import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import models
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import csv
import datetime
import pyperclip
import os


class DNASequenceExtractor:
    def __init__(self, model_path="model/latest_model.h5", use_separation=True, use_dilation=True ):  # Added use_separation parameter
        self.model_path = Path(model_path)
        self.model = None
        self.input_shape = (32, 32, 1)
        self.char_mapping = {0: 'a', 1: 'c', 2: 'g', 3: 't'}
        self.confidence_threshold = 0.75  # Minimum confidence for prediction
        self.use_separation = use_separation
        self.use_dilation = use_dilation
        
        self.load_model()
    
    def set_use_separation(self, value):
        """Set whether to use character separation"""
        self.use_separation = value
        print(f"Character separation set to: {value}")
        
    def set_use_dilation(self, value):
        """Set whether to use vertical dilation"""
        self.use_dilation = value
        print(f"Vertical dilation set to: {value}")
    
    def load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print("Loading model...")
        self.model = models.load_model(str(self.model_path))
        print("Model loaded successfully!")
    
    def preprocess_image(self, padded_image):
        """Preprocess the input image with advanced techniques"""
        # Ensure input is numpy array with proper type
        img = np.array(padded_image).astype(np.uint8)
        if img is None or img.size == 0:
            raise ValueError("Could not load image")

        # Show original
        cv2.imshow('1. Original', img)
        cv2.waitKey(0)

        # Convert to grayscale
        if len(img.shape) == 3:  # Color image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            gray = img

        # Normalize intensity values
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(normalized)

        # Apply mild Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure white background, black text
        if np.mean(binary[0:10, 0:10]) < 127:
            binary = cv2.bitwise_not(binary)

        cv2.imshow('2. Initial Binary', binary)
        cv2.waitKey(0)

        if self.use_dilation:
            # Dilate vertically to connect broken parts within characters
            kernel_vertical = np.ones((3,1), np.uint8)
            dilated_vertical = cv2.dilate(binary, kernel_vertical, iterations=1)
            cv2.imshow('3. Dilated Vertical', dilated_vertical)
            cv2.waitKey(0)
        else:
            dilated_vertical = binary
            cv2.imshow('3. Without Dilation', dilated_vertical)
            cv2.waitKey(0)

        # Identify text lines
        horizontal_projection = np.sum(255 - dilated_vertical, axis=1)
        line_threshold = np.max(horizontal_projection) * 0.1

        # Find line boundaries
        lines = []
        in_line = False
        start_y = 0

        for y in range(len(horizontal_projection)):
            if not in_line and horizontal_projection[y] > line_threshold:
                in_line = True
                start_y = y
            elif in_line and horizontal_projection[y] <= line_threshold:
                in_line = False
                lines.append((start_y, y))

        # Visualize lines
        line_vis = dilated_vertical.copy()
        line_vis = cv2.cvtColor(line_vis, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored lines
        line_vis = line_vis.astype(np.uint8)
        for start_y, end_y in lines:
            cv2.line(line_vis, (0, start_y), (dilated_vertical.shape[1], start_y), (0, 255, 0), 1)
            cv2.line(line_vis, (0, end_y), (dilated_vertical.shape[1], end_y), (0, 255, 0), 1)

        cv2.imshow('4. Detected Lines', line_vis)
        cv2.waitKey(0)

        if self.use_separation:
            # Separate touching characters
            separated = dilated_vertical.copy()
            min_black_pixels = 8  # Minimum black pixels to keep

            for start_y, end_y in lines:
                line_height = end_y - start_y
                bottom_quarter_start = end_y - line_height // 4
                line_region = separated[start_y:end_y, :]
    
                # Process each vertical column in the line
                for x in range(line_region.shape[1]):
                    column = line_region[:, x]
                    black_pixels = np.sum(column == 0)
        
                    # Check if black pixels are only in bottom quarter
                    bottom_quarter = column[bottom_quarter_start-start_y:]
                    black_pixels_bottom = np.sum(bottom_quarter == 0)
        
                    if 0 < black_pixels <= min_black_pixels and black_pixels == black_pixels_bottom:
                        line_region[:, x] = 255  # Convert to white

            # Ensure proper type for visualization
            separated = separated.astype(np.uint8)
            cv2.imshow('5. Separated Characters', separated)
            cv2.waitKey(0)
            final_image = separated
        else:
            final_image = dilated_vertical
            
        cv2.destroyAllWindows()
        return final_image, img
    
    def process_image(self, image_path, visualize=True):
        try:
            # Add padding to image
            original_image = cv2.imread(str(image_path))
            padding = 50
            padded_image = cv2.copyMakeBorder(original_image, padding, padding, padding, padding,
                                             cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
            # Store padded image
            self.original_image = padded_image
        
            # Process the padded image
            processed_image, _ = self.preprocess_image(padded_image)
    
            # Segment characters
            characters, positions, split_positions = self.segment_characters(processed_image)
    
            if len(characters) == 0:
                raise ValueError("No characters found in image")
    
            # Show component ordering
            ordering_vis = self.visualize_component_ordering(positions, processed_image)
            cv2.imshow('Component Ordering', ordering_vis)
            cv2.waitKey(0)
            cv2.destroyWindow('Component Ordering')
    
            # Predict sequence
            sequence = self.predict_sequence(characters, positions)
    
            # Initialize visualization result
            vis_result = None
        
            # Visualize if requested
            if visualize:
                vis_result, sequence = self.visualize_predictions(padded_image, sequence, split_positions)
    
            # Create final results
            final_sequence = [pred for pred in sequence if not pred.get('deleted', False)]
    
            results = {
                'sequence': ''.join(pred['char'] for pred in final_sequence),
                'detailed_predictions': final_sequence,
                'average_confidence': np.mean([pred['confidence'] for pred in final_sequence]),
                'split_positions': split_positions
            }
    
            if visualize and vis_result is not None:
                results['visualization'] = vis_result
    
            return results
    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise
        

    def segment_characters(self, processed_image):
        """Segment individual characters from the image"""
        padding = 10
        padded = cv2.copyMakeBorder(processed_image, padding, padding, padding, padding,
                                   cv2.BORDER_CONSTANT, value=255)

        # Invert for contour detection
        inverted = cv2.bitwise_not(padded)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        # First pass: collect area and width statistics
        areas = []
        components = []
        widths = []

        # Skip first label (background)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            center_y = y + h/2
        
            components.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'area': area,
                'center_y': center_y
            })
            areas.append(area)
            widths.append(w)

        # Calculate average character area and filter small components
        avg_area = np.mean(areas)
        min_area = avg_area * 0.5
        filtered_indices = [i for i, comp in enumerate(components) if comp['area'] >= min_area]
        components = [components[i] for i in filtered_indices]
        widths = [widths[i] for i in filtered_indices]

        # Get y-coordinates for clustering
        y_coords = np.array([comp['center_y'] for comp in components])
    
        # Sort y-coordinates to find natural groupings
        sorted_y = np.sort(y_coords)
        y_diffs = np.diff(sorted_y)
    
        # Find significant gaps in y-coordinates (potential line breaks)
        mean_diff = np.mean(y_diffs)
        std_diff = np.std(y_diffs)
        significant_gaps = np.where(y_diffs > mean_diff + std_diff)[0]
    
        # Number of lines is one more than number of significant gaps
        n_lines = len(significant_gaps) + 1 if len(significant_gaps) > 0 else 1

        # Perform k-means clustering on y-coordinates
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_lines, random_state=42)
        y_clusters = kmeans.fit_predict(y_coords.reshape(-1, 1))

        # Group components by cluster and sort within each cluster
        lines = [[] for _ in range(n_lines)]
        for idx, (comp, cluster) in enumerate(zip(components, y_clusters)):
            lines[cluster].append((idx, comp))

        # Sort each line by x-coordinate
        for line in lines:
            line.sort(key=lambda x: x[1]['x'])

        # Sort lines by average y-coordinate
        lines.sort(key=lambda line: np.mean([comp['center_y'] for _, comp in line]))

        # Create final ordered sequence
        ordered_components = []
        for line in lines:
            ordered_components.extend(line)

        # Visualize the ordering and line grouping
        vis_image = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    
        # Draw line indicators
        for line in lines:
            if len(line) > 1:
                # Calculate average y for the line
                avg_y = int(np.mean([comp['center_y'] for _, comp in line]))
                # Get x coordinates for start and end of line
                x_start = min(comp['x'] for _, comp in line)
                x_end = max(comp['x'] + comp['w'] for _, comp in line)
                # Draw line
                cv2.line(vis_image, (x_start, avg_y), (x_end, avg_y), (0, 255, 0), 1)

        # Draw boxes and numbers
        for order, (_, comp) in enumerate(ordered_components, 1):
            x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(vis_image, str(order), (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Character Ordering', vis_image)
        cv2.waitKey(0)

        # Process components in the new order
        characters = []
        positions = []
        split_positions = []
        current_idx = 0

        # Analyze width distribution for splitting
        widths = np.array([comp['w'] for _, comp in ordered_components])
        widths_hist = np.histogram(widths, bins=20)
        most_common_width = widths_hist[1][np.argmax(widths_hist[0])]
        single_char_width = most_common_width

        # Process components in order
        for _, comp in ordered_components:
            width_ratio = comp['w'] / single_char_width
            num_chars = round(width_ratio)

            if num_chars <= 1:
                # Single character
                char_img = padded[comp['y']:comp['y']+comp['h'], 
                                comp['x']:comp['x']+comp['w']]
                char_img = cv2.resize(char_img, (32, 32))
                char_img = char_img.astype(float) / 255.0
                char_img = char_img.reshape(32, 32, 1)

                characters.append(char_img)
                positions.append((comp['x'] - padding, comp['y'] - padding, 
                                comp['w'], comp['h']))
                current_idx += 1
            else:
                # Multiple characters
                split_width = comp['w'] // num_chars
                for i in range(num_chars):
                    split_positions.append(current_idx)

                    split_x = comp['x'] + (i * split_width)
                    x_start = max(split_x - 1, comp['x'])
                    x_end = min(split_x + split_width + 1, comp['x'] + comp['w'])

                    char_img = padded[comp['y']:comp['y']+comp['h'], 
                                    x_start:x_end]
                    char_img = cv2.resize(char_img, (32, 32))
                    char_img = char_img.astype(float) / 255.0
                    char_img = char_img.reshape(32, 32, 1)

                    characters.append(char_img)
                    positions.append((split_x - padding, comp['y'] - padding, 
                                    split_width, comp['h']))
                    current_idx += 1

        print(f"Found {len(characters)} characters")
        return np.array(characters), positions, split_positions
    
    
    def visualize_component_ordering(self, positions, processed_image):
        """Simply visualize the component ordering"""
        vis_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    
        # Draw boxes and order numbers
        for idx, pos in enumerate(positions, 1):
            x, y, w, h = pos
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(vis_image, str(idx), (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
        return vis_image

    def update_visualization(self, vis_image, sequence, split_positions, selected_idx=None):
        """Update the visualization with current predictions"""
        # Ensure vis_image is a numpy array with correct dtype
        if not isinstance(vis_image, np.ndarray):
            raise ValueError("vis_image must be a numpy array")
    
        # Make a copy of the original image
        display_image = self.original_image.copy()
    
        # Ensure the image is in BGR format
        if len(display_image.shape) == 2:  # If grayscale
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        elif display_image.shape[2] == 4:  # If RGBA
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGBA2BGR)
    
        # Draw only non-deleted boxes
        visible_idx = 0  # Counter for visible (non-deleted) boxes
        for idx, pred in enumerate(sequence):
            if pred['position'] and not pred.get('deleted', False):
                x, y, w, h = pred['position']
                confidence = pred['confidence']
                char = pred['char'].replace('[', '').replace('?]', '')
            
                # Determine base color
                if idx in split_positions:
                    base_color = (0, 165, 255)  # Orange for split boxes
                elif confidence > 0.9:
                    base_color = (0, 255, 0)    # Green for high confidence
                else:
                    base_color = (0, 0, 255)    # Red for low confidence
            
                # Highlight selected box
                if idx == selected_idx:
                    cv2.rectangle(display_image, 
                                (x-2, y-2), 
                                (x+w+2, y+h+2), 
                                (255, 255, 0), 2)  # Cyan highlight
                    thickness = 2
                else:
                    thickness = 1
            
                # Draw box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), base_color, thickness)
            
                # Draw character and position
                text = f"{char}"
                cv2.putText(display_image, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, thickness+1)
            
                visible_idx += 1
    
        # Update the original vis_image with our changes
        vis_image[:] = display_image

    def visualize_predictions(self, image, sequence, split_positions=None):
        """Visualize predictions with interactive correction and insertion"""
        #Ensure image is in correct format
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
    
        # Convert image to BGR if needed
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # If RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
        selected_idx = None
        history = []
        insert_mode = False
        final_vis_image = None
    
        def handle_key(key):
            """Handle keyboard input for the visualization"""
            nonlocal selected_idx, insert_mode, history
        
            if selected_idx is not None:
                if key == ord('i'):
                    insert_mode = True
                    self.update_visualization(vis_image, sequence, split_positions, selected_idx)
                    return
            
                if insert_mode and key in [ord(c) for c in 'acgtACGT']:
                    # Save current state for undo
                    history.append([dict(pred) for pred in sequence])
                
                    # Get current character and append new one
                    current_char = sequence[selected_idx]['char'].replace('[', '').replace('?]', '')
                    new_char = current_char + chr(key).upper()
                
                    # Update the character
                    sequence[selected_idx]['char'] = new_char
                    sequence[selected_idx]['confidence'] = 1.0  # User-modified confidence
                
                    # Keep the same box selected
                    self.update_visualization(vis_image, sequence, split_positions, selected_idx)
                    return
            
                if not insert_mode:
                    if key in [ord(c) for c in 'acgtACGT']:
                        history.append([dict(pred) for pred in sequence])
                        sequence[selected_idx]['char'] = chr(key).upper()
                        sequence[selected_idx]['confidence'] = 1.0
                    elif key in [8, ord('d')]:  # Backspace or 'd'
                        history.append([dict(pred) for pred in sequence])
                        sequence[selected_idx]['deleted'] = True
                        selected_idx = None
                        insert_mode = False
            
                if key == ord('\t'):  # Tab
                    old_selected = selected_idx
                    next_idx = (selected_idx + 1) % len(sequence)
                    while next_idx != selected_idx and sequence[next_idx].get('deleted', False):
                        next_idx = (next_idx + 1) % len(sequence)
                    selected_idx = next_idx if not sequence[next_idx].get('deleted', False) else None
                    if old_selected != selected_idx:
                        insert_mode = False
                elif key == ord('`'):  # Back-tab
                    old_selected = selected_idx
                    prev_idx = (selected_idx - 1) % len(sequence)
                    while prev_idx != selected_idx and sequence[prev_idx].get('deleted', False):
                        prev_idx = (prev_idx - 1) % len(sequence)
                    selected_idx = prev_idx if not sequence[prev_idx].get('deleted', False) else None
                    if old_selected != selected_idx:
                        insert_mode = False
        
            elif key == ord('\t'):
                # If nothing selected, select first non-deleted box
                for idx, pred in enumerate(sequence):
                    if not pred.get('deleted', False):
                        selected_idx = idx
                        insert_mode = False
                        break
        
            self.update_visualization(vis_image, sequence, split_positions, selected_idx)
    
        def on_mouse_click(event, x, y, flags, param):
            nonlocal selected_idx, insert_mode
            if event == cv2.EVENT_LBUTTONDOWN:
                old_selected = selected_idx
                selected_idx = None
                for idx, pred in enumerate(sequence):
                    if pred['position'] and not pred.get('deleted', False):
                        x1, y1, w, h = pred['position']
                        if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                            selected_idx = idx
                            break
            
                # Only reset insert mode if we've selected a different box
                if old_selected != selected_idx:
                    insert_mode = False
            
                self.update_visualization(vis_image, sequence, split_positions, selected_idx)
    
        def update_status_bar(help_image):
            """Update status bar with current mode"""
            status_img = help_image.copy()
            mode_text = "INSERT MODE" if insert_mode else "NORMAL MODE"
            cv2.putText(status_img, f"Current Mode: {mode_text}", (10, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            return status_img
    
        # Create help image with proper format
        help_image = np.full((450, 400, 3), 255, dtype=np.uint8)
        help_text = [
            "Keyboard Shortcuts:",
            "Click/Tab: Select box",
            "A/C/G/T: Change letter",
            "I: Enter insert mode",
            "   (append characters)",
            "Backspace/D: Delete box",
            "U: Undo",
            "Tab/Shift-Tab: Navigate",
            "Enter: Finish",
            "H: Toggle help",
            "Esc: Cancel"
        ]
    
        for i, line in enumerate(help_text):
            cv2.putText(help_image, line, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
        # Create initial visualization image
        vis_image = image.copy()
        cv2.namedWindow('DNA Sequence Editor')
        cv2.setMouseCallback('DNA Sequence Editor', on_mouse_click)
    
        show_help = True
    
        # Initial visualization
        self.update_visualization(vis_image, sequence, split_positions, selected_idx)
    
        while True:
            # Ensure vis_image is in correct format before showing
            if not isinstance(vis_image, np.ndarray):
                break
            
            cv2.imshow('DNA Sequence Editor', vis_image)
            if show_help:
                status_help = update_status_bar(help_image)
                cv2.imshow('Keyboard Shortcuts', status_help)
        
            key = cv2.waitKey(1) & 0xFF
        
            if key == 27:  # Esc
                vis_image = None
                break
            elif key == 13:  # Enter
                break
            elif key == ord('u') and history:  # Undo
                sequence = [dict(pred) for pred in history.pop()]
                insert_mode = False
            elif key == ord('h'):  # Toggle help
                show_help = not show_help
                if not show_help:
                    cv2.destroyWindow('Keyboard Shortcuts')
            else:
                handle_key(key)
    
        # Clean up windows
        cv2.destroyAllWindows()
    
        # Return the final visualization and sequence
        return (final_vis_image, sequence)

    def get_mouse_position(self, window_name):
        """Get current mouse position in window"""
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                param[0] = x
                param[1] = y
    
        pos = [None, None]
        cv2.setMouseCallback(window_name, on_mouse, pos)
        cv2.waitKey(1)
        return pos[0], pos[1]
        
    def update_visualization(self, vis_image, sequence, split_positions, selected_idx=None):
        """Update the visualization with current predictions"""
        # Ensure vis_image is a numpy array with correct dtype
        if not isinstance(vis_image, np.ndarray):
            raise ValueError("vis_image must be a numpy array")
    
        # Make a copy of the original image
        display_image = self.original_image.copy()
    
        # Ensure the image is in BGR format
        if len(display_image.shape) == 2:  # If grayscale
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        elif display_image.shape[2] == 4:  # If RGBA
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGBA2BGR)
    
        # Draw only non-deleted boxes
        visible_idx = 0  # Counter for visible (non-deleted) boxes
        for idx, pred in enumerate(sequence):
            if pred['position'] and not pred.get('deleted', False):
                x, y, w, h = pred['position']
                confidence = pred['confidence']
                char = pred['char'].replace('[', '').replace('?]', '')
            
                # Determine base color
                if idx in split_positions:
                    base_color = (0, 165, 255)  # Orange for split boxes
                elif confidence > 0.9:
                    base_color = (0, 255, 0)    # Green for high confidence
                else:
                    base_color = (0, 0, 255)    # Red for low confidence
            
                # Highlight selected box
                if idx == selected_idx:
                    cv2.rectangle(display_image, 
                                (x-2, y-2), 
                                (x+w+2, y+h+2), 
                                (255, 255, 0), 2)  # Cyan highlight
                    thickness = 2
                else:
                    thickness = 1
            
                # Draw box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), base_color, thickness)
            
                # Draw character and position
                text = f"{char}"
                cv2.putText(display_image, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, thickness+1)
            
                visible_idx += 1
    
        # Update the original vis_image with our changes
        vis_image[:] = display_image

    
    def predict_sequence(self, chars, positions=None, original_image=None):
        """Predict DNA sequence from segmented characters"""
        # Get predictions for all characters
        predictions = self.model.predict(chars, verbose=0)
    
        sequence = []
    
        # For each character prediction
        for idx, pred in enumerate(predictions):
            # Get probabilities for each class
            class_probs = {self.char_mapping[i].upper(): float(p) for i, p in enumerate(pred)}
        
            # Get the highest probability prediction
            predicted_char = max(class_probs.items(), key=lambda x: x[1])
            char, confidence = predicted_char
        
            # Add confidence threshold marking
            if confidence < self.confidence_threshold:
                char_display = f"[{char}?]"
            else:
                char_display = char
            
            sequence.append({
                'char': char_display,
                'confidence': confidence,
                'position': positions[idx] if positions else None,
                'all_probs': class_probs  # Include all probabilities for debugging
            })
    
        print("\nDetailed predictions:")
        for idx, pred in enumerate(sequence):
            print(f"\nCharacter {idx + 1}:")
            print(f"Predicted: {pred['char']} (Confidence: {pred['confidence']:.2%})")
            print("All probabilities:")
            for char, prob in pred['all_probs'].items():
                print(f"  {char}: {prob:.2%}")
    
        return sequence
    
        
class DNASequenceExporter:
    """Handles exporting of DNA sequence results"""
    
    @staticmethod
    def export_to_json(results, filepath):
        """Export results to JSON file"""
        export_data = {
            'date': datetime.datetime.now().isoformat(),
            'sequence': results['sequence'],
            'average_confidence': float(results['average_confidence']),
            'detailed_predictions': [
                {
                    'char': pred['char'],
                    'confidence': float(pred['confidence']),
                    'position': pred['position']
                }
                for pred in results['detailed_predictions']
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    @staticmethod
    def export_to_csv(results, filepath):
        """Export results to CSV file"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', datetime.datetime.now().isoformat()])
            writer.writerow(['Complete Sequence', results['sequence']])
            writer.writerow(['Average Confidence', f"{results['average_confidence']:.2%}"])
            writer.writerow([])
            writer.writerow(['Position', 'Character', 'Confidence', 'X', 'Y', 'Width', 'Height'])
            
            for idx, pred in enumerate(results['detailed_predictions'], 1):
                pos = pred['position']
                writer.writerow([
                    idx,
                    pred['char'],
                    f"{pred['confidence']:.2%}",
                    pos[0] if pos else 'N/A',
                    pos[1] if pos else 'N/A',
                    pos[2] if pos else 'N/A',
                    pos[3] if pos else 'N/A'
                ])

    @staticmethod
    def export_to_text(results, filepath):
        """Export results to plain text file"""
        with open(filepath, 'w') as f:
            f.write(f"DNA Sequence Analysis - {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Extracted Sequence: {results['sequence']}\n")
            f.write(f"Average Confidence: {results['average_confidence']:.2%}\n\n")
            f.write("Detailed Predictions:\n")
            f.write("-" * 30 + "\n")
            
            for idx, pred in enumerate(results['detailed_predictions'], 1):
                f.write(f"Position {idx}: {pred['char']} "
                       f"(Confidence: {pred['confidence']:.2%})\n")

class DNASequenceExtractorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DNA Sequence Extractor")
        self.root.geometry("800x600")
        
        # Create settings first
        self.use_separation = tk.BooleanVar(value=True)
        self.use_dilation = tk.BooleanVar(value=True)
        
        # Create extractor with initial settings
        self.extractor = DNASequenceExtractor(
            use_separation=self.use_separation.get(),
            use_dilation=self.use_dilation.get()
        )
        self.exporter = DNASequenceExporter()
        self.current_results = None
        self.setup_gui()
    
    def update_settings(self):
        """Update extractor settings when checkboxes change"""
        self.extractor.set_use_separation(self.use_separation.get())
        self.extractor.set_use_dilation(self.use_dilation.get())
        
    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Separation checkbox
        ttk.Checkbutton(settings_frame, 
                       text="Use character separation (helps with touching characters)",
                       variable=self.use_separation,
                       command=self.update_settings).grid(row=0, column=0, padx=5, sticky='w')
        
        # Dilation checkbox
        ttk.Checkbutton(settings_frame, 
                       text="Use vertical dilation (helps with broken characters)",
                       variable=self.use_dilation,
                       command=self.update_settings).grid(row=1, column=0, padx=5, sticky='w')
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(btn_frame, text="Select Image", 
                  command=self.process_image).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Export Results", 
                  command=self.show_export_options).grid(row=0, column=1, padx=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text widget for results
        self.results_text = tk.Text(self.results_frame, height=20, width=80)
        self.results_text.grid(row=0, column=0)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, 
                                command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        status_bar.grid(row=3, column=0, pady=5)

    
    def process_image(self):
        """Process selected image"""
        image_path = filedialog.askopenfilename(
            title="Select DNA Sequence Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not image_path:
            return
        
        try:
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Process image
            self.current_results = self.extractor.process_image(image_path)
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            
            # Display results
            self.results_text.insert(tk.END, "Extracted Sequence:\n")
            self.results_text.insert(tk.END, f"{self.current_results['sequence']}\n\n")
            
            # Copy sequence to clipboard
            pyperclip.copy(self.current_results['sequence'])
            
            self.results_text.insert(tk.END, 
                f"Average Confidence: {self.current_results['average_confidence']:.2%}\n\n")
            
            self.results_text.insert(tk.END, "Detailed Predictions:\n")
            for idx, pred in enumerate(self.current_results['detailed_predictions'], 1):
                self.results_text.insert(tk.END, 
                    f"Position {idx}: {pred['char']} "
                    f"(Confidence: {pred['confidence']:.2%})\n")
            
            # Show visualization
            if 'visualization' in self.current_results:
                cv2.imshow('Predictions', self.current_results['visualization'])
            
            self.status_var.set("Processing complete! Sequence copied to clipboard.")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error processing image")
    
    def show_export_options(self):
        """Show export options dialog"""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Options")
        export_window.geometry("300x200")
        
        ttk.Label(export_window, text="Select export format:").pack(pady=10)
        
        def export_results(format_type):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                output_dir = Path("exports")
                output_dir.mkdir(exist_ok=True)
                
                if format_type == "json":
                    filepath = output_dir / f"dna_sequence_{timestamp}.json"
                    self.exporter.export_to_json(self.current_results, filepath)
                elif format_type == "csv":
                    filepath = output_dir / f"dna_sequence_{timestamp}.csv"
                    self.exporter.export_to_csv(self.current_results, filepath)
                elif format_type == "txt":
                    filepath = output_dir / f"dna_sequence_{timestamp}.txt"
                    self.exporter.export_to_text(self.current_results, filepath)
                
                messagebox.showinfo("Success", 
                    f"Results exported to:\n{filepath}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
            
            export_window.destroy()
        
        ttk.Button(export_window, text="JSON", 
                  command=lambda: export_results("json")).pack(pady=5)
        ttk.Button(export_window, text="CSV", 
                  command=lambda: export_results("csv")).pack(pady=5)
        ttk.Button(export_window, text="Text", 
                  command=lambda: export_results("txt")).pack(pady=5)
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    app = DNASequenceExtractorGUI()
    app.run()

if __name__ == "__main__":
    main()
