import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from pathlib import Path
import json

class DNAImageProcessor:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.current_char = 'a'  # Start with 'a'
        self.current_mode = 'train'  # Start with training mode
        self.selections = []
        self.window_name = 'DNA Character Selector'
        self.train_dir = Path('dna_training_data')
        self.test_dir = Path('dna_test_data')
        self.config_file = Path('processor_config.json')
        self.roi_size = 40  # Size of selection box
        self.current_position = None
        self.saved_count = self.load_saved_counts()
        
    def load_saved_counts(self):
        """Load previously saved character counts"""
        default_counts = {
            'train': {'a': 0, 'c': 0, 'g': 0, 't': 0},
            'test': {'a': 0, 'c': 0, 'g': 0, 't': 0}
        }
    
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_counts = json.load(f)
                    # Ensure all required keys exist
                    if not all(mode in saved_counts for mode in ['train', 'test']):
                        return default_counts
                    if not all(char in saved_counts['train'] for char in ['a', 'c', 'g', 't']):
                        return default_counts
                    if not all(char in saved_counts['test'] for char in ['a', 'c', 'g', 't']):
                        return default_counts
                    return saved_counts
            except (json.JSONDecodeError, KeyError):
                return default_counts
        return default_counts
        
    def save_counts(self):
        """Save current character counts"""
        with open(self.config_file, 'w') as f:
            json.dump(self.saved_count, f)

    def preprocess_image(self, image_path):
        """Load and preprocess the image for better character visibility"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Invert if necessary (ensure white background, black text)
        if np.mean(binary[0:10, 0:10]) < 127:  # Check corner for background color
            binary = cv2.bitwise_not(binary)
            
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        self.image = img
        self.processed_image = denoised
        return denoised

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for character selection"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_position = (x, y)
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Extract and save character
            half_size = self.roi_size // 2
            x1, y1 = x - half_size, y - half_size
            x2, y2 = x + half_size, y + half_size
            
            # Ensure within bounds
            h, w = self.processed_image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 > 0 and y2 - y1 > 0:
                char_img = self.processed_image[y1:y2, x1:x2]
                self.save_character(char_img)

    def save_character(self, char_img):
        """Save the selected character"""
        # Determine target directory based on mode
        base_dir = self.train_dir if self.current_mode == 'train' else self.test_dir
        char_dir = base_dir / self.current_char
        char_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        self.saved_count[self.current_mode][self.current_char] += 1
        filename = f"{self.current_char}{self.saved_count[self.current_mode][self.current_char]}.png"
        
        # Save image
        cv2.imwrite(str(char_dir / filename), char_img)
        print(f"Saved {filename} to {char_dir} ({self.current_mode} set)")
        self.save_counts()

    def update_display(self):
        """Update the display with current selection box and mode"""
        display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        
        if self.current_position:
            x, y = self.current_position
            half_size = self.roi_size // 2
            
            # Draw selection box (green for train, blue for test)
            color = (0, 255, 0) if self.current_mode == 'train' else (255, 0, 0)
            cv2.rectangle(
                display_img,
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                color,
                1
            )
            
            # Show current character and mode being selected
            cv2.putText(
                display_img,
                f"Mode: {self.current_mode.upper()}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            cv2.putText(
                display_img,
                f"Character: {self.current_char.upper()} "
                f"({self.saved_count[self.current_mode][self.current_char]} saved)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
        cv2.imshow(self.window_name, display_img)

    def process_image(self, image_path):
        """Main processing loop"""
        # Preprocess image
        processed = self.preprocess_image(image_path)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nControls:")
        print("Left Click: Select and save character")
        print("A: Switch to selecting 'A's")
        print("C: Switch to selecting 'C's")
        print("G: Switch to selecting 'G's")
        print("T: Switch to selecting 'T's")
        print("M: Toggle between train/test mode")
        print("+/-: Adjust selection box size")
        print("ESC: Exit")
        
        while True:
            self.update_display()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('a'):
                self.current_char = 'a'
            elif key == ord('c'):
                self.current_char = 'c'
            elif key == ord('g'):
                self.current_char = 'g'
            elif key == ord('t'):
                self.current_char = 't'
            elif key == ord('m'):
                # Toggle between train and test mode
                self.current_mode = 'test' if self.current_mode == 'train' else 'train'
                print(f"\nSwitched to {self.current_mode.upper()} mode")
            elif key == ord('+') or key == ord('='):
                self.roi_size = min(100, self.roi_size + 2)
            elif key == ord('-'):
                self.roi_size = max(10, self.roi_size - 2)
                
        cv2.destroyAllWindows()
        self.save_counts()

def main():
    # Create GUI for file selection
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Select image file
    image_path = filedialog.askopenfilename(
        title="Select DNA Sequence Image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("No image selected")
        return
        
    try:
        processor = DNAImageProcessor()
        processor.process_image(image_path)
        
        print("\nProcessing complete!")
        print("\nCharacter counts:")
        print("\nTraining set:")
        for char, count in processor.saved_count['train'].items():
            print(f"{char.upper()}: {count}")
        print("\nTest set:")
        for char, count in processor.saved_count['test'].items():
            print(f"{char.upper()}: {count}")
            
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
