import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import datetime
import random
import cv2

class DNADataAugmenter:
    def __init__(self, rotation_range=5, scale_range=0.1, 
                 shear_range=2, translation_range=2):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        
    def augment_image(self, image):
        """Generate augmented versions of a single image"""
        augmented_images = []
    
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # Ensure image is 2D (remove any extra dimensions)
            image = image.squeeze()
            image = Image.fromarray((image * 255).astype('uint8'))
    
        # Original image
        augmented_images.append(np.array(image) / 255.0)
    
        # 1. Slight rotations
        for angle in [-self.rotation_range, self.rotation_range]:
            rotated = image.rotate(angle, fillcolor=255, expand=False)
            augmented_images.append(np.array(rotated) / 255.0)
    
        # 2. Small scale variations
        for scale in [1.0 - self.scale_range, 1.0 + self.scale_range]:
            new_size = tuple(int(dim * scale) for dim in image.size)
            scaled = image.resize(new_size, Image.Resampling.BICUBIC)  # Changed from LANCZOS
            scaled = self._pad_or_crop_to_size(scaled, image.size)
            augmented_images.append(np.array(scaled) / 255.0)
    
        # 3. Slight translations
        for dx in [-self.translation_range, self.translation_range]:
            for dy in [-self.translation_range, self.translation_range]:
                translated = Image.new('L', image.size, 255)
                translated.paste(image, (dx, dy))
                augmented_images.append(np.array(translated) / 255.0)
    
        # 4. Small shear transformations
        for shear in [-self.shear_range, self.shear_range]:
            width, height = image.size
            m = 1.0
            xshift = abs(m) * height
            new_width = width + int(round(xshift))
            sheared = image.transform(
                (new_width, height),
                Image.AFFINE,
                (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                Image.Resampling.BICUBIC,  # Changed from LANCZOS
                fillcolor=255
            )
            sheared = self._pad_or_crop_to_size(sheared, image.size)
            augmented_images.append(np.array(sheared) / 255.0)
    
        # Ensure all images have the same shape and add channel dimension
        augmented_images = [img.reshape(32, 32, 1) for img in augmented_images]
    
        return augmented_images
    
    def _pad_or_crop_to_size(self, image, target_size):
        """Pad or crop image to match target size"""
        if image.size == target_size:
            return image
            
        result = Image.new('L', target_size, 255)
        x = (target_size[0] - image.size[0]) // 2
        y = (target_size[1] - image.size[1]) // 2
        result.paste(image, (max(0, x), max(0, y)))
        return result

class DNAOCRIncrementalTrainer:
    def __init__(self, data_dir="dna_training_data", model_dir="model"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.input_shape = (32, 32, 1)
        self.char_mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}
        
        # Load or create training history
        self.history_file = self.model_dir / 'training_history.json'
        self.training_history = self.load_training_history()
        
        # Load or create model
        self.model = self.load_or_create_model()
    
    def load_training_history(self):
        """Load or initialize training history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            'last_training_date': None,
            'total_samples_trained': {char: 0 for char in self.char_mapping},
            'training_sessions': []
        }
    
    def save_training_history(self):
        """Save training history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _build_model(self):
        """Build the CNN model using functional API"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # First Convolutional Block
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Dense Layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(4, activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        latest_model = self.model_dir / 'latest_model.h5'
        if latest_model.exists():
            print("Loading existing model...")
            # Load the model architecture and weights
            model = models.load_model(str(latest_model))
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        print("Creating new model...")
        return self._build_model()
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((32, 32), Image.Resampling.BICUBIC)  # Changed from LANCZOS
            img_array = np.array(img) / 255.0
            return img_array.reshape(32, 32, 1)  # Ensure correct shape with channel
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def augmented_load_new_samples(self):
        """Load new samples with augmentation"""
        X = []
        y = []
        new_samples = {char: 0 for char in self.char_mapping}
        augmented_samples = {char: 0 for char in self.char_mapping}
        
        # Create augmenter
        augmenter = DNADataAugmenter()
        
        for char in self.char_mapping.keys():
            char_dir = self.data_dir / char
            if not char_dir.exists():
                continue
            
            # Get all image files
            image_files = list(char_dir.glob("*.png"))
            total_files = len(image_files)
            
            # Calculate new samples
            previously_trained = self.training_history['total_samples_trained'][char]
            new_files = image_files[previously_trained:]
            
            for img_path in new_files:
                img_array = self.load_and_preprocess_image(img_path)
                if img_array is not None:
                    # Add original image
                    X.append(img_array)
                    y.append(self.char_mapping[char])
                    new_samples[char] += 1
                    
                    # Generate and add augmented versions
                    augmented_images = augmenter.augment_image(img_array)
                    # Skip first image as it's the original
                    for aug_img in augmented_images[1:]:
                        X.append(aug_img)
                        y.append(self.char_mapping[char])
                        augmented_samples[char] += 1
        
        if not X:
            return None, None, new_samples, augmented_samples
        
        return np.array(X), np.array(y), new_samples, augmented_samples
    
    def incremental_train(self, epochs=50, batch_size=32, validation_split=0.2):
        """Perform incremental training on new samples with augmentation"""
        # Load new samples with augmentation
        X, y, new_samples, augmented_samples = self.augmented_load_new_samples()
        
        if X is None or len(X) == 0:
            print("No new samples found for training!")
            return None
        
        print("\nNew samples found:")
        for char, count in new_samples.items():
            if count > 0:
                print(f"{char.upper()}: {count} original + "
                      f"{augmented_samples[char]} augmented = "
                      f"{count + augmented_samples[char]} total")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        print("\nStarting incremental training...")
        print(f"Training on {len(X_train)} samples "
              f"({len(X_val)} validation samples)")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Update training history
        session_info = {
            'date': datetime.datetime.now().isoformat(),
            'new_samples': {
                char: {
                    'original': new_samples[char],
                    'augmented': augmented_samples[char],
                    'total': new_samples[char] + augmented_samples[char]
                } for char in self.char_mapping
            },
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        self.training_history['training_sessions'].append(session_info)
        
        # Update total samples trained (only count original samples)
        for char, count in new_samples.items():
            self.training_history['total_samples_trained'][char] += count
        
        self.training_history['last_training_date'] = session_info['date']
        
        # Save updated history and model
        self.save_training_history()
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save current model state"""
        # Save latest model
        latest_path = self.model_dir / 'latest_model.h5'
        self.model.save(str(latest_path))
        
        # Also save a versioned copy
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_path = self.model_dir / f'model_{timestamp}.h5'
        self.model.save(str(versioned_path))
        
        print(f"\nModel saved as '{latest_path}' and '{versioned_path}'")
    
    def print_training_summary(self):
        """Print summary of all training sessions with augmentation details"""
        print("\nTraining History Summary:")
        print("========================")
        print(f"Total original samples trained:")
        for char, count in self.training_history['total_samples_trained'].items():
            print(f"{char.upper()}: {count}")
        
        print("\nTraining sessions:")
        for session in self.training_history['training_sessions']:
            print(f"\nDate: {session['date']}")
            print("Samples processed:")
            for char, counts in session['new_samples'].items():
                if counts['total'] > 0:
                    print(f"  {char.upper()}: {counts['original']} original + "
                          f"{counts['augmented']} augmented = "
                          f"{counts['total']} total")
            print(f"Final accuracy: {session['final_accuracy']:.2%}")
            print(f"Final validation accuracy: {session['final_val_accuracy']:.2%}")
    
    def test_predictions(self, num_samples=5, test_dir="dna_test_data"):
        """Test model on samples from test directory"""
        test_dir = Path(test_dir)
        X = []
        y = []
        file_paths = []  # Keep track of file paths for reporting
    
        print("\nLooking for test samples in:", test_dir)
    
        # Load samples from test directory
        for char in self.char_mapping.keys():
            char_dir = test_dir / char
            if not char_dir.exists():
                print(f"Directory not found for {char.upper()}: {char_dir}")
                continue
        
            # Get all image files
            image_files = list(char_dir.glob("*.png"))
        
            if image_files:
                print(f"Found {len(image_files)} test samples for {char.upper()}")
            
                for img_path in image_files:
                    img_array = self.load_and_preprocess_image(img_path)
                    if img_array is not None:
                        X.append(img_array)
                        y.append(self.char_mapping[char])
                        file_paths.append(img_path)
            else:
                print(f"No test samples found for {char.upper()}")
    
        if not X:
            print(f"\nNo test samples found in {test_dir}!")
            print("Please add test images to the appropriate directories:")
            print(f"{test_dir}/a/  - for A's")
            print(f"{test_dir}/c/  - for C's")
            print(f"{test_dir}/g/  - for G's")
            print(f"{test_dir}/t/  - for T's")
            return
    
        print(f"\nTotal test samples found: {len(X)}")
    
        X = np.array(X)
        y = np.array(y)
    
        # Select random samples
        total_samples = len(X)
        num_to_test = min(num_samples, total_samples)
        indices = np.random.choice(total_samples, num_to_test)
    
        print(f"\nTesting {num_to_test} random samples:")
        correct = 0
    
        for idx in indices:
            pred = self.model.predict(X[idx:idx+1], verbose=0)[0]
            true_char = self.reverse_mapping[y[idx]]
            pred_char = self.reverse_mapping[np.argmax(pred)]
            confidence = np.max(pred) * 100
        
            is_correct = true_char == pred_char
            correct += int(is_correct)
        
            status = "✓" if is_correct else "✗"
        
            print(f"\nTest image: {file_paths[idx].name}")
            print(f"True: {true_char.upper()}, Predicted: {pred_char.upper()}, "
              f"Confidence: {confidence:.2f}% {status}")
    
        # Print summary
        accuracy = (correct / num_to_test) * 100
        print(f"\nTest Summary:")
        print(f"Correctly predicted: {correct}/{num_to_test}")
        print(f"Accuracy: {accuracy:.2f}%")

def main():
    trainer = DNAOCRIncrementalTrainer()
    
    while True:
        print("\nDNA OCR Incremental Trainer")
        print("==========================")
        print("1. Train on new samples")
        print("2. View training history")
        print("3. Test current model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            history = trainer.incremental_train()
            if history:
                trainer.print_training_summary()  # Added parentheses here
        elif choice == '2':
            trainer.print_training_summary()
        elif choice == '3':
            trainer.test_predictions()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")
        
        input("\nPress Enter to continue...")  # Pause before showing menu again

if __name__ == "__main__":
    main()