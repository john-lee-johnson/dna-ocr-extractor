# DNA Sequence Image Processing and OCR System

## Overview
This system processes DNA sequence images through three main steps:
1. Training data collection
2. Model training
3. Sequence extraction and correction

## Step 1: Image Processing for Training Data Collection
Run `dna_training_image_processor.py` to collect training examples.

1. Select an image containing DNA sequences
2. Create training examples:
  - Save 30+ examples of each letter (A, C, G, T) for training
  - Save 5+ examples of each letter for testing
  - The training and test folders:
    - `dna_training_data/[a/c/g/t]/`
    - `dna_test_data/[a/c/g/t]/`

### Keyboard Shortcuts for Image Processor
- `A`: Switch to collecting A's
- `C`: Switch to collecting C's
- `G`: Switch to collecting G's
- `T`: Switch to collecting T's
- `M`: Toggle between train/test mode
- `+/-`: Adjust selection box size
- Left Click: Select and save character
- ESC: Exit

## Step 2: Model Training
Run `dna_ocr_incremental_training.py` to:
- Train a new model with collected images
- Use newly images and retrain
- Test model performance

## Step 3: DNA Sequence Extraction
Run `dna_sequence_extractor.py` to process clean DNA sequence images.

### Initial Processing
If characters are touching or bulky (i.e., serif fonts):
1. Enable character separation options in GUI
2. Adjust parameters until characters are properly separated

If characters are well-separated (i.e., sans-serif fonts:
- Leave separation options disabled

### Visualization Steps
The system will show several processing steps:
1. Original image
2. Preprocessing stages
3. Line identification
4. Character segmentation
5. Final sequence visualization

### Sequence Editor
In the final sequence editor window:

#### Color Coding
- **Green**: High confidence calls (>90%)
- **Orange**: Split multi-character boxes
- **Red**: Low confidence calls (<90%)
- **Cyan highlight**: Currently selected box

#### Keyboard Shortcuts
- Click/Tab: Select box
- A/C/G/T: Change letter
- Backspace/D: Delete box
- U: Undo
- Tab/Shift-Tab: Navigate boxes
- H: Toggle help overlay
- Enter: Finish
- Esc: Cancel

### Output
- Sequence is automatically copied to clipboard
- Optional export to JSON/CSV/TXT in 'exports' folder

## Tips
1. Collect diverse training examples
2. Test model performance before processing important sequences
3. Use the sequence editor to correct any misidentifications
4. Always verify critical sequences
5. If model performance is low, delete all models and retrain the full model

## Requirements
```python
opencv-python
numpy
tensorflow
PIL
tkinter
File Structure
Copy├── dna_training_data/
│   ├── a/
│   ├── c/
│   ├── g/
│   └── t/
├── dna_test_data/
│   ├── a/
│   ├── c/
│   ├── g/
│   └── t/
├── exports/
├── model/
├── dna_training_image_processor.py
├── dna_ocr_incremental_training.py
└── dna_sequence_extractor.py
