# Virtual Piano with Finger Detection

## Overview
This project implements a virtual piano that detects finger positions using computer vision and plays corresponding notes in real-time. The system uses a CNN model to classify whether a finger is pressing a piano key based on camera input.

## Features
- Real-time finger detection on piano keys
- 8-note piano (C4 to C5) with sound playback
- Perspective correction for accurate key detection
- CNN-based classification model (~100% test accuracy)
- GPU acceleration support (falls back to CPU if GPU not available)

## Technical Details
### Model Architecture
- Input: 125x700 RGB images of piano keys
- 3 Conv2D layers with ReLU activation
- 2 MaxPooling2D layers
- 2 Dense layers (final sigmoid activation for binary classification)
- Trained for 10 epochs (achieved 100% test accuracy)

### Computer Vision Pipeline
1. Camera feed capture
2. Edge detection and contour finding
3. Perspective correction to align piano keys
4. Region-of-interest extraction for each key
5. CNN inference to detect finger presence
6. Sound playback for pressed keys

## Requirements
- Python 3.10+
- TensorFlow 2.x
- OpenCV
- Pygame
- NumPy

## Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download sound files (AIFF format) for each note (C4.aiff, D4.aiff, etc.)
4. Place sound files in `sounds/` directory
5. Run the notebook: `jupyter notebook Virtual_Piano_CUBA.ipynb`

## Usage
1. Point camera at piano keyboard
2. Press 's' to lock perspective when keyboard is detected
3. Virtual piano will play notes when fingers are detected
4. Press 'q' to quit

## Dataset
- 105 finger images
- 106 no-finger images
- Train/Val/Test split: 70%/20%/10%