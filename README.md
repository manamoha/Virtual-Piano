# Virtual Piano with Finger Detection

## Overview
This project implements a virtual piano that detects finger positions using computer vision and plays corresponding notes in real-time. The system includes two main components:
1. A dataset builder tool (`building_dataset.py`)
2. The virtual piano application (`Virtual_Piano_CUBA.ipynb`)

## Features
- **Dataset Builder**:
  - Captures images of piano keys with/without finger presence
  - Automatic perspective correction
  - Saves labeled images for training
- **Virtual Piano**:
  - Real-time finger detection on piano keys
  - 8-note piano (C4 to C5) with sound playback
  - CNN-based classification model (~100% test accuracy)
  - GPU acceleration support

## Setup Instructions

### 1. Dataset Preparation
Run `building_dataset.py` first to create your training dataset:
```bash
python building_dataset.py
```
- Point your camera at the piano keyboard
- Press 's' to lock perspective when keyboard is detected
- Press 'f' to save images with finger present (saves to `dataset/finger/C4/`)
- Press 'n' to save images without finger (saves to `dataset/no_finger/C4/`)
- Collect at least 100 images for each class

### 2. Virtual Piano Installation
1. Clone the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Download sound files (AIFF format) for each note (C4.aiff to C5.aiff)
4. Place sound files in `sounds/` directory

### 3. Run the Virtual Piano
```bash
jupyter notebook Virtual_Piano_CUBA.ipynb
```

## Technical Details

### Dataset Builder (`building_dataset.py`)
- Uses OpenCV for image capture and processing
- Automatic perspective correction with 4-point transform
- Saves 125x700px images of individual keys
- Currently captures only C4 key (modify to capture all keys)

### Virtual Piano (`Virtual_Piano_CUBA.ipynb`)
- **Model Architecture**:
  - 3 Conv2D layers with ReLU activation
  - 2 MaxPooling2D layers
  - 2 Dense layers (final sigmoid activation)
  - Trained for 10 epochs (achieved 100% test accuracy)

- **Computer Vision Pipeline**:
  1. Camera feed capture
  2. Edge detection and contour finding
  3. Perspective correction
  4. Region-of-interest extraction
  5. CNN inference
  6. Sound playback

## Directory Structure
```
project/
├── dataset/
│   ├── finger/
│   │   └── C4/
│   └── no_finger/
│       └── C4/
├── sounds/
│   ├── C4.aiff
│   ├── D4.aiff
│   └── ... 
├── building_dataset.py
├── Virtual_Piano_CUBA.ipynb
└── requirements.txt
```
