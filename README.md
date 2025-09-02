# Facial Emotion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-74.43%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*Real-time emotion detection with bias correction*

[Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation)

</div>

---

## Overview

This facial emotion detection system provides real-time emotion recognition using deep learning. Built with PyTorch and optimized for CPU inference on all PCs, it achieves 99.43% accuracy on the FER2013 dataset with intelligent bias correction for improved real-world performance.

## Model Download

[https://huggingface.co/Tricc/facial-emotion-detector/tree/main](https://huggingface.co/Naweiner/facial-emotion-detector/tree/main)

## Key Features

- **74.43% Accuracy** - Trained on FER2013 dataset
- **Universal Compatibility** - Works on any operating system/PC
- **Live Camera Interface** - Real-time emotion detection from webcam
- **Screenshot Capture** - Save detection results with timestamps

## Features

### Emotion Recognition
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Real-time confidence scoring for each emotion
- Visual emotion probability bars
- Sadness may be difficult for the model to recognize due to the bias correction

### AI Model
- ResNet18-based CNN architecture with 28M parameters
- Sad bias reduction to prevent false positives
- Neutral emotion boosting for balanced detection

### User Interface
- Real-time video display with emotion overlay
- Performance metrics
- Keyboard controls for screenshots
- Emotion confidence visuals

## Quick Start

### Prerequisites
```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/nawjasinghe/facial-emotion-detector.git
   cd facial-emotion-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Windows (recommended)
   ./run_emotion_detector.bat
   
   # Or manually
   python emotion_detector.py
   ```

## Controls

| Key | Action |
|-----|--------|
| **S** | Save screenshot |
| **Q** | Quit application |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT CAMERA FEED                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 FACE DETECTION                              │                         
│   • Fallback detection methods                             │
│   • Continuous face tracking                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              EMOTION RECOGNITION                            │
│   • ResNet18-based CNN (28M parameters)                    │
│   • 74.43% base accuracy                                   │
│   • 7 emotion categories                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              BIAS CORRECTION                                │
│   • 60% sad emotion reduction                               │
│   • 70% neutral emotion boosting                            │                       
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               DISPLAY OUTPUT                                │
│   • Video feed with emotion overlay                        │
│   • Emotion confidence bars                                │
│   • Performance metrics                                    │
│   • Screenshot capture                                     │
└─────────────────────────────────────────────────────────────┘
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 74.43% |
| **Processing Speed** | 10 FPS |
| **Model Size** | 28M parameters |
| **Memory Usage** | ~2GB RAM |

### Bias Correction Algorithm
```python
# Bias correction implementation
sad_probability *= 0.4  # 60% reduction
neutral_probability += (original_sad * 0.6 * 0.7)  # 70% boost
```

### Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+ (CPU version)
- **OpenCV**: 4.0+
- **Memory**: 2GB+ RAM recommended

## Project Structure

```
facial-emotion-detection/
├── emotion_detector.py              # Main application
├── emotion_model.pth                # Trained model
├── run_emotion_detector.bat         # Quick launcher
├── requirements.txt                 # Dependencies
├── README.md                        # Documentation
```

## Development

This system was developed using machine learning techniques:

- **Data Augmentation**: Rotation, scaling, brightness adjustment
- **Transfer Learning**: ResNet18 pre-trained backbone
- **Bias Analysis**: Statistical analysis of emotion distribution
- **Optimization**: Frame rate control and memory management

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools
- FER2013 dataset contributors
- Research community for emotion recognition advances

---

<div align="center">

**Star this repository if you found it neat!**

</div>
