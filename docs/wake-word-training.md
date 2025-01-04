# Wake Word Model Training Guide

This guide explains how to train your own wake word detection model for Jarvis.

## Overview

The wake word detection system uses a Convolutional Neural Network (CNN) trained on MFCC (Mel-frequency cepstral coefficients) features to detect when someone says "Jarvis". The model is designed to be:
- Fast (< 500ms detection time)
- Accurate (> 95% accuracy)
- Resource-efficient (< 5% CPU usage)

## Training Process

### 1. Data Collection

Run the data collector:
```bash
python src/voice/training/data_collector.py
```

#### Controls
- Press 'p' to record a positive sample (saying "Jarvis")
- Press 'n' to record a negative sample (other words/noise)
- Press 'q' to quit

#### Requirements
- **Positive Samples**: At least 100 recordings of "Jarvis"
  - Different voice tones (high, low, neutral)
  - Different speaking speeds
  - Different distances from microphone
  - Different emotional states (calm, excited, tired)
  
- **Negative Samples**: At least 200 recordings
  - Similar-sounding words
  - Common background noises
  - Regular conversation
  - Music or TV sounds
  - Other people speaking

#### Tips for Quality Data
1. **Environment**
   - Record in your typical usage environment
   - Include typical background noise
   - Use your regular microphone setup

2. **Variation**
   - Record at different times of day
   - Include different voice variations
   - Move around while recording
   - Include both clear and slightly muffled speech

3. **Similar Words**
   - Record similar-sounding words as negative samples
   - Include names that start with "J"
   - Include words that rhyme with "Jarvis"

### 2. Model Training

Run the training script:
```bash
python src/voice/training/train_model.py
```

The script will:
1. Process all collected audio samples
2. Extract MFCC features
3. Train a CNN model
4. Save the best model to `models/wake_word_model.h5`

#### Training Parameters
- Epochs: 50 (default)
- Batch Size: 32 (default)
- Validation Split: 20%
- Early Stopping: 5 epochs patience

You can modify these in `src/voice/training/train_model.py` if needed.

### 3. Validation

The training script will output:
- Training accuracy
- Validation accuracy
- Model size
- Training time

Aim for:
- Validation accuracy > 95%
- False positive rate < 1%
- False negative rate < 0.5%

### 4. Integration

The trained model will automatically be used by the wake word detector. No additional setup is required.

## Troubleshooting

### Common Issues

1. **Poor Detection Accuracy**
   - Collect more training data
   - Ensure good quality recordings
   - Add more variety to samples
   - Check microphone quality

2. **Slow Detection**
   - Reduce model complexity
   - Check system resources
   - Ensure proper audio setup

3. **High False Positives**
   - Add more negative samples
   - Include similar-sounding words
   - Increase detection threshold

4. **High False Negatives**
   - Add more positive samples
   - Include voice variations
   - Decrease detection threshold

## Advanced Customization

The model architecture and training parameters can be customized in:
- `src/voice/training/train_model.py` - Model architecture and training
- `config/settings.py` - Detection thresholds and audio parameters 