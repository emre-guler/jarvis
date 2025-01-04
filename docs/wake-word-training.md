# Wake Word Detection Training Guide

This document provides detailed instructions for training and optimizing the wake word detection model for Jarvis.

## Overview

The wake word detection system uses a Convolutional Neural Network (CNN) trained on Mel-frequency cepstral coefficients (MFCC) features extracted from audio samples. The model is designed to detect the wake word "Jarvis" with high accuracy while minimizing false positives and computational overhead.

## Training Data Collection

There are two methods to collect training data:

### Method 1: Auto-Segmentation (Recommended)
This method allows you to record continuous audio files that are automatically segmented into training samples.

1. **Record Positive Samples**:
   ```bash
   python src/voice/training/auto_segment.py
   ```
   - Record a 2-3 minute audio file saying "Jarvis" multiple times
   - Leave 1-2 seconds gap between each utterance
   - Include variations in:
     - Volume (normal, loud, quiet)
     - Speed (normal, fast, slow)
     - Tone (formal, casual)
     - Distance from microphone
     - Accent and pronunciation

2. **Record Negative Samples**:
   - Record another 2-3 minute audio file containing:
     - Similar-sounding words ("Service", "Jarred", etc.)
     - Normal conversations
     - Background noise
     - Environmental sounds

The auto-segmenter will:
- Detect speech segments automatically
- Extract and normalize audio samples
- Save them in the appropriate directories

### Method 2: Manual Recording (Legacy)
```bash
python src/voice/training/data_collector.py
```
- Press 'p' to record positive samples
- Press 'n' to record negative samples
- Press 'q' to quit

## Training Process

1. **Prepare Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train Model**:
   ```bash
   python src/voice/training/train_model.py
   ```

3. **Test Model**:
   ```bash
   python src/voice/training/test_model.py
   ```
   - Say "Jarvis" to test detection
   - Press 'y' if detection was correct
   - Press 'n' if detection was incorrect

## Best Practices

### Recording Guidelines
1. **Positive Samples**:
   - Minimum 100 samples
   - Mix different speaking styles
   - Include various environments
   - Vary distance from microphone
   - Use different intonations

2. **Negative Samples**:
   - Minimum 200 samples
   - Include similar-sounding words
   - Record ambient noise
   - Add background conversations
   - Mix in music/media playback

### Training Tips
- Balance positive and negative samples
- Use diverse recording environments
- Include challenging scenarios
- Monitor validation accuracy
- Test in real conditions

## Performance Metrics

Target metrics for the wake word system:
- Detection latency < 500ms
- False positive rate < 1%
- False negative rate < 0.5%
- CPU usage < 5% in standby
- Accuracy > 98%

## Troubleshooting

1. **Poor Detection Rate**:
   - Add more training samples
   - Include more variations
   - Adjust model parameters
   - Check audio quality

2. **High False Positives**:
   - Add more negative samples
   - Include similar words
   - Tune detection threshold
   - Verify audio normalization

3. **System Performance**:
   - Monitor CPU usage
   - Check audio buffer size
   - Verify sample rate
   - Optimize model size

## Maintenance

Regular updates to the training data are recommended:
- Add new variations
- Include failed detections
- Update with user feedback
- Retrain periodically 