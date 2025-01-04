# Wake Word Detection Training Guide

This document provides detailed instructions for training and optimizing the wake word detection model for Jarvis.

## Overview

The wake word detection system uses a Convolutional Neural Network (CNN) trained on Mel-frequency cepstral coefficients (MFCC) features extracted from audio samples. The model is designed to detect the wake word "Jarvis" with high accuracy while minimizing false positives and computational overhead.

## System Requirements

- Python 3.8 or higher
- PyAudio with working microphone input
- TensorFlow 2.5 or higher
- 500MB+ free disk space for audio samples
- 2GB+ RAM recommended

## Data Collection

### Running the Data Collector

```bash
python src/voice/training/data_collector.py
```

### Controls
- `p`: Record a positive sample (saying "Jarvis")
- `n`: Record a negative sample (other words/noise)
- `q`: Quit the data collection session

### Sample Requirements

Minimum recommended samples:
- 100+ positive samples (saying "Jarvis")
- 200+ negative samples (other words/background noise)

For better model performance, aim for:
- 500+ positive samples
- 1000+ negative samples

### Recording Guidelines

1. Positive Samples:
   - Vary your speaking speed
   - Use different intonations
   - Record from different distances
   - Include different accents if possible
   - Mix formal/casual pronunciations

2. Negative Samples:
   - Similar-sounding words (e.g., "Jarred", "Service")
   - Common background noises
   - Conversations without the wake word
   - Music and media playback
   - Household sounds

3. Environmental Considerations:
   - Record in different rooms
   - Include samples with background noise
   - Mix quiet and noisy environments
   - Various times of day
   - Different microphone positions

## Model Training

### Running the Training Script

```bash
python src/voice/training/train_model.py
```

### Training Parameters

Default parameters (configurable in `config/settings.py`):
- Epochs: 50
- Batch size: 32
- Validation split: 20%
- Learning rate: 0.001
- Early stopping patience: 5

### Model Architecture

The CNN model consists of:
1. Input layer (MFCC features)
2. Convolutional layers with max pooling
3. Dropout for regularization
4. Dense layers
5. Binary classification output

### Training Process

1. Data Preprocessing:
   - Audio normalization
   - MFCC feature extraction
   - Feature standardization
   - Data augmentation (optional)

2. Model Training:
   - Cross-validation
   - Early stopping
   - Model checkpointing
   - Performance monitoring

3. Evaluation:
   - Accuracy metrics
   - Confusion matrix
   - ROC curve
   - Resource usage stats

## Performance Optimization

### Accuracy Improvement

If detection accuracy is low:
1. Collect more training data
2. Increase model complexity
3. Adjust detection threshold
4. Implement data augmentation
5. Try transfer learning

### Latency Optimization

If detection is slow:
1. Reduce model size
2. Optimize chunk size
3. Enable TensorFlow optimizations
4. Adjust feature extraction parameters
5. Profile and optimize bottlenecks

### Resource Usage

To minimize resource usage:
1. Use quantized models
2. Optimize buffer sizes
3. Implement batch processing
4. Enable TensorFlow lite
5. Profile memory usage

## Monitoring and Metrics

The system tracks:
1. Detection accuracy
2. Processing latency
3. CPU/memory usage
4. False positive rate
5. Energy levels

View metrics in:
- Real-time logs
- JSON reports in `data/metrics/`
- Performance graphs (if enabled)

## Troubleshooting

### Common Issues

1. Poor Detection Rate:
   - Insufficient training data
   - Unbalanced dataset
   - Noisy training samples
   - Incorrect threshold

2. High Latency:
   - Model too complex
   - Resource contention
   - Buffer size issues
   - System overload

3. Resource Usage:
   - Memory leaks
   - Excessive logging
   - Background processes
   - Unoptimized model

### Solutions

1. Model Issues:
   - Retrain with more data
   - Adjust model architecture
   - Tune hyperparameters
   - Validate training data

2. System Issues:
   - Check audio setup
   - Monitor resource usage
   - Optimize configurations
   - Update dependencies

3. Performance Issues:
   - Profile the system
   - Optimize bottlenecks
   - Adjust buffer sizes
   - Enable optimizations

## Best Practices

1. Data Collection:
   - Regular data updates
   - Diverse sample sources
   - Quality validation
   - Proper labeling

2. Training:
   - Cross-validation
   - Regular evaluation
   - Version control
   - Parameter tuning

3. Deployment:
   - Gradual rollout
   - Performance monitoring
   - Regular updates
   - User feedback

4. Maintenance:
   - Regular retraining
   - Performance audits
   - System updates
   - Documentation updates 