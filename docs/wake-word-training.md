# Wake Word Training Guide

## Overview
This guide explains how to prepare and process training data for the wake word detection system, including the use of the auto-segmentation feature for processing long audio recordings.

## Prerequisites
- Python 3.9 or higher
- Required packages installed (see requirements.txt)
- Microphone for recording
- Quiet environment for recording

## Recording Guidelines

### Positive Samples (Wake Word "Jarvis")
1. Record in your typical usage environment
2. Speak clearly with ~1 second pause between each "Jarvis"
3. Vary your:
   - Speaking speed
   - Tone
   - Volume
   - Distance from microphone
4. Include some background noise variations
5. Aim for at least 100 samples

### Negative Samples (Non-Wake Words)
1. Include similar-sounding words
2. Record common background noises
3. Include typical commands and phrases
4. Vary environmental conditions
5. Aim for at least 200 samples

## Auto-Segmentation Feature

### Overview
The auto-segmentation feature automatically processes long audio recordings into individual word samples, making it easy to create a large dataset of training samples.

### Key Features
- Automatic word boundary detection
- Clean word isolation
- Consistent sample length
- Intelligent silence detection
- Quality assurance checks

### Parameters
- Minimum word length: 300ms
- Maximum word length: 1.2s
- Pre/post padding: 100ms each
- Silence threshold: -45dB
- Sample rate: 16kHz
- Channels: Mono

### Usage

1. **Recording Long Audio Files**
   ```bash
   # Record positive samples (multiple "Jarvis" utterances)
   rec data/audio/wake_word/jarvis_samples.wav
   
   # Record negative samples (other words/phrases)
   rec data/audio/wake_word/negative_samples.wav
   ```

2. **Running Auto-Segmentation**
   ```bash
   python src/voice/training/auto_segment.py
   ```
   - Follow the prompts to process your audio files
   - The script will automatically:
     1. Detect word boundaries
     2. Extract clean samples
     3. Save to appropriate directories
     4. Show progress and statistics

3. **Verifying Samples**
   - Check the output directories:
     - `data/audio/wake_word/positive/`
     - `data/audio/wake_word/negative/`
   - Listen to a few samples to ensure quality
   - Verify clean word boundaries
   - Check sample counts

### Tips for Better Results
1. Record in a relatively quiet environment
2. Leave clear pauses between words
3. Maintain consistent volume
4. Vary your speaking style naturally
5. Include some background noise
6. Monitor the segmentation output
7. Verify sample quality regularly

## Training Process

1. **Prepare Training Data**
   - Use auto-segmentation for long recordings
   - Manually record individual samples if preferred
   - Verify sample quality and counts

2. **Train the Model**
   ```bash
   python src/voice/training/train_model.py
   ```

3. **Test the Model**
   ```bash
   python src/voice/training/test_model.py
   ```

## Troubleshooting

### Common Issues
1. **Poor Segmentation**
   - Ensure clear pauses between words
   - Adjust silence threshold if needed
   - Check recording volume levels

2. **Missing Words**
   - Words might be too short/long
   - Adjust length parameters
   - Check silence threshold

3. **Extra Segments**
   - Background noise too high
   - Silence threshold too low
   - Insufficient pauses between words

### Parameter Adjustment
If needed, adjust these parameters in `auto_segment.py`:
```python
self.min_silence_len = 0.2     # minimum silence between words
self.silence_thresh = -45      # silence threshold in dB
self.min_word_length = 0.3     # minimum word length
self.max_word_length = 1.2     # maximum word length
```

## Best Practices
1. Start with default parameters
2. Monitor segmentation results
3. Adjust parameters if needed
4. Verify sample quality
5. Maintain consistent recording conditions
6. Back up your original recordings
7. Document any parameter changes

## Next Steps
1. Train your wake word model
2. Test in various conditions
3. Fine-tune as needed
4. Deploy for real-world use 