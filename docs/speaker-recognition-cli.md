# Speaker Recognition CLI Guide

## Overview
The Speaker Recognition CLI provides a comprehensive set of commands to manage voice profiles, enroll new users, and verify speaker identities. The system includes advanced features such as continuous authentication, emotion recognition, and voice aging adaptation.

## Installation
Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

## Available Commands

### 1. Enroll a New User
Enroll a new user with voice samples.
```bash
python -m src.voice.recognition.cli enroll USER_ID --samples NUMBER_OF_SAMPLES
```
- `USER_ID`: Your chosen identifier (e.g., username)
- `--samples`: Number of voice samples to collect (default: 5)

Example:
```bash
python -m src.voice.recognition.cli enroll john_doe --samples 5
```

### 2. Verify Speaker Identity
Verify if a speaker matches their enrolled voice profile.
```bash
python -m src.voice.recognition.cli verify USER_ID [--continuous]
```
Options:
- `--continuous`: Enable continuous authentication
- `--emotion`: Include emotion detection
- `--adapt`: Enable voice adaptation

Example:
```bash
python -m src.voice.recognition.cli verify john_doe --continuous --emotion
```

### 3. Add More Voice Samples
Add additional voice samples to an existing profile.
```bash
python -m src.voice.recognition.cli add-samples USER_ID
```
- You can add multiple samples
- Each sample will be checked for similarity with existing profile
- Continue adding samples until satisfied

### 4. List Enrolled Users
Display all enrolled users and their profile details.
```bash
python -m src.voice.recognition.cli list-users
```
This shows:
- User IDs
- Number of samples
- Creation date
- Last update date
- Adaptation history
- Additional metadata

### 5. Delete User Profile
Remove a user's voice profile from the system.
```bash
python -m src.voice.recognition.cli delete USER_ID
```

## Advanced Features

### Continuous Authentication
- Real-time speaker verification during sessions
- Configurable verification interval
- Automatic session termination on verification failure
- Background processing for seamless experience

### Emotion Recognition
- Detects 7 emotional states:
  - Neutral
  - Happy
  - Sad
  - Angry
  - Fearful
  - Surprised
  - Disgusted
- Provides emotion confidence scores
- Real-time emotion tracking

### Voice Adaptation
- Automatic profile updates for aging voices
- Drift detection and compensation
- Historical adaptation tracking
- Configurable adaptation thresholds

## Best Practices

### Recording Voice Samples
1. Maintain consistent distance from microphone (about 6-8 inches)
2. Speak clearly and naturally
3. Vary your tone slightly between samples
4. Record in a quiet environment
5. Wait for the countdown before speaking
6. Continue speaking until "Done recording!" appears

### Profile Management
- Collect at least 5 voice samples for reliable recognition
- Add more samples if verification accuracy is low
- Enable voice adaptation for long-term reliability
- Keep profiles updated with periodic new samples

### Troubleshooting
If you encounter audio recording issues:
1. Check your microphone connection
2. Ensure microphone permissions are granted
3. Try reducing background noise
4. Restart the application if issues persist
5. Check system audio settings

## Security Features
- Voice profiles are stored locally in the `data/profiles` directory
- Each profile is encrypted using industry-standard encryption
- Anti-spoofing detection protects against:
  - Replay attacks
  - Synthetic speech
  - Audio manipulation
- Continuous authentication ensures ongoing security
- Regular security audits and updates recommended

## Performance Specifications
- Enrollment time: ~15-20 seconds for 5 samples
- Verification time: < 1 second
- Recognition accuracy: > 99% in ideal conditions
- False acceptance rate (FAR): < 0.1%
- False rejection rate (FRR): < 1%
- CPU usage: Typically < 30%
- Memory usage: < 100MB
- Emotion detection accuracy: > 90%
- Adaptation accuracy: > 95%

## Testing
### Running Tests
Execute the test suite using pytest:
```bash
# Run all tests
python -m pytest tests/voice/ -v

# Run specific test categories
python -m pytest tests/voice/test_security.py -v     # Security tests
python -m pytest tests/voice/test_performance.py -v  # Performance tests
python -m pytest tests/voice/test_emotion.py -v      # Emotion recognition tests
python -m pytest tests/voice/test_adaptation.py -v   # Voice adaptation tests
```

### Test Coverage
The test suite includes:
1. Security Tests (11 tests)
   - Profile encryption/decryption
   - Anti-spoofing detection
   - Key generation
   - Replay attack prevention
   
2. Performance Tests (7 tests)
   - System initialization
   - Verification metrics
   - Processing time
   - System metrics collection
   - Performance under load
   - Settings optimization

3. Feature Tests
   - Continuous authentication
   - Emotion recognition
   - Voice adaptation
   - Profile management

4. Benchmark Tests
   - CPU usage monitoring
   - Memory consumption
   - Processing speed
   - Real-time performance

### Continuous Integration
- All tests must pass before deployment
- Performance benchmarks are tracked over time
- Security tests are required for each update
- Automated testing pipeline ensures quality 