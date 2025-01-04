# Jarvis AI Assistant

A sophisticated AI-powered personal assistant inspired by Iron Man's JARVIS, featuring voice recognition, AI interaction, system control, and personalized information services.

> **Note:** This project was architected and designed with the assistance of AI (Claude 3.5 Sonnet by Anthropic). The entire system architecture, including RFCs, documentation, and technical specifications, was carefully crafted to create a personal voice assistant inspired by J.A.R.V.I.S. from the Iron Man movies. The project aims to demonstrate the potential of modern AI technologies in creating sophisticated personal assistants.

## 🌟 Features

### Voice Interaction
- Wake word detection with "Jarvis"
  - CNN-based model with MFCC features
  - Automatic audio segmentation for training
  - Intelligent speech detection
  - < 500ms detection latency
  - < 5% CPU usage in standby
  - > 98% accuracy target
  - Real-time user feedback system
  - Environmental noise resistance
  - Multi-speaker support
  - Performance monitoring and metrics
  - Automatic energy level adjustment
- Secure speaker recognition
- Natural voice synthesis with customizable voice cloning
- Real-time voice processing

### AI Capabilities
- Advanced natural language understanding
- Context-aware conversations
- Personalized responses
- Multi-turn dialogue support
- Learning from interactions

### System Control
- Voice-controlled computer operations
- Application management
- File system navigation
- System settings control
- Process management

### Information Services
- Real-time weather updates
- Calendar management
- Personalized news delivery
- Event scheduling
- Location-based services

### Security
- Voice biometric authentication
- End-to-end encryption
- Secure data storage
- Access control
- Privacy protection

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Modern multi-core processor
- Minimum 16GB RAM
- High-quality microphone
- Stable internet connection
- GPU recommended for optimal performance

### Installation

1. Clone the repository:
```bash
git clone https://github.com/emre-guler/jarvis.git
cd jarvis
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

5. Train the wake word model:
```bash
# Method 1: Auto-Segmentation (Recommended)
python src/voice/training/auto_segment.py
# Follow the prompts to record continuous audio samples

# Method 2: Manual Recording
python src/voice/training/data_collector.py
# Press 'p' for positive samples (saying "Jarvis")
# Press 'n' for negative samples (other sounds)
# Press 'q' to quit

# Train the model
python src/voice/training/train_model.py
```

6. Run initial setup:
```bash
python scripts/setup.py
```

7. Start Jarvis:
```bash
python scripts/run.py
```

## 🛠️ Architecture

### Core Components
1. **Voice Processing**
   - Wake word detection
   - Speaker recognition
   - Voice synthesis

2. **AI Engine**
   - LLM integration
   - Context management
   - Knowledge base

3. **System Interface**
   - Computer control
   - Application management
   - File operations

4. **Information Services**
   - Weather integration
   - Calendar sync
   - News aggregation

5. **Security Framework**
   - Authentication
   - Encryption
   - Access control

## 📊 Performance

- Wake word detection < 500ms
- Voice authentication < 1s
- Command execution < 2s
- System resource usage < 20% CPU
- Memory usage < 2GB

## 🔒 Security

- Biometric voice authentication
- End-to-end encryption
- Secure API key storage
- Regular security audits
- Privacy-first design

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

Test wake word detection:
```bash
# Run wake word detection tests
pytest tests/voice/test_wake_word.py -v

# Start wake word detection
python scripts/run_wake_word.py

# After each detection:
# Press 'y' if detection was correct
# Press 'n' if detection was incorrect
# Press Ctrl+C to stop
```

Performance metrics are automatically collected and saved in `data/metrics/`.

## 📚 Documentation

- [Product Requirements Document](PRD.md)
- [Feature Specifications](FEATURES.md)
- [API Documentation](docs/API.md)
- [RFCs](rfcs/)
- [Wake Word Training Guide](docs/wake-word-training.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Coqui for TTS
- OpenWeatherMap for weather data
- Various open-source contributors

## 🔄 Version History

- v0.1.0 - Initial development version
- v0.2.0 - Wake word detection system
  - CNN-based wake word model
  - Real-time user feedback
  - Performance monitoring
  - Environmental testing
  - Resource usage optimization
- Future releases TBD

## ⚠️ Disclaimer

This is an experimental project. Use at your own risk. Not recommended for production use without proper security review. 