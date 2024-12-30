# Jarvis - Personal Voice Assistant

> **Note**: This project is a personal voice assistant inspired by Jarvis from the Iron Man movies. It is designed to work on MacOS.

## Features

- Continuous voice listening and speaker recognition
- Customized voice responses
- System control (brightness, volume, applications, etc.)
- Conversation history
- Speaker verification

## Requirements

- Python 3.8+
- PyAudio
- OpenAI Whisper
- TTS (Text-to-Speech)
- Other requirements are listed in requirements.txt

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Create your voice profile:
- When you first run Jarvis, you'll need to create a profile for voice recognition
- You'll be asked to speak for a few seconds

## Usage

To start the assistant:
```bash
python jarvis.py
```

## Security

- System commands may require sudo privileges
- Voice profile data is stored locally
- All processing is done locally

## License

MIT
