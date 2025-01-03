"""
Jarvis Configuration Settings
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "buffer_duration": 0.5,  # seconds
    "format": "int16",
}

# Wake word settings
WAKE_WORD_CONFIG = {
    "wake_word": "jarvis",
    "detection_threshold": 0.5,
    "min_detection_confidence": 0.8,
    "detection_timeout": 0.5,  # seconds
    "cooldown_period": 1.0,  # seconds
    "max_energy_threshold": 4000,
    "min_energy_threshold": 100,
}

# Model settings
MODEL_CONFIG = {
    "feature_type": "mfcc",
    "n_mfcc": 13,
    "n_mel": 40,
    "window_size": 0.025,  # seconds
    "hop_size": 0.010,  # seconds
    "model_type": "cnn",
} 