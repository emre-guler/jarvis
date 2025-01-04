"""
Configuration settings for Jarvis
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
AUDIO_DIR = DATA_DIR / "audio"
METRICS_DIR = DATA_DIR / "metrics"

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "Int16",
    "input_device": None,  # None uses default input device
}

# Wake word detection settings
WAKE_WORD_CONFIG = {
    "detection_threshold": 0.5,  # Confidence threshold for wake word detection
    "min_detection_confidence": 0.6,  # Minimum confidence for callback trigger
    "cooldown_period": 1.0,  # Seconds to wait between detections
    "max_energy_threshold": 0.0,  # Maximum energy threshold in dB
    "min_energy_threshold": -65.0,  # More lenient minimum energy threshold in dB
    "model_path": MODELS_DIR / "wake_word_model.keras",
    "positive_samples_dir": AUDIO_DIR / "wake_word/positive",
    "negative_samples_dir": AUDIO_DIR / "wake_word/negative",
    "sample_duration": 2.0,  # Duration of each training sample in seconds
    "training_epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
}

# Performance monitoring settings
MONITORING_CONFIG = {
    "metrics_dir": METRICS_DIR,
    "log_interval": 60,  # Seconds between metrics logging
    "save_interval": 300,  # Seconds between metrics saving
    "max_detection_history": 1000,  # Maximum number of detection events to keep
} 