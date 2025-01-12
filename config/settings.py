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
    "chunk_size": 4096,
    "format": "Float32",
    "input_device": None,  # None uses default input device
    "input_overflow_handler": "ignore",  # Handle overflow
    "input_latency": 0.2,  # Added latency buffer
    "max_recording_time": 5  # Maximum recording time in seconds
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
    "log_interval": 1,  # Seconds between metrics logging (reduced from 60)
    "save_interval": 60,  # Seconds between metrics saving (reduced from 300)
    "max_detection_history": 1000,  # Maximum number of detection events to keep
}

# Model settings
MODEL_CONFIG = {
    "emotion": {
        "model_path": MODELS_DIR / "emotion_model.keras",
        "input_shape": (220,),
        "num_classes": 7,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "validation_split": 0.2,
    }
} 