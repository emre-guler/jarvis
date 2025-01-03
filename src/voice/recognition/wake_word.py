"""
Wake Word Detection Module for Jarvis
"""
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyaudio
import tensorflow as tf
import webrtcvad
from python_speech_features import mfcc

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Wake Word Detection System for Jarvis"""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the wake word detector
        
        Args:
            model_path: Path to the wake word model file
        """
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = queue.Queue()
        self.is_running = False
        self.callback = None
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        
        # Load model
        self.model_path = model_path
        self._load_model()
        
        # Initialize audio parameters
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.format = getattr(pyaudio, "paInt16")
        
        # Detection parameters
        self.detection_threshold = WAKE_WORD_CONFIG["detection_threshold"]
        self.min_confidence = WAKE_WORD_CONFIG["min_detection_confidence"]
        self.last_detection_time = 0
        self.cooldown_period = WAKE_WORD_CONFIG["cooldown_period"]
        self.energy_threshold = {
            "max": WAKE_WORD_CONFIG["max_energy_threshold"],
            "min": WAKE_WORD_CONFIG["min_energy_threshold"]
        }

    def _load_model(self):
        """Load the wake word detection model"""
        try:
            if self.model_path and self.model_path.exists():
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Wake word model loaded from {self.model_path}")
            else:
                logger.warning("No model file found, using dummy model for testing")
                self.model = self._create_dummy_model()
            
            # Warm up the model
            dummy_input = np.zeros((1, 13, 49, 1))
            self.model.predict(dummy_input)
            
        except Exception as e:
            logger.error(f"Error loading wake word model: {e}")
            raise

    def _create_dummy_model(self):
        """Create a simple model for testing"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(13, 49, 1)),  # MFCC features
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def start(self, callback: Callable[[float], None]):
        """Start wake word detection
        
        Args:
            callback: Function to call when wake word is detected
                     Takes confidence score as parameter
        """
        if self.is_running:
            logger.warning("Wake word detector is already running")
            return

        self.callback = callback
        self.is_running = True

        try:
            # Start audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=AUDIO_SETTINGS["channels"],
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            logger.info("Wake word detector started successfully")
        except Exception as e:
            logger.error(f"Failed to start wake word detector: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop wake word detection"""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        
        try:
            self.audio.terminate()
        except Exception as e:
            logger.error(f"Error terminating audio: {e}")
        
        logger.info("Wake word detector stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        if self.is_running:
            self.audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    def _check_audio_energy(self, audio_data: np.ndarray) -> bool:
        """Check if audio energy is within acceptable range"""
        energy = np.sqrt(np.mean(audio_data**2))
        return self.energy_threshold["min"] <= energy <= self.energy_threshold["max"]

    def _extract_features(self, audio_data: bytes) -> np.ndarray:
        """Extract MFCC features from audio data
        
        Args:
            audio_data: Raw audio data bytes
            
        Returns:
            np.ndarray: MFCC features
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check energy levels
            if not self._check_audio_energy(audio_np):
                return None
            
            # Check VAD
            if not self.vad.is_speech(audio_data, self.sample_rate):
                return None
            
            # Extract MFCC features
            features = mfcc(
                audio_np,
                samplerate=self.sample_rate,
                numcep=MODEL_CONFIG["n_mfcc"],
                nfilt=MODEL_CONFIG["n_mel"],
                winlen=MODEL_CONFIG["window_size"],
                winstep=MODEL_CONFIG["hop_size"]
            )
            
            # Normalize features
            features = (features - np.mean(features)) / np.std(features)
            
            # Reshape for model input
            features = features.reshape(1, features.shape[0], features.shape[1], 1)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def _process_audio(self):
        """Process audio data and detect wake word"""
        while self.is_running:
            try:
                # Get audio data from buffer
                audio_data = self.audio_buffer.get(timeout=1.0)
                
                # Check if enough time has passed since last detection
                current_time = time.time()
                if current_time - self.last_detection_time < self.cooldown_period:
                    continue

                # Extract features
                features = self._extract_features(audio_data)
                if features is None:
                    continue
                
                # Predict
                confidence = float(self.model.predict(features, verbose=0)[0][0])
                
                # Check if wake word detected
                if confidence > self.detection_threshold:
                    logger.info(f"Wake word detected with confidence: {confidence:.2f}")
                    self.last_detection_time = current_time
                    
                    # Call callback if confidence meets minimum threshold
                    if confidence >= self.min_confidence and self.callback:
                        self.callback(confidence)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def is_active(self) -> bool:
        """Check if wake word detector is active"""
        return self.is_running

    @property
    def detection_threshold(self) -> float:
        """Get current detection threshold"""
        return self._detection_threshold

    @detection_threshold.setter
    def detection_threshold(self, value: float):
        """Set detection threshold"""
        if not 0 <= value <= 1:
            raise ValueError("Detection threshold must be between 0 and 1")
        self._detection_threshold = value 