"""
Wake Word Detection System for Jarvis
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

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG
from src.voice.monitoring.metrics import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Wake Word Detection System for Jarvis"""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the wake word detector
        
        Args:
            model_path: Path to the wake word model file
        """
        self.audio = pyaudio.PyAudio()
        
        # List available audio devices
        logger.info("\n=== Available Audio Devices ===")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            logger.info(f"Device {i}: {device_info['name']}")
            if device_info['maxInputChannels'] > 0:
                logger.info(f"    Input channels: {device_info['maxInputChannels']}")
                logger.info(f"    Default sample rate: {device_info['defaultSampleRate']}")
        logger.info("="*30 + "\n")
        
        self.stream = None
        self.audio_buffer = queue.Queue()
        self.accumulated_audio = bytearray()  # Buffer to accumulate audio chunks
        self.is_running = False
        self.callback = None
        
        # Initialize VAD with higher aggressiveness to reduce false positives
        self.vad = webrtcvad.Vad(3)  # Increased from 2 to 3 for better voice detection
        
        # Load model
        self.model_path = model_path
        self._load_model()
        
        # Initialize audio parameters
        self.target_sample_rate = AUDIO_SETTINGS["sample_rate"]  # Target sample rate for model
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.format = getattr(pyaudio, "paInt16")
        
        # Detection parameters
        self.detection_threshold = WAKE_WORD_CONFIG["detection_threshold"]
        self.min_confidence = WAKE_WORD_CONFIG["min_detection_confidence"]
        self.last_detection_time = 0
        self.cooldown_period = WAKE_WORD_CONFIG["cooldown_period"]
        
        # Performance optimization parameters
        self.process_every_n_chunks = 2  # Process every other chunk
        self.chunk_counter = 0
        self.batch_size = 4
        self.processing_batch = []
        
        # Stricter energy thresholds
        self.energy_threshold = {
            "max": 0.0,    # 0 dB (maximum)
            "min": -55.0   # Adjusted for better sensitivity
        }
        
        # Power saving settings
        self.power_mode = "balanced"  # Balance between performance and power
        self.processing_interval = 0.05  # 50ms interval
        self.active_processing = True  # Flag for active processing
        
        # Initialize performance monitoring
        self.monitor = PerformanceMonitor()
        self.last_metrics_time = time.time()
        self.metrics_interval = 1.0  # Record metrics every second
        
        # Start periodic metrics logging
        self.metrics_thread = threading.Thread(target=self._log_metrics_periodically)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()

    def _create_dummy_model(self):
        """Create a simple model for testing"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(200, 13, 1)),  # Match feature shape
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def _load_model(self):
        """Load the wake word detection model"""
        try:
            model_path = Path("models/wake_word_model.keras")
            if model_path.exists():
                try:
                    self.model = tf.keras.models.load_model(str(model_path))
                    logger.info(f"Wake word model loaded from {model_path}")
                except Exception as e:
                    logger.warning(f"Error loading saved model: {e}")
                    logger.info("Creating new model...")
                    self.model = self._create_dummy_model()
            else:
                logger.warning("No trained model found, creating new model...")
                self.model = self._create_dummy_model()
            
            # Warm up the model
            dummy_input = np.zeros((1, 200, 13, 1))  # Shape matches our training data
            self.model.predict(dummy_input, verbose=0)
            
        except Exception as e:
            logger.error(f"Error loading wake word model: {e}")
            raise

    def start(self, callback: Callable[[float], None]):
        """Start wake word detection
        
        Args:
            callback: Function to call when wake word is detected
        """
        if self.is_running:
            return
            
        self.callback = callback
        self.is_running = True
        self.active_processing = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=1,
            rate=self.target_sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Print status
        logger.info("\n==================================================")
        logger.info("🎤 Wake Word Detection Active")
        logger.info("Say 'Jarvis' to trigger")
        logger.info("Press Ctrl+C to stop")
        logger.info("==================================================\n")
        
        self.monitor.start_time = time.time()  # Reset monitor start time
        
    def _process_audio_stream(self):
        """Process audio stream for wake word detection"""
        while self.is_running:
            try:
                # Power saving: Sleep when not actively processing
                if not self.active_processing:
                    time.sleep(0.1)  # Longer sleep when inactive
                    continue
                    
                # Get audio chunk from buffer
                audio_chunk = self.audio_buffer.get(timeout=1.0)
                
                # Process every nth chunk
                self.chunk_counter += 1
                if self.chunk_counter % self.process_every_n_chunks != 0:
                    continue
                    
                # Check energy level
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                if not self._check_audio_energy(audio_data):
                    self.active_processing = False  # Reduce processing when no speech
                    continue
                    
                self.active_processing = True  # Resume active processing
                
                # Extract features
                features = self._extract_features(audio_chunk)
                if features is None:
                    continue
                    
                # Add to batch
                self.processing_batch.append(features)
                
                # Process batch
                if len(self.processing_batch) >= self.batch_size:
                    batch = np.array(self.processing_batch)
                    prediction = self.model.predict(batch, verbose=0)
                    
                    # Check for wake word
                    max_confidence = np.max(prediction)
                    if max_confidence > self.min_confidence:
                        current_time = time.time()
                        if current_time - self.last_detection_time > self.cooldown_period:
                            self.last_detection_time = current_time
                            if self.callback:
                                self.callback(max_confidence)
                                
                    # Clear batch
                    self.processing_batch = []
                    
                # Power saving: Sleep between processing
                time.sleep(self.processing_interval)
                
                # Record and save metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_time >= self.metrics_interval:
                    self.monitor.record_system_metrics()
                    self.monitor.save_metrics()  # Save metrics to file
                    self.last_metrics_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def _handle_keyboard_input(self):
        """Handle keyboard input for detection feedback"""
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if key.char == 'y':
                        self.monitor.mark_detection_feedback(self.last_detection_time, True)
                        logger.info("✅ Marked last detection as correct")
                    elif key.char == 'n':
                        self.monitor.mark_detection_feedback(self.last_detection_time, False)
                        logger.info("❌ Marked last detection as incorrect")
                except AttributeError:
                    pass  # Special key pressed

            # Start keyboard listener
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()

        except Exception as e:
            logger.error(f"Error handling keyboard input: {e}")

    def stop(self):
        """Stop wake word detection"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Save final metrics
        if hasattr(self, 'monitor'):
            self.monitor.save_metrics()
            
        # Clean up resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Clear buffers
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                break
                
        self.accumulated_audio.clear()
        self.processing_batch.clear()
        
        logger.info("Wake word detector stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if self.is_running:
            try:
                # Debug audio levels
                audio_np = np.frombuffer(in_data, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float**2))
                db = 20 * np.log10(max(rms, 1e-10))
                
                # Always add to buffer and let the processor handle filtering
                self.audio_buffer.put(in_data)
                
                if db > -50:  # Only log when there's significant audio
                    logger.debug(f"Incoming audio level: {db:.1f} dB")
                    
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                
        return (in_data, pyaudio.paContinue)

    def _check_audio_energy(self, audio_data: np.ndarray) -> bool:
        """Check if audio energy is within acceptable range"""
        # Calculate RMS energy with proper handling of zero/negative values
        squared = np.abs(audio_data)**2
        mean_square = np.mean(squared) if len(squared) > 0 else 0
        
        if mean_square <= 1e-10:  # Avoid log of zero or negative
            energy_db = -100
        else:
            energy = np.sqrt(mean_square)
            energy_db = 20 * np.log10(max(energy, 1e-10))  # Ensure positive value for log
            
        # Only log energy levels when they're in range
        min_db = self.energy_threshold["min"]
        max_db = self.energy_threshold["max"]
        is_valid = min_db <= energy_db <= max_db
        
        return is_valid

    def _extract_features(self, audio_data: bytes) -> np.ndarray:
        """Extract MFCC features from audio data"""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            features = mfcc(
                audio_float,
                samplerate=self.target_sample_rate,
                numcep=13,
                nfilt=40,
                winlen=0.025,
                winstep=0.01,
                nfft=2048,
                preemph=0.97,
                appendEnergy=True
            )
            
            # Skip if features are invalid
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.debug("Invalid features detected (NaN or Inf)")
                return None
            
            # Normalize features
            features = (features - np.mean(features)) / (np.std(features) + 1e-10)
            
            # Ensure correct shape (200 frames x 13 features)
            target_frames = 200
            current_frames = features.shape[0]
            
            if current_frames > target_frames:
                # Take center portion
                start = (current_frames - target_frames) // 2
                features = features[start:start + target_frames]
            elif current_frames < target_frames:
                # Pad with zeros
                pad_frames = target_frames - current_frames
                pad_left = pad_frames // 2
                pad_right = pad_frames - pad_left
                features = np.pad(features, ((pad_left, pad_right), (0, 0)), mode='constant')
            
            # Reshape for model input (batch_size=1, frames=200, features=13, channels=1)
            features = features.reshape(1, 200, 13, 1)
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            return None

    def _log_metrics_periodically(self):
        """Log metrics periodically in a separate thread"""
        while self.is_running:
            try:
                time.sleep(60)  # Log every minute
                if self.is_running:
                    self.monitor.log_current_metrics()
                    self.monitor.save_metrics()  # Save metrics to file
            except Exception as e:
                logger.error(f"Error logging metrics: {e}")
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