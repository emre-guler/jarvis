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
        self.process_every_n_chunks = 2  # Reduced for better responsiveness
        self.chunk_counter = 0
        self.batch_size = 4
        self.processing_batch = []
        
        # Stricter energy thresholds
        self.energy_threshold = {
            "max": 0.0,    # 0 dB (maximum)
            "min": -55.0   # Adjusted for better sensitivity
        }
        
        # Power saving settings
        self.power_mode = "performance"  # Changed to performance mode
        self.processing_interval = 0.02  # Reduced for faster response
        
        # Initialize performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Start periodic metrics logging with reduced frequency
        self.metrics_thread = threading.Thread(target=self._log_metrics_periodically)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()

    def _load_model(self):
        """Load the wake word detection model"""
        try:
            model_path = Path("models/wake_word_model.keras")
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
                logger.info(f"Wake word model loaded from {model_path}")
            else:
                logger.warning("No trained model found at models/wake_word_model.keras")
                logger.warning("Please train the model first using src/voice/training/train_model.py")
                raise FileNotFoundError("Wake word model not found")
            
            # Warm up the model
            dummy_input = np.zeros((1, 200, 13, 1))  # Shape matches our training data
            self.model.predict(dummy_input, verbose=0)
            
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
        """Start wake word detection"""
        if self.is_running:
            logger.warning("Wake word detector is already running")
            return

        self.callback = callback
        self.is_running = True

        try:
            # Use the default input device from settings or system default
            input_device_index = AUDIO_SETTINGS.get("input_device", None)
            
            if input_device_index is None:
                # Find the first available input device
                for i in range(self.audio.get_device_count()):
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        input_device_index = i
                        logger.info(f"Using input device: {device_info['name']} (device index: {input_device_index})")
                        break

            if input_device_index is None:
                logger.error("No input devices found")
                return

            # Get the device info to use its native sample rate
            device_info = self.audio.get_device_info_by_index(input_device_index)
            self.device_sample_rate = int(device_info['defaultSampleRate'])
            
            # Start audio stream with the device's native sample rate
            self.stream = self.audio.open(
                format=self.format,
                channels=AUDIO_SETTINGS["channels"],
                rate=self.device_sample_rate,
                input=True,
                frames_per_buffer=1024,  # Fixed buffer size
                input_device_index=input_device_index
            )

            # Verify stream is active
            if not self.stream.is_active():
                logger.error("Failed to start audio stream")
                return
                
            logger.info("Audio stream is active and receiving input")
            logger.info(f"Device sample rate: {self.device_sample_rate} Hz")
            logger.info(f"Target sample rate: {self.target_sample_rate} Hz")

            print("\n" + "="*50)
            print("🎤 Wake Word Detection Active")
            print("Say 'Jarvis' to trigger")
            print("Press Ctrl+C to stop")
            print("="*50 + "\n")

            # Start processing in the main thread
            self._process_audio_stream()

        except Exception as e:
            logger.error(f"Failed to start wake word detector: {e}")
            self.stop()
            raise

    def _process_audio_stream(self):
        """Process audio stream in real-time"""
        logger.info("Starting audio processing")
        
        chunk_size = 1024
        audio_buffer = []
        required_samples = int(self.target_sample_rate * 0.5)  # 500ms of audio
        
        while self.is_running:
            try:
                # Read audio data
                audio_data = self.stream.read(chunk_size, exception_on_overflow=False)
                audio_buffer.append(audio_data)
                
                # Process when we have enough data
                total_samples = len(audio_buffer) * chunk_size
                if total_samples >= required_samples:
                    # Concatenate audio chunks
                    audio_bytes = b''.join(audio_buffer)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Resample to target sample rate
                    if self.device_sample_rate != self.target_sample_rate:
                        ratio = self.target_sample_rate / self.device_sample_rate
                        new_length = int(len(audio_np) * ratio)
                        x_old = np.linspace(0, 1, len(audio_np))
                        x_new = np.linspace(0, 1, new_length)
                        audio_np = np.interp(x_new, x_old, audio_np)
                    
                    # Calculate energy
                    audio_float = audio_np.astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(audio_float**2))
                    energy_db = 20 * np.log10(max(rms, 1e-10))
                    
                    logger.debug(f"Processing audio - Energy: {energy_db:.1f} dB")
                    
                    # Only process if energy is above threshold
                    if energy_db > -40:  # Increased threshold for better detection
                        # Extract features
                        features = self._extract_features(audio_np.tobytes())
                        
                        if features is not None:
                            # Get prediction
                            confidence = float(self.model.predict(features, verbose=0)[0][0])
                            logger.debug(f"Wake word confidence: {confidence:.3f}")
                            
                            # Check for wake word
                            if confidence > 0.5:  # Lower threshold for testing
                                print(f"\n✨ Wake word detected! (confidence: {confidence:.2f}, energy: {energy_db:.1f}dB)")
                                if self.callback:
                                    self.callback(confidence)
                                time.sleep(0.5)  # Cooldown after detection
                    
                    # Keep only the most recent chunk
                    audio_buffer = audio_buffer[-2:]
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                logger.debug(f"Error details: {str(e)}", exc_info=True)
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
            
        # Log final accuracy metrics
        metrics = self.monitor.get_current_metrics()
        if 'accuracy' in metrics:
            logger.info("\n=== Final Accuracy Metrics ===")
            logger.info(f"Total Detections: {metrics['detections']['total']}")
            logger.info(f"True Positives: {metrics['detections']['true_positives']}")
            logger.info(f"False Positives: {metrics['detections']['false_positives']}")
            logger.info(f"Accuracy: {metrics['accuracy']['value']:.1%}")
        
        # Save final metrics
        self.monitor.save_metrics()
        
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
        """Log metrics periodically"""
        while self.is_running:
            try:
                # Record system metrics less frequently
                self.monitor.record_system_metrics()
                time.sleep(60)  # Increased from 30s to 60s
                
            except Exception as e:
                logger.error(f"Error logging metrics: {e}")
                time.sleep(60)

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