import os
import wave
import time
import logging
import threading
from pathlib import Path

import numpy as np
import pyaudio
import tensorflow as tf
from python_speech_features import mfcc
from pynput import keyboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordTester:
    def __init__(self, model_path: str = "models/wake_word_model.keras"):
        """Initialize the wake word tester
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = Path(model_path)
        
        # Audio settings (must match training settings)
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.record_seconds = 2
        
        # MFCC settings (must match training settings)
        self.n_mfcc = 13
        self.n_mel = 40
        self.window_size = 0.025
        self.hop_size = 0.01
        
        # Recording state
        self.is_recording = False
        self.current_audio_data = []
        
        # Load model
        self._load_model()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def _load_model(self):
        """Load the trained wake word model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def start_testing(self):
        """Start testing the wake word model"""
        logger.info("Starting wake word testing session...")
        logger.info("Press 't' to test a sample (say 'Jarvis' or any other word)")
        logger.info("Press 'q' to quit")
        
        # Setup keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            
            # Keep the main thread alive
            while self.keyboard_listener.is_alive():
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error during testing: {e}")
        finally:
            self.stop_testing()
            
    def stop_testing(self):
        """Stop testing and cleanup"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logger.info("Testing session ended")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if self.is_recording:
            self.current_audio_data.append(in_data)
            
        return (in_data, pyaudio.paContinue)
    
    def _on_key_press(self, key):
        """Handle keyboard events"""
        try:
            if key.char == 't':
                self._test_sample()
            elif key.char == 'q':
                return False  # Stop listener
        except AttributeError:
            pass
            
    def _test_sample(self):
        """Record and test a single audio sample"""
        if self.is_recording:
            logger.warning("Already recording!")
            return
            
        logger.info("Recording in 3 seconds...")
        time.sleep(3)  # Give time to prepare
        
        self.is_recording = True
        self.current_audio_data = []
        
        logger.info("Recording...")
        time.sleep(self.record_seconds)
        
        self.is_recording = False
        self._process_sample()
        
    def _process_sample(self):
        """Process and evaluate the recorded sample"""
        if not self.current_audio_data:
            logger.warning("No audio data to process!")
            return
            
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(b''.join(self.current_audio_data), dtype=np.int16)
            
            # Extract MFCC features
            features = mfcc(
                audio_data,
                samplerate=self.sample_rate,
                numcep=self.n_mfcc,
                nfilt=self.n_mel,
                winlen=self.window_size,
                winstep=self.hop_size
            )
            
            # Normalize features
            features = (features - np.mean(features)) / np.std(features)
            
            # Pad or truncate to match expected length
            target_length = int(2 * self.sample_rate / (self.hop_size * self.sample_rate))
            if features.shape[0] > target_length:
                features = features[:target_length, :]
            elif features.shape[0] < target_length:
                pad_width = ((0, target_length - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            
            # Reshape for model input
            features = features.reshape(1, features.shape[0], features.shape[1], 1)
            
            # Get prediction
            prediction = float(self.model.predict(features, verbose=0)[0][0])
            
            # Display result
            if prediction > 0.5:
                logger.info(f"✅ Wake word detected! (confidence: {prediction:.2%})")
            else:
                logger.info(f"❌ Not a wake word (confidence: {1-prediction:.2%})")
                
        except Exception as e:
            logger.error(f"Error processing sample: {e}")

if __name__ == "__main__":
    tester = WakeWordTester()
    tester.start_testing() 