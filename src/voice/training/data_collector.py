import os
import wave
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pyaudio
from pynput import keyboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDataCollector:
    def __init__(self, output_dir: str = "data/wake_word_samples"):
        """Initialize the data collector
        
        Args:
            output_dir: Directory to save audio samples
        """
        self.output_dir = Path(output_dir)
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        
        # Create directories
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.record_seconds = 2  # Each sample is 2 seconds
        
        # Recording state
        self.is_recording = False
        self.current_audio_data = []
        self.recording_positive = False
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        """Start recording audio samples"""
        logger.info("Starting recording session...")
        logger.info("Press 'p' to record a positive sample (saying 'Jarvis')")
        logger.info("Press 'n' to record a negative sample (other words/noise)")
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
            logger.error(f"Error during recording: {e}")
        finally:
            self.stop_recording()
            
    def stop_recording(self):
        """Stop recording and cleanup"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logger.info("Recording session ended")
        
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
            if key.char == 'p':
                self._record_sample(positive=True)
            elif key.char == 'n':
                self._record_sample(positive=False)
            elif key.char == 'q':
                return False  # Stop listener
        except AttributeError:
            pass
        
    def _record_sample(self, positive: bool):
        """Record a single audio sample"""
        if self.is_recording:
            logger.warning("Already recording!")
            return
            
        sample_type = "positive" if positive else "negative"
        logger.info(f"Recording {sample_type} sample in 3 seconds...")
        time.sleep(3)  # Give time to prepare
        
        self.is_recording = True
        self.current_audio_data = []
        self.recording_positive = positive
        
        logger.info("Recording...")
        time.sleep(self.record_seconds)
        
        self.is_recording = False
        self._save_sample()
        
        logger.info(f"{sample_type.capitalize()} sample recorded!")
        
    def _save_sample(self):
        """Save the recorded audio sample"""
        if not self.current_audio_data:
            logger.warning("No audio data to save!")
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.positive_dir if self.recording_positive else self.negative_dir
        filename = output_dir / f"sample_{timestamp}.wav"
        
        # Save as WAV file
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.current_audio_data))
            
        logger.info(f"Saved sample to {filename}")

if __name__ == "__main__":
    collector = WakeWordDataCollector()
    collector.start_recording() 