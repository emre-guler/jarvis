"""
Data Collection Script for Wake Word Training
"""
import logging
import sys
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import pyaudio
from pynput import keyboard

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WakeWordDataCollector:
    def __init__(self):
        """Initialize the data collector"""
        # Audio settings
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.channels = AUDIO_SETTINGS["channels"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.format = pyaudio.paInt16
        self.sample_duration = WAKE_WORD_CONFIG["sample_duration"]
        
        # Create directories
        self.positive_dir = WAKE_WORD_CONFIG["positive_samples_dir"]
        self.negative_dir = WAKE_WORD_CONFIG["negative_samples_dir"]
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Recording state
        self.is_recording = False
        self.current_audio = []
        self.recording_positive = False
        
    def start(self):
        """Start the data collection process"""
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
            
            print("\n" + "="*50)
            print("🎤 Wake Word Data Collection")
            print("Press 'p' to record a positive sample (saying 'Jarvis')")
            print("Press 'n' to record a negative sample (other words/noise)")
            print("Press 'q' to quit")
            print("="*50 + "\n")
            
            # Start keyboard listener
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                listener.join()
                
        except Exception as e:
            logger.error(f"Error starting data collection: {e}")
            self.stop()
            sys.exit(1)
            
    def stop(self):
        """Stop data collection"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data"""
        if self.is_recording:
            self.current_audio.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _on_key_press(self, key):
        """Handle keyboard input"""
        try:
            if key.char == 'p':
                if not self.is_recording:
                    print("\n🎤 Recording POSITIVE sample (say 'Jarvis')...")
                    self._start_recording(positive=True)
            elif key.char == 'n':
                if not self.is_recording:
                    print("\n🎤 Recording NEGATIVE sample...")
                    self._start_recording(positive=False)
            elif key.char == 'q':
                print("\n👋 Stopping data collection...")
                self.stop()
                return False
        except AttributeError:
            pass
        return True
    
    def _start_recording(self, positive: bool):
        """Start recording a sample"""
        self.is_recording = True
        self.recording_positive = positive
        self.current_audio = []
        
        # Calculate frames needed for sample duration
        frames_needed = int(self.sample_rate * self.sample_duration)
        
        # Record for the specified duration
        while len(b''.join(self.current_audio)) < frames_needed * self.channels * 2:  # 2 bytes per sample
            pass
        
        # Stop recording and save
        self.is_recording = False
        self._save_sample()
        
    def _save_sample(self):
        """Save the recorded audio sample"""
        if not self.current_audio:
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_type = "positive" if self.recording_positive else "negative"
        filename = f"sample_{sample_type}_{timestamp}.wav"
        
        # Choose directory
        save_dir = self.positive_dir if self.recording_positive else self.negative_dir
        filepath = save_dir / filename
        
        # Save as WAV file
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.current_audio))
        
        # Log success
        logger.info(f"✅ Saved {sample_type} sample: {filename}")
        
        # Show counts
        positive_count = len(list(self.positive_dir.glob("*.wav")))
        negative_count = len(list(self.negative_dir.glob("*.wav")))
        print(f"\n📊 Current samples:")
        print(f"   Positive: {positive_count} (target: 100+)")
        print(f"   Negative: {negative_count} (target: 200+)")
        print("\nReady for next recording...")

if __name__ == "__main__":
    collector = WakeWordDataCollector()
    collector.start() 