"""
Process a long audio file into wake word training samples
"""
import logging
import wave
import numpy as np
from pathlib import Path
import librosa
from pydantic import BaseModel
from typing import List, Tuple
import json

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioSegment(BaseModel):
    start_time: float
    end_time: float
    label: str  # "positive" or "negative"

class AudioProcessor:
    def __init__(self):
        """Initialize the audio processor"""
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.sample_duration = WAKE_WORD_CONFIG["sample_duration"]
        
        # Create directories
        self.positive_dir = WAKE_WORD_CONFIG["positive_samples_dir"]
        self.negative_dir = WAKE_WORD_CONFIG["negative_samples_dir"]
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)

    def process_audio_file(self, audio_file: str, segments_file: str):
        """Process a long audio file using timestamp annotations"""
        logger.info(f"Processing audio file: {audio_file}")
        
        # Load audio file
        audio_data, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        
        # Load segments
        with open(segments_file, 'r') as f:
            segments = [AudioSegment(**s) for s in json.load(f)]
        
        # Process each segment
        for i, segment in enumerate(segments):
            start_sample = int(segment.start_time * self.sample_rate)
            end_sample = int(segment.end_time * self.sample_rate)
            
            # Extract audio segment
            audio_segment = audio_data[start_sample:end_sample]
            
            # Save segment
            self._save_segment(audio_segment, segment.label, i)
            
        logger.info("Audio processing complete!")
        
        # Show counts
        positive_count = len(list(self.positive_dir.glob("*.wav")))
        negative_count = len(list(self.negative_dir.glob("*.wav")))
        print(f"\n📊 Processed samples:")
        print(f"   Positive: {positive_count}")
        print(f"   Negative: {negative_count}")

    def _save_segment(self, audio_data: np.ndarray, label: str, index: int):
        """Save an audio segment as a WAV file"""
        # Choose directory
        save_dir = self.positive_dir if label == "positive" else self.negative_dir
        filename = f"sample_{label}_{index:04d}.wav"
        filepath = save_dir / filename
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        logger.info(f"Saved {label} sample: {filename}")

def main():
    """Main function"""
    print("\n" + "="*50)
    print("🎤 Long Audio File Processor")
    print("This script processes a long audio file into training samples")
    print("="*50 + "\n")
    
    # Get input files
    audio_file = input("Enter the path to your audio file: ")
    segments_file = input("Enter the path to your segments JSON file: ")
    
    # Process the audio
    processor = AudioProcessor()
    processor.process_audio_file(audio_file, segments_file)

if __name__ == "__main__":
    main() 