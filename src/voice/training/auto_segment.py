"""
Automatically segment long audio recordings into training samples
"""
import logging
import wave
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from scipy.io import wavfile
from pydantic import BaseModel
import json

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoSegmenter:
    def __init__(self):
        """Initialize the auto segmenter"""
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.sample_duration = WAKE_WORD_CONFIG["sample_duration"]
        
        # Create directories
        self.positive_dir = WAKE_WORD_CONFIG["positive_samples_dir"]
        self.negative_dir = WAKE_WORD_CONFIG["negative_samples_dir"]
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)
        
        # Segmentation parameters - adjusted for better detection
        self.min_silence_len = 0.3    # reduced minimum silence length
        self.silence_thresh = -50     # more lenient silence threshold
        self.chunk_size = 2.0        # chunk size in seconds for positive samples
        
    def detect_speech_segments(self, audio_data: np.ndarray) -> list:
        """Detect speech segments in audio using energy-based VAD"""
        # Calculate energy in dB
        energy = librosa.amplitude_to_db(np.abs(audio_data), ref=np.max)
        
        # Apply smoothing to energy
        window_size = int(0.02 * self.sample_rate)  # 20ms window
        energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        
        # Find regions above silence threshold
        is_speech = energy_smooth > self.silence_thresh
        
        # Find speech segment boundaries
        boundaries = np.where(np.diff(is_speech.astype(int)))[0]
        
        if len(boundaries) < 2:
            logger.warning("No clear speech segments detected. Trying with entire audio...")
            return [(0, len(audio_data)/self.sample_rate)]
            
        segments = []
        start_idx = 0 if is_speech[0] else boundaries[0]
        
        for i in range(0, len(boundaries)-1, 2):
            if i + 1 >= len(boundaries):
                break
                
            end_idx = boundaries[i+1]
            
            # Convert to seconds
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            
            # Only keep segments longer than min_silence_len
            if end_time - start_time >= self.min_silence_len:
                segments.append((start_time, end_time))
                logger.info(f"Detected segment: {start_time:.2f}s - {end_time:.2f}s")
            
            if i + 2 < len(boundaries):
                start_idx = boundaries[i+2]
        
        if not segments:
            logger.warning("No segments found after filtering. Using entire audio...")
            return [(0, len(audio_data)/self.sample_rate)]
            
        return segments
        
    def process_audio_file(self, audio_file: str, is_positive: bool = True):
        """Process a long audio file and segment it into samples"""
        logger.info(f"Processing audio file: {audio_file}")
        
        # Load audio file
        audio_data, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        logger.info(f"Audio duration: {len(audio_data)/self.sample_rate:.2f} seconds")
        
        # Detect speech segments
        segments = self.detect_speech_segments(audio_data)
        
        logger.info(f"Found {len(segments)} speech segments")
        
        # Process each segment
        for i, (start_time, end_time) in enumerate(segments):
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Extract audio segment
            audio_segment = audio_data[start_sample:end_sample]
            
            # Pad or trim to chunk_size
            target_length = int(self.chunk_size * self.sample_rate)
            if len(audio_segment) < target_length:
                # Pad with silence
                padding = target_length - len(audio_segment)
                audio_segment = np.pad(audio_segment, (0, padding), mode='constant')
            else:
                # Center and trim
                start = (len(audio_segment) - target_length) // 2
                audio_segment = audio_segment[start:start+target_length]
            
            # Save segment
            self._save_segment(audio_segment, "positive" if is_positive else "negative", i)
        
        # Show counts
        positive_count = len(list(self.positive_dir.glob("*.wav")))
        negative_count = len(list(self.negative_dir.glob("*.wav")))
        print(f"\n📊 Current samples:")
        print(f"   Positive: {positive_count}")
        print(f"   Negative: {negative_count}")
    
    def _save_segment(self, audio_data: np.ndarray, label: str, index: int):
        """Save an audio segment as a WAV file"""
        # Choose directory
        save_dir = self.positive_dir if label == "positive" else self.negative_dir
        filename = f"sample_{label}_{index:04d}.wav"
        filepath = save_dir / filename
        
        # Save as WAV using soundfile (better handling of float32 audio)
        sf.write(str(filepath), audio_data, self.sample_rate)
        
        logger.info(f"Saved {label} sample: {filename}")

def main():
    """Main function"""
    print("\n" + "="*50)
    print("🎤 Auto Audio Segmenter")
    print("\nThis script will automatically segment your audio into training samples.")
    print("Record two types of audio files:")
    print("1. Positive samples: You saying 'Jarvis' multiple times with pauses")
    print("2. Negative samples: Other words and background noise")
    print("="*50 + "\n")
    
    # Process positive samples
    print("\n1. Processing POSITIVE samples (recordings of 'Jarvis')")
    audio_file = input("Enter the path to your positive audio file (with multiple 'Jarvis' utterances): ")
    
    processor = AutoSegmenter()
    processor.process_audio_file(audio_file, is_positive=True)
    
    # Process negative samples
    print("\n2. Processing NEGATIVE samples (other words/noise)")
    audio_file = input("Enter the path to your negative audio file: ")
    processor.process_audio_file(audio_file, is_positive=False)
    
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main() 