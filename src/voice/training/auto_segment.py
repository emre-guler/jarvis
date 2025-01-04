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

# Path Constants
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio" / "wake_word"
POSITIVE_SAMPLES_PATH = AUDIO_DIR / "jarvis_samples.wav"
NEGATIVE_SAMPLES_PATH = AUDIO_DIR / "negative_samples.wav"

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
        
        # Improved segmentation parameters for cleaner word boundaries
        self.min_silence_len = 0.2     # minimum silence between words (200ms)
        self.silence_thresh = -45      # silence threshold in dB
        self.chunk_size = 2.0          # chunk size in seconds
        self.pre_padding = 0.1         # padding before word (100ms)
        self.post_padding = 0.1        # padding after word (100ms)
        self.min_word_length = 0.3     # minimum word length (300ms)
        self.max_word_length = 1.2     # maximum word length (1.2s)
        
    def detect_speech_segments(self, audio_data: np.ndarray) -> list:
        """Detect speech segments in audio using enhanced VAD"""
        # Calculate energy in dB
        energy = librosa.amplitude_to_db(np.abs(audio_data), ref=np.max)
        
        # Multi-scale energy smoothing for better boundary detection
        scales = [
            int(0.01 * self.sample_rate),  # 10ms for fine details
            int(0.02 * self.sample_rate),  # 20ms for word-level
            int(0.05 * self.sample_rate)   # 50ms for overall trend
        ]
        
        energy_smooth = np.zeros_like(energy)
        for scale in scales:
            window = np.hanning(scale)
            window = window / window.sum()
            energy_smooth += np.convolve(energy, window, mode='same')
        energy_smooth /= len(scales)
        
        # Dynamic threshold using local statistics
        window_size = int(0.5 * self.sample_rate)  # 500ms window
        local_mean = np.convolve(energy_smooth, np.ones(window_size)/window_size, mode='same')
        local_std = np.sqrt(np.convolve((energy_smooth - local_mean)**2, np.ones(window_size)/window_size, mode='same'))
        dynamic_threshold = local_mean - local_std
        
        # Find speech regions
        is_speech = energy_smooth > np.maximum(self.silence_thresh, dynamic_threshold)
        
        # Clean up speech regions
        min_samples = int(self.min_word_length * self.sample_rate)
        max_samples = int(self.max_word_length * self.sample_rate)
        
        # Remove short segments
        labels = librosa.effects.split(audio_data, top_db=-self.silence_thresh)
        segments = []
        
        for start, end in labels:
            duration = (end - start) / self.sample_rate
            
            # Skip if segment is too short or too long
            if duration < self.min_word_length or duration > self.max_word_length:
                continue
            
            # Add padding
            start_pad = max(0, start - int(self.pre_padding * self.sample_rate))
            end_pad = min(len(audio_data), end + int(self.post_padding * self.sample_rate))
            
            segments.append((start_pad / self.sample_rate, end_pad / self.sample_rate))
            logger.info(f"Detected word segment: {start_pad/self.sample_rate:.2f}s - {end_pad/self.sample_rate:.2f}s")
        
        if not segments:
            logger.warning("No valid word segments found!")
            return []
            
        return segments
        
    def process_audio_file(self, audio_file: str, is_positive: bool = True):
        """Process a long audio file and segment it into samples"""
        logger.info(f"Processing audio file: {audio_file}")
        
        try:
            # Load and normalize audio
            logger.info("Loading audio file...")
            audio_data, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            audio_data = librosa.util.normalize(audio_data)
            
            logger.info(f"Audio duration: {len(audio_data)/self.sample_rate:.2f} seconds")
            
            # Detect word segments
            logger.info("Detecting word segments...")
            segments = self.detect_speech_segments(audio_data)
            
            if not segments:
                logger.error("No valid segments found in the audio file!")
                return
            
            logger.info(f"Found {len(segments)} valid word segments")
            
            # Process each segment
            for i, (start_time, end_time) in enumerate(segments):
                logger.info(f"Processing segment {i+1}/{len(segments)}")
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # Extract audio segment
                audio_segment = audio_data[start_sample:end_sample]
                
                # Normalize segment
                audio_segment = librosa.util.normalize(audio_segment)
                
                # Center the word in the chunk if needed
                target_length = int(self.chunk_size * self.sample_rate)
                if len(audio_segment) < target_length:
                    # Calculate padding for centering
                    pad_left = (target_length - len(audio_segment)) // 2
                    pad_right = target_length - len(audio_segment) - pad_left
                    audio_segment = np.pad(audio_segment, (pad_left, pad_right), mode='constant')
                else:
                    # Center and trim
                    center = len(audio_segment) // 2
                    start = center - (target_length // 2)
                    audio_segment = audio_segment[start:start+target_length]
                
                # Save segment
                self._save_segment(audio_segment, "positive" if is_positive else "negative", i)
            
            # Show counts
            positive_count = len(list(self.positive_dir.glob("*.wav")))
            negative_count = len(list(self.negative_dir.glob("*.wav")))
            print(f"\n📊 Current samples:")
            print(f"   Positive: {positive_count}")
            print(f"   Negative: {negative_count}")
            
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file}")
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise
    
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
    print("🎤 Enhanced Word Segmentation for Wake Word Training")
    print("\nThis script will automatically segment your audio into individual word samples.")
    print("\nGuidelines for recording:")
    print("1. Positive samples (saying 'Jarvis'):")
    print("   - Speak clearly with ~1 second pause between each 'Jarvis'")
    print("   - Vary your tone and speed slightly for better training")
    print("   - Record in your typical usage environment")
    print("\n2. Negative samples (other words/noise):")
    print("   - Include similar-sounding words")
    print("   - Add common background noises")
    print("   - Speak other commands or phrases you commonly use")
    print("="*50 + "\n")
    
    print(f"\nDefault file paths:")
    print(f"- Positive samples: {POSITIVE_SAMPLES_PATH}")
    print(f"- Negative samples: {NEGATIVE_SAMPLES_PATH}")
    
    try:
        # Process positive samples
        print("\n1. Processing POSITIVE samples (recordings of 'Jarvis')")
        audio_file = input(f"Enter the path to your positive audio file (press Enter to use default: {POSITIVE_SAMPLES_PATH}): ")
        audio_file = audio_file.strip() if audio_file.strip() else str(POSITIVE_SAMPLES_PATH)
        
        # Check if file exists
        if not Path(audio_file).exists():
            print(f"\n❌ Error: File not found: {audio_file}")
            return
        
        processor = AutoSegmenter()
        processor.process_audio_file(audio_file, is_positive=True)
        
        # Process negative samples
        print("\n2. Processing NEGATIVE samples (other words/noise)")
        audio_file = input(f"Enter the path to your negative audio file (press Enter to use default: {NEGATIVE_SAMPLES_PATH}): ")
        audio_file = audio_file.strip() if audio_file.strip() else str(NEGATIVE_SAMPLES_PATH)
        
        # Check if file exists
        if not Path(audio_file).exists():
            print(f"\n❌ Error: File not found: {audio_file}")
            return
            
        processor.process_audio_file(audio_file, is_positive=False)
        
        print("\n✅ Processing complete!")
        print("\nPlease verify the segmented samples in the output directories:")
        print(f"- Positive samples: {WAKE_WORD_CONFIG['positive_samples_dir']}")
        print(f"- Negative samples: {WAKE_WORD_CONFIG['negative_samples_dir']}")
        print("\nListen to a few samples to ensure clean word boundaries before training.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Error during processing: {e}", exc_info=True)

if __name__ == "__main__":
    main() 