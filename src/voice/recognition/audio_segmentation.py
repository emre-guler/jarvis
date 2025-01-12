"""
Audio segmentation module for wake word detection system.
Handles automatic word boundary detection and clean segmentation of audio recordings.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional

class AudioSegmenter:
    """Handles automatic segmentation of audio for wake word detection"""
    
    def __init__(self, 
                 min_word_length: float = 0.3,  # 300ms
                 max_word_length: float = 1.2,  # 1.2s
                 padding: float = 0.1,  # 100ms
                 silence_threshold: float = -45.0,  # -45dB
                 sample_rate: int = 16000):
        """
        Initialize the audio segmenter.
        
        Args:
            min_word_length: Minimum word length in seconds
            max_word_length: Maximum word length in seconds
            padding: Pre/post padding in seconds
            silence_threshold: Silence threshold in dB
            sample_rate: Audio sample rate in Hz
        """
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.padding = padding
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        
    def detect_word_boundaries(self, audio: np.ndarray) -> Tuple[int, int]:
        """
        Detect word boundaries in audio using energy-based segmentation.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (start_sample, end_sample)
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)
            
        # Calculate energy in dB
        energy_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find regions above silence threshold
        active = energy_db > self.silence_threshold
        
        # Find start and end points
        start_idx = np.argmax(active)
        end_idx = len(audio) - np.argmax(active[::-1])
        
        # Apply minimum word length
        min_samples = int(self.min_word_length * self.sample_rate)
        if end_idx - start_idx < min_samples:
            center = (start_idx + end_idx) // 2
            start_idx = max(0, center - min_samples // 2)
            end_idx = min(len(audio), center + min_samples // 2)
            
        # Apply maximum word length
        max_samples = int(self.max_word_length * self.sample_rate)
        if end_idx - start_idx > max_samples:
            center = (start_idx + end_idx) // 2
            start_idx = center - max_samples // 2
            end_idx = center + max_samples // 2
            
        # Add padding
        pad_samples = int(self.padding * self.sample_rate)
        start_idx = max(0, start_idx - pad_samples)
        end_idx = min(len(audio), end_idx + pad_samples)
        
        return start_idx, end_idx
        
    def segment_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment audio to extract the word region.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Segmented audio array or None if no valid segment found
        """
        start_idx, end_idx = self.detect_word_boundaries(audio)
        
        # Get segment length
        segment_length = end_idx - start_idx
        min_samples = int(self.min_word_length * self.sample_rate)
        max_samples = int(self.max_word_length * self.sample_rate)
        
        # Handle short segments by padding with zeros
        if segment_length < min_samples:
            pad_amount = min_samples - segment_length
            left_pad = pad_amount // 2
            right_pad = pad_amount - left_pad
            segment = np.pad(audio[start_idx:end_idx], (left_pad, right_pad), mode='constant')
            return segment
            
        # Handle long segments by centering and truncating
        if segment_length > max_samples:
            center = (start_idx + end_idx) // 2
            start_idx = center - max_samples // 2
            end_idx = start_idx + max_samples
            
        return audio[start_idx:end_idx]
        
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Process an audio file and optionally save the segmented result.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional path to save segmented audio
            
        Returns:
            Path to segmented audio file if output_path provided, else None
        """
        # Load audio file
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Segment audio
        segmented = self.segment_audio(audio)
        if segmented is None:
            return None
            
        # Save if output path provided
        if output_path:
            sf.write(output_path, segmented, self.sample_rate)
            return output_path
            
        return None 