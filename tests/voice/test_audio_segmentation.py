"""
Test suite for Audio Segmentation System
"""
import os
import wave
import shutil
import logging
import numpy as np
import librosa
from pathlib import Path
from unittest import TestCase, main

from src.voice.recognition.audio_segmentation import AudioSegmenter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAudioSegmentation(TestCase):
    """Test cases for audio segmentation system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test directory
        self.test_dir = Path(__file__).parent / "data" / "test_segmentation"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize segmenter with default parameters
        self.segmenter = AudioSegmenter()
        
        # Create test audio files
        self.create_test_audio()
        
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def create_test_audio(self):
        """Create test audio files with known characteristics"""
        # Create short word (200ms)
        self.short_file = self.test_dir / "short_word.wav"
        self._create_audio_file(self.short_file, duration=0.2)
        
        # Create normal word (500ms)
        self.normal_file = self.test_dir / "normal_word.wav"
        self._create_audio_file(self.normal_file, duration=0.5)
        
        # Create long word (1.5s)
        self.long_file = self.test_dir / "long_word.wav"
        self._create_audio_file(self.long_file, duration=1.5)
        
    def _create_audio_file(self, file_path: Path, duration: float, frequency: int = 440):
        """Helper to create test audio files"""
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Add silence padding
        silence_duration = 0.2
        silence = np.zeros(int(sample_rate * silence_duration))
        padded_audio = np.concatenate([silence, audio_data, silence])
        
        # Normalize and convert to int16
        padded_audio = (padded_audio * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(str(file_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(padded_audio.tobytes())
            
    def test_initialization(self):
        """Test segmenter initialization with default parameters"""
        self.assertEqual(self.segmenter.min_word_length, 0.3)
        self.assertEqual(self.segmenter.max_word_length, 1.2)
        self.assertEqual(self.segmenter.padding, 0.1)
        self.assertEqual(self.segmenter.silence_threshold, -45.0)
        self.assertEqual(self.segmenter.sample_rate, 16000)
        
    def test_word_boundary_detection(self):
        """Test word boundary detection"""
        # Test with normal word
        audio, _ = librosa.load(self.normal_file, sr=self.segmenter.sample_rate)
        start_idx, end_idx = self.segmenter.detect_word_boundaries(audio)
        
        # Check if boundaries are within expected range
        expected_start = int(0.2 * self.segmenter.sample_rate)  # After initial silence
        expected_end = int(0.7 * self.segmenter.sample_rate)    # Before final silence
        
        # Allow for padding
        pad_samples = int(self.segmenter.padding * self.segmenter.sample_rate)
        self.assertLessEqual(abs(start_idx - expected_start), pad_samples)
        self.assertLessEqual(abs(end_idx - expected_end), pad_samples)
        
    def test_segmentation_length_constraints(self):
        """Test minimum and maximum length constraints"""
        # Test short word (should be padded to minimum length)
        audio, _ = librosa.load(self.short_file, sr=self.segmenter.sample_rate)
        segmented = self.segmenter.segment_audio(audio)
        min_samples = int(self.segmenter.min_word_length * self.segmenter.sample_rate)
        self.assertGreaterEqual(len(segmented), min_samples)
        
        # Test long word (should be truncated to maximum length)
        audio, _ = librosa.load(self.long_file, sr=self.segmenter.sample_rate)
        segmented = self.segmenter.segment_audio(audio)
        max_samples = int(self.segmenter.max_word_length * self.segmenter.sample_rate)
        self.assertLessEqual(len(segmented), max_samples)
        
    def test_file_processing(self):
        """Test end-to-end file processing"""
        output_file = self.test_dir / "output.wav"
        
        # Process normal word file
        result = self.segmenter.process_file(str(self.normal_file), str(output_file))
        
        # Check if output file was created
        self.assertIsNotNone(result)
        self.assertTrue(output_file.exists())
        
        # Verify output audio properties
        with wave.open(str(output_file), 'rb') as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), self.segmenter.sample_rate)
            
            # Check duration is within bounds
            duration = wf.getnframes() / wf.getframerate()
            self.assertGreaterEqual(duration, self.segmenter.min_word_length)
            self.assertLessEqual(duration, self.segmenter.max_word_length)
            
    def test_silence_detection(self):
        """Test silence detection and thresholding"""
        # Create audio with varying levels
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create segments with different amplitudes
        silence = np.zeros(int(0.2 * sample_rate))
        quiet = np.sin(2 * np.pi * 440 * t[:int(0.3 * sample_rate)]) * 0.1
        loud = np.sin(2 * np.pi * 440 * t[:int(0.3 * sample_rate)]) * 0.8
        
        # Combine segments
        audio = np.concatenate([silence, quiet, loud, silence])
        
        # Test boundary detection
        start_idx, end_idx = self.segmenter.detect_word_boundaries(audio)
        
        # Verify that quiet section is included but silence is excluded
        self.assertLess(start_idx, len(silence) + len(quiet))
        self.assertGreater(end_idx, len(silence) + len(quiet))
        
if __name__ == '__main__':
    main() 