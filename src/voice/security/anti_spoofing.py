"""Anti-spoofing detection for voice security"""
import logging
import numpy as np
import librosa
from typing import Tuple

logger = logging.getLogger(__name__)

class AntiSpoofing:
    def __init__(self):
        """Initialize anti-spoofing detection system"""
        # Thresholds for different checks
        self.energy_threshold = 0.1
        self.frequency_threshold = 0.7
        self.temporal_threshold = 0.6
        
    def check_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """Check audio for signs of spoofing
        
        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate
            
        Returns:
            Tuple[bool, float]: (is_genuine, confidence)
        """
        try:
            # Run multiple checks
            energy_score = self._check_energy_distribution(audio_data)
            freq_score = self._check_frequency_distribution(audio_data, sample_rate)
            temporal_score = self._check_temporal_patterns(audio_data, sample_rate)
            
            # Combine scores with weights
            weights = [0.3, 0.4, 0.3]  # Energy, Frequency, Temporal
            total_score = (
                weights[0] * energy_score +
                weights[1] * freq_score +
                weights[2] * temporal_score
            )
            
            # Determine if genuine
            is_genuine = bool(
                energy_score >= self.energy_threshold and
                freq_score >= self.frequency_threshold and
                temporal_score >= self.temporal_threshold
            )
            
            return is_genuine, total_score
            
        except Exception as e:
            logger.error(f"Error checking audio: {e}")
            return False, 0.0
            
    def _check_energy_distribution(self, audio_data: np.ndarray) -> float:
        """Check energy distribution for replay signs"""
        try:
            # Calculate energy in different frequency bands
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Split into frequency bands
            n_bands = 8
            band_size = magnitude.shape[0] // n_bands
            band_energies = [
                np.mean(magnitude[i:i+band_size]) 
                for i in range(0, magnitude.shape[0], band_size)
            ]
            
            # Check energy distribution
            energy_std = np.std(band_energies)
            energy_ratio = max(band_energies) / (min(band_energies) + 1e-10)
            
            # Natural speech should have varied energy distribution
            score = min(1.0, energy_std / 2.0) * min(1.0, 5.0 / energy_ratio)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in energy check: {e}")
            return 0.0
    
    # Alias for backward compatibility
    _check_energy = _check_energy_distribution
            
    def _check_frequency_distribution(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Check frequency distribution for synthetic signs"""
        try:
            # Calculate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=128
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Calculate statistics
            freq_std = np.std(mel_spec_db, axis=1)
            freq_range = np.ptp(mel_spec_db, axis=1)
            
            # Natural speech has varied frequency content
            score = min(1.0, np.mean(freq_std) / 30.0) * min(1.0, np.mean(freq_range) / 60.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in frequency check: {e}")
            return 0.0
            
    def _check_temporal_patterns(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Check temporal patterns for replay signs"""
        try:
            # Calculate temporal features
            zero_crossings = librosa.zero_crossings(audio_data)
            zero_crossing_rate = np.mean(zero_crossings)
            
            # Natural speech typically has varying zero-crossing rates
            frames = librosa.util.frame(audio_data, frame_length=2048, hop_length=512)
            frame_zero_crossings = np.array([
                np.mean(librosa.zero_crossings(frame)) 
                for frame in frames.T
            ])
            
            # Calculate variation in zero-crossing rate
            zcr_std = np.std(frame_zero_crossings)
            score = min(1.0, zcr_std * 10)  # Normalize to [0,1]
            
            return score
            
        except Exception as e:
            logger.error(f"Error in temporal check: {e}")
            return 0.0 