"""Voice feature extractor for speaker recognition"""
import logging
import numpy as np
from typing import Dict, List, Optional, Union
import librosa
from python_speech_features import mfcc, delta

from config.settings import AUDIO_SETTINGS

logger = logging.getLogger(__name__)

class VoiceFeatureExtractor:
    def __init__(self):
        """Initialize the voice feature extractor"""
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        
        # Feature extraction settings
        self.n_mfcc = 20          # Number of MFCC coefficients
        self.n_mels = 40          # Number of Mel bands
        self.frame_length = 0.025  # Frame length in seconds
        self.frame_shift = 0.010   # Frame shift in seconds
        self.n_fft = 2048         # FFT window size
        
    def extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice features from audio data
        
        Args:
            audio_data: Audio signal data
            
        Returns:
            Optional[np.ndarray]: Extracted feature vector or None on error
        """
        try:
            # Ensure audio data is normalized
            audio_data = librosa.util.normalize(audio_data)
            
            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=int(self.frame_shift * self.sample_rate),
                win_length=int(self.frame_length * self.sample_rate)
            )
            
            # Calculate delta and delta-delta features
            delta_features = librosa.feature.delta(mfcc_features)
            delta2_features = librosa.feature.delta(mfcc_features, order=2)
            
            # Spectral features
            cent = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            # Prosodic features
            f0, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)
            
            # Compute statistics and concatenate
            features = np.concatenate([
                np.mean(mfcc_features, axis=1),
                np.std(mfcc_features, axis=1),
                np.mean(delta_features, axis=1),
                np.std(delta_features, axis=1),
                np.mean(delta2_features, axis=1),
                np.std(delta2_features, axis=1),
                [np.mean(cent)],
                [np.mean(bandwidth)],
                [np.mean(rolloff)],
                [np.nanmean(f0[voiced_flag]) if np.any(voiced_flag) else 0.0],
                [np.nanstd(f0[voiced_flag]) if np.any(voiced_flag) else 0.0],
                [np.mean(zcr)],
                [np.mean(rms)]
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
            
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two sets of voice features
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            if features1 is None or features2 is None:
                return 0.0
                
            # Ensure same length
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            # Calculate weighted cosine similarity
            # Give more weight to MFCC and delta features (first 120 features)
            weights = np.ones(min_len)
            weights[:40] = 2.0   # Higher weight for MFCC features
            weights[40:80] = 1.5 # Medium weight for delta features
            weights[80:120] = 1.2 # Lower weight for delta2 features
            
            # Apply weights
            features1_weighted = features1 * weights
            features2_weighted = features2 * weights
            
            # Calculate cosine similarity
            dot_product = np.dot(features1_weighted, features2_weighted)
            norm1 = np.linalg.norm(features1_weighted)
            norm2 = np.linalg.norm(features2_weighted)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Apply non-linear scaling to make the similarity more discriminative
            similarity = (similarity + 1) / 2  # Normalize to [0, 1]
            
            # Calculate feature-wise differences for fine-grained comparison
            diff = np.abs(features1 - features2)
            mfcc_diff = np.mean(diff[:40])  # MFCC differences
            delta_diff = np.mean(diff[40:80])  # Delta differences
            
            # Calculate amplitude difference (for handling amplitude modifications)
            amp_diff = abs(np.mean(features1[120:]) - np.mean(features2[120:]))
            
            # Calculate frequency difference (for handling pitch changes)
            freq_diff = abs(features1[-4] - features2[-4]) / max(abs(features1[-4]), abs(features2[-4]))
            
            # Adjust similarity based on feature differences
            if freq_diff > 0.3:  # Different fundamental frequency
                similarity *= 0.5  # Significantly reduce similarity for different pitches
            elif amp_diff > 0.05:  # Amplitude modification detected
                similarity = min(0.75, similarity * 0.8)  # Cap similarity for amplitude changes
            elif mfcc_diff < 0.1 and delta_diff < 0.1 and amp_diff < 0.05:  # Nearly identical
                similarity = min(0.95, similarity * 1.05)  # Boost very similar voices
            elif mfcc_diff > 0.3 or delta_diff > 0.4:  # Large differences in key features
                similarity *= 0.5
            elif similarity > 0.9:  # Very similar voices
                similarity = 0.9 + (similarity - 0.9) * 0.5
            elif similarity < 0.7:  # Different voices
                similarity *= 0.7
            
            # Additional scaling for amplitude modifications
            if amp_diff > 0.05:
                similarity = min(0.8, similarity)  # Ensure amplitude changes don't get too high confidence
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing features: {e}")
            return 0.0
            
    def get_feature_stats(self, features_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate statistics over multiple feature sets
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            Dictionary containing feature statistics
        """
        try:
            if not features_list or not all(f is not None for f in features_list):
                return {}
                
            # Convert to numpy array
            features_array = np.array(features_list)
            
            # Calculate statistics
            stats = {
                'mean': np.mean(features_array, axis=0),
                'std': np.std(features_array, axis=0),
                'min': np.min(features_array, axis=0),
                'max': np.max(features_array, axis=0)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {e}")
            return {} 