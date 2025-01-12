"""
Feature Extractor for Voice Recognition

This module handles feature extraction from voice samples.
"""
import numpy as np
import librosa
from typing import Dict, List

class VoiceFeatureExtractor:
    def __init__(self, sample_rate: int = 16000):
        """Initialize the feature extractor
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features from audio data
        
        Args:
            audio_data: Raw audio samples
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=20)
            
            # Calculate delta features
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Calculate statistics
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            delta_mean = np.mean(mfcc_delta, axis=1)
            delta_std = np.std(mfcc_delta, axis=1)
            delta2_mean = np.mean(mfcc_delta2, axis=1)
            delta2_std = np.std(mfcc_delta2, axis=1)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate))
            
            # Extract prosodic features
            rms = np.sqrt(np.mean(np.square(audio_data)))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            f0, voiced_flag, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_mean = np.mean(f0[voiced_flag]) if np.any(voiced_flag) else 0
            f0_std = np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                delta_mean, delta_std,
                delta2_mean, delta2_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [rms, zero_crossing_rate, f0_mean, f0_std]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
            
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Split features into components
            mfcc_size = 20  # Number of MFCC coefficients
            spectral_size = 3  # Number of spectral features
            prosodic_size = 4  # Number of prosodic features
            
            # Extract feature components
            mfcc1 = features1[:mfcc_size * 6]  # MFCC and deltas
            mfcc2 = features2[:mfcc_size * 6]
            
            spectral1 = features1[mfcc_size * 6:mfcc_size * 6 + spectral_size]
            spectral2 = features2[mfcc_size * 6:mfcc_size * 6 + spectral_size]
            
            prosodic1 = features1[-prosodic_size:]
            prosodic2 = features2[-prosodic_size:]
            
            # Calculate component similarities
            mfcc_sim = 1 - np.mean(np.abs(mfcc1 - mfcc2) / (np.abs(mfcc1) + np.abs(mfcc2) + 1e-6))
            spectral_sim = 1 - np.mean(np.abs(spectral1 - spectral2) / (np.abs(spectral1) + np.abs(spectral2) + 1e-6))
            
            # Calculate prosodic similarity with higher weight for RMS
            rms_sim = 1 - np.abs(prosodic1[0] - prosodic2[0]) / (max(prosodic1[0], prosodic2[0]) + 1e-6)
            other_prosodic_sim = 1 - np.mean(np.abs(prosodic1[1:] - prosodic2[1:]) / (np.abs(prosodic1[1:]) + np.abs(prosodic2[1:]) + 1e-6))
            prosodic_sim = 0.6 * rms_sim + 0.4 * other_prosodic_sim
            
            # Combine similarities with weights
            similarity = 0.5 * mfcc_sim + 0.3 * spectral_sim + 0.2 * prosodic_sim
            
            # Apply non-linear scaling to make it more sensitive to differences
            similarity = np.power(similarity, 1.2)
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error comparing features: {e}")
            return 0.0
            
    def get_feature_stats(self, features_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate statistics over multiple feature sets
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            Dict[str, np.ndarray]: Statistics including mean, std, min, max
        """
        try:
            if not features_list:
                return {}
                
            # Convert to numpy array
            features_array = np.array(features_list)
            
            # Calculate statistics
            return {
                'mean': np.mean(features_array, axis=0),
                'std': np.std(features_array, axis=0),
                'min': np.min(features_array, axis=0),
                'max': np.max(features_array, axis=0)
            }
            
        except Exception as e:
            print(f"Error calculating feature statistics: {e}")
            return {} 