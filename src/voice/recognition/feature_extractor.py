"""
Voice Feature Extractor for Speaker Recognition

This module handles the extraction of voice characteristics and features
used for speaker recognition and verification.
"""
import logging
import numpy as np
from typing import Dict, List
import librosa
from python_speech_features import mfcc, delta

from config.settings import AUDIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, List[float]]:
        """Extract voice features from audio data
        
        Args:
            audio_data: Audio signal data
            
        Returns:
            Dictionary containing different feature types
        """
        features = {}
        
        try:
            # Ensure audio data is normalized
            audio_data = librosa.util.normalize(audio_data)
            
            # 1. Basic MFCC features
            mfcc_features = mfcc(
                audio_data,
                samplerate=self.sample_rate,
                numcep=self.n_mfcc,
                nfilt=self.n_mels,
                nfft=self.n_fft,
                winlen=self.frame_length,
                winstep=self.frame_shift
            )
            
            # Calculate delta and delta-delta features
            delta_features = delta(mfcc_features, 2)
            delta2_features = delta(delta_features, 2)
            
            # 2. Spectral features
            # Spectral centroid
            cent = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.n_fft
            )
            
            # 3. Prosodic features
            # Fundamental frequency (F0)
            f0, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=audio_data)
            
            # Store all features
            features.update({
                'mfcc_mean': np.nan_to_num(np.mean(mfcc_features, axis=0)).tolist(),
                'mfcc_std': np.nan_to_num(np.std(mfcc_features, axis=0)).tolist(),
                'delta_mean': np.nan_to_num(np.mean(delta_features, axis=0)).tolist(),
                'delta_std': np.nan_to_num(np.std(delta_features, axis=0)).tolist(),
                'delta2_mean': np.nan_to_num(np.mean(delta2_features, axis=0)).tolist(),
                'delta2_std': np.nan_to_num(np.std(delta2_features, axis=0)).tolist(),
                'centroid_mean': [float(np.nan_to_num(np.mean(cent)))],
                'bandwidth_mean': [float(np.nan_to_num(np.mean(bandwidth)))],
                'rolloff_mean': [float(np.nan_to_num(np.mean(rolloff)))],
                'f0_mean': [float(np.nan_to_num(np.nanmean(f0[voiced_flag])) if np.any(voiced_flag) else 0.0)],
                'f0_std': [float(np.nan_to_num(np.nanstd(f0[voiced_flag])) if np.any(voiced_flag) else 0.0)],
                'zcr_mean': [float(np.nan_to_num(np.mean(zcr)))],
                'rms_mean': [float(np.nan_to_num(np.mean(rms)))]
            })
            
            logger.info("Successfully extracted voice features")
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
        
        return features
    
    def compare_features(self, features1: Dict[str, List[float]], 
                        features2: Dict[str, List[float]]) -> float:
        """Compare two sets of voice features
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Calculate cosine similarity for each feature type
            similarities = []
            
            for key in features1.keys():
                if key in features2:
                    v1 = np.array(features1[key], dtype=np.float32)
                    v2 = np.array(features2[key], dtype=np.float32)
                    
                    # Ensure non-zero vectors
                    if np.any(v1) and np.any(v2):
                        # Calculate cosine similarity
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        # Handle NaN values
                        if not np.isnan(similarity):
                            similarities.append(similarity)
            
            # Return average similarity or 0 if no valid comparisons
            if similarities:
                return float(np.mean(similarities))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error comparing features: {e}")
            raise
    
    def get_feature_stats(self, features_list: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """Calculate statistics over multiple feature sets
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Dictionary containing feature statistics
        """
        try:
            stats = {}
            
            # Combine all features
            for key in features_list[0].keys():
                values = [np.array(f[key], dtype=np.float32) for f in features_list]
                stats[f"{key}_mean"] = np.nan_to_num(np.mean(values, axis=0)).tolist()
                stats[f"{key}_std"] = np.nan_to_num(np.std(values, axis=0)).tolist()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {e}")
            raise 