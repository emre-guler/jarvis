"""
Speaker Recognition Module for Jarvis
"""
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from python_speech_features import mfcc
from sklearn.preprocessing import StandardScaler
import librosa

from config.settings import AUDIO_SETTINGS, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerRecognition:
    """Speaker Recognition System for Jarvis"""

    def __init__(self):
        """Initialize the speaker recognition system with enhanced model"""
        self.model = self._create_enhanced_model()
        self.feature_extractor = self._create_feature_extractor()
        self.scaler = StandardScaler()
        self.threshold = 0.95  # Increased threshold for stricter verification
        
    def _create_enhanced_model(self):
        """Create an enhanced deep neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
        
    def _create_feature_extractor(self):
        """Create an enhanced feature extractor"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None,)),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Conv1D(64, 8, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 8, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(256, 8, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        
    def extract_features(self, audio_data):
        """Extract enhanced voice features"""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=16000)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=16000)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=16000)
        
        # Extract prosodic features
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=50, fmax=600)
        
        # Combine all features
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.mean(f0[~np.isnan(f0)]) if len(f0[~np.isnan(f0)]) > 0 else [0]
        ])
        
        return features

    def create_profile(self, user_id: str, audio_samples: List[np.ndarray]) -> bool:
        """Create a voice profile for a user
        
        Args:
            user_id: Unique identifier for the user
            audio_samples: List of audio samples for profile creation
            
        Returns:
            bool: True if profile creation was successful
        """
        try:
            # Extract features from all samples
            features = []
            for sample in audio_samples:
                sample_features = self._extract_features(sample)
                if sample_features is not None:
                    features.append(sample_features)
            
            if not features:
                logger.error("No valid features extracted from audio samples")
                return False
            
            # Generate embedding by averaging features
            embeddings = [self._generate_embedding(f) for f in features]
            profile_embedding = np.mean(embeddings, axis=0)
            
            # Save profile
            self.voice_profiles[user_id] = profile_embedding
            self._save_profile(user_id, profile_embedding)
            
            logger.info(f"Voice profile created for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating voice profile: {e}")
            return False

    def verify_speaker(self, audio_data: np.ndarray, claimed_id: str) -> Tuple[bool, float]:
        """Verify if the audio matches the claimed user's voice profile
        
        Args:
            audio_data: Audio data to verify
            claimed_id: ID of the claimed user
            
        Returns:
            Tuple[bool, float]: Verification result and confidence score
        """
        try:
            if claimed_id not in self.voice_profiles:
                logger.warning(f"No voice profile found for user {claimed_id}")
                return False, 0.0
            
            # Extract features and generate embedding
            features = self._extract_features(audio_data)
            if features is None:
                return False, 0.0
                
            embedding = self._generate_embedding(features)
            
            # Compare with stored profile
            similarity = self._compute_similarity(embedding, self.voice_profiles[claimed_id])
            
            # Verify based on similarity threshold
            threshold = MODEL_CONFIG.get("speaker_verification_threshold", 0.85)
            is_verified = similarity > threshold
            
            return is_verified, similarity
            
        except Exception as e:
            logger.error(f"Error during speaker verification: {e}")
            return False, 0.0

    def _extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features from audio data"""
        try:
            features = mfcc(
                audio_data,
                samplerate=AUDIO_SETTINGS["sample_rate"],
                numcep=MODEL_CONFIG["n_mfcc"],
                nfilt=MODEL_CONFIG["n_mel"],
                winlen=MODEL_CONFIG["window_size"],
                winstep=MODEL_CONFIG["hop_size"]
            )
            
            # Normalize features
            features = self.scaler.fit_transform(features)
            
            # Reshape for model input
            features = features.reshape(1, features.shape[0], features.shape[1], 1)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def _generate_embedding(self, features: np.ndarray) -> np.ndarray:
        """Generate voice embedding from features"""
        embedding = self.model.predict(features, verbose=0)
        return embedding.flatten()

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)

    def load_profiles(self):
        """Load all voice profiles from disk"""
        try:
            for profile_file in self.profiles_dir.glob("*.pkl"):
                user_id = profile_file.stem
                with open(profile_file, 'rb') as f:
                    self.voice_profiles[user_id] = pickle.load(f)
            logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
        except Exception as e:
            logger.error(f"Error loading voice profiles: {e}")

    def _save_profile(self, user_id: str, embedding: np.ndarray):
        """Save a voice profile to disk"""
        try:
            profile_path = self.profiles_dir / f"{user_id}.pkl"
            with open(profile_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Error saving voice profile: {e}")
            raise 