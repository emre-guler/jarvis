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

from config.settings import AUDIO_SETTINGS, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerRecognition:
    """Speaker Recognition System for Jarvis"""

    def __init__(self, model_path: Optional[Path] = None, profiles_dir: Optional[Path] = None):
        """Initialize the speaker recognition system
        
        Args:
            model_path: Path to the speaker recognition model file
            profiles_dir: Directory to store voice profiles
        """
        self.model_path = model_path
        self.profiles_dir = profiles_dir or Path("data/voice_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # Load voice profiles
        self.voice_profiles: Dict[str, np.ndarray] = {}
        self.load_profiles()
        
        # Initialize feature scaler
        self.scaler = StandardScaler()

    def _load_model(self):
        """Load the speaker recognition model"""
        try:
            if self.model_path and self.model_path.exists():
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Speaker recognition model loaded from {self.model_path}")
            else:
                logger.warning("No model file found, using dummy model for testing")
                self.model = self._create_dummy_model()
            
            # Warm up the model
            dummy_input = np.zeros((1, 13, 49, 1))
            self.model.predict(dummy_input)
            
        except Exception as e:
            logger.error(f"Error loading speaker recognition model: {e}")
            raise

    def _create_dummy_model(self):
        """Create a simple model for testing"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(13, 49, 1)),  # MFCC features
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32)  # Embedding dimension
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

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