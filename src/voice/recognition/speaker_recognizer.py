"""
Speaker Recognition System

This module provides speaker recognition and verification capabilities
by combining voice profile management and feature extraction.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .profile_manager import VoiceProfileManager, VoiceProfile
from .feature_extractor import VoiceFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeakerRecognizer:
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize the speaker recognition system
        
        Args:
            profiles_dir: Directory to store voice profiles
        """
        self.profile_manager = VoiceProfileManager(profiles_dir)
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Recognition settings
        self.min_similarity_threshold = 0.85  # Minimum similarity score for verification
        self.min_samples_required = 5        # Minimum samples needed for reliable verification
        
    def enroll_user(self, user_id: str, audio_data: np.ndarray, metadata: dict = None) -> VoiceProfile:
        """Enroll a new user in the system
        
        Args:
            user_id: Unique identifier for the user
            audio_data: Initial voice sample
            metadata: Additional user information
            
        Returns:
            Created voice profile
        """
        try:
            # Create new profile
            profile = self.profile_manager.create_profile(user_id, metadata)
            
            # Extract and add features
            features = self.feature_extractor.extract_features(audio_data)
            profile.features = features
            
            # Add voice sample
            self.profile_manager.add_voice_sample(user_id, audio_data)
            
            # Update profile
            self.profile_manager.update_profile(profile)
            
            logger.info(f"Enrolled new user: {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error enrolling user: {e}")
            raise
    
    def add_voice_sample(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Add a voice sample to an existing profile
        
        Args:
            user_id: User identifier
            audio_data: Voice sample data
            
        Returns:
            Tuple of (success, similarity_score)
        """
        try:
            # Get existing profile
            profile = self.profile_manager.get_profile(user_id)
            if not profile:
                raise ValueError(f"No profile found for user: {user_id}")
            
            # Extract features from new sample
            new_features = self.feature_extractor.extract_features(audio_data)
            
            # Compare with existing features
            similarity = self.feature_extractor.compare_features(profile.features, new_features)
            
            # If similarity is high enough, add sample
            if similarity >= self.min_similarity_threshold:
                self.profile_manager.add_voice_sample(user_id, audio_data)
                
                # Update profile features with running average
                if profile.samples_count >= 1:
                    # Calculate weighted average of features
                    weight_old = profile.samples_count / (profile.samples_count + 1)
                    weight_new = 1 / (profile.samples_count + 1)
                    
                    for key in profile.features:
                        old_value = np.array(profile.features[key])
                        new_value = np.array(new_features[key])
                        profile.features[key] = (old_value * weight_old + new_value * weight_new).tolist()
                
                self.profile_manager.update_profile(profile)
                logger.info(f"Added voice sample for user {user_id} (similarity: {similarity:.2%})")
                return True, similarity
            else:
                logger.warning(f"Voice sample rejected for user {user_id} (similarity: {similarity:.2%})")
                return False, similarity
                
        except Exception as e:
            logger.error(f"Error adding voice sample: {e}")
            raise
    
    def verify_speaker(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Verify if an audio sample matches a user's voice profile
        
        Args:
            user_id: User identifier
            audio_data: Voice sample to verify
            
        Returns:
            Tuple of (is_verified, confidence_score)
        """
        try:
            # Get user profile
            profile = self.profile_manager.get_profile(user_id)
            if not profile:
                raise ValueError(f"No profile found for user: {user_id}")
            
            # Check if we have enough samples
            if profile.samples_count < self.min_samples_required:
                logger.warning(f"Insufficient samples for reliable verification (have: {profile.samples_count}, need: {self.min_samples_required})")
                return False, 0.0
            
            # Extract features
            test_features = self.feature_extractor.extract_features(audio_data)
            
            # Compare features
            similarity = self.feature_extractor.compare_features(profile.features, test_features)
            
            # Verify based on similarity threshold
            is_verified = similarity >= self.min_similarity_threshold
            
            if is_verified:
                logger.info(f"Speaker verified: {user_id} (confidence: {similarity:.2%})")
            else:
                logger.warning(f"Speaker verification failed: {user_id} (confidence: {similarity:.2%})")
            
            return is_verified, similarity
            
        except Exception as e:
            logger.error(f"Error during speaker verification: {e}")
            raise
    
    def delete_user(self, user_id: str):
        """Remove a user's voice profile from the system
        
        Args:
            user_id: User identifier
        """
        try:
            self.profile_manager.delete_profile(user_id)
            logger.info(f"Deleted user profile: {user_id}")
        except Exception as e:
            logger.error(f"Error deleting user profile: {e}")
            raise 