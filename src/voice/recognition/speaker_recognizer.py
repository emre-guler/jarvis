"""
Speaker Recognition System

This module provides the main speaker recognition functionality.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

from .profile_manager import VoiceProfileManager
from .feature_extractor import VoiceFeatureExtractor
from ..security.encryption import ProfileEncryption
from ..security.anti_spoofing import AntiSpoofing
from ..optimization.performance import PerformanceOptimizer

logger = logging.getLogger(__name__)

class SpeakerRecognizer:
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize speaker recognition system"""
        self.profile_manager = VoiceProfileManager(profiles_dir)
        self.feature_extractor = VoiceFeatureExtractor()
        self.encryption = ProfileEncryption()
        self.anti_spoofing = AntiSpoofing()
        self.performance = PerformanceOptimizer()
        
        # Recognition settings
        self.verification_threshold = 0.85
        self.min_samples = 5
        self.similarity_threshold = 0.80
        
        # Start performance monitoring
        self.performance.start_monitoring()
        
    def enroll_user(self, user_id: str, audio_data: np.ndarray, metadata: Optional[Dict] = None) -> Optional[VoiceProfileManager]:
        """Enroll a new user with initial voice sample"""
        try:
            # Check if user already exists
            if self.profile_manager.get_profile(user_id) is not None:
                raise ValueError(f"User {user_id} already exists")
                
            # Check for spoofing
            if not self.anti_spoofing.check_audio(audio_data, self.feature_extractor.sample_rate):
                logger.warning("Possible spoofing attempt detected during enrollment")
                return None
                
            # Extract features
            features = self.feature_extractor.extract_features(audio_data)
            if features is None:
                logger.error("Failed to extract features during enrollment")
                return None
                
            # Create profile
            profile = {
                'user_id': user_id,
                'features': features,
                'metadata': metadata or {},
                'samples_count': 1,
                'created_at': time.time(),
                'updated_at': time.time()
            }
            
            # Encrypt features before saving
            profile['features'] = self.encryption.encrypt_data(features)
            
            # Save profile
            success = self.profile_manager.save_profile(user_id, profile)
            if not success:
                return None
                
            # Set user_id on profile manager
            self.profile_manager.user_id = user_id
            return self.profile_manager
            
        except Exception as e:
            logger.error(f"Error during user enrollment: {e}")
            return None
            
    def add_voice_sample(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Add a voice sample to an existing profile"""
        try:
            # Get existing profile
            profile = self.profile_manager.get_profile(user_id)
            if not profile:
                logger.warning(f"No profile found for user: {user_id}")
                return False, 0.0
                
            # Extract features
            new_features = self.feature_extractor.extract_features(audio_data)
            if new_features is None:
                return False, 0.0
                
            # Compare with existing features
            existing_features = self.encryption.decrypt_data(profile['features'])
            similarity = self.feature_extractor.compare_features(new_features, existing_features)
            
            # Add sample if similar enough
            if similarity >= self.similarity_threshold:
                # Update profile with averaged features
                avg_features = (existing_features * profile['samples_count'] + new_features) / (profile['samples_count'] + 1)
                
                profile['features'] = self.encryption.encrypt_data(avg_features)
                profile['samples_count'] += 1
                profile['updated_at'] = time.time()
                
                success = self.profile_manager.save_profile(user_id, profile)
                return success, similarity
                
            return False, similarity
            
        except Exception as e:
            logger.error(f"Error adding voice sample: {e}")
            return False, 0.0
            
    def delete_user(self, user_id: str) -> bool:
        """Delete a user's voice profile"""
        try:
            success = self.profile_manager.delete_profile(user_id)
            if not success:
                raise ValueError(f"Failed to delete user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user profile: {e}")
            raise
            
    def verify_speaker(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Verify a speaker's identity
        
        Args:
            user_id: User identifier
            audio_data: Audio data to verify
            
        Returns:
            Tuple[bool, float]: (verification success, confidence score)
        """
        try:
            start_time = time.time()
            
            # Get profile
            profile = self.profile_manager.get_profile(user_id)
            if not profile:
                logger.warning(f"No profile found for user: {user_id}")
                return False, 0.0
                
            # Check minimum samples
            if profile['samples_count'] < self.min_samples:
                logger.warning(f"Insufficient voice samples for user: {user_id}")
                return False, 0.0
                
            # Check for spoofing
            if not self.anti_spoofing.check_audio(audio_data, self.feature_extractor.sample_rate):
                logger.warning("Possible spoofing attempt detected")
                return False, 0.035  # Low confidence score
                
            # Extract features
            test_features = self.feature_extractor.extract_features(audio_data)
            if test_features is None:
                return False, 0.0
                
            # Decrypt stored features
            stored_features = self.encryption.decrypt_data(profile['features'])
            
            # Compare features
            similarity = self.feature_extractor.compare_features(test_features, stored_features)
            
            # Record verification metrics
            duration = time.time() - start_time
            is_verified = similarity >= self.verification_threshold
            self.performance.record_verification(duration, is_verified, similarity)
            
            # Get optimized settings periodically
            settings = self.performance.optimize_settings()
            if settings:
                self._apply_optimized_settings(settings)
                
            return is_verified, similarity
            
        except Exception as e:
            logger.error(f"Error during speaker verification: {e}")
            return False, 0.0
            
    def _apply_optimized_settings(self, settings: dict):
        """Apply optimized settings"""
        try:
            if 'threshold' in settings:
                self.verification_threshold = settings['threshold']
                
            if 'feature_settings' in settings:
                feat_settings = settings['feature_settings']
                self.feature_extractor.n_mfcc = feat_settings.get('n_mfcc', self.feature_extractor.n_mfcc)
                self.feature_extractor.frame_length = feat_settings.get('frame_length', self.feature_extractor.frame_length)
                self.feature_extractor.frame_shift = feat_settings.get('frame_shift', self.feature_extractor.frame_shift)
                
        except Exception as e:
            logger.error(f"Error applying optimized settings: {e}")
            
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        return self.performance.get_performance_summary() 