"""
Speaker Recognition System

This module provides the main speaker recognition functionality.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime
import json

import numpy as np

from .profile_manager import VoiceProfileManager
from .feature_extractor import VoiceFeatureExtractor
from ..security.encryption import ProfileEncryption
from ..security.anti_spoofing import AntiSpoofing
from ..optimization.performance import PerformanceOptimizer

logger = logging.getLogger(__name__)

class VoiceProfileManager:
    def __init__(self, user_id: str, metadata: Optional[Dict] = None):
        self.user_id = user_id
        self.metadata = metadata or {}
        self.features = []
        self.samples_count = 0
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

class SpeakerRecognizer:
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize speaker recognition system"""
        self.profiles_dir = profiles_dir or Path("data/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, VoiceProfileManager] = {}
        self.min_samples = 5
        self.verification_threshold = 0.85
        
        # Initialize components
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Load existing profiles
        self._load_profiles()
        
    def _load_profiles(self):
        """Load existing voice profiles"""
        try:
            for profile_file in self.profiles_dir.glob("*.json"):
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = VoiceProfileManager(data['user_id'], data.get('metadata'))
                    profile.features = [np.array(f) for f in data['features']]
                    profile.samples_count = data['samples_count']
                    profile.created_at = datetime.fromisoformat(data['created_at'])
                    profile.last_updated = datetime.fromisoformat(data['last_updated'])
                    self.profiles[profile.user_id] = profile
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            
    def _save_profile(self, profile: VoiceProfileManager):
        """Save profile to disk"""
        try:
            profile_data = {
                'user_id': profile.user_id,
                'metadata': profile.metadata,
                'features': [f.tolist() for f in profile.features],
                'samples_count': profile.samples_count,
                'created_at': profile.created_at.isoformat(),
                'last_updated': profile.last_updated.isoformat()
            }
            
            profile_path = self.profiles_dir / f"{profile.user_id}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f)
                
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            raise
            
    def enroll_user(self, user_id: str, audio_data: np.ndarray, metadata: Optional[Dict] = None) -> VoiceProfileManager:
        """Enroll a new user
        
        Args:
            user_id: User identifier
            audio_data: Voice sample
            metadata: Optional user metadata
            
        Returns:
            VoiceProfileManager: Created profile
        """
        try:
            # Create new profile
            profile = VoiceProfileManager(user_id, metadata)
            
            # Extract and add initial features
            features = self.feature_extractor.extract_features(audio_data)
            if features is not None:
                profile.features.append(features)
                profile.samples_count += 1
                
            # Save profile
            self.profiles[user_id] = profile
            self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error enrolling user: {e}")
            raise
            
    def verify_speaker(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Verify a speaker's identity
        
        Args:
            user_id: User identifier
            audio_data: Voice sample to verify
            
        Returns:
            Tuple[bool, float]: Verification result and confidence score
        """
        try:
            # Get user profile
            profile = self.profiles.get(user_id)
            if not profile:
                logger.warning(f"No profile found for user: {user_id}")
                return False, 0.0
                
            # Check minimum samples
            if profile.samples_count < self.min_samples:
                logger.warning(f"Insufficient samples for user {user_id}: {profile.samples_count}/{self.min_samples}")
                return False, 0.0
                
            # Extract features
            features = self.feature_extractor.extract_features(audio_data)
            if features is None:
                return False, 0.0
                
            # Compare with profile features
            similarities = []
            for profile_features in profile.features:
                similarity = self.feature_extractor.compare_features(features, profile_features)
                similarities.append(similarity)
                
            # Calculate confidence score
            confidence = float(np.mean(similarities))
            
            # Verify against threshold
            is_verified = confidence >= self.verification_threshold
            
            return is_verified, confidence
            
        except Exception as e:
            logger.error(f"Error verifying speaker: {e}")
            return False, 0.0
            
    def delete_user(self, user_id: str):
        """Delete a user profile"""
        try:
            if user_id not in self.profiles:
                raise ValueError(f"No profile found for user: {user_id}")
                
            # Remove from memory
            del self.profiles[user_id]
            
            # Remove from disk
            profile_path = self.profiles_dir / f"{user_id}.json"
            if profile_path.exists():
                profile_path.unlink()
                
        except Exception as e:
            logger.error(f"Error deleting user profile: {e}")
            raise 