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

class SpeakerRecognizer:
    def __init__(self, profiles_dir: Optional[Path] = None, min_samples: int = 5):
        """Initialize the speaker recognizer

        Args:
            profiles_dir: Directory to store voice profiles
            min_samples: Minimum number of voice samples required for verification
        """
        self.profiles_dir = profiles_dir or Path("data/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, VoiceProfileManager] = {}
        self.min_samples = min_samples
        self.verification_threshold = 0.7  # Threshold for verification
        
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
                    profile.audio_samples = [np.array(s) for s in data.get('audio_samples', [])]
                    profile.samples_count = data['samples_count']
                    profile.created_at = datetime.fromisoformat(data['created_at'])
                    profile.last_updated_at = datetime.fromisoformat(data['last_updated_at'])
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
                'audio_samples': [s.tolist() for s in profile.audio_samples],
                'samples_count': profile.samples_count,
                'created_at': profile.created_at.isoformat(),
                'last_updated_at': profile.last_updated_at.isoformat()
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
                profile.add_sample(audio_data, features)
                
            # Save profile
            self.profiles[user_id] = profile
            self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error enrolling user: {e}")
            raise
            
    def add_voice_sample(self, user_id: str, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Add a voice sample to an existing profile
        
        Args:
            user_id: User identifier
            audio_data: Voice sample to add
            
        Returns:
            Tuple[bool, float]: Success status and similarity score
        """
        try:
            # Get user profile
            profile = self.profiles.get(user_id)
            if not profile:
                logger.warning(f"No profile found for user: {user_id}")
                return False, 0.0
                
            # Extract features
            features = self.feature_extractor.extract_features(audio_data)
            if features is None:
                return False, 0.0
                
            # Compare with existing features
            similarities = []
            for profile_features in profile.features:
                similarity = self.feature_extractor.compare_features(features, profile_features)
                similarities.append(similarity)
                
            # Calculate average similarity
            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
            
            # Add sample if similar enough
            if avg_similarity >= 0.7 or profile.samples_count < 2:
                profile.add_sample(audio_data, features)
                self._save_profile(profile)
                return True, avg_similarity
                
            return False, avg_similarity
            
        except Exception as e:
            logger.error(f"Error adding voice sample: {e}")
            return False, 0.0
            
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
                raise ValueError(f"No profile found for user: {user_id}")
                
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
            
            # Calculate confidence based on similarity statistics
            raw_confidence = float(np.mean(similarities))
            min_similarity = float(min(similarities))
            std_similarity = float(np.std(similarities)) if len(similarities) > 1 else 0.0
            max_similarity = float(max(similarities))
            
            # Calculate amplitude ratio and noise level
            if profile.audio_samples:
                # Calculate RMS amplitude for current audio
                current_rms = np.sqrt(np.mean(np.square(audio_data)))
                
                # Calculate RMS amplitudes for profile samples
                profile_rms = [np.sqrt(np.mean(np.square(s))) for s in profile.audio_samples]
                avg_profile_rms = np.mean(profile_rms)
                
                # Calculate amplitude ratio and variation
                amplitude_ratio = np.abs(1 - current_rms / avg_profile_rms)
                amplitude_variation = np.std(profile_rms) / avg_profile_rms
                
                # Estimate noise level using high-frequency components
                noise_level = np.std(np.diff(audio_data)) / np.std(audio_data)
            else:
                amplitude_ratio = 0.0
                amplitude_variation = 0.0
                noise_level = 0.0
            
            # Check for modifications or different voice
            is_amplitude_modified = (
                amplitude_ratio > max(0.05, 2 * amplitude_variation) and  # Significant amplitude change
                noise_level < 0.15  # Not due to noise
            )
            is_modified = (
                (min_similarity < 0.8 and noise_level < 0.2) or  # Low similarity but not due to noise
                (std_similarity > 0.15) or  # High variance in similarities
                is_amplitude_modified  # Significant amplitude change but not due to noise
            )
            is_different = raw_confidence < 0.75 or min_similarity < 0.6
            
            # Apply adaptive scaling based on voice characteristics
            if is_different:
                # Different voice - ensure low confidence
                confidence = raw_confidence * 0.6
            elif is_modified or is_amplitude_modified:
                # Start with base confidence for modified audio
                base_confidence = 0.7
                
                # Calculate normalized confidence between 0 and 1
                normalized = min(1.0, max(0.0, (raw_confidence - 0.7) / 0.3))
                
                # Scale to range [0.7, 0.85] for modified audio
                confidence = base_confidence + 0.15 * normalized
                
                # Additional scaling based on modification severity
                if is_amplitude_modified:
                    # Scale down based on amplitude ratio
                    reduction = min(0.15, amplitude_ratio * 0.2)  # Cap reduction at 0.15
                    confidence = max(0.7, confidence - reduction)
            elif noise_level > 0.15:
                # Noisy audio - reduce confidence slightly but keep verification
                confidence = min(0.95, raw_confidence * 0.95)
            elif raw_confidence > 0.9 and std_similarity < 0.05:
                # Identical voice - high confidence
                confidence = max(0.92, min(1.0, raw_confidence + 0.03))
            elif raw_confidence > 0.85:
                # Very similar voice
                confidence = 0.85 + (raw_confidence - 0.85) * 0.3
            else:
                # Similar voice
                confidence = raw_confidence * 0.9
            
            # Return verification result and confidence
            return confidence >= self.verification_threshold, confidence
            
        except Exception as e:
            logger.error(f"Error verifying speaker: {e}")
            raise
            
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
            
    def get_profile(self, user_id: str) -> VoiceProfileManager:
        """Get a user's voice profile
        
        Args:
            user_id: User identifier
            
        Returns:
            VoiceProfileManager: User's voice profile
            
        Raises:
            ValueError: If profile not found
        """
        if user_id not in self.profiles:
            raise ValueError(f"No profile found for user: {user_id}")
        return self.profiles[user_id] 