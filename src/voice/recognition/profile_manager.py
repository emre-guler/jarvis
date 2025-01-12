"""
Voice Profile Manager

This module manages voice profiles for speaker recognition.
"""
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import json

class VoiceProfileManager:
    def __init__(self, user_id: str, metadata: Optional[Dict] = None):
        """Initialize a voice profile
        
        Args:
            user_id: User identifier
            metadata: Optional user metadata
        """
        self.user_id = user_id
        self.metadata = metadata or {}
        self.features: List[np.ndarray] = []
        self.audio_samples: List[np.ndarray] = []  # Store audio samples for amplitude comparison
        self.samples_count = 0
        self.created_at = datetime.now()
        self.last_updated_at = self.created_at
        
    def add_sample(self, audio_data: np.ndarray, features: np.ndarray):
        """Add a voice sample to the profile
        
        Args:
            audio_data: Raw audio data
            features: Extracted features
        """
        self.features.append(features)
        self.audio_samples.append(audio_data)
        self.samples_count += 1
        self.last_updated_at = datetime.now()
        
    def get_average_features(self) -> np.ndarray:
        """Get average features across all samples
        
        Returns:
            np.ndarray: Average feature vector
        """
        if not self.features:
            raise ValueError("No features available")
        return np.mean(self.features, axis=0)
        
    def get_feature_statistics(self) -> Dict[str, np.ndarray]:
        """Get feature statistics across all samples
        
        Returns:
            Dict[str, np.ndarray]: Statistics including mean, std, min, max
        """
        if not self.features:
            raise ValueError("No features available")
            
        features_array = np.array(self.features)
        return {
            'mean': np.mean(features_array, axis=0),
            'std': np.std(features_array, axis=0),
            'min': np.min(features_array, axis=0),
            'max': np.max(features_array, axis=0)
        }
        
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialization
        
        Returns:
            Dict: Serializable dictionary
        """
        return {
            'user_id': self.user_id,
            'metadata': self.metadata,
            'features': [f.tolist() for f in self.features],
            'audio_samples': [s.tolist() for s in self.audio_samples],
            'samples_count': self.samples_count,
            'created_at': self.created_at.isoformat(),
            'last_updated_at': self.last_updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceProfileManager':
        """Create profile from dictionary
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            VoiceProfileManager: Created profile
        """
        profile = cls(data['user_id'], data.get('metadata'))
        profile.features = [np.array(f) for f in data['features']]
        profile.audio_samples = [np.array(s) for s in data.get('audio_samples', [])]
        profile.samples_count = data['samples_count']
        profile.created_at = datetime.fromisoformat(data['created_at'])
        profile.last_updated_at = datetime.fromisoformat(data['last_updated_at'])
        return profile 