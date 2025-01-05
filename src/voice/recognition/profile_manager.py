"""
Voice Profile Manager for Speaker Recognition

This module handles the creation, storage, and management of voice profiles
for speaker recognition and authentication.
"""
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import soundfile as sf
from pydantic import BaseModel

from config.settings import AUDIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceProfile(BaseModel):
    """Voice profile data model"""
    user_id: str
    created_at: datetime
    updated_at: datetime
    samples_count: int
    features: Dict[str, List[float]]
    metadata: Dict[str, str]

class VoiceProfileManager:
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize the voice profile manager
        
        Args:
            profiles_dir: Directory to store voice profiles
        """
        # Set up directories
        self.workspace_root = Path(__file__).parent.parent.parent.parent
        self.profiles_dir = profiles_dir or (self.workspace_root / "data" / "profiles" / "voice")
        self.samples_dir = self.workspace_root / "data" / "profiles" / "samples"
        
        # Create directories if they don't exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio settings
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.channels = AUDIO_SETTINGS["channels"]
        
        # Load existing profiles
        self.profiles: Dict[str, VoiceProfile] = {}
        self._load_profiles()
        
    def _load_profiles(self):
        """Load existing voice profiles from disk"""
        logger.info("Loading voice profiles...")
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    profile = VoiceProfile(**profile_data)
                    self.profiles[profile.user_id] = profile
                    logger.info(f"Loaded profile for user: {profile.user_id}")
            except Exception as e:
                logger.error(f"Error loading profile {profile_file}: {e}")
    
    def create_profile(self, user_id: str, metadata: Dict[str, str] = None) -> VoiceProfile:
        """Create a new voice profile
        
        Args:
            user_id: Unique identifier for the user
            metadata: Additional profile information
            
        Returns:
            VoiceProfile: The created profile
        """
        if user_id in self.profiles:
            raise ValueError(f"Profile already exists for user: {user_id}")
        
        now = datetime.now()
        profile = VoiceProfile(
            user_id=user_id,
            created_at=now,
            updated_at=now,
            samples_count=0,
            features={},
            metadata=metadata or {}
        )
        
        # Save profile
        self._save_profile(profile)
        self.profiles[user_id] = profile
        
        logger.info(f"Created new profile for user: {user_id}")
        return profile
    
    def add_voice_sample(self, user_id: str, audio_data: np.ndarray, sample_rate: int = None):
        """Add a voice sample to a user's profile
        
        Args:
            user_id: User identifier
            audio_data: Audio sample data
            sample_rate: Sample rate of the audio
        """
        if user_id not in self.profiles:
            raise ValueError(f"No profile found for user: {user_id}")
        
        profile = self.profiles[user_id]
        
        # Save audio sample
        sample_path = self.samples_dir / f"{user_id}_{profile.samples_count:04d}.wav"
        sf.write(str(sample_path), audio_data, sample_rate or self.sample_rate)
        
        # Update profile
        profile.samples_count += 1
        profile.updated_at = datetime.now()
        self._save_profile(profile)
        
        logger.info(f"Added voice sample for user: {user_id} (total: {profile.samples_count})")
    
    def get_profile(self, user_id: str) -> Optional[VoiceProfile]:
        """Get a user's voice profile
        
        Args:
            user_id: User identifier
            
        Returns:
            VoiceProfile if found, None otherwise
        """
        return self.profiles.get(user_id)
    
    def update_profile(self, profile: VoiceProfile):
        """Update a voice profile
        
        Args:
            profile: Updated profile
        """
        if profile.user_id not in self.profiles:
            raise ValueError(f"No profile found for user: {profile.user_id}")
        
        profile.updated_at = datetime.now()
        self._save_profile(profile)
        self.profiles[profile.user_id] = profile
        
        logger.info(f"Updated profile for user: {profile.user_id}")
    
    def delete_profile(self, user_id: str):
        """Delete a user's voice profile
        
        Args:
            user_id: User identifier
        """
        if user_id not in self.profiles:
            raise ValueError(f"No profile found for user: {user_id}")
        
        # Remove profile file
        profile_path = self.profiles_dir / f"{user_id}.json"
        profile_path.unlink(missing_ok=True)
        
        # Remove sample files
        for sample_file in self.samples_dir.glob(f"{user_id}_*.wav"):
            sample_file.unlink()
        
        # Remove from memory
        del self.profiles[user_id]
        
        logger.info(f"Deleted profile for user: {user_id}")
    
    def _save_profile(self, profile: VoiceProfile):
        """Save a profile to disk
        
        Args:
            profile: Profile to save
        """
        profile_path = self.profiles_dir / f"{profile.user_id}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.dict(), f, indent=2, default=str) 