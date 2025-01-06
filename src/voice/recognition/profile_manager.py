"""Voice profile manager for speaker recognition"""
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class VoiceProfileManager:
    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize the voice profile manager
        
        Args:
            profiles_dir: Directory to store voice profiles
        """
        # Set up directories
        self.workspace_root = Path(__file__).parent.parent.parent.parent
        self.profiles_dir = profiles_dir or (self.workspace_root / "data" / "profiles" / "voice")
        
        # Create directory if it doesn't exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing profiles
        self.profiles: Dict[str, Dict] = {}
        self._load_profiles()
        
    def _load_profiles(self):
        """Load existing voice profiles from disk"""
        logger.info("Loading voice profiles...")
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    self.profiles[profile_data['user_id']] = profile_data
                    logger.info(f"Loaded profile for user: {profile_data['user_id']}")
            except Exception as e:
                logger.error(f"Error loading profile {profile_file}: {e}")
                
    def save_profile(self, user_id: str, profile: Dict) -> bool:
        """Save a profile to disk
        
        Args:
            user_id: User identifier
            profile: Profile data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            profile_path = self.profiles_dir / f"{user_id}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            # Update in-memory cache
            self.profiles[user_id] = profile
            
            logger.info(f"Saved profile for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            return False
            
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """Get a user's voice profile
        
        Args:
            user_id: User identifier
            
        Returns:
            Optional[Dict]: Profile data if found, None otherwise
        """
        return self.profiles.get(user_id)
        
    def delete_profile(self, user_id: str) -> bool:
        """Delete a user's voice profile
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove profile file
            profile_path = self.profiles_dir / f"{user_id}.json"
            if profile_path.exists():
                profile_path.unlink()
            
            # Remove from memory
            if user_id in self.profiles:
                del self.profiles[user_id]
                
            logger.info(f"Deleted profile for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False 