"""Voice adaptation module for handling voice changes over time"""
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class VoiceAdaptation:
    def __init__(self, profiles_dir: Path):
        """Initialize voice adaptation system
        
        Args:
            profiles_dir: Directory to store adaptation history
        """
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self.adaptation_history: Dict[str, List[dict]] = {}
        self.adaptation_threshold = 0.85
        self.min_adaptation_interval = timedelta(days=30)  # Minimum time between adaptations
        self.max_history_size = 10  # Maximum number of adaptations to keep
        
        # Load existing adaptation history
        self._load_history()
        
    def adapt_profile(self, user_id: str, features: np.ndarray, score: float) -> Optional[np.ndarray]:
        """Adapt user profile based on new features
        
        Args:
            user_id: User ID
            features: New feature vector
            score: Verification score
            
        Returns:
            Optional[np.ndarray]: Adapted features if adaptation occurred, None otherwise
        """
        try:
            # Always adapt for new users
            if user_id not in self.adaptation_history:
                self.adaptation_history[user_id] = {
                    "last_adaptation": time.time(),
                    "feature_history": [features],
                    "score_history": [score]
                }
                return features
                
            # Check if adaptation is needed based on score and time
            time_since_last = time.time() - self.adaptation_history[user_id]["last_adaptation"]
            
            # Adapt if:
            # 1. Score is above threshold (0.85) AND
            # 2. Either:
            #    a. Time since last adaptation > 24 hours OR
            #    b. Score difference > 0.05 from average
            if score >= 0.85:
                avg_score = np.mean(self.adaptation_history[user_id]["score_history"])
                score_diff = abs(score - avg_score)
                
                if time_since_last > 86400 or score_diff > 0.05:
                    # Update history
                    self.adaptation_history[user_id]["last_adaptation"] = time.time()
                    self.adaptation_history[user_id]["feature_history"].append(features)
                    self.adaptation_history[user_id]["score_history"].append(score)
                    
                    # Keep only last 5 samples
                    if len(self.adaptation_history[user_id]["feature_history"]) > 5:
                        self.adaptation_history[user_id]["feature_history"] = self.adaptation_history[user_id]["feature_history"][-5:]
                        self.adaptation_history[user_id]["score_history"] = self.adaptation_history[user_id]["score_history"][-5:]
                    
                    # Calculate adapted features
                    adapted_features = np.mean(self.adaptation_history[user_id]["feature_history"], axis=0)
                    return adapted_features
                    
            return None
            
        except Exception as e:
            logger.error(f"Error adapting profile: {e}")
            return None
            
    def _should_adapt(self, user_id: str, verification_score: float) -> bool:
        """Determine if adaptation should occur"""
        # Always adapt if no history exists
        if user_id not in self.adaptation_history or not self.adaptation_history[user_id]:
            return True
            
        history = self.adaptation_history[user_id]
        last_adaptation = datetime.fromisoformat(history[-1]['timestamp'])
        time_since_last = datetime.now() - last_adaptation
        
        # Time-based adaptation (30 days)
        if time_since_last >= self.min_adaptation_interval:
            return verification_score >= self.adaptation_threshold
            
        # Score-based adaptation
        if verification_score < self.adaptation_threshold:
            return False
            
        # Check score difference
        last_score = float(history[-1]['verification_score'])
        score_diff = abs(verification_score - last_score)
        return score_diff >= 0.05  # 5% difference threshold
        
    def _calculate_adapted_features(self, history: List[dict]) -> np.ndarray:
        """Calculate adapted features from history"""
        try:
            # Extract features from history
            feature_arrays = []
            weights = []
            
            for record in history:
                try:
                    features = np.array(record['features'])
                    score = float(record['verification_score'])
                    feature_arrays.append(features)
                    weights.append(score)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid history record: {e}")
                    continue
            
            if not feature_arrays:
                if history and 'features' in history[-1]:
                    return np.array(history[-1]['features'])
                raise ValueError("No valid feature arrays found in history")
                
            # Convert to numpy arrays
            feature_arrays = np.array(feature_arrays)
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Calculate weighted average
            adapted_features = np.average(feature_arrays, weights=weights, axis=0)
            
            return adapted_features
            
        except Exception as e:
            logger.error(f"Error calculating adapted features: {e}")
            if history and 'features' in history[-1]:
                return np.array(history[-1]['features'])
            raise
        
    def _load_history(self):
        """Load adaptation history from disk"""
        try:
            history_file = self.profiles_dir / 'adaptation_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.adaptation_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading adaptation history: {e}")
            self.adaptation_history = {}
            
    def _save_history(self):
        """Save adaptation history to disk"""
        try:
            history_file = self.profiles_dir / 'adaptation_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.adaptation_history, f)
        except Exception as e:
            logger.error(f"Error saving adaptation history: {e}") 