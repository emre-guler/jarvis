"""Tests for voice adaptation"""
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.voice.recognition.voice_adaptation import VoiceAdaptation

@pytest.fixture
def profiles_dir(tmp_path):
    """Create temporary profiles directory"""
    return tmp_path / "profiles"

@pytest.fixture
def adaptation(profiles_dir):
    """Create voice adaptation instance"""
    profiles_dir.mkdir(exist_ok=True)
    return VoiceAdaptation(profiles_dir)

@pytest.fixture
def sample_features():
    """Create sample voice features"""
    return np.random.random(128)

class TestVoiceAdaptation:
    def test_initialization(self, adaptation, profiles_dir):
        """Test adaptation system initialization"""
        assert adaptation.profiles_dir == profiles_dir
        assert isinstance(adaptation.adaptation_history, dict)
        
    def test_history_management(self, adaptation, sample_features):
        """Test adaptation history management"""
        user_id = "test_user"
        
        # Add adaptation record
        adapted = adaptation.adapt_profile(user_id, sample_features, 0.95)
        assert adapted is not None
        
        # Check history
        assert user_id in adaptation.adaptation_history
        assert len(adaptation.adaptation_history[user_id]) == 1
        
        record = adaptation.adaptation_history[user_id][0]
        assert "timestamp" in record
        assert "features" in record
        assert "verification_score" in record
        
    def test_adaptation_threshold(self, adaptation, sample_features):
        """Test adaptation threshold conditions"""
        user_id = "test_user"
        
        # Initial adaptation
        adaptation.adapt_profile(user_id, sample_features, 0.95)
        
        # Immediate re-adaptation should not occur
        adapted = adaptation.adapt_profile(user_id, sample_features * 1.01, 0.94)
        assert adapted is None
        
        # Significant feature drift should trigger adaptation
        drifted_features = sample_features * 1.5
        adapted = adaptation.adapt_profile(user_id, drifted_features, 0.85)
        assert adapted is not None
        
    def test_time_based_adaptation(self, adaptation, sample_features, monkeypatch):
        """Test time-based adaptation triggers"""
        user_id = "test_user"
        
        # Initial adaptation
        current_time = datetime.now()
        monkeypatch.setattr('src.voice.recognition.voice_adaptation.datetime',
                           type('MockDateTime', (), {
                               'now': lambda: current_time,
                               'fromisoformat': datetime.fromisoformat
                           }))
        
        adaptation.adapt_profile(user_id, sample_features, 0.95)
        
        # Advance time by 4 months
        future_time = current_time + timedelta(days=120)
        monkeypatch.setattr('src.voice.recognition.voice_adaptation.datetime',
                           type('MockDateTime', (), {
                               'now': lambda: future_time,
                               'fromisoformat': datetime.fromisoformat
                           }))
        
        # Should trigger adaptation
        adapted = adaptation.adapt_profile(user_id, sample_features * 1.01, 0.93)
        assert adapted is not None
        
    def test_feature_averaging(self, adaptation, sample_features):
        """Test feature averaging in adaptation"""
        user_id = "test_user"
        
        # Add multiple adaptations
        features_list = [
            sample_features,
            sample_features * 1.1,
            sample_features * 0.9,
            sample_features * 1.2,
            sample_features * 0.8
        ]
        
        for features in features_list:
            adapted = adaptation.adapt_profile(user_id, features, 0.9)
            
        # Check that final adaptation is average of recent samples
        expected_average = np.mean(features_list[-5:], axis=0)
        np.testing.assert_array_almost_equal(adapted, expected_average)
        
    def test_persistence(self, adaptation, sample_features, profiles_dir):
        """Test persistence of adaptation history"""
        user_id = "test_user"
        
        # Add adaptation record
        adaptation.adapt_profile(user_id, sample_features, 0.95)
        
        # Create new instance
        new_adaptation = VoiceAdaptation(profiles_dir)
        
        # Check history is loaded
        assert user_id in new_adaptation.adaptation_history
        assert len(new_adaptation.adaptation_history[user_id]) == 1 