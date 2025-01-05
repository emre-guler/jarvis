"""
Unit Tests for Speaker Recognition System
"""
import pytest
import numpy as np
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

from src.voice.recognition.speaker_recognizer import SpeakerRecognizer
from src.voice.recognition.profile_manager import VoiceProfile
from src.voice.recognition.feature_extractor import VoiceFeatureExtractor

@pytest.fixture
def temp_profiles_dir():
    """Create a temporary directory for test profiles"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing"""
    # Generate 1 second of synthetic audio (sine wave)
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t)

@pytest.fixture
def feature_extractor():
    """Create a feature extractor instance"""
    return VoiceFeatureExtractor()

@pytest.fixture
def speaker_recognizer(temp_profiles_dir):
    """Create a speaker recognizer instance with temporary directory"""
    return SpeakerRecognizer(profiles_dir=temp_profiles_dir)

def test_feature_extraction(feature_extractor, mock_audio_data):
    """Test voice feature extraction"""
    features = feature_extractor.extract_features(mock_audio_data)
    
    # Verify feature structure
    assert isinstance(features, dict)
    assert 'mfcc_mean' in features
    assert 'f0_mean' in features
    assert 'zcr_mean' in features
    
    # Verify feature dimensions
    assert len(features['mfcc_mean']) == feature_extractor.n_mfcc
    assert isinstance(features['f0_mean'], list)
    assert isinstance(features['zcr_mean'], list)
    assert len(features['f0_mean']) == 1
    assert len(features['zcr_mean']) == 1

def test_feature_comparison(feature_extractor, mock_audio_data):
    """Test feature comparison functionality"""
    # Extract features from same audio twice
    features1 = feature_extractor.extract_features(mock_audio_data)
    features2 = feature_extractor.extract_features(mock_audio_data)
    
    # Compare identical features
    similarity = feature_extractor.compare_features(features1, features2)
    assert similarity >= 0.95  # Should be nearly identical
    
    # Compare with slightly modified audio
    modified_audio = mock_audio_data * 1.1  # Change amplitude
    features3 = feature_extractor.extract_features(modified_audio)
    similarity = feature_extractor.compare_features(features1, features3)
    assert 0.7 <= similarity <= 1.0  # Should be similar but not identical

def test_profile_creation(speaker_recognizer, mock_audio_data):
    """Test user profile creation"""
    user_id = "test_user"
    metadata = {"name": "Test User", "age": "30"}
    
    # Create profile
    profile = speaker_recognizer.enroll_user(user_id, mock_audio_data, metadata)
    
    # Verify profile
    assert isinstance(profile, VoiceProfile)
    assert profile.user_id == user_id
    assert profile.metadata == metadata
    assert profile.samples_count == 1
    assert isinstance(profile.created_at, datetime)
    assert len(profile.features) > 0

def test_speaker_verification(speaker_recognizer, mock_audio_data):
    """Test speaker verification"""
    user_id = "test_user"
    
    # Enroll user
    speaker_recognizer.enroll_user(user_id, mock_audio_data)
    
    # Add more samples to meet minimum requirement
    for _ in range(4):
        speaker_recognizer.add_voice_sample(user_id, mock_audio_data)
    
    # Test verification with same audio
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, mock_audio_data)
    assert is_verified
    assert confidence > 0.9
    
    # Test verification with different audio
    different_audio = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 16000))  # Different frequency
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, different_audio)
    assert not is_verified
    assert confidence < 0.85

def test_profile_management(speaker_recognizer, mock_audio_data):
    """Test profile management operations"""
    user_id = "test_user"
    
    # Create profile
    profile = speaker_recognizer.enroll_user(user_id, mock_audio_data)
    assert profile is not None
    
    # Add sample
    success, similarity = speaker_recognizer.add_voice_sample(user_id, mock_audio_data)
    assert success
    assert similarity > 0.9
    
    # Delete profile
    speaker_recognizer.delete_user(user_id)
    with pytest.raises(ValueError):
        speaker_recognizer.verify_speaker(user_id, mock_audio_data)

def test_minimum_samples_requirement(speaker_recognizer, mock_audio_data):
    """Test minimum samples requirement for verification"""
    user_id = "test_user"
    
    # Create profile with one sample
    speaker_recognizer.enroll_user(user_id, mock_audio_data)
    
    # Verify with insufficient samples
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, mock_audio_data)
    assert not is_verified
    assert confidence == 0.0
    
    # Add required samples
    for _ in range(4):
        speaker_recognizer.add_voice_sample(user_id, mock_audio_data)
    
    # Verify again
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, mock_audio_data)
    assert is_verified
    assert confidence > 0.0

def test_feature_statistics(feature_extractor, mock_audio_data):
    """Test feature statistics calculation"""
    # Generate multiple feature sets
    features_list = [
        feature_extractor.extract_features(mock_audio_data),
        feature_extractor.extract_features(mock_audio_data * 1.1),
        feature_extractor.extract_features(mock_audio_data * 0.9)
    ]
    
    # Calculate statistics
    stats = feature_extractor.get_feature_stats(features_list)
    
    # Verify statistics
    assert isinstance(stats, dict)
    assert all('_mean' in key or '_std' in key for key in stats.keys())
    assert all(isinstance(value, list) for value in stats.values())
    # Verify non-empty lists
    assert all(len(value) > 0 for value in stats.values()) 