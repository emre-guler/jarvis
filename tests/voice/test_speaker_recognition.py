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
from src.voice.recognition.profile_manager import VoiceProfileManager
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
    assert isinstance(features, np.ndarray)
    assert features is not None
    assert len(features) > 0
    
    # Expected feature count:
    # - MFCC mean and std (20 * 2)
    # - Delta mean and std (20 * 2)
    # - Delta2 mean and std (20 * 2)
    # - Spectral features (3)
    # - Prosodic features (4)
    expected_features = (20 * 6) + 7
    assert len(features) == expected_features, f"Expected {expected_features} features, got {len(features)}"

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
    assert isinstance(profile, VoiceProfileManager)
    assert profile.user_id == user_id
    assert profile.metadata == metadata
    assert profile.samples_count == 1
    assert isinstance(profile.created_at, datetime)
    assert profile.last_updated_at >= profile.created_at
    assert profile.features is not None
    assert len(profile.features) > 0
    
    # Test profile retrieval
    retrieved_profile = speaker_recognizer.get_profile(user_id)
    assert retrieved_profile is not None
    assert retrieved_profile.user_id == user_id
    assert retrieved_profile.metadata == metadata

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
    assert all(key in stats for key in ['mean', 'std', 'min', 'max'])
    assert all(isinstance(value, np.ndarray) for value in stats.values())
    # Verify non-empty arrays
    assert all(len(value) > 0 for value in stats.values()) 

def test_verification_scenarios(speaker_recognizer, mock_audio_data):
    """Test different speaker verification scenarios"""
    user_id = "test_user"
    
    # Enroll user
    speaker_recognizer.enroll_user(user_id, mock_audio_data)
    
    # Add required samples
    for _ in range(4):
        speaker_recognizer.add_voice_sample(user_id, mock_audio_data)
    
    # Test 1: Verify with same audio (should pass)
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, mock_audio_data)
    assert is_verified
    assert confidence > 0.9
    
    # Test 2: Verify with slightly modified audio (should pass with lower confidence)
    modified_audio = mock_audio_data * 1.1  # Change amplitude
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, modified_audio)
    assert is_verified
    assert 0.7 <= confidence <= 0.9
    
    # Test 3: Verify with completely different audio (should fail)
    different_audio = np.sin(2 * np.pi * 880 * np.linspace(0, 1, len(mock_audio_data)))
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, different_audio)
    assert not is_verified
    assert confidence < 0.6
    
    # Test 4: Verify with noisy audio
    noise = np.random.normal(0, 0.1, len(mock_audio_data))
    noisy_audio = mock_audio_data + noise
    is_verified, confidence = speaker_recognizer.verify_speaker(user_id, noisy_audio)
    assert is_verified  # Should still verify with noise
    assert confidence > 0.7
    
    # Test 5: Verify non-existent user
    with pytest.raises(ValueError, match="User profile not found"):
        speaker_recognizer.verify_speaker("non_existent_user", mock_audio_data) 