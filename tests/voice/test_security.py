"""
Tests for Speaker Recognition Security Components
"""
import pytest
import numpy as np
from pathlib import Path
import os
import json
from unittest.mock import Mock

from src.voice.security.encryption import ProfileEncryption
from src.voice.security.anti_spoofing import AntiSpoofing

@pytest.fixture
def encryption():
    """Create a test encryption instance"""
    test_key_file = Path("tests/data/test_encryption.key")
    encryption = ProfileEncryption(key_file=test_key_file)
    yield encryption
    # Cleanup
    if test_key_file.exists():
        test_key_file.unlink()

@pytest.fixture
def anti_spoofing():
    """Create anti-spoofing instance"""
    return AntiSpoofing()

@pytest.fixture
def sample_audio():
    """Create sample audio data"""
    # Generate synthetic audio for testing
    duration = 3  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a clean sine wave
    frequency = 440  # Hz
    audio = np.sin(2 * np.pi * frequency * t)
    return audio

class TestEncryption:
    def test_key_generation(self, encryption):
        """Test encryption key generation"""
        assert encryption.key_file.exists()
        assert encryption.fernet is not None

    def test_encrypt_decrypt(self, encryption):
        """Test encryption and decryption"""
        test_data = b"test profile data"
        
        # Encrypt
        encrypted = encryption.encrypt_profile(test_data)
        assert encrypted != test_data
        
        # Decrypt
        decrypted = encryption.decrypt_profile(encrypted)
        assert decrypted == test_data

    def test_encrypt_large_data(self, encryption):
        """Test encryption of large profile data"""
        large_data = os.urandom(1024 * 1024)  # 1MB of random data
        
        encrypted = encryption.encrypt_profile(large_data)
        decrypted = encryption.decrypt_profile(encrypted)
        
        assert decrypted == large_data

    def test_invalid_decrypt(self, encryption):
        """Test decryption of invalid data"""
        invalid_data = b"invalid encrypted data"
        
        with pytest.raises(Exception):
            encryption.decrypt_profile(invalid_data)

class TestAntiSpoofing:
    def test_genuine_audio(self, anti_spoofing, sample_audio):
        """Test detection of genuine audio"""
        is_genuine, confidence = anti_spoofing.check_audio(sample_audio, 16000)
        assert isinstance(is_genuine, bool)
        assert 0 <= confidence <= 1.0

    def test_energy_check(self, anti_spoofing):
        """Test energy-based spoofing detection"""
        # Test with very low energy audio (potential replay attack)
        low_energy_audio = np.zeros(16000)  # 1 second of silence
        score = anti_spoofing._check_energy(low_energy_audio)
        assert score == 0.0  # Should fail energy check

        # Test with normal energy audio
        normal_audio = np.random.normal(0, 0.5, 16000)
        score = anti_spoofing._check_energy(normal_audio)
        assert score > 0.0  # Should pass energy check

    def test_frequency_check(self, anti_spoofing, sample_audio):
        """Test frequency distribution check"""
        score = anti_spoofing._check_frequency_distribution(sample_audio, 16000)
        assert 0 <= score <= 1.0

    def test_temporal_check(self, anti_spoofing, sample_audio):
        """Test temporal pattern check"""
        score = anti_spoofing._check_temporal_patterns(sample_audio, 16000)
        assert 0 <= score <= 1.0

    def test_replay_attack(self, anti_spoofing):
        """Test detection of replay attacks"""
        # Simulate replay attack with repeated patterns
        t = np.linspace(0, 1, 16000)
        frequency = 440
        clean_audio = np.sin(2 * np.pi * frequency * t)
        replay_audio = np.tile(clean_audio[:800], 20)  # Repeat same pattern
        
        is_genuine, confidence = anti_spoofing.check_audio(replay_audio, 16000)
        assert not is_genuine  # Should detect replay attack
        assert confidence < 0.8  # Should have low confidence

    def test_synthetic_speech(self, anti_spoofing):
        """Test detection of synthetic speech"""
        # Create synthetic speech with perfect sine waves
        t = np.linspace(0, 1, 16000)
        synthetic_audio = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)
        
        is_genuine, confidence = anti_spoofing.check_audio(synthetic_audio, 16000)
        assert confidence < 0.9  # Should have lower confidence for synthetic speech

    def test_noise_handling(self, anti_spoofing):
        """Test handling of noisy audio"""
        # Create noisy audio
        noise = np.random.normal(0, 0.1, 16000)
        t = np.linspace(0, 1, 16000)
        signal = np.sin(2 * np.pi * 440 * t)
        noisy_audio = signal + noise
        
        is_genuine, confidence = anti_spoofing.check_audio(noisy_audio, 16000)
        assert isinstance(is_genuine, bool)
        assert 0 <= confidence <= 1.0 