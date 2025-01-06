"""Tests for continuous authentication"""
import pytest
import numpy as np
import time
from unittest.mock import Mock
from src.voice.recognition.continuous_auth import ContinuousAuthenticator

@pytest.fixture
def speaker_recognizer():
    """Mock speaker recognizer"""
    recognizer = Mock()
    recognizer.verify_speaker.return_value = (True, 0.95)
    return recognizer

@pytest.fixture
def authenticator(speaker_recognizer):
    """Create authenticator instance"""
    return ContinuousAuthenticator(speaker_recognizer, verification_interval=0.1)

class TestContinuousAuth:
    def test_initialization(self, authenticator):
        """Test authenticator initialization"""
        assert authenticator.verification_interval == 0.1
        assert not authenticator.is_running
        assert authenticator.current_user is None
        
    def test_start_stop(self, authenticator):
        """Test starting and stopping authentication"""
        # Start authentication
        authenticator.start("test_user")
        assert authenticator.is_running
        assert authenticator.current_user == "test_user"
        
        # Stop authentication
        authenticator.stop()
        assert not authenticator.is_running
        
    def test_audio_processing(self, authenticator):
        """Test audio data processing"""
        authenticator.start("test_user")
        
        # Add audio data
        audio_data = np.random.random(16000)
        authenticator.add_audio(audio_data)
        
        # Wait for processing
        time.sleep(0.2)
        
        # Verify speaker verification was called
        authenticator.speaker_recognizer.verify_speaker.assert_called_once()
        
        authenticator.stop()
        
    def test_auth_failure_callback(self, authenticator, speaker_recognizer):
        """Test authentication failure callback"""
        # Set up mock for failed verification
        speaker_recognizer.verify_speaker.return_value = (False, 0.3)
        
        # Set up callback
        callback_called = False
        def on_failure(confidence):
            nonlocal callback_called
            callback_called = True
            
        # Start authentication
        authenticator.start("test_user", on_auth_failed=on_failure)
        
        # Add audio data
        audio_data = np.random.random(16000)
        authenticator.add_audio(audio_data)
        
        # Wait for processing
        time.sleep(0.2)
        
        # Verify callback was called
        assert callback_called
        
        authenticator.stop()
        
    def test_continuous_verification(self, authenticator):
        """Test continuous verification over time"""
        authenticator.start("test_user")
        
        # Add multiple audio samples
        for _ in range(3):
            audio_data = np.random.random(16000)
            authenticator.add_audio(audio_data)
            time.sleep(0.15)  # Wait for processing
            
        # Verify multiple verifications occurred
        assert authenticator.speaker_recognizer.verify_speaker.call_count == 3
        
        authenticator.stop() 