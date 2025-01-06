"""Tests for emotion recognition"""
import pytest
import numpy as np
from src.voice.recognition.emotion_recognition import EmotionRecognizer

@pytest.fixture
def emotion_recognizer():
    """Create emotion recognizer instance"""
    return EmotionRecognizer()

@pytest.fixture
def sample_audio():
    """Create sample audio data"""
    # Generate a simple sine wave
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # Hz
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * frequency * t)

class TestEmotionRecognition:
    def test_initialization(self, emotion_recognizer):
        """Test emotion recognizer initialization"""
        assert emotion_recognizer.model is not None
        assert len(emotion_recognizer.emotions) == 7
        assert all(emotion in emotion_recognizer.emotions.values() for emotion in [
            "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"
        ])
        
    def test_feature_extraction(self, emotion_recognizer, sample_audio):
        """Test emotion feature extraction"""
        features = emotion_recognizer.extract_emotion_features(sample_audio)
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert features.shape[0] > 0
        
    def test_emotion_detection(self, emotion_recognizer, sample_audio):
        """Test emotion detection"""
        emotion, probabilities = emotion_recognizer.detect_emotion(sample_audio)
        
        # Check emotion
        assert emotion in emotion_recognizer.emotions.values()
        
        # Check probabilities
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 7
        assert all(0 <= prob <= 1 for prob in probabilities.values())
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6
        
    def test_different_emotions(self, emotion_recognizer):
        """Test detection of different emotions"""
        # Create audio samples with different characteristics
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test with different frequencies and amplitudes
        test_cases = [
            (440, 0.5),  # Normal speech-like
            (880, 0.8),  # Higher pitch
            (220, 0.3),  # Lower pitch
        ]
        
        emotions_detected = set()
        for freq, amp in test_cases:
            audio = amp * np.sin(2 * np.pi * freq * t)
            emotion, _ = emotion_recognizer.detect_emotion(audio)
            emotions_detected.add(emotion)
            
        # Should detect at least 2 different emotions
        assert len(emotions_detected) >= 2
        
    def test_confidence_scores(self, emotion_recognizer, sample_audio):
        """Test emotion confidence scores"""
        _, probabilities = emotion_recognizer.detect_emotion(sample_audio)
        
        # Get highest confidence emotion
        max_prob = max(probabilities.values())
        
        # Should have clear confidence in prediction
        assert max_prob > 0.3  # At least 30% confidence
        
        # Should have some uncertainty
        assert len([p for p in probabilities.values() if p > 0.1]) > 1 