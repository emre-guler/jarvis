"""Emotion recognition from voice"""
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple

import librosa
from sklearn.preprocessing import StandardScaler

from config.settings import AUDIO_SETTINGS, MODEL_CONFIG

logger = logging.getLogger(__name__)

class EmotionRecognizer:
    def __init__(self):
        """Initialize emotion recognition system"""
        self.emotions = {
            0: "neutral",
            1: "happy",
            2: "sad",
            3: "angry",
            4: "fearful",
            5: "surprised",
            6: "disgusted"
        }
        
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        
    def _create_model(self) -> tf.keras.Model:
        """Create emotion recognition model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(220,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def extract_emotion_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract emotion-relevant features from audio
        
        Args:
            audio_data: Raw audio signal
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Extract features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=40
            )
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate
            )
            mel = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate
            )
            
            # Add pitch and energy features
            pitch = librosa.yin(audio_data, fmin=50, fmax=500)
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Compute statistics
            features = []
            for feat in [mfcc, chroma, mel]:
                features.extend([
                    np.mean(feat, axis=1),
                    np.std(feat, axis=1),
                    np.max(feat, axis=1),
                    np.min(feat, axis=1)
                ])
                
            # Add pitch and energy statistics
            for feat in [pitch, energy]:
                features.extend([
                    np.mean(feat),
                    np.std(feat),
                    np.max(feat),
                    np.min(feat)
                ])
                
            # Flatten and concatenate
            features = np.concatenate([f.flatten() for f in features])
            
            # Ensure fixed size (220 features)
            if len(features) > 220:
                features = features[:220]
            elif len(features) < 220:
                features = np.pad(features, (0, 220 - len(features)))
                
            # Normalize
            features = self.scaler.fit_transform(features.reshape(1, -1))[0]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting emotion features: {e}")
            return np.zeros(220)  # Return zero features on error
            
    def detect_emotion(self, audio_data: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Detect emotion from audio data
        
        Args:
            audio_data: Raw audio signal
            
        Returns:
            Tuple[str, Dict[str, float]]: Detected emotion and confidence scores
        """
        try:
            # Extract features
            features = self.extract_emotion_features(audio_data)
            
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1), verbose=1)
            
            # Apply softmax with temperature
            temperature = 0.5  # Lower temperature = more certainty
            scaled_prediction = prediction[0] / temperature
            probabilities = tf.nn.softmax(scaled_prediction).numpy()
            
            # Get emotion label and probabilities
            emotion_idx = np.argmax(probabilities)
            emotion = self.emotions[emotion_idx]
            
            # Calculate confidence scores
            confidence_scores = {
                self.emotions[i]: float(score) 
                for i, score in enumerate(probabilities)
            }
            
            # Adjust confidence based on audio characteristics
            energy = np.mean(np.abs(audio_data))
            pitch = librosa.yin(audio_data, fmin=50, fmax=500)
            pitch_mean = np.nanmean(pitch)
            
            # High energy suggests active emotions (angry, happy, surprised)
            if energy > 0.5:
                for e in ["angry", "happy", "surprised"]:
                    confidence_scores[e] *= 1.5
                    
            # High pitch suggests happy/surprised, low pitch suggests sad/angry
            if pitch_mean > 200:  # High pitch
                confidence_scores["happy"] *= 1.3
                confidence_scores["surprised"] *= 1.3
            elif pitch_mean < 100:  # Low pitch
                confidence_scores["sad"] *= 1.3
                confidence_scores["angry"] *= 1.3
                
            # Normalize scores
            total = sum(confidence_scores.values())
            confidence_scores = {k: v/total for k, v in confidence_scores.items()}
            
            # Get highest confidence emotion
            emotion = max(confidence_scores.items(), key=lambda x: x[1])[0]
            
            return emotion, confidence_scores
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral", {e: 1.0/7.0 for e in self.emotions.values()} 