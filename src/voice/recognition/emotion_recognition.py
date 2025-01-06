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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
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
            
            # Reshape for model input
            features = features.reshape(1, -1)
            
            # Get predictions
            predictions = self.model.predict(features, verbose=0)[0]
            
            # Apply softmax temperature scaling for better calibration
            temperature = 0.5  # Lower temperature for sharper probabilities
            scaled_predictions = tf.nn.softmax(predictions / temperature).numpy()
            
            # Adjust probabilities based on audio characteristics
            energy = np.mean(np.abs(audio_data))
            pitch = librosa.yin(audio_data, fmin=50, fmax=500)
            pitch_mean = np.nanmean(pitch)
            
            # Convert to probabilities dict
            probabilities = {
                emotion: float(prob)
                for emotion, prob in zip(self.emotions.values(), scaled_predictions)
            }
            
            # Adjust probabilities based on audio characteristics
            if energy > 0.5:  # High energy
                probabilities["angry"] *= 1.5
                probabilities["happy"] *= 1.5
                probabilities["surprised"] *= 1.5
            else:  # Low energy
                probabilities["sad"] *= 1.5
                probabilities["neutral"] *= 1.5
                
            if pitch_mean > 200:  # High pitch
                probabilities["happy"] *= 1.3
                probabilities["surprised"] *= 1.3
            elif pitch_mean < 100:  # Low pitch
                probabilities["sad"] *= 1.3
                probabilities["angry"] *= 1.3
                
            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
            # Get emotion with highest probability
            max_emotion = max(probabilities.items(), key=lambda x: x[1])
            
            return max_emotion[0], probabilities
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral", {emotion: 1.0/len(self.emotions) for emotion in self.emotions.values()} 