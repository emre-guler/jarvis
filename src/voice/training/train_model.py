import os
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from python_speech_features import mfcc
import librosa
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordModelTrainer:
    def __init__(self, data_dir: str = "data/wake_word_samples", model_dir: str = "models"):
        """Initialize the model trainer
        
        Args:
            data_dir: Directory containing positive and negative samples
            model_dir: Directory to save the trained model
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio processing settings
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_mel = 40
        self.window_size = 0.025
        self.hop_size = 0.01
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from audio samples"""
        logger.info("Preparing training data...")
        
        X = []  # Features
        y = []  # Labels
        
        # Process positive samples
        positive_dir = self.data_dir / "positive"
        for audio_file in positive_dir.glob("*.wav"):
            features = self._extract_features(audio_file)
            if features is not None:
                X.append(features)
                y.append(1)
                
        # Process negative samples
        negative_dir = self.data_dir / "negative"
        for audio_file in negative_dir.glob("*.wav"):
            features = self._extract_features(audio_file)
            if features is not None:
                X.append(features)
                y.append(0)
                
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared {len(y)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
        return X, y
        
    def _extract_features(self, audio_file: Path) -> np.ndarray:
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            audio, _ = librosa.load(audio_file, sr=self.sample_rate)
            
            # Extract MFCC features
            features = mfcc(
                audio,
                samplerate=self.sample_rate,
                numcep=self.n_mfcc,
                nfilt=self.n_mel,
                winlen=self.window_size,
                winstep=self.hop_size
            )
            
            # Normalize features
            features = (features - np.mean(features)) / np.std(features)
            
            # Reshape for CNN input (add channel dimension)
            features = features.reshape(features.shape[0], features.shape[1], 1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            return None
            
    def create_model(self) -> tf.keras.Model:
        """Create the wake word detection model"""
        model = tf.keras.Sequential([
            # CNN layers
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(self.n_mfcc, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Train the wake word detection model"""
        # Prepare data
        X, y = self.prepare_data()
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Create and compile model
        model = self.create_model()
        logger.info(model.summary())
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_dir / 'wake_word_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Save final model
        model.save(self.model_dir / 'wake_word_model_final.h5')
        logger.info(f"Model saved to {self.model_dir}")
        
        return history

if __name__ == "__main__":
    trainer = WakeWordModelTrainer()
    trainer.train() 