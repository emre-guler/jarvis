"""
Train the Wake Word Detection Model
"""
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from python_speech_features import mfcc
import wave

from config.settings import AUDIO_SETTINGS, WAKE_WORD_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_audio_file(file_path: Path) -> np.ndarray:
    """Load and preprocess audio file"""
    with wave.open(str(file_path), 'rb') as wf:
        # Read audio data
        audio_data = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to [-1, 1]
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        return audio_np

def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract MFCC features from audio data"""
    features = mfcc(
        audio_data,
        samplerate=sample_rate,
        numcep=13,
        nfilt=40,
        nfft=2048,
        winlen=0.025,
        winstep=0.01,
        preemph=0.97,
        appendEnergy=True
    )
    
    # Normalize features
    features = (features - np.mean(features)) / (np.std(features) + 1e-10)
    
    return features

def create_model(input_shape):
    """Create the CNN model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """Prepare training data from audio samples"""
    logger.info("Preparing training data...")
    
    # Get file paths
    positive_dir = WAKE_WORD_CONFIG["positive_samples_dir"]
    negative_dir = WAKE_WORD_CONFIG["negative_samples_dir"]
    
    positive_files = list(positive_dir.glob("*.wav"))
    negative_files = list(negative_dir.glob("*.wav"))
    
    logger.info(f"Found {len(positive_files)} positive and {len(negative_files)} negative samples")
    
    # Prepare data arrays
    features_list = []
    labels = []
    
    # Calculate target length (2 seconds of audio)
    target_frames = int(2.0 / 0.01)  # 2 seconds with 10ms step size
    
    # Process positive samples
    for file_path in positive_files:
        try:
            audio_data = load_audio_file(file_path)
            features = extract_features(audio_data, AUDIO_SETTINGS["sample_rate"])
            
            # Pad or truncate to target length
            if features.shape[0] > target_frames:
                features = features[:target_frames]
            elif features.shape[0] < target_frames:
                pad_width = ((0, target_frames - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            
            features_list.append(features)
            labels.append(1)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Process negative samples
    for file_path in negative_files:
        try:
            audio_data = load_audio_file(file_path)
            features = extract_features(audio_data, AUDIO_SETTINGS["sample_rate"])
            
            # Pad or truncate to target length
            if features.shape[0] > target_frames:
                features = features[:target_frames]
            elif features.shape[0] < target_frames:
                pad_width = ((0, target_frames - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            
            features_list.append(features)
            labels.append(0)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    # Reshape for CNN input (samples, time, features, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    logger.info(f"Prepared {len(X)} samples with shape {X.shape}")
    
    return X, y

def train_model():
    """Train the wake word detection model"""
    logger.info("Starting model training...")
    
    # Prepare data
    X, y = prepare_data()
    
    # Create and compile model
    model = create_model(input_shape=(X.shape[1], X.shape[2], 1))
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WAKE_WORD_CONFIG["model_path"]),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X, y,
        epochs=WAKE_WORD_CONFIG["training_epochs"],
        batch_size=WAKE_WORD_CONFIG["batch_size"],
        validation_split=WAKE_WORD_CONFIG["validation_split"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(WAKE_WORD_CONFIG["model_path"])
    logger.info(f"Model saved to {WAKE_WORD_CONFIG['model_path']}")
    
    # Print final metrics
    val_accuracy = max(history.history['val_accuracy'])
    logger.info(f"Best validation accuracy: {val_accuracy:.2%}")
    
    return history

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("🧠 Training Wake Word Detection Model")
        print("="*50 + "\n")
        
        history = train_model()
        
        print("\n" + "="*50)
        print("✅ Training Complete!")
        print(f"Model saved to: {WAKE_WORD_CONFIG['model_path']}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise 