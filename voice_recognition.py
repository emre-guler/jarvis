import numpy as np
from scipy.io import wavfile
import librosa
import pickle
import os

class VoiceRecognition:
    def __init__(self, model_path="voice_profile.pkl"
    ):
        self.model_path = model_path
        self.voice_profile = None
        self.load_voice_profile()
        self.threshold = 0.6
    
    def extract_features(self, audio_data, sample_rate):
        """Extract MFCC features from audio"""
        try:
            # Normalize audio data
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate,
                                       n_mfcc=13,
                                       hop_length=512,
                                       n_fft=2048)
            
            # Add delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            combined_features = np.concatenate([
                np.mean(mfccs.T, axis=0),
                np.mean(delta_mfccs.T, axis=0),
                np.mean(delta2_mfccs.T, axis=0)
            ])
            
            # Normalize features
            combined_features = (combined_features - np.mean(combined_features)) / np.std(combined_features)
            
            return combined_features
            
        except Exception as e:
            print(f"Özellik çıkarma sırasında hata: {e}")
            return None
    
    def create_voice_profile(self, audio_samples):
        """Create a voice profile from multiple audio samples"""
        try:
            features_list = []
            for audio_data, sample_rate in audio_samples:
                features = self.extract_features(audio_data, sample_rate)
                if features is not None:
                    features_list.append(features)
            
            if features_list:
                self.voice_profile = np.mean(features_list, axis=0)
                self.save_voice_profile()
                return True
                
            return False
            
        except Exception as e:
            print(f"Ses profili oluşturma sırasında hata: {e}")
            return False
    
    def save_voice_profile(self):
        """Save voice profile to file"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.voice_profile, f)
            print(f"Ses profili kaydedildi: {self.model_path}")
        except Exception as e:
            print(f"Ses profili kaydedilirken hata: {e}")
    
    def load_voice_profile(self):
        """Load voice profile from file"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.voice_profile = pickle.load(f)
                print(f"Ses profili yüklendi: {self.model_path}")
            except Exception as e:
                print(f"Ses profili yüklenirken hata: {e}")
    
    def verify_speaker(self, audio_data, sample_rate, threshold=None):
        """Verify if the audio matches the voice profile"""
        if self.voice_profile is None:
            print("Ses profili bulunamadı!")
            return False
        
        # Use instance threshold if none provided
        threshold = threshold if threshold is not None else self.threshold
        
        try:
            # Extract features from the new audio
            features = self.extract_features(audio_data, sample_rate)
            if features is None:
                return False
            
            # Calculate cosine similarity
            similarity = np.dot(features, self.voice_profile) / (
                np.linalg.norm(features) * np.linalg.norm(self.voice_profile)
            )
            
            # Print similarity score for debugging
            print(f"Benzerlik skoru: {similarity:.2f}")
            
            return similarity > threshold
            
        except Exception as e:
            print(f"Ses doğrulama sırasında hata: {e}")
            return False 