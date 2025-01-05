"""
User Enrollment Script for Speaker Recognition

This script handles the collection of voice samples and user enrollment
in the speaker recognition system.
"""
import logging
import wave
import numpy as np
import pyaudio
import time
from pathlib import Path
from typing import Optional

from .speaker_recognizer import SpeakerRecognizer
from config.settings import AUDIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceEnrollment:
    def __init__(self):
        """Initialize voice enrollment system"""
        self.recognizer = SpeakerRecognizer()
        
        # Audio settings
        self.format = pyaudio.paFloat32
        self.channels = AUDIO_SETTINGS["channels"]
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.record_seconds = 3
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def record_sample(self) -> Optional[np.ndarray]:
        """Record a voice sample
        
        Returns:
            Recorded audio data or None if recording failed
        """
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("\n🎤 Recording in 3 seconds...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            print("Recording... Speak now!")
            
            # Record audio
            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * self.record_seconds)):
                data = stream.read(self.chunk_size)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            print("Done recording!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = np.concatenate(frames)
            return audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
        
    def enroll_new_user(self, user_id: str, num_samples: int = 5):
        """Enroll a new user with multiple voice samples
        
        Args:
            user_id: Unique identifier for the user
            num_samples: Number of voice samples to collect
        """
        try:
            print(f"\n📝 Enrolling new user: {user_id}")
            print(f"We'll collect {num_samples} voice samples.")
            print("Please speak naturally and vary your tone slightly.")
            
            # Record first sample and create profile
            print("\n🎤 Recording sample 1...")
            audio_data = self.record_sample()
            if audio_data is None:
                print("❌ Failed to record audio. Please try again.")
                return
            
            # Create profile with first sample
            profile = self.recognizer.enroll_user(user_id, audio_data)
            print("✅ Profile created successfully!")
            
            # Collect additional samples
            for i in range(2, num_samples + 1):
                print(f"\n🎤 Recording sample {i}...")
                audio_data = self.record_sample()
                if audio_data is None:
                    print("❌ Failed to record sample. Skipping...")
                    continue
                
                # Add sample to profile
                success, similarity = self.recognizer.add_voice_sample(user_id, audio_data)
                if success:
                    print(f"✅ Sample added successfully! (similarity: {similarity:.2%})")
                else:
                    print(f"❌ Sample rejected (similarity: {similarity:.2%})")
                    print("Please try again...")
                    i -= 1  # Retry this sample
            
            print(f"\n✅ Enrollment complete for user: {user_id}")
            print(f"Collected {profile.samples_count} voice samples")
            
        except Exception as e:
            logger.error(f"Error during enrollment: {e}")
            print(f"\n❌ Enrollment failed: {str(e)}")
        finally:
            self.audio.terminate()

def main():
    """Main enrollment function"""
    print("\n" + "="*50)
    print("🎤 Speaker Recognition Enrollment")
    print("="*50)
    
    # Get user information
    user_id = input("\nEnter user ID: ").strip()
    if not user_id:
        print("❌ User ID cannot be empty")
        return
    
    # Start enrollment
    enrollor = VoiceEnrollment()
    enrollor.enroll_new_user(user_id)

if __name__ == "__main__":
    main() 