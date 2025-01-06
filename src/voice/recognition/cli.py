"""
Command Line Interface for Speaker Recognition System

This module provides a CLI for testing and using the speaker recognition system.
"""
import logging
import click
import numpy as np
import time
import pyaudio
from pathlib import Path

from .speaker_recognizer import SpeakerRecognizer
from .enroll_user import VoiceEnrollment
from config.settings import AUDIO_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self):
        """Initialize audio recorder"""
        self.format = pyaudio.paFloat32
        self.channels = AUDIO_SETTINGS["channels"]
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.chunk_size = AUDIO_SETTINGS["chunk_size"]
        self.audio = pyaudio.PyAudio()
        
    def record(self, duration: float = 3.0) -> np.ndarray:
        """Record audio for specified duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio data
        """
        try:
            # Find the best input device
            device_index = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    device_index = i
                    logger.info(f"Using input device: {device_info['name']}")
                    break

            if device_index is None:
                raise RuntimeError("No input device found")

            # Open stream with more robust settings
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                start=False  # Don't start immediately
            )
            
            print("\n🎤 Recording in 3 seconds...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            
            # Start the stream
            stream.start_stream()
            print("Recording... Speak now!")
            
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            # Record audio with overflow handling
            for _ in range(num_chunks):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                except Exception as e:
                    logger.warning(f"Dropped frame: {e}")
                    continue
            
            print("Done recording!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            if not frames:
                raise RuntimeError("No audio frames recorded")
            
            # Convert to numpy array
            audio_data = np.concatenate(frames)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise
        
    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()

@click.group()
def cli():
    """Speaker Recognition System CLI"""
    pass

@cli.command()
@click.argument('user_id')
@click.option('--samples', default=5, help='Number of voice samples to collect')
def enroll(user_id: str, samples: int):
    """Enroll a new user in the system"""
    enrollor = VoiceEnrollment()
    enrollor.enroll_new_user(user_id, samples)

@cli.command()
@click.argument('user_id')
def verify(user_id: str):
    """Verify a speaker's identity"""
    try:
        recognizer = SpeakerRecognizer()
        recorder = AudioRecorder()
        
        # Record test sample
        audio_data = recorder.record()
        
        # Verify speaker
        is_verified, confidence = recognizer.verify_speaker(user_id, audio_data)
        
        if is_verified:
            print(f"\n✅ Speaker verified as {user_id}!")
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"\n❌ Speaker verification failed!")
            print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        print(f"\n❌ Verification failed: {str(e)}")
    finally:
        recorder.cleanup()

@cli.command()
@click.argument('user_id')
def add_samples(user_id: str):
    """Add more voice samples to an existing profile"""
    try:
        recognizer = SpeakerRecognizer()
        recorder = AudioRecorder()
        
        while True:
            # Record sample
            audio_data = recorder.record()
            
            # Add sample
            success, similarity = recognizer.add_voice_sample(user_id, audio_data)
            
            if success:
                print(f"\n✅ Sample added successfully!")
                print(f"Similarity: {similarity:.2%}")
            else:
                print(f"\n❌ Sample rejected!")
                print(f"Similarity: {similarity:.2%}")
            
            # Ask to continue
            if not click.confirm("\nAdd another sample?"):
                break
                
    except Exception as e:
        logger.error(f"Error adding samples: {e}")
        print(f"\n❌ Failed to add samples: {str(e)}")
    finally:
        recorder.cleanup()

@cli.command()
@click.argument('user_id')
def delete(user_id: str):
    """Delete a user's voice profile"""
    try:
        if click.confirm(f"Are you sure you want to delete profile for {user_id}?"):
            recognizer = SpeakerRecognizer()
            recognizer.delete_user(user_id)
            print(f"\n✅ Profile deleted for user: {user_id}")
    except Exception as e:
        logger.error(f"Error deleting profile: {e}")
        print(f"\n❌ Failed to delete profile: {str(e)}")

@cli.command()
def list_users():
    """List all enrolled users"""
    try:
        recognizer = SpeakerRecognizer()
        profiles_dir = Path(recognizer.profile_manager.profiles_dir)
        
        print("\n📋 Enrolled Users:")
        for profile_file in profiles_dir.glob("*.json"):
            user_id = profile_file.stem
            profile = recognizer.profile_manager.get_profile(user_id)
            if profile:
                print(f"\n- User ID: {user_id}")
                print(f"  Samples: {profile.samples_count}")
                print(f"  Created: {profile.created_at}")
                print(f"  Updated: {profile.updated_at}")
                if profile.metadata:
                    print("  Metadata:")
                    for key, value in profile.metadata.items():
                        print(f"    {key}: {value}")
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        print(f"\n❌ Failed to list users: {str(e)}")

if __name__ == "__main__":
    cli() 