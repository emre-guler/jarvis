"""
Continuous Authentication Module

This module provides continuous speaker verification during ongoing interactions.
"""
import numpy as np
import threading
import time
from typing import Optional, Callable
from queue import Queue

class ContinuousAuthenticator:
    def __init__(self, speaker_recognizer, verification_interval: float = 5.0):
        """
        Initialize continuous authentication.
        
        Args:
            speaker_recognizer: Instance of SpeakerRecognition
            verification_interval: Time between verifications in seconds
        """
        self.speaker_recognizer = speaker_recognizer
        self.verification_interval = verification_interval
        self.audio_queue = Queue()
        self.is_running = False
        self.current_user: Optional[str] = None
        self.on_auth_failed: Optional[Callable] = None
        
    def start(self, user_id: str, on_auth_failed: Callable = None):
        """
        Start continuous authentication for a user.
        
        Args:
            user_id: ID of the user to authenticate
            on_auth_failed: Callback function when authentication fails
        """
        self.current_user = user_id
        self.on_auth_failed = on_auth_failed
        self.is_running = True
        
        # Start authentication thread
        self.auth_thread = threading.Thread(target=self._auth_loop)
        self.auth_thread.daemon = True
        self.auth_thread.start()
        
    def stop(self):
        """Stop continuous authentication"""
        self.is_running = False
        if hasattr(self, 'auth_thread'):
            self.auth_thread.join()
            
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data for verification"""
        self.audio_queue.put(audio_data)
        
    def _auth_loop(self):
        """Main authentication loop"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                
                # Verify speaker
                is_verified, confidence = self.speaker_recognizer.verify_speaker(
                    self.current_user,
                    audio_data
                )
                
                # Handle failed verification
                if not is_verified and self.on_auth_failed:
                    self.on_auth_failed(confidence)
                    
            time.sleep(0.1)  # Prevent busy waiting 