import speech_recognition as sr
import whisper
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import json
import os
from datetime import datetime
import threading
import queue
import time
from dotenv import load_dotenv
from voice_recognition import VoiceRecognition
from system_controls import SystemControls
import subprocess
import sys
import platform

class Jarvis:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.conversation_history = []
        self.is_listening = True
        self.language = 'en'  # Default language is English
        
        print("Starting Jarvis...")
        print("Checking system controls...")
        
        # Check if running on macOS
        if platform.system() != 'Darwin':
            print("Warning: This application is currently fully supported only on macOS.")
        
        # Check permissions first
        if not self.check_permissions():
            print("Could not obtain permissions. Terminating program...")
            sys.exit(1)
        
        # Initialize voice recognition
        print("Initializing voice recognition system...")
        self.voice_recognition = VoiceRecognition()
        
        # Initialize system controls
        print("Initializing system controls...")
        self.system_controls = SystemControls()
        
        # Whisper model for speech recognition
        print("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            sys.exit(1)
        
        # Initialize TTS
        print("Loading TTS model...")
        try:
            # Initialize TTS with English model
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True)
            # Set speaker and language
            self.tts_model.to("cpu")  # Use CPU for inference
            
            # Test TTS with a simple message
            test_text = "Test message"
            self.tts_model.tts_to_file(text=test_text, file_path="test.wav")
            os.remove("test.wav")
            
            print("TTS model loaded successfully!")
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            print("Error detail:", str(e.__class__.__name__))
            sys.exit(1)
        
        # Voice recognition threshold
        self.voice_threshold = 0.8
        
        # Load environment variables
        load_dotenv()
        
        print("\nJarvis started successfully!")
        self.speak("Hello! I'm Jarvis. How can I help you?")
    
    def check_permissions(self):
        """Check and request necessary permissions"""
        try:
            print("\nChecking required permissions...")
            
            # Check microphone permission
            print("Checking microphone permission...")
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✓ Microphone permission granted!")
            except OSError:
                print("✗ Could not get microphone permission!")
                print("\nPlease follow these steps:")
                print("1. Open Apple menu > System Settings")
                print("2. Go to Privacy & Security > Microphone")
                print("3. Grant microphone permission to the application")
                self._open_system_preferences()
                return False
            
            # Check audio output permission
            print("\nChecking audio output...")
            try:
                sd.check_output_settings()
                print("✓ Audio output permission granted!")
            except Exception:
                print("✗ Could not get audio output permission!")
                print("\nPlease follow these steps:")
                print("1. Open Apple menu > System Settings")
                print("2. Check Sound settings")
                self._open_system_preferences()
                return False
            
            print("\nAll permissions granted successfully!")
            return True
            
        except Exception as e:
            print(f"\nUnexpected error occurred: {e}")
            return False
    
    def _open_system_preferences(self):
        """Helper method to open System Preferences"""
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security"])
            print("\nOpening System Settings...")
            print("After granting permissions, please restart the program.")
        except Exception as e:
            print(f"Error opening System Settings: {e}")
            print("Please open System Settings manually.")
    
    def create_voice_profile(self):
        """Create a voice profile for the user"""
        print("\nStarting voice profile creation...")
        print("\nTo create your voice profile:")
        print("1. Please ensure you are in a quiet environment")
        print("2. Speak in a normal tone in English")
        print("3. Your voice will be recorded for 5 seconds")
        
        input("\nPress ENTER when ready...")
        
        audio_samples = []
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nRecording starting... (5 seconds)")
            
            try:
                audio = self.recognizer.listen(source, timeout=5)
                audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                sample_rate = 16000  # Default sample rate
                audio_samples.append((audio_data, sample_rate))
                
                print("\nVoice sample captured, processing...")
                success = self.voice_recognition.create_voice_profile(audio_samples)
                
                if success:
                    print("✓ Voice profile created successfully!")
                    return True
                else:
                    print("✗ Could not create voice profile.")
                    print("Please try again.")
                    return False
                    
            except sr.WaitTimeoutError:
                print("✗ Voice recording timed out.")
                print("Please try again and make sure you're speaking.")
                return False
            except Exception as e:
                print(f"✗ Error creating voice profile: {e}")
                return False
    
    def process_command(self, text):
        """Process the recognized text command"""
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": text
        })
        
        # Convert text to lowercase and remove extra spaces
        text = ' '.join(text.lower().split())
        
        try:
            # Command keywords and their variations
            commands = {
                'shutdown': ['shutdown computer', 'shutdown system', 'shutdown'],
                'sleep': ['sleep mode', 'sleep', 'go to sleep'],
                'volume': {
                    'up': ['volume up', 'increase volume', 'turn up volume'],
                    'down': ['volume down', 'decrease volume', 'turn down volume'],
                    'mute': ['mute volume', 'mute'],
                    'unmute': ['unmute volume', 'unmute']
                },
                'brightness': {
                    'up': ['increase brightness', 'brighten screen', 'brightness up'],
                    'down': ['decrease brightness', 'dim screen', 'brightness down']
                },
                'battery': ['battery', 'battery status', 'power status', 'charge status']
            }
            
            # Check for commands in text
            for command_type, variations in commands.items():
                if isinstance(variations, dict):
                    # Handle nested commands (like volume and brightness)
                    for action, sub_variations in variations.items():
                        if any(var in text for var in sub_variations):
                            if command_type == 'volume':
                                if action == 'up':
                                    self.system_controls.control_volume(75)
                                    self.speak("Volume increased")
                                elif action == 'down':
                                    self.system_controls.control_volume(25)
                                    self.speak("Volume decreased")
                                elif action == 'mute':
                                    self.system_controls.mute_volume()
                                    self.speak("Volume muted")
                                elif action == 'unmute':
                                    self.system_controls.unmute_volume()
                                    self.speak("Volume unmuted")
                            elif command_type == 'brightness':
                                if action == 'up':
                                    self.system_controls.control_screen_brightness(100)
                                    self.speak("Brightness increased")
                                elif action == 'down':
                                    self.system_controls.control_screen_brightness(50)
                                    self.speak("Brightness decreased")
                            return
                else:
                    # Handle simple commands
                    if any(var in text for var in variations):
                        if command_type == 'shutdown':
                            self.speak("Shutting down computer...")
                            self.system_controls.shutdown_computer()
                        elif command_type == 'sleep':
                            self.speak("Putting computer to sleep...")
                            self.system_controls.sleep_computer()
                        elif command_type == 'battery':
                            status = self.system_controls.get_battery_status()
                            self.speak(f"Battery status: {status}")
                        return
            
            # Handle application commands
            if "open" in text:
                app_name = text.split("open")[-1].strip()
                if app_name and not any(cmd in app_name for cmd in ['volume', 'mute']):
                    self.system_controls.open_application(app_name)
                    self.speak(f"Opening {app_name}")
                    return
            
            if "shutdown" in text and "application" in text:
                app_name = text.split("shutdown")[-1].strip()
                if app_name:
                    self.system_controls.close_application(app_name)
                    self.speak(f"Closing {app_name}")
                    return
            
            # If no command is recognized, give feedback
            self.speak(f"Recognized command: {text}")
            
        except Exception as e:
            print(f"Error processing command: {e}")
            self.speak("Sorry, I couldn't process that command.")
    
    def listen_continuously(self):
        """Continuously listen for audio input"""
        with sr.Microphone() as source:
            print("\nCalibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ready to listen! Say 'Jarvis' or 'Carvis' to activate me.")
            
            # Set dynamic energy threshold
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 2000  # Lower threshold for better sensitivity
            self.recognizer.pause_threshold = 0.5    # Shorter pause threshold
            self.recognizer.phrase_threshold = 0.2   # Lower phrase threshold
            
            while self.is_listening:
                try:
                    print("\nWaiting for wake word...")
                    wake_audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=3)  # Increased time limit
                    
                    try:
                        # Try to detect wake word
                        wake_text = self.recognizer.recognize_google(wake_audio, language='en-US').lower()
                        wake_words = ["jarvis", "carvis", "cervis", "carviz", "jarvi", "carvi"]  # Added variations
                        if any(word in wake_text for word in wake_words):
                            print("\nListening...")
                            self.speak("Listening")
                            
                            # Listen for the actual command
                            command_audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            print("Command received, processing...")
                            self.audio_queue.put(command_audio)
                        
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        continue
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Error during listening: {e}")
                    continue
    
    def process_audio(self):
        """Process audio from the queue"""
        while self.is_listening:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    # Convert audio to numpy array and normalize to float32
                    audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Verify speaker
                    if self.voice_recognition.verify_speaker(audio_data, 16000):
                        print("\nProcessing audio...")
                        
                        # Try Google Speech Recognition first
                        try:
                            text = self.recognizer.recognize_google(audio, language='en-US')
                            if text.strip():
                                print(f"\nRecognized command: {text}")
                                self.process_command(text)
                                continue
                        except sr.UnknownValueError:
                            print("\nGoogle Speech Recognition failed, trying Whisper...")
                        except sr.RequestError:
                            print("\nCould not connect to Google Speech Recognition, trying Whisper...")
                        
                        # If Google fails, try Whisper
                        try:
                            result = self.whisper_model.transcribe(
                                audio_data,
                                language="en",
                                task="transcribe",
                                fp16=False,
                                initial_prompt="This is a voice command system."
                            )
                            text = result["text"].strip()
                            
                            if text and not text.startswith("This is a description of the directory"):
                                print(f"\nRecognized command: {text}")
                                self.process_command(text)
                            
                        except Exception as e:
                            print(f"\nError processing Whisper: {e}")
                    else:
                        print("\nWarning: Speaker not recognized. Only authorized users can issue commands.")
                        
                except Exception as e:
                    print(f"\nError processing audio: {e}")
                    print("Error detail:", str(e.__class__.__name__))
            else:
                time.sleep(0.1)
    
    def speak(self, text):
        """Convert text to speech using TTS"""
        try:
            print(f"\nJarvis: {text}")
            
            # Convert text to lowercase for better TTS performance
            text = text.lower()
            
            # Replace problematic characters
            char_map = {
                'İ': 'i', 'I': 'ı', 'Ğ': 'ğ', 'Ü': 'ü', 'Ş': 'ş', 'Ö': 'ö', 'Ç': 'ç',
                '%': ' yüzde ', '&': ' ve ', '+': ' artı ', '=': ' eşittir ',
                '_': ' ', '|': ' ', '/': ' bölü ', '\\': ' ',
                '@': ' et ', '#': ' numara ', '$': ' dolar '
            }
            
            for k, v in char_map.items():
                text = text.replace(k, v)
            
            # Generate speech with error handling
            try:
                # Convert text to speech
                wav = self.tts_model.tts(text=text)
                
                # Save and play audio
                import soundfile as sf
                sf.write("response.wav", wav, self.tts_model.synthesizer.output_sample_rate)
                
                # Play the audio
                data, samplerate = sf.read("response.wav")
                sd.play(data, samplerate)
                sd.wait()
                
                # Clean up
                os.remove("response.wav")
                
            except Exception as tts_error:
                print(f"\nError synthesizing speech: {tts_error}")
                print("Text will be displayed instead...")
                print(f"Text: {text}")
            
        except Exception as e:
            print(f"\nGeneral error: {e}")
            print("Text will be displayed instead...")
            print(f"Text: {text}")
    
    def start(self):
        """Start Jarvis"""
        try:
            print("\nStarting Jarvis...")
            
            # Create voice profile if it doesn't exist
            if not os.path.exists("voice_profile.pkl"):
                print("\nVoice profile not found. Creating new profile...")
                if not self.create_voice_profile():
                    print("\nCould not create voice profile. Terminating program...")
                    return
            
            print("\nVoice profile loaded successfully!")
            print("Starting listening and processing threads...")
            
            # Start the listening thread
            listen_thread = threading.Thread(target=self.listen_continuously)
            listen_thread.start()
            
            # Start the processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.start()
            
            print("\nJarvis active and listening!")
            print("Speak to give commands or type 'exit' to quit.")
            print("(Press Ctrl+C to terminate the program)")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nTerminating program...")
                self.is_listening = False
                listen_thread.join()
                process_thread.join()
                print("Program terminated. Goodbye!")
                
        except Exception as e:
            print(f"\nError starting Jarvis: {e}")
            self.is_listening = False
            sys.exit(1)

if __name__ == "__main__":
    try:
        jarvis = Jarvis()
        jarvis.start()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1) 