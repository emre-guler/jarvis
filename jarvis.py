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
        self.language = 'tr'  # Default language is Turkish
        
        print("Jarvis başlatılıyor...")
        print("Sistem kontrolleri yapılıyor...")
        
        # Check if running on macOS
        if platform.system() != 'Darwin':
            print("Uyarı: Bu uygulama şu anda sadece macOS'ta tam olarak desteklenmektedir.")
        
        # Check permissions first
        if not self.check_permissions():
            print("İzinler alınamadı. Program sonlandırılıyor...")
            sys.exit(1)
        
        # Initialize voice recognition
        print("Ses tanıma sistemi başlatılıyor...")
        self.voice_recognition = VoiceRecognition()
        
        # Initialize system controls
        print("Sistem kontrolleri başlatılıyor...")
        self.system_controls = SystemControls()
        
        # Whisper model for speech recognition
        print("Whisper modeli yükleniyor...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("Whisper modeli başarıyla yüklendi!")
        except Exception as e:
            print(f"Whisper modeli yüklenirken hata oluştu: {e}")
            sys.exit(1)
        
        # Initialize TTS
        print("TTS modeli yükleniyor...")
        try:
            # Initialize TTS with a different Turkish model
            self.tts_model = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=True)
            # Set speaker and language
            self.tts_model.to("cpu")  # Use CPU for inference
            
            # Test TTS with a simple message
            test_text = "Test mesajı"
            self.tts_model.tts_to_file(text=test_text, file_path="test.wav")
            os.remove("test.wav")
            
            print("TTS modeli başarıyla yüklendi!")
        except Exception as e:
            print(f"TTS modeli yüklenirken hata oluştu: {e}")
            print("Hata detayı:", str(e.__class__.__name__))
            sys.exit(1)
        
        # Voice recognition threshold
        self.voice_threshold = 0.8
        
        # Load environment variables
        load_dotenv()
        
        print("\nJarvis başarıyla başlatıldı!")
        self.speak("Merhaba! Ben Jarvis. Size nasıl yardımcı olabilirim?")
    
    def check_permissions(self):
        """Check and request necessary permissions"""
        try:
            print("\nGerekli izinler kontrol ediliyor...")
            
            # Check microphone permission
            print("Mikrofon izni kontrol ediliyor...")
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✓ Mikrofon izni alındı!")
            except OSError:
                print("✗ Mikrofon izni alınamadı!")
                print("\nLütfen şu adımları takip edin:")
                print("1. Apple menüsü > Sistem Ayarları'nı açın")
                print("2. Gizlilik ve Güvenlik > Mikrofon'a gidin")
                print("3. Uygulamaya mikrofon iznini verin")
                self._open_system_preferences()
                return False
            
            # Check audio output permission
            print("\nSes çıkışı kontrol ediliyor...")
            try:
                sd.check_output_settings()
                print("✓ Ses çıkışı izni alındı!")
            except Exception:
                print("✗ Ses çıkışı izni alınamadı!")
                print("\nLütfen şu adımları takip edin:")
                print("1. Apple menüsü > Sistem Ayarları'nı açın")
                print("2. Ses ayarlarını kontrol edin")
                self._open_system_preferences()
                return False
            
            print("\nTüm izinler başarıyla alındı!")
            return True
            
        except Exception as e:
            print(f"\nBeklenmeyen bir hata oluştu: {e}")
            return False
    
    def _open_system_preferences(self):
        """Helper method to open System Preferences"""
        try:
            subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security"])
            print("\nSistem Ayarları açılıyor...")
            print("İzinleri verdikten sonra programı tekrar başlatın.")
        except Exception as e:
            print(f"Sistem Ayarları açılırken hata oluştu: {e}")
            print("Lütfen Sistem Ayarları'nı manuel olarak açın.")
    
    def create_voice_profile(self):
        """Create a voice profile for the user"""
        print("\nSes profili oluşturma başlatılıyor...")
        print("\nSes profilinizi oluşturmak için:")
        print("1. Lütfen sessiz bir ortamda olduğunuzdan emin olun")
        print("2. Normal bir ses tonuyla Türkçe konuşun")
        print("3. 5 saniye boyunca konuşmanız kaydedilecek")
        
        input("\nHazır olduğunuzda ENTER tuşuna basın...")
        
        audio_samples = []
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nKayıt başlıyor... (5 saniye)")
            
            try:
                audio = self.recognizer.listen(source, timeout=5)
                audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                sample_rate = 16000  # Default sample rate
                audio_samples.append((audio_data, sample_rate))
                
                print("\nSes örneği alındı, işleniyor...")
                success = self.voice_recognition.create_voice_profile(audio_samples)
                
                if success:
                    print("✓ Ses profili başarıyla oluşturuldu!")
                    return True
                else:
                    print("✗ Ses profili oluşturulamadı.")
                    print("Lütfen tekrar deneyin.")
                    return False
                    
            except sr.WaitTimeoutError:
                print("✗ Ses kaydı zaman aşımına uğradı.")
                print("Lütfen tekrar deneyin ve konuştuğunuzdan emin olun.")
                return False
            except Exception as e:
                print(f"✗ Ses profili oluşturulurken hata: {e}")
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
                'kapat': ['bilgisayarı kapat', 'sistemi kapat', 'kapat'],
                'uyku': ['uyku moduna al', 'uyku', 'uyu'],
                'ses': {
                    'yükselt': ['sesi yükselt', 'sesi artır', 'ses yükselt', 'sesi aç'],
                    'azalt': ['sesi azalt', 'sesi düşür', 'ses azalt', 'sesi kıs'],
                    'kapat': ['sesi kapat', 'sessiz'],
                    'aç': ['sesi aç', 'sessizi aç']
                },
                'parlaklık': {
                    'yükselt': ['parlaklığı yükselt', 'parlaklığı artır', 'ekranı parlat'],
                    'azalt': ['parlaklığı azalt', 'parlaklığı düşür', 'ekranı karat']
                },
                'batarya': ['batarya', 'pil', 'pil durumu', 'şarj durumu']
            }
            
            # Check for commands in text
            for command_type, variations in commands.items():
                if isinstance(variations, dict):
                    # Handle nested commands (like ses and parlaklık)
                    for action, sub_variations in variations.items():
                        if any(var in text for var in sub_variations):
                            if command_type == 'ses':
                                if action == 'yükselt':
                                    self.system_controls.control_volume(75)
                                    self.speak("Ses yükseltildi")
                                elif action == 'azalt':
                                    self.system_controls.control_volume(25)
                                    self.speak("Ses azaltıldı")
                                elif action == 'kapat':
                                    self.system_controls.mute_volume()
                                    self.speak("Ses kapatıldı")
                                elif action == 'aç':
                                    self.system_controls.unmute_volume()
                                    self.speak("Ses açıldı")
                            elif command_type == 'parlaklık':
                                if action == 'yükselt':
                                    self.system_controls.control_screen_brightness(100)
                                    self.speak("Parlaklık artırıldı")
                                elif action == 'azalt':
                                    self.system_controls.control_screen_brightness(50)
                                    self.speak("Parlaklık azaltıldı")
                            return
                else:
                    # Handle simple commands
                    if any(var in text for var in variations):
                        if command_type == 'kapat':
                            self.speak("Bilgisayar kapatılıyor...")
                            self.system_controls.shutdown_computer()
                        elif command_type == 'uyku':
                            self.speak("Bilgisayar uyku moduna alınıyor...")
                            self.system_controls.sleep_computer()
                        elif command_type == 'batarya':
                            status = self.system_controls.get_battery_status()
                            self.speak(f"Pil durumu: {status}")
                        return
            
            # Handle application commands
            if "aç" in text:
                app_name = text.split("aç")[-1].strip()
                if app_name and not any(cmd in app_name for cmd in ['ses', 'sessiz']):
                    self.system_controls.open_application(app_name)
                    self.speak(f"{app_name} açılıyor")
                    return
            
            if "kapat" in text and "uygulama" in text:
                app_name = text.split("kapat")[-1].strip()
                if app_name:
                    self.system_controls.close_application(app_name)
                    self.speak(f"{app_name} kapatılıyor")
                    return
            
            # If no command is recognized, give feedback
            self.speak(f"Algılanan komut: {text}")
            
        except Exception as e:
            print(f"Komut işlenirken hata oluştu: {e}")
            self.speak("Üzgünüm, komutu işlerken bir hata oluştu.")
    
    def listen_continuously(self):
        """Continuously listen for audio input"""
        with sr.Microphone() as source:
            print("\nOrtam sesi kalibre ediliyor...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Dinlemeye hazır! 'Jarvis' veya 'Carvis' diyerek beni aktifleştirebilirsiniz.")
            
            # Set dynamic energy threshold
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 2000  # Lower threshold for better sensitivity
            self.recognizer.pause_threshold = 0.5    # Shorter pause threshold
            self.recognizer.phrase_threshold = 0.2   # Lower phrase threshold
            
            while self.is_listening:
                try:
                    print("\nUyandırma kelimesi bekleniyor...")
                    wake_audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=3)  # Increased time limit
                    
                    try:
                        # Try to detect wake word
                        wake_text = self.recognizer.recognize_google(wake_audio, language='tr-TR').lower()
                        wake_words = ["jarvis", "carvis", "cervis", "carviz", "jarvi", "carvi"]  # Added variations
                        if any(word in wake_text for word in wake_words):
                            print("\nSizi dinliyorum...")
                            self.speak("Sizi dinliyorum")
                            
                            # Listen for the actual command
                            command_audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            print("Komut alındı, işleniyor...")
                            self.audio_queue.put(command_audio)
                        
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        continue
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Dinleme sırasında hata: {e}")
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
                        print("\nSes işleniyor...")
                        
                        # Try Google Speech Recognition first
                        try:
                            text = self.recognizer.recognize_google(audio, language='tr-TR')
                            if text.strip():
                                print(f"\nAlgılanan komut: {text}")
                                self.process_command(text)
                                continue
                        except sr.UnknownValueError:
                            print("\nGoogle algılayamadı, Whisper deneniyor...")
                        except sr.RequestError:
                            print("\nGoogle servisine erişilemiyor, Whisper deneniyor...")
                        
                        # If Google fails, try Whisper
                        try:
                            result = self.whisper_model.transcribe(
                                audio_data,
                                language="tr",
                                task="transcribe",
                                fp16=False,
                                initial_prompt="Bu bir sesli komut sistemidir."
                            )
                            text = result["text"].strip()
                            
                            if text and not text.startswith("Bu dizinin betimlemesi"):
                                print(f"\nAlgılanan komut: {text}")
                                self.process_command(text)
                            
                        except Exception as e:
                            print(f"\nWhisper ses işleme hatası: {e}")
                    else:
                        print("\nUyarı: Konuşmacı tanınmadı. Sadece kayıtlı kullanıcı komut verebilir.")
                        
                except Exception as e:
                    print(f"\nSes işleme sırasında hata: {e}")
                    print("Hata detayı:", str(e.__class__.__name__))
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
                print(f"\nSes sentezi sırasında hata: {tts_error}")
                print("Metin çıktısı olarak gösteriliyor...")
                print(f"Metin: {text}")
            
        except Exception as e:
            print(f"\nGenel bir hata oluştu: {e}")
            print("Metin çıktısı:", text)
    
    def start(self):
        """Start Jarvis"""
        try:
            print("\nJarvis başlatılıyor...")
            
            # Create voice profile if it doesn't exist
            if not os.path.exists("voice_profile.pkl"):
                print("\nSes profili bulunamadı. Yeni profil oluşturulacak...")
                if not self.create_voice_profile():
                    print("\nSes profili oluşturulamadı. Program sonlandırılıyor...")
                    return
            
            print("\nSes profili hazır!")
            print("Dinleme ve işleme sistemleri başlatılıyor...")
            
            # Start the listening thread
            listen_thread = threading.Thread(target=self.listen_continuously)
            listen_thread.start()
            
            # Start the processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.start()
            
            print("\nJarvis aktif ve dinliyor!")
            print("Komut vermek için konuşabilirsiniz.")
            print("(Programı sonlandırmak için Ctrl+C tuşlarına basın)")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nProgram sonlandırılıyor...")
                self.is_listening = False
                listen_thread.join()
                process_thread.join()
                print("Program sonlandırıldı. Güle güle!")
                
        except Exception as e:
            print(f"\nProgram başlatılırken hata: {e}")
            self.is_listening = False
            sys.exit(1)

if __name__ == "__main__":
    try:
        jarvis = Jarvis()
        jarvis.start()
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
        sys.exit(0)
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {e}")
        sys.exit(1) 