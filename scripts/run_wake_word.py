#!/usr/bin/env python3
"""
Run the wake word detection system
"""
import logging
import signal
import sys
import time
from pathlib import Path

from src.voice.recognition.wake_word import WakeWordDetector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def on_wake_word(confidence: float):
    """Callback function when wake word is detected"""
    logger.info(f"🎯 Wake word detected with confidence: {confidence:.2f}")
    # Add a visual indicator
    print("\n" + "="*50)
    print(f"🎤 Wake Word 'Jarvis' Detected! (confidence: {confidence:.2f})")
    print("="*50 + "\n")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("Stopping wake word detection...")
    if detector:
        detector.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    detector = None
    try:
        # Initialize wake word detector
        model_path = Path("models/wake_word_model.keras")
        detector = WakeWordDetector(model_path=model_path)
        
        # Start detection
        detector.start(callback=on_wake_word)
        
        print("\n" + "="*50)
        print("🎤 Wake Word Detection Active")
        print("Say 'Jarvis' to trigger")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error running wake word detection: {e}")
        if detector:
            detector.stop()
        sys.exit(1) 