#!/usr/bin/env python3
"""
Main script to run Jarvis
"""
import logging
import signal
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.voice.recognition.wake_word import WakeWordDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def wake_word_callback(confidence: float):
    """Callback function for wake word detection"""
    logger.info(f"Wake word detected! (confidence: {confidence:.2f})")
    # TODO: Add additional wake word response actions here

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("Shutting down Jarvis...")
    if detector:
        detector.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize wake word detector
        detector = WakeWordDetector()
        
        # Start detection
        logger.info("Starting Jarvis wake word detection...")
        detector.start(wake_word_callback)
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error running Jarvis: {e}")
        if detector:
            detector.stop()
        sys.exit(1) 