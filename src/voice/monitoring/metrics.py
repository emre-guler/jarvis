"""
Performance Monitoring System for Wake Word Detection
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionEvent:
    """Represents a single wake word detection event"""
    timestamp: float
    confidence: float
    detection_time: float  # Time taken to process
    energy_level: float
    cpu_usage: float
    memory_usage: float
    is_true_positive: Optional[bool] = None  # To be set by user feedback

class PerformanceMonitor:
    """Monitors and collects performance metrics for wake word detection"""

    def __init__(self, metrics_dir: str = "data/metrics"):
        """Initialize the performance monitor
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.detection_events: List[DetectionEvent] = []
        self.cpu_usage_samples: List[float] = []
        self.memory_usage_samples: List[float] = []
        self.latency_samples: List[float] = []
        
        # Performance thresholds from RFC
        self.target_latency = 0.5  # 500ms
        self.target_cpu_idle = 5.0  # 5% in standby
        self.target_accuracy = 0.98  # 98%
        
        # Start time for this monitoring session
        self.start_time = time.time()
        
        # Initialize with first measurements
        try:
            process = psutil.Process()
            # Get initial CPU usage
            initial_cpu = process.cpu_percent(interval=0.1)
            normalized_cpu = initial_cpu / psutil.cpu_count()
            self.cpu_usage_samples.append(normalized_cpu)
            
            # Get initial memory usage
            initial_memory = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_samples.append(initial_memory)
            
            logger.info(f"Initialized monitoring - Initial CPU: {normalized_cpu:.1f}%, Memory: {initial_memory:.1f}MB")
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
            self.cpu_usage_samples.append(0.0)
            self.memory_usage_samples.append(0.0)
        
    def record_detection(self, confidence: float, detection_time: float, energy_level: float = None):
        """Record a wake word detection event"""
        event = DetectionEvent(
            timestamp=time.time(),
            confidence=confidence,
            detection_time=detection_time,
            energy_level=energy_level if energy_level is not None else -1,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024  # MB
        )
        self.detection_events.append(event)
        self.latency_samples.append(detection_time)
        logger.debug(f"Recorded detection event: confidence={confidence:.2f}, latency={detection_time*1000:.1f}ms")
        
    def record_system_metrics(self):
        """Record system resource usage"""
        try:
            # Get current process
            process = psutil.Process()
            
            # Get CPU usage with shorter interval for accuracy
            cpu_percent = process.cpu_percent(interval=0.1)  # Reduced from 1.0
            
            # Normalize by CPU count and add to samples
            normalized_cpu = cpu_percent / psutil.cpu_count()
            self.cpu_usage_samples.append(normalized_cpu)
            
            # Get memory usage in MB (RSS - Resident Set Size)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_samples.append(memory_mb)
            
            # Keep only last 10 minutes of samples (600 samples at 1 per second)
            max_samples = 600  # Reduced from 3600
            if len(self.cpu_usage_samples) > max_samples:
                self.cpu_usage_samples = self.cpu_usage_samples[-max_samples:]
            if len(self.memory_usage_samples) > max_samples:
                self.memory_usage_samples = self.memory_usage_samples[-max_samples:]
            
            logger.info(f"System metrics - CPU: {normalized_cpu:.1f}%, Memory: {memory_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
            # Keep last known values if available
            if self.cpu_usage_samples:
                self.cpu_usage_samples.append(self.cpu_usage_samples[-1])
            else:
                self.cpu_usage_samples.append(0.0)
            if self.memory_usage_samples:
                self.memory_usage_samples.append(self.memory_usage_samples[-1])
            else:
                self.memory_usage_samples.append(0.0)
        
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        # Convert numpy values to Python native types
        def to_native(value):
            if isinstance(value, np.generic):
                return value.item()
            return value
        
        metrics = {
            "latency": {
                "current": to_native(np.mean(self.latency_samples[-100:]) if self.latency_samples else 0),
                "min": to_native(min(self.latency_samples) if self.latency_samples else 0),
                "max": to_native(max(self.latency_samples) if self.latency_samples else 0),
                "avg": to_native(np.mean(self.latency_samples) if self.latency_samples else 0),
                "meets_target": to_native((np.mean(self.latency_samples) if self.latency_samples else 1) < self.target_latency)
            },
            "cpu_usage": {
                "current": to_native(np.mean(self.cpu_usage_samples[-100:]) if self.cpu_usage_samples else 0),
                "min": to_native(min(self.cpu_usage_samples) if self.cpu_usage_samples else 0),
                "max": to_native(max(self.cpu_usage_samples) if self.cpu_usage_samples else 0),
                "avg": to_native(np.mean(self.cpu_usage_samples) if self.cpu_usage_samples else 0),
                "meets_target": to_native((np.mean(self.cpu_usage_samples) if self.cpu_usage_samples else 100) < self.target_cpu_idle)
            },
            "memory_usage": {
                "current": to_native(self.memory_usage_samples[-1] if self.memory_usage_samples else 0),
                "min": to_native(min(self.memory_usage_samples) if self.memory_usage_samples else 0),
                "max": to_native(max(self.memory_usage_samples) if self.memory_usage_samples else 0),
                "avg": to_native(np.mean(self.memory_usage_samples) if self.memory_usage_samples else 0)
            },
            "detections": {
                "total": len(self.detection_events),
                "true_positives": sum(1 for e in self.detection_events if e.is_true_positive is True),
                "false_positives": sum(1 for e in self.detection_events if e.is_true_positive is False),
                "unverified": sum(1 for e in self.detection_events if e.is_true_positive is None)
            }
        }
        
        # Calculate accuracy if we have verified detections
        verified_detections = [e for e in self.detection_events if e.is_true_positive is not None]
        if verified_detections:
            true_positives = sum(1 for e in verified_detections if e.is_true_positive)
            accuracy = true_positives / len(verified_detections)
            metrics["accuracy"] = {
                "value": to_native(accuracy),
                "meets_target": to_native(accuracy >= self.target_accuracy)
            }
            
        return metrics
        
    def save_metrics(self):
        """Save current metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"
        
        try:
            # Get current metrics
            metrics = self.get_current_metrics()
            
            # Add additional metadata
            metrics["timestamp"] = timestamp
            metrics["duration"] = time.time() - self.start_time
            metrics["status"] = "complete"
            
            # Write to temporary file first
            temp_file = self.metrics_dir / f"temp_{timestamp}.json"
            with open(temp_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                f.write('\n')  # Add newline at end of file
            
            # Rename temporary file to final filename
            temp_file.rename(metrics_file)
            
            logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            # If temp file exists, try to clean it up
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
    def log_current_metrics(self):
        """Log current performance metrics"""
        metrics = self.get_current_metrics()
        
        logger.info("=== Performance Metrics ===")
        logger.info(f"Latency: {metrics['latency']['current']*1000:.1f}ms (target: <{self.target_latency*1000}ms)")
        logger.info(f"CPU Usage: {metrics['cpu_usage']['current']:.1f}% (target: <{self.target_cpu_idle}%)")
        logger.info(f"Memory Usage: {metrics['memory_usage']['current']:.1f}MB")
        logger.info(f"Total Detections: {metrics['detections']['total']}")
        
        if 'accuracy' in metrics:
            logger.info(f"Accuracy: {metrics['accuracy']['value']:.1%} (target: {self.target_accuracy:.1%})")
            
    def mark_detection_feedback(self, timestamp: float, was_correct: bool):
        """Mark a detection event as true/false positive based on user feedback"""
        # Find the closest detection event to the given timestamp
        if self.detection_events:
            closest_event = min(self.detection_events, key=lambda e: abs(e.timestamp - timestamp))
            if abs(closest_event.timestamp - timestamp) < 5:  # Within 5 seconds
                closest_event.is_true_positive = was_correct
                logger.debug(f"Marked detection at {timestamp} as {'correct' if was_correct else 'incorrect'}")
            else:
                logger.warning("No matching detection event found within 5 seconds") 