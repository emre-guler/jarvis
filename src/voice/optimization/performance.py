"""Performance optimization for speaker recognition"""
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self):
        """Initialize performance optimization system"""
        # Performance metrics
        self.verification_times: deque = deque(maxlen=100)
        self.verification_results: deque = deque(maxlen=100)
        self.similarity_scores: deque = deque(maxlen=100)
        self.processing_times: deque = deque(maxlen=100)
        
        # System metrics
        self.cpu_usage: deque = deque(maxlen=100)
        self.memory_usage: deque = deque(maxlen=100)
        
        # Initialize performance stats
        self.performance_stats = {
            "verification_times": [],
            "verification_results": [],
            "similarity_scores": [],
            "processing_times": [],
            "cpu_usage": [],
            "memory_usage": [],
            "accuracy_scores": []  # Add accuracy scores
        }
        
        # Thresholds
        self.max_verification_time = 1.0  # seconds
        self.min_verification_rate = 0.95  # 95% success rate
        self.max_cpu_usage = 80.0  # percentage
        
        # Optimization state
        self.is_monitoring = False
        self.last_optimization = time.time()
        self.optimization_interval = 300  # 5 minutes
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        logger.info("Started performance monitoring")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        logger.info("Stopped performance monitoring")
        
    def record_verification(self, duration: float, success: bool, similarity: Optional[float] = None):
        """Record verification metrics
        
        Args:
            duration: Time taken for verification
            success: Whether verification was successful
            similarity: Optional similarity score (used as confidence)
        """
        if duration > self.max_verification_time:
            logger.warning(f"Verification time ({duration:.2f}s) exceeded threshold")
            
        self.verification_times.append(duration)
        self.verification_results.append(success)
        
        if similarity is not None:
            self.similarity_scores.append(similarity)
            self.performance_stats["similarity_scores"].append(similarity)
            self.performance_stats["accuracy_scores"].append(similarity)
        
        # Update performance stats
        self.performance_stats["verification_times"].append(duration)
        self.performance_stats["verification_results"].append(success)
        
    def record_processing(self, duration: float):
        """Record processing time
        
        Args:
            duration: Time taken for processing
        """
        self.processing_times.append(duration)
        self.performance_stats["processing_times"].append(duration)
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics
        
        Returns:
            Dict containing CPU and memory usage
        """
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info()
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": memory.rss / 1024 / 1024
        }
        
        # Update stats
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(metrics["memory_percent"])
        
        self.performance_stats["cpu_usage"].append(cpu_percent)
        self.performance_stats["memory_usage"].append(metrics["memory_percent"])
        
        return metrics
        
    def get_verification_rate(self) -> float:
        """Get verification success rate"""
        if not self.verification_results:
            return 0.0
        return sum(self.verification_results) / len(self.verification_results)
        
    def get_average_time(self) -> float:
        """Get average verification time"""
        if not self.verification_times:
            return 0.0
        return np.mean(self.verification_times)
        
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary
        
        Returns:
            Dict containing performance metrics
        """
        # Calculate basic metrics
        avg_verification_time = np.mean(self.verification_times) if self.verification_times else 0.0
        verification_rate = self.get_verification_rate()
        avg_similarity = np.mean(self.similarity_scores) if self.similarity_scores else 0.0
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0.0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0.0
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            'avg_verification_time': float(avg_verification_time),
            'verification_rate': float(verification_rate),
            'avg_similarity': float(avg_similarity),
            'avg_cpu_usage': float(avg_cpu),
            'avg_memory_usage': float(avg_memory),
            'avg_processing_time': float(avg_processing),
            'accuracy': float(verification_rate),
            'verification': {
                'success_rate': float(verification_rate),
                'avg_time': float(avg_verification_time),
                'avg_similarity': float(avg_similarity)
            },
            'processing': {
                'avg_time': float(avg_processing),
                'cpu_usage': float(avg_cpu),
                'memory_usage': float(avg_memory)
            },
            'system': {
                'cpu_usage': float(avg_cpu),
                'memory_usage': float(avg_memory),
                'verification_rate': float(verification_rate)
            }
        }
        
    def optimize_settings(self) -> Dict:
        """Optimize recognition settings based on performance
        
        Returns:
            Dict containing optimized settings
        """
        try:
            # Initialize with default settings
            settings = {
                'chunk_size': 1024,
                'buffer_size': 4096,
                'threshold': 0.5,
                'timeout': 5.0,
                'processing_threads': 2,
                'feature_settings': {
                    'n_mfcc': 20,
                    'frame_length': 0.025,
                    'frame_shift': 0.010
                }
            }
            
            # Get current metrics
            metrics = self.get_performance_summary()
            
            # Optimize based on verification time
            if metrics['avg_verification_time'] > self.max_verification_time:
                settings.update({
                    'chunk_size': 512,
                    'buffer_size': 2048,
                    'processing_threads': 4,
                    'feature_settings': {
                        'n_mfcc': 13,
                        'frame_length': 0.020,
                        'frame_shift': 0.010
                    }
                })
            
            # Optimize based on CPU usage
            if metrics['avg_cpu_usage'] > self.max_cpu_usage:
                settings.update({
                    'chunk_size': 2048,
                    'buffer_size': 8192,
                    'processing_threads': 1,
                    'timeout': 3.0
                })
                
            # Ensure all values are numeric
            settings = {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in settings.items()}
                
            return settings
            
        except Exception as e:
            logger.error(f"Error optimizing settings: {e}")
            return {} 