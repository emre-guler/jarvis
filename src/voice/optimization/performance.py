"""Performance optimization for speaker recognition"""
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Union
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
        
    def record_verification(self, duration: float, success: bool, score: float = None):
        """Record verification performance metrics"""
        try:
            self.performance_stats["verification_times"].append(duration)
            self.performance_stats["verification_success"].append(success)
            
            if score is not None:
                self.performance_stats["accuracy_scores"].append(score)
                
            # Log warning if verification time exceeds threshold
            if duration > 1.0:
                logger.warning(f"Verification time ({duration:.2f}s) exceeded threshold")
                
        except Exception as e:
            logger.error(f"Error recording verification metrics: {e}")
        
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
        
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics"""
        try:
            verification_times = self.performance_stats["verification_times"]
            processing_times = self.performance_stats["processing_times"]
            accuracy_scores = self.performance_stats["accuracy_scores"]
            cpu_usage = self.performance_stats["cpu_usage"]
            memory_usage = self.performance_stats["memory_usage"]
            
            summary = {
                "verification": {
                    "mean_time": float(np.mean(verification_times)),
                    "std_time": float(np.std(verification_times)),
                    "min_time": float(np.min(verification_times)),
                    "max_time": float(np.max(verification_times))
                },
                "accuracy": {
                    "mean_score": float(np.mean(accuracy_scores)),
                    "std_score": float(np.std(accuracy_scores))
                },
                "processing": {
                    "mean_time": float(np.mean(processing_times)),
                    "std_time": float(np.std(processing_times))
                },
                "system": {
                    "mean_cpu": float(np.mean(cpu_usage)),
                    "mean_memory": float(np.mean(memory_usage))
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}
            
    def optimize_settings(self) -> Dict[str, Union[int, float]]:
        """Optimize performance settings based on metrics"""
        try:
            summary = self.get_performance_summary()
            
            # Default settings
            settings = {
                "chunk_size": 1024,
                "buffer_size": 4096,
                "processing_threads": 2
            }
            
            # Adjust based on performance metrics
            if summary:
                # CPU usage based adjustments
                if summary["system"]["mean_cpu"] > 70:
                    settings["processing_threads"] = 1
                    settings["chunk_size"] = 2048
                elif summary["system"]["mean_cpu"] < 30:
                    settings["processing_threads"] = 4
                    settings["chunk_size"] = 512
                    
                # Memory usage based adjustments
                if summary["system"]["mean_memory"] > 1000:  # MB
                    settings["buffer_size"] = 2048
                elif summary["system"]["mean_memory"] < 500:  # MB
                    settings["buffer_size"] = 8192
                    
                # Latency based adjustments
                if "verification" in summary and summary["verification"]["mean_time"] > 1.0:
                    settings["chunk_size"] = min(settings["chunk_size"] * 2, 4096)
                    
            return settings
            
        except Exception as e:
            logger.error(f"Error optimizing settings: {e}")
            return {
                "chunk_size": 1024,
                "buffer_size": 4096,
                "processing_threads": 2
            } 