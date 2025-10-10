"""Performance monitoring utilities for spatial optimization"""
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps
import psutil
import os

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.metrics[operation] = duration
        logger.info(f"Operation '{operation}' completed in {duration:.3f}s")
        
        # Clean up
        del self.start_times[operation]
        return duration
    
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        process = psutil.Process(os.getpid())
        
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    def log_system_metrics(self, operation: str = "system_check"):
        """Log current system metrics"""
        metrics = self.get_system_metrics()
        logger.info(f"System metrics for {operation}: {metrics}")
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "operation_times": self.metrics.copy(),
            "system_metrics": self.get_system_metrics(),
            "total_operations": len(self.metrics)
        }


def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Get or create monitor instance
            monitor = getattr(wrapper, '_monitor', None)
            if monitor is None:
                monitor = PerformanceMonitor()
                wrapper._monitor = monitor
            
            with monitor.timer(name):
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator


# Global performance monitor instance
global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return global_monitor


def log_performance_summary():
    """Log a summary of all performance metrics"""
    summary = global_monitor.get_summary()
    
    logger.info("=== Performance Summary ===")
    for operation, duration in summary["operation_times"].items():
        logger.info(f"{operation}: {duration:.3f}s")
    
    logger.info("=== System Metrics ===")
    for metric, value in summary["system_metrics"].items():
        logger.info(f"{metric}: {value}")
    
    return summary
