"""
MARK5 System Monitor Utilities
==============================
Consolidated utilities for system information and health monitoring.
"""

import platform
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information
    
    Returns:
        Dictionary containing system metrics:
        - platform: Operating system
        - platform_version: OS version
        - processor: CPU information
        - python_version: Python version
        - cpu_percent: CPU usage percentage
        - memory_total_gb: Total RAM in GB
        - memory_available_gb: Available RAM in GB
        - memory_percent: Memory usage percentage
        - disk_total_gb: Total disk space in GB
        - disk_free_gb: Free disk space in GB
        - disk_percent: Disk usage percentage
        - timestamp: Current ISO timestamp
    """
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat()
    }

    if psutil:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            info.update({
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": disk.percent,
            })
        except Exception as e:
            info["error"] = f"Could not retrieve system metrics: {str(e)}"
    else:
        info["error"] = "psutil not installed - limited system info available"
        
    return info


class SystemHealthMonitor:
    """
    Centralized monitor for system health, resources, and error tracking.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.critical_errors: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.last_check = 0
        self.health_status = "HEALTHY"
        
        # Thresholds
        self.cpu_threshold = 90.0  # %
        self.memory_threshold = 90.0  # %
        self.disk_threshold = 90.0  # %
        self.error_rate_threshold = 10  # errors per minute

    def check_resources(self) -> Dict[str, Any]:
        """Check system resources and return metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat()
        }
        
        if psutil:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics.update({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                })
                
                # Check thresholds
                if cpu_percent > self.cpu_threshold:
                    self._log_warning(f"High CPU usage: {cpu_percent}%")
                if memory.percent > self.memory_threshold:
                    self._log_warning(f"High Memory usage: {memory.percent}%")
                if disk.percent > self.disk_threshold:
                    self._log_warning(f"Low Disk Space: {disk.percent}% used")
            except Exception as e:
                self.logger.error(f"Error checking resources: {e}")
                metrics['error'] = str(e)
        else:
            metrics['error'] = "psutil not installed"
            
        return metrics

    def log_error(self, component: str, error: str, critical: bool = False):
        """Log an error and update stats."""
        timestamp = datetime.now().isoformat()
        
        if component not in self.error_counts:
            self.error_counts[component] = 0
        self.error_counts[component] += 1
        
        error_entry = {
            'timestamp': timestamp,
            'component': component,
            'error': error,
            'critical': critical
        }
        
        if critical:
            self.critical_errors.append(error_entry)
            self.logger.critical(f"CRITICAL ERROR in {component}: {error}")
            self.health_status = "CRITICAL"
        else:
            self.logger.error(f"Error in {component}: {error}")
            if self.health_status == "HEALTHY":
                self.health_status = "DEGRADED"

    def get_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        uptime = time.time() - self.start_time
        return {
            'status': self.health_status,
            'uptime_seconds': uptime,
            'total_errors': sum(self.error_counts.values()),
            'critical_errors': len(self.critical_errors),
            'component_errors': self.error_counts
        }

    def _log_warning(self, message: str):
        self.logger.warning(message)
