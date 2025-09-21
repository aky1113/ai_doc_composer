"""
Rate limiting for AI Documentary Composer API calls.

Implements robust rate limiting for external APIs, particularly Gemini API
which has a 15 requests per minute limit on the free tier.
"""

import time
import threading
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_minute: int
    requests_per_hour: int = None
    requests_per_day: int = None


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, rate_limit: RateLimit, safety_margin: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Rate limit configuration
            safety_margin: Reduce limit by this amount for safety (default: 1)
        """
        self.requests_per_minute = max(1, rate_limit.requests_per_minute - safety_margin)
        self.requests_per_hour = rate_limit.requests_per_hour
        self.requests_per_day = rate_limit.requests_per_day
        
        # Track request timestamps
        self._minute_requests: List[float] = []
        self._hour_requests: List[float] = []
        self._day_requests: List[float] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized: {self.requests_per_minute} req/min (safety margin: {safety_margin})")
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests older than the tracking window."""
        # Remove requests older than 1 minute
        minute_cutoff = current_time - 60
        self._minute_requests = [req for req in self._minute_requests if req > minute_cutoff]
        
        # Remove requests older than 1 hour (if tracking hourly)
        if self.requests_per_hour:
            hour_cutoff = current_time - 3600
            self._hour_requests = [req for req in self._hour_requests if req > hour_cutoff]
        
        # Remove requests older than 1 day (if tracking daily)
        if self.requests_per_day:
            day_cutoff = current_time - 86400
            self._day_requests = [req for req in self._day_requests if req > day_cutoff]
    
    def acquire(self, timeout: float = 300) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < timeout:
            with self._lock:
                current_time = time.time()
                self._cleanup_old_requests(current_time)
                
                # Check minute limit
                if len(self._minute_requests) >= self.requests_per_minute:
                    oldest_request = min(self._minute_requests)
                    wait_time = 60 - (current_time - oldest_request)
                    if wait_time > 0:
                        logger.info(f"Rate limit: waiting {wait_time:.1f}s for minute window")
                        time.sleep(min(wait_time + 0.1, timeout - (time.monotonic() - start_time)))
                        continue
                
                # Check hour limit (if configured)
                if self.requests_per_hour and len(self._hour_requests) >= self.requests_per_hour:
                    oldest_request = min(self._hour_requests)
                    wait_time = 3600 - (current_time - oldest_request)
                    if wait_time > 0:
                        logger.warning(f"Hourly rate limit exceeded, waiting {wait_time/60:.1f} minutes")
                        return False  # Don't wait for hour limits
                
                # Check day limit (if configured)
                if self.requests_per_day and len(self._day_requests) >= self.requests_per_day:
                    logger.error("Daily rate limit exceeded")
                    return False
                
                # Permission granted - record the request
                self._minute_requests.append(current_time)
                if self.requests_per_hour:
                    self._hour_requests.append(current_time)
                if self.requests_per_day:
                    self._day_requests.append(current_time)
                
                return True
        
        logger.error(f"Rate limiter timeout after {timeout}s")
        return False
    
    def get_status(self) -> Dict[str, int]:
        """Get current rate limiting status."""
        with self._lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time)
            
            return {
                "minute_requests": len(self._minute_requests),
                "minute_limit": self.requests_per_minute,
                "hour_requests": len(self._hour_requests) if self.requests_per_hour else None,
                "hour_limit": self.requests_per_hour,
                "day_requests": len(self._day_requests) if self.requests_per_day else None,
                "day_limit": self.requests_per_day
            }


# Global rate limiters for different services
_rate_limiters: Dict[str, RateLimiter] = {}

# Predefined rate limits
GEMINI_FLASH_FREE_TIER = RateLimit(
    requests_per_minute=15,
    requests_per_hour=1000,  # Conservative estimate
    requests_per_day=50000   # Conservative estimate
)

GEMINI_FLASH_PAID_TIER = RateLimit(
    requests_per_minute=1000,
    requests_per_hour=None,  # No strict hourly limit
    requests_per_day=None    # No strict daily limit
)


def get_rate_limiter(service: str, rate_limit: RateLimit = None) -> RateLimiter:
    """
    Get or create a rate limiter for a service.
    
    Args:
        service: Service name (e.g., 'gemini-flash')
        rate_limit: Rate limit configuration (optional)
    
    Returns:
        RateLimiter instance
    """
    if service not in _rate_limiters:
        if rate_limit is None:
            # Default to Gemini Flash free tier limits
            rate_limit = GEMINI_FLASH_FREE_TIER
        
        _rate_limiters[service] = RateLimiter(rate_limit)
    
    return _rate_limiters[service]


def rate_limited_call(service: str, func, *args, **kwargs):
    """
    Execute a function with rate limiting.
    
    Args:
        service: Service name for rate limiting
        func: Function to call
        *args, **kwargs: Arguments for the function
    
    Returns:
        Function result
        
    Raises:
        RuntimeError: If rate limit cannot be acquired
    """
    limiter = get_rate_limiter(service)
    
    if not limiter.acquire():
        status = limiter.get_status()
        raise RuntimeError(
            f"Rate limit exceeded for {service}. "
            f"Current: {status['minute_requests']}/{status['minute_limit']} req/min. "
            f"Please wait before retrying."
        )
    
    logger.debug(f"Rate limit acquired for {service}, executing request")
    return func(*args, **kwargs)


def reset_rate_limiter(service: str) -> None:
    """Reset rate limiter for a service (useful for testing)."""
    if service in _rate_limiters:
        del _rate_limiters[service]


def get_all_status() -> Dict[str, Dict[str, int]]:
    """Get status of all active rate limiters."""
    return {
        service: limiter.get_status() 
        for service, limiter in _rate_limiters.items()
    }