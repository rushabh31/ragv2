"""In-memory local cache manager implementation."""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import OrderedDict
from threading import RLock

logger = logging.getLogger(__name__)

class LocalCacheManager:
    """Thread-safe in-memory cache implementation.
    
    This implements a local in-memory cache with LRU (least recently used) eviction policy
    and time-based expiration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the local cache manager.
        
        Args:
            config: Configuration dictionary for the cache
        """
        self._config = config or {}
        self._default_ttl = self._config.get("default_ttl_seconds", 3600)  # 1 hour default
        self._max_size = self._config.get("max_size", 10000)  # Max cache entries
        
        # Use OrderedDict for LRU functionality
        self._cache: Dict[str, Tuple[Any, float]] = OrderedDict()
        
        # Lock for thread safety
        self._lock = RLock()
        
        # Cleanup interval in seconds
        self._cleanup_interval = self._config.get("cleanup_interval_seconds", 60)
        
        # Cleanup task reference
        self._cleanup_task = None
        
        # Schedule cleanup task only if there's a running event loop
        if self._config.get("auto_cleanup", True):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._cleanup_task = loop.create_task(self._background_cleanup())
                    logger.debug("Background cache cleanup task scheduled")
                else:
                    logger.debug("No running event loop, background cleanup disabled")
            except RuntimeError as e:
                logger.debug(f"Skipping background cleanup task: {str(e)}")
        
        logger.info(f"Local cache initialized with max size {self._max_size} and default TTL {self._default_ttl}s")
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # Check if expired
            if expiry < time.time():
                self._cache.pop(key)
                return None
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            return value
    
    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire_seconds: Optional TTL in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        ttl = expire_seconds if expire_seconds is not None else self._default_ttl
        expiry = time.time() + ttl
        
        with self._lock:
            # If at capacity and this is a new key, remove the oldest item
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._cache.popitem(last=False)  # Remove first item (oldest)
            
            self._cache[key] = (value, expiry)
            # Move to end to mark as most recently used
            self._cache.move_to_end(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            self._cache.clear()
        return True
    
    async def _background_cleanup(self) -> None:
        """Background task to clean up expired entries."""
        try:
            while True:
                # Sleep first to avoid cleanup right at startup
                await asyncio.sleep(self._cleanup_interval)
                
                # Run cleanup
                await self._cleanup_expired()
                
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}", exc_info=True)
    
    async def _cleanup_expired(self) -> int:
        """Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            # Create a list of expired keys
            expired_keys = [key for key, (_, expiry) in self._cache.items() if expiry < now]
            
            # Remove expired keys
            for key in expired_keys:
                self._cache.pop(key)
                removed += 1
        
        if removed > 0:
            logger.debug(f"Removed {removed} expired entries from cache")
        
        return removed
    
    async def shutdown(self) -> None:
        """Shutdown the cache manager and cancel background tasks."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                # Wait for task to be cancelled
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
                logger.debug("Cache cleanup task cancelled successfully")
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Timeout waiting for cache cleanup task to cancel")
            except Exception as e:
                logger.error(f"Error shutting down cache manager: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            now = time.time()
            total_items = len(self._cache)
            expired_items = sum(1 for _, expiry in self._cache.values() if expiry < now)
            
            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "active_items": total_items - expired_items,
                "max_size": self._max_size,
                "usage_percent": (total_items / self._max_size) * 100 if self._max_size > 0 else 0
            }
