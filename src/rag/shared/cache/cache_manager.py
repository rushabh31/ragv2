import asyncio
import json
import hashlib
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Awaitable, Union, List

logger = logging.getLogger(__name__)

# Import cache implementations conditionally to avoid unnecessary dependencies
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

T = TypeVar('T')

class CacheManager:
    """Cache manager for the RAG system.
    
    Provides a unified interface for caching operations using different backends.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the cache manager.
        
        Args:
            config: Configuration dictionary for the cache
        """
        self._config = config or {}
        self._enabled = self._config.get("enabled", True)
        self._default_ttl = self._config.get("default_ttl_seconds", 3600)  # Default: 1 hour
        self._provider = self._config.get("provider", "local").lower()
        
        # Cache client reference
        self._cache_client = None
        
        if not self._enabled:
            logger.info("Caching is disabled")
            return
            
        # Initialize cache based on provider
        if self._provider == "redis":
            if not REDIS_AVAILABLE:
                logger.warning("Redis package not installed. Falling back to local cache.")
                self._init_local_cache()
            else:
                self._init_redis_cache()
        elif self._provider == "local":
            self._init_local_cache()
        else:
            logger.warning(f"Unsupported cache provider '{self._provider}'. Falling back to local cache.")
            self._init_local_cache()
    
    def _init_redis_cache(self):
        """Initialize Redis cache client."""
        import redis.asyncio as redis
        redis_url = self._config.get("redis_url", "redis://localhost:6379/0")
        try:
            self._cache_client = redis.from_url(redis_url, decode_responses=True)
            logger.info(f"Connected to Redis cache at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            logger.info("Falling back to local cache")
            self._init_local_cache()
    
    def _init_local_cache(self):
        """Initialize local in-memory cache client."""
        try:
            from src.rag.shared.cache.local_cache_manager import LocalCacheManager
            self._cache_client = LocalCacheManager(self._config)
            logger.info("Initialized local in-memory cache")
        except Exception as e:
            logger.error(f"Failed to initialize local cache: {str(e)}")
            self._cache_client = None
            self._enabled = False
    
    async def shutdown(self) -> None:
        """Shutdown the cache manager and cleanup resources.
        
        This method should be called during application shutdown to ensure proper resource cleanup.
        """
        if not self._enabled or not self._cache_client:
            return
            
        try:
            # Handle different cache providers
            if self._provider == "local" and hasattr(self._cache_client, "shutdown"):
                await self._cache_client.shutdown()
                logger.info("Local cache manager shut down successfully")
            elif self._provider == "redis" and hasattr(self._cache_client, "close"):
                await self._cache_client.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error shutting down cache manager: {str(e)}")
    
    async def check_connection(self) -> bool:
        """Check if the cache connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not self._enabled or not self._cache_client:
            return False
        
        try:
            if self._provider == "redis":
                await self._cache_client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache connection failed: {str(e)}")
            return False
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or disabled
        """
        if not self._enabled or not self._cache_client:
            return None
        
        try:
            # Get value from cache
            if self._provider == "redis":
                # Get the value from Redis
                value = await self._cache_client.get(key)
                
                # Deserialize JSON if it exists
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # If not valid JSON, return as is
                        return value
            else:
                # Get value from local cache (already deserialized)
                return await self._cache_client.get(key)
            
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, uses default_ttl if None
            
        Returns:
            True if successful, False otherwise
        """
        if not self._enabled or not self._cache_client:
            return False
        
        if ttl is None:
            ttl = self._default_ttl
            
        try:
            if self._provider == "redis":
                # Set the value in Redis
                if isinstance(value, (int, float, str, bool)):
                    return await self._cache_client.setex(key, ttl, value)
                else:
                    return await self._cache_client.setex(key, ttl, json.dumps(value))
            else:
                # Set value in local cache
                return await self._cache_client.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        if not self._enabled or not self._cache_client:
            return False
        
        try:
            if self._provider == "redis":
                # Delete the key from Redis
                result = await self._cache_client.delete(key)
                return result > 0
            else:
                # Delete from local cache
                return await self._cache_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    def generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from parameters.
        
        Args:
            prefix: Key prefix
            params: Dictionary of parameters to include in the key
            
        Returns:
            Generated cache key
        """
        key_string = f"{prefix}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_or_set(self, key: str, 
                         func: Callable[..., Awaitable[T]], 
                         ttl: Optional[int] = None,
                         serialize: Callable[[T], str] = json.dumps,
                         deserialize: Callable[[str], T] = json.loads) -> T:
        """Get a value from cache or compute and store it.
        
        Args:
            key: Cache key
            func: Async function to compute the value if not in cache
            ttl: Time-to-live in seconds
            serialize: Function to serialize the value to a string
            deserialize: Function to deserialize the value from a string
            
        Returns:
            Retrieved or computed value
        """
        cached_value = await self.get(key)
        
        if cached_value is not None:
            logger.debug(f"Cache hit for key: {key}")
            return deserialize(cached_value)
        
        logger.debug(f"Cache miss for key: {key}")
        result = await func()
        await self.set(key, serialize(result), ttl)
        return result
    
    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with a specific prefix.
        
        Args:
            prefix: Key prefix to clear
            
        Returns:
            Number of keys deleted
        """
        if not self._enabled or not self._cache_client:
            return 0
        
        try:
            if self._provider == "redis":
                cursor = b'0'
                count = 0
                
                while cursor:
                    cursor, keys = await self._cache_client.scan(
                        cursor=cursor, 
                        match=f"{prefix}*", 
                        count=100
                    )
                    
                    if keys:
                        count += await self._cache_client.delete(*keys)
                        
                    if cursor == b'0':
                        break
                        
                return count
            else:
                # Clear from local cache
                return await self._cache_client.clear_prefix(prefix)
        except Exception as e:
            logger.error(f"Error clearing cache prefix {prefix}: {str(e)}")
            return 0
