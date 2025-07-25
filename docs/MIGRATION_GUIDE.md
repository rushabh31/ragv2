# Migration Guide: Redis to Local Cache

This guide explains how to migrate your RAG System deployment from Redis-based caching to the new local in-memory cache implementation.

## Table of Contents

1. [Overview](#overview)
2. [Migration Steps](#migration-steps)
3. [Configuration Changes](#configuration-changes)
4. [API and Service Updates](#api-and-service-updates)
5. [Performance Considerations](#performance-considerations)
6. [Troubleshooting](#troubleshooting)

## Overview

The RAG System previously required Redis for caching and rate limiting. With the latest update, the system now supports a local in-memory cache option that eliminates the Redis dependency while maintaining all functionality.

### Benefits of Migration

- **Simplified Deployment**: No need to provision and maintain a Redis server
- **Reduced Operational Complexity**: Fewer dependencies to manage
- **Improved Local Development**: Easier setup for development environments
- **Consistent API**: The cache interface remains the same, so application code is unaffected

### Components Affected

- Cache Manager
- Rate Limiting Middleware
- Configuration Files

## Migration Steps

### Step 1: Update Configuration

1. Edit your `config.yaml` file to change the cache provider:

```yaml
# Before
cache:
  enabled: true
  provider: redis
  redis_url: "${REDIS_URL}"
  default_ttl_seconds: 3600

# After
cache:
  enabled: true
  provider: local
  default_ttl_seconds: 3600
  max_size: 10000
  cleanup_interval_seconds: 300
```

2. Update rate limiting configuration:

```yaml
# Before
api:
  chatbot:
    rate_limiting:
      enabled: true
      redis_url: "${REDIS_URL}"
      requests: 100
      period_seconds: 60

# After
api:
  chatbot:
    rate_limiting:
      enabled: true
      provider: local
      requests: 100
      period_seconds: 60
```

### Step 2: Remove Redis Environment Variables

If you're using environment variables for configuration, you can remove Redis-related variables:

```bash
# Before
export REDIS_URL=redis://localhost:6379/0
export API_KEY=your_api_key_here
export ADMIN_API_KEY=your_admin_api_key_here

# After
export API_KEY=your_api_key_here
export ADMIN_API_KEY=your_admin_api_key_here
```

### Step 3: Update Dependencies (Optional)

If you're no longer using Redis for any part of your application, you can remove Redis-related dependencies:

```bash
# Remove Redis dependencies if no longer needed
pip uninstall redis aioredis fastapi-limiter
```

Note: If you still need Redis for other parts of your application or want to maintain the option to switch back, keep these dependencies installed.

## Configuration Changes

### Local Cache Options

The local cache implementation supports the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `max_size` | Maximum number of items in the cache | 10000 |
| `cleanup_interval_seconds` | Interval for TTL cleanup task | 300 |
| `default_ttl_seconds` | Default TTL for cached items | 3600 |

### Rate Limiting Options

The local rate limiter supports the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `requests` | Maximum number of requests allowed per period | 100 |
| `period_seconds` | Time period for rate limiting in seconds | 60 |
| `exempted_paths` | Paths exempt from rate limiting | ["/docs", "/redoc", "/openapi.json", "/health"] |

## API and Service Updates

### Cache Manager API

The API for the cache manager remains consistent, so no code changes are needed:

```python
# Before and after migration - the API remains the same
await cache_manager.get(key)
await cache_manager.set(key, value, ttl=3600)
await cache_manager.delete(key)
```

### Service Configuration

If you directly instantiate the cache manager in your code:

```python
# Before
from controlsgenai.funcs.rag.src.shared.cache.cache_manager import CacheManager

cache_manager = CacheManager(provider="redis", redis_url="redis://localhost:6379/0")

# After
from controlsgenai.funcs.rag.src.shared.cache.cache_manager import CacheManager

cache_manager = CacheManager(provider="local", max_size=10000, cleanup_interval_seconds=300)
```

## Performance Considerations

### Memory Usage

The local cache stores all data in memory. Consider the following when configuring:

- Set an appropriate `max_size` based on your server's available memory
- Monitor memory usage during operation
- Adjust `default_ttl_seconds` to control how long items remain in cache
- The LRU (Least Recently Used) eviction policy will remove older items when the cache reaches `max_size`

### Concurrency

The local cache implementation is thread-safe and designed for concurrent access:

- Uses read-write locks for cache operations
- Supports high concurrency for read operations
- Write operations may block during high contention

### Persistence

Unlike Redis, the local cache is not persistent:

- Cache contents are lost when the service restarts
- Consider using a warm-up strategy if cold caches impact performance
- For mission-critical deployments that require persistence, consider keeping Redis as an option

## Troubleshooting

### Common Issues

#### Cache Misses After Migration

**Problem**: Higher than expected cache miss rate after migration.

**Solution**: 
- Verify that the cache is enabled in configuration
- Check that the `max_size` is large enough for your workload
- Ensure services are running with enough memory allocation

#### Memory Pressure

**Problem**: System memory usage grows too high.

**Solution**:
- Reduce the `max_size` configuration
- Decrease the `default_ttl_seconds` value
- Consider increasing the frequency of the cleanup task by lowering `cleanup_interval_seconds`

#### Rate Limiting Issues

**Problem**: Rate limiting behaves inconsistently across service restarts.

**Solution**:
- Local rate limiting state is reset when services restart
- If you need consistent rate limiting across restarts, consider keeping Redis for this component

### Fallback to Redis

If you encounter issues with the local cache implementation, you can always revert to Redis:

1. Update the configuration:
```yaml
cache:
  provider: redis
  redis_url: "${REDIS_URL}"
```

2. Ensure Redis is installed and running:
```bash
# Install Redis client if needed
pip install redis aioredis fastapi-limiter

# Set Redis URL
export REDIS_URL=redis://localhost:6379/0
```

## Conclusion

Migrating from Redis to the local cache implementation simplifies deployment and reduces dependencies while maintaining functionality. For most use cases, the local cache provides excellent performance with minimal configuration.

However, if your deployment requires distributed caching, persistence across restarts, or very large cache sizes, you may want to continue using Redis by simply changing the provider configuration.
