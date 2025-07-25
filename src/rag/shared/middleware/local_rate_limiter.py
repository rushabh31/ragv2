"""Local in-memory rate limiter implementation."""

import time
import logging
from collections import defaultdict, deque
from threading import RLock
from typing import Dict, Deque, Tuple, Optional, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

logger = logging.getLogger(__name__)

class LocalRateLimiter(BaseHTTPMiddleware):
    """Rate limiting middleware using local in-memory storage.
    
    This middleware tracks request rates by client IP or other identifiers
    and enforces rate limits without requiring a Redis server.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_period: int = 100,
        period_seconds: int = 60,
        identifier_key: str = "client_ip",
        headers_enabled: bool = True,
        exempted_paths: list = None,
        api_key_exempt: bool = True,
        config: Dict[str, Any] = None
    ):
        """Initialize the rate limiter.
        
        Args:
            app: ASGI application
            requests_per_period: Maximum number of requests allowed in the period
            period_seconds: Time period in seconds for the rate limit window
            identifier_key: Request attribute to use as rate limit key
            headers_enabled: Whether to include rate limit headers in responses
            exempted_paths: List of path prefixes that are exempt from rate limiting
            api_key_exempt: Whether requests with API keys are exempt from rate limiting
            config: Optional configuration dictionary
        """
        super().__init__(app)
        
        # Use config values if provided, otherwise use parameters
        if config:
            self.requests_per_period = config.get("requests", requests_per_period)
            self.period_seconds = config.get("period_seconds", period_seconds)
            self.identifier_key = config.get("identifier_key", identifier_key)
            self.headers_enabled = config.get("headers_enabled", headers_enabled)
            self.exempted_paths = config.get("exempted_paths", exempted_paths or [])
            self.api_key_exempt = config.get("api_key_exempt", api_key_exempt)
        else:
            self.requests_per_period = requests_per_period
            self.period_seconds = period_seconds
            self.identifier_key = identifier_key
            self.headers_enabled = headers_enabled
            self.exempted_paths = exempted_paths or []
            self.api_key_exempt = api_key_exempt
        
        # Store request timestamps for each client
        self.request_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.requests_per_period * 2))
        
        # Lock for thread safety
        self.lock = RLock()
        
        logger.info(
            f"Local rate limiter initialized with {self.requests_per_period} "
            f"requests per {self.period_seconds} seconds"
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process a request through the rate limiter.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
            
        Returns:
            The response
        """
        # Skip rate limiting for exempted paths
        if any(request.url.path.startswith(path) for path in self.exempted_paths):
            return await call_next(request)
        
        # Skip rate limiting if the request has an API key and api_key_exempt is True
        if self.api_key_exempt and (
            request.headers.get("X-API-Key") or 
            request.headers.get("Authorization") or
            request.query_params.get("api_key")
        ):
            return await call_next(request)
        
        # Get client identifier
        identifier = self._get_identifier(request)
        
        # Check rate limit
        allowed, limit_info = self._check_rate_limit(identifier)
        
        if not allowed:
            # Return 429 Too Many Requests
            response = Response(
                content={"detail": "Rate limit exceeded"},
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json"
            )
            
            # Add rate limit headers if enabled
            if self.headers_enabled:
                self._add_rate_limit_headers(response, limit_info)
            
            return response
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response if enabled
        if self.headers_enabled:
            self._add_rate_limit_headers(response, limit_info)
        
        return response
    
    def _get_identifier(self, request: Request) -> str:
        """Get the client identifier from the request.
        
        Args:
            request: The incoming request
            
        Returns:
            Client identifier string
        """
        if self.identifier_key == "client_ip":
            # Get client IP from headers or connection
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # Get first IP if multiple are provided
                return forwarded.split(",")[0].strip()
            else:
                return request.client.host if request.client else "unknown"
        else:
            # Use specified request attribute
            return str(getattr(request, self.identifier_key, "unknown"))
    
    def _check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, int]]:
        """Check if the request is within rate limits.
        
        Args:
            identifier: Client identifier
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        window_start = current_time - self.period_seconds
        
        with self.lock:
            # Add current request timestamp
            self.request_history[identifier].append(current_time)
            
            # Count requests in the current window
            requests_in_window = sum(1 for ts in self.request_history[identifier] if ts > window_start)
            
            # Clean up old entries periodically
            if len(self.request_history[identifier]) > self.requests_per_period:
                # Remove timestamps outside the window
                while (
                    self.request_history[identifier] and 
                    self.request_history[identifier][0] <= window_start
                ):
                    self.request_history[identifier].popleft()
            
            # Calculate remaining requests
            remaining = max(0, self.requests_per_period - requests_in_window)
            
            # Calculate reset time in seconds
            if requests_in_window >= self.requests_per_period and self.request_history[identifier]:
                oldest_timestamp = self.request_history[identifier][0]
                reset_seconds = int(oldest_timestamp + self.period_seconds - current_time)
            else:
                reset_seconds = self.period_seconds
            
            # Check if allowed
            allowed = requests_in_window <= self.requests_per_period
            
            limit_info = {
                "limit": self.requests_per_period,
                "remaining": remaining,
                "reset": reset_seconds,
                "used": requests_in_window
            }
            
            return allowed, limit_info
    
    def _add_rate_limit_headers(self, response: Response, limit_info: Dict[str, int]):
        """Add rate limit headers to the response.
        
        Args:
            response: The response object
            limit_info: Rate limit information
        """
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
        
        if limit_info["remaining"] <= 0:
            response.headers["Retry-After"] = str(limit_info["reset"])
