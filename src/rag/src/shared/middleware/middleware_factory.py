"""Factory for creating middleware components based on configuration."""

import logging
from typing import Dict, Any, List, Callable, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class MiddlewareFactory:
    """Factory for creating and configuring middleware components."""
    
    @staticmethod
    def add_cors_middleware(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Add CORS middleware to the application.
        
        Args:
            app: FastAPI application
            config: CORS configuration
        """
        if not config:
            config = {}
            
        allow_origins = config.get("allow_origins", ["*"])
        allow_credentials = config.get("allow_credentials", True)
        allow_methods = config.get("allow_methods", ["*"])
        allow_headers = config.get("allow_headers", ["*"])
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers
        )
        
        logger.info(f"Added CORS middleware with allow_origins={allow_origins}")
    
    @staticmethod
    def add_auth_middleware(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Add authentication middleware to the application.
        
        Args:
            app: FastAPI application
            config: Authentication configuration
        """
        from src.rag.src.shared.middleware.api_key_auth import APIKeyMiddleware
        
        if not config:
            config = {}
            
        api_keys = config.get("api_keys", [])
        header_name = config.get("header_name", "X-API-Key")
        allow_query_param = config.get("allow_query_param", True)
        
        app.add_middleware(
            APIKeyMiddleware,
            api_keys=api_keys,
            header_name=header_name,
            allow_query_param=allow_query_param
        )
        
        logger.info(f"Added API key authentication middleware with {len(api_keys)} keys")
    
    @staticmethod
    def add_rate_limiting_middleware(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Add rate limiting middleware to the application.
        
        Args:
            app: FastAPI application
            config: Rate limiting configuration
        """
        if not config:
            config = {}
            
        if not config.get("enabled", False):
            logger.info("Rate limiting is disabled")
            return
            
        provider = config.get("provider", "local").lower()
        
        if provider == "redis":
            try:
                from fastapi_limiter import FastAPILimiter
                from fastapi_limiter.depends import RateLimiter
                import redis.asyncio as redis
                
                redis_url = config.get("redis_url", "redis://localhost:6379/0")
                requests = config.get("requests", 100)
                period_seconds = config.get("period_seconds", 60)
                
                # Create a dependency for rate limiting
                @app.on_event("startup")
                async def connect_to_redis():
                    redis_client = redis.from_url(redis_url, decode_responses=True)
                    await FastAPILimiter.init(redis_client)
                    logger.info(f"Connected to Redis for rate limiting at {redis_url}")
                
                # Add rate limiter dependency to required routes
                logger.info(f"Added Redis-based rate limiting: {requests} requests per {period_seconds} seconds")
                
            except ImportError:
                logger.warning("Redis rate limiting requires fastapi-limiter package")
                logger.info("Falling back to local rate limiting")
                MiddlewareFactory._add_local_rate_limiting(app, config)
                
        else:  # Use local rate limiting
            MiddlewareFactory._add_local_rate_limiting(app, config)
    
    @staticmethod
    def _add_local_rate_limiting(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Add local in-memory rate limiting middleware.
        
        Args:
            app: FastAPI application
            config: Rate limiting configuration
        """
        from src.rag.src.shared.middleware.local_rate_limiter import LocalRateLimiter
        
        if not config:
            config = {}
            
        requests = config.get("requests", 100)
        period_seconds = config.get("period_seconds", 60)
        
        app.add_middleware(
            LocalRateLimiter,
            requests_per_period=requests,
            period_seconds=period_seconds,
            config=config
        )
        
        logger.info(f"Added local rate limiting: {requests} requests per {period_seconds} seconds")
    
    @staticmethod
    def add_request_logging_middleware(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Add request logging middleware to the application.
        
        Args:
            app: FastAPI application
            config: Logging configuration
        """
        from src.rag.src.shared.middleware.request_logger import RequestLoggingMiddleware
        
        if not config:
            config = {}
            
        log_request_body = config.get("log_request_body", False)
        log_response_body = config.get("log_response_body", False)
        log_headers = config.get("log_headers", False)
        
        app.add_middleware(
            RequestLoggingMiddleware,
            log_request_body=log_request_body,
            log_response_body=log_response_body,
            log_headers=log_headers
        )
        
        logger.info("Added request logging middleware")
    
    @staticmethod
    def configure_app_middleware(app: FastAPI, config: Dict[str, Any] = None) -> None:
        """Configure all middleware for a FastAPI application.
        
        Args:
            app: FastAPI application
            config: Middleware configuration
        """
        if not config:
            config = {}
            
        # Add middleware in the correct order
        # 1. CORS
        cors_config = config.get("cors", {})
        MiddlewareFactory.add_cors_middleware(app, cors_config)
        
        # 2. Authentication
        auth_config = config.get("authentication", {})
        if auth_config.get("enabled", True):
            MiddlewareFactory.add_auth_middleware(app, auth_config)
        
        # 3. Rate limiting
        rate_limiting_config = config.get("rate_limiting", {})
        if rate_limiting_config.get("enabled", True):
            MiddlewareFactory.add_rate_limiting_middleware(app, rate_limiting_config)
        
        # 4. Request logging
        logging_config = config.get("logging", {})
        if logging_config.get("enabled", True):
            MiddlewareFactory.add_request_logging_middleware(app, logging_config)
