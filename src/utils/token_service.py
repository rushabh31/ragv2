"""
Universal Token Service for OAuth2 and API Authentication.

This module provides a professional, enterprise-grade token service that handles
OAuth2 token acquisition, caching, refresh, and validation across all services.
"""

import asyncio
import logging
import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.rag.shared.utils.env_manager import env_manager

logger = logging.getLogger(__name__)


@dataclass
class TokenConfig:
    """Configuration for token service."""
    endpoint_url: str
    client_id: str
    client_secret: str
    scope: str
    grant_type: str = "client_credentials"
    timeout: int = 10
    verify_ssl: bool = False
    max_retries: int = 3
    backoff_factor: float = 0.3
    
    @classmethod
    def from_env(cls) -> 'TokenConfig':
        """Create TokenConfig from environment variables."""
        return cls(
            endpoint_url=env_manager.get("COIN_CONSUMER_ENDPOINT_URL", ""),
            client_id=env_manager.get("COIN_CONSUMER_CLIENT_ID", ""),
            client_secret=env_manager.get("COIN_CONSUMER_CLIENT_SECRET", ""),
            scope=env_manager.get("COIN_CONSUMER_SCOPE", ""),
            timeout=env_manager.get_int("TOKEN_REQUEST_TIMEOUT", 10),
            verify_ssl=env_manager.get_bool("TOKEN_VERIFY_SSL", False),
            max_retries=env_manager.get_int("TOKEN_MAX_RETRIES", 3),
            backoff_factor=env_manager.get_float("TOKEN_BACKOFF_FACTOR", 0.3)
        )


@dataclass
class TokenInfo:
    """Token information with metadata."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    issued_at: float = None
    
    def __post_init__(self):
        if self.issued_at is None:
            self.issued_at = time.time()
    
    @property
    def expires_at(self) -> Optional[float]:
        """Calculate token expiration timestamp."""
        if self.expires_in is None:
            return None
        return self.issued_at + self.expires_in
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 5-minute buffer)."""
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - 300)  # 5-minute buffer
    
    @property
    def time_until_expiry(self) -> Optional[float]:
        """Get seconds until token expires."""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())


class TokenService:
    """
    Enterprise-grade token service for OAuth2 authentication.
    
    Features:
    - Automatic token refresh with configurable buffer
    - Thread-safe operations with proper locking
    - Comprehensive error handling and retry logic
    - Token caching and validation
    - Health monitoring and circuit breaker patterns
    - Detailed logging and metrics
    """
    
    def __init__(self, config: TokenConfig, service_name: str = "default"):
        """
        Initialize TokenService.
        
        Args:
            config: Token configuration
            service_name: Name of the service using this token service
        """
        self.config = config
        self.service_name = service_name
        self._token_info: Optional[TokenInfo] = None
        self._lock = threading.RLock()
        self._session = None
        self._health_status = "healthy"
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._circuit_breaker_until: Optional[float] = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized TokenService for {service_name}")
    
    def _validate_config(self):
        """Validate token configuration."""
        required_fields = ["endpoint_url", "client_id", "client_secret", "scope"]
        missing_fields = [field for field in required_fields if not getattr(self.config, field)]
        
        if missing_fields:
            raise ValueError(f"Missing required token configuration: {', '.join(missing_fields)}")
    
    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with retry strategy."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        
        return self._session
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_until is None:
            return False
        
        if time.time() >= self._circuit_breaker_until:
            self._circuit_breaker_until = None
            self._error_count = 0
            logger.info(f"Circuit breaker reset for {self.service_name}")
            return False
        
        return True
    
    def _open_circuit_breaker(self):
        """Open circuit breaker for 5 minutes."""
        self._circuit_breaker_until = time.time() + 300  # 5 minutes
        self._health_status = "circuit_breaker_open"
        logger.warning(f"Circuit breaker opened for {self.service_name} due to repeated failures")
    
    async def _fetch_token(self) -> TokenInfo:
        """Fetch new token from OAuth2 endpoint."""
        if self._is_circuit_breaker_open():
            raise RuntimeError(f"Circuit breaker is open for {self.service_name}")
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": self.config.grant_type,
            "scope": self.config.scope,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        
        try:
            logger.debug(f"Requesting new token for {self.service_name}")
            
            session = self._get_session()
            response = await asyncio.to_thread(
                session.post,
                self.config.endpoint_url,
                headers=headers,
                data=data,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token_info = TokenInfo(
                    access_token=token_data.get('access_token'),
                    token_type=token_data.get('token_type', 'Bearer'),
                    expires_in=token_data.get('expires_in'),
                    scope=token_data.get('scope')
                )
                
                # Reset error tracking on success
                self._error_count = 0
                self._last_error = None
                self._health_status = "healthy"
                
                logger.info(f"Successfully obtained token for {self.service_name}")
                return token_info
            else:
                error_msg = f"Token request failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            self._health_status = "unhealthy"
            
            # Open circuit breaker after 3 consecutive failures
            if self._error_count >= 3:
                self._open_circuit_breaker()
            
            logger.error(f"Failed to fetch token for {self.service_name}: {str(e)}")
            raise
    
    async def get_token(self, force_refresh: bool = False) -> str:
        """
        Get valid access token, refreshing if necessary.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Valid access token
        """
        with self._lock:
            # Check if we need to fetch/refresh token
            if (force_refresh or 
                self._token_info is None or 
                self._token_info.is_expired):
                
                logger.debug(f"Fetching new token for {self.service_name}")
                self._token_info = await self._fetch_token()
            
            return self._token_info.access_token
    
    def get_token_sync(self, force_refresh: bool = False) -> str:
        """
        Synchronous version of get_token.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Valid access token
        """
        return asyncio.run(self.get_token(force_refresh))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of the token service."""
        with self._lock:
            status = {
                "service_name": self.service_name,
                "health_status": self._health_status,
                "has_token": self._token_info is not None,
                "token_expired": self._token_info.is_expired if self._token_info else None,
                "time_until_expiry": self._token_info.time_until_expiry if self._token_info else None,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "circuit_breaker_open": self._is_circuit_breaker_open(),
                "circuit_breaker_until": self._circuit_breaker_until
            }
            
            return status
    
    async def validate_token(self) -> bool:
        """Validate current token by attempting to refresh it."""
        try:
            await self.get_token(force_refresh=True)
            return True
        except Exception as e:
            logger.error(f"Token validation failed for {self.service_name}: {str(e)}")
            return False
    
    def invalidate_token(self):
        """Invalidate current token to force refresh on next request."""
        with self._lock:
            self._token_info = None
            logger.info(f"Token invalidated for {self.service_name}")
    
    def close(self):
        """Clean up resources."""
        if self._session:
            self._session.close()
            self._session = None
        logger.info(f"TokenService closed for {self.service_name}")
