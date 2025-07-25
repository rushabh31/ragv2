"""
Universal Authentication Manager for Multi-Service Token Management.

This module provides enterprise-grade authentication management that can be used
across all models and services (Vertex AI, OpenAI, etc.) with unified token handling.
"""

import asyncio
import logging
import os
import threading
from typing import Dict, Any, Optional, Union, Type
from dataclasses import dataclass
from credentials import Credentials
import vertexai

from .token_service import TokenService, TokenConfig

logger = logging.getLogger(__name__)


class TokenCredentials(Credentials):
    """Custom credentials class for token-based authentication."""
    
    def __init__(self, token: str):
        """
        Initialize TokenCredentials.
        
        Args:
            token: Bearer token for authentication
        """
        super().__init__()
        self.token = token
        
    def refresh(self, request):
        """Refresh method required by Credentials interface."""
        pass
        
    def before_request(self, request, method, url, headers):
        """Add authorization header before making requests."""
        headers["Authorization"] = f"Bearer {self.token}"


@dataclass
class ServiceConfig:
    """Configuration for individual services."""
    service_name: str
    project_id: Optional[str] = None
    location: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_transport: str = "grpc"
    
    @classmethod
    def from_env(cls, service_name: str) -> 'ServiceConfig':
        """Create ServiceConfig from environment variables."""
        return cls(
            service_name=service_name,
            project_id=os.environ.get("PROJECT_ID"),
            location=os.environ.get("VERTEXAI_LOCATION", "us-central1"),
            api_endpoint=os.environ.get("VERTEXAI_API_ENDPOINT"),
            api_transport=os.environ.get("VERTEXAI_API_TRANSPORT", "grpc")
        )


class UniversalAuthManager:
    """
    Universal Authentication Manager for multi-service token management.
    
    This class provides enterprise-grade authentication management that can be used
    across all models and services with unified token handling, health monitoring,
    and automatic refresh capabilities.
    
    Features:
    - Unified token management across all services
    - Service-specific configuration and health monitoring
    - Thread-safe operations with proper locking
    - Automatic token refresh and validation
    - Circuit breaker patterns for resilience
    - Comprehensive logging and error handling
    - Support for multiple authentication providers
    """
    
    _instances: Dict[str, 'UniversalAuthManager'] = {}
    _lock = threading.RLock()
    
    def __new__(cls, service_name: str = "default") -> 'UniversalAuthManager':
        """Singleton pattern per service name."""
        with cls._lock:
            if service_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[service_name] = instance
                instance._initialized = False
            return cls._instances[service_name]
    
    def __init__(self, service_name: str = "default"):
        """
        Initialize UniversalAuthManager.
        
        Args:
            service_name: Name of the service using this auth manager
        """
        if self._initialized:
            return
            
        self.service_name = service_name
        self._token_service: Optional[TokenService] = None
        self._service_config: Optional[ServiceConfig] = None
        self._authenticated_clients: Dict[str, Any] = {}
        self._health_status = "initializing"
        self._last_error: Optional[str] = None
        self._lock = threading.RLock()
        self._initialized = True
        
        logger.info(f"Initialized UniversalAuthManager for {service_name}")
    
    def configure(self, 
                  token_config: Optional[TokenConfig] = None,
                  service_config: Optional[ServiceConfig] = None):
        """
        Configure the authentication manager.
        
        Args:
            token_config: Token service configuration
            service_config: Service-specific configuration
        """
        with self._lock:
            try:
                # Use environment-based config if not provided
                if token_config is None:
                    token_config = TokenConfig.from_env()
                
                if service_config is None:
                    service_config = ServiceConfig.from_env(self.service_name)
                
                # Initialize token service
                self._token_service = TokenService(token_config, self.service_name)
                self._service_config = service_config
                self._health_status = "configured"
                
                logger.info(f"Configured UniversalAuthManager for {self.service_name}")
                
            except Exception as e:
                self._last_error = str(e)
                self._health_status = "configuration_failed"
                logger.error(f"Failed to configure UniversalAuthManager for {self.service_name}: {str(e)}")
                raise
    
    async def get_token(self, force_refresh: bool = False) -> str:
        """
        Get valid authentication token.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Valid access token
        """
        if self._token_service is None:
            self.configure()
        
        try:
            token = await self._token_service.get_token(force_refresh)
            self._health_status = "healthy"
            self._last_error = None
            return token
        except Exception as e:
            self._last_error = str(e)
            self._health_status = "token_failed"
            logger.error(f"Failed to get token for {self.service_name}: {str(e)}")
            raise
    
    def get_token_sync(self, force_refresh: bool = False) -> str:
        """
        Synchronous version of get_token.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Valid access token
        """
        return asyncio.run(self.get_token(force_refresh))
    
    async def get_credentials(self, force_refresh: bool = False) -> TokenCredentials:
        """
        Get TokenCredentials object for API authentication.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            TokenCredentials object
        """
        token = await self.get_token(force_refresh)
        return TokenCredentials(token)
    
    def get_credentials_sync(self, force_refresh: bool = False) -> TokenCredentials:
        """
        Synchronous version of get_credentials.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            TokenCredentials object
        """
        return asyncio.run(self.get_credentials(force_refresh))
    
    async def initialize_vertex_ai(self, force_refresh: bool = False) -> None:
        """
        Initialize Vertex AI with authenticated credentials.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
        """
        if self._service_config is None:
            self.configure()
        
        try:
            credentials = await self.get_credentials(force_refresh)
            
            # Initialize Vertex AI with authenticated credentials
            vertexai.init(
                project=self._service_config.project_id,
                location=self._service_config.location,
                api_transport=self._service_config.api_transport,
                api_endpoint=self._service_config.api_endpoint,
                credentials=credentials,
            )
            
            logger.info(f"Initialized Vertex AI for {self.service_name}")
            
        except Exception as e:
            self._last_error = str(e)
            self._health_status = "vertex_init_failed"
            logger.error(f"Failed to initialize Vertex AI for {self.service_name}: {str(e)}")
            raise
    
    def initialize_vertex_ai_sync(self, force_refresh: bool = False) -> None:
        """
        Synchronous version of initialize_vertex_ai.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
        """
        asyncio.run(self.initialize_vertex_ai(force_refresh))
    
    async def get_openai_headers(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get headers for OpenAI API authentication.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Dictionary of headers for OpenAI API
        """
        token = await self.get_token(force_refresh)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def get_openai_headers_sync(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Synchronous version of get_openai_headers.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Dictionary of headers for OpenAI API
        """
        return asyncio.run(self.get_openai_headers(force_refresh))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the authentication manager."""
        with self._lock:
            status = {
                "service_name": self.service_name,
                "auth_manager_status": self._health_status,
                "last_error": self._last_error,
                "configured": self._token_service is not None,
                "service_config": {
                    "project_id": self._service_config.project_id if self._service_config else None,
                    "location": self._service_config.location if self._service_config else None,
                    "api_endpoint": self._service_config.api_endpoint if self._service_config else None,
                } if self._service_config else None
            }
            
            # Add token service health if available
            if self._token_service:
                status["token_service"] = self._token_service.get_health_status()
            
            return status
    
    async def validate_authentication(self) -> bool:
        """
        Validate authentication by testing token acquisition.
        
        Returns:
            True if authentication is working, False otherwise
        """
        try:
            if self._token_service is None:
                self.configure()
            
            # Test token acquisition
            await self.get_token(force_refresh=True)
            
            # Test Vertex AI initialization if service config is available
            if self._service_config and self._service_config.project_id:
                await self.initialize_vertex_ai(force_refresh=True)
            
            self._health_status = "validated"
            return True
            
        except Exception as e:
            self._last_error = str(e)
            self._health_status = "validation_failed"
            logger.error(f"Authentication validation failed for {self.service_name}: {str(e)}")
            return False
    
    def invalidate_tokens(self):
        """Invalidate all cached tokens to force refresh."""
        with self._lock:
            if self._token_service:
                self._token_service.invalidate_token()
            
            self._authenticated_clients.clear()
            logger.info(f"Invalidated all tokens for {self.service_name}")
    
    def close(self):
        """Clean up resources."""
        with self._lock:
            if self._token_service:
                self._token_service.close()
                self._token_service = None
            
            self._authenticated_clients.clear()
            self._health_status = "closed"
            logger.info(f"Closed UniversalAuthManager for {self.service_name}")
    
    @classmethod
    def get_instance(cls, service_name: str = "default") -> 'UniversalAuthManager':
        """Get existing instance of UniversalAuthManager."""
        with cls._lock:
            if service_name not in cls._instances:
                raise ValueError(f"No UniversalAuthManager instance found for {service_name}")
            return cls._instances[service_name]
    
    @classmethod
    def list_instances(cls) -> Dict[str, Dict[str, Any]]:
        """List all active UniversalAuthManager instances and their health status."""
        with cls._lock:
            return {
                name: instance.get_health_status() 
                for name, instance in cls._instances.items()
            }
