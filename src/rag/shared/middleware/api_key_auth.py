"""API key authentication middleware for FastAPI applications."""

import logging
from typing import Dict, List, Any, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.
    
    This middleware validates API keys provided in request headers or query parameters.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        api_keys: List[str],
        header_name: str = "X-API-Key",
        allow_query_param: bool = True,
        query_param_name: str = "api_key",
        exempt_paths: List[str] = None,
        admin_keys: List[str] = None,
        config: Dict[str, Any] = None
    ):
        """Initialize the API key middleware.
        
        Args:
            app: ASGI application
            api_keys: List of valid API keys
            header_name: Name of the header containing the API key
            allow_query_param: Whether to allow API key in query parameters
            query_param_name: Name of the query parameter containing the API key
            exempt_paths: List of paths that are exempt from API key validation
            admin_keys: List of API keys that have admin privileges
            config: Optional configuration dictionary
        """
        super().__init__(app)
        
        # Use config values if provided, otherwise use parameters
        if config:
            self.api_keys = config.get("api_keys", api_keys)
            self.header_name = config.get("header_name", header_name)
            self.allow_query_param = config.get("allow_query_param", allow_query_param)
            self.query_param_name = config.get("query_param_name", query_param_name)
            self.exempt_paths = config.get("exempt_paths", exempt_paths or [])
            self.admin_keys = config.get("admin_keys", admin_keys or [])
        else:
            self.api_keys = api_keys
            self.header_name = header_name
            self.allow_query_param = allow_query_param
            self.query_param_name = query_param_name
            self.exempt_paths = exempt_paths or []
            self.admin_keys = admin_keys or []
        
        logger.info(f"API key middleware initialized with {len(self.api_keys)} keys")
    
    async def dispatch(self, request: Request, call_next):
        """Process a request through the API key authentication.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
            
        Returns:
            The response
        """
        # Skip authentication for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get API key from header or query parameter
        api_key = self._get_api_key(request)
        
        # Validate API key
        if not api_key or not self._is_valid_key(api_key):
            logger.warning(f"Invalid or missing API key for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"}
            )
        
        # Add API key info to request state
        request.state.api_key = api_key
        request.state.is_admin = api_key in self.admin_keys
        
        # Process the request
        return await call_next(request)
    
    def _get_api_key(self, request: Request) -> Optional[str]:
        """Get API key from request headers or query parameters.
        
        Args:
            request: The incoming request
            
        Returns:
            API key if found, None otherwise
        """
        # Try to get API key from header
        api_key = request.headers.get(self.header_name)
        
        # If not found in header and query parameters are allowed, try query parameter
        if not api_key and self.allow_query_param:
            api_key = request.query_params.get(self.query_param_name)
        
        # If still not found, try Authorization header (Bearer token)
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        return api_key
    
    def _is_valid_key(self, api_key: str) -> bool:
        """Check if API key is valid.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key is valid, False otherwise
        """
        # If no API keys configured, authentication is disabled
        if not self.api_keys and not self.admin_keys:
            return True
            
        # Check if API key is in the list of valid keys
        return api_key in self.api_keys or api_key in self.admin_keys


def api_key_auth(request: Request, api_keys: List[str] = None, admin_keys: List[str] = None) -> bool:
    """Dependency for API key authentication in FastAPI endpoints.
    
    Args:
        request: FastAPI request
        api_keys: List of valid API keys (optional if already in request state)
        admin_keys: List of admin API keys (optional if already in request state)
        
    Returns:
        True if authentication passed
        
    Raises:
        HTTPException: If authentication failed
    """
    # If API key is already validated by middleware
    if hasattr(request.state, "api_key"):
        return True
    
    # Get API key from header or query parameter
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        api_key = request.query_params.get("api_key")
    
    # If still not found, try Authorization header (Bearer token)
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
    
    # Validate API key
    if not api_key:
        logger.warning(f"Missing API key for {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Check if API key is valid
    if (api_keys and api_key in api_keys) or (admin_keys and api_key in admin_keys):
        request.state.api_key = api_key
        request.state.is_admin = admin_keys and api_key in admin_keys
        return True
    
    logger.warning(f"Invalid API key for {request.url.path}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key"
    )
