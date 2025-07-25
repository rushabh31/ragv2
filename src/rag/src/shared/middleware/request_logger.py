"""Request logging middleware for FastAPI applications."""

import time
import logging
import json
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request and response details.
    
    This middleware logs request and response details for debugging and monitoring.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        log_headers: bool = False,
        log_level: str = "INFO",
        config: Dict[str, Any] = None
    ):
        """Initialize the request logging middleware.
        
        Args:
            app: ASGI application
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            log_headers: Whether to log headers
            log_level: Logging level (INFO, DEBUG, etc.)
            config: Optional configuration dictionary
        """
        super().__init__(app)
        
        # Use config values if provided, otherwise use parameters
        if config:
            self.log_request_body = config.get("log_request_body", log_request_body)
            self.log_response_body = config.get("log_response_body", log_response_body)
            self.log_headers = config.get("log_headers", log_headers)
            self.log_level = config.get("log_level", log_level).upper()
        else:
            self.log_request_body = log_request_body
            self.log_response_body = log_response_body
            self.log_headers = log_headers
            self.log_level = log_level.upper()
        
        # Set log level
        self.numeric_level = getattr(logging, self.log_level, logging.INFO)
        
        logger.info("Request logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Process a request and log details.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
            
        Returns:
            The response
        """
        # Start timer
        start_time = time.time()
        
        # Get request details
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        method = request.method
        query_params = dict(request.query_params)
        
        # Log request details
        log_data = {
            "request": {
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "query_params": query_params
            }
        }
        
        # Log headers if enabled
        if self.log_headers:
            log_data["request"]["headers"] = dict(request.headers)
        
        # Log request body if enabled
        if self.log_request_body:
            try:
                # Try to read request body (this can only be done once)
                body = await request.body()
                # Create a new request with the same body to avoid consuming it
                # Request object is immutable, so we need to create a new one
                request = Request(request.scope, request._receive, request._send)
                
                # Try to parse body as JSON
                try:
                    body_text = body.decode("utf-8")
                    if body_text:
                        try:
                            body_json = json.loads(body_text)
                            log_data["request"]["body"] = body_json
                        except json.JSONDecodeError:
                            log_data["request"]["body"] = body_text
                except UnicodeDecodeError:
                    log_data["request"]["body"] = "<binary data>"
            except Exception as e:
                log_data["request"]["body_error"] = str(e)
        
        logger.log(self.numeric_level, f"Request: {method} {path} from {client_ip}")
        
        # Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            process_time = time.time() - start_time
            log_data["response"] = {
                "status_code": 500,
                "process_time": f"{process_time:.4f}s",
                "error": str(e)
            }
            logger.error(
                f"Error processing {method} {path}: {str(e)}",
                extra={"log_data": log_data}
            )
            raise
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response details
        log_data["response"] = {
            "status_code": response.status_code,
            "process_time": f"{process_time:.4f}s"
        }
        
        # Log response headers if enabled
        if self.log_headers:
            log_data["response"]["headers"] = dict(response.headers)
        
        # Log response body if enabled
        if self.log_response_body:
            try:
                # Get response body from original_response if available
                body = getattr(response, "body", None)
                if body:
                    try:
                        body_text = body.decode("utf-8")
                        try:
                            body_json = json.loads(body_text)
                            log_data["response"]["body"] = body_json
                        except json.JSONDecodeError:
                            log_data["response"]["body"] = body_text
                    except UnicodeDecodeError:
                        log_data["response"]["body"] = "<binary data>"
            except Exception as e:
                log_data["response"]["body_error"] = str(e)
        
        # Log response details
        logger.log(
            self.numeric_level,
            f"Response: {method} {path} - Status: {response.status_code} - Time: {process_time:.4f}s"
        )
        
        if self.log_level == "DEBUG":
            logger.debug(f"Request/Response details: {json.dumps(log_data, default=str)}")
        
        return response
