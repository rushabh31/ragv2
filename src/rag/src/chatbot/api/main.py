import logging
import os
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

from src.rag.src.chatbot.api.router import router as chatbot_router
from src.rag.src.shared.utils.config_manager import ConfigManager
from src.rag.src.shared.middleware.middleware_factory import MiddlewareFactory
from src.rag.src.core.exceptions.exceptions import (
    RetrievalError,
    RerankerError,
    GenerationError,
    MemoryError,
    ConfigError,
    AuthenticationError,
    RateLimitError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chatbot-api")

# Load configuration
config_manager = ConfigManager()
config = config_manager.get_config()
app_config = config.get("api", {}).get("chatbot", {})

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for conversational retrieval and question answering",
    version="1.0.0"
)

# Configure middleware using the factory
middleware_config = {
    "cors": {
        "allow_origins": app_config.get("cors_origins", ["*"]),
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"]
    },
    "authentication": {
        "enabled": app_config.get("require_api_key", False),
        "api_keys": app_config.get("api_keys", []),
        "header_name": "X-API-Key",
        "allow_query_param": True,
        "exempt_paths": ["/docs", "/redoc", "/openapi.json", "/health"]
    },
    "rate_limiting": {
        "enabled": app_config.get("rate_limit", {}).get("enabled", False),
        "provider": app_config.get("rate_limit", {}).get("provider", "local"),
        "redis_url": app_config.get("rate_limit", {}).get("redis_url", "redis://localhost:6379/0"),
        "requests": app_config.get("rate_limit", {}).get("times", 100),
        "period_seconds": app_config.get("rate_limit", {}).get("seconds", 60),
        "exempted_paths": ["/docs", "/redoc", "/openapi.json", "/health"]
    },
    "logging": {
        "enabled": True,
        "log_request_body": False,
        "log_response_body": False,
        "log_headers": False
    }
}

# Apply all configured middleware
MiddlewareFactory.configure_app_middleware(app, middleware_config)

# Include routers
app.include_router(chatbot_router)

# Exception handlers
@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(RerankerError)
async def reranker_error_handler(request: Request, exc: RerankerError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": str(exc)}
    )

@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "chatbot-api"}

def main():
    """Entry point for the chatbot API server."""
    port = int(os.environ.get("PORT", 8001))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "src.rag.src.chatbot.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=app_config.get("reload", False)
    )

if __name__ == "__main__":
    main()
