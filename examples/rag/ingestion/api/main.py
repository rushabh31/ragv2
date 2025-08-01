import os
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request
from dotenv import load_dotenv
load_dotenv()

from src.rag.shared.utils.config_manager import ConfigManager
from examples.rag.ingestion.api.router import router as ingestion_router
from src.rag.shared.middleware.middleware_factory import MiddlewareFactory
from src.rag.core.exceptions.exceptions import (
    DocumentProcessingError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    ConfigError,
    AuthenticationError,
    RateLimitError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ingestion-api")

# Load configuration
config_manager = ConfigManager()
config = config_manager.get_config()
app_config = config.get("api", {}).get("ingestion", {})

# Create FastAPI app
app = FastAPI(
    title="RAG Ingestion API",
    description="API for document ingestion, processing, and indexing",
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
app.include_router(ingestion_router)

# Exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_error_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(ChunkingError)
async def chunking_error_handler(request: Request, exc: ChunkingError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

@app.exception_handler(VectorStoreError)
async def vector_store_error_handler(request: Request, exc: VectorStoreError):
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
    return {"status": "ok", "service": "ingestion-api"}

if __name__ == "__main__":
    # Use environment manager for configuration
    from src.utils.env_manager import env
    
    port = env.get_int("PORT", 8000)
    log_level = env.get_string("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=app_config.get("reload", False)
    )
