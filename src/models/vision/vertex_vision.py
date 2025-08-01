"""
Vertex AI Vision Model with Universal Authentication.

This module provides a standardized interface for Vertex AI vision models
using the universal authentication system.
"""

import asyncio
import base64
import logging
import os
from typing import Dict, Any, Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, Image

from src.utils.auth_manager import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class VertexVisionAI:
    """Vertex AI Vision model with universal authentication."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        max_pages: Optional[int] = None,
        max_concurrent_pages: Optional[int] = None,
        **kwargs
    ):
        """Initialize the Vertex AI Vision model.
        
        Args:
            model_name: Name of the vision model to use (overrides config)
            project_id: GCP project ID (overrides env var)
            location: GCP location for the model (overrides config)
            max_pages: Maximum pages to process (overrides config)
            max_concurrent_pages: Max concurrent pages (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        vision_config = config_manager.get_section("vision", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or vision_config.get("model", "gemini-1.5-pro-002")
        # Use environment manager for configuration
        from src.utils.env_manager import env
        self.project_id = project_id or env.get_string("PROJECT_ID")
        self.location = location or vision_config.get("location", "us-central1")
        self.max_pages = max_pages or vision_config.get("max_pages", 100)
        self.max_concurrent_pages = max_concurrent_pages or vision_config.get("max_concurrent_pages", 5)
        
        # Store additional config parameters
        self.config = {**vision_config, **kwargs}
        
        # Initialize universal auth manager
        self.auth_manager = UniversalAuthManager(f"vertex_vision_{self.model_name}")
        self.auth_manager.configure()
        
        # Model will be initialized lazily
        self._model = None
        self._initialized = False
        
        logger.info(f"Initialized VertexVisionAI with model: {self.model_name} (from config)")
    
    async def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager.
        
        Returns:
            Authentication token or None if unavailable
        """
        try:
            return await self.auth_manager.get_token("vertex_ai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def get_auth_health_status(self) -> Dict[str, Any]:
        """Get authentication health status.
        
        Returns:
            Dictionary containing health status information
        """
        token = await self.get_coin_token()
        return {
            "service": "vertex_ai_vision",
            "model_name": self.model_name,
            "status": "healthy" if token else "unhealthy",
            "token_available": bool(token)
        }
    
    async def validate_authentication(self) -> bool:
        """Validate Vertex AI authentication.
        
        Returns:
            True if authentication is valid, False otherwise
        """
        try:
            token = await self.get_coin_token()
            if not token:
                logger.error("Vertex AI token not available")
                return False
            
            # Initialize if needed to test authentication
            await self._ensure_initialized()
            logger.info("Vertex AI vision authentication validated successfully")
            return True
        except Exception as e:
            logger.error(f"Vertex AI vision authentication validation error: {str(e)}")
            return False
    
    async def _ensure_initialized(self) -> None:
        """Ensure the model is initialized with proper authentication."""
        if self._initialized:
            return
        
        try:
            # Ensure we have a project ID
            if not self.project_id:
                raise ValueError("Project ID is required. Set PROJECT_ID environment variable or pass project_id parameter.")
            
            # Set SSL certificate path if provided
            ssl_cert_path = env.get_string("SSL_CERT_FILE")
            if ssl_cert_path:
                env.set("SSL_CERT_FILE", ssl_cert_path)
                logger.info(f"Using SSL certificate from: {ssl_cert_path}")
            
            # Initialize Vertex AI with authenticated credentials
            credentials = await self.auth_manager.get_credentials()
            vertexai.init(
                project=self.project_id,
                location=self.location,
                api_transport="rest",  # uses UAT PROJECT
                api_endpoint=env.get_string("VERTEXAI_API_ENDPOINT"),  # uses R2D2 UAT
                credentials=credentials
            )
            
            # Set metadata for user tracking
            self.metadata = [("x-r2d2-user", os.getenv("USERNAME", ""))]
            
            # Initialize the generative model
            self._model = GenerativeModel(self.model_name)
            self._initialized = True
            
            logger.info(f"Successfully initialized Vertex AI Vision model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI Vision model: {str(e)}")
            raise
    
    async def parse_text_from_image(
        self,
        base64_encoded: str,
        prompt: str,
        **kwargs
    ) -> str:
        """Parse text from a base64-encoded image using vision model.
        
        Args:
            base64_encoded: Base64-encoded image data
            prompt: Text prompt for the vision model
            **kwargs: Additional generation parameters
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If vision processing fails
        """
        try:
            await self._ensure_initialized()
            
            # Create content with text and optional image
            parts = []
            if prompt:
                parts.append(Part.from_text(prompt))
            
            # Add image if provided
            if base64_encoded:
                # Determine mime type based on image format (default to PNG)
                mime_type = kwargs.get("mime_type", "image/png")
                parts.append(Part.from_data(
                    data=base64.b64decode(base64_encoded),
                    mime_type=mime_type
                ))
            
            # Create contents list with user role
            contents = [Content(role="user", parts=parts)]
            
            # Set default generation config if not provided
            generation_config = {
                "temperature": kwargs.get("temperature", 0.7),
                "max_output_tokens": kwargs.get("max_output_tokens", 2048),
                "top_p": kwargs.get("top_p", 1.0),
                "top_k": kwargs.get("top_k", 40)
            }
            
            # Make the API call with timeout
            timeout = kwargs.get("timeout", 60)  # Default 60 second timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self._model.generate_content(
                        contents,
                        generation_config=generation_config
                    )
                ),
                timeout=timeout
            )
            
            # Extract text from response
            if response.text:
                logger.info("Successfully extracted text from image using vision model")
                return response.text.strip()
            else:
                logger.warning("Vision model returned empty response")
                return ""
                
        except Exception as e:
            logger.error(f"Vision text extraction failed: {str(e)}")
            raise
    
    async def analyze_image(
        self,
        base64_encoded: str,
        analysis_prompt: str,
        **kwargs
    ) -> str:
        """Analyze an image with a custom prompt.
        
        Args:
            base64_encoded: Base64-encoded image data
            analysis_prompt: Custom analysis prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Analysis result
        """
        return await self.parse_text_from_image(
            base64_encoded=base64_encoded,
            prompt=analysis_prompt,
            **kwargs
        )
    
    async def extract_structured_data(
        self,
        base64_encoded: str,
        schema_prompt: str,
        **kwargs
    ) -> str:
        """Extract structured data from an image.
        
        Args:
            base64_encoded: Base64-encoded image data
            schema_prompt: Prompt describing the desired data structure
            **kwargs: Additional generation parameters
            
        Returns:
            Structured data as text
        """
        return await self.parse_text_from_image(
            base64_encoded=base64_encoded,
            prompt=schema_prompt,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "provider": "vertex_ai",
            "type": "vision",
            "initialized": self._initialized,
            "auth_health": self.get_auth_health_status()
        }


# Convenience functions for easy model creation
def create_vertex_vision_model(
    model_name: str = "gemini-1.5-pro-002",
    **kwargs
) -> VertexVisionAI:
    """Create a Vertex AI vision model with default settings.
    
    Args:
        model_name: Name of the vision model
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured VertexVisionAI instance
    """
    return VertexVisionAI(model_name=model_name, **kwargs)


async def example_usage():
    """Example of how to use VertexVisionAI."""
    # Create vision model
    vision_model = VertexVisionAI()
    
    # Check authentication
    token = vision_model.get_coin_token()
    print(f"Authentication token available: {'Yes' if token else 'No'}")
    
    # Example image analysis (you would provide actual base64 image data)
    # base64_image = "your_base64_encoded_image_here"
    # prompt = "Extract all text from this document and format it as markdown."
    # result = await vision_model.parse_text_from_image(base64_image, prompt)
    # print(f"Extracted text: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
