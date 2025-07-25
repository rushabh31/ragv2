"""
Vertex AI Vision Model with Universal Authentication.

This module provides a standardized interface for Vertex AI vision models
using the universal authentication system.
"""

import base64
import logging
from typing import Dict, Any, Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from src.utils.auth_manager import UniversalAuthManager

logger = logging.getLogger(__name__)


class VertexVisionAI:
    """Vertex AI Vision model with universal authentication."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro-002",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs
    ):
        """Initialize the Vertex AI Vision model.
        
        Args:
            model_name: Name of the vision model to use
            project_id: GCP project ID (will be retrieved from auth manager if not provided)
            location: GCP location for the model
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.config = kwargs
        
        # Initialize universal auth manager
        self._auth_manager = UniversalAuthManager.get_instance("vertex_ai")
        
        # Model will be initialized lazily
        self._model = None
        self._initialized = False
        
        logger.info(f"Initialized VertexVisionAI with model: {model_name}")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager.
        
        Returns:
            Authentication token or None if unavailable
        """
        try:
            return self._auth_manager.get_token()
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get authentication health status.
        
        Returns:
            Dictionary containing authentication health information
        """
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication status.
        
        Returns:
            True if authentication is valid, False otherwise
        """
        try:
            token = self.get_coin_token()
            if not token:
                return False
            
            # Initialize if needed to test authentication
            await self._ensure_initialized()
            return True
        except Exception as e:
            logger.error(f"Authentication validation failed: {str(e)}")
            return False
    
    async def _ensure_initialized(self) -> None:
        """Ensure the model is initialized with proper authentication."""
        if self._initialized:
            return
        
        try:
            # Get project ID from auth manager if not provided
            if not self.project_id:
                self.project_id = self._auth_manager.get_project_id()
            
            # Initialize Vertex AI with authenticated credentials
            credentials = self._auth_manager.get_credentials()
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )
            
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
            
            # Create image part from base64 data
            image_part = Part.from_data(
                data=base64.b64decode(base64_encoded),
                mime_type="image/png"
            )
            
            # Generate content with image and prompt
            response = await self._model.generate_content_async(
                [prompt, image_part],
                generation_config={
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.8),
                    "top_k": kwargs.get("top_k", 40),
                    "max_output_tokens": kwargs.get("max_output_tokens", 8192)
                }
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
