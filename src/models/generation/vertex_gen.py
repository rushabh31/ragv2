"""
Vertex AI Generation Model.

This module provides a standardized interface for Google Vertex AI generative models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
from vertexai.generative_models import GenerativeModel
import vertexai

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class VertexGenAI:
    """
    Vertex AI generation model with universal authentication.
    
    This class provides a standardized interface for Google Vertex AI generative models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs):
        """
        Initialize Vertex AI client.
        
        Args:
            model_name: Name of the Vertex AI model to use (overrides config)
            project_id: GCP project ID (overrides env var)
            location: GCP location for Vertex AI (overrides config)
            temperature: Temperature for generation (overrides config)
            max_output_tokens: Max output tokens (overrides config)
            top_p: Top-p for generation (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        generation_config = config_manager.get_section("generation", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or generation_config.get("model", "gemini-1.5-pro-002")
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        self.location = location or generation_config.get("location", "us-central1")
        self.temperature = temperature or generation_config.get("temperature", 0.2)
        self.max_output_tokens = max_output_tokens or generation_config.get("max_output_tokens", 1024)
        self.top_p = top_p or generation_config.get("top_p", 0.95)
        
        # Store additional config parameters
        self.config = {**generation_config, **kwargs}
        
        self._model = None
        self._auth_manager = UniversalAuthManager(f"vertex_gen_{self.model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized VertexGenAI with model: {self.model_name} (from config)")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("vertex_ai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_model(self):
        """Initialize Vertex AI model lazily."""
        if self._model is None:
            try:
                # Get credentials using universal auth manager
                credentials = await self._auth_manager.get_credentials()
                
                # Initialize Vertex AI
                vertexai.init(
                    project=self.project_id,
                    api_transport="rest",  # uses UAT PROJECT
                    api_endpoint=os.environ.get("VERTEXAI_API_ENDPOINT"),  # uses R2D2 UAT
                    credentials=credentials,
                )
                
                # Set metadata for user tracking
                self.metadata = [("x-r2d2-user", os.getenv("USERNAME", ""))]
                
                # Initialize the generative model
                self._model = GenerativeModel(self.model_name)
                
                logger.info(f"Initialized Vertex AI model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI model: {str(e)}")
                raise
    
    async def generate_content(self,
                         prompt: str,
                         temperature: Optional[float] = None,
                         max_output_tokens: Optional[int] = None,
                         top_p: Optional[float] = None,
                         top_k: Optional[int] = None,
                         **kwargs) -> Any:
        """
        Generate content using Vertex AI.
        
        Args:
            prompt: Text prompt for generation
            temperature: Temperature for generation (uses config default if None)
            max_output_tokens: Maximum tokens to generate (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        await self._init_model()
        
        try:
            # Use provided values or fall back to instance defaults from config
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
            effective_top_p = top_p if top_p is not None else self.top_p
            
            # Set generation config
            generation_config = {
                "temperature": effective_temperature,
                "max_output_tokens": effective_max_tokens,
                "top_p": effective_top_p,
                "top_k": top_k or 40,
                **kwargs
            }
            
            # Generate content
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config=generation_config,
                metadata=self.metadata,
            )
            
            logger.info(f"Generated content using {self.model_name}")
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self,
                        messages: Optional[List[Dict[str, Any]]] = None,
                        prompt: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_output_tokens: Optional[int] = None,
                        **kwargs) -> str:
        """
        Chat completion interface with message history support.
        
        Args:
            messages: List of message dictionaries with 'role' and 'parts' keys
            prompt: Simple prompt string (alternative to messages)
            temperature: Temperature for generation (uses config default if None)
            max_output_tokens: Maximum tokens to generate (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        await self._init_model()
        
        try:
            # Use provided values or fall back to instance defaults from config
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
            
            # Set generation config
            generation_config = {
                "temperature": effective_temperature,
                "max_output_tokens": effective_max_tokens,
                "top_p": self.top_p,
                **kwargs
            }
            
            if messages:
                # Use chat with message history
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    messages,
                    generation_config=generation_config,
                    metadata=self.metadata,
                )
            elif prompt:
                # Use simple prompt
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    prompt,
                    generation_config=generation_config,
                    metadata=self.metadata,
                )
            else:
                raise ValueError("Either 'messages' or 'prompt' must be provided")
            
            logger.info(f"Generated chat completion using {self.model_name}")
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get health status of the authentication system."""
        return self._auth_manager.get_health_status()
    
    async def validate_authentication(self) -> bool:
        """Validate authentication by testing token acquisition."""
        return await self._auth_manager.validate_authentication()
    
    async def generate_response(self,
                              query: str,
                              documents: Optional[List[Dict[str, Any]]] = None,
                              conversation_history: Optional[List[Dict[str, str]]] = None,
                              temperature: Optional[float] = None,
                              max_output_tokens: Optional[int] = None,
                              **kwargs) -> str:
        """
        Generate a response for RAG use case with documents and conversation history.
        
        Args:
            query: User query string
            documents: List of relevant documents with 'content' field
            conversation_history: Optional conversation history with 'role' and 'content' fields
            temperature: Temperature for generation (uses config default if None)
            max_output_tokens: Maximum tokens to generate (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        await self._init_model()
        
        try:
            # Build context from documents
            context = ""
            if documents:
                context = "\n\nContext:\n"
                for i, doc in enumerate(documents, 1):
                    content = doc.get('content', str(doc)) if isinstance(doc, dict) else str(doc)
                    context += f"Document {i}:\n{content}\n\n"
            
            # Build conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n\nPrevious conversation:\n"
                for msg in conversation_history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
            
            # Create the full prompt
            full_prompt = f"""
{context}{history_text}
Question: {query}

Instructions:
1. Answer the question based on the provided context and conversation history.
2. If the context doesn't contain enough information, say "I don't have enough information to answer that question."
3. Provide specific references to the context when possible.
4. Be concise and accurate.

Answer:
"""
            
            # Use provided values or fall back to instance defaults from config
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
            
            # Set generation config
            generation_config = {
                "temperature": effective_temperature,
                "max_output_tokens": effective_max_tokens,
                "top_p": self.top_p,
                **kwargs
            }
            
            # Generate response
            response = await asyncio.to_thread(
                self._model.generate_content,
                full_prompt,
                generation_config=generation_config,
                metadata=self.metadata,
            )
            
            logger.info(f"Generated RAG response using {self.model_name}")
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {str(e)}")
            raise


# Example usage
async def example_usage():
    """Example of how to use VertexGenAI."""
    
    # Set environment variables for testing
    os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem files>"
    
    # Initialize client
    vertex_gen_ai = VertexGenAI()
    
    try:
        # Simple chat completion
        response = await vertex_gen_ai.chat_completion("Provide interesting trivia.")
        print(f"Response: {response}")
        
        # Advanced generation
        full_response = await vertex_gen_ai.generate_content(
            prompt="Provide interesting trivia.",
            temperature=0.8,
            max_output_tokens=1024
        )
        print(f"Full Response: {full_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
