"""
Azure OpenAI Generation Model.

This module provides a standardized interface for Azure OpenAI models
using the universal authentication system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken

from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AzureOpenAIGenAI:
    """
    Azure OpenAI generation model with universal authentication.
    
    This class provides a standardized interface for Azure OpenAI models
    using the universal authentication system.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 api_version: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 **kwargs):
        """
        Initialize Azure OpenAI client.
        
        Args:
            model_name: Name of the Azure OpenAI model to use (overrides config)
            azure_endpoint: Azure OpenAI endpoint URL (overrides config)
            api_version: API version to use (overrides config)
            temperature: Temperature for generation (overrides config)
            max_tokens: Max tokens for generation (overrides config)
            top_p: Top-p for generation (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        generation_config = config_manager.get_section("generation", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or generation_config.get("model", "GPT4-o")
        self.azure_endpoint = azure_endpoint or generation_config.get("azure_endpoint")
        self.api_version = api_version or generation_config.get("api_version", "2023-05-15")
        self.temperature = temperature or generation_config.get("temperature", 0.2)
        self.max_tokens = max_tokens or generation_config.get("max_tokens", 1024)
        self.top_p = top_p or generation_config.get("top_p", 0.95)
        
        # Store additional config parameters
        self.config = {**generation_config, **kwargs}
        
        self._client = None
        self._auth_manager = UniversalAuthManager(f"azure_openai_{self.model_name}")
        self._auth_manager.configure()
        
        logger.info(f"Initialized AzureOpenAIGenAI with model: {self.model_name} (from config)")
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager."""
        try:
            return self._auth_manager.get_token("azure_openai")
        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            return None
    
    async def _init_client(self):
        """Initialize Azure OpenAI client lazily."""
        if self._client is None:
            try:
                # Set SSL certificate path
                os.environ["SSL_CERT_FILE"] = "Path\\to\\ca-bundle.cer"
                # Ensure you have the correct path to your SSL certificate bundle
                
                # Get authentication token
                token = self.get_coin_token()
                if not token:
                    raise ValueError("Failed to get authentication token")
                
                # Initialize Azure OpenAI client
                self._client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token_provider=lambda: token,
                    api_version=self.api_version,
                )
                
                logger.info(f"Initialized Azure OpenAI client for {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
                raise
    
    async def generate_content(self,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.2,
                             stream: bool = False,
                             max_tokens: Optional[int] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Generate content using Azure OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        await self._init_client()
        
        try:
            # Make the API call
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                user=os.getenv("USERNAME", ""),
                temperature=temperature,
                stream=stream,
                max_tokens=max_tokens,
                **kwargs
            )
            
            logger.info(f"Generated content using {self.model_name}")
            return completion
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise
    
    async def chat_completion(self,
                            prompt: str,
                            system_message: Optional[str] = None,
                            temperature: float = 0.2,
                            stream: bool = False,
                            max_tokens: Optional[int] = None,
                            **kwargs) -> str:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Temperature for generation
            stream: Whether to stream the response
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.generate_content(
            messages=messages,
            temperature=temperature,
            stream=stream,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content if response.choices else ""
    
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
                              max_tokens: Optional[int] = None,
                              **kwargs) -> str:
        """
        Generate a response for RAG use case with documents and conversation history.
        
        Args:
            query: User query string
            documents: List of relevant documents with 'content' field
            conversation_history: Optional conversation history with 'role' and 'content' fields
            temperature: Temperature for generation (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
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
            effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Create messages for Azure OpenAI API
            messages = [{"role": "user", "content": full_prompt}]
            
            # Generate response using Azure OpenAI API
            response = await self.generate_content(
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
                **kwargs
            )
            
            logger.info(f"Generated RAG response using {self.model_name}")
            return response.choices[0].message.content if response.choices else ""
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {str(e)}")
            raise


# Example usage
async def example_usage():
    """Example of how to use AzureOpenAIGenAI."""
    
    # Initialize client
    client = AzureOpenAIGenAI()
    
    try:
        # Simple chat completion
        response = await client.chat_completion(
            prompt="Explain the difference between a bank and a credit union.",
            system_message="You are an expert on financial institutions.",
            temperature=0.2,
            stream=False
        )
        print(f"Response: {response}")
        
        # Advanced message format
        messages = [
            {"role": "system", "content": "You are an expert on financial institutions."},
            {"role": "user", "content": "Explain the difference between a bank and a credit union."}
        ]
        
        full_response = await client.generate_content(
            messages=messages,
            temperature=0.2,
            stream=False
        )
        print(f"Full Response: {full_response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
