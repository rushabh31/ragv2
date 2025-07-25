"""OpenAI-based embedder for generating text embeddings."""

import logging
import asyncio
import aiohttp
import json
import requests
from typing import Dict, Any, List, Optional

from src.rag.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    """Embedder that uses OpenAI embedding models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the embedder with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - model: Name of the embedding model to use (default: text-embedding-3-large)
                - batch_size: Number of texts to embed in a single API call
                - api_endpoint: OpenAI API endpoint URL
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "text-embedding-3-large")
        self.batch_size = self.config.get("batch_size", 50)
        self.api_endpoint = self.config.get("api_endpoint", "https://api.openai.com/v1/embeddings")
        
        # Load configuration for token-based authentication
        config_manager = ConfigManager()
        self.vertex_config = config_manager.get_config("vertex")
        
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token for OpenAI API access using COIN token provider.
        
        Returns:
            Authentication token or None if token retrieval fails
        """
        if not self.vertex_config:
            logger.warning("Vertex configuration not found. Cannot retrieve token.")
            return None
            
        url = self.vertex_config.get("COIN_CONSUMER_ENDPOINT_URL")
        if not url:
            logger.error("COIN_CONSUMER_ENDPOINT_URL not found in configuration")
            return None
            
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "scope": self.vertex_config.get("COIN_CONSUMER_SCOPE"),
            "client_id": self.vertex_config.get("COIN_CONSUMER_CLIENT_ID"),
            "client_secret": self.vertex_config.get("COIN_CONSUMER_CLIENT_SECRET"),
        }
        
        logger.info("Requesting API token for OpenAI embeddings")
        
        try:
            response = requests.post(url, headers=headers, data=body, verify=False, timeout=10)
            
            if response.status_code == 200:
                logger.info("Token received successfully for OpenAI embeddings")
                return response.json().get('access_token', None)
            else:
                logger.error(f"Failed to get token! Status code: {response.status_code}, Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error requesting token: {str(e)}")
            return None
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings using OpenAI embedding model.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for embedding
            
        Returns:
            List of embedding vectors
        """
        # Get configuration
        batch_size = config.get("batch_size", self.batch_size)
        api_endpoint = config.get("api_endpoint", self.api_endpoint)
        model = config.get("model", self.model_name)
        
        # Get API token for authentication
        api_key = self.get_coin_token()
        if not api_key:
            api_key = config.get("api_key")
            if not api_key:
                logger.error("No API key available for OpenAI embeddings")
                raise ValueError("OpenAI API key not available")
        
        # Process texts in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._embed_batch(batch, api_endpoint, model, api_key)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    async def _embed_batch(self, texts: List[str], api_endpoint: str, model: str, api_key: str) -> List[List[float]]:
        """Embed a batch of texts using the OpenAI API.
        
        Args:
            texts: Batch of texts to embed
            api_endpoint: API endpoint URL
            model: Model name to use
            api_key: API key or token for authentication
            
        Returns:
            List of embedding vectors for the batch
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "input": texts,
            "model": model,
            "encoding_format": "float"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, json=payload, headers=headers, ssl=False) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise ValueError(f"OpenAI API returned error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    # Extract embeddings from the response
                    embeddings = []
                    for item in result.get("data", []):
                        embeddings.append(item.get("embedding", []))
                    
                    # Ensure we got the expected number of embeddings
                    if len(embeddings) != len(texts):
                        logger.warning(f"Expected {len(texts)} embeddings but received {len(embeddings)}")
                    
                    return embeddings
                    
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {str(e)}")
            raise ValueError(f"Failed to generate OpenAI embeddings: {str(e)}")
