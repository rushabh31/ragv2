"""
Sentence Transformer Embedding Model with Universal Authentication Support.

This module provides a local sentence transformer-based embedding model that integrates 
with the universal authentication system and supports batch embedding operations.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import UniversalAuthManager
from src.rag.shared.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingAI:
    """Sentence Transformer Embedding model with universal authentication."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 device: Optional[str] = None,
                 normalize_embeddings: Optional[bool] = None,
                 **kwargs):
        """Initialize Sentence Transformer Embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use (overrides config)
            batch_size: Batch size for embedding requests (overrides config)
            device: Device to use for computation (overrides config)
            normalize_embeddings: Whether to normalize embeddings (overrides config)
            **kwargs: Additional configuration parameters
        """
        # Load configuration from ConfigManager
        config_manager = ConfigManager()
        embedding_config = config_manager.get_section("embedding", {})
        
        # Use provided values or fall back to config, then to defaults
        self.model_name = model_name or embedding_config.get("model", "all-mpnet-base-v2")
        self.batch_size = batch_size or embedding_config.get("batch_size", 32)
        self.device = device or embedding_config.get("device", "cpu")
        self.normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else embedding_config.get("normalize_embeddings", True)
        
        # Store additional config parameters
        self.config = {**embedding_config, **kwargs}
        
        # Initialize universal auth manager
        self.auth_manager = UniversalAuthManager()
        
        # Model will be loaded lazily
        self._model = None
        
        logger.info(f"Initialized SentenceTransformerEmbeddingAI with model: {self.model_name} (from config)")
    
    def _load_model(self):
        """Load the sentence transformer model lazily."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Successfully loaded model on device: {self.device}")
                
            except ImportError:
                error_msg = "sentence-transformers package not installed. Install with: pip install sentence-transformers"
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                error_msg = f"Failed to load sentence transformer model: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    def get_coin_token(self) -> Optional[str]:
        """Get authentication token using universal auth manager.
        
        Note: Sentence transformers run locally, so this is mainly for consistency.
        
        Returns:
            Authentication token or None if unavailable
        """
        try:
            return self.auth_manager.get_token("sentence_transformer")
        except Exception as e:
            logger.debug(f"Sentence transformer token not available (expected for local model): {str(e)}")
            return "local_model_token"  # Return placeholder for local model
    
    async def validate_authentication(self) -> bool:
        """Validate sentence transformer model availability.
        
        Returns:
            True if model can be loaded, False otherwise
        """
        try:
            self._load_model()
            logger.info("Sentence transformer model validation successful")
            return True
        except Exception as e:
            logger.error(f"Sentence transformer model validation failed: {str(e)}")
            return False
    
    def get_auth_health_status(self) -> Dict[str, Any]:
        """Get model health status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            model_loaded = self._model is not None
            if not model_loaded:
                # Try to load model to check availability
                self._load_model()
                model_loaded = True
        except Exception:
            model_loaded = False
        
        return {
            "service": "sentence_transformer",
            "model_name": self.model_name,
            "model_loaded": model_loaded,
            "device": self.device,
            "status": "healthy" if model_loaded else "unhealthy"
        }
    
    async def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Load model if not already loaded
            self._load_model()
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100  # Show progress for large batches
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings with dimension {len(embeddings_list[0])}")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Sentence transformer embedding generation failed: {str(e)}")
            raise
    
    async def embed_single(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            embeddings = await self.get_embeddings([text], **kwargs)
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Single text embedding failed: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        try:
            self._load_model()
            return self._model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {str(e)}")
            # Return common dimension for all-mpnet-base-v2
            return 768 if "mpnet" in self.model_name.lower() else 384
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            self._load_model()
            return {
                "model_name": self.model_name,
                "embedding_dimension": self.get_embedding_dimension(),
                "device": self.device,
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize_embeddings,
                "max_seq_length": getattr(self._model, 'max_seq_length', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {
                "model_name": self.model_name,
                "error": str(e)
            }
