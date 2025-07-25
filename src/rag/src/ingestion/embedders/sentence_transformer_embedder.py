"""Sentence Transformer embedder for generating embeddings."""

import logging
import os
from typing import Dict, List, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.rag.src.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.src.core.exceptions.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using Sentence Transformers library."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Sentence Transformer embedder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._model = None
        self._executor = None
        self._model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self._batch_size = config.get("batch_size", 32)
        self._max_workers = config.get("max_workers", 1)
    
    async def _init_model(self):
        """Initialize the Sentence Transformer model."""
        if self._model is None:
            try:
                # Create thread pool executor for running the model
                self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
                
                # Load the model - this happens in a separate thread to avoid blocking
                def load_model():
                    return SentenceTransformer(self._model_name)
                
                self._model = await asyncio.get_event_loop().run_in_executor(self._executor, load_model)
                logger.info(f"Initialized Sentence Transformer model: {self._model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Sentence Transformer model: {str(e)}", exc_info=True)
                raise EmbeddingError(f"Failed to initialize Sentence Transformer model: {str(e)}")
    
    async def _generate_embeddings(self, texts: List[str], config: Dict[str, Any] = None) -> List[List[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            config: Optional configuration parameters
            
        Returns:
            List of embedding vectors
        """
        return await self._embed_texts(texts)
        
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Internal method to generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        await self._init_model()
        
        try:
            # Skip empty batches
            if not texts:
                return []
            
            # Encode texts in batches using thread pool
            async def encode_batch(batch):
                return await asyncio.get_event_loop().run_in_executor(
                    self._executor, 
                    lambda: self._model.encode(batch).tolist()
                )
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i:i + self._batch_size]
                batch_embeddings = await encode_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    async def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
            logger.info("Sentence Transformer embedder resources released")
