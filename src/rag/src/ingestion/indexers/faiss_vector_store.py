import logging
import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
from pathlib import Path

from src.rag.src.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.src.core.interfaces.base import Chunk, SearchResult
from src.rag.src.core.exceptions.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class FAISSVectorStore(BaseVectorStore):
    """Vector store using FAISS for efficient similarity search."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the FAISS vector store with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - dimension: Dimension of the embedding vectors
                - index_type: FAISS index type (e.g., "HNSW", "Flat", "IVF")
                - index_path: Path to save/load the index
                - metadata_path: Path to save/load the metadata
        """
        super().__init__(config)
        self.dimension = self.config.get("dimension", 768)
        self.index_type = self.config.get("index_type", "HNSW")
        self.index_path = self.config.get("index_path")
        self.metadata_path = self.config.get("metadata_path")
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []  # Store chunk objects for retrieval
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the FAISS index.
        
        This must be called before adding or searching vectors.
        """
        if self._initialized:
            return
        
        try:
            # Try to load existing index if paths are provided
            if self.index_path and os.path.exists(self.index_path) and self.metadata_path and os.path.exists(self.metadata_path):
                await self._load_index()
            else:
                # Create new index
                await self._create_index()
            
            self._initialized = True
            logger.info(f"Initialized FAISS index of type {self.index_type}")
        except Exception as e:
            error_msg = f"Failed to initialize FAISS index: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def _create_index(self) -> None:
        """Create a new FAISS index based on the configured type."""
        logger.info(f"Creating FAISS index with dimension {self.dimension}")
        if self.index_type == "Flat":
            # Exact search
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors per node
        elif self.index_type == "IVF":
            # Inverted File index
            quantizer = faiss.IndexFlatIP(self.dimension)
            n_cells = 100  # Number of centroids
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_cells, faiss.METRIC_INNER_PRODUCT)
            # Need to train this index with some vectors before using
            self.index_needs_training = True
        else:
            # Default to flat index
            logger.warning(f"Unknown index type {self.index_type}, using Flat index")
            self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created FAISS {self.index_type} index with dimension {self.dimension}")
    
    async def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            # Create new index as fallback
            await self._create_index()
    
    async def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if not self.index_path or not self.metadata_path:
            logger.warning("No index_path or metadata_path provided, skipping save")
            return
        
        try:
            # Create directories if they don't exist
            for path in [self.index_path, self.metadata_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Saved FAISS index with {len(self.chunks)} chunks to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
    
    async def _add_vectors(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add vectors to the FAISS index.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings
        """
        try:
            logger.info("Starting vector indexing process")
            
            if not self._initialized:
                logger.info("Initializing FAISS index")
                await self.initialize()
                logger.info("FAISS index initialized")
            
            # Get embeddings from chunks if not provided
            if embeddings is None:
                logger.info("Extracting embeddings from chunks")
                embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
                # Filter chunks to match embeddings
                chunks = [chunk for chunk in chunks if chunk.embedding]
                logger.info(f"Extracted {len(embeddings)} embeddings from chunks")
            
            if not embeddings:
                logger.warning("No embeddings provided and none found in chunks")
                return
            
            # Convert embeddings to numpy array
            logger.info("Converting embeddings to numpy array")
            vectors = np.array(embeddings).astype('float32')
            logger.info(f"Converted embeddings to array of shape {vectors.shape}")
            
            # Check if dimensions match and recreate index if needed
            actual_dim = vectors.shape[1]
            if self.index is not None and actual_dim != self.dimension:
                logger.warning(f"Dimension mismatch: index has {self.dimension} dimensions, but embeddings have {actual_dim} dimensions")
                logger.info(f"Recreating index with correct dimension {actual_dim}")
                self.dimension = actual_dim
                await self._create_index()
            
            # Normalize vectors for inner product
            logger.info("Normalizing vectors using custom implementation")
            try:
                # Custom normalization to avoid FAISS OpenMP issues
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms == 0] = 1.0
                vectors = vectors / norms
                logger.info("Vector normalization completed")
            except Exception as e:
                logger.error(f"Error during normalization: {str(e)}")
                # Fallback to simpler implementation if needed
                for i in range(len(vectors)):
                    norm = np.linalg.norm(vectors[i])
                    if norm > 0:
                        vectors[i] = vectors[i] / norm
            
            # Train index if needed (IVF requires training)
            if hasattr(self, 'index_needs_training') and self.index_needs_training and self.index.ntotal == 0:
                logger.info("Training FAISS index")
                self.index.train(vectors)
                self.index_needs_training = False
                logger.info("FAISS index trained")
            
            # Add vectors to index
            logger.info(f"Adding {len(vectors)} vectors to FAISS index")
            import time
            start_time = time.time()
            
            # Add vectors in smaller batches to prevent hanging
            batch_size = 100  # Add vectors in smaller batches
            for i in range(0, len(vectors), batch_size):
                end_idx = min(i + batch_size, len(vectors))
                logger.info(f"Adding batch of vectors {i}-{end_idx}")
                self.index.add(vectors[i:end_idx])
            
            # Store chunks for retrieval
            self.chunks.extend(chunks)
            logger.info(f"Added {len(chunks)} chunks to metadata store")
            
            # Save index and metadata
            logger.info(f"Saving FAISS index with {len(self.chunks)} total chunks")
            await self._save_index()
            elapsed_time = time.time() - start_time
            logger.info(f"Vector indexing completed in {elapsed_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error in vector indexing: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to add vectors to index: {str(e)}")
    
    async def _search_vectors(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search for similar vectors in the FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
        """
        if not self._initialized:
            await self.initialize()
        
        # Check if index is empty
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no results to return")
            return []
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize query vector using custom implementation
        try:
            # Custom normalization to avoid FAISS OpenMP issues
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            logger.info("Query vector normalized")
        except Exception as e:
            logger.error(f"Error normalizing query vector: {str(e)}")
            # Simple fallback
            for i in range(len(query_vector)):
                norm = np.linalg.norm(query_vector[i])
                if norm > 0:
                    query_vector[i] = query_vector[i] / norm
        
        # Search index
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Create search results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue  # Skip invalid indices
            
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(score)
            ))
        
        return results
