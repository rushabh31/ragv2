import logging
import os
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

# Import chromadb conditionally to handle missing dependency gracefully
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.rag.src.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.src.core.interfaces.base import Chunk, SearchResult
from src.rag.src.core.exceptions.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Vector store using ChromaDB for efficient similarity search and management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ChromaDB vector store with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - dimension: Dimension of the embedding vectors
                - persist_directory: Directory to persist ChromaDB data
                - collection_name: Name of the ChromaDB collection
                - metadata_path: Optional path to save metadata backup
        """
        super().__init__(config)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with 'pip install chromadb'")
            
        self.dimension = self.config.get("dimension", 768)
        self.persist_directory = self.config.get("persist_directory", "./data/chroma_db")
        self.collection_name = self.config.get("collection_name", "document_embeddings")
        self.metadata_path = self.config.get("metadata_path")
        self.distance_function = self.config.get("distance_function", "cosine")
        
        # Will be initialized later
        self.client = None
        self.collection = None
        self.chunks = []  # Store chunk objects for retrieval
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the ChromaDB store.
        
        Creates the necessary client, collection, and loads any existing data.
        """
        if self._initialized:
            return
        
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.dimension},
                    embedding_function=None  # We provide our own embeddings
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
            # Try to load existing chunks from metadata if path provided
            if self.metadata_path and os.path.exists(self.metadata_path):
                await self._load_metadata()
            else:
                # Load chunks from ChromaDB
                await self._load_chunks_from_chroma()
                
            self._initialized = True
            logger.info(f"Initialized ChromaDB store with collection {self.collection_name}")
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def _load_metadata(self) -> None:
        """Load chunks metadata from disk."""
        try:
            with open(self.metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks from {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            # Fall back to loading from ChromaDB
            await self._load_chunks_from_chroma()
    
    async def _load_chunks_from_chroma(self) -> None:
        """Load chunks from ChromaDB collection."""
        try:
            # Get all items from the collection
            result = self.collection.get(include=['documents', 'metadatas'])
            
            self.chunks = []
            if result and 'ids' in result and result['ids']:
                for i, chunk_id in enumerate(result['ids']):
                    content = result['documents'][i] if 'documents' in result and i < len(result['documents']) else ""
                    metadata = result['metadatas'][i] if 'metadatas' in result and i < len(result['metadatas']) else {}
                    
                    # Create chunk object (without embeddings to save memory)
                    chunk = Chunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        embedding=None
                    )
                    self.chunks.append(chunk)
                    
                logger.info(f"Loaded {len(self.chunks)} chunks from ChromaDB collection")
            else:
                logger.info("No existing chunks found in ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to load chunks from ChromaDB: {str(e)}")
            # Initialize with empty chunks
            self.chunks = []
    
    async def _save_metadata(self) -> None:
        """Save chunks metadata to disk."""
        if not self.metadata_path:
            logger.warning("No metadata_path provided, skipping metadata save")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            # Save metadata
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Saved {len(self.chunks)} chunks metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
    
    async def _add_vectors(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add vectors to the ChromaDB store.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
            
        try:
            start_time = datetime.now()
            logger.info(f"Starting vector indexing process for {len(chunks)} chunks")
            
            # Extract embeddings from chunks if not provided
            if embeddings is None:
                logger.info("Extracting embeddings from chunks")
                embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
                if len(embeddings) != len(chunks):
                    raise VectorStoreError("Not all chunks have embeddings")
            
            # Ensure all vectors have the correct dimension
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.dimension:
                    logger.warning(f"Vector {i} has dimension {len(embedding)}, expected {self.dimension}")
                    raise VectorStoreError(f"Vector dimension mismatch: got {len(embedding)}, expected {self.dimension}")
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate chunk ID if not present
                if not hasattr(chunk, 'id') or not chunk.id:
                    chunk.id = f"chunk_{len(self.chunks) + i}_{uuid.uuid4()}"
                
                ids.append(chunk.id)
                documents.append(chunk.content)
                metadatas.append(chunk.metadata or {})
            
            # Add chunks to ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # Add or update chunks in our in-memory store (without embeddings to save memory)
            for i, chunk in enumerate(chunks):
                # Check if chunk already exists
                existing_idx = next((i for i, c in enumerate(self.chunks) if c.id == chunk.id), None)
                if existing_idx is not None:
                    # Update existing chunk
                    self.chunks[existing_idx] = Chunk(
                        id=chunk.id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        embedding=None  # Don't store embeddings in memory
                    )
                else:
                    # Add new chunk
                    self.chunks.append(Chunk(
                        id=chunk.id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        embedding=None  # Don't store embeddings in memory
                    ))
            
            # Save metadata backup
            await self._save_metadata()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Added {len(chunks)} chunks to ChromaDB store in {duration:.2f} seconds")
        except Exception as e:
            error_msg = f"Failed to add vectors to ChromaDB store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def _search_vectors(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search vectors in the ChromaDB store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate query embedding dimension
            if len(query_embedding) != self.dimension:
                raise VectorStoreError(f"Query dimension mismatch: got {len(query_embedding)}, expected {self.dimension}")
                
            # Execute vector similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            
            # Process results
            if results and 'ids' in results and results['ids']:
                ids = results['ids'][0]  # First query's results
                distances = results['distances'][0]  # First query's distances
                documents = results['documents'][0]  # First query's documents
                metadatas = results['metadatas'][0]  # First query's metadatas
                
                for i, chunk_id in enumerate(ids):
                    # Convert distance to similarity score (ChromaDB returns distances, not similarities)
                    # For cosine distance, similarity = 1 - distance
                    similarity = 1.0 - float(distances[i])
                    
                    content = documents[i] if i < len(documents) else ""
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    # Create search result
                    search_result = SearchResult(
                        chunk=Chunk(
                            id=chunk_id,
                            content=content,
                            metadata=metadata,
                            embedding=None  # We don't include embeddings in search results
                        ),
                        similarity=similarity
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def add(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings. If not provided, embeddings 
                       from the chunks will be used if available.
        
        Raises:
            VectorStoreError: If adding vectors fails
        """
        if not self._initialized:
            await self.initialize()
            
        await self._add_vectors(chunks, embeddings)
    
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search vectors in the store by similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
            
        Raises:
            VectorStoreError: If search fails
        """
        if not self._initialized:
            await self.initialize()
            
        return await self._search_vectors(query_embedding, top_k)
    
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks from the store by ID.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Raises:
            VectorStoreError: If deletion fails
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Delete from ChromaDB
            self.collection.delete(ids=chunk_ids)
            
            # Delete from in-memory store
            self.chunks = [chunk for chunk in self.chunks if chunk.id not in chunk_ids]
            
            # Save updated metadata
            await self._save_metadata()
            
            logger.info(f"Deleted {len(chunk_ids)} chunks from ChromaDB store")
        except Exception as e:
            error_msg = f"Failed to delete chunks from ChromaDB store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
