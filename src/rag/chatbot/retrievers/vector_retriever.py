import logging
from typing import Dict, Any, List, Optional

from src.rag.chatbot.retrievers.base_retriever import BaseRetriever
from src.rag.core.interfaces.base import Document
from src.rag.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.ingestion.indexers.faiss_vector_store import FAISSVectorStore
from src.rag.ingestion.indexers.pgvector_store import PgVectorStore
from src.models.embedding.embedding_factory import EmbeddingModelFactory
from src.rag.core.exceptions.exceptions import RetrievalError

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """Retriever that uses vector embeddings for similarity search."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the vector retriever with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - top_k: Number of documents to retrieve
                - search_type: Type of search (similarity, mmr)
                - vector_store: Vector store configuration
                - embedder: Embedder configuration
        """
        super().__init__(config)
        self.search_type = self.config.get("search_type", "similarity")
        self._vector_store = None
        self._embedder = None
    
    async def _init_components(self):
        """Initialize vector store and embedder components lazily."""
        logger.info("Initializing vector retriever components...")
        
        if self._vector_store is None:
            try:
                # Get vector store configuration from chatbot section
                vector_store_config = self.config_manager.get_section("chatbot.retrieval.vector_store", {})
                logger.info(f"Vector store config from chatbot section: {vector_store_config}")
                
                # If chatbot config is empty, try the main vector store config
                if not vector_store_config:
                    vector_store_config = self.config_manager.get_section("vector_store", {})
                    logger.info(f"Fallback vector store config: {vector_store_config}")
                
                vector_store_type = vector_store_config.get("type", "pgvector")
                logger.info(f"Using vector store type: {vector_store_type}")
                
                # Support for different vector store types
                if vector_store_type == "faiss":
                    # Configure paths for index and metadata
                    data_dir = self.config.get("data_dir", "./data")
                    index_dir = f"{data_dir}/indices"
                    
                    # Ensure directory exists
                    import os
                    os.makedirs(index_dir, exist_ok=True)
                    
                    # Set up vector store config with proper paths and defaults
                    faiss_config = {
                        "type": "faiss",
                        "dimension": vector_store_config.get("dimension", 768),  # Default for text-embedding-004
                        "index_type": vector_store_config.get("index_type", "HNSW"),
                        "index_path": f"{index_dir}/faiss.index",
                        "metadata_path": f"{index_dir}/metadata.pickle",
                        **vector_store_config  # Override with any additional config
                    }
                    
                    logger.info(f"FAISS config: {faiss_config}")
                    self._vector_store = FAISSVectorStore(faiss_config)
                    logger.info(f"FAISSVectorStore object created: {self._vector_store}")
                    
                    logger.info("Calling FAISS initialize()...")
                    await self._vector_store.initialize()
                    logger.info(f"FAISS initialize() completed. _initialized = {getattr(self._vector_store, '_initialized', 'MISSING')}")
                    
                    # Double-check the state after initialization
                    if hasattr(self._vector_store, '_initialized'):
                        logger.info(f"Vector store _initialized attribute: {self._vector_store._initialized}")
                    else:
                        logger.error("Vector store missing _initialized attribute!")
                    
                    if hasattr(self._vector_store, 'index') and self._vector_store.index:
                        logger.info(f"FAISS index exists with {self._vector_store.index.ntotal} vectors")
                    else:
                        logger.warning("FAISS index is None or missing")
                    
                    logger.info(f"FAISS vector store setup complete")
                    
                elif vector_store_type == "pgvector":
                    # Use pgvector configuration
                    self._vector_store = PgVectorStore(vector_store_config)
                    await self._vector_store.initialize()
                    logger.info("PgVector store initialized")
                else:
                    raise ValueError(f"Unsupported vector store type: {vector_store_type}. Supported types: faiss, pgvector")
                    
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
                raise
        
        if self._embedder is None:
            try:
                # Use factory to create embedder - it will automatically read config from YAML
                logger.info("Creating embedder using factory...")
                self._embedder = EmbeddingModelFactory.create_model()
                logger.info(f"Embedder created: {type(self._embedder).__name__}")
            except Exception as e:
                logger.error(f"Failed to create embedder: {str(e)}", exc_info=True)
                raise
        
        logger.info("Vector retriever components initialized successfully")
    
    async def _retrieve_documents(self, query: str, config: Dict[str, Any]) -> List[Document]:
        """Retrieve documents using vector similarity.
        
        Args:
            query: User query string
            config: Configuration parameters for retrieval
            
        Returns:
            List of retrieved documents
        """
        await self._init_components()
        
        # Get configuration
        top_k = config.get("top_k", self.top_k)
        search_type = config.get("search_type", self.search_type)
        
        try:
            # Generate query embedding using the correct method
            logger.info(f"Generating embedding for query: '{query[:100]}...'")
            query_embedding = await self._embedder.embed_single(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            logger.info(f"Generated embedding with dimension: {len(query_embedding)}")
            
            # Check if vector store is initialized and has data
            if not hasattr(self._vector_store, '_initialized') or not self._vector_store._initialized:
                logger.error("Vector store is not initialized - this should not happen after _init_components()")
                raise RetrievalError("Vector store initialization failed")
            
            # Check if vector store has any chunks
            if hasattr(self._vector_store, 'chunks'):
                total_chunks = len(self._vector_store.chunks) if self._vector_store.chunks else 0
                logger.info(f"Vector store has {total_chunks} chunks available")
                if total_chunks == 0:
                    logger.warning("Vector store has no chunks - no documents have been ingested")
                    logger.warning("Please use the ingestion API to upload and process documents first")
                    logger.warning("Example: POST /ingest/upload with a document file")
                    return []
            
            # Check if FAISS index has vectors
            if hasattr(self._vector_store, 'index') and self._vector_store.index:
                index_total = self._vector_store.index.ntotal
                logger.info(f"FAISS index has {index_total} vectors")
                if index_total == 0:
                    logger.warning("FAISS index is empty - no vectors have been indexed")
                    logger.warning("This could mean documents were uploaded but embedding/indexing failed")
                    return []
            
            # Perform vector search
            logger.info(f"Performing {search_type} search with top_k={top_k}")
            if search_type == "similarity":
                search_results = await self._vector_store.search(query_embedding, top_k=top_k)
            elif search_type == "mmr":
                # Maximum Marginal Relevance search - not implemented yet
                search_results = await self._vector_store.search(query_embedding, top_k=top_k)
            else:
                logger.warning(f"Unknown search type '{search_type}', falling back to similarity search")
                search_results = await self._vector_store.search(query_embedding, top_k=top_k)
            
            logger.info(f"Vector search returned {len(search_results)} results")
            
            # Convert search results to documents
            documents = []
            for i, result in enumerate(search_results):
                try:
                    # Create document from chunk
                    chunk = result.chunk
                    logger.debug(f"Processing result {i}: score={result.score}, content_length={len(chunk.content)}")
                    document = Document(
                        content=chunk.content,
                        metadata={
                            **chunk.metadata,
                            "score": result.score
                        }
                    )
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Error processing search result {i}: {str(e)}")
                    continue
            
            logger.info(f"Successfully converted {len(documents)} search results to documents")
            return documents
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}", exc_info=True)
            raise RetrievalError(f"Vector retrieval failed: {str(e)}") from e
