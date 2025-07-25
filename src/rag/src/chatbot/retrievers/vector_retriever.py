import logging
from typing import Dict, Any, List, Optional

from src.rag.src.chatbot.retrievers.base_retriever import BaseRetriever
from src.rag.src.core.interfaces.base import Document
from src.rag.src.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.src.ingestion.indexers.faiss_vector_store import FAISSVectorStore
from src.rag.src.ingestion.indexers.pgvector_store import PgVectorStore
from src.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
from src.rag.src.core.exceptions.exceptions import RetrievalError

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
        if self._vector_store is None:
            # Get vector store configuration from chatbot section
            vector_store_config = self.config_manager.get_section("chatbot.retrieval.vector_store", {})
            vector_store_type = vector_store_config.get("type", "faiss")
            
            # Support for different vector store types
            if vector_store_type == "faiss":
                # Configure paths for index and metadata
                data_dir = self.config.get("data_dir", "./data")
                index_dir = f"{data_dir}/indices"
                vector_store_config["index_path"] = f"{index_dir}/faiss.index"
                vector_store_config["metadata_path"] = f"{index_dir}/metadata.pickle"
                self._vector_store = FAISSVectorStore(vector_store_config)
                await self._vector_store.initialize()
            elif vector_store_type == "pgvector":
                # Use pgvector configuration
                self._vector_store = PgVectorStore(vector_store_config)
                await self._vector_store.initialize()
            else:
                raise ValueError(f"Unsupported vector store type: {vector_store_type}. Supported types: faiss, pgvector")
        
        if self._embedder is None:
            # Get embedder configuration
            embedder_config = self.config_manager.get_section("ingestion.embedding", {})
            # Use factory to create the appropriate embedder based on config
            self._embedder = await EmbedderFactory.create_embedder(embedder_config)
    
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
            # Generate query embedding
            query_embedding = await self._embedder.embed([query])
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search
            if search_type == "similarity":
                search_results = await self._vector_store.search(query_embedding[0], top_k=top_k)
            elif search_type == "mmr":
                # Maximum Marginal Relevance search - not implemented yet
                search_results = await self._vector_store.search(query_embedding[0], top_k=top_k)
            else:
                logger.warning(f"Unknown search type '{search_type}', falling back to similarity search")
                search_results = await self._vector_store.search(query_embedding[0], top_k=top_k)
            
            # Convert search results to documents
            documents = []
            for result in search_results:
                # Create document from chunk
                chunk = result.chunk
                document = Document(
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "score": result.score
                    }
                )
                documents.append(document)
            
            return documents
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}", exc_info=True)
            raise RetrievalError(f"Vector retrieval failed: {str(e)}") from e
