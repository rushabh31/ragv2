import logging
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder

from src.rag.chatbot.rerankers.base_reranker import BaseReranker
from src.rag.core.interfaces.base import Document
from src.rag.core.exceptions.exceptions import RerankerError

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseReranker):
    """Reranker that uses cross-encoder models for more precise relevance scoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the cross-encoder reranker with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - model: Name of the cross-encoder model
                - enabled: Whether reranking is enabled
                - top_k: Number of documents to keep after reranking
        """
        super().__init__(config)
        self.model_name = self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._model = None
    
    async def _init_model(self):
        """Initialize cross-encoder model lazily."""
        if self._model is None:
            try:
                # Note: This is not async, but could be wrapped to run in a thread pool
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Initialized cross-encoder model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize cross-encoder model: {str(e)}")
                raise
    
    async def _rerank_documents(self, query: str, documents: List[Document], config: Dict[str, Any]) -> List[Document]:
        """Rerank documents using a cross-encoder model.
        
        Args:
            query: User query string
            documents: List of documents to rerank
            config: Configuration parameters for reranking
            
        Returns:
            Reranked list of documents
        """
        try:
            await self._init_model()
            
            if not documents:
                return []
            
            # Prepare document-query pairs for scoring
            pairs = [(query, doc.content) for doc in documents]
            
            # Score document-query pairs
            # Note: This is not async, but could be wrapped to run in a thread pool
            scores = self._model.predict(pairs)
            
            # Attach scores to documents
            for i, doc in enumerate(documents):
                doc.metadata["reranker_score"] = float(scores[i])
            
            # Sort documents by score in descending order
            reranked_documents = sorted(documents, key=lambda doc: doc.metadata["reranker_score"], reverse=True)
            
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {str(e)}", exc_info=True)
            # Fall back to original order if reranking fails
            logger.warning("Falling back to original document order")
            return documents
