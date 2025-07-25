"""
Custom reranker implementation without external models.
"""

import logging
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import math

from src.rag.chatbot.rerankers.base_reranker import BaseReranker
from src.rag.core.interfaces.base import Document

logger = logging.getLogger(__name__)


class CustomReranker(BaseReranker):
    """
    Custom reranker that uses text-based similarity metrics without external models.
    
    This reranker combines multiple scoring methods:
    1. TF-IDF similarity
    2. Keyword overlap
    3. Length normalization
    4. Position-based scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the custom reranker.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - keyword_weight: Weight for keyword overlap scoring (default: 0.3)
                - tfidf_weight: Weight for TF-IDF scoring (default: 0.4)
                - length_weight: Weight for length normalization (default: 0.2)
                - position_weight: Weight for position-based scoring (default: 0.1)
                - min_score_threshold: Minimum score threshold (default: 0.0)
        """
        super().__init__(config)
        
        # Scoring weights
        self.keyword_weight = self.config.get("keyword_weight", 0.3)
        self.tfidf_weight = self.config.get("tfidf_weight", 0.4)
        self.length_weight = self.config.get("length_weight", 0.2)
        self.position_weight = self.config.get("position_weight", 0.1)
        self.min_score_threshold = self.config.get("min_score_threshold", 0.0)
        
        logger.info(f"Custom reranker initialized with weights: "
                   f"keyword={self.keyword_weight}, tfidf={self.tfidf_weight}, "
                   f"length={self.length_weight}, position={self.position_weight}")
    
    async def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Rerank documents using custom text-based similarity metrics.
        
        Args:
            query: User query string
            documents: List of documents to rerank
            top_k: Number of top documents to return (default: all documents)
            
        Returns:
            List of reranked documents sorted by relevance score
        """
        if not documents:
            return documents
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            # Preprocess query
            query_tokens = self._tokenize_and_normalize(query)
            query_keywords = self._extract_keywords(query)
            
            # Calculate scores for each document
            scored_documents = []
            for idx, doc in enumerate(documents):
                score = self._calculate_document_score(
                    query, query_tokens, query_keywords, doc, idx, len(documents)
                )
                
                # Filter by minimum score threshold
                if score >= self.min_score_threshold:
                    # Store score in metadata
                    doc.metadata = doc.metadata or {}
                    doc.metadata["rerank_score"] = score
                    scored_documents.append((score, doc))
            
            # Sort by score (descending) and return top_k
            scored_documents.sort(key=lambda x: x[0], reverse=True)
            reranked_docs = [doc for _, doc in scored_documents[:top_k]]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            # Return original documents if reranking fails
            return documents[:top_k] if top_k else documents
    
    def _calculate_document_score(self, query: str, query_tokens: List[str], 
                                 query_keywords: List[str], document: Document, 
                                 position: int, total_docs: int) -> float:
        """Calculate combined relevance score for a document."""
        
        doc_content = document.content.lower()
        doc_tokens = self._tokenize_and_normalize(doc_content)
        
        # 1. Keyword overlap score
        keyword_score = self._calculate_keyword_overlap(query_keywords, doc_content)
        
        # 2. TF-IDF similarity score
        tfidf_score = self._calculate_tfidf_similarity(query_tokens, doc_tokens)
        
        # 3. Length normalization score (prefer documents with reasonable length)
        length_score = self._calculate_length_score(doc_content)
        
        # 4. Position-based score (slight preference for earlier results)
        position_score = self._calculate_position_score(position, total_docs)
        
        # Combine scores with weights
        final_score = (
            self.keyword_weight * keyword_score +
            self.tfidf_weight * tfidf_score +
            self.length_weight * length_score +
            self.position_weight * position_score
        )
        
        return final_score
    
    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        # Filter out very short words
        words = [word for word in words if len(word) > 2]
        return words
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        tokens = self._tokenize_and_normalize(text)
        
        # Simple keyword extraction: remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 
            'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'can', 
            'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must',
            'have', 'has', 'had', 'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'been', 'being', 'get', 'got', 'getting'
        }
        
        keywords = [token for token in tokens if token not in stop_words]
        return keywords
    
    def _calculate_keyword_overlap(self, query_keywords: List[str], doc_content: str) -> float:
        """Calculate keyword overlap score."""
        if not query_keywords:
            return 0.0
        
        doc_content_lower = doc_content.lower()
        matches = sum(1 for keyword in query_keywords if keyword in doc_content_lower)
        return matches / len(query_keywords)
    
    def _calculate_tfidf_similarity(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Calculate simplified TF-IDF similarity."""
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Calculate term frequencies
        query_tf = Counter(query_tokens)
        doc_tf = Counter(doc_tokens)
        
        # Calculate similarity using cosine similarity of TF vectors
        common_terms = set(query_tf.keys()) & set(doc_tf.keys())
        
        if not common_terms:
            return 0.0
        
        # Simplified cosine similarity
        numerator = sum(query_tf[term] * doc_tf[term] for term in common_terms)
        
        query_norm = math.sqrt(sum(count ** 2 for count in query_tf.values()))
        doc_norm = math.sqrt(sum(count ** 2 for count in doc_tf.values()))
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return numerator / (query_norm * doc_norm)
    
    def _calculate_length_score(self, doc_content: str) -> float:
        """Calculate length-based score (prefer documents with reasonable length)."""
        length = len(doc_content)
        
        # Optimal length range (adjust as needed)
        min_optimal = 100
        max_optimal = 2000
        
        if min_optimal <= length <= max_optimal:
            return 1.0
        elif length < min_optimal:
            # Penalize very short documents
            return length / min_optimal
        else:
            # Penalize very long documents
            return max_optimal / length
    
    def _calculate_position_score(self, position: int, total_docs: int) -> float:
        """Calculate position-based score (slight preference for earlier results)."""
        if total_docs <= 1:
            return 1.0
        
        # Linear decay from 1.0 to 0.5
        return 1.0 - (0.5 * position / (total_docs - 1))
