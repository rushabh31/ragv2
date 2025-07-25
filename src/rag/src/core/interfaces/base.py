from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic
from pydantic import BaseModel

# Generic type variables
T = TypeVar('T')
U = TypeVar('U')

class Document(BaseModel):
    """Base document model representing a document with content and metadata."""
    content: str
    metadata: Dict[str, Any]

class Chunk(BaseModel):
    """Chunk model representing a segment of a document with content and metadata."""
    id: Optional[str] = None  # <-- Added field for chunk ID
    content: str
    metadata: Dict[str, Any]
    # Optional embedding field for vector representation
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """Search result model containing a chunk and its similarity score."""
    chunk: Chunk
    score: float

class DocumentParser(ABC):
    """Abstract interface for document parsers."""
    
    @abstractmethod
    async def parse(self, file_path: str, config: Dict[str, Any]) -> List[Document]:
        """Parse a document from a file path into a list of Document objects.
        
        Args:
            file_path: Path to the document file
            config: Configuration parameters for the parser
            
        Returns:
            List of parsed Document objects
        """
        pass

class Chunker(ABC):
    """Abstract interface for document chunking strategies."""
    
    @abstractmethod
    async def chunk(self, documents: List[Document], config: Dict[str, Any]) -> List[Chunk]:
        """Split documents into chunks according to a specific strategy.
        
        Args:
            documents: List of documents to chunk
            config: Configuration parameters for the chunker
            
        Returns:
            List of document chunks
        """
        pass

class Embedder(ABC):
    """Abstract interface for text embedders."""
    
    @abstractmethod
    async def embed(self, texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to embed
            config: Configuration parameters for the embedder
            
        Returns:
            List of embedding vectors
        """
        pass

class VectorStore(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    async def add(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of chunks to add
            embeddings: Optional list of embeddings. If not provided, embeddings 
                       from the chunks will be used if available.
        """
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search for similar chunks in the vector store.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results sorted by similarity
        """
        pass

class Retriever(ABC):
    """Abstract interface for document retrievers."""
    
    @abstractmethod
    async def retrieve(self, query: str, config: Dict[str, Any]) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            config: Configuration parameters for retrieval
            
        Returns:
            List of retrieved documents
        """
        pass

class Reranker(ABC):
    """Abstract interface for document rerankers."""
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[Document], config: Dict[str, Any]) -> List[Document]:
        """Rerank a list of documents based on relevance to the query.
        
        Args:
            query: User query string
            documents: List of documents to rerank
            config: Configuration parameters for reranking
            
        Returns:
            Reranked list of documents
        """
        pass

class Generator(ABC):
    """Abstract interface for response generators."""
    
    @abstractmethod
    async def generate(self, query: str, documents: List[Document], 
                      conversation_history: Optional[List[Dict[str, str]]] = None, 
                      config: Dict[str, Any] = None) -> str:
        """Generate a response based on the query and relevant documents.
        
        Args:
            query: User query string
            documents: List of relevant documents
            conversation_history: Optional conversation history
            config: Configuration parameters for generation
            
        Returns:
            Generated response
        """
        pass

class Memory(ABC):
    """Abstract interface for conversation memory."""
    
    @abstractmethod
    async def add(self, user_id: str, messages: List[Dict[str, str]]) -> None:
        """Add messages to the conversation memory.
        
        Args:
            user_id: User identifier
            messages: List of message dictionaries with 'role' and 'content' keys
        """
        pass
    
    @abstractmethod
    async def get(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        pass
