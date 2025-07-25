import logging
import os
import shutil
import uuid
import asyncio
from typing import Dict, Any, List, Optional, BinaryIO
from datetime import datetime
from fastapi import UploadFile

from src.rag.src.shared.models.schema import DocumentMetadata, DocumentType, ProcessingJob
from src.rag.src.shared.utils.config_manager import ConfigManager
from src.rag.src.core.exceptions.exceptions import DocumentProcessingError, ParsingError, ChunkingError, IndexingError, EmbeddingError
from src.rag.src.ingestion.parsers.base_parser import BaseDocumentParser
from src.rag.src.ingestion.parsers.vision_parser import VisionParser
from src.rag.src.ingestion.parsers.groq_simple_parser import GroqSimpleParser
from src.rag.src.ingestion.parsers.openai_vision_parser import OpenAIVisionParser
from src.rag.src.ingestion.parsers.simple_text_parser import SimpleTextParser
from src.rag.src.ingestion.chunkers.base_chunker import BaseChunker
from src.rag.src.ingestion.chunkers.fixed_size_chunker import FixedSizeChunker
from src.rag.src.ingestion.chunkers.semantic_chunker import SemanticChunker
from src.rag.src.ingestion.chunkers.page_based_chunker import PageBasedChunker
from src.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
from src.rag.src.ingestion.embedders.base_embedder import BaseEmbedder
from src.rag.src.ingestion.embedders.vertex_embedder import VertexEmbedder
from src.rag.src.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.src.ingestion.indexers.faiss_vector_store import FAISSVectorStore
from src.rag.src.ingestion.indexers.pgvector_store import PgVectorStore
from src.rag.src.ingestion.indexers.chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class IngestionService:
    """Service for document ingestion pipeline."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize the ingestion service.
        
        Args:
            data_dir: Directory to store uploaded documents and indices
        """
        self.config_manager = ConfigManager()
        self.data_dir = data_dir
        self.uploads_dir = os.path.join(data_dir, "uploads")
        self.index_dir = os.path.join(data_dir, "indices")
        self.jobs: Dict[str, ProcessingJob] = {}
        
        # Create directories if they don't exist
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize components
        self._parsers = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None
    
    async def _init_components(self):
        """Initialize all components with current configuration."""
        config = self.config_manager.get_config()
        
        # Initialize parsers
        self._parsers = {}
        parser_configs = config["ingestion"]["parsers"]

        # Register simple text parser
        if parser_configs.get("simple_text", {}).get("enabled", False):
            self._parsers["simple_text"] = SimpleTextParser(parser_configs.get("simple_text", {}))

        # Register groq_simple_parser
        if parser_configs.get("groq_simplev_vision", {}).get("enabled", False):
            self._parsers["groq_simplev_vision"] = GroqSimpleParser(parser_configs.get("groq_simplev_vision", {}))

        if parser_configs.get("pymupdf", {}).get("enabled", True):
            self._parsers["pymupdf"] = PyMuPDFParser(parser_configs.get("pymupdf", {}))
        if parser_configs.get("vision_parser", {}).get("enabled", True):
            self._parsers["vision_parser"] = VisionParser(parser_configs.get("vision_parser", {}))
        if parser_configs.get("openai_vision", {}).get("enabled", False):
            self._parsers["openai_vision"] = OpenAIVisionParser(parser_configs.get("openai_vision", {}))
        
        # Initialize chunker based on configuration
        chunker_config = config["ingestion"]["chunking"]
        chunker_strategy = chunker_config.get("strategy", "fixed_size")
        
        if chunker_strategy == "semantic":
            self._chunker = SemanticChunker(chunker_config)
        elif chunker_strategy == "page_based":
            logger.info("Using page-based chunking strategy for documents")
            self._chunker = PageBasedChunker(chunker_config)
        else:
            self._chunker = FixedSizeChunker(chunker_config)
        
        # Initialize embedder using factory
        try:
            embedder_config = config["ingestion"]["embedding"]
            self._embedder = await EmbedderFactory.create_embedder(embedder_config)
            logger.info(f"Embedder initialized successfully with provider: {embedder_config.get('provider', 'default')}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {str(e)}")
            raise EmbeddingError(f"Failed to initialize embedder: {str(e)}")
        
        # Initialize vector store
        vector_store_config = config["ingestion"]["vector_store"]
        vector_store_type = vector_store_config.get("type", "faiss")
        
        # Set dimension from config
        dimension = vector_store_config.get("dimension", 384)
        
        if vector_store_type == "faiss":
            # Create FAISS config with necessary parameters
            faiss_config = {
                "dimension": dimension,
                "index_type": vector_store_config.get("faiss", {}).get("index_type", "HNSW"),
                "index_path": os.path.join(self.index_dir, "faiss.index"),
                "metadata_path": os.path.join(self.index_dir, "metadata.pickle")
            }
            # Add additional FAISS parameters if available
            if "faiss" in vector_store_config:
                faiss_params = vector_store_config["faiss"]
                faiss_config.update({
                    key: value for key, value in faiss_params.items()
                    if key not in ["index_path", "metadata_path"]
                })
            
            self._vector_store = FAISSVectorStore(faiss_config)
            logger.info(f"Initializing FAISS vector store with index type: {faiss_config.get('index_type', 'HNSW')}")
            
        elif vector_store_type == "pgvector":
            # Create PgVector config with necessary parameters
            pg_config = {
                "dimension": dimension,
                "connection_string": vector_store_config.get("pgvector", {}).get("connection_string", ""),
                "table_name": vector_store_config.get("pgvector", {}).get("table_name", "document_embeddings"),
                "index_method": vector_store_config.get("pgvector", {}).get("index_method", "ivfflat"),
                "metadata_path": os.path.join(self.index_dir, "pgvector_metadata.pickle")
            }
            
            self._vector_store = PgVectorStore(pg_config)
            logger.info(f"Initializing PgVector store with table: {pg_config.get('table_name')}")
            
        elif vector_store_type == "chromadb":
            # Create ChromaDB config with necessary parameters
            chroma_config = {
                "dimension": dimension,
                "persist_directory": vector_store_config.get("chromadb", {}).get("persist_directory", "./data/chromadb"),
                "collection_name": vector_store_config.get("chromadb", {}).get("collection_name", "document_embeddings"),
                "distance_function": vector_store_config.get("chromadb", {}).get("distance_function", "cosine"),
                "metadata_path": os.path.join(self.index_dir, "chromadb_metadata.pickle")
            }
            
            self._vector_store = ChromaVectorStore(chroma_config)
            logger.info(f"Initializing ChromaDB vector store with collection: {chroma_config.get('collection_name')}")
            
        else:
            # Default to FAISS if vector store type is not recognized
            logger.warning(f"Unknown vector store type: {vector_store_type}, defaulting to FAISS")
            faiss_config = {
                "dimension": dimension,
                "index_path": os.path.join(self.index_dir, "faiss.index"),
                "metadata_path": os.path.join(self.index_dir, "metadata.pickle")
            }
            self._vector_store = FAISSVectorStore(faiss_config)
            
        # Initialize the selected vector store
        await self._vector_store.initialize()
    
    async def upload_document(self, file_path: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingJob:
        """Upload and start processing a document.
        
        Args:
            file_path: Path to the uploaded file
            user_id: ID of the user uploading the document
            metadata: Optional metadata for the document
            
        Returns:
            ProcessingJob object with job_id
        """
        job_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        
        # Create job record
        job = ProcessingJob(
            job_id=job_id,
            document_id=document_id,
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            user_id=user_id
        )
        self.jobs[job_id] = job
        
        # Start processing in background
        asyncio.create_task(self._process_document(job, file_path, metadata))
        
        return job
    
    async def _process_document(self, job: ProcessingJob, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Process document through the ingestion pipeline.
        
        Args:
            job: Processing job record
            file_path: Path to the uploaded file
            metadata: Optional metadata for the document
        """
        try:
            # Initialize components if not already done
            if self._parsers is None:
                await self._init_components()
            
            # Update job status
            job.status = "processing"
            job.updated_at = datetime.now()
            job.progress = 0.0
            
            # Determine document type
            doc_type = self._get_document_type(file_path)
            
            # Parse document
            job.progress = 0.1
            documents = await self._parse_document(file_path, doc_type)
            
            # Extract and enhance metadata
            if documents:
                doc_metadata = documents[0].metadata
                
                # Ensure user_id is set in all document metadata
                for doc in documents:
                    if not doc.metadata:
                        doc.metadata = {}
                    
                    # Add user_id if not present
                    if 'user_id' not in doc.metadata:
                        doc.metadata['user_id'] = job.user_id
                    
                    # Add document_id if not present
                    if 'document_id' not in doc.metadata:
                        doc.metadata['document_id'] = job.document_id
                        
                    # Add created_at timestamp if not present
                    if 'created_at' not in doc.metadata:
                        doc.metadata['created_at'] = job.created_at.isoformat()
                        
                    # Add filename if not present
                    if 'filename' not in doc.metadata and file_path:
                        doc.metadata['filename'] = os.path.basename(file_path)
                
                job.progress = 0.3
            
            # Chunk documents
            job.progress = 0.4
            chunks = await self._chunker.chunk(documents, self.config_manager.get_section("ingestion.chunking"))
            
            # Generate embeddings
            job.progress = 0.6
            texts = [chunk.content for chunk in chunks]
            embeddings = await self._embedder.embed(texts)
            
            # Add to vector store
            job.progress = 0.8
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            await self._vector_store.add(chunks)
            
            # Update job status
            job.status = "completed"
            job.progress = 1.0
            job.updated_at = datetime.now()
            
        except Exception as e:
            # Update job status on failure
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.now()
            logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            
            # Clean up file if needed
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
    
    def _get_document_type(self, file_path: str) -> DocumentType:
        """Determine document type from file extension."""
        return BaseDocumentParser.get_document_type(file_path)
    
    async def _parse_document(self, file_path: str, doc_type: DocumentType) -> List[Any]:
        """Parse document using appropriate parser.
        
        Args:
            file_path: Path to document file
            doc_type: Document type
            
        Returns:
            List of parsed document objects
        """
        if doc_type == DocumentType.PDF:
            # Get default parser from config
            config = self.config_manager.get_config()
            default_parser = config["ingestion"]["parsers"].get("default_parser", "pymupdf")

            # Try the default parser first
            if default_parser in self._parsers:
                try:
                    logger.info(f"Using {default_parser} parser for PDF processing")
                    return await self._parsers[default_parser].parse(file_path, {})
                except Exception as e:
                    logger.warning(f"{default_parser} parser failed: {str(e)}")

            # Try available parsers in priority order (include groq_simplev_vision)
            for parser_name in ["groq_simplev_vision", "openai_vision", "vision_parser", "pymupdf"]:
                if parser_name in self._parsers and parser_name != default_parser:
                    try:
                        logger.info(f"Trying fallback parser: {parser_name}")
                        return await self._parsers[parser_name].parse(file_path, {})
                    except Exception as e:
                        logger.warning(f"{parser_name} parser failed: {str(e)}")
                        continue
        
        # For other document types, add handlers here
        
        raise DocumentProcessingError(f"No parser available for document type {doc_type}")
    
    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProcessingJob object or None if not found
        """
        return self.jobs.get(job_id)
    
    async def list_documents(self, user_id: Optional[str] = None, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """List ingested documents.
        
        Args:
            user_id: Optional user ID to filter documents
            page: Page number
            page_size: Number of documents per page
            
        Returns:
            Dictionary with documents, total count, and pagination info
        """
        try:
            # Initialize components if needed
            if not self._vector_store:
                await self._init_components()
            
            # Get all chunks from vector store
            if not hasattr(self._vector_store, 'chunks') or not self._vector_store.chunks:
                logger.info("No documents found in vector store")
                return {
                    "documents": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size
                }
            
            # Extract unique document IDs and metadata
            unique_docs = {}
            for chunk in self._vector_store.chunks:
                if chunk.metadata and 'document_id' in chunk.metadata:
                    doc_id = chunk.metadata['document_id']
                    if doc_id not in unique_docs:
                        # Filter by user_id if provided
                        if user_id and chunk.metadata.get('user_id') != user_id:
                            continue
                        
                        # Extract relevant metadata
                        # Use created_at for ingestion_time if available, or current datetime
                        created_at = chunk.metadata.get('created_at')
                        doc_metadata = {
                            'document_id': doc_id,
                            'title': chunk.metadata.get('title', 'Untitled Document'),
                            'filename': chunk.metadata.get('filename', 'unknown.pdf'),
                            'created_at': created_at,
                            'user_id': chunk.metadata.get('user_id'),
                            'document_type': chunk.metadata.get('document_type', 'pdf'),
                            'page_count': chunk.metadata.get('page_count', 1),
                            'chunk_count': 1,  # Start counting chunks
                            'ingestion_time': created_at or datetime.now()  # Add required ingestion_time field
                        }
                        unique_docs[doc_id] = doc_metadata
                    else:
                        # Update chunk count for existing document
                        unique_docs[doc_id]['chunk_count'] += 1
            
            # Convert to list and sort by created_at (newest first)
            docs_list = list(unique_docs.values())
            docs_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            # Apply pagination
            total = len(docs_list)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_docs = docs_list[start_idx:end_idx]
            
            logger.info(f"Found {total} documents, returning {len(paginated_docs)} for page {page}")
            
            return {
                "documents": paginated_docs,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}", exc_info=True)
            return {
                "documents": [],
                "total": 0,
                "page": page,
                "page_size": page_size
            }
    
    async def update_config(self, config_path: str, value: Any) -> bool:
        """Update ingestion configuration.
        
        Args:
            config_path: Path to configuration setting (e.g., 'ingestion.chunking.strategy')
            value: New value for the setting
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            self.config_manager.update_section(config_path, value)
            # Re-initialize components on next use
            self._parsers = None
            self._chunker = None
            self._embedder = None
            self._vector_store = None
            return True
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            return False
