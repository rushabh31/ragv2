import logging
import os
import pickle
import json
from typing import Dict, Any, List, Optional, Tuple
import asyncpg
import numpy as np
from datetime import datetime

from src.rag.src.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.src.core.interfaces.base import Chunk, SearchResult
from src.rag.src.core.exceptions.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class PgVectorStore(BaseVectorStore):
    """Vector store using PostgreSQL with pgvector extension for similarity search."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pgvector store with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - dimension: Dimension of the embedding vectors
                - connection_string: PostgreSQL connection string
                - table_name: Name of the table to store vectors
                - metadata_path: Optional path to save metadata backup
        """
        super().__init__(config)
        self.dimension = self.config.get("dimension", 768)
        self.connection_string = self.config.get("connection_string")
        self.table_name = self.config.get("table_name", "document_embeddings")
        self.metadata_path = self.config.get("metadata_path")
        self.index_method = self.config.get("index_method", "ivfflat")  # ivfflat, hnsw
        
        self.conn_pool = None
        self.chunks = []  # Store chunk objects for retrieval
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the pgvector store.
        
        Creates the necessary database connection, extensions, tables, and indexes.
        """
        if self._initialized:
            return
            
        try:
            # Create connection pool
            self.conn_pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=1,
                max_size=10
            )
            
            # Set up database
            await self._setup_database()
            
            # Try to load existing chunks if path provided
            if self.metadata_path and os.path.exists(self.metadata_path):
                await self._load_metadata()
            else:
                # Load existing chunks from database
                await self._load_chunks_from_db()
                
            self._initialized = True
            logger.info(f"Initialized pgvector store with table {self.table_name}")
        except Exception as e:
            error_msg = f"Failed to initialize pgvector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def _setup_database(self) -> None:
        """Set up the database with required extensions and tables."""
        async with self.conn_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create table if it doesn't exist
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    soeid TEXT,
                    session_id TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self.dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_number INTEGER,
                    file_name TEXT,
                    document_id TEXT,
                    chunk_index INTEGER
                )
            ''')
            
            # Create index based on the specified method
            if self.index_method == 'ivfflat':
                try:
                    # Create IVFFlat index (approximate nearest neighbor)
                    await conn.execute(f'''
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops) 
                        WITH (lists = 100)
                    ''')
                except Exception as e:
                    logger.warning(f"Failed to create IVFFlat index: {str(e)}. This may happen if the table is empty.")
            elif self.index_method == 'hnsw':
                try:
                    # Create HNSW index for newer PostgreSQL versions with pgvector 0.5.0+
                    await conn.execute(f'''
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx 
                        ON {self.table_name} 
                        USING hnsw (embedding vector_cosine_ops) 
                        WITH (m = 16, ef_construction = 64)
                    ''')
                except Exception as e:
                    logger.warning(f"Failed to create HNSW index: {str(e)}. This may require pgvector 0.5.0+ and PostgreSQL 15+.")
            
            # Create index on chunk_id for faster lookups
            await conn.execute(f'CREATE INDEX IF NOT EXISTS {self.table_name}_chunk_id_idx ON {self.table_name} (chunk_id)')
            
            logger.info(f"Set up pgvector database table {self.table_name} with {self.index_method} index")
    
    async def _load_metadata(self) -> None:
        """Load chunks metadata from disk."""
        try:
            with open(self.metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks from {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            # Fall back to loading from database
            await self._load_chunks_from_db()
    
    async def _load_chunks_from_db(self) -> None:
        """Load chunks from the database."""
        try:
            async with self.conn_pool.acquire() as conn:
                rows = await conn.fetch(f'SELECT chunk_id, content, metadata FROM {self.table_name}')
                
                self.chunks = []
                for row in rows:
                    chunk_id = row['chunk_id']
                    content = row['content']
                    metadata = row['metadata']

                    # Ensure metadata is a dictionary
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode metadata for chunk {chunk_id}", exc_info=True)
                            metadata = {}
                    elif metadata is None:
                        metadata = {}
                    
                    chunk = Chunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        embedding=None  # We don't load embeddings into memory
                    )
                    self.chunks.append(chunk)
                
                logger.info(f"Loaded {len(self.chunks)} chunks from database")
        except Exception as e:
            logger.error(f"Failed to load chunks from database: {str(e)}")
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
        """Add vectors to the pgvector store.
        
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
            
            # Insert chunks into database
            async with self.conn_pool.acquire() as conn:
                # Start transaction
                async with conn.transaction():
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        # Generate chunk ID if not present
                        if not hasattr(chunk, 'id') or not chunk.id:
                            chunk.id = f"chunk_{len(self.chunks) + i}"
                        
                        # Convert metadata to JSON
                        metadata_json = json.dumps(chunk.metadata, default=str) if chunk.metadata else '{}'
                        
                        # Extract additional fields from metadata
                        soeid = chunk.metadata.get('soeid') if chunk.metadata else None
                        session_id = chunk.metadata.get('session_id') if chunk.metadata else None
                        page_number = chunk.metadata.get('page_number') if chunk.metadata else None
                        file_name = chunk.metadata.get('file_name') if chunk.metadata else None
                        document_id = chunk.metadata.get('document_id') if chunk.metadata else None
                        chunk_index = chunk.metadata.get('chunk_index') if chunk.metadata else None
                        
                        # Convert embedding list to string format for pgvector
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        # Insert or update chunk
                        await conn.execute(f'''
                            INSERT INTO {self.table_name} 
                            (chunk_id, soeid, session_id, content, metadata, embedding, 
                             page_number, file_name, document_id, chunk_index)
                            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9, $10)
                            ON CONFLICT (chunk_id)
                            DO UPDATE SET
                                soeid = $2,
                                session_id = $3,
                                content = $4,
                                metadata = $5,
                                embedding = $6::vector,
                                page_number = $7,
                                file_name = $8,
                                document_id = $9,
                                chunk_index = $10,
                                upload_timestamp = CURRENT_TIMESTAMP
                        ''', chunk.id, soeid, session_id, chunk.content, metadata_json, 
                             embedding_str, page_number, file_name, document_id, chunk_index)
            
            # Add chunks to in-memory store (without embeddings to save memory)
            for chunk in chunks:
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
            logger.info(f"Added {len(chunks)} chunks to pgvector store in {duration:.2f} seconds")
        except Exception as e:
            error_msg = f"Failed to add vectors to pgvector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg) from e
    
    async def _search_vectors(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search vectors in the pgvector store.
        
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
            # Convert embedding list to string format for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            async with self.conn_pool.acquire() as conn:
                results = await conn.fetch(f'''
                    SELECT chunk_id, content, metadata, 
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT $2
                ''', embedding_str, top_k)
            
            search_results = []
            for row in results:
                chunk_id = row['chunk_id']
                content = row['content']
                metadata = row['metadata']
                similarity = float(row['similarity'])

                # Ensure metadata is a dictionary
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode metadata for chunk {chunk_id}", exc_info=True)
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                
                # Create search result
                search_result = SearchResult(
                    chunk=Chunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        embedding=None  # We don't include embeddings in search results
                    ),
                    score=similarity
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
            
    async def close(self) -> None:
        """Close the pgvector store connections."""
        if self.conn_pool:
            await self.conn_pool.close()
            logger.info("Closed pgvector database connections")
