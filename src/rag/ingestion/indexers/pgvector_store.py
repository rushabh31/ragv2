import logging
import os
import pickle
import json
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.ext.asyncio.session import async_sessionmaker
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import select, insert, update
import numpy as np
from datetime import datetime

from src.rag.ingestion.indexers.base_vector_store import BaseVectorStore
from src.rag.core.interfaces.base import Chunk, SearchResult
from src.rag.core.exceptions.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class PgVectorStore(BaseVectorStore):
    """Vector store using PostgreSQL with pgvector extension for similarity search."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pgvector store with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - dimension: Dimension of the embedding vectors  
                - table_name: Name of the table to store vectors
                - schema_name: PostgreSQL schema name (default: 'public')
                - metadata_path: Optional path to save metadata backup
                - index_method: Vector index method ('ivfflat' or 'hnsw')
                - database: Optional database name override
                - ssl_mode: SSL mode for connection (default: 'require' for production)
                
        Note: Connection details and credentials are loaded from environment variables:
              - PGVECTOR_URL: Base connection URL with host, port, database
              - PostgreSQL credentials from YAML secrets file
        """
        super().__init__(config)
        self.dimension = self.config.get("dimension", 768)
        self.table_name = self.config.get("table_name", "document_embeddings")
        self.schema_name = self.config.get("schema_name", "public")
        self.metadata_path = self.config.get("metadata_path")
        self.index_method = self.config.get("index_method", "ivfflat")  # ivfflat, hnsw
        
        # Connection configuration (loaded from environment)
        self.database_override = self.config.get("database")  # Optional database name override
        self.ssl_mode = self.config.get("ssl_mode", "require")  # Production-grade SSL by default
        
        # Create fully qualified table name with schema
        self.qualified_table_name = f"{self.schema_name}.{self.table_name}"
        
        # SQLAlchemy async engine and session factory
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory = None
        self.chunks = []  # Store chunk objects for retrieval
        self._initialized = False
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string using environment variables and YAML secrets.
        
        Returns:
            PostgreSQL connection string for asyncpg
            
        Raises:
            VectorStoreError: If connection string cannot be built
        """
        try:
            from src.utils.env_manager import env
            
            # Build connection string using env manager with PGVECTOR_URL and secrets
            connection_string = env.build_postgresql_connection_string(
                database=self.database_override,
                schema=self.schema_name if self.schema_name != "public" else None,
                ssl_mode=self.ssl_mode
            )
            
            logger.info("Successfully built PostgreSQL connection string from environment variables and YAML secrets")
            return connection_string
            
        except Exception as e:
            error_msg = (
                f"Failed to build PostgreSQL connection string from environment: {e}. "
                "Ensure PGVECTOR_URL is set and PostgreSQL credentials are available in YAML secrets file."
            )
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    async def initialize(self) -> None:
        """Initialize the pgvector store.
        
        Creates the necessary database connection, extensions, tables, and indexes.
        """
        if self._initialized:
            return
            
        try:
            # Build connection string from environment variables and secrets
            connection_string = self._build_connection_string()
            
            # Convert to SQLAlchemy async format
            if connection_string.startswith('postgresql://'):
                sqlalchemy_url = connection_string.replace('postgresql://', 'postgresql+asyncpg://')
            else:
                sqlalchemy_url = f"postgresql+asyncpg://{connection_string}"
            
            # Remove SSL mode and schema options from URL and handle them separately
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(sqlalchemy_url)
            ssl_mode_param = None
            schema_option = None
            
            if parsed.query:
                query_params = dict(urlparse.parse_qsl(parsed.query))
                ssl_mode_param = query_params.pop('sslmode', None)
                
                # Remove schema options (these cause issues with asyncpg)
                options_param = query_params.pop('options', None)
                if options_param:
                    # Extract schema from options parameter
                    import re
                    schema_match = re.search(r'search_path[%=]([^%&,]+)', options_param)
                    if schema_match:
                        schema_option = schema_match.group(1)
                
                # Rebuild URL without problematic parameters
                if query_params:
                    new_query = urlparse.urlencode(query_params)
                    sqlalchemy_url = urlparse.urlunparse(
                        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
                    )
                else:
                    sqlalchemy_url = urlparse.urlunparse(
                        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment)
                    )
            
            # Configure SSL mode for asyncpg
            connect_args = {"server_settings": {"application_name": "pgvector_indexer"}}
            if ssl_mode_param:
                if ssl_mode_param == 'require':
                    connect_args['ssl'] = 'require'
                elif ssl_mode_param == 'prefer':
                    connect_args['ssl'] = 'prefer'
                elif ssl_mode_param == 'disable':
                    connect_args['ssl'] = 'disable'
            
            # Set schema in server settings if specified
            if schema_option or self.schema_name != "public":
                schema_to_use = schema_option or self.schema_name
                connect_args["server_settings"]["search_path"] = f"{schema_to_use},public"
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                sqlalchemy_url,
                pool_size=self.config.get("pool_size", 10),
                max_overflow=self.config.get("max_overflow", 20),
                pool_pre_ping=True,
                echo=self.config.get("echo_sql", False),  # Set to True for SQL debugging
                pool_recycle=3600,  # Recycle connections after 1 hour
                connect_args=connect_args
            )
            
            # Create async session factory
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
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
        """Set up the database with required extensions, schema, and tables."""
        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            
            # Create schema if it doesn't exist (skip for 'public' schema)
            if self.schema_name != 'public':
                await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {self.schema_name}'))
                logger.info(f"Created schema: {self.schema_name}")
            
            # Create table if it doesn't exist
            await conn.execute(text(f'''
                CREATE TABLE IF NOT EXISTS {self.qualified_table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    soeid TEXT,
                    session_id TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self.dimension}),
                    base64_image TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_number INTEGER,
                    file_name TEXT,
                    document_id TEXT,
                    chunk_index INTEGER
                )
            '''))
            
            # Create index based on the specified method
            if self.index_method == 'ivfflat':
                try:
                    # Create IVFFlat index (approximate nearest neighbor)
                    await conn.execute(text(f'''
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.qualified_table_name} 
                        USING ivfflat (embedding vector_cosine_ops) 
                        WITH (lists = 100)
                    '''))
                except Exception as e:
                    logger.warning(f"Failed to create IVFFlat index: {str(e)}. This may happen if the table is empty.")
            elif self.index_method == 'hnsw':
                try:
                    # Create HNSW index for newer PostgreSQL versions with pgvector 0.5.0+
                    await conn.execute(text(f'''
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx 
                        ON {self.qualified_table_name} 
                        USING hnsw (embedding vector_cosine_ops) 
                        WITH (m = 16, ef_construction = 64)
                    '''))
                except Exception as e:
                    logger.warning(f"Failed to create HNSW index: {str(e)}. This may require pgvector 0.5.0+ and PostgreSQL 15+.")
            
            # Create additional indexes for efficient querying
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS {self.table_name}_chunk_id_idx ON {self.qualified_table_name} (chunk_id)'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS {self.table_name}_soeid_idx ON {self.qualified_table_name} (soeid)'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS {self.table_name}_session_id_idx ON {self.qualified_table_name} (session_id)'))
            await conn.execute(text(f'CREATE INDEX IF NOT EXISTS {self.table_name}_upload_timestamp_idx ON {self.qualified_table_name} (upload_timestamp)'))
            
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
            async with self.async_session_factory() as session:
                result = await session.execute(
                    text(f'SELECT chunk_id, content, metadata FROM {self.qualified_table_name}')
                )
                rows = result.fetchall()
                
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
            async with self.async_session_factory() as session:
                # Start transaction (automatically handled by session)
                try:
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
                        
                        # Extract image data from metadata
                        base64_image = chunk.metadata.get('base64_image') if chunk.metadata else None
                        image_width = chunk.metadata.get('image_width') if chunk.metadata else None
                        image_height = chunk.metadata.get('image_height') if chunk.metadata else None
                        
                        # Convert embedding list to string format for pgvector
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        # Insert or update chunk using SQLAlchemy text with parameters
                        await session.execute(
                            text(f'''
                                INSERT INTO {self.qualified_table_name} 
                                (chunk_id, soeid, session_id, content, metadata, embedding, 
                                 base64_image, image_width, image_height,
                                 page_number, file_name, document_id, chunk_index, created_at)
                                VALUES (:chunk_id, :soeid, :session_id, :content, :metadata, :embedding::vector, 
                                        :base64_image, :image_width, :image_height,
                                        :page_number, :file_name, :document_id, :chunk_index, CURRENT_TIMESTAMP)
                                ON CONFLICT (chunk_id)
                                DO UPDATE SET
                                    soeid = EXCLUDED.soeid,
                                    session_id = EXCLUDED.session_id,
                                    content = EXCLUDED.content,
                                    metadata = EXCLUDED.metadata,
                                    embedding = EXCLUDED.embedding,
                                    base64_image = EXCLUDED.base64_image,
                                    image_width = EXCLUDED.image_width,
                                    image_height = EXCLUDED.image_height,
                                    page_number = EXCLUDED.page_number,
                                    file_name = EXCLUDED.file_name,
                                    document_id = EXCLUDED.document_id,
                                    chunk_index = EXCLUDED.chunk_index,
                                    upload_timestamp = CURRENT_TIMESTAMP
                            '''),
                            {
                                'chunk_id': chunk.id,
                                'soeid': soeid,
                                'session_id': session_id,
                                'content': chunk.content,
                                'metadata': metadata_json,
                                'embedding': embedding_str,
                                'base64_image': base64_image,
                                'image_width': image_width,
                                'image_height': image_height,
                                'page_number': page_number,
                                'file_name': file_name,
                                'document_id': document_id,
                                'chunk_index': chunk_index
                            }
                        )
                    
                    # Commit the transaction
                    await session.commit()
                    
                except Exception as e:
                    # Rollback on error
                    await session.rollback()
                    raise
            
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
            async with self.async_session_factory() as session:
                result = await session.execute(
                    text(f'''
                        SELECT chunk_id, content, metadata, base64_image, image_width, image_height,
                               1 - (embedding <=> :embedding::vector) AS similarity
                        FROM {self.qualified_table_name}
                        ORDER BY similarity DESC
                        LIMIT :top_k
                    '''),
                    {
                        'embedding': embedding_str,
                        'top_k': top_k
                    }
                )
                results = result.fetchall()
            
            search_results = []
            for row in results:
                chunk_id = row['chunk_id']
                content = row['content']
                metadata = row['metadata']
                base64_image = row['base64_image']
                image_width = row['image_width']
                image_height = row['image_height']
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
                
                # Add image data to metadata if available
                if base64_image:
                    metadata['base64_image'] = base64_image
                    metadata['image_width'] = image_width
                    metadata['image_height'] = image_height
                
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
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.async_session_factory = None
            logger.info("Closed pgvector database connections")
