#!/usr/bin/env python3
"""
PostgreSQL Database Setup Script for RAG System with pgvector
============================================================

This script sets up all PostgreSQL databases required for the RAG system:
1. rag_vector_db - For document embeddings and vector search
2. langgraph_memory_db - For LangGraph conversation memory

Features:
- Creates databases with pgvector extension
- Sets up proper tables and indexes
- Validates configuration consistency
- Provides comprehensive error handling and logging

Usage:
    python setup_pgvector_databases.py
"""

import asyncio
import asyncpg
import logging
import os
import sys
from typing import Dict, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PgVectorDatabaseSetup:
    """Setup and configure PostgreSQL databases for RAG system with pgvector."""
    
    def __init__(self):
        """Initialize the database setup."""
        self.base_connection_string = "postgresql://rushabhsmacbook@localhost:5432"
        self.databases = {
            "rag_vector_db": {
                "description": "Vector store for document embeddings",
                "schemas": {
                    "public": ["document_embeddings"],
                    "rag_prod": ["document_embeddings"],
                    "rag_dev": ["document_embeddings"],
                    "rag_test": ["document_embeddings"]
                },
                "extensions": ["vector"]
            },
            "langgraph_memory_db": {
                "description": "LangGraph conversation memory storage",
                "schemas": {
                    "public": ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"]
                },
                "extensions": ["vector"]  # May be needed for future memory features
            }
        }
    
    async def create_database(self, db_name: str) -> bool:
        """Create a database if it doesn't exist."""
        try:
            # Connect to default postgres database to create new database
            conn = await asyncpg.connect(f"{self.base_connection_string}/postgres")
            
            # Check if database exists
            result = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            
            if result:
                logger.info(f"Database '{db_name}' already exists")
                await conn.close()
                return True
            
            # Create database
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Created database: {db_name}")
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database {db_name}: {str(e)}")
            return False
    
    async def setup_extensions(self, db_name: str, extensions: list) -> bool:
        """Set up required extensions for a database."""
        try:
            conn = await asyncpg.connect(f"{self.base_connection_string}/{db_name}")
            
            for extension in extensions:
                try:
                    await conn.execute(f'CREATE EXTENSION IF NOT EXISTS {extension}')
                    logger.info(f"Enabled extension '{extension}' in {db_name}")
                except Exception as e:
                    logger.warning(f"Could not enable extension '{extension}' in {db_name}: {str(e)}")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup extensions for {db_name}: {str(e)}")
            return False
    
    async def setup_vector_store_table(self, db_name: str) -> bool:
        """Set up the document_embeddings table for vector storage."""
        try:
            conn = await asyncpg.connect(f"{self.base_connection_string}/{db_name}")
            
            # Create document_embeddings table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    soeid TEXT,
                    session_id TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_number INTEGER,
                    file_name TEXT,
                    document_id TEXT,
                    chunk_index INTEGER
                )
            ''')
            logger.info(f"Created document_embeddings table in {db_name}")
            
            # Create HNSW index for efficient vector search
            try:
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS document_embeddings_embedding_hnsw_idx 
                    ON document_embeddings 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                ''')
                logger.info(f"Created HNSW index on embeddings in {db_name}")
            except Exception as e:
                logger.warning(f"Could not create HNSW index (trying IVFFlat): {str(e)}")
                
                # Fallback to IVFFlat index
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS document_embeddings_embedding_ivfflat_idx 
                    ON document_embeddings 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                ''')
                logger.info(f"Created IVFFlat index on embeddings in {db_name}")
            
            # Create additional indexes for efficient querying
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_soeid 
                ON document_embeddings (soeid)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_session_id 
                ON document_embeddings (session_id)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_document_embeddings_upload_timestamp 
                ON document_embeddings (upload_timestamp)
            ''')
            
            logger.info(f"Created additional indexes in {db_name}")
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vector store table in {db_name}: {str(e)}")
            return False
    
    async def validate_database(self, db_name: str) -> Dict[str, Any]:
        """Validate database setup and return status information."""
        try:
            conn = await asyncpg.connect(f"{self.base_connection_string}/{db_name}")
            
            # Check extensions
            extensions = await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector')"
            )
            
            # Check tables
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            
            # Check indexes
            indexes = await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            )
            
            await conn.close()
            
            return {
                "status": "success",
                "extensions": [ext["extname"] for ext in extensions],
                "tables": [table["tablename"] for table in tables],
                "indexes": [idx["indexname"] for idx in indexes]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def setup_all_databases(self) -> bool:
        """Set up all required databases for the RAG system."""
        logger.info("Starting PostgreSQL database setup for RAG system...")
        
        success = True
        
        for db_name, config in self.databases.items():
            logger.info(f"\n=== Setting up {db_name} ===")
            logger.info(f"Description: {config['description']}")
            
            # Create database
            if not await self.create_database(db_name):
                success = False
                continue
            
            # Setup extensions
            if not await self.setup_extensions(db_name, config["extensions"]):
                success = False
                continue
            
            # Setup vector store table if needed
            if "document_embeddings" in config["tables"]:
                if not await self.setup_vector_store_table(db_name):
                    success = False
                    continue
            
            # Validate setup
            validation = await self.validate_database(db_name)
            if validation["status"] == "success":
                logger.info(f"✅ {db_name} setup completed successfully")
                logger.info(f"   Extensions: {validation['extensions']}")
                logger.info(f"   Tables: {validation['tables']}")
                logger.info(f"   Indexes: {len(validation['indexes'])} created")
            else:
                logger.error(f"❌ {db_name} validation failed: {validation['error']}")
                success = False
        
        return success
    
    def validate_config_files(self) -> bool:
        """Validate that configuration files are consistent with database setup."""
        logger.info("\n=== Validating Configuration Files ===")
        
        config_files = [
            "config/config.yaml",
            "examples/rag/ingestion/config.yaml",
            "examples/rag/chatbot/config.yaml"
        ]
        
        success = True
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                logger.warning(f"Config file not found: {config_file}")
                continue
            
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check vector store configuration
                vector_configs = []
                
                # Check main vector store config
                if 'vector_store' in config:
                    vector_configs.append(('vector_store', config['vector_store']))
                
                # Check chatbot retrieval config
                if 'chatbot' in config and 'retrieval' in config['chatbot']:
                    if 'vector_store' in config['chatbot']['retrieval']:
                        vector_configs.append(
                            ('chatbot.retrieval.vector_store', 
                             config['chatbot']['retrieval']['vector_store'])
                        )
                
                for config_path, vector_config in vector_configs:
                    if vector_config.get('type') == 'pgvector' or vector_config.get('provider') == 'pgvector':
                        connection_string = vector_config.get('connection_string', '')
                        if 'rag_vector_db' in connection_string:
                            logger.info(f"✅ {config_file}:{config_path} - pgvector configured correctly")
                        else:
                            logger.warning(f"⚠️  {config_file}:{config_path} - connection string may be incorrect")
                    else:
                        logger.warning(f"⚠️  {config_file}:{config_path} - not using pgvector")
                
            except Exception as e:
                logger.error(f"❌ Failed to validate {config_file}: {str(e)}")
                success = False
        
        return success

async def main():
    """Main setup function."""
    setup = PgVectorDatabaseSetup()
    
    logger.info("🚀 RAG System PostgreSQL Database Setup")
    logger.info("=" * 50)
    
    # Setup databases
    if await setup.setup_all_databases():
        logger.info("\n✅ All databases set up successfully!")
    else:
        logger.error("\n❌ Some databases failed to set up properly")
        sys.exit(1)
    
    # Validate configuration files
    if setup.validate_config_files():
        logger.info("\n✅ Configuration files validated successfully!")
    else:
        logger.warning("\n⚠️  Some configuration issues detected")
    
    logger.info("\n🎉 PostgreSQL setup completed!")
    logger.info("\nNext steps:")
    logger.info("1. Verify PostgreSQL is running: pg_ctl status")
    logger.info("2. Test ingestion: python -m examples.rag.ingestion.main")
    logger.info("3. Test chatbot: python -m examples.rag.chatbot.main")
    logger.info("4. Check vector store: psql -d rag_vector_db -c '\\dt'")

if __name__ == "__main__":
    asyncio.run(main())
