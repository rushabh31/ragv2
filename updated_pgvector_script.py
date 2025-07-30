# Store the generated vector embeddings in a PostgreSQL table.
# This code may run for a few minutes.

import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration variables - update these with your actual values
connection_string = "postgresql://rushabhsmacbook@localhost:5432/rag_vector_db"
schema_name = "public"  # or your preferred schema
table_name = "product_embeddings"

async def main():
    # Create connection to PostgreSQL database using asyncpg
    conn: asyncpg.Connection = await asyncpg.connect(connection_string)

    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        logger.info("Enabled pgvector extension")

        # Create schema if it doesn't exist (skip for 'public' schema)
        if schema_name != 'public':
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name}')
            logger.info(f"Created schema: {schema_name}")

        # Create fully qualified table name
        qualified_table_name = f"{schema_name}.{table_name}"

        # Drop existing table if needed (optional - remove if you want to keep existing data)
        await conn.execute(f"DROP TABLE IF EXISTS {qualified_table_name}")
        logger.info(f"Dropped existing table: {qualified_table_name}")

        # Create the enhanced `product_embeddings` table with additional metadata columns
        await conn.execute(f"""
            CREATE TABLE {qualified_table_name} (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(1024) NOT NULL,
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
                chunk_index INTEGER,
                UNIQUE(product_id)
            )
        """)
        logger.info(f"Created table: {qualified_table_name}")

        # Create indexes for efficient querying
        await conn.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_product_id_idx ON {qualified_table_name} (product_id)')
        await conn.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_soeid_idx ON {qualified_table_name} (soeid)')
        await conn.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_session_id_idx ON {qualified_table_name} (session_id)')
        await conn.execute(f'CREATE INDEX IF NOT EXISTS {table_name}_upload_timestamp_idx ON {qualified_table_name} (upload_timestamp)')
        
        # Create vector index (HNSW for better performance)
        try:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx 
                ON {qualified_table_name} 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
            logger.info("Created HNSW vector index")
        except Exception as e:
            logger.warning(f"Failed to create HNSW index, trying IVFFlat: {str(e)}")
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                    ON {qualified_table_name} 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100)
                """)
                logger.info("Created IVFFlat vector index")
            except Exception as e2:
                logger.warning(f"Failed to create vector index: {str(e2)}")

        # Store all the generated embeddings back into the database using UPSERT
        logger.info(f"Starting to insert {len(product_embeddings)} embeddings")
        
        for index, row in product_embeddings.iterrows():
            try:
                # Prepare metadata (you can add more fields as needed)
                metadata = {
                    "source": "product_catalog",
                    "index": index,
                    "processed_at": datetime.now().isoformat()
                }
                metadata_json = json.dumps(metadata)
                
                # Convert embedding to numpy array if it isn't already
                embedding_array = np.array(row["embedding"]) if not isinstance(row["embedding"], np.ndarray) else row["embedding"]
                
                # Use UPSERT (INSERT ... ON CONFLICT) for robust insertion
                await conn.execute(f"""
                    INSERT INTO {qualified_table_name} 
                    (product_id, content, embedding, metadata, created_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                    ON CONFLICT (product_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        upload_timestamp = CURRENT_TIMESTAMP
                """,
                    row["product_id"],
                    row["content"],
                    embedding_array,
                    metadata_json
                )
                
                if (index + 1) % 100 == 0:  # Log progress every 100 insertions
                    logger.info(f"Inserted {index + 1}/{len(product_embeddings)} embeddings")
                    
            except Exception as e:
                logger.error(f"Failed to insert embedding for product {row['product_id']}: {str(e)}")
                # Continue with next embedding instead of failing completely
                continue

        logger.info(f"Successfully inserted embeddings into {qualified_table_name}")
        
        # Verify the insertion
        count_result = await conn.fetchval(f"SELECT COUNT(*) FROM {qualified_table_name}")
        logger.info(f"Total embeddings in table: {count_result}")

    except Exception as e:
        logger.error(f"Error during database operations: {str(e)}")
        raise
    finally:
        await conn.close()
        logger.info("Database connection closed")


# Example usage with error handling
async def run_with_error_handling():
    """Run the main function with proper error handling."""
    try:
        await main()
        logger.info("Embedding storage completed successfully!")
    except Exception as e:
        logger.error(f"Failed to store embeddings: {str(e)}")
        raise


# Run the SQL commands now.
if __name__ == "__main__":
    # For Jupyter notebook, use: await run_with_error_handling()
    # For regular Python script, use:
    asyncio.run(run_with_error_handling())
