# PostgreSQL Setup

## 🎯 Overview

PostgreSQL serves as the persistent storage backend for the RAG system's memory components, providing enterprise-grade reliability, scalability, and performance. This guide covers complete PostgreSQL setup, configuration, optimization, and maintenance for the LangGraph checkpoint memory system with pgvector support for vector operations.

## 🚀 Quick Start

### **Installation**

**macOS (Homebrew):**
```bash
# Install PostgreSQL
brew install postgresql@15

# Install pgvector extension
brew install pgvector

# Start PostgreSQL service
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15

# Install pgvector
sudo apt install postgresql-15-pgvector

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Docker (Recommended for Development):**
```bash
# Run PostgreSQL with pgvector
docker run --name rag-postgres \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpass \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg15
```

### **Database Setup**
```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE ragdb;
CREATE USER raguser WITH ENCRYPTED PASSWORD 'ragpass';
GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;

# Connect to the RAG database
\c ragdb

# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Grant permissions
GRANT ALL ON SCHEMA public TO raguser;
```

## 🏗️ Database Schema

### **LangGraph Tables**
The LangGraph checkpoint system automatically creates these tables:

```sql
-- Checkpoints table (main conversation state)
CREATE TABLE IF NOT EXISTS rag_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes table (individual message writes)
CREATE TABLE IF NOT EXISTS rag_checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Checkpoint blobs table (large data storage)
CREATE TABLE IF NOT EXISTS rag_checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, type)
);

-- Migrations table (schema versioning)
CREATE TABLE IF NOT EXISTS rag_checkpoint_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### **Performance Indexes**
```sql
-- Essential indexes for performance
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id 
ON rag_checkpoints(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at 
ON rag_checkpoints(created_at);

CREATE INDEX IF NOT EXISTS idx_checkpoints_metadata_soeid 
ON rag_checkpoints USING GIN ((metadata->>'soeid'));

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id 
ON rag_checkpoint_writes(thread_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_created_at 
ON rag_checkpoint_writes(created_at);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_channel 
ON rag_checkpoint_writes(channel);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_created 
ON rag_checkpoints(thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_writes_thread_checkpoint 
ON rag_checkpoint_writes(thread_id, checkpoint_id);
```

### **Additional Indexes for Chat History**
```sql
-- Indexes for SOEID-based queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_soeid_created 
ON rag_checkpoints((metadata->>'soeid'), created_at DESC) 
WHERE metadata->>'soeid' IS NOT NULL;

-- Indexes for temporal queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_date 
ON rag_checkpoints(DATE(created_at));

-- Partial indexes for active sessions
CREATE INDEX IF NOT EXISTS idx_checkpoints_recent 
ON rag_checkpoints(thread_id, created_at DESC) 
WHERE created_at > NOW() - INTERVAL '30 days';
```

## 📋 Configuration

### **PostgreSQL Configuration (postgresql.conf)**
```ini
# Memory settings
shared_buffers = 256MB                  # 25% of RAM for dedicated server
effective_cache_size = 1GB              # 75% of RAM
work_mem = 16MB                         # Per-operation memory
maintenance_work_mem = 256MB            # Maintenance operations

# Connection settings
max_connections = 100                   # Adjust based on load
listen_addresses = '*'                  # Listen on all interfaces
port = 5432                            # Default PostgreSQL port

# Write-ahead logging
wal_buffers = 16MB                     # WAL buffer size
checkpoint_completion_target = 0.9      # Checkpoint completion target
max_wal_size = 1GB                     # Maximum WAL size
min_wal_size = 80MB                    # Minimum WAL size

# Query planner
random_page_cost = 1.1                 # SSD optimization
effective_io_concurrency = 200         # SSD optimization

# Logging
log_destination = 'stderr'             # Log destination
logging_collector = on                 # Enable log collector
log_directory = 'log'                  # Log directory
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000      # Log slow queries (1 second)
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Performance monitoring
track_activities = on                  # Track activities
track_counts = on                      # Track statistics
track_io_timing = on                   # Track I/O timing
track_functions = pl                   # Track function calls
```

### **Connection Pool Configuration (pgbouncer.ini)**
```ini
[databases]
ragdb = host=localhost port=5432 dbname=ragdb user=raguser

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Pool settings
pool_mode = transaction                # Transaction pooling
max_client_conn = 200                  # Maximum client connections
default_pool_size = 25                 # Default pool size
min_pool_size = 5                      # Minimum pool size
reserve_pool_size = 5                  # Reserve pool size

# Connection limits
server_lifetime = 3600                 # Server connection lifetime
server_idle_timeout = 600              # Server idle timeout
client_idle_timeout = 0                # Client idle timeout (disabled)

# Logging
log_connections = 1                    # Log connections
log_disconnections = 1                 # Log disconnections
log_pooler_errors = 1                  # Log pooler errors
```

### **Application Configuration**
```yaml
# config.yaml
chatbot:
  memory:
    type: "langgraph_checkpoint"
    store_type: "postgres"
    
    postgres:
      # Connection settings
      connection_string: "postgresql://raguser:ragpass@localhost:5432/ragdb"
      
      # Pool configuration
      pool_size: 20                    # Connection pool size
      max_overflow: 30                 # Additional connections
      pool_timeout: 30                 # Connection timeout (seconds)
      pool_recycle: 3600               # Recycle connections (1 hour)
      pool_pre_ping: true              # Validate connections
      
      # Table configuration
      table_prefix: "rag_"             # Table name prefix
      schema: "public"                 # Database schema
      
      # Performance settings
      batch_size: 100                  # Batch operations
      connection_timeout: 60           # Connection timeout
      query_timeout: 30                # Query timeout
      statement_timeout: 45            # Statement timeout
      
      # Retry configuration
      retry_attempts: 3                # Connection retry attempts
      retry_delay: 1                   # Retry delay (seconds)
      
      # SSL settings (for production)
      ssl_mode: "require"              # SSL mode
      ssl_cert: "/path/to/client.crt"  # Client certificate
      ssl_key: "/path/to/client.key"   # Client key
      ssl_ca: "/path/to/ca.crt"        # CA certificate
```

## 🛠️ Database Management

### **Connection Management**
```python
import asyncpg
import asyncio
from contextlib import asynccontextmanager

class PostgreSQLManager:
    def __init__(self, config):
        self.config = config
        self.pool = None
    
    async def initialize_pool(self):
        """Initialize connection pool."""
        
        self.pool = await asyncpg.create_pool(
            dsn=self.config["connection_string"],
            min_size=self.config.get("pool_size", 10),
            max_size=self.config.get("max_overflow", 20),
            command_timeout=self.config.get("query_timeout", 30),
            server_settings={
                'application_name': 'rag_chatbot',
                'timezone': 'UTC'
            }
        )
        
        # Test connection
        async with self.pool.acquire() as conn:
            await conn.execute('SELECT 1')
        
        logger.info("PostgreSQL connection pool initialized")
    
    async def close_pool(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        
        if not self.pool:
            await self.initialize_pool()
        
        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                # Log error and re-raise
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def execute_query(self, query, *args):
        """Execute a query with error handling."""
        
        async with self.get_connection() as conn:
            try:
                return await conn.fetch(query, *args)
            except asyncpg.PostgresError as e:
                logger.error(f"PostgreSQL error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

# Usage example
async def main():
    config = {
        "connection_string": "postgresql://raguser:ragpass@localhost:5432/ragdb",
        "pool_size": 10,
        "max_overflow": 20,
        "query_timeout": 30
    }
    
    db_manager = PostgreSQLManager(config)
    
    try:
        # Initialize pool
        await db_manager.initialize_pool()
        
        # Execute query
        result = await db_manager.execute_query(
            "SELECT COUNT(*) FROM rag_checkpoints WHERE created_at > $1",
            datetime.now() - timedelta(days=7)
        )
        
        print(f"Recent checkpoints: {result[0]['count']}")
        
    finally:
        await db_manager.close_pool()
```

### **Database Maintenance**
```python
class DatabaseMaintenance:
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    async def cleanup_old_data(self, days_to_keep=30):
        """Clean up old checkpoint data."""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.db_manager.get_connection() as conn:
            # Start transaction
            async with conn.transaction():
                # Delete old checkpoint writes
                writes_deleted = await conn.execute(
                    "DELETE FROM rag_checkpoint_writes WHERE created_at < $1",
                    cutoff_date
                )
                
                # Delete old checkpoint blobs
                blobs_deleted = await conn.execute(
                    "DELETE FROM rag_checkpoint_blobs WHERE created_at < $1",
                    cutoff_date
                )
                
                # Delete old checkpoints
                checkpoints_deleted = await conn.execute(
                    "DELETE FROM rag_checkpoints WHERE created_at < $1",
                    cutoff_date
                )
                
                logger.info(f"Cleanup completed: {checkpoints_deleted} checkpoints, "
                          f"{writes_deleted} writes, {blobs_deleted} blobs deleted")
    
    async def vacuum_tables(self):
        """Vacuum and analyze tables for performance."""
        
        tables = [
            "rag_checkpoints",
            "rag_checkpoint_writes", 
            "rag_checkpoint_blobs"
        ]
        
        async with self.db_manager.get_connection() as conn:
            for table in tables:
                await conn.execute(f"VACUUM ANALYZE {table}")
                logger.info(f"Vacuumed table: {table}")
    
    async def reindex_tables(self):
        """Rebuild indexes for optimal performance."""
        
        indexes = [
            "idx_checkpoints_thread_id",
            "idx_checkpoints_created_at",
            "idx_checkpoints_metadata_soeid",
            "idx_checkpoint_writes_thread_id",
            "idx_checkpoint_writes_created_at"
        ]
        
        async with self.db_manager.get_connection() as conn:
            for index in indexes:
                await conn.execute(f"REINDEX INDEX {index}")
                logger.info(f"Reindexed: {index}")
    
    async def get_database_stats(self):
        """Get comprehensive database statistics."""
        
        async with self.db_manager.get_connection() as conn:
            # Table sizes
            table_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE tablename LIKE 'rag_%'
                ORDER BY size_bytes DESC
            """)
            
            # Index usage
            index_stats = await conn.fetch("""
                SELECT 
                    indexrelname as index_name,
                    idx_tup_read,
                    idx_tup_fetch,
                    idx_scan
                FROM pg_stat_user_indexes 
                WHERE indexrelname LIKE 'idx_%'
                ORDER BY idx_scan DESC
            """)
            
            # Connection stats
            connection_stats = await conn.fetch("""
                SELECT 
                    state,
                    COUNT(*) as count
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state
            """)
            
            return {
                "table_stats": [dict(row) for row in table_stats],
                "index_stats": [dict(row) for row in index_stats],
                "connection_stats": [dict(row) for row in connection_stats]
            }
```

## 📊 Performance Optimization

### **Query Optimization**
```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM rag_checkpoints 
WHERE metadata->>'soeid' = 'john.doe' 
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC 
LIMIT 20;

-- Optimize with proper indexing
CREATE INDEX CONCURRENTLY idx_checkpoints_soeid_recent 
ON rag_checkpoints((metadata->>'soeid'), created_at DESC) 
WHERE created_at > NOW() - INTERVAL '30 days';

-- Partition large tables by date
CREATE TABLE rag_checkpoints_2024_01 PARTITION OF rag_checkpoints
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Analyze table statistics
ANALYZE rag_checkpoints;
```

### **Connection Pool Tuning**
```python
# Optimal connection pool configuration
POOL_CONFIG = {
    # For high-traffic applications
    "high_traffic": {
        "pool_size": 30,
        "max_overflow": 50,
        "pool_timeout": 10,
        "pool_recycle": 1800  # 30 minutes
    },
    
    # For moderate traffic
    "moderate_traffic": {
        "pool_size": 15,
        "max_overflow": 25,
        "pool_timeout": 20,
        "pool_recycle": 3600  # 1 hour
    },
    
    # For development
    "development": {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 7200  # 2 hours
    }
}

def get_pool_config(environment="moderate_traffic"):
    """Get optimized pool configuration for environment."""
    return POOL_CONFIG.get(environment, POOL_CONFIG["moderate_traffic"])
```

### **Monitoring Queries**
```sql
-- Monitor slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
WHERE query LIKE '%rag_%'
ORDER BY mean_time DESC 
LIMIT 10;

-- Monitor table bloat
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables 
WHERE tablename LIKE 'rag_%';

-- Monitor index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexrelname LIKE 'idx_%'
ORDER BY idx_scan DESC;
```

## 🔒 Security Configuration

### **Authentication Setup**
```sql
-- Create dedicated user with limited privileges
CREATE USER rag_app WITH ENCRYPTED PASSWORD 'secure_password';

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE ragdb TO rag_app;
GRANT USAGE ON SCHEMA public TO rag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_app;

-- Create read-only user for analytics
CREATE USER rag_readonly WITH ENCRYPTED PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE ragdb TO rag_readonly;
GRANT USAGE ON SCHEMA public TO rag_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;
```

### **SSL Configuration**
```ini
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_ca_file = 'ca.crt'
ssl_crl_file = 'server.crl'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on
```

### **Connection Security**
```python
# Secure connection string
SECURE_CONNECTION_STRING = (
    "postgresql://rag_app:secure_password@localhost:5432/ragdb"
    "?sslmode=require"
    "&sslcert=/path/to/client.crt"
    "&sslkey=/path/to/client.key"
    "&sslrootcert=/path/to/ca.crt"
)

# Connection with SSL verification
async def create_secure_connection():
    """Create secure database connection."""
    
    return await asyncpg.connect(
        dsn=SECURE_CONNECTION_STRING,
        server_settings={
            'application_name': 'rag_chatbot_secure',
            'timezone': 'UTC'
        },
        ssl='require'
    )
```

## 🚨 Backup and Recovery

### **Automated Backup Script**
```bash
#!/bin/bash
# backup_rag_db.sh

DB_NAME="ragdb"
DB_USER="raguser"
BACKUP_DIR="/var/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/ragdb_backup_${DATE}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
pg_dump -U $DB_USER -h localhost -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove backups older than 30 days
find $BACKUP_DIR -name "ragdb_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

### **Point-in-Time Recovery Setup**
```ini
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
max_wal_senders = 3
checkpoint_completion_target = 0.9
```

### **Recovery Script**
```bash
#!/bin/bash
# restore_rag_db.sh

BACKUP_FILE="$1"
DB_NAME="ragdb"
DB_USER="raguser"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
systemctl stop rag-chatbot

# Drop and recreate database
dropdb -U postgres $DB_NAME
createdb -U postgres $DB_NAME
psql -U postgres -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Restore from backup
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | psql -U $DB_USER -d $DB_NAME
else
    psql -U $DB_USER -d $DB_NAME < $BACKUP_FILE
fi

# Start application
systemctl start rag-chatbot

echo "Database restored from: $BACKUP_FILE"
```

## 🔍 Troubleshooting

### **Common Issues and Solutions**

**Connection Pool Exhaustion:**
```python
# Issue: "pool exhausted" errors
# Solution: Increase pool size or optimize connection usage

# Monitor pool status
async def monitor_pool_status(pool):
    """Monitor connection pool status."""
    
    status = {
        "size": pool.get_size(),
        "checked_in": pool.get_size() - pool.get_idle_size(),
        "checked_out": pool.get_idle_size(),
        "overflow": pool.get_overflow(),
        "invalid": pool.get_invalidated()
    }
    
    logger.info(f"Pool status: {status}")
    
    if status["checked_out"] / status["size"] > 0.8:
        logger.warning("Pool utilization high, consider increasing pool size")
    
    return status
```

**Slow Queries:**
```sql
-- Find slow queries
SELECT 
    query,
    calls,
    total_time / calls as avg_time_ms,
    rows / calls as avg_rows
FROM pg_stat_statements 
WHERE total_time / calls > 1000  -- Slower than 1 second
ORDER BY avg_time_ms DESC;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_missing_index 
ON table_name(column_name);
```

**Lock Contention:**
```sql
-- Monitor locks
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity 
    ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity 
    ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

## 📚 Related Documentation

- **[Memory Systems Overview](./README.md)** - Complete memory architecture
- **[Chat History](./chat-history.md)** - Chat history implementation
- **[Memory API](./api.md)** - Memory management endpoints

## 🚀 Quick Setup Commands

### **Docker Setup**
```bash
# Start PostgreSQL with pgvector
docker run --name rag-postgres \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpass \
  -p 5432:5432 \
  -d pgvector/pgvector:pg15

# Connect and setup
docker exec -it rag-postgres psql -U raguser -d ragdb
```

### **Local Setup**
```bash
# Install and start PostgreSQL
brew install postgresql@15 pgvector
brew services start postgresql@15

# Create database
createdb ragdb
psql ragdb -c "CREATE EXTENSION vector;"
```

---

**Next Steps**: 
- [Configure Memory API](./api.md)
- [Set up Chat History](./chat-history.md)
- [Use Memory Systems](./README.md)
