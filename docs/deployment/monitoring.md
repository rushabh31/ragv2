# Monitoring & Logging

Comprehensive monitoring and logging setup for the RAG system to ensure production reliability, performance tracking, and effective troubleshooting.

## Table of Contents

- [Overview](#overview)
- [Logging Configuration](#logging-configuration)
- [Metrics Collection](#metrics-collection)
- [Health Monitoring](#health-monitoring)
- [Performance Monitoring](#performance-monitoring)
- [Error Tracking](#error-tracking)
- [Alerting](#alerting)
- [Log Aggregation](#log-aggregation)
- [Monitoring Tools](#monitoring-tools)
- [Dashboard Setup](#dashboard-setup)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The RAG system includes comprehensive monitoring and logging capabilities designed for production environments:

### Key Features
- **Structured Logging**: JSON-formatted logs with consistent fields
- **Metrics Collection**: Performance and business metrics
- **Health Checks**: System and component health monitoring
- **Error Tracking**: Comprehensive error logging and alerting
- **Performance Monitoring**: Response times, throughput, and resource usage
- **Security Monitoring**: Authentication and access logging

### Architecture
```
Application Layer
├── Structured Logging (JSON)
├── Metrics Export (Prometheus)
├── Health Endpoints
└── Error Tracking

Monitoring Layer
├── Log Aggregation (ELK/Fluentd)
├── Metrics Storage (Prometheus)
├── Alerting (AlertManager)
└── Dashboards (Grafana)

Infrastructure Layer
├── Container Metrics (cAdvisor)
├── Node Metrics (Node Exporter)
└── Network Monitoring
```

## Logging Configuration

### Application Logging

Configure structured logging in your application:

```yaml
# config.yaml
logging:
  level: INFO
  format: json
  output: stdout
  fields:
    service: rag-system
    version: "1.0.0"
    environment: production
  
  # Component-specific logging
  components:
    ingestion:
      level: INFO
      include_metrics: true
    chatbot:
      level: INFO
      include_response_times: true
    memory:
      level: DEBUG
      include_sql_queries: false
    models:
      level: INFO
      include_token_usage: true
```

### Python Logging Setup

```python
# src/utils/logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(config: Dict[str, Any]):
    """Setup application logging configuration."""
    
    # Create formatter
    formatter = JSONFormatter()
    
    # Setup handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    root_logger.addHandler(handler)
    
    # Configure component loggers
    components = config.get('components', {})
    for component, component_config in components.items():
        logger = logging.getLogger(f"rag.{component}")
        logger.setLevel(getattr(logging, component_config.get('level', 'INFO')))

# Usage in application
from src.utils.logging_config import setup_logging
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
setup_logging(config['logging'])
logger = logging.getLogger(__name__)
```

### Logging Best Practices

```python
# Example logging in RAG components
import logging
from typing import Dict, Any

logger = logging.getLogger("rag.ingestion.parser")

class VisionParser:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Vision parser initialized", extra={
            'extra_fields': {
                'component': 'vision_parser',
                'model': config.get('model'),
                'max_pages': config.get('max_pages')
            }
        })
    
    async def parse_document(self, file_path: str) -> List[Document]:
        start_time = time.time()
        
        logger.info("Starting document parsing", extra={
            'extra_fields': {
                'file_path': file_path,
                'operation': 'parse_document'
            }
        })
        
        try:
            documents = await self._parse_file(file_path)
            
            processing_time = time.time() - start_time
            logger.info("Document parsing completed", extra={
                'extra_fields': {
                    'file_path': file_path,
                    'documents_count': len(documents),
                    'processing_time_seconds': processing_time,
                    'pages_per_second': len(documents) / processing_time
                }
            })
            
            return documents
            
        except Exception as e:
            logger.error("Document parsing failed", extra={
                'extra_fields': {
                    'file_path': file_path,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            }, exc_info=True)
            raise
```

## Metrics Collection

### Prometheus Metrics

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total number of requests',
    ['component', 'operation', 'status']
)

REQUEST_DURATION = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['component', 'operation']
)

ACTIVE_SESSIONS = Gauge(
    'rag_active_sessions',
    'Number of active chat sessions'
)

MODEL_TOKEN_USAGE = Counter(
    'rag_model_tokens_total',
    'Total tokens used by models',
    ['provider', 'model', 'type']
)

DOCUMENT_COUNT = Gauge(
    'rag_documents_total',
    'Total number of documents in vector store'
)

def track_metrics(component: str, operation: str):
    """Decorator to track metrics for operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(
                    component=component,
                    operation=operation,
                    status='success'
                ).inc()
                return result
                
            except Exception as e:
                REQUEST_COUNT.labels(
                    component=component,
                    operation=operation,
                    status='error'
                ).inc()
                raise
                
            finally:
                REQUEST_DURATION.labels(
                    component=component,
                    operation=operation
                ).observe(time.time() - start_time)
                
        return wrapper
    return decorator

# Usage example
@track_metrics('ingestion', 'parse_document')
async def parse_document(self, file_path: str):
    # Implementation
    pass

# Start metrics server
def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server."""
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")
```

## Health Monitoring

### Health Check Endpoints

```python
# src/api/health.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
import time

router = APIRouter()

class HealthChecker:
    """Comprehensive health checking for RAG system."""
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check PostgreSQL database health."""
        try:
            from src.memory.langgraph_checkpoint_memory import LangGraphCheckpointMemory
            memory = LangGraphCheckpointMemory({
                'store_type': 'postgres',
                'postgres': {
                    'connection_string': 'postgresql://...'
                }
            })
            
            # Simple health check
            await memory.add('health_check', 'test', 'test', {})
            await memory.clear_session('health_check')
            
            return {
                'status': 'healthy',
                'response_time_ms': 50,
                'details': 'Database connection successful'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Database connection failed'
            }
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check AI model health."""
        try:
            from src.models.generation import GenerationModelFactory
            
            # Test model authentication
            model = GenerationModelFactory.create_model('vertex')
            health_status = await model.get_auth_health_status()
            
            return {
                'status': 'healthy' if health_status['authenticated'] else 'unhealthy',
                'details': health_status
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Model authentication failed'
            }

health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check for all components."""
    
    checks = await asyncio.gather(
        health_checker.check_database_health(),
        health_checker.check_model_health(),
        return_exceptions=True
    )
    
    database_health, model_health = checks
    
    overall_status = "healthy"
    if any(check.get('status') == 'unhealthy' for check in checks):
        overall_status = "unhealthy"
    
    return {
        "overall_status": overall_status,
        "timestamp": time.time(),
        "components": {
            "database": database_health,
            "models": model_health
        }
    }

@router.get("/health/readiness")
async def readiness_check():
    """Kubernetes readiness probe."""
    try:
        detailed_health = await detailed_health_check()
        if detailed_health["overall_status"] == "healthy":
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": time.time()}
```

## Performance Monitoring

### Response Time Tracking

```python
# src/utils/performance.py
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration: float
    timestamp: float
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """Performance monitoring and tracking."""
    
    def __init__(self, max_samples: int = 1000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.max_samples = max_samples
    
    def record_operation(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Record operation performance."""
        metric = PerformanceMetrics(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.metrics[operation].append(metric)
    
    def get_statistics(self, operation: str) -> Dict[str, Any]:
        """Get performance statistics for operation."""
        if operation not in self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics[operation]]
        
        return {
            'operation': operation,
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'p95_duration': self._percentile(durations, 95),
            'p99_duration': self._percentile(durations, 99)
        }
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

# Global performance monitor
performance_monitor = PerformanceMonitor()
```

## Error Tracking

### Error Logging and Categorization

```python
# src/utils/error_tracking.py
import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
from collections import defaultdict

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    AUTHENTICATION = "authentication"
    MODEL_API = "model_api"
    DATABASE = "database"
    VALIDATION = "validation"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    operation: str
    timestamp: float
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorTracker:
    """Centralized error tracking and reporting."""
    
    def __init__(self):
        self.logger = logging.getLogger("rag.errors")
        self.error_counts = defaultdict(int)
    
    def track_error(self, 
                   error: Exception,
                   component: str,
                   operation: str,
                   category: ErrorCategory = ErrorCategory.UNKNOWN,
                   severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track and log error with structured information."""
        
        error_id = f"{component}_{operation}_{int(time.time())}"
        
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            component=component,
            operation=operation,
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            metadata=metadata
        )
        
        # Log structured error
        self.logger.error("Error tracked", extra={
            'extra_fields': {
                'error_id': error_id,
                'category': category.value,
                'severity': severity.value,
                'component': component,
                'operation': operation,
                'error_type': type(error).__name__,
                'metadata': metadata
            }
        })
        
        # Update error counts
        self.error_counts[f"{component}_{category.value}"] += 1
        
        return error_id

# Global error tracker
error_tracker = ErrorTracker()
```

## Alerting

### Alert Configuration

```yaml
# alerts.yaml
alerts:
  rules:
    - name: high_error_rate
      condition: error_rate > 0.05  # 5% error rate
      duration: 5m
      severity: warning
      channels: ["slack", "email"]
      
    - name: response_time_high
      condition: avg_response_time > 10s
      duration: 2m
      severity: critical
      channels: ["pagerduty", "slack"]
      
    - name: database_down
      condition: database_health != "healthy"
      duration: 1m
      severity: critical
      channels: ["pagerduty", "slack", "email"]
      
    - name: model_authentication_failed
      condition: model_auth_failures > 3
      duration: 5m
      severity: high
      channels: ["slack", "email"]

  channels:
    slack:
      webhook_url: "https://hooks.slack.com/services/..."
      channel: "#rag-alerts"
      
    email:
      smtp_server: "smtp.company.com"
      recipients: ["team@company.com"]
      
    pagerduty:
      integration_key: "your-pagerduty-key"
```

## Log Aggregation

### ELK Stack Configuration

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "rag-system" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "rag-logs-%{+YYYY.MM.dd}"
  }
}
```

## Monitoring Tools

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['localhost:8000']  # Metrics endpoint
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_requests_total[5m])",
            "legendFormat": "{{component}} - {{operation}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(rag_requests_total{status=\"error\"}[5m]) / rate(rag_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### Monitoring Best Practices

1. **Structured Logging**:
   ```python
   # Good: Structured logging with consistent fields
   logger.info("Document processed", extra={
       'extra_fields': {
           'document_id': doc_id,
           'processing_time': duration,
           'page_count': pages
       }
   })
   ```

2. **Meaningful Metrics**:
   ```python
   # Track business metrics, not just technical metrics
   DOCUMENTS_PROCESSED = Counter('documents_processed_total', ['type', 'status'])
   QUERY_ACCURACY = Histogram('query_accuracy_score', ['model'])
   ```

3. **Alert Fatigue Prevention**:
   - Set appropriate thresholds
   - Use alert grouping and suppression
   - Implement escalation policies
   - Regular alert review and tuning

4. **Performance Monitoring**:
   - Monitor end-to-end user experience
   - Track resource utilization trends
   - Set up capacity planning alerts
   - Monitor external dependencies

### Security Monitoring

```python
# Security event logging
security_logger = logging.getLogger("rag.security")

def log_authentication_event(event_type: str, user_id: str, success: bool):
    security_logger.info("Authentication event", extra={
        'extra_fields': {
            'event_type': event_type,
            'user_id': user_id,
            'success': success,
            'ip_address': request.client.host
        }
    })
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**:
   - Check for memory leaks in model loading
   - Monitor vector store memory usage
   - Implement proper cleanup in parsers

2. **Slow Response Times**:
   - Check model API latency
   - Monitor database query performance
   - Verify network connectivity

3. **Authentication Failures**:
   - Check token expiration
   - Verify environment variables
   - Monitor OAuth2 flow

4. **Database Connection Issues**:
   - Check PostgreSQL health
   - Monitor connection pool
   - Verify network connectivity

### Debug Commands

```bash
# Check application logs
kubectl logs -f deployment/rag-system -n rag-system

# Check metrics endpoint
curl http://localhost:8000/metrics

# Check health endpoints
curl http://localhost:8001/health/detailed

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana dashboards
curl http://localhost:3000/api/dashboards/home
```

This monitoring and logging setup provides comprehensive observability for the RAG system, enabling proactive monitoring, quick troubleshooting, and performance optimization in production environments.
