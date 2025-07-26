# Deployment Guide

This guide provides comprehensive instructions for deploying the RAG system in various environments, from development to production. It covers containerization, cloud deployment, scaling strategies, security considerations, and operational best practices.

## Overview

The RAG system supports multiple deployment patterns:

- **Local Development**: Single-machine setup for development and testing
- **Docker Containers**: Containerized deployment for consistency and portability
- **Cloud Native**: Kubernetes deployment with auto-scaling and high availability
- **Serverless**: Function-based deployment for cost optimization
- **Hybrid**: Mixed deployment patterns for different components

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│   (nginx/ALB)   │    │   (Kong/AWS)    │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Ingestion API  │    │  Chatbot API    │    │   Grafana       │
│   (FastAPI)     │    │   (FastAPI)     │    │  (Dashboard)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
         ▼                       ▼                       
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Store   │    │  Memory Store   │    │  File Storage   │
│    (FAISS)      │    │ (PostgreSQL)    │    │    (S3/GCS)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Dependencies

- **Ingestion Service**: Requires vision models, embedding models, vector storage
- **Chatbot Service**: Requires generation models, retrieval, memory, vector storage
- **Memory Service**: Requires PostgreSQL with pgvector extension
- **Authentication**: Universal OAuth2 token service
- **Storage**: File storage for documents, vector storage for embeddings

## Local Development Deployment

### Prerequisites

```bash
# System requirements
Python 3.8+
PostgreSQL 12+
Docker 20.10+
Docker Compose 2.0+

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

**1. Environment Variables:**
```bash
# .env file
# Universal Authentication
COIN_CONSUMER_ENDPOINT_URL=https://oauth-server/oauth2/token
COIN_CONSUMER_CLIENT_ID=your-client-id
COIN_CONSUMER_CLIENT_SECRET=your-client-secret
COIN_CONSUMER_SCOPE=https://www.googleapis.com/auth/cloud-platform

# Google Cloud
PROJECT_ID=your-gcp-project-id
VERTEXAI_API_ENDPOINT=us-central1-aiplatform.googleapis.com

# Database
POSTGRES_CONNECTION_STRING=postgresql://rag_user:rag_password@localhost:5432/rag_db

# API Keys (optional)
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key

# Service Configuration
INGESTION_PORT=8000
CHATBOT_PORT=8001
LOG_LEVEL=INFO
```

**2. Database Setup:**
```bash
# Create PostgreSQL database
createdb rag_db

# Create user and grant permissions
psql rag_db -c "
CREATE USER rag_user WITH PASSWORD 'rag_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
CREATE EXTENSION IF NOT EXISTS vector;
"

# Initialize LangGraph tables
python scripts/setup_langgraph_db.py
```

**3. Start Services:**
```bash
# Start ingestion service
cd examples/rag/ingestion
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start chatbot service (in another terminal)
cd examples/rag/chatbot
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker Deployment

### Containerization

**Multi-stage Dockerfile:**
```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app examples/ ./examples/
COPY --chown=app:app config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "examples.rag.ingestion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  ingestion:
    build:
      context: .
      dockerfile: Dockerfile.ingestion
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_CONNECTION_STRING=postgresql://rag_user:rag_password@postgres:5432/rag_db
      - COIN_CONSUMER_ENDPOINT_URL=${COIN_CONSUMER_ENDPOINT_URL}
      - COIN_CONSUMER_CLIENT_ID=${COIN_CONSUMER_CLIENT_ID}
      - COIN_CONSUMER_CLIENT_SECRET=${COIN_CONSUMER_CLIENT_SECRET}
      - PROJECT_ID=${PROJECT_ID}
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  chatbot:
    build:
      context: .
      dockerfile: Dockerfile.chatbot
    ports:
      - "8001:8001"
    environment:
      - POSTGRES_CONNECTION_STRING=postgresql://rag_user:rag_password@postgres:5432/rag_db
      - COIN_CONSUMER_ENDPOINT_URL=${COIN_CONSUMER_ENDPOINT_URL}
      - COIN_CONSUMER_CLIENT_ID=${COIN_CONSUMER_CLIENT_ID}
      - COIN_CONSUMER_CLIENT_SECRET=${COIN_CONSUMER_CLIENT_SECRET}
      - PROJECT_ID=${PROJECT_ID}
    depends_on:
      postgres:
        condition: service_healthy
      ingestion:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ingestion
      - chatbot

volumes:
  postgres_data:
```

### Container Commands

```bash
# Build images
docker build -f Dockerfile.ingestion -t rag-ingestion:latest .
docker build -f Dockerfile.chatbot -t rag-chatbot:latest .

# Run with Docker Compose
docker-compose up -d

# Scale services
docker-compose up -d --scale chatbot=3 --scale ingestion=2

# View logs
docker-compose logs -f chatbot
docker-compose logs -f ingestion

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Basic Kubernetes Manifests

**1. Namespace and ConfigMap:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  POSTGRES_DB: "rag_db"
  POSTGRES_USER: "rag_user"
  LOG_LEVEL: "INFO"
  INGESTION_PORT: "8000"
  CHATBOT_PORT: "8001"
```

**2. Application Deployment:**
```yaml
# k8s/ingestion.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingestion
  namespace: rag-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ingestion
  template:
    metadata:
      labels:
        app: ingestion
    spec:
      containers:
      - name: ingestion
        image: rag-ingestion:latest
        ports:
        - containerPort: 8000
        env:
        - name: POSTGRES_CONNECTION_STRING
          value: "postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@postgres-service:5432/$(POSTGRES_DB)"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: ingestion-service
  namespace: rag-system
spec:
  selector:
    app: ingestion
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

**3. Auto-scaling:**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ingestion-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ingestion
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Kubernetes Commands

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-system
kubectl get services -n rag-system
kubectl get hpa -n rag-system

# View logs
kubectl logs -f deployment/ingestion -n rag-system

# Scale manually
kubectl scale deployment ingestion --replicas=5 -n rag-system

# Port forward for testing
kubectl port-forward service/ingestion-service 8000:8000 -n rag-system
```

## Cloud Platform Deployment

### Google Cloud Platform (GCP)

```bash
# Create GKE cluster
gcloud container clusters create rag-cluster \
    --zone=us-central1-a \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10

# Get credentials
gcloud container clusters get-credentials rag-cluster --zone=us-central1-a

# Deploy to GKE
kubectl apply -f k8s/
```

### Amazon Web Services (AWS)

```bash
# Create EKS cluster
eksctl create cluster \
    --name rag-cluster \
    --region us-west-2 \
    --nodegroup-name standard-workers \
    --node-type m5.large \
    --nodes 3 \
    --nodes-min 1 \
    --nodes-max 10

# Deploy to EKS
kubectl apply -f k8s/
```

## Security

### Authentication and Authorization

**Service Account Setup:**
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-service-account
  namespace: rag-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rag-role
  namespace: rag-system
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
```

### SSL/TLS Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.rag-system.com
    secretName: rag-tls
  rules:
  - host: api.rag-system.com
    http:
      paths:
      - path: /ingestion
        pathType: Prefix
        backend:
          service:
            name: ingestion-service
            port:
              number: 8000
```

## Monitoring and Observability

### Metrics Collection

**Prometheus Configuration:**
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-ingestion'
    static_configs:
      - targets: ['ingestion-service:8000']
    metrics_path: /metrics

  - job_name: 'rag-chatbot'
    static_configs:
      - targets: ['chatbot-service:8001']
    metrics_path: /metrics
```

### Logging

**Centralized Logging:**
```yaml
# logging/fluentd.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    metadata:
      labels:
        name: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
```

## Performance Optimization

### Scaling Strategies

**1. Horizontal Pod Autoscaling:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chatbot
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**2. Resource Optimization:**
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Caching

**Redis Cache:**
```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: rag-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Troubleshooting

### Common Issues

**1. Pod Startup Issues:**
```bash
# Check pod status
kubectl get pods -n rag-system

# Describe pod for events
kubectl describe pod <pod-name> -n rag-system

# Check logs
kubectl logs <pod-name> -n rag-system
```

**2. Service Discovery Issues:**
```bash
# Test service connectivity
kubectl exec -it <pod-name> -n rag-system -- curl http://ingestion-service:8000/health

# Check service endpoints
kubectl get endpoints -n rag-system
```

**3. Resource Issues:**
```bash
# Check resource usage
kubectl top pods -n rag-system
kubectl top nodes

# Check resource quotas
kubectl describe resourcequota -n rag-system
```

### Health Checks

**Application Health Endpoints:**
```python
# Add to FastAPI applications
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    # Check dependencies (database, external APIs)
    try:
        # Test database connection
        await test_database_connection()
        # Test external API
        await test_external_api()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")
```

## Best Practices

### Deployment Best Practices

1. **Use multi-stage Docker builds** for smaller images
2. **Implement proper health checks** for all services
3. **Use resource limits** to prevent resource exhaustion
4. **Implement graceful shutdown** handling
5. **Use secrets management** for sensitive data
6. **Enable monitoring and logging** from day one
7. **Implement proper backup strategies** for data
8. **Use blue-green or rolling deployments** for zero downtime

### Security Best Practices

1. **Use least privilege principle** for service accounts
2. **Implement network policies** to restrict traffic
3. **Use TLS everywhere** for data in transit
4. **Encrypt data at rest** using cloud provider encryption
5. **Regularly update dependencies** and base images
6. **Implement proper authentication and authorization**
7. **Use secret scanning** in CI/CD pipelines
8. **Monitor for security vulnerabilities**

## Related Documentation

- [Monitoring & Logging](./monitoring.md) - Detailed monitoring setup
- [Configuration Guide](../configuration.md) - Configuration reference
- [Development Guide](../development/README.md) - Development setup
- [Testing Guide](../development/testing.md) - Testing strategies

## Useful Commands

```bash
# Docker commands
docker-compose up -d                    # Start services
docker-compose logs -f <service>        # View logs
docker-compose scale <service>=3        # Scale service

# Kubernetes commands
kubectl apply -f k8s/                   # Deploy manifests
kubectl get pods -n rag-system          # Check pods
kubectl logs -f deployment/ingestion    # View logs
kubectl scale deployment ingestion --replicas=5  # Scale deployment

# Cloud commands
gcloud container clusters get-credentials rag-cluster  # GKE
eksctl create cluster --name rag-cluster              # EKS
az aks get-credentials --name rag-cluster              # AKS
```
