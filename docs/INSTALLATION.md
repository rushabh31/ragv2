# Detailed Installation Guide

This guide provides comprehensive instructions for installing, configuring, and deploying the RAG system in different environments.

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Provider-Specific Setup](#provider-specific-setup)

## Local Development Setup

### Prerequisites
- Python 3.9+ installed
- pip and virtualenv
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following variables:
   ```
   API_KEY=your_api_key_here
   ADMIN_API_KEY=your_admin_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-credentials.json
   ```

5. **Copy and customize configuration**
   ```bash
   cp config_sample.yaml config.yaml
   # Edit config.yaml according to your needs
   ```

6. **Run the services**
   ```bash
   # Start Ingestion API
   python -m controlsgenai.funcs.rag.src.ingestion.api.main
   
   # In another terminal
   python -m controlsgenai.funcs.rag.src.chatbot.api.main
   ```

## Production Deployment

### Prerequisites
- Server with Python 3.9+
- Nginx or similar web server (optional, for reverse proxy)
- Supervisor or systemd (for service management)

### Step-by-Step Deployment

1. **Clone the repository on your server**
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up environment and configuration**
   ```bash
   cp config_sample.yaml config.yaml
   # Edit config.yaml for production settings
   
   # Create .env file with production credentials
   ```

4. **Set up Supervisor for service management**

   Create `/etc/supervisor/conf.d/rag-ingestion.conf`:
   ```
   [program:rag-ingestion]
   command=/path/to/controlsgenai/venv/bin/python -m controlsgenai.funcs.rag.src.ingestion.api.main
   directory=/path/to/rag-system
   user=www-data
   autostart=true
   autorestart=true
   redirect_stderr=true
   stdout_logfile=/var/log/rag-ingestion.log
   environment=
       PYTHONPATH="/path/to/rag-system",
       API_KEY="your_api_key_here",
       ADMIN_API_KEY="your_admin_api_key_here",
       GROQ_API_KEY="your_groq_api_key_here"
   ```

   Create `/etc/supervisor/conf.d/rag-chatbot.conf`:
   ```
   [program:rag-chatbot]
   command=/path/to/controlsgenai/venv/bin/python -m controlsgenai.funcs.rag.src.chatbot.api.main
   directory=/path/to/rag-system
   user=www-data
   autostart=true
   autorestart=true
   redirect_stderr=true
   stdout_logfile=/var/log/rag-chatbot.log
   environment=
       PYTHONPATH="/path/to/rag-system",
       API_KEY="your_api_key_here",
       ADMIN_API_KEY="your_admin_api_key_here",
       GROQ_API_KEY="your_groq_api_key_here"
   ```

5. **Update Supervisor and start services**
   ```bash
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start rag-ingestion
   sudo supervisorctl start rag-chatbot
   ```

6. **Set up Nginx as reverse proxy (optional)**

   Create `/etc/nginx/sites-available/rag-system`:
   ```
   server {
       listen 80;
       server_name ingestion-api.yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   
   server {
       listen 80;
       server_name chatbot-api.yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

   Enable the site:
   ```bash
   sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed

### Step-by-Step Docker Setup

1. **Create a Dockerfile**
   Create a file named `Dockerfile` in the project root:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["python", "-m", "controlsgenai.funcs.rag.src.ingestion.api.main"]
   ```

2. **Create Docker Compose configuration**
   Create a file named `docker-compose.yml` in the project root:
   ```yaml
   version: '3'
   
   services:
     ingestion-api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - API_KEY=${API_KEY}
         - ADMIN_API_KEY=${ADMIN_API_KEY}
         - GROQ_API_KEY=${GROQ_API_KEY}
         - PYTHONPATH=/app
       volumes:
         - ./config.yaml:/app/config.yaml
         - ./data:/app/data
       command: python -m controlsgenai.funcs.rag.src.ingestion.api.main
   
     chatbot-api:
       build: .
       ports:
         - "8001:8001"
       environment:
         - API_KEY=${API_KEY}
         - ADMIN_API_KEY=${ADMIN_API_KEY}
         - GROQ_API_KEY=${GROQ_API_KEY}
         - PYTHONPATH=/app
       volumes:
         - ./config.yaml:/app/config.yaml
         - ./data:/app/data
       command: python -m controlsgenai.funcs.rag.src.chatbot.api.main
   ```

3. **Build and run with Docker Compose**
   ```bash
   # Create .env file with required API keys
   echo "API_KEY=your_api_key_here" > .env
   echo "ADMIN_API_KEY=your_admin_api_key_here" >> .env
   echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
   
   # Build and start services
   docker-compose up -d
   ```

## Environment Configuration

### Required Environment Variables

| Variable | Description | Required For |
|----------|-------------|-------------|
| `API_KEY` | General API key for authentication | All services |
| `ADMIN_API_KEY` | Admin API key with elevated privileges | Administrative endpoints |
| `GROQ_API_KEY` | Groq API key | Groq generation provider |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google service account JSON | Vertex AI providers |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Port for the API service | 8000/8001 |
| `REDIS_URL` | URL for Redis connection (if using Redis cache) | redis://localhost:6379/0 |

## Provider-Specific Setup

### Vertex AI Setup

1. **Create a Google Cloud project**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable the Vertex AI API**
   - Go to "APIs & Services" > "Library"
   - Search for "Vertex AI API" and enable it

3. **Create a service account**
   - Go to "IAM & Admin" > "Service Accounts"
   - Create a new service account
   - Assign the "Vertex AI User" role
   - Create and download a JSON key

4. **Set the credentials path**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
   ```

5. **Configure the system to use Vertex AI**
   ```yaml
   # In config.yaml
   chatbot:
     generation:
       provider: vertex
       model_name: gemini-1.0-pro
       
   ingestion:
     embedding:
       provider: vertex
       model_name: textembedding-gecko@001
   ```

### Groq Setup

1. **Sign up for Groq**
   - Create an account at [Groq](https://console.groq.com/)
   - Generate an API key

2. **Set the API key**
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Configure the system to use Groq**
   ```yaml
   # In config.yaml
   chatbot:
     generation:
       provider: groq
       model_name: llama3-8b-8192
   ```

### Sentence Transformers Setup

1. **Configure the system to use Sentence Transformers**
   ```yaml
   # In config.yaml
   ingestion:
     embedding:
       provider: sentence_transformer
       model_name: all-MiniLM-L6-v2
   ```

   Note: Sentence Transformers will automatically download the model when first used.

2. **Model selection considerations**
   - Smaller models (like `all-MiniLM-L6-v2`) are faster but may be less accurate
   - Larger models (like `all-mpnet-base-v2`) provide better quality but require more resources

## Additional Resources

- [RAG System GitHub Repository](https://github.com/yourusername/rag-system)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Groq Documentation](https://console.groq.com/docs)
- [Sentence Transformers Documentation](https://www.sbert.net/)
