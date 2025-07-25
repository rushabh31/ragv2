#!/usr/bin/env python3

import os
import sys

# Add the project root to the Python path (go up 3 levels to reach controlsgenai root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Set the config file path to the local config
os.environ['CONFIG_PATH'] = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Import and run the application
from examples.rag.ingestion.api.main import app
import uvicorn

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
