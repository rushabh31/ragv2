# ControlGenAI Installation Guide

This guide explains how to install and use the restructured ControlGenAI library.

## Installation

### Option 1: Development Installation (Recommended)

1. **Navigate to the project directory:**
   ```bash
   cd /Users/rushabhsmacbook/Documents/controlsgenai
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Regular Installation

1. **Install from the directory:**
   ```bash
   pip install /Users/rushabhsmacbook/Documents/controlsgenai
   ```

## Verification

Run the test script to verify the installation:

```bash
python test_installation.py
```

## Usage

After installation, you can import and use the library with the new import pattern:

### Basic Imports

```python
# Import the main package
import controlsgenai

# Import specific modules
from controlsgenai.funcs.rag.src.shared.utils.config_manager import ConfigManager
from controlsgenai.funcs.rag.src.chatbot.generators.generator_factory import GeneratorFactory
from controlsgenai.funcs.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
```

### Example Usage

```python
# Configuration management
from controlsgenai.funcs.rag.src.shared.utils.config_manager import ConfigManager
config_manager = ConfigManager()

# Generator factory
from controlsgenai.funcs.rag.src.chatbot.generators.generator_factory import GeneratorFactory
generator = GeneratorFactory.create_generator("groq", {"api_key": "your-key"})

# Embedder factory
from controlsgenai.funcs.rag.src.ingestion.embedders.embedder_factory import EmbedderFactory
embedder = EmbedderFactory.create_embedder("sentence_transformer")
```

## Running the Services

### Ingestion API
```bash
python -m controlsgenai.funcs.rag.src.ingestion.api.main
```

### Chatbot API
```bash
python -m controlsgenai.funcs.rag.src.chatbot.api.main
```

### Using the Run Scripts
```bash
# From the main directory
python run_ingestion.py
python run_chatbot.py
```

### Using Console Scripts (after installation)
```bash
# These commands are available after installing the package
controlsgenai-ingestion
controlsgenai-chatbot
```

## Package Structure

The new package structure follows this pattern:

```
controlsgenai/
├── __init__.py
├── funcs/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       └── src/
│           ├── __init__.py
│           ├── chatbot/
│           ├── core/
│           ├── ingestion/
│           └── shared/
```

All imports now use the pattern: `controlsgenai.funcs.rag.src.*`

## Migration from Old Structure

If you have existing code using the old `rag_system.src.*` imports, simply replace them with `controlsgenai.funcs.rag.src.*`:

```python
# Old import
from rag_system.src.shared.utils.config_manager import ConfigManager

# New import
from controlsgenai.funcs.rag.src.shared.utils.config_manager import ConfigManager
```

## Running Tests

Run the test suite to verify everything is working:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=controlsgenai

# Run specific test file
python -m pytest tests/test_singleton_memory.py
```

## Troubleshooting

1. **Import errors**: Make sure you've installed the package correctly and all dependencies are installed.

2. **Module not found**: Verify that you're using the correct import path: `controlsgenai.funcs.rag.src.*`

3. **Configuration issues**: Check that your config files are in the correct location and properly formatted.

4. **Test failures**: Make sure you have the test dependencies installed: `pip install -e .[test]`

For more detailed information, see the documentation in the `docs/` directory.
