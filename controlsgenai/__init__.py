"""
ControlsGenAI - Retrieval Augmented Generation System
"""

__version__ = "1.0.0"
__author__ = "ControlsGenAI Team"

# Import main modules from src to make them accessible
import sys
import os

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import rag
except ImportError:
    rag = None

__all__ = ['rag']