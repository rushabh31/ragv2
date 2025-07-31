"""Centralized environment variable management for the RAG system."""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """Centralized manager for environment variables."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one environment manager instance exists."""
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._env_cache = {}
        return cls._instance
    
    def __init__(self):
        """Initialize the environment manager."""
        if not hasattr(self, '_env_cache'):
            self._env_cache = {}
    
    def get(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get environment variable with caching and validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            ValueError: If required variable is not found
        """
        if key in self._env_cache:
            return self._env_cache[key]
            
        value = os.environ.get(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
            
        self._env_cache[key] = value
        logger.debug(f"Retrieved environment variable: {key}")
        return value
    
    def get_int(self, key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
        """Get environment variable as integer."""
        value = self.get(key, str(default) if default is not None else None, required)
        return int(value) if value is not None else None
    
    def get_bool(self, key: str, default: Optional[bool] = None, required: bool = False) -> Optional[bool]:
        """Get environment variable as boolean."""
        value = self.get(key, str(default).lower() if default is not None else None, required)
        return value.lower() in ('true', '1', 'yes', 'on') if value is not None else None
    
    def get_float(self, key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
        """Get environment variable as float."""
        value = self.get(key, str(default) if default is not None else None, required)
        return float(value) if value is not None else None
    
    def set(self, key: str, value: str) -> None:
        """Set environment variable and update cache.
        
        Args:
            key: Environment variable name
            value: Environment variable value
        """
        os.environ[key] = value
        self._env_cache[key] = value
        logger.debug(f"Set environment variable: {key}")
    
    def clear_cache(self) -> None:
        """Clear the environment variable cache."""
        self._env_cache.clear()
        logger.debug("Cleared environment variable cache")
    
    def get_all_with_prefix(self, prefix: str) -> Dict[str, str]:
        """Get all environment variables with a specific prefix.
        
        Args:
            prefix: Prefix to filter environment variables
            
        Returns:
            Dictionary of environment variables with the prefix
        """
        result = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                result[key] = value
        return result
    
    def validate_required_vars(self, required_vars: list) -> None:
        """Validate that all required environment variables are set.
        
        Args:
            required_vars: List of required environment variable names
            
        Raises:
            ValueError: If any required variable is missing
        """
        missing_vars = []
        for var in required_vars:
            if not self.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

env_manager = EnvironmentManager()
