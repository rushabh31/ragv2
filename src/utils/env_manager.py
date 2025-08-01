"""
Advanced Environment Variable Manager

This module provides a centralized, type-safe, and efficient way to manage
environment variables across the entire application. It supports validation,
default values, type conversion, and caching for optimal performance.

Author: Expert Python Developer
"""

import os
import logging
from typing import Any, Dict, Optional, Union, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from functools import lru_cache

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnvVarType(Enum):
    """Supported environment variable types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"
    JSON = "json"
    LIST = "list"


@dataclass
class EnvVarConfig:
    """Configuration for an environment variable."""
    name: str
    var_type: EnvVarType
    default: Any = None
    required: bool = False
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    transformer: Optional[Callable[[str], Any]] = None


class EnvVarError(Exception):
    """Exception raised for environment variable related errors."""
    pass


class EnvironmentManager:
    """
    Advanced Environment Variable Manager
    
    Provides centralized management of environment variables with:
    - Type safety and validation
    - Caching for performance
    - Default value support
    - Custom transformers and validators
    - Comprehensive error handling
    """
    
    def __init__(self, auto_load_dotenv: bool = True):
        """
        Initialize the Environment Manager.
        
        Args:
            auto_load_dotenv: Whether to automatically load .env files
        """
        self._cache: Dict[str, Any] = {}
        self._configs: Dict[str, EnvVarConfig] = {}
        
        if auto_load_dotenv:
            self._load_dotenv()
        
        # Register common environment variables
        self._register_common_vars()
    
    def _load_dotenv(self) -> None:
        """Load environment variables from .env files."""
        try:
            from dotenv import load_dotenv
            
            # Look for .env files in common locations
            env_files = [
                Path.cwd() / ".env",
                Path.cwd() / "config" / ".env",
                Path(__file__).parent.parent.parent / ".env"
            ]
            
            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    logger.info(f"Loaded environment variables from {env_file}")
                    break
            else:
                logger.debug("No .env file found in common locations")
                
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file loading")
    
    def _register_common_vars(self) -> None:
        """Register commonly used environment variables."""
        
        # Authentication variables
        self.register_var(EnvVarConfig(
            name="COIN_CONSUMER_ENDPOINT_URL",
            var_type=EnvVarType.STRING,
            description="OAuth2 token endpoint URL for authentication"
        ))
        
        self.register_var(EnvVarConfig(
            name="COIN_CONSUMER_CLIENT_ID",
            var_type=EnvVarType.STRING,
            description="OAuth2 client ID for authentication"
        ))
        
        self.register_var(EnvVarConfig(
            name="COIN_CONSUMER_CLIENT_SECRET",
            var_type=EnvVarType.STRING,
            description="OAuth2 client secret for authentication"
        ))
        
        self.register_var(EnvVarConfig(
            name="COIN_CONSUMER_SCOPE",
            var_type=EnvVarType.STRING,
            default="https://www.googleapis.com/auth/cloud-platform",
            description="OAuth2 scope for authentication"
        ))
        
        # Google Cloud variables
        self.register_var(EnvVarConfig(
            name="PROJECT_ID",
            var_type=EnvVarType.STRING,
            description="Google Cloud Project ID"
        ))
        
        self.register_var(EnvVarConfig(
            name="VERTEXAI_API_ENDPOINT",
            var_type=EnvVarType.STRING,
            default="us-central1-aiplatform.googleapis.com",
            description="Vertex AI API endpoint"
        ))
        
        self.register_var(EnvVarConfig(
            name="VERTEXAI_API_TRANSPORT",
            var_type=EnvVarType.STRING,
            default="rest",
            description="Vertex AI API transport method"
        ))
        
        # API Keys
        self.register_var(EnvVarConfig(
            name="GROQ_API_KEY",
            var_type=EnvVarType.STRING,
            description="Groq API key for language models"
        ))
        
        self.register_var(EnvVarConfig(
            name="OPENAI_API_KEY",
            var_type=EnvVarType.STRING,
            description="OpenAI API key"
        ))
        
        self.register_var(EnvVarConfig(
            name="AZURE_OPENAI_API_KEY",
            var_type=EnvVarType.STRING,
            description="Azure OpenAI API key"
        ))
        
        # SSL Configuration
        self.register_var(EnvVarConfig(
            name="SSL_CERT_FILE",
            var_type=EnvVarType.PATH,
            default="config/certs.pem",
            description="Path to SSL certificate file"
        ))
        
        # Database Configuration
        self.register_var(EnvVarConfig(
            name="DATABASE_URL",
            var_type=EnvVarType.STRING,
            description="Database connection URL"
        ))
        
        self.register_var(EnvVarConfig(
            name="POSTGRES_HOST",
            var_type=EnvVarType.STRING,
            default="localhost",
            description="PostgreSQL host"
        ))
        
        self.register_var(EnvVarConfig(
            name="POSTGRES_PORT",
            var_type=EnvVarType.INTEGER,
            default=5432,
            description="PostgreSQL port"
        ))
        
        self.register_var(EnvVarConfig(
            name="POSTGRES_DB",
            var_type=EnvVarType.STRING,
            description="PostgreSQL database name"
        ))
        
        self.register_var(EnvVarConfig(
            name="POSTGRES_USER",
            var_type=EnvVarType.STRING,
            description="PostgreSQL username"
        ))
        
        self.register_var(EnvVarConfig(
            name="POSTGRES_PASSWORD",
            var_type=EnvVarType.STRING,
            description="PostgreSQL password"
        ))
        
        # Application Configuration
        self.register_var(EnvVarConfig(
            name="LOG_LEVEL",
            var_type=EnvVarType.STRING,
            default="INFO",
            description="Application log level",
            validator=lambda x: x.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ))
        
        self.register_var(EnvVarConfig(
            name="ENVIRONMENT",
            var_type=EnvVarType.STRING,
            default="development",
            description="Application environment",
            validator=lambda x: x.lower() in ["development", "staging", "production"]
        ))
        
        # Performance Configuration
        self.register_var(EnvVarConfig(
            name="MAX_WORKERS",
            var_type=EnvVarType.INTEGER,
            default=4,
            description="Maximum number of worker threads",
            validator=lambda x: 1 <= x <= 32
        ))
        
        self.register_var(EnvVarConfig(
            name="CACHE_TTL",
            var_type=EnvVarType.INTEGER,
            default=3600,
            description="Cache TTL in seconds",
            validator=lambda x: x > 0
        ))
    
    def register_var(self, config: EnvVarConfig) -> None:
        """
        Register an environment variable configuration.
        
        Args:
            config: Environment variable configuration
        """
        self._configs[config.name] = config
        logger.debug(f"Registered environment variable: {config.name}")
    
    def _convert_value(self, value: str, var_type: EnvVarType, transformer: Optional[Callable] = None) -> Any:
        """
        Convert string value to the specified type.
        
        Args:
            value: String value from environment
            var_type: Target type for conversion
            transformer: Optional custom transformer function
            
        Returns:
            Converted value
            
        Raises:
            EnvVarError: If conversion fails
        """
        if transformer:
            try:
                return transformer(value)
            except Exception as e:
                raise EnvVarError(f"Custom transformer failed: {e}")
        
        try:
            if var_type == EnvVarType.STRING:
                return value
            elif var_type == EnvVarType.INTEGER:
                return int(value)
            elif var_type == EnvVarType.FLOAT:
                return float(value)
            elif var_type == EnvVarType.BOOLEAN:
                return value.lower() in ("true", "1", "yes", "on")
            elif var_type == EnvVarType.PATH:
                return Path(value)
            elif var_type == EnvVarType.JSON:
                return json.loads(value)
            elif var_type == EnvVarType.LIST:
                return [item.strip() for item in value.split(",") if item.strip()]
            else:
                raise EnvVarError(f"Unsupported type: {var_type}")
                
        except (ValueError, json.JSONDecodeError) as e:
            raise EnvVarError(f"Failed to convert '{value}' to {var_type.value}: {e}")
    
    def get(self, name: str, default: Any = None, var_type: Optional[EnvVarType] = None) -> Any:
        """
        Get an environment variable value with type conversion and caching.
        
        Args:
            name: Environment variable name
            default: Default value if not found
            var_type: Expected type (if not registered)
            
        Returns:
            Environment variable value
            
        Raises:
            EnvVarError: If required variable is missing or validation fails
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        # Get configuration if registered
        config = self._configs.get(name)
        
        # Get raw value from environment
        raw_value = os.environ.get(name)
        
        if raw_value is None:
            if config and config.required:
                raise EnvVarError(f"Required environment variable '{name}' is not set")
            
            # Use default value
            final_default = config.default if config else default
            if final_default is not None:
                self._cache[name] = final_default
                return final_default
            
            return None
        
        # Determine type and convert
        target_type = config.var_type if config else (var_type or EnvVarType.STRING)
        transformer = config.transformer if config else None
        
        try:
            converted_value = self._convert_value(raw_value, target_type, transformer)
            
            # Validate if validator is provided
            if config and config.validator and not config.validator(converted_value):
                raise EnvVarError(f"Validation failed for '{name}': {converted_value}")
            
            # Cache and return
            self._cache[name] = converted_value
            return converted_value
            
        except EnvVarError:
            raise
        except Exception as e:
            raise EnvVarError(f"Error processing environment variable '{name}': {e}")
    
    def get_string(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get string environment variable."""
        return self.get(name, default, EnvVarType.STRING)
    
    def get_int(self, name: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer environment variable."""
        return self.get(name, default, EnvVarType.INTEGER)
    
    def get_float(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get float environment variable."""
        return self.get(name, default, EnvVarType.FLOAT)
    
    def get_bool(self, name: str, default: Optional[bool] = None) -> Optional[bool]:
        """Get boolean environment variable."""
        return self.get(name, default, EnvVarType.BOOLEAN)
    
    def get_path(self, name: str, default: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Get path environment variable."""
        result = self.get(name, default, EnvVarType.PATH)
        return Path(result) if result and not isinstance(result, Path) else result
    
    def get_json(self, name: str, default: Any = None) -> Any:
        """Get JSON environment variable."""
        return self.get(name, default, EnvVarType.JSON)
    
    def get_list(self, name: str, default: Optional[list] = None) -> Optional[list]:
        """Get list environment variable (comma-separated)."""
        return self.get(name, default, EnvVarType.LIST)
    
    def set(self, name: str, value: Any) -> None:
        """
        Set an environment variable and update cache.
        
        Args:
            name: Environment variable name
            value: Value to set
        """
        os.environ[name] = str(value)
        if name in self._cache:
            del self._cache[name]  # Clear cache to force reload
        logger.debug(f"Set environment variable: {name}")
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        Clear cached values.
        
        Args:
            name: Specific variable to clear, or None to clear all
        """
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()
        logger.debug(f"Cleared cache for: {name or 'all variables'}")
    
    def validate_all(self) -> Dict[str, str]:
        """
        Validate all registered environment variables.
        
        Returns:
            Dictionary of validation errors (empty if all valid)
        """
        errors = {}
        
        for name, config in self._configs.items():
            try:
                self.get(name)
            except EnvVarError as e:
                errors[name] = str(e)
        
        return errors
    
    def get_all_values(self) -> Dict[str, Any]:
        """
        Get all registered environment variable values.
        
        Returns:
            Dictionary of all environment variable values
        """
        values = {}
        
        for name in self._configs:
            try:
                values[name] = self.get(name)
            except EnvVarError:
                values[name] = None
        
        return values
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all registered environment variables.
        
        Returns:
            Dictionary with variable info and current values
        """
        summary = {}
        
        for name, config in self._configs.items():
            try:
                current_value = self.get(name)
                is_set = current_value is not None
            except EnvVarError:
                current_value = None
                is_set = False
            
            summary[name] = {
                "description": config.description,
                "type": config.var_type.value,
                "required": config.required,
                "default": config.default,
                "is_set": is_set,
                "current_value": current_value if not name.endswith(("_KEY", "_SECRET", "_PASSWORD")) else "***"
            }
        
        return summary


# Global instance for easy access
env = EnvironmentManager()


# Convenience functions for backward compatibility
def get_env(name: str, default: Any = None, var_type: Optional[EnvVarType] = None) -> Any:
    """Get environment variable using global manager."""
    return env.get(name, default, var_type)


def get_env_string(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get string environment variable using global manager."""
    return env.get_string(name, default)


def get_env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Get integer environment variable using global manager."""
    return env.get_int(name, default)


def get_env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    """Get boolean environment variable using global manager."""
    return env.get_bool(name, default)


def get_env_path(name: str, default: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Get path environment variable using global manager."""
    return env.get_path(name, default)
