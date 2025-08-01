import os
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.rag.core.exceptions.exceptions import ConfigError

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manager for loading and accessing configuration settings."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one config manager instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigError: If configuration file cannot be loaded
        """
        if config_path is None:
            # Check for CONFIG_PATH environment variable first
            # Use environment manager for configuration
            from src.utils.env_manager import env
            env_config_path = env.get_string('CONFIG_PATH')
            if env_config_path:
                config_path = Path(env_config_path)
            else:
                # Default path is in the project's config directory
                # Traverse up to find the project root (where setup.py is located)
                current_dir = Path(__file__).resolve().parent
                project_root = None
                while current_dir != current_dir.parent:
                    if (current_dir / 'setup.py').exists():
                        project_root = current_dir
                        break
                    current_dir = current_dir.parent

                if project_root is None:
                    raise ConfigError("Could not find the project root directory. Make sure 'setup.py' is present.")

                config_path = project_root / 'config' / 'config.yaml'
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Process environment variables in the configuration
            self._process_env_vars(self._config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return self._config
        except Exception as e:
            error_msg = f"Failed to load configuration from {config_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e
    
    def get_config(self, reload: bool = False, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Get the loaded configuration.
        
        Args:
            reload: Whether to reload the configuration from disk
            config_path: Path to the configuration file. If None, uses default path.
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigError: If configuration has not been loaded yet
        """
        if self._config is None or reload:
            return self.load_config(config_path=config_path)
        return self._config
    
    def get_section(self, section_path: str, default: Any = None, config_path: Optional[str] = None) -> Any:
        """Get a specific section from the configuration using dot notation.
        
        Args:
            section_path: Path to the configuration section (e.g., 'ingestion.parsers')
            default: Default value to return if section is not found
            config_path: Path to the configuration file. If None, uses default path.
            
        Returns:
            Configuration section or default value if not found
        """
        if self._config is None:
            self.load_config(config_path=config_path)
        
        parts = section_path.split('.')
        current = self._config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            logger.warning(f"Configuration section '{section_path}' not found, using default")
            return default
    
    def update_section(self, section_path: str, value: Any, config_path: Optional[str] = None) -> None:
        """Update a specific section in the configuration using dot notation.
        
        Args:
            section_path: Path to the configuration section (e.g., 'ingestion.parsers.vision_parser.enabled')
            value: New value for the section
            config_path: Path to the configuration file. If None, uses default path.
            
        Raises:
            ConfigError: If section_path is invalid or configuration has not been loaded
        """
        if self._config is None:
            self.load_config(config_path=config_path)
        
        parts = section_path.split('.')
        current = self._config
        
        # Navigate to the parent of the target section
        for part in parts[:-1]:
            if part not in current:
                error_msg = f"Invalid section path: '{section_path}'"
                logger.error(error_msg)
                raise ConfigError(error_msg)
            current = current[part]
        
        # Update the target section
        current[parts[-1]] = value
    
    def _process_env_vars(self, config_section):
        """Recursively process environment variables in the configuration.
        
        Replaces strings like "${ENV_VAR_NAME}" with the value of the corresponding
        environment variable if it exists.
        
        Args:
            config_section: Configuration section to process
        """
        if isinstance(config_section, dict):
            for key, value in config_section.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str) and value.startswith("${"): 
                    env_var = value.strip("${}") 
                    env_value = env.get_string(env_var)
                    if env_value is not None:
                        logger.debug(f"Replacing config value with environment variable {env_var}")
                        config_section[key] = env_value
                    else:
                        logger.warning(f"Environment variable {env_var} not found, keeping placeholder")
        elif isinstance(config_section, list):
            for i, item in enumerate(config_section):
                if isinstance(item, (dict, list)):
                    self._process_env_vars(item)
                elif isinstance(item, str) and item.startswith("${"):
                    env_var = item.strip("${}")
                    env_value = env.get_string(env_var)
                    if env_value is not None:
                        logger.debug(f"Replacing config value with environment variable {env_var}")
                        config_section[i] = env_value
                    else:
                        logger.warning(f"Environment variable {env_var} not found, keeping placeholder")
