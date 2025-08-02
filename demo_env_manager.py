#!/usr/bin/env python3
"""
Environment Manager Demonstration Script

This script demonstrates the advanced environment variable management
capabilities and shows how to refactor existing scripts to use the new
centralized environment manager.

Author: Expert Python Developer
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.env_manager import env, EnvironmentManager, EnvVarConfig, EnvVarType


def demonstrate_basic_usage():
    """Demonstrate basic environment manager usage."""
    print("🔧 Basic Environment Manager Usage")
    print("=" * 50)
    
    # Get common environment variables with defaults
    project_id = env.get_string("PROJECT_ID", "default-project")
    postgres_port = env.get_int("POSTGRES_PORT", 5432)
    debug_mode = env.get_bool("DEBUG_MODE", False)
    ssl_cert_path = env.get_path("SSL_CERT_FILE", "config/certs.pem")
    
    print(f"Project ID: {project_id}")
    print(f"PostgreSQL Port: {postgres_port} (type: {type(postgres_port)})")
    print(f"Debug Mode: {debug_mode} (type: {type(debug_mode)})")
    print(f"SSL Cert Path: {ssl_cert_path} (type: {type(ssl_cert_path)})")
    
    # Demonstrate JSON and list parsing
    json_config = env.get_json("JSON_CONFIG", {"default": "value"})
    api_endpoints = env.get_list("API_ENDPOINTS", ["localhost:8000", "localhost:8001"])
    
    print(f"JSON Config: {json_config}")
    print(f"API Endpoints: {api_endpoints}")
    print()


def demonstrate_custom_configuration():
    """Demonstrate custom environment variable configuration."""
    print("⚙️ Custom Environment Variable Configuration")
    print("=" * 50)
    
    # Create a custom environment manager
    custom_env = EnvironmentManager()
    
    # Register custom variables with validation
    custom_env.register_var(EnvVarConfig(
        name="MAX_RETRIES",
        var_type=EnvVarType.INTEGER,
        default=3,
        description="Maximum number of retry attempts",
        validator=lambda x: 1 <= x <= 10
    ))
    
    custom_env.register_var(EnvVarConfig(
        name="API_TIMEOUT",
        var_type=EnvVarType.FLOAT,
        default=30.0,
        description="API timeout in seconds",
        validator=lambda x: x > 0
    ))
    
    custom_env.register_var(EnvVarConfig(
        name="ENVIRONMENT",
        var_type=EnvVarType.STRING,
        default="development",
        description="Application environment",
        validator=lambda x: x in ["development", "staging", "production"]
    ))
    
    # Get values with validation
    max_retries = custom_env.get("MAX_RETRIES")
    api_timeout = custom_env.get("API_TIMEOUT")
    environment = custom_env.get("ENVIRONMENT")
    
    print(f"Max Retries: {max_retries}")
    print(f"API Timeout: {api_timeout}")
    print(f"Environment: {environment}")
    print()


def demonstrate_validation_and_errors():
    """Demonstrate validation and error handling."""
    print("🛡️ Validation and Error Handling")
    print("=" * 50)
    
    custom_env = EnvironmentManager()
    
    # Register a required variable
    custom_env.register_var(EnvVarConfig(
        name="REQUIRED_API_KEY",
        var_type=EnvVarType.STRING,
        required=True,
        description="Required API key for authentication"
    ))
    
    # Try to get a required variable that doesn't exist
    try:
        api_key = custom_env.get("REQUIRED_API_KEY")
        print(f"API Key: {api_key}")
    except Exception as e:
        print(f"Expected error for missing required variable: {e}")
    
    # Demonstrate validation errors
    custom_env.register_var(EnvVarConfig(
        name="INVALID_PORT",
        var_type=EnvVarType.INTEGER,
        default=8080,
        validator=lambda x: 1024 <= x <= 65535
    ))
    
    # Set an invalid value
    custom_env.set("INVALID_PORT", "80")  # Below valid range
    try:
        port = custom_env.get("INVALID_PORT")
        print(f"Port: {port}")
    except Exception as e:
        print(f"Expected validation error: {e}")
    
    print()


def demonstrate_summary_and_stats():
    """Demonstrate environment summary and statistics."""
    print("📊 Environment Summary and Statistics")
    print("=" * 50)
    
    # Get summary of all registered variables
    summary = env.get_summary()
    
    print("Registered Environment Variables:")
    for name, info in summary.items():
        status = "✅ SET" if info["is_set"] else "❌ NOT SET"
        required = "REQUIRED" if info["required"] else "OPTIONAL"
        print(f"  {name}: {status} ({required}) - {info['description']}")
    
    print(f"\nTotal registered variables: {len(summary)}")
    set_variables = sum(1 for info in summary.values() if info["is_set"])
    print(f"Variables with values: {set_variables}")
    print()


def demonstrate_legacy_refactoring():
    """Demonstrate how to refactor legacy code to use the environment manager."""
    print("🔄 Legacy Code Refactoring Example")
    print("=" * 50)
    
    print("BEFORE (Legacy approach):")
    print("""
import os

# Scattered environment variable access
project_id = os.environ.get("PROJECT_ID", "default")
port = int(os.environ.get("PORT", "8080"))
debug = os.environ.get("DEBUG", "false").lower() == "true"
ssl_cert = os.environ.get("SSL_CERT_FILE", "config/certs.pem")
""")
    
    print("AFTER (Using Environment Manager):")
    print("""
from src.utils.env_manager import env

# Centralized, type-safe environment variable access
project_id = env.get_string("PROJECT_ID", "default")
port = env.get_int("PORT", 8080)
debug = env.get_bool("DEBUG", False)
ssl_cert = env.get_path("SSL_CERT_FILE", "config/certs.pem")
""")
    
    # Show actual usage
    project_id = env.get_string("PROJECT_ID", "default")
    port = env.get_int("PORT", 8080)
    debug = env.get_bool("DEBUG", False)
    ssl_cert = env.get_path("SSL_CERT_FILE", "config/certs.pem")
    
    print(f"\nActual values:")
    print(f"Project ID: {project_id}")
    print(f"Port: {port} (type: {type(port)})")
    print(f"Debug: {debug} (type: {type(debug)})")
    print(f"SSL Cert: {ssl_cert} (type: {type(ssl_cert)})")
    print()


def demonstrate_database_configuration():
    """Demonstrate database configuration using environment manager."""
    print("🗄️ Database Configuration Example")
    print("=" * 50)
    
    # PostgreSQL configuration
    postgres_host = env.get_string("POSTGRES_HOST", "localhost")
    postgres_port = env.get_int("POSTGRES_PORT", 5432)
    postgres_db = env.get_string("POSTGRES_DB", "rag_db")
    postgres_user = env.get_string("POSTGRES_USER", "postgres")
    postgres_password = env.get_string("POSTGRES_PASSWORD")
    
    # Build connection string
    if postgres_password:
        connection_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    else:
        connection_string = f"postgresql://{postgres_user}@{postgres_host}:{postgres_port}/{postgres_db}"
    
    print(f"PostgreSQL Configuration:")
    print(f"  Host: {postgres_host}")
    print(f"  Port: {postgres_port}")
    print(f"  Database: {postgres_db}")
    print(f"  User: {postgres_user}")
    print(f"  Password: {'***' if postgres_password else 'Not set'}")
    print(f"  Connection String: {connection_string[:50]}...")
    print()


def main():
    """Main demonstration function."""
    print("🚀 Advanced Environment Manager Demonstration")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_custom_configuration()
    demonstrate_validation_and_errors()
    demonstrate_summary_and_stats()
    demonstrate_legacy_refactoring()
    demonstrate_database_configuration()
    
    print("✅ Environment Manager demonstration completed!")
    print()
    print("Key Benefits:")
    print("- Type-safe environment variable access")
    print("- Centralized configuration management")
    print("- Validation and error handling")
    print("- Default value support")
    print("- Comprehensive documentation and monitoring")
    print("- Easy migration from legacy os.environ usage")


if __name__ == "__main__":
    main()
