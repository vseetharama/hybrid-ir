"""
Configuration Management Module for Hybrid Information Retrieval System

This module provides centralized configuration management with support for:
- Default settings
- Environment variable overrides
- File-based configuration (YAML/JSON)
- Environment-specific configuration overrides
- Secure credential redaction in logs
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class Config:
    """
    Configuration class for the Hybrid IR System.
    
    Manages all system settings including database connection, model parameters,
    search configuration, paths, and environment-specific overrides.
    """
    
    # Database Configuration
    mongodb_uri: str
    database_name: str
    collection_name: str
    
    # Model Configuration
    embedding_model_name: str
    embedding_dimension: int
    
    # Search Configuration
    rrf_k_constant: int
    max_features_tfidf: int
    min_df: float
    max_df: float
    default_top_k: int
    
    # Path Configuration
    data_dir: str
    models_dir: str
    indices_dir: str
    logs_dir: str
    
    # Environment
    environment: str
    debug: bool
    
    def __init__(
        self,
        mongodb_uri: str = "mongodb+srv://",
        database_name: str = "hybrid_ir",
        collection_name: str = "documents",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        rrf_k_constant: int = 60,
        max_features_tfidf: int = 5000,
        min_df: float = 1,
        max_df: float = 0.95,
        default_top_k: int = 10,
        data_dir: str = "./data",
        models_dir: str = "./models",
        indices_dir: str = "./indices",
        logs_dir: str = "./logs",
        environment: str = "dev",
        debug: bool = False,
    ):
        """
        Initialize Config with provided values.
        
        Args:
            mongodb_uri: MongoDB connection string
            database_name: MongoDB database name (default: "hybrid_ir")
            collection_name: MongoDB collection name (default: "documents")
            embedding_model_name: Sentence-transformers model name
            embedding_dimension: Embedding vector dimension (default: 384)
            rrf_k_constant: RRF k parameter (default: 60)
            max_features_tfidf: Maximum TF-IDF features (default: 5000)
            min_df: Minimum document frequency for TF-IDF (default: 1)
            max_df: Maximum document frequency for TF-IDF (default: 0.95)
            default_top_k: Default number of results to return (default: 10)
            data_dir: Data storage directory (default: "./data")
            models_dir: Downloaded models directory (default: "./models")
            indices_dir: Persisted indices directory (default: "./indices")
            logs_dir: Log files directory (default: "./logs")
            environment: Environment type: "dev", "staging", or "prod" (default: "dev")
            debug: Enable debug mode (default: False)
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.rrf_k_constant = rrf_k_constant
        self.max_features_tfidf = max_features_tfidf
        self.min_df = min_df
        self.max_df = max_df
        self.default_top_k = default_top_k
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.indices_dir = indices_dir
        self.logs_dir = logs_dir
        self.environment = environment
        self.debug = debug
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.
        
        Reads environment variables and falls back to defaults if not set.
        Supported environment variables:
            - MONGODB_URI: MongoDB connection string
            - DATABASE_NAME: MongoDB database name
            - COLLECTION_NAME: MongoDB collection name
            - EMBEDDING_MODEL_NAME: Sentence-transformers model name
            - EMBEDDING_DIMENSION: Embedding vector dimension
            - RRF_K_CONSTANT: RRF k parameter
            - MAX_FEATURES_TFIDF: Maximum TF-IDF features
            - MIN_DF: Minimum document frequency
            - MAX_DF: Maximum document frequency
            - DEFAULT_TOP_K: Default results to return
            - DATA_DIR: Data storage directory
            - MODELS_DIR: Downloaded models directory
            - INDICES_DIR: Persisted indices directory
            - LOGS_DIR: Log files directory
            - ENVIRONMENT: Environment type (dev/staging/prod)
            - DEBUG: Enable debug mode (true/false)
        
        Returns:
            Config instance populated from environment variables and defaults
        """
        # Parse environment variables with type conversion
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://")
        database_name = os.getenv("DATABASE_NAME", "hybrid_ir")
        collection_name = os.getenv("COLLECTION_NAME", "documents")
        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        rrf_k_constant = int(os.getenv("RRF_K_CONSTANT", "60"))
        max_features_tfidf = int(os.getenv("MAX_FEATURES_TFIDF", "5000"))
        min_df = float(os.getenv("MIN_DF", "1"))
        max_df = float(os.getenv("MAX_DF", "0.95"))
        default_top_k = int(os.getenv("DEFAULT_TOP_K", "10"))
        data_dir = os.getenv("DATA_DIR", "./data")
        models_dir = os.getenv("MODELS_DIR", "./models")
        indices_dir = os.getenv("INDICES_DIR", "./indices")
        logs_dir = os.getenv("LOGS_DIR", "./logs")
        environment = os.getenv("ENVIRONMENT", "dev")
        debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        
        return cls(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_dimension=embedding_dimension,
            rrf_k_constant=rrf_k_constant,
            max_features_tfidf=max_features_tfidf,
            min_df=min_df,
            max_df=max_df,
            default_top_k=default_top_k,
            data_dir=data_dir,
            models_dir=models_dir,
            indices_dir=indices_dir,
            logs_dir=logs_dir,
            environment=environment,
            debug=debug,
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML or JSON file.
        
        Supports both YAML and JSON formats. Environment-specific config files
        can override base settings:
            - config.yaml or config.json (base configuration)
            - config.dev.yaml, config.staging.yaml, config.prod.yaml (environment-specific)
        
        Environment-specific overrides take precedence over base configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Config instance populated from file
            
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If file format is unsupported or invalid
            yaml.YAMLError: If YAML parsing fails (when pyyaml available)
            json.JSONDecodeError: If JSON parsing fails
        """
        path = Path(config_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = path.suffix.lower()
        
        # Load base configuration
        config_data = {}
        
        if suffix == ".yaml" or suffix == ".yml":
            if not HAS_YAML:
                raise ValueError(
                    "YAML support requires PyYAML. Install with: pip install pyyaml"
                )
            with open(path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        elif suffix == ".json":
            with open(path, "r") as f:
                config_data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                f"Supported formats: .yaml, .yml, .json"
            )
        
        # Try to load environment-specific overrides
        environment = config_data.get("environment", "dev")
        if environment in ("dev", "staging", "prod"):
            env_config_path = path.parent / f"{path.stem}.{environment}{path.suffix}"
            if env_config_path.exists():
                if suffix == ".yaml" or suffix == ".yml":
                    with open(env_config_path, "r") as f:
                        env_data = yaml.safe_load(f) or {}
                else:
                    with open(env_config_path, "r") as f:
                        env_data = json.load(f)
                # Merge environment-specific config (takes precedence)
                config_data.update(env_data)
        
        # Create config with file values, falling back to defaults
        return cls(
            mongodb_uri=config_data.get("mongodb_uri", "mongodb+srv://"),
            database_name=config_data.get("database_name", "hybrid_ir"),
            collection_name=config_data.get("collection_name", "documents"),
            embedding_model_name=config_data.get(
                "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dimension=config_data.get("embedding_dimension", 384),
            rrf_k_constant=config_data.get("rrf_k_constant", 60),
            max_features_tfidf=config_data.get("max_features_tfidf", 5000),
            min_df=config_data.get("min_df", 1),
            max_df=config_data.get("max_df", 0.95),
            default_top_k=config_data.get("default_top_k", 10),
            data_dir=config_data.get("data_dir", "./data"),
            models_dir=config_data.get("models_dir", "./models"),
            indices_dir=config_data.get("indices_dir", "./indices"),
            logs_dir=config_data.get("logs_dir", "./logs"),
            environment=config_data.get("environment", "dev"),
            debug=config_data.get("debug", False),
        )
    
    def log_settings(self) -> None:
        """
        Log all configuration values.
        
        Logs all configuration settings using Python's logging module.
        Sensitive credentials (mongodb_uri) are redacted to show "[REDACTED]".
        Format: "Setting: value"
        
        Returns:
            None
        """
        logger = logging.getLogger(__name__)
        
        # Create a copy of settings with redacted credentials
        settings = {
            "mongodb_uri": "[REDACTED]",  # Always redact sensitive credentials
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "rrf_k_constant": self.rrf_k_constant,
            "max_features_tfidf": self.max_features_tfidf,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "default_top_k": self.default_top_k,
            "data_dir": self.data_dir,
            "models_dir": self.models_dir,
            "indices_dir": self.indices_dir,
            "logs_dir": self.logs_dir,
            "environment": self.environment,
            "debug": self.debug,
        }
        
        # Log header
        logger.info("=" * 60)
        logger.info("Configuration Settings")
        logger.info("=" * 60)
        
        # Log each setting
        for key, value in settings.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration (credentials not redacted)
        """
        return {
            "mongodb_uri": self.mongodb_uri,
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "rrf_k_constant": self.rrf_k_constant,
            "max_features_tfidf": self.max_features_tfidf,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "default_top_k": self.default_top_k,
            "data_dir": self.data_dir,
            "models_dir": self.models_dir,
            "indices_dir": self.indices_dir,
            "logs_dir": self.logs_dir,
            "environment": self.environment,
            "debug": self.debug,
        }


if __name__ == "__main__":
    # Configure logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test 1: Create Config with defaults
    print("Test 1: Creating Config with defaults")
    config = Config()
    print(f"  Database: {config.database_name}")
    print(f"  Environment: {config.environment}")
    print(f"  Debug: {config.debug}")
    print()
    
    # Test 2: Load from environment variables
    print("Test 2: Loading from environment variables")
    os.environ["DATABASE_NAME"] = "test_db"
    os.environ["ENVIRONMENT"] = "prod"
    os.environ["DEBUG"] = "true"
    config_env = Config.from_env()
    print(f"  Database: {config_env.database_name}")
    print(f"  Environment: {config_env.environment}")
    print(f"  Debug: {config_env.debug}")
    print()
    
    # Test 3: Log settings with credential redaction
    print("Test 3: Logging settings (credentials redacted)")
    config.log_settings()
