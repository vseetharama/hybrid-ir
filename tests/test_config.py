"""
Property-Based Tests for Configuration Management

This module tests the Config class with property-based testing using Hypothesis.
It validates that configuration loads correctly from environment variables and files,
and that environment-specific overrides work as expected.

**Validates: Requirements 11.1, 11.5**
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the Config class from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


# ============================================================================
# STRATEGY DEFINITIONS (Input Generators)
# ============================================================================

# Strategies for generating valid configuration values
valid_mongodb_uris = st.just("mongodb+srv://user:pass@cluster.mongodb.net/")
valid_database_names = st.text(
    alphabet=st.characters(blacklist_characters="\x00", codec='utf-8'),
    min_size=1,
    max_size=64
).filter(lambda x: x.strip())
valid_collection_names = st.text(
    alphabet=st.characters(blacklist_characters="\x00", codec='utf-8'),
    min_size=1,
    max_size=64
).filter(lambda x: x.strip())
valid_model_names = st.just("sentence-transformers/all-MiniLM-L6-v2")
valid_embedding_dimensions = st.integers(min_value=100, max_value=1000)
valid_rrf_k_constants = st.integers(min_value=1, max_value=1000)
valid_max_features_tfidf = st.integers(min_value=100, max_value=50000)
valid_min_df = st.floats(min_value=0.0, max_value=1.0)
valid_max_df = st.floats(min_value=0.0, max_value=1.0)
valid_default_top_k = st.integers(min_value=1, max_value=1000)
valid_paths = st.just("./test_data")
valid_environments = st.sampled_from(["dev", "staging", "prod"])
valid_debug_flags = st.booleans()


# ============================================================================
# PROPERTY 42: CONFIGURATION LOADS REQUIRED FIELDS
# ============================================================================

class TestProperty42ConfigurationLoadsRequiredFields:
    """
    Property 42: Configuration loads required fields
    
    Validates that all required fields in the Config class are properly loaded
    from both from_env() and from_file() methods.
    """
    
    @given(
        mongodb_uri=valid_mongodb_uris,
        database_name=valid_database_names,
        collection_name=valid_collection_names,
        embedding_model_name=valid_model_names,
        embedding_dimension=valid_embedding_dimensions,
        rrf_k_constant=valid_rrf_k_constants,
        max_features_tfidf=valid_max_features_tfidf,
        min_df=valid_min_df,
        max_df=valid_max_df,
        default_top_k=valid_default_top_k,
        data_dir=valid_paths,
        models_dir=valid_paths,
        indices_dir=valid_paths,
        logs_dir=valid_paths,
        environment=valid_environments,
        debug=valid_debug_flags,
    )
    @settings(max_examples=20, deadline=30000)
    def test_config_loads_from_env_all_required_fields_present(
        self,
        mongodb_uri,
        database_name,
        collection_name,
        embedding_model_name,
        embedding_dimension,
        rrf_k_constant,
        max_features_tfidf,
        min_df,
        max_df,
        default_top_k,
        data_dir,
        models_dir,
        indices_dir,
        logs_dir,
        environment,
        debug,
    ):
        """
        Test that Config.from_env() loads all required fields from environment variables.
        
        Given: Random but valid configuration values in environment variables
        When: Config.from_env() is called
        Then: All required fields are present in the returned Config instance
              with correct types and values
        
        **Validates: Requirements 11.1, 11.5**
        """
        # Set environment variables
        os.environ["MONGODB_URI"] = mongodb_uri
        os.environ["DATABASE_NAME"] = database_name
        os.environ["COLLECTION_NAME"] = collection_name
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model_name
        os.environ["EMBEDDING_DIMENSION"] = str(embedding_dimension)
        os.environ["RRF_K_CONSTANT"] = str(rrf_k_constant)
        os.environ["MAX_FEATURES_TFIDF"] = str(max_features_tfidf)
        os.environ["MIN_DF"] = str(min_df)
        os.environ["MAX_DF"] = str(max_df)
        os.environ["DEFAULT_TOP_K"] = str(default_top_k)
        os.environ["DATA_DIR"] = data_dir
        os.environ["MODELS_DIR"] = models_dir
        os.environ["INDICES_DIR"] = indices_dir
        os.environ["LOGS_DIR"] = logs_dir
        os.environ["ENVIRONMENT"] = environment
        os.environ["DEBUG"] = str(debug).lower()
        
        try:
            # Load configuration from environment
            config = Config.from_env()
            
            # Assert all required fields are present
            assert hasattr(config, 'mongodb_uri'), "Missing mongodb_uri field"
            assert hasattr(config, 'database_name'), "Missing database_name field"
            assert hasattr(config, 'collection_name'), "Missing collection_name field"
            assert hasattr(config, 'embedding_model_name'), "Missing embedding_model_name field"
            assert hasattr(config, 'embedding_dimension'), "Missing embedding_dimension field"
            assert hasattr(config, 'rrf_k_constant'), "Missing rrf_k_constant field"
            assert hasattr(config, 'max_features_tfidf'), "Missing max_features_tfidf field"
            assert hasattr(config, 'min_df'), "Missing min_df field"
            assert hasattr(config, 'max_df'), "Missing max_df field"
            assert hasattr(config, 'default_top_k'), "Missing default_top_k field"
            assert hasattr(config, 'data_dir'), "Missing data_dir field"
            assert hasattr(config, 'models_dir'), "Missing models_dir field"
            assert hasattr(config, 'indices_dir'), "Missing indices_dir field"
            assert hasattr(config, 'logs_dir'), "Missing logs_dir field"
            assert hasattr(config, 'environment'), "Missing environment field"
            assert hasattr(config, 'debug'), "Missing debug field"
            
            # Verify field types are correct
            assert isinstance(config.mongodb_uri, str), "mongodb_uri must be str"
            assert isinstance(config.database_name, str), "database_name must be str"
            assert isinstance(config.collection_name, str), "collection_name must be str"
            assert isinstance(config.embedding_model_name, str), "embedding_model_name must be str"
            assert isinstance(config.embedding_dimension, int), "embedding_dimension must be int"
            assert isinstance(config.rrf_k_constant, int), "rrf_k_constant must be int"
            assert isinstance(config.max_features_tfidf, int), "max_features_tfidf must be int"
            assert isinstance(config.min_df, float), "min_df must be float"
            assert isinstance(config.max_df, float), "max_df must be float"
            assert isinstance(config.default_top_k, int), "default_top_k must be int"
            assert isinstance(config.data_dir, str), "data_dir must be str"
            assert isinstance(config.models_dir, str), "models_dir must be str"
            assert isinstance(config.indices_dir, str), "indices_dir must be str"
            assert isinstance(config.logs_dir, str), "logs_dir must be str"
            assert isinstance(config.environment, str), "environment must be str"
            assert isinstance(config.debug, bool), "debug must be bool"
            
            # Verify field values match what was set
            assert config.mongodb_uri == mongodb_uri, f"mongodb_uri mismatch"
            assert config.database_name == database_name, f"database_name mismatch"
            assert config.collection_name == collection_name, f"collection_name mismatch"
            assert config.embedding_model_name == embedding_model_name, f"embedding_model_name mismatch"
            assert config.embedding_dimension == embedding_dimension, f"embedding_dimension mismatch"
            assert config.rrf_k_constant == rrf_k_constant, f"rrf_k_constant mismatch"
            assert config.max_features_tfidf == max_features_tfidf, f"max_features_tfidf mismatch"
            assert abs(config.min_df - min_df) < 1e-6, f"min_df mismatch"
            assert abs(config.max_df - max_df) < 1e-6, f"max_df mismatch"
            assert config.default_top_k == default_top_k, f"default_top_k mismatch"
            assert config.data_dir == data_dir, f"data_dir mismatch"
            assert config.models_dir == models_dir, f"models_dir mismatch"
            assert config.indices_dir == indices_dir, f"indices_dir mismatch"
            assert config.logs_dir == logs_dir, f"logs_dir mismatch"
            assert config.environment == environment, f"environment mismatch"
            assert config.debug == debug, f"debug mismatch"
            
        finally:
            # Clean up environment variables
            os.environ.pop("MONGODB_URI", None)
            os.environ.pop("DATABASE_NAME", None)
            os.environ.pop("COLLECTION_NAME", None)
            os.environ.pop("EMBEDDING_MODEL_NAME", None)
            os.environ.pop("EMBEDDING_DIMENSION", None)
            os.environ.pop("RRF_K_CONSTANT", None)
            os.environ.pop("MAX_FEATURES_TFIDF", None)
            os.environ.pop("MIN_DF", None)
            os.environ.pop("MAX_DF", None)
            os.environ.pop("DEFAULT_TOP_K", None)
            os.environ.pop("DATA_DIR", None)
            os.environ.pop("MODELS_DIR", None)
            os.environ.pop("INDICES_DIR", None)
            os.environ.pop("LOGS_DIR", None)
            os.environ.pop("ENVIRONMENT", None)
            os.environ.pop("DEBUG", None)
    
    @given(
        database_name=valid_database_names,
        collection_name=valid_collection_names,
        embedding_dimension=valid_embedding_dimensions,
        rrf_k_constant=valid_rrf_k_constants,
        max_features_tfidf=valid_max_features_tfidf,
        default_top_k=valid_default_top_k,
        environment=valid_environments,
    )
    @settings(max_examples=20, deadline=30000)
    def test_config_loads_from_json_file_all_required_fields_present(
        self,
        database_name,
        collection_name,
        embedding_dimension,
        rrf_k_constant,
        max_features_tfidf,
        default_top_k,
        environment,
    ):
        """
        Test that Config.from_file() loads all required fields from JSON configuration file.
        
        Given: A JSON config file with valid configuration values
        When: Config.from_file(json_path) is called
        Then: All required fields are present with correct types
        
        **Validates: Requirements 11.1, 11.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON config file with test values
            config_data = {
                "mongodb_uri": "mongodb+srv://test:pass@cluster.mongodb.net/",
                "database_name": database_name,
                "collection_name": collection_name,
                "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": embedding_dimension,
                "rrf_k_constant": rrf_k_constant,
                "max_features_tfidf": max_features_tfidf,
                "min_df": 1,
                "max_df": 0.95,
                "default_top_k": default_top_k,
                "data_dir": "./data",
                "models_dir": "./models",
                "indices_dir": "./indices",
                "logs_dir": "./logs",
                "environment": environment,
                "debug": False,
            }
            
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            # Load configuration from JSON file
            config = Config.from_file(str(config_path))
            
            # Assert all required fields are present and have correct types
            assert hasattr(config, 'mongodb_uri'), "Missing mongodb_uri"
            assert hasattr(config, 'database_name'), "Missing database_name"
            assert hasattr(config, 'collection_name'), "Missing collection_name"
            assert hasattr(config, 'embedding_dimension'), "Missing embedding_dimension"
            assert hasattr(config, 'rrf_k_constant'), "Missing rrf_k_constant"
            assert hasattr(config, 'max_features_tfidf'), "Missing max_features_tfidf"
            assert hasattr(config, 'environment'), "Missing environment"
            
            # Verify types
            assert isinstance(config.embedding_dimension, int)
            assert isinstance(config.rrf_k_constant, int)
            assert isinstance(config.max_features_tfidf, int)
            assert isinstance(config.database_name, str)
            assert isinstance(config.collection_name, str)
            assert isinstance(config.environment, str)
            
            # Verify values match
            assert config.database_name == database_name
            assert config.collection_name == collection_name
            assert config.embedding_dimension == embedding_dimension
            assert config.rrf_k_constant == rrf_k_constant
            assert config.max_features_tfidf == max_features_tfidf
            assert config.default_top_k == default_top_k
            assert config.environment == environment


# ============================================================================
# PROPERTY 43: CONFIGURATION SUPPORTS ENVIRONMENT OVERRIDES
# ============================================================================

class TestProperty43ConfigurationSupportsEnvironmentOverrides:
    """
    Property 43: Configuration supports environment overrides
    
    Validates that environment-specific overrides work correctly,
    with environment variables and environment-specific files taking precedence.
    """
    
    @given(
        dev_rrf_k=st.integers(min_value=1, max_value=100),
        staging_rrf_k=st.integers(min_value=101, max_value=200),
        prod_rrf_k=st.integers(min_value=201, max_value=300),
        dev_top_k=st.integers(min_value=1, max_value=50),
        staging_top_k=st.integers(min_value=51, max_value=100),
        prod_top_k=st.integers(min_value=101, max_value=200),
    )
    @settings(max_examples=20, deadline=30000)
    def test_environment_specific_values_override_defaults_via_env_vars(
        self,
        dev_rrf_k,
        staging_rrf_k,
        prod_rrf_k,
        dev_top_k,
        staging_top_k,
        prod_top_k,
    ):
        """
        Test that environment-specific values override defaults from environment variables.
        
        Given: Environment variables set for dev, staging, and prod
        When: Config.from_env() is called for each environment
        Then: Environment-specific values override defaults
              Non-overridden values keep their defaults
        
        **Validates: Requirements 11.1, 11.5**
        """
        base_env = {
            "MONGODB_URI": "mongodb+srv://base:pass@cluster.mongodb.net/",
            "DATABASE_NAME": "hybrid_ir",
            "COLLECTION_NAME": "documents",
        }
        
        try:
            # Test DEV environment
            os.environ.update(base_env)
            os.environ["ENVIRONMENT"] = "dev"
            os.environ["RRF_K_CONSTANT"] = str(dev_rrf_k)
            os.environ["DEFAULT_TOP_K"] = str(dev_top_k)
            
            config_dev = Config.from_env()
            assert config_dev.environment == "dev"
            assert config_dev.rrf_k_constant == dev_rrf_k
            assert config_dev.default_top_k == dev_top_k
            assert config_dev.database_name == "hybrid_ir"  # Not overridden
            
            # Test STAGING environment
            os.environ["ENVIRONMENT"] = "staging"
            os.environ["RRF_K_CONSTANT"] = str(staging_rrf_k)
            os.environ["DEFAULT_TOP_K"] = str(staging_top_k)
            
            config_staging = Config.from_env()
            assert config_staging.environment == "staging"
            assert config_staging.rrf_k_constant == staging_rrf_k
            assert config_staging.default_top_k == staging_top_k
            assert config_staging.database_name == "hybrid_ir"  # Not overridden
            
            # Test PROD environment
            os.environ["ENVIRONMENT"] = "prod"
            os.environ["RRF_K_CONSTANT"] = str(prod_rrf_k)
            os.environ["DEFAULT_TOP_K"] = str(prod_top_k)
            
            config_prod = Config.from_env()
            assert config_prod.environment == "prod"
            assert config_prod.rrf_k_constant == prod_rrf_k
            assert config_prod.default_top_k == prod_top_k
            assert config_prod.database_name == "hybrid_ir"  # Not overridden
            
            # Verify each environment has different values
            assert config_dev.rrf_k_constant != config_staging.rrf_k_constant
            assert config_staging.rrf_k_constant != config_prod.rrf_k_constant
            assert config_dev.default_top_k != config_staging.default_top_k
            assert config_staging.default_top_k != config_prod.default_top_k
            
        finally:
            # Clean up
            for key in list(os.environ.keys()):
                if key in base_env or key in [
                    "ENVIRONMENT", "RRF_K_CONSTANT", "DEFAULT_TOP_K",
                    "EMBEDDING_DIMENSION", "MAX_FEATURES_TFIDF"
                ]:
                    os.environ.pop(key, None)
    
    @given(
        base_rrf_k=st.integers(min_value=1, max_value=100),
        override_rrf_k=st.integers(min_value=101, max_value=200),
        base_top_k=st.integers(min_value=1, max_value=50),
        override_top_k=st.integers(min_value=51, max_value=100),
    )
    @settings(max_examples=20, deadline=30000)
    def test_environment_specific_json_file_overrides_base_config(
        self,
        base_rrf_k,
        override_rrf_k,
        base_top_k,
        override_top_k,
    ):
        """
        Test that environment-specific JSON files override base configuration.
        
        Given: Base config.json and environment-specific config.dev.json files
        When: Config.from_file(config.json) is called
        Then: Environment-specific values override base configuration
              Environment variable takes precedence over files
        
        **Validates: Requirements 11.1, 11.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create base config.json
            base_config = {
                "mongodb_uri": "mongodb+srv://base:pass@cluster.mongodb.net/",
                "database_name": "hybrid_ir_base",
                "collection_name": "documents",
                "embedding_dimension": 384,
                "rrf_k_constant": base_rrf_k,
                "default_top_k": base_top_k,
                "environment": "dev",
            }
            
            base_config_path = tmpdir_path / "config.json"
            with open(base_config_path, "w") as f:
                json.dump(base_config, f)
            
            # Create environment-specific config.dev.json
            dev_config = {
                "rrf_k_constant": override_rrf_k,
                "default_top_k": override_top_k,
            }
            
            dev_config_path = tmpdir_path / "config.dev.json"
            with open(dev_config_path, "w") as f:
                json.dump(dev_config, f)
            
            # Load config - should merge base + environment-specific
            config = Config.from_file(str(base_config_path))
            
            # Environment-specific values should override base
            assert config.rrf_k_constant == override_rrf_k
            assert config.default_top_k == override_top_k
            
            # Non-overridden values should remain from base
            assert config.database_name == "hybrid_ir_base"
            assert config.collection_name == "documents"
    
    @given(
        env_var_rrf_k=st.integers(min_value=1, max_value=100),
        file_rrf_k=st.integers(min_value=101, max_value=200),
    )
    @settings(max_examples=20, deadline=30000)
    def test_environment_variable_precedence_over_file_values(
        self,
        env_var_rrf_k,
        file_rrf_k,
    ):
        """
        Test that environment variables take precedence over file-based configuration.
        
        Given: A config file with rrf_k_constant value AND environment variable set
        When: Config is loaded
        Then: Environment variable value is used (takes precedence)
        
        **Validates: Requirements 11.1, 11.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_data = {
                "mongodb_uri": "mongodb+srv://file:pass@cluster.mongodb.net/",
                "database_name": "hybrid_ir",
                "collection_name": "documents",
                "rrf_k_constant": file_rrf_k,
                "embedding_dimension": 384,
                "default_top_k": 10,
                "environment": "dev",
            }
            
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            try:
                # Set environment variable to different value than file
                os.environ["RRF_K_CONSTANT"] = str(env_var_rrf_k)
                
                # Load from environment (should use env var)
                config = Config.from_env()
                
                # Environment variable should take precedence
                assert config.rrf_k_constant == env_var_rrf_k
                assert config.rrf_k_constant != file_rrf_k
                
            finally:
                os.environ.pop("RRF_K_CONSTANT", None)
    
    @given(
        dev_debug=st.booleans(),
        staging_debug=st.booleans(),
        prod_debug=st.booleans(),
    )
    @settings(max_examples=20, deadline=30000)
    def test_non_overridden_values_keep_defaults(
        self,
        dev_debug,
        staging_debug,
        prod_debug,
    ):
        """
        Test that non-overridden configuration values keep their defaults.
        
        Given: Only certain fields are overridden in environment
        When: Config.from_env() is called
        Then: Non-overridden fields retain their default values
              Overridden fields have the provided values
        
        **Validates: Requirements 11.1, 11.5**
        """
        try:
            # Clear environment
            for key in list(os.environ.keys()):
                if key.startswith(("DATABASE_", "COLLECTION_", "EMBEDDING_",
                                   "RRF_", "MAX_", "MIN_", "DEFAULT_", 
                                   "DATA_", "MODELS_", "INDICES_", "LOGS_",
                                   "ENVIRONMENT", "DEBUG", "MONGODB_")):
                    os.environ.pop(key, None)
            
            # Only set a few environment variables
            os.environ["DATABASE_NAME"] = "test_db"
            os.environ["ENVIRONMENT"] = "staging"
            os.environ["DEBUG"] = "true"
            
            config = Config.from_env()
            
            # Overridden values should match what we set
            assert config.database_name == "test_db"
            assert config.environment == "staging"
            assert config.debug == True
            
            # Non-overridden values should be defaults
            assert config.collection_name == "documents"  # Default
            assert config.embedding_dimension == 384  # Default
            assert config.rrf_k_constant == 60  # Default
            assert config.max_features_tfidf == 5000  # Default
            assert config.default_top_k == 10  # Default
            
        finally:
            os.environ.pop("DATABASE_NAME", None)
            os.environ.pop("ENVIRONMENT", None)
            os.environ.pop("DEBUG", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
