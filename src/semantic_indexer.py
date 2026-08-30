"""
Semantic Search Indexing Module for Hybrid IR System.

This module provides the SemanticIndexer class for generating dense embeddings
using sentence-transformers and building FAISS indices for efficient semantic search.

The module supports:
- Loading pre-trained sentence-transformers models
- Generating normalized dense embeddings (384-dimensional)
- Building and managing FAISS IndexFlatIP indices
- Appending new embeddings to existing indices
- Persisting and loading indices to/from disk
- Validating index integrity and rebuilding on corruption

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 13.6
"""

import json
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from .config import Config
except ImportError:  # Support direct imports when src is on sys.path.
    from config import Config

logger = logging.getLogger(__name__)


class SemanticIndexer:
    """
    Build and manage FAISS index for semantic search.
    
    This class handles:
    - Loading sentence-transformers models
    - Generating dense embeddings for document texts
    - Building and managing FAISS indices
    - Appending new embeddings to existing indices
    - Persisting indices to disk
    - Validating and recovering from index corruption
    
    Attributes:
        config (Config): Configuration object with model and path settings
        model (SentenceTransformer): Loaded sentence-transformers model
        faiss_index (faiss.Index): FAISS IndexFlatIP for semantic search
        document_ids (List[str]): Mapping of index positions to document IDs
        embeddings (np.ndarray): Array of embeddings with shape (n_docs, 384)
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 13.6
    """
    
    def __init__(self, config: Config):
        """
        Initialize the SemanticIndexer.
        
        Args:
            config (Config): Configuration object with model name and paths
            
        Raises:
            ImportError: If faiss or sentence-transformers not installed
        """
        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.config = config
        self.model = None
        self.faiss_index = None
        self.document_ids = []
        self.embeddings = None
        logger.info("SemanticIndexer initialized with config")
    
    def load_model(self) -> bool:
        """
        Load sentence-transformers model.
        
        Loads the model specified in config (default: 'all-MiniLM-L6-v2')
        which produces 384-dimensional embeddings.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Requirement: 3.1
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> success = indexer.load_model()
            >>> assert success is True
            >>> assert indexer.model is not None
        """
        try:
            logger.info(f"Loading model: {self.config.embedding_model_name}")
            self.model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.config.embedding_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.config.embedding_model_name}: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts.
        
        Encodes texts using the loaded sentence-transformers model and normalizes
        embeddings to unit length (L2 norm = 1.0) for cosine similarity search.
        
        Args:
            texts (List[str]): List of document content strings to encode
            
        Returns:
            np.ndarray: Normalized embeddings with shape (len(texts), 384)
            
        Raises:
            RuntimeError: If model not loaded
            
        Requirement: 3.2, 3.3
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> texts = ["Hello world", "Machine learning"]
            >>> embeddings = indexer.generate_embeddings(texts)
            >>> assert embeddings.shape == (2, 384)
            >>> # Check normalization (unit length)
            >>> norms = np.linalg.norm(embeddings, axis=1)
            >>> np.allclose(norms, 1.0)
            True
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings to unit length (L2 norm = 1.0)
            # This enables cosine similarity via inner product in FAISS
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms
            
            logger.debug(f"Generated {len(texts)} embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def build_faiss_index(self, documents: List[Dict]) -> bool:
        """
        Build FAISS IndexFlatIP index from documents.
        
        Extracts content from documents, generates embeddings, normalizes them,
        builds a FAISS IndexFlatIP index (inner product = cosine for unit vectors),
        and stores document ID mapping.
        
        Args:
            documents (List[Dict]): List of document dicts with structure:
                {
                    'doc_id': str,
                    'content': str,
                    'metadata': dict (optional)
                }
                
        Returns:
            bool: True if index built successfully, False otherwise
            
        Requirement: 3.4, 3.5
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> docs = [
            ...     {'doc_id': 'doc1', 'content': 'Machine learning basics'},
            ...     {'doc_id': 'doc2', 'content': 'Deep learning guide'}
            ... ]
            >>> success = indexer.build_faiss_index(docs)
            >>> assert success is True
            >>> assert indexer.faiss_index.ntotal == 2
            >>> assert len(indexer.document_ids) == 2
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return False
            
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return False
            
            # Extract content from documents
            contents = [doc.get('content', '') for doc in documents]
            doc_ids = [doc.get('doc_id', '') for doc in documents]
            
            if not all(contents) or not all(doc_ids):
                logger.error("Documents must have 'content' and 'doc_id' fields")
                return False
            
            # Generate embeddings
            embeddings = self.generate_embeddings(contents)
            
            # Create FAISS index (IndexFlatIP for inner product = cosine similarity)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype(np.float32))
            
            # Store document IDs and embeddings
            self.document_ids = doc_ids
            self.embeddings = embeddings
            
            logger.info(f"FAISS index built successfully with {self.faiss_index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def add_embeddings(self, documents: List[Dict]) -> bool:
        """
        Add new embeddings to existing FAISS index (append operation).
        
        Generates embeddings for new documents and appends them to the existing
        FAISS index. Also appends document IDs and embeddings arrays.
        
        Args:
            documents (List[Dict]): List of new document dicts with 'doc_id' and 'content'
            
        Returns:
            bool: True if embeddings added successfully, False otherwise
            
        Requirement: 3.6
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> docs1 = [{'doc_id': 'doc1', 'content': 'First document'}]
            >>> indexer.build_faiss_index(docs1)
            >>> assert indexer.faiss_index.ntotal == 1
            >>> 
            >>> docs2 = [{'doc_id': 'doc2', 'content': 'Second document'}]
            >>> indexer.add_embeddings(docs2)
            >>> assert indexer.faiss_index.ntotal == 2
        """
        try:
            if not documents:
                logger.warning("No documents provided for appending")
                return False
            
            if self.faiss_index is None or self.embeddings is None:
                logger.error("No existing index. Use build_faiss_index() first.")
                return False
            
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return False
            
            # Extract content and IDs
            contents = [doc.get('content', '') for doc in documents]
            doc_ids = [doc.get('doc_id', '') for doc in documents]
            
            if not all(contents) or not all(doc_ids):
                logger.error("Documents must have 'content' and 'doc_id' fields")
                return False
            
            # Generate embeddings for new documents
            new_embeddings = self.generate_embeddings(contents)
            
            # Add to FAISS index
            self.faiss_index.add(new_embeddings.astype(np.float32))
            
            # Append to document IDs and embeddings array
            self.document_ids.extend(doc_ids)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            logger.info(f"Added {len(documents)} embeddings. Total: {self.faiss_index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        """
        Return embeddings array.
        
        Returns the full embeddings matrix with shape (n_docs, 384).
        
        Returns:
            np.ndarray: Embeddings array with shape (n_docs, 384), or None if not built
            
        Requirement: 3.7
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> docs = [{'doc_id': 'doc1', 'content': 'Sample text'}]
            >>> indexer.build_faiss_index(docs)
            >>> embeddings = indexer.get_embeddings()
            >>> assert embeddings.shape == (1, 384)
        """
        return self.embeddings
    
    def get_faiss_index(self):
        """
        Return FAISS index object.
        
        Returns:
            faiss.Index: FAISS index object, or None if not built
            
        Requirement: 3.7
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> indexer.build_faiss_index([{'doc_id': 'd1', 'content': 'text'}])
            >>> index = indexer.get_faiss_index()
            >>> assert index is not None
            >>> assert index.ntotal == 1
        """
        return self.faiss_index
    
    def save_index(self, path: Optional[str] = None) -> bool:
        """
        Persist FAISS index and embeddings to disk.
        
        Saves:
        - FAISS index to {path}/faiss.index using faiss.write_index()
        - Embeddings to {path}/embeddings.npy using numpy
        - Document ID mapping to {path}/doc_ids.json
        
        Args:
            path (Optional[str]): Directory path for saving. If None, uses config.indices_dir
            
        Returns:
            bool: True if saved successfully, False otherwise
            
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> indexer.build_faiss_index([{'doc_id': 'd1', 'content': 'text'}])
            >>> success = indexer.save_index('./indices')
            >>> assert success is True
        """
        try:
            if self.faiss_index is None or self.embeddings is None:
                logger.warning("No index to save")
                return False
            
            if path is None:
                path = self.config.indices_dir
            
            # Create directory if it doesn't exist
            index_dir = Path(path)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss_path = str(index_dir / "faiss.index")
            faiss.write_index(self.faiss_index, faiss_path)
            logger.info(f"FAISS index saved to {faiss_path}")
            
            # Save embeddings
            embeddings_path = str(index_dir / "embeddings.npy")
            np.save(embeddings_path, self.embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")
            
            # Save document ID mapping
            doc_ids_path = str(index_dir / "doc_ids.json")
            with open(doc_ids_path, 'w') as f:
                json.dump(self.document_ids, f)
            logger.info(f"Document IDs saved to {doc_ids_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """
        Load FAISS index and embeddings from disk.
        
        Loads:
        - FAISS index from {path}/faiss.index
        - Embeddings from {path}/embeddings.npy
        - Document ID mapping from {path}/doc_ids.json
        
        Args:
            path (Optional[str]): Directory path for loading. If None, uses config.indices_dir
            
        Returns:
            bool: True if loaded successfully, False otherwise
            
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> # After saving an index...
            >>> success = indexer.load_index('./indices')
            >>> assert success is True
        """
        try:
            if path is None:
                path = self.config.indices_dir
            
            index_dir = Path(path)
            
            # Load FAISS index
            faiss_path = str(index_dir / "faiss.index")
            if not Path(faiss_path).exists():
                logger.error(f"FAISS index not found at {faiss_path}")
                return False
            
            self.faiss_index = faiss.read_index(faiss_path)
            logger.info(f"FAISS index loaded from {faiss_path}")
            
            # Load embeddings
            embeddings_path = str(index_dir / "embeddings.npy")
            if not Path(embeddings_path).exists():
                logger.error(f"Embeddings not found at {embeddings_path}")
                return False
            
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Embeddings loaded from {embeddings_path}")
            
            # Load document ID mapping
            doc_ids_path = str(index_dir / "doc_ids.json")
            if not Path(doc_ids_path).exists():
                logger.error(f"Document IDs not found at {doc_ids_path}")
                return False
            
            with open(doc_ids_path, 'r') as f:
                self.document_ids = json.load(f)
            logger.info(f"Document IDs loaded from {doc_ids_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def validate_index(self) -> bool:
        """
        Check FAISS index integrity.
        
        Validates:
        - Index exists
        - Embeddings array exists and matches index size
        - Document IDs count matches index size
        - All embeddings are normalized (L2 norm ≈ 1.0)
        
        If corruption is detected, rebuilds index from embeddings.
        
        Returns:
            bool: True if index is valid or successfully rebuilt, False if rebuild failed
            
        Requirement: 13.6
        
        Example:
            >>> indexer = SemanticIndexer(config)
            >>> indexer.load_model()
            >>> indexer.build_faiss_index([{'doc_id': 'd1', 'content': 'text'}])
            >>> is_valid = indexer.validate_index()
            >>> assert is_valid is True
        """
        try:
            # Check if index exists
            if self.faiss_index is None or self.embeddings is None:
                logger.warning("Index not initialized")
                return False
            
            # Check consistency between index and embeddings
            if self.faiss_index.ntotal != len(self.embeddings):
                logger.error(
                    f"Index size mismatch: FAISS has {self.faiss_index.ntotal} "
                    f"but embeddings has {len(self.embeddings)}"
                )
                return self._rebuild_index()
            
            # Check document IDs count
            if len(self.document_ids) != self.faiss_index.ntotal:
                logger.error(
                    f"Document IDs mismatch: {len(self.document_ids)} IDs "
                    f"but index has {self.faiss_index.ntotal} documents"
                )
                return self._rebuild_index()
            
            # Check embedding normalization (L2 norm should be ~1.0)
            norms = np.linalg.norm(self.embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-6):
                logger.warning("Embeddings not properly normalized, normalizing...")
                norms = np.maximum(norms, 1e-12)
                self.embeddings = self.embeddings / norms[:, np.newaxis]
                return self._rebuild_index()
            
            logger.info("Index validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return self._rebuild_index()
    
    def _rebuild_index(self) -> bool:
        """
        Rebuild FAISS index from embeddings array.
        
        Used internally by validate_index() when corruption is detected.
        
        Returns:
            bool: True if rebuild successful, False otherwise
        """
        try:
            if self.embeddings is None:
                logger.error("Cannot rebuild: embeddings not available")
                return False
            
            logger.info("Rebuilding FAISS index from embeddings...")
            
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings.astype(np.float32))
            
            logger.info(f"Index rebuilt successfully with {self.faiss_index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False
