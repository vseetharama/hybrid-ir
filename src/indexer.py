"""
BM25 and TF-IDF Indexing Module for Hybrid Information Retrieval System

This module provides an IndexingEngine class that manages the building,
persistence, and retrieval of BM25 and TF-IDF indices.

Requirements: 2.1, 2.3, 2.5, 2.7
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    import pickle

try:
    from .config import Config
    from .preprocess import TextPreprocessor
except ImportError:  # Support direct imports when src is on sys.path.
    from config import Config
    from preprocess import TextPreprocessor


logger = logging.getLogger(__name__)


class IndexingEngine:
    """
    Build and manage BM25 and TF-IDF indices.
    
    This class handles:
    - Building BM25Okapi indices from preprocessed documents
    - Building TF-IDF indices using scikit-learn
    - Storing document ID mappings
    - Persisting indices to disk
    - Loading indices from disk
    
    Attributes:
        config (Config): Configuration object
        preprocessor (TextPreprocessor): Text preprocessing pipeline
        bm25_index (BM25Okapi): BM25 index object
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer instance
        tfidf_matrix: Sparse TF-IDF matrix (scipy.sparse matrix)
        document_ids (List[str]): List of document IDs ordered by index position
        tfidf_vocabulary (Dict[str, int]): Vocabulary mapping feature names to indices
    """
    
    def __init__(self, config: Config, preprocessor: TextPreprocessor):
        """
        Initialize IndexingEngine.
        
        Args:
            config (Config): Configuration object containing model and path settings
            preprocessor (TextPreprocessor): TextPreprocessor instance for document processing
        """
        self.config = config
        self.preprocessor = preprocessor
        
        # BM25 indices
        self.bm25_index: Optional[BM25Okapi] = None
        self.document_ids: List[str] = []
        
        # TF-IDF indices
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.tfidf_vocabulary: Optional[List[str]] = None
        
        logger.info("IndexingEngine initialized")
    
    def build_bm25_index(self, documents: List[Dict]) -> bool:
        """
        Build BM25Okapi index from documents.
        
        Preprocesses each document's content to create a token list,
        then builds a BM25Okapi index with default parameters (k1=1.5, b=0.75).
        Stores document ID mapping for result retrieval.
        
        Args:
            documents (List[Dict]): List of document dicts with 'doc_id' and 'content' keys
                                   Example: [{'doc_id': 'doc1', 'content': 'text...', 'metadata': {...}}, ...]
        
        Returns:
            bool: True if index built successfully, False otherwise
            
        Requirements: 2.1, 2.3, 2.5
        
        Raises:
            ValueError: If documents list is empty or documents lack required fields
        """
        try:
            if not documents:
                logger.warning("Empty documents list provided to build_bm25_index")
                return False
            
            # Verify all documents have required fields
            for doc in documents:
                if 'doc_id' not in doc or 'content' not in doc:
                    logger.error(f"Document missing 'doc_id' or 'content': {doc}")
                    return False
            
            # Preprocess documents to token lists
            tokenized_documents = []
            self.document_ids = []
            
            for doc in documents:
                doc_id = doc['doc_id']
                content = doc['content']
                
                # Preprocess content to tokens
                tokens = self.preprocessor.preprocess(content)
                
                tokenized_documents.append(tokens)
                self.document_ids.append(doc_id)
            
            # Create BM25Okapi index with default parameters (k1=1.5, b=0.75)
            self.bm25_index = BM25Okapi(tokenized_documents, k1=1.5, b=0.75)
            
            logger.info(f"BM25 index built successfully for {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {str(e)}")
            return False
    
    def build_tfidf_index(self, documents: List[Dict]) -> bool:
        """
        Build TF-IDF index from documents.
        
        Creates a TfidfVectorizer with specified parameters (max_features=5000,
        min_df=1, max_df=0.95) and fits it on preprocessed document tokens.
        
        Args:
            documents (List[Dict]): List of document dicts with 'doc_id' and 'content' keys
        
        Returns:
            bool: True if index built successfully, False otherwise
            
        Requirements: 2.2, 2.4, 2.6, 2.7
        """
        try:
            if not documents:
                logger.warning("Empty documents list provided to build_tfidf_index")
                return False
            
            # Verify all documents have required fields
            for doc in documents:
                if 'doc_id' not in doc or 'content' not in doc:
                    logger.error(f"Document missing 'doc_id' or 'content': {doc}")
                    return False
            
            # Preprocess documents to token strings for TF-IDF
            # TF-IDF typically works with raw text or pre-tokenized space-separated strings
            preprocessed_texts = []
            document_ids = []
            
            for doc in documents:
                doc_id = doc['doc_id']
                content = doc['content']
                
                # Preprocess to tokens
                tokens = self.preprocessor.preprocess(content)
                
                # Join tokens back to string (space-separated)
                preprocessed_text = ' '.join(tokens)
                preprocessed_texts.append(preprocessed_text)
                document_ids.append(doc_id)
            
            # Create and fit TF-IDF vectorizer
            # Using parameters from config: max_features=5000, min_df=1, max_df=0.95
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_features_tfidf,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                lowercase=False,  # Already lowercased during preprocessing
                token_pattern=r"(?u)\b\w+\b"  # Match word tokens
            )
            
            # Fit vectorizer and transform documents
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_texts)
            
            # Store vocabulary (feature names in order)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            self.tfidf_vocabulary = list(feature_names)
            
            logger.info(
                f"TF-IDF index built successfully for {len(documents)} documents, "
                f"vocabulary size: {len(self.tfidf_vocabulary)}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index: {str(e)}")
            return False
    
    def get_bm25_index(self) -> Optional[BM25Okapi]:
        """
        Get the BM25 index object.
        
        Returns:
            BM25Okapi: The BM25 index object, or None if not built
            
        Requirement: 2.7
        """
        return self.bm25_index
    
    def get_tfidf_matrix(self) -> Optional[Any]:
        """
        Get the TF-IDF sparse matrix.
        
        Returns:
            scipy.sparse matrix: The TF-IDF matrix, or None if not built
            
        Requirement: 2.7
        """
        return self.tfidf_matrix
    
    def get_tfidf_vectorizer(self) -> Optional[TfidfVectorizer]:
        """
        Get the TF-IDF vectorizer object.
        
        Returns:
            TfidfVectorizer: The vectorizer instance, or None if not built
        """
        return self.tfidf_vectorizer
    
    def get_tfidf_vocabulary(self) -> Optional[List[str]]:
        """
        Get TF-IDF vocabulary (feature names).
        
        Returns:
            List[str]: List of vocabulary terms in order, or None if not built
            
        Requirement: 2.7
        """
        return self.tfidf_vocabulary
    
    def get_document_ids(self) -> List[str]:
        """
        Get the document ID mapping.
        
        Returns:
            List[str]: List of document IDs in order of index position
        """
        return self.document_ids
    
    def save_indices(self, path: Optional[str] = None) -> bool:
        """
        Persist indices to disk.
        
        Saves BM25 index, TF-IDF vectorizer and matrix, and document ID mapping
        to the specified directory (or config.indices_dir if not specified).
        
        Uses joblib if available (better for sklearn objects), falls back to pickle.
        
        Args:
            path (Optional[str]): Directory path to save indices. If None, uses config.indices_dir
        
        Returns:
            bool: True if save successful, False otherwise
            
        Requirement: 2.7
        """
        try:
            save_dir = Path(path) if path else Path(self.config.indices_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save BM25 index
            bm25_path = save_dir / "bm25_index.pkl"
            if self.bm25_index:
                if HAS_JOBLIB:
                    joblib.dump(self.bm25_index, bm25_path)
                else:
                    import pickle
                    with open(bm25_path, 'wb') as f:
                        pickle.dump(self.bm25_index, f)
                logger.info(f"BM25 index saved to {bm25_path}")
            
            # Save TF-IDF vectorizer and matrix
            tfidf_vectorizer_path = save_dir / "tfidf_vectorizer.pkl"
            tfidf_matrix_path = save_dir / "tfidf_matrix.pkl"
            
            if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                if HAS_JOBLIB:
                    joblib.dump(self.tfidf_vectorizer, tfidf_vectorizer_path)
                    joblib.dump(self.tfidf_matrix, tfidf_matrix_path)
                else:
                    import pickle
                    with open(tfidf_vectorizer_path, 'wb') as f:
                        pickle.dump(self.tfidf_vectorizer, f)
                    with open(tfidf_matrix_path, 'wb') as f:
                        pickle.dump(self.tfidf_matrix, f)
                logger.info(f"TF-IDF vectorizer and matrix saved to {tfidf_vectorizer_path} and {tfidf_matrix_path}")
            
            # Save document ID mapping
            doc_ids_path = save_dir / "document_ids.json"
            with open(doc_ids_path, 'w') as f:
                json.dump(self.document_ids, f)
            logger.info(f"Document IDs saved to {doc_ids_path}")
            
            logger.info(f"All indices saved successfully to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save indices: {str(e)}")
            return False
    
    def load_indices(self, path: Optional[str] = None) -> bool:
        """
        Load indices from disk.
        
        Loads BM25 index, TF-IDF vectorizer and matrix, and document ID mapping
        from the specified directory (or config.indices_dir if not specified).
        
        Args:
            path (Optional[str]): Directory path to load indices from. If None, uses config.indices_dir
        
        Returns:
            bool: True if load successful, False otherwise
            
        Requirement: 2.7
        """
        try:
            load_dir = Path(path) if path else Path(self.config.indices_dir)
            
            if not load_dir.exists():
                logger.warning(f"Indices directory does not exist: {load_dir}")
                return False
            
            # Load BM25 index
            bm25_path = load_dir / "bm25_index.pkl"
            if bm25_path.exists():
                if HAS_JOBLIB:
                    self.bm25_index = joblib.load(bm25_path)
                else:
                    import pickle
                    with open(bm25_path, 'rb') as f:
                        self.bm25_index = pickle.load(f)
                logger.info(f"BM25 index loaded from {bm25_path}")
            
            # Load TF-IDF vectorizer and matrix
            tfidf_vectorizer_path = load_dir / "tfidf_vectorizer.pkl"
            tfidf_matrix_path = load_dir / "tfidf_matrix.pkl"
            
            if tfidf_vectorizer_path.exists() and tfidf_matrix_path.exists():
                if HAS_JOBLIB:
                    self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
                    self.tfidf_matrix = joblib.load(tfidf_matrix_path)
                else:
                    import pickle
                    with open(tfidf_vectorizer_path, 'rb') as f:
                        self.tfidf_vectorizer = pickle.load(f)
                    with open(tfidf_matrix_path, 'rb') as f:
                        self.tfidf_matrix = pickle.load(f)
                
                # Restore vocabulary from vectorizer
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                self.tfidf_vocabulary = list(feature_names)
                logger.info(f"TF-IDF vectorizer and matrix loaded from {tfidf_vectorizer_path}")
            
            # Load document ID mapping
            doc_ids_path = load_dir / "document_ids.json"
            if doc_ids_path.exists():
                with open(doc_ids_path, 'r') as f:
                    self.document_ids = json.load(f)
                logger.info(f"Document IDs loaded from {doc_ids_path}")
            
            logger.info(f"All indices loaded successfully from {load_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {str(e)}")
            return False


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test IndexingEngine
    print("=" * 60)
    print("Testing IndexingEngine")
    print("=" * 60)
    
    # Create config and preprocessor
    config = Config()
    preprocessor = TextPreprocessor()
    
    # Create indexing engine
    indexer = IndexingEngine(config, preprocessor)
    
    # Sample documents
    test_documents = [
        {
            'doc_id': 'doc1',
            'content': 'The quick brown fox jumps over the lazy dog',
            'metadata': {'title': 'Document 1'}
        },
        {
            'doc_id': 'doc2',
            'content': 'Machine learning and artificial intelligence are transforming technology',
            'metadata': {'title': 'Document 2'}
        },
        {
            'doc_id': 'doc3',
            'content': 'Natural language processing is a subfield of artificial intelligence',
            'metadata': {'title': 'Document 3'}
        },
    ]
    
    # Test BM25 indexing
    print("\n1. Building BM25 index...")
    success = indexer.build_bm25_index(test_documents)
    print(f"   BM25 index built: {success}")
    
    # Test TF-IDF indexing
    print("\n2. Building TF-IDF index...")
    success = indexer.build_tfidf_index(test_documents)
    print(f"   TF-IDF index built: {success}")
    
    # Test retrieval
    print("\n3. Retrieving indices...")
    bm25_index = indexer.get_bm25_index()
    tfidf_matrix = indexer.get_tfidf_matrix()
    doc_ids = indexer.get_document_ids()
    print(f"   BM25 index available: {bm25_index is not None}")
    print(f"   TF-IDF matrix available: {tfidf_matrix is not None}")
    print(f"   Document IDs: {doc_ids}")
    
    # Test persistence
    print("\n4. Saving indices...")
    success = indexer.save_indices()
    print(f"   Indices saved: {success}")
    
    print("\n5. Loading indices...")
    indexer2 = IndexingEngine(config, preprocessor)
    success = indexer2.load_indices()
    print(f"   Indices loaded: {success}")
    print(f"   Loaded document IDs: {indexer2.get_document_ids()}")
