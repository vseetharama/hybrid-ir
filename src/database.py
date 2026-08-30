"""
Database manager module for MongoDB operations in the Hybrid IR System.

This module provides the DatabaseManager class for handling all database
operations including document storage, retrieval, and batch operations.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 13.1
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

try:
    from .config import Config
except ImportError:  # Support direct imports when src is on sys.path.
    from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    MongoDB connection and document management.
    
    Handles all database operations for storing and retrieving documents
    from MongoDB Atlas. Implements retry logic with exponential backoff
    for connection failures and supports batch operations.
    
    Attributes:
        config (Config): Configuration object
        client (Optional[MongoClient]): MongoDB client instance
        db: MongoDB database instance
        collection: MongoDB collection instance
    
    Requirements: 7.1-7.8, 13.1
    """
    
    def __init__(self, config: Config):
        """
        Initialize the DatabaseManager.
        
        Args:
            config (Config): Configuration object containing MongoDB connection string
        """
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        logger.debug(f"DatabaseManager initialized with config: {config.database_name}")
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Implements retry logic with exponential backoff:
        - Attempt 1: 1 second delay
        - Attempt 2: 2 second delay
        - Attempt 3: 4 second delay
        
        Returns:
            bool: True if successful, False otherwise
            
        Requirement: 7.1, 13.1
        """
        backoff_times = [1, 2, 4]  # Exponential backoff in seconds
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"MongoDB connection attempt {attempt + 1}/{max_retries}")
                
                # Create client with connection timeout
                self.client = MongoClient(
                    self.config.mongodb_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                    retryWrites=True
                )
                
                # Test the connection by accessing server info
                self.client.admin.command('ping')
                
                # Set database and collection
                self.db = self.client[self.config.database_name]
                self.collection = self.db[self.config.collection_name]
                
                logger.info(f"Successfully connected to MongoDB: {self.config.database_name}")
                return True
                
            except (ServerSelectionTimeoutError, ConnectionFailure) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to connect to MongoDB after {max_retries} attempts. "
                        f"URI: {self.config.mongodb_uri}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Unexpected error during connection: {str(e)}")
                return False
        
        return False
    
    def disconnect(self) -> None:
        """
        Close MongoDB connection.
        
        Requirement: 7.2
        """
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {str(e)}")
            finally:
                self.client = None
                self.db = None
                self.collection = None
    
    def insert_document(self, doc_id: str, content: str, metadata: Dict) -> bool:
        """
        Insert a single document into MongoDB.
        
        Stores document with the following fields:
        - _id: Unique document identifier (doc_id)
        - content: Document text content
        - metadata: Dictionary with arbitrary metadata
        - indexed_timestamp: ISO format timestamp when document was indexed
        
        Args:
            doc_id (str): Unique document identifier
            content (str): Document text content
            metadata (Dict): Dictionary with arbitrary metadata
            
        Returns:
            bool: True if successful, False otherwise
            
        Requirement: 7.3
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            document = {
                "_id": doc_id,
                "content": content,
                "metadata": metadata,
                "indexed_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Use replace_one with upsert to handle duplicate IDs
            result = self.collection.replace_one(
                {"_id": doc_id},
                document,
                upsert=True
            )
            
            logger.debug(f"Document {doc_id} inserted/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting document {doc_id}: {str(e)}")
            return False
    
    def insert_batch(self, documents: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple documents in batch.
        
        Processes a list of documents and attempts to insert each one.
        Tracks successful and failed insertions separately.
        
        Args:
            documents (List[Dict]): List of dicts with 'doc_id', 'content', 'metadata'
            
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
            
        Requirement: 7.4, 13.4
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return 0, len(documents)
        
        successful_count = 0
        failed_count = 0
        
        logger.info(f"Starting batch insert of {len(documents)} documents")
        
        for idx, doc in enumerate(documents):
            try:
                doc_id = doc.get("doc_id")
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if not doc_id:
                    logger.warning(f"Document at index {idx} missing doc_id, skipping")
                    failed_count += 1
                    continue
                
                if self.insert_document(doc_id, content, metadata):
                    successful_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing document at index {idx}: {str(e)}")
                failed_count += 1
        
        logger.info(
            f"Batch insert completed: {successful_count} successful, {failed_count} failed"
        )
        return successful_count, failed_count
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a single document by ID.
        
        Args:
            doc_id (str): Unique document identifier
            
        Returns:
            Optional[Dict]: Dict with 'content' and 'metadata' keys, or None if not found
            
        Requirement: 7.5
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return None
        
        try:
            document = self.collection.find_one({"_id": doc_id})
            
            if document:
                return {
                    "content": document.get("content", ""),
                    "metadata": document.get("metadata", {}),
                    "indexed_timestamp": document.get("indexed_timestamp")
                }
            else:
                logger.debug(f"Document {doc_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def get_documents_batch(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            doc_ids (List[str]): List of document identifiers
            
        Returns:
            Dict[str, Dict]: Dictionary mapping doc_id → {content, metadata}
                            Only includes documents that were found
            
        Requirement: 7.6
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return {}
        
        result = {}
        
        try:
            documents = self.collection.find({"_id": {"$in": doc_ids}})
            
            for document in documents:
                doc_id = document.get("_id")
                result[doc_id] = {
                    "content": document.get("content", ""),
                    "metadata": document.get("metadata", {}),
                    "indexed_timestamp": document.get("indexed_timestamp")
                }
            
            logger.debug(f"Retrieved {len(result)} documents from batch of {len(doc_ids)}")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving batch documents: {str(e)}")
            return {}
    
    def create_index(self) -> bool:
        """
        Create index on _id field for fast lookups.
        
        Returns:
            bool: True if successful or index already exists, False otherwise
            
        Requirement: 7.7
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            # MongoDB automatically creates an index on _id, but we can ensure it exists
            # by attempting to create it (create_index is idempotent)
            self.collection.create_index("_id", unique=True)
            logger.debug("Index on _id field created/verified")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id (str): Unique document identifier
            
        Returns:
            bool: True if document was deleted, False otherwise
            
        Requirement: 7.8
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            result = self.collection.delete_one({"_id": doc_id})
            
            if result.deleted_count > 0:
                logger.info(f"Document {doc_id} deleted successfully")
                return True
            else:
                logger.warning(f"Document {doc_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Remove all documents from the collection.
        
        Use with caution - this operation cannot be undone.
        
        Returns:
            bool: True if successful, False otherwise
            
        Requirement: 7.8
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            result = self.collection.delete_many({})
            logger.warning(f"Cleared {result.deleted_count} documents from collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dict: Dictionary containing collection statistics including:
                - document_count: Number of documents in collection
                - indexes: List of index names
                - avg_doc_size: Average document size in bytes (if available)
                
        Requirement: 7.8
        """
        if not self.collection:
            logger.error("Not connected to MongoDB")
            return {}
        
        try:
            stats = {
                "document_count": self.collection.count_documents({}),
                "indexes": list(self.collection.list_indexes()),
            }
            
            # Try to get collection statistics from database
            try:
                db_stats = self.db.command("collStats", self.config.collection_name)
                stats["avg_doc_size"] = db_stats.get("avgObjSize")
                stats["storage_size"] = db_stats.get("storageSize")
            except Exception:
                # Some deployments might not support collStats
                pass
            
            logger.debug(f"Collection stats: {stats['document_count']} documents")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def is_connected(self) -> bool:
        """
        Check if currently connected to MongoDB.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None and self.collection is not None
