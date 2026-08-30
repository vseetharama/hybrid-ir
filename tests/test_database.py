"""
Property-Based Tests for MongoDB Database Manager

This module tests the DatabaseManager class with property-based testing using Hypothesis.
It validates database operations including document insertion, retrieval, batch operations,
and error handling using mongomock for in-memory MongoDB testing.

**Validates: Requirements 7.3-7.8, 13.1, 13.4**
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import mongomock
from datetime import datetime, timezone
import json

# Import the DatabaseManager and Config classes from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import DatabaseManager
from config import Config


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = MagicMock(spec=Config)
    config.mongodb_uri = "mongodb://test"
    config.database_name = "test_db"
    config.collection_name = "test_collection"
    return config


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client using mongomock."""
    client = mongomock.MongoClient()
    return client


@pytest.fixture
def db_manager_with_mock(test_config, mock_mongo_client):
    """Create a DatabaseManager with mocked MongoDB."""
    manager = DatabaseManager(test_config)
    
    # Mock the MongoClient to return our mongomock client
    with patch('database.MongoClient', return_value=mock_mongo_client):
        manager.connect()
    
    return manager


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

# Strategy for document IDs
doc_ids = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=122),
    min_size=1,
    max_size=50
)

# Strategy for document content
doc_content = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=1000
)

# Strategy for metadata (dictionary) - use constrained values
metadata_values = st.one_of(
    st.just(None),
    st.booleans(),
    st.integers(min_value=-9223372036854775808, max_value=9223372036854775807),  # 64-bit range
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    st.text(max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
)

metadata_dict = st.dictionaries(
    keys=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
    values=metadata_values,
    min_size=0,
    max_size=5
)

# Strategy for document objects
documents = st.lists(
    st.fixed_dictionaries({
        "doc_id": doc_ids,
        "content": doc_content,
        "metadata": metadata_dict
    }),
    min_size=1,
    max_size=20,
    unique_by=lambda x: x["doc_id"]  # Ensure unique doc_ids
)


# ============================================================================
# PROPERTY 31: Document Storage Round-Trip
# ============================================================================

@given(
    doc_id=doc_ids,
    content=doc_content,
    metadata=metadata_dict
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_31_document_round_trip(db_manager_with_mock, doc_id, content, metadata):
    """
    **Property 31: Document storage round-trip**
    
    When a document is inserted into the database and then retrieved,
    the content and metadata should match the original values.
    
    Validates: Requirement 7.5
    """
    # Insert document
    success = db_manager_with_mock.insert_document(doc_id, content, metadata)
    assert success, "Document insertion should succeed"
    
    # Retrieve document
    retrieved = db_manager_with_mock.get_document(doc_id)
    assert retrieved is not None, "Retrieved document should not be None"
    
    # Verify content and metadata match
    assert retrieved["content"] == content, "Content should match original"
    assert retrieved["metadata"] == metadata, "Metadata should match original"
    
    # Verify timestamp is present and valid ISO format
    assert "indexed_timestamp" in retrieved, "Should have indexed_timestamp"
    assert isinstance(retrieved["indexed_timestamp"], str), "Timestamp should be string"
    
    # Try to parse ISO format timestamp
    try:
        datetime.fromisoformat(retrieved["indexed_timestamp"])
    except ValueError:
        pytest.fail("Timestamp should be valid ISO format")


# ============================================================================
# PROPERTY 32: Batch Retrieval Returns All Requested Documents
# ============================================================================

@given(documents_to_insert=documents)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_32_batch_retrieval(db_manager_with_mock, documents_to_insert):
    """
    **Property 32: Batch retrieval returns all requested documents**
    
    When multiple documents are inserted and retrieved as a batch,
    the batch retrieval should return all documents that were found.
    
    Validates: Requirement 7.6
    """
    # Insert all documents
    for doc in documents_to_insert:
        success = db_manager_with_mock.insert_document(
            doc["doc_id"],
            doc["content"],
            doc["metadata"]
        )
        assert success, f"Document {doc['doc_id']} insertion should succeed"
    
    # Extract doc_ids
    doc_ids_to_retrieve = [doc["doc_id"] for doc in documents_to_insert]
    
    # Retrieve as batch
    retrieved_batch = db_manager_with_mock.get_documents_batch(doc_ids_to_retrieve)
    
    # Verify all documents were retrieved
    assert len(retrieved_batch) == len(documents_to_insert), \
        "Should retrieve all inserted documents"
    
    # Verify each document's content and metadata
    for original_doc in documents_to_insert:
        doc_id = original_doc["doc_id"]
        assert doc_id in retrieved_batch, f"Document {doc_id} should be in batch"
        
        retrieved_doc = retrieved_batch[doc_id]
        assert retrieved_doc["content"] == original_doc["content"], \
            f"Content mismatch for {doc_id}"
        assert retrieved_doc["metadata"] == original_doc["metadata"], \
            f"Metadata mismatch for {doc_id}"


# ============================================================================
# PROPERTY 33: Bulk Insert Creates All Documents
# ============================================================================

@given(documents_to_insert=documents)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_33_bulk_insert(db_manager_with_mock, documents_to_insert):
    """
    **Property 33: Bulk insert creates all documents**
    
    When documents are inserted via insert_batch, the returned counts
    should match the number of documents inserted, and all should be
    retrievable from the database.
    
    Validates: Requirement 7.4, 13.4
    """
    # Perform bulk insert
    successful_count, failed_count = db_manager_with_mock.insert_batch(documents_to_insert)
    
    # Verify counts
    assert successful_count == len(documents_to_insert), \
        f"All {len(documents_to_insert)} documents should succeed, got {successful_count}"
    assert failed_count == 0, f"No documents should fail, got {failed_count}"
    
    # Verify all documents are retrievable
    doc_ids = [doc["doc_id"] for doc in documents_to_insert]
    retrieved_batch = db_manager_with_mock.get_documents_batch(doc_ids)
    
    assert len(retrieved_batch) == len(documents_to_insert), \
        "All inserted documents should be retrievable"


# ============================================================================
# PROPERTY 44: Database Retry Logic Executes 3 Times
# ============================================================================

def test_property_44_retry_logic_executes_three_times(test_config):
    """
    **Property 44: Database retry logic executes 3 times**
    
    When MongoDB connection fails, the DatabaseManager should attempt
    3 retries with exponential backoff before giving up.
    
    Validates: Requirement 13.1
    """
    from unittest.mock import patch, call, MagicMock
    from pymongo.errors import ServerSelectionTimeoutError
    
    manager = DatabaseManager(test_config)
    
    # Mock MongoClient to always fail with proper exception
    with patch('database.MongoClient') as mock_client_class:
        # Make it raise the proper exception
        mock_client_class.side_effect = ServerSelectionTimeoutError("Connection failed")
        
        # Attempt connection
        result = manager.connect()
        
        # Should have failed
        assert result is False, "Should return False on connection failure"
        
        # Verify MongoClient was called 3 times
        assert mock_client_class.call_count == 3, \
            f"Should attempt connection 3 times, got {mock_client_class.call_count}"


def test_property_44_exponential_backoff(test_config):
    """
    **Property 44 Extended: Exponential backoff timing**
    
    When retrying, the delays should follow: 1s, 2s, 4s
    
    Validates: Requirement 13.1
    """
    from unittest.mock import patch, call, MagicMock
    from pymongo.errors import ServerSelectionTimeoutError
    
    manager = DatabaseManager(test_config)
    
    # Mock both MongoClient and time.sleep
    with patch('database.MongoClient') as mock_client_class, \
         patch('database.time.sleep') as mock_sleep:
        
        # Make MongoClient raise the proper exception
        mock_client_class.side_effect = ServerSelectionTimeoutError("Connection failed")
        
        # Attempt connection
        result = manager.connect()
        
        # Verify sleep was called with correct backoff times
        # Between 3 attempts, we should sleep 2 times: after attempt 1 (1s), after attempt 2 (2s)
        assert mock_sleep.call_count == 2, \
            f"Should sleep 2 times (between 3 attempts), got {mock_sleep.call_count}"
        
        # Check the backoff times: 1, 2
        expected_calls = [call(1), call(2)]
        actual_calls = mock_sleep.call_args_list
        assert actual_calls == expected_calls, \
            f"Expected sleep calls {expected_calls}, got {actual_calls}"


# ============================================================================
# UNIT TESTS (Additional coverage)
# ============================================================================

def test_insert_document_stores_timestamp(db_manager_with_mock):
    """Test that insert_document stores ISO timestamp."""
    doc_id = "test_doc_1"
    content = "Test content"
    metadata = {"key": "value"}
    
    before_insert = datetime.now(timezone.utc)
    success = db_manager_with_mock.insert_document(doc_id, content, metadata)
    after_insert = datetime.now(timezone.utc)
    
    assert success
    retrieved = db_manager_with_mock.get_document(doc_id)
    assert retrieved is not None
    
    # Parse timestamp
    stored_time = datetime.fromisoformat(retrieved["indexed_timestamp"])
    
    # Verify it's within expected range
    assert before_insert <= stored_time <= after_insert, \
        "Stored timestamp should be between before and after insert times"


def test_get_document_nonexistent_returns_none(db_manager_with_mock):
    """Test that getting a nonexistent document returns None."""
    result = db_manager_with_mock.get_document("nonexistent_doc")
    assert result is None


def test_get_documents_batch_partial_results(db_manager_with_mock):
    """Test batch retrieval with some nonexistent documents."""
    # Insert one document
    db_manager_with_mock.insert_document("doc1", "content1", {})
    
    # Request two documents (one exists, one doesn't)
    result = db_manager_with_mock.get_documents_batch(["doc1", "doc2"])
    
    # Should only return the one that exists
    assert len(result) == 1
    assert "doc1" in result
    assert "doc2" not in result


def test_delete_document_removes_from_database(db_manager_with_mock):
    """Test that delete_document removes the document."""
    doc_id = "to_delete"
    db_manager_with_mock.insert_document(doc_id, "content", {})
    
    # Verify it exists
    assert db_manager_with_mock.get_document(doc_id) is not None
    
    # Delete it
    success = db_manager_with_mock.delete_document(doc_id)
    assert success
    
    # Verify it's gone
    assert db_manager_with_mock.get_document(doc_id) is None


def test_delete_document_nonexistent_returns_false(db_manager_with_mock):
    """Test that deleting a nonexistent document returns False."""
    result = db_manager_with_mock.delete_document("nonexistent")
    assert result is False


def test_clear_collection_removes_all_documents(db_manager_with_mock):
    """Test that clear_collection removes all documents."""
    # Insert multiple documents
    for i in range(5):
        db_manager_with_mock.insert_document(f"doc{i}", f"content{i}", {})
    
    # Verify they exist
    stats_before = db_manager_with_mock.get_collection_stats()
    assert stats_before["document_count"] == 5
    
    # Clear collection
    success = db_manager_with_mock.clear_collection()
    assert success
    
    # Verify collection is empty
    stats_after = db_manager_with_mock.get_collection_stats()
    assert stats_after["document_count"] == 0


def test_get_collection_stats_returns_document_count(db_manager_with_mock):
    """Test that get_collection_stats returns document count."""
    # Insert documents
    for i in range(3):
        db_manager_with_mock.insert_document(f"doc{i}", f"content{i}", {})
    
    stats = db_manager_with_mock.get_collection_stats()
    
    assert "document_count" in stats
    assert stats["document_count"] == 3


def test_is_connected_returns_true_after_connect(db_manager_with_mock):
    """Test that is_connected returns True after successful connection."""
    assert db_manager_with_mock.is_connected() is True


def test_is_connected_returns_false_before_connect(test_config):
    """Test that is_connected returns False before connection."""
    manager = DatabaseManager(test_config)
    assert manager.is_connected() is False


def test_disconnect_closes_connection(db_manager_with_mock):
    """Test that disconnect closes the connection."""
    assert db_manager_with_mock.is_connected() is True
    
    db_manager_with_mock.disconnect()
    
    assert db_manager_with_mock.is_connected() is False


def test_insert_batch_with_missing_doc_id(db_manager_with_mock):
    """Test that insert_batch handles documents with missing doc_id."""
    documents = [
        {"doc_id": "doc1", "content": "content1", "metadata": {}},
        {"content": "content2", "metadata": {}},  # Missing doc_id
        {"doc_id": "doc3", "content": "content3", "metadata": {}},
    ]
    
    successful_count, failed_count = db_manager_with_mock.insert_batch(documents)
    
    assert successful_count == 2, "Should insert 2 valid documents"
    assert failed_count == 1, "Should fail 1 document"


def test_insert_batch_progress_tracking(db_manager_with_mock):
    """Test that insert_batch tracks progress correctly."""
    documents = [
        {"doc_id": f"doc{i}", "content": f"content{i}", "metadata": {"index": i}}
        for i in range(10)
    ]
    
    successful_count, failed_count = db_manager_with_mock.insert_batch(documents)
    
    assert successful_count + failed_count == len(documents), \
        "Total count should equal number of documents"


def test_insert_document_with_duplicate_id_updates(db_manager_with_mock):
    """Test that inserting with duplicate ID updates the document."""
    doc_id = "duplicate_doc"
    
    # Insert first version
    db_manager_with_mock.insert_document(doc_id, "original content", {"version": 1})
    retrieved1 = db_manager_with_mock.get_document(doc_id)
    timestamp1 = retrieved1["indexed_timestamp"]
    
    # Insert second version with same ID
    db_manager_with_mock.insert_document(doc_id, "updated content", {"version": 2})
    retrieved2 = db_manager_with_mock.get_document(doc_id)
    
    # Should have new content and metadata
    assert retrieved2["content"] == "updated content"
    assert retrieved2["metadata"]["version"] == 2


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_operations_fail_gracefully_without_connection(test_config):
    """Test that operations fail gracefully when not connected."""
    manager = DatabaseManager(test_config)
    
    # Try operations without connecting
    assert manager.insert_document("doc", "content", {}) is False
    assert manager.get_document("doc") is None
    assert manager.get_documents_batch(["doc"]) == {}
    assert manager.delete_document("doc") is False
    assert manager.clear_collection() is False
    assert manager.get_collection_stats() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
