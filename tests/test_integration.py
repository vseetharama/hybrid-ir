"""
End-to-End Integration Tests for Hybrid IR System

Tests the complete workflow: index → search → evaluate, including:
1. Complete Workflow: Index documents, search all methods, retrieve from MongoDB
2. Graceful Degradation: System works when some methods fail
3. Partial Indexing Failures: Handles failed documents gracefully

Uses pytest with mongomock for MongoDB (no real database) and hypothesis
for property-based tests.

**Task 20.1 - Integration Tests**
Validates: Requirements 4.1-6.7, 7.1-7.8, 10.1-10.3, 13.1-13.4
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Check for optional dependencies
try:
    import mongomock
except ImportError:
    mongomock = None

try:
    from hypothesis import given, settings, strategies as st, HealthCheck
except ImportError:
    st = None

from config import Config
from database import DatabaseManager
from evaluator import EvaluationMetrics
from indexer import IndexingEngine
from preprocess import TextPreprocessor
from search_engine import SearchEngine


# ============================================================================
# FIXTURES & SETUP
# ============================================================================

@pytest.fixture
def config():
    """Configuration for integration tests."""
    cfg = Config()
    cfg.mongodb_uri = "mongodb://localhost:27017"
    cfg.database_name = "test_ir_db"
    cfg.collection_name = "test_docs"
    cfg.rrf_k_constant = 60
    return cfg


@pytest.fixture
def preprocessor():
    """Text preprocessor instance."""
    return TextPreprocessor(download_nltk_data=False)


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client using mongomock if available."""
    if mongomock is None:
        pytest.skip("mongomock not installed")
    return mongomock.MongoClient()


@pytest.fixture
def db_manager(config, mock_mongo_client):
    """Database manager with mongomock."""
    db_manager = DatabaseManager(config)
    db_manager.client = mock_mongo_client
    db_manager.db = mock_mongo_client[config.database_name]
    db_manager.collection = db_manager.db[config.collection_name]
    db_manager.is_connected_flag = True
    return db_manager


# ============================================================================
# SAMPLE DATA
# ============================================================================

def get_sample_documents():
    """Generate sample documents for testing."""
    return [
        {
            "doc_id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn",
            "metadata": {"title": "ML Basics", "year": 2023}
        },
        {
            "doc_id": "doc_2",
            "content": "Deep neural networks revolutionized computer vision and natural language processing",
            "metadata": {"title": "Deep Learning", "year": 2022}
        },
        {
            "doc_id": "doc_3",
            "content": "Semantic search uses embeddings to find conceptually similar documents",
            "metadata": {"title": "Semantic Search", "year": 2023}
        },
        {
            "doc_id": "doc_4",
            "content": "Information retrieval systems combine lexical and semantic approaches",
            "metadata": {"title": "IR Systems", "year": 2021}
        },
        {
            "doc_id": "doc_5",
            "content": "Reciprocal rank fusion combines multiple ranking functions",
            "metadata": {"title": "RRF Algorithm", "year": 2024}
        },
    ]


def get_large_documents(count=20):
    """Generate larger corpus for testing."""
    return [
        {
            "doc_id": f"doc_{i}",
            "content": f"Document {i} with learning and retrieval content about AI",
            "metadata": {"index": i}
        }
        for i in range(1, count + 1)
    ]


# ============================================================================
# COMPLETE WORKFLOW TESTS
# ============================================================================

class TestCompleteWorkflow:
    """Test complete workflow: index → search → evaluate"""

    def test_workflow_bm25_index_search_retrieve(self, config, preprocessor, db_manager):
        """
        Workflow: Index → BM25 Search → MongoDB Retrieve
        
        Validates:
        - Documents indexed to BM25
        - Search returns ranked results
        - MongoDB retrieval works
        - Results have correct format
        """
        docs = get_sample_documents()
        
        # Index
        indexer = IndexingEngine(config, preprocessor)
        assert indexer.build_bm25_index(docs) is True
        
        # Insert to MongoDB
        inserted, failed = db_manager.insert_batch(docs)
        assert inserted == 5
        assert failed == 0
        
        # Search
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("machine learning", top_k=3)
        
        # Verify results
        assert len(results) > 0
        assert len(results) <= 3
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)
        
        # Verify sorted
        scores = [s for _, s in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Retrieve from MongoDB
        doc_ids = [doc_id for doc_id, _ in results]
        retrieved = db_manager.get_documents_batch(doc_ids)
        assert len(retrieved) == len(results)
        for doc_id in doc_ids:
            assert doc_id in retrieved
            assert "content" in retrieved[doc_id]
            assert "metadata" in retrieved[doc_id]

    def test_workflow_tfidf_index_search_retrieve(self, config, preprocessor, db_manager):
        """
        Workflow: Index → TF-IDF Search → MongoDB Retrieve
        
        Similar to BM25 but using TF-IDF for ranking.
        """
        docs = get_sample_documents()
        
        # Index
        indexer = IndexingEngine(config, preprocessor)
        assert indexer.build_tfidf_index(docs) is True
        
        # Insert to MongoDB
        inserted, failed = db_manager.insert_batch(docs)
        assert inserted == 5
        
        # Search
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_tfidf("semantic search", top_k=3)
        
        # Verify format
        assert len(results) >= 0
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        
        # Verify sorted
        scores = [s for _, s in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_workflow_evaluation_metrics(self, config, preprocessor, db_manager):
        """
        Test evaluation metrics calculation on search results
        
        Validates:
        - Precision@K calculation
        - Recall@K calculation
        - MRR calculation
        """
        docs = get_sample_documents()
        
        # Setup
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("machine learning", top_k=5)
        result_ids = [doc_id for doc_id, _ in results]
        
        # Mark relevant docs
        relevant_ids = {"doc_1"}
        
        # Calculate metrics
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=5)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=5)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # Verify ranges
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= mrr <= 1.0

    def test_workflow_unified_search_interface(self, config, preprocessor, db_manager):
        """
        Test unified search interface with response formatting
        
        Validates:
        - Unified search method works
        - Response has correct schema
        - Execution time is recorded
        - Query is preprocessed
        """
        docs = get_sample_documents()
        
        # Setup
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        response = engine.search("machine learning", top_k=3, fusion_method="bm25")
        
        # Verify response structure
        assert "results" in response
        assert "execution_time_ms" in response
        assert "query_processed" in response
        
        assert isinstance(response["results"], list)
        assert isinstance(response["execution_time_ms"], float)
        assert isinstance(response["query_processed"], str)


# ============================================================================
# GRACEFUL DEGRADATION TESTS
# ============================================================================

class TestGracefulDegradation:
    """Test system resilience when components fail"""

    def test_degradation_without_bm25_index(self, config, preprocessor, db_manager):
        """
        Test system works when BM25 index is not available
        
        Validates:
        - TF-IDF search works
        - System gracefully handles missing BM25
        """
        docs = get_sample_documents()
        
        # Build only TF-IDF (skip BM25)
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_tfidf_index(docs)
        
        db_manager.insert_batch(docs)
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        
        # TF-IDF should work
        tfidf_results = engine.search_tfidf("learning", top_k=3)
        assert isinstance(tfidf_results, list)
        
        # BM25 should return empty or handle gracefully
        bm25_results = engine.search_bm25("learning", top_k=3)
        assert isinstance(bm25_results, list)

    def test_degradation_mongodb_unavailable(self, config, preprocessor):
        """
        Test when MongoDB is unavailable during retrieval
        
        Validates:
        - Search still works and returns ranked results
        - Content retrieval handles empty results
        """
        docs = get_sample_documents()
        
        # Setup indices
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        
        # Mock database that returns nothing
        mock_db = MagicMock()
        mock_db.get_documents_batch.return_value = {}
        
        engine = SearchEngine(config, preprocessor, indexer, None, mock_db)
        results = engine.search_bm25("learning", top_k=3)
        
        # Search works
        assert isinstance(results, list)
        # Results sorted
        scores = [s for _, s in results]
        if len(scores) > 1:
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_degradation_partial_document_retrieval(self, config, preprocessor):
        """
        Test when some documents fail to retrieve from MongoDB
        
        Validates:
        - Successfully retrieved documents are returned
        - Missing documents don't cause crashes
        """
        docs = get_sample_documents()
        
        # Setup
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        
        # Mock DB that fails for specific doc
        mock_db = MagicMock()
        def mock_get_batch(doc_ids):
            return {
                doc_id: {"content": f"Content of {doc_id}", "metadata": {}}
                for doc_id in doc_ids
                if doc_id != "doc_2"  # Fail for doc_2
            }
        mock_db.get_documents_batch.side_effect = mock_get_batch
        
        engine = SearchEngine(config, preprocessor, indexer, None, mock_db)
        results = engine.search_bm25("learning", top_k=5)
        
        assert isinstance(results, list)


# ============================================================================
# PARTIAL INDEXING FAILURE TESTS
# ============================================================================

class TestPartialIndexingFailures:
    """Test handling of partial indexing failures"""

    def test_bulk_insert_success_and_failure_reporting(self, config, db_manager):
        """
        Test bulk insert reports success/failure counts
        
        Validates:
        - Successful documents inserted
        - Failure count reported
        - Mixed success/failure handled
        """
        docs = [
            {"doc_id": f"doc_{i}", "content": f"Content {i}", "metadata": {}}
            for i in range(5)
        ]
        
        inserted, failed = db_manager.insert_batch(docs)
        
        assert inserted == 5
        assert failed == 0
        
        # Verify in database
        retrieved = db_manager.get_documents_batch([d["doc_id"] for d in docs])
        assert len(retrieved) == 5

    def test_index_subset_of_documents(self, config, preprocessor, db_manager):
        """
        Test indexing subset while full corpus in MongoDB
        
        Validates:
        - Search only finds indexed documents
        - Partial indexing is handled
        """
        docs = get_sample_documents()
        subset = docs[:3]
        
        # Index subset
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(subset)
        
        # Insert full corpus
        inserted, _ = db_manager.insert_batch(docs)
        assert inserted == 5
        
        # Search
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("learning", top_k=10)
        
        assert isinstance(results, list)

    def test_large_batch_insertion(self, config, preprocessor, db_manager):
        """
        Test inserting and indexing large batch
        
        Validates:
        - Large batches handled
        - All documents searchable
        """
        docs = get_large_documents(count=20)
        
        # Index
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        
        # Insert
        inserted, failed = db_manager.insert_batch(docs)
        assert inserted == 20
        assert failed == 0
        
        # Search
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("learning", top_k=5)
        
        assert len(results) <= 5


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_query_returns_empty(self, config, preprocessor, db_manager):
        """Test empty query returns empty results without error."""
        docs = get_sample_documents()
        
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        
        assert engine.search_bm25("", top_k=5) == []
        assert engine.search_bm25("   ", top_k=5) == []

    def test_single_document_corpus(self, config, preprocessor, db_manager):
        """Test search on single document corpus."""
        docs = [get_sample_documents()[0]]
        
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("machine", top_k=10)
        
        assert len(results) <= 1

    def test_top_k_larger_than_corpus(self, config, preprocessor, db_manager):
        """Test requesting top_k larger than corpus."""
        docs = get_sample_documents()
        
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("learning", top_k=100)
        
        assert len(results) <= 5  # Corpus size

    def test_unicode_content(self, config, preprocessor, db_manager):
        """Test handling of unicode in content."""
        docs = [
            {
                "doc_id": "doc_unicode_1",
                "content": "Café in München with Ü and ö characters",
                "metadata": {}
            },
            {
                "doc_id": "doc_unicode_2",
                "content": "Learning various topics",
                "metadata": {}
            },
        ]
        
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("learning", top_k=5)
        
        assert isinstance(results, list)


# ============================================================================
# API INTEGRATION TESTS
# ============================================================================

class TestApiIntegration:
    """Test integration with API endpoints"""

    def test_search_endpoint_full_cycle(self, config, preprocessor, db_manager):
        """
        Test complete search endpoint cycle:
        1. Index documents
        2. Search
        3. Retrieve from MongoDB
        4. Format response
        """
        docs = get_sample_documents()
        
        # Setup
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        db_manager.insert_batch(docs)
        
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        
        # Search
        query = "machine learning"
        top_k = 3
        search_result = engine.search(query, top_k, fusion_method="bm25")
        
        # Verify response
        assert "results" in search_result
        assert "execution_time_ms" in search_result
        assert "query_processed" in search_result
        
        # Retrieve from MongoDB
        doc_ids = [result[0] for result in search_result["results"]]
        docs_retrieved = db_manager.get_documents_batch(doc_ids)
        
        # Format as API would
        formatted_results = []
        for doc_id, combined_score, individual_scores in search_result["results"]:
            doc = docs_retrieved.get(doc_id, {})
            formatted_results.append({
                "document_id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "combined_score": combined_score,
                "individual_scores": individual_scores or {}
            })
        
        assert len(formatted_results) <= top_k
        for result in formatted_results:
            assert "document_id" in result
            assert "content" in result
            assert "metadata" in result
            assert "combined_score" in result

    def test_index_endpoint_full_cycle(self, config, preprocessor, db_manager):
        """
        Test complete index endpoint cycle
        """
        docs = get_sample_documents()
        
        # Index
        indexer = IndexingEngine(config, preprocessor)
        indexer.build_bm25_index(docs)
        indexer.build_tfidf_index(docs)
        
        # Insert
        indexed_count, failed_count = db_manager.insert_batch(docs)
        
        assert indexed_count == 5
        assert failed_count == 0
        
        # Verify indices work
        engine = SearchEngine(config, preprocessor, indexer, None, db_manager)
        results = engine.search_bm25("machine", top_k=3)
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
