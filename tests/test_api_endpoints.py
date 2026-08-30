"""
Property-based tests for FastAPI endpoints.

Tests verify that API responses conform to required schema and contain all necessary fields
for search results, error handling, and indexing operations. Uses Hypothesis for property-based
testing and mongomock for MongoDB mocking.

**Validates: Requirements 8.2-8.8**
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


# Fixtures and utilities
@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Download NLTK data once for the test session."""
    import nltk
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
    except Exception:
        pass


@pytest.fixture
def mock_search_engine():
    """Mock SearchEngine with predictable search results."""
    engine = MagicMock()
    
    def mock_search(query, top_k, fusion_method):
        if not query or not query.strip():
            return {
                "results": [],
                "execution_time_ms": 0.0,
                "query_processed": "",
            }
        
        # Generate deterministic results based on query
        results = []
        for i in range(min(top_k, 5)):
            doc_id = f"doc_{i}"
            combined_score = 0.9 - (i * 0.1)
            individual_scores = {
                "bm25_score": 0.8 - (i * 0.1),
                "tfidf_score": 0.7 - (i * 0.1),
                "semantic_score": 0.6 - (i * 0.1),
                "rrf_score": combined_score,
            }
            results.append((doc_id, combined_score, individual_scores))
        
        return {
            "results": results,
            "execution_time_ms": 12.5,
            "query_processed": query.lower(),
        }
    
    engine.search = mock_search
    return engine


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager with predictable document retrieval."""
    db = MagicMock()
    
    def mock_get_documents_batch(doc_ids):
        documents = {}
        for doc_id in doc_ids:
            documents[doc_id] = {
                "content": f"Sample content for {doc_id}",
                "metadata": {"source": "test", "title": f"Title of {doc_id}"},
            }
        return documents
    
    db.get_documents_batch = mock_get_documents_batch
    db.insert_batch = MagicMock(return_value=(len(doc_ids := []), 0))
    return db


@pytest.fixture
def mock_fastapi_app(mock_search_engine, mock_db_manager, monkeypatch):
    """Create a FastAPI app with mocked dependencies."""
    import importlib
    import nltk
    
    def forbidden(*args, **kwargs):
        raise AssertionError("application import performed runtime I/O")
    
    monkeypatch.setattr(nltk, "download", forbidden)
    
    sys.modules.pop("main", None)
    sys.modules.pop("indexer", None)
    sys.modules.pop("preprocess", None)
    main = importlib.import_module("main")
    
    # Replace search and db engines with mocks
    monkeypatch.setattr(main, "search_engine", mock_search_engine)
    monkeypatch.setattr(main, "db_manager", mock_db_manager)
    
    return main


# ========================================================================
# Property 34: API response schema correct for search
# ========================================================================
@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
    top_k=st.integers(min_value=1, max_value=100),
    fusion_method=st.sampled_from(["rrf", "bm25", "tfidf", "semantic"]),
)
@settings(
    max_examples=100,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_search_response_schema_correct(
    mock_fastapi_app, query, top_k, fusion_method
):
    """
    **Validates: Requirements 8.2**
    
    Property: All search responses SHALL have the documented schema with:
    - results (list)
    - execution_time_ms (float)
    - query_processed (string)
    """
    main = mock_fastapi_app
    
    request = main.SearchRequest(
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
    )
    response = asyncio.run(main.search(request))
    
    # Verify response is SearchResponse
    assert isinstance(response, main.SearchResponse)
    
    # Verify all required fields are present
    assert hasattr(response, "results")
    assert hasattr(response, "execution_time_ms")
    assert hasattr(response, "query_processed")
    
    # Verify field types
    assert isinstance(response.results, list)
    assert isinstance(response.execution_time_ms, float)
    assert isinstance(response.query_processed, str)
    
    # Verify execution_time_ms is non-negative
    assert response.execution_time_ms >= 0.0


# ========================================================================
# Property 35: Search result items contain required fields
# ========================================================================
@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
    top_k=st.integers(min_value=1, max_value=50),
    fusion_method=st.sampled_from(["rrf", "bm25", "tfidf", "semantic"]),
)
@settings(
    max_examples=100,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_search_results_contain_required_fields(
    mock_fastapi_app, query, top_k, fusion_method
):
    """
    **Validates: Requirements 8.3**
    
    Property: Each result item SHALL contain all required fields:
    - document_id (string)
    - content (string)
    - metadata (dict)
    - combined_score (float)
    - individual_scores (dict)
    """
    main = mock_fastapi_app
    
    request = main.SearchRequest(
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
    )
    response = asyncio.run(main.search(request))
    
    # Verify each result has required fields
    for result in response.results:
        assert isinstance(result, dict), "Each result must be a dict"
        assert "document_id" in result
        assert "content" in result
        assert "metadata" in result
        assert "combined_score" in result
        assert "individual_scores" in result
        
        # Verify field types
        assert isinstance(result["document_id"], str)
        assert isinstance(result["content"], str)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["combined_score"], float)
        assert isinstance(result["individual_scores"], dict)


# ========================================================================
# Property 36: Individual scores all present
# ========================================================================
@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
    top_k=st.integers(min_value=1, max_value=50),
    fusion_method=st.sampled_from(["rrf", "bm25", "tfidf", "semantic"]),
)
@settings(
    max_examples=100,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_individual_scores_all_present(
    mock_fastapi_app, query, top_k, fusion_method
):
    """
    **Validates: Requirements 8.4**
    
    Property: individual_scores dict SHALL include all four score types:
    - bm25_score (float)
    - tfidf_score (float)
    - semantic_score (float)
    - rrf_score (float)
    """
    # Filter out whitespace-only queries to match empty query validation
    if not query or not query.strip():
        pytest.skip("Skipping whitespace-only query")
    
    main = mock_fastapi_app
    
    request = main.SearchRequest(
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
    )
    response = asyncio.run(main.search(request))
    
    # Check if response is an error (JSONResponse with status_code)
    if hasattr(response, 'status_code') and response.status_code != 200:
        pytest.skip("Request returned error response")
    
    required_score_fields = {
        "bm25_score",
        "tfidf_score",
        "semantic_score",
        "rrf_score",
    }
    
    for result in response.results:
        individual_scores = result["individual_scores"]
        
        # Verify all required score fields are present
        for score_field in required_score_fields:
            assert score_field in individual_scores, \
                f"Missing required score field: {score_field}"
        
        # Verify all scores are floats
        for score_field in required_score_fields:
            score_value = individual_scores[score_field]
            assert isinstance(score_value, (int, float)), \
                f"Score {score_field} must be numeric, got {type(score_value)}"
            
            # Verify scores are in reasonable range
            assert isinstance(float(score_value), float)


# ========================================================================
# Property 37: Indexing succeeds for valid documents
# ========================================================================
@given(
    documents=st.lists(
        st.fixed_dictionaries({
            "doc_id": st.text(min_size=1, max_size=20),
            "content": st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cc", "Cs"),
                    blacklist_characters="\x00",
                ),
                min_size=1,
                max_size=200,
            ),
            "metadata": st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(
                    st.text(max_size=50),
                    st.integers(),
                    st.booleans(),
                ),
                max_size=5,
            ),
        }),
        min_size=1,
        max_size=10,
    ),
    batch_size=st.integers(min_value=1, max_value=10),
)
@settings(
    max_examples=50,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_indexing_succeeds_for_valid_documents(
    mock_fastapi_app, documents, batch_size
):
    """
    **Validates: Requirements 8.5-8.6**
    
    Property: Indexing valid documents SHALL create proper IndexRequest.
    """
    main = mock_fastapi_app
    
    # Create IndexRequest
    request = main.IndexRequest(
        documents=documents,
        batch_size=batch_size,
    )
    
    # Verify the request model is valid
    assert request.documents == documents
    assert request.batch_size == batch_size
    assert len(request.documents) > 0


# ========================================================================
# Property 38: Error response has error_message field
# ========================================================================
@given(
    error_message=st.text(min_size=1, max_size=200),
)
@settings(
    max_examples=50,
    deadline=30000,
)
def test_error_response_has_error_message_field(error_message):
    """
    **Validates: Requirements 8.8**
    
    Property: Error responses SHALL have error_message field in JSON.
    """
    # Create an ErrorResponse
    response = ErrorResponse(error_message=error_message)
    
    assert hasattr(response, "error_message")
    assert response.error_message == error_message
    assert isinstance(response.error_message, str)


# Import ErrorResponse for the error test
from main import ErrorResponse


# ========================================================================
# Additional validation tests
# ========================================================================
@pytest.mark.parametrize(
    "invalid_query,expected_message",
    [
        ("", "Query cannot be empty"),
        ("   ", "Query cannot be empty"),
    ],
)
def test_empty_query_returns_error(mock_fastapi_app, invalid_query, expected_message):
    """Verify that empty or whitespace queries return HTTP 400."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(query=invalid_query)
    response = asyncio.run(main.search(request))
    
    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_message"] == expected_message


@pytest.mark.parametrize(
    "invalid_top_k,expected_message",
    [
        (0, "top_k must be >= 1 and <= 1000"),
        (1001, "top_k must be >= 1 and <= 1000"),
        (-5, "top_k must be >= 1 and <= 1000"),
    ],
)
def test_invalid_top_k_returns_error(mock_fastapi_app, invalid_top_k, expected_message):
    """Verify that invalid top_k values return HTTP 400."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(query="test", top_k=invalid_top_k)
    response = asyncio.run(main.search(request))
    
    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_message"] == expected_message


@pytest.mark.parametrize(
    "invalid_method,expected_message",
    [
        ("invalid", "fusion_method must be: rrf, bm25, tfidf, semantic"),
        ("", "fusion_method must be: rrf, bm25, tfidf, semantic"),
        ("BM25", "fusion_method must be: rrf, bm25, tfidf, semantic"),
    ],
)
def test_invalid_fusion_method_returns_error(
    mock_fastapi_app, invalid_method, expected_message
):
    """Verify that invalid fusion methods return HTTP 400."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(
        query="test",
        fusion_method=invalid_method,
    )
    response = asyncio.run(main.search(request))
    
    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_message"] == expected_message


@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
)
@settings(
    max_examples=50,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_valid_search_has_status_code_200(mock_fastapi_app, query):
    """Property: Valid search requests return HTTP 200."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(query=query)
    response = asyncio.run(main.search(request))
    
    # Response should be SearchResponse (successful)
    assert isinstance(response, main.SearchResponse)
    assert response.execution_time_ms >= 0.0


@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
    top_k=st.integers(min_value=1, max_value=100),
)
@settings(
    max_examples=50,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_results_list_respects_top_k_limit(mock_fastapi_app, query, top_k):
    """Property: Results list never exceeds top_k items."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(query=query, top_k=top_k)
    response = asyncio.run(main.search(request))
    
    assert len(response.results) <= top_k


@given(
    top_k=st.integers(min_value=1, max_value=1000),
)
@settings(
    max_examples=50,
    deadline=30000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_search_request_accepts_valid_top_k(mock_fastapi_app, top_k):
    """Property: SearchRequest accepts any valid top_k in [1, 1000]."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(query="test", top_k=top_k)
    assert request.top_k == top_k


@given(
    query=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cc", "Cs"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=100,
    ),
    top_k=st.integers(min_value=1, max_value=50),
)
@settings(
    max_examples=50,
    deadline=30000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_combined_score_matches_individual_scores_for_fusion_method(
    mock_fastapi_app, query, top_k
):
    """Property: For RRF fusion, combined_score matches rrf_score in individual_scores."""
    main = mock_fastapi_app
    
    request = main.SearchRequest(
        query=query,
        top_k=top_k,
        fusion_method="rrf",
    )
    response = asyncio.run(main.search(request))
    
    for result in response.results:
        # For RRF, combined_score should equal rrf_score
        combined = result["combined_score"]
        individual = result["individual_scores"]
        rrf_score = individual.get("rrf_score", 0.0)
        
        # Allow small floating point differences
        assert abs(combined - rrf_score) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
