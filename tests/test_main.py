"""Focused unit tests for the FastAPI application and search endpoint."""

import asyncio
import importlib
import json
import sys
from pathlib import Path

import nltk
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


def _import_main_without_runtime_services(monkeypatch):
    from database import DatabaseManager
    from semantic_indexer import SemanticIndexer

    def forbidden(*args, **kwargs):
        raise AssertionError("application import performed runtime I/O")

    monkeypatch.setattr(nltk, "download", forbidden)
    monkeypatch.setattr(DatabaseManager, "connect", forbidden)
    monkeypatch.setattr(SemanticIndexer, "load_model", forbidden)

    # Ensure preprocessing is imported under the download guard for this test.
    sys.modules.pop("main", None)
    sys.modules.pop("indexer", None)
    sys.modules.pop("preprocess", None)
    return importlib.import_module("main")


def _call_search(main, **request_data):
    request = main.SearchRequest(**request_data)
    return asyncio.run(main.search(request))


def _json_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_models_define_required_fields_and_defaults(monkeypatch):
    main = _import_main_without_runtime_services(monkeypatch)

    search = main.SearchRequest(query="hybrid retrieval")
    assert search.top_k == 10
    assert search.fusion_method == "rrf"

    index = main.IndexRequest(documents=[])
    assert index.batch_size == 100

    assert main.SearchResponse(
        results=[], execution_time_ms=1.5, query_processed="hybrid retrieval"
    ).results == []
    assert main.IndexResponse(
        indexed_count=2, failed_count=0, status="ok", message="indexed"
    ).indexed_count == 2
    assert main.HealthResponse(status="ok").status == "ok"
    assert main.ErrorResponse(error_message="failed").error_message == "failed"


def test_app_cors_and_global_components_are_initialized(monkeypatch):
    main = _import_main_without_runtime_services(monkeypatch)

    assert main.app.title == "Hybrid IR System"
    cors = next(
        middleware
        for middleware in main.app.user_middleware
        if middleware.cls is main.CORSMiddleware
    )
    options = getattr(cors, "kwargs", None)
    if options is None:
        options = cors.options
    assert options["allow_origins"] == ["*"]
    assert options["allow_methods"] == ["*"]
    assert options["allow_headers"] == ["*"]

    assert isinstance(main.config, main.Config)
    assert isinstance(main.preprocessor, main.TextPreprocessor)
    assert isinstance(main.indexer, main.IndexingEngine)
    assert isinstance(main.semantic_indexer, main.SemanticIndexer)
    assert isinstance(main.db_manager, main.DatabaseManager)
    assert isinstance(main.search_engine, main.SearchEngine)

    assert main.indexer.config is main.config
    assert main.semantic_indexer.config is main.config
    assert main.db_manager.config is main.config
    assert main.search_engine.database_manager is main.db_manager
    assert main.semantic_indexer.model is None
    assert main.db_manager.client is None

    paths = {route.path for route in main.app.routes}
    assert "/search" in paths
    assert "/health" in paths
    assert "/index" not in paths


def test_health_returns_ok_status(monkeypatch):
    main = _import_main_without_runtime_services(monkeypatch)

    response = asyncio.run(main.health())

    assert isinstance(response, main.HealthResponse)
    assert response.status == "ok"


def test_search_returns_hydrated_ranked_results(monkeypatch, caplog):
    main = _import_main_without_runtime_services(monkeypatch)
    calls = {}

    def fake_search(query, top_k, fusion_method):
        calls["search"] = (query, top_k, fusion_method)
        return {
            "results": [
                (
                    "doc-2",
                    0.75,
                    {"bm25_score": 1.2, "semantic_score": 0.8},
                ),
                ("doc-1", 0.5, {"tfidf_score": 0.5}),
            ],
            "execution_time_ms": 4.25,
            "query_processed": "hybrid retrieval",
        }

    def fake_get_documents_batch(document_ids):
        calls["document_ids"] = document_ids
        return {
            "doc-1": {"content": "First", "metadata": {"title": "One"}},
            "doc-2": {"content": "Second", "metadata": {"title": "Two"}},
        }

    monkeypatch.setattr(main.search_engine, "search", fake_search)
    monkeypatch.setattr(main.db_manager, "get_documents_batch", fake_get_documents_batch)

    response = _call_search(
        main, query="hybrid retrieval", top_k=2, fusion_method="rrf"
    )

    assert isinstance(response, main.SearchResponse)
    assert calls["search"] == ("hybrid retrieval", 2, "rrf")
    assert calls["document_ids"] == ["doc-2", "doc-1"]
    assert response.execution_time_ms == 4.25
    assert response.query_processed == "hybrid retrieval"
    assert [item["document_id"] for item in response.results] == ["doc-2", "doc-1"]
    assert response.results[0]["content"] == "Second"
    assert response.results[0]["metadata"] == {"title": "Two"}
    assert response.results[0]["combined_score"] == 0.75
    assert response.results[0]["individual_scores"] == {
        "bm25_score": 1.2,
        "tfidf_score": 0.0,
        "semantic_score": 0.8,
        "rrf_score": 0.0,
    }
    assert "endpoint=/search" in caplog.text
    assert "status_code=200" in caplog.text


@pytest.mark.parametrize(
    ("request_data", "message"),
    [
        ({"query": ""}, "Query cannot be empty"),
        ({"query": "   "}, "Query cannot be empty"),
        ({"query": "valid", "top_k": 0}, "top_k must be >= 1 and <= 1000"),
        ({"query": "valid", "top_k": 1001}, "top_k must be >= 1 and <= 1000"),
        (
            {"query": "valid", "fusion_method": "invalid"},
            "fusion_method must be: rrf, bm25, tfidf, semantic",
        ),
    ],
)
def test_search_rejects_invalid_requests(monkeypatch, request_data, message):
    main = _import_main_without_runtime_services(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("search engine must not run for invalid requests")

    monkeypatch.setattr(main.search_engine, "search", forbidden)
    response = _call_search(main, **request_data)

    assert response.status_code == 400
    assert _json_body(response) == {"error_message": message}


def test_search_returns_empty_results_without_database_query(monkeypatch):
    main = _import_main_without_runtime_services(monkeypatch)
    monkeypatch.setattr(
        main.search_engine,
        "search",
        lambda *args: {
            "results": [],
            "execution_time_ms": 0.5,
            "query_processed": "missing",
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("database must not be queried when there are no results")

    monkeypatch.setattr(main.db_manager, "get_documents_batch", forbidden)
    response = _call_search(main, query="missing")

    assert isinstance(response, main.SearchResponse)
    assert response.results == []


def test_search_failure_returns_error_response(monkeypatch, caplog):
    main = _import_main_without_runtime_services(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(main.search_engine, "search", fail)
    response = _call_search(main, query="valid")

    assert response.status_code == 500
    assert _json_body(response) == {"error_message": "index unavailable"}
    assert "endpoint=/search" in caplog.text
    assert "status_code=500" in caplog.text
