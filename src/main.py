"""FastAPI application for the Hybrid IR System."""

import time
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:  # Package import (for ``uvicorn src.main:app``)
    from .config import Config
    from .database import DatabaseManager
    from .indexer import IndexingEngine
    from .logger import get_logger
    from .preprocess import TextPreprocessor
    from .search_engine import SearchEngine
    from .semantic_indexer import SemanticIndexer
except ImportError:  # Direct module import when ``src`` is on sys.path
    from config import Config
    from database import DatabaseManager
    from indexer import IndexingEngine
    from logger import get_logger
    from preprocess import TextPreprocessor
    from search_engine import SearchEngine
    from semantic_indexer import SemanticIndexer


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    fusion_method: str = "rrf"


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    execution_time_ms: float
    query_processed: str


class IndexRequest(BaseModel):
    documents: List[Dict[str, Any]]
    batch_size: int = 100


class IndexResponse(BaseModel):
    indexed_count: int
    failed_count: int
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error_message: str


app = FastAPI(title="Hybrid IR System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory document cache (for when MongoDB is unavailable)
_document_cache = {}

# Construct dependencies without connecting to MongoDB or loading/downloading
# the sentence-transformer model. Endpoint tasks own those runtime operations.
config = Config.from_env()
config.log_settings()
logger = get_logger(__name__, config)
preprocessor = TextPreprocessor(download_nltk_data=False)
indexer = IndexingEngine(config, preprocessor)
semantic_indexer = SemanticIndexer(config)
db_manager = DatabaseManager(config)
search_engine = SearchEngine(
    config,
    preprocessor,
    indexer,
    semantic_indexer,
    database_manager=db_manager,
)


def _error_response(status_code: int, message: str) -> JSONResponse:
    """Build the API's documented error response shape."""
    error = ErrorResponse(error_message=message)
    return JSONResponse(status_code=status_code, content=error.model_dump())


def _log_search_request(started_at: float, status_code: int) -> None:
    """Log endpoint timing and outcome; the logger formatter adds the timestamp."""
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info(
        "endpoint=/search execution_time_ms=%.3f status_code=%d",
        elapsed_ms,
        status_code,
    )


def _log_index_request(started_at: float, status_code: int, doc_count: int = 0) -> None:
    """Log index endpoint timing and outcome."""
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info(
        "endpoint=/index execution_time_ms=%.3f status_code=%d documents=%d",
        elapsed_ms,
        status_code,
        doc_count,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post(
    "/search",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def search(request: SearchRequest):
    """Search indexed documents and hydrate ranked results from MongoDB."""
    started_at = time.perf_counter()

    if not request.query or not request.query.strip():
        _log_search_request(started_at, 400)
        return _error_response(400, "Query cannot be empty")

    if not 1 <= request.top_k <= 1000:
        _log_search_request(started_at, 400)
        return _error_response(400, "top_k must be >= 1 and <= 1000")

    valid_methods = {"rrf", "bm25", "tfidf", "semantic"}
    if request.fusion_method not in valid_methods:
        _log_search_request(started_at, 400)
        return _error_response(
            400, "fusion_method must be: rrf, bm25, tfidf, semantic"
        )

    try:
        search_output = search_engine.search(
            request.query, request.top_k, request.fusion_method
        )
        ranked_results = search_output.get("results", [])[: request.top_k]
        document_ids = [result[0] for result in ranked_results]
        
        # Try MongoDB first, fall back to cache
        documents = {}
        if db_manager.is_connected() or document_ids:
            try:
                documents = db_manager.get_documents_batch(document_ids) if document_ids else {}
            except:
                pass
        
        # Fill in from cache if not found in MongoDB
        for doc_id in document_ids:
            if str(doc_id) not in documents and str(doc_id) in _document_cache:
                documents[str(doc_id)] = _document_cache[str(doc_id)]

        score_names = (
            "bm25_score",
            "tfidf_score",
            "semantic_score",
            "rrf_score",
        )
        formatted_results = []
        for document_id, combined_score, individual_scores in ranked_results:
            document = documents.get(str(document_id), documents.get(document_id, {}))
            scores = individual_scores or {}
            formatted_results.append(
                {
                    "document_id": str(document_id),
                    "content": document.get("content", ""),
                    "metadata": document.get("metadata", {}),
                    "combined_score": float(combined_score),
                    "individual_scores": {
                        name: float(scores.get(name, 0.0)) for name in score_names
                    },
                }
            )

        response = SearchResponse(
            results=formatted_results,
            execution_time_ms=float(search_output.get("execution_time_ms", 0.0)),
            query_processed=str(search_output.get("query_processed", "")),
        )
        _log_search_request(started_at, 200)
        return response
    except Exception as exc:
        logger.exception("Search request failed: %s", exc)
        _log_search_request(started_at, 500)
        return _error_response(500, str(exc))


@app.post(
    "/index",
    response_model=IndexResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def index(request: IndexRequest):
    """Index documents and build search indices."""
    started_at = time.perf_counter()

    if not request.documents:
        _log_index_request(started_at, 400, 0)
        return _error_response(400, "Documents list cannot be empty")

    try:
        # Validate document structure
        for doc in request.documents:
            if "doc_id" not in doc or "content" not in doc:
                _log_index_request(started_at, 400, len(request.documents))
                return _error_response(400, "Each document must have doc_id and content")

        # Process in batches
        total_indexed = 0
        total_failed = 0
        batch_size = request.batch_size or 100

        for i in range(0, len(request.documents), batch_size):
            batch = request.documents[i : i + batch_size]

            try:
                # Cache documents in memory
                for doc in batch:
                    _document_cache[str(doc.get("doc_id"))] = {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    }

                # Connect to MongoDB if not already connected
                if not db_manager.is_connected():
                    if not db_manager.connect():
                        logger.warning("MongoDB connection failed, continuing without DB")

                # Build indices
                indexer.build_bm25_index(batch)
                indexer.build_tfidf_index(batch)

                # Try to load and use semantic indexer
                try:
                    semantic_indexer.load_model()
                    semantic_indexer.build_faiss_index(batch)
                except Exception as se:
                    logger.warning("Semantic indexing failed: %s, continuing", se)

                # Insert to MongoDB if connected
                if db_manager.is_connected():
                    indexed, failed = db_manager.insert_batch(batch)
                    total_indexed += indexed
                    total_failed += failed
                else:
                    total_indexed += len(batch)

            except Exception as e:
                logger.error("Batch indexing failed: %s", e)
                total_failed += len(batch)

        response = IndexResponse(
            indexed_count=total_indexed,
            failed_count=total_failed,
            status="success" if total_failed == 0 else "partial",
            message=f"Indexed {total_indexed} documents"
            + (f", {total_failed} failed" if total_failed > 0 else ""),
        )
        _log_index_request(started_at, 200, total_indexed)
        return response

    except Exception as exc:
        logger.exception("Index request failed: %s", exc)
        _log_index_request(started_at, 500, 0)
        return _error_response(500, str(exc))

