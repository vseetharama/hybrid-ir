"""
Property-Based Tests for Search Algorithms (SearchEngine)

This module tests the SearchEngine class from src/search_engine.py using
property-based testing with Hypothesis, complemented by targeted unit tests.

Properties covered (task 9.1):
- Property 15: Query preprocessing matches document preprocessing
- Property 16: BM25 scores all documents
- Property 17: BM25 results sorted descending
- Property 18: BM25 returns top-K documents
- Property 19: BM25 result format correct
- Property 20: TF-IDF scores in valid range
- Property 21: Query embeddings have dimension 384
- Property 22: Query embeddings are unit length
- Property 23: Semantic search scores in [0, 1]
- Property 24: Semantic results sorted descending
- Property 25: Semantic search returns top-K
- Property 26: Semantic result format correct

**Validates: Requirements 4.1-4.7, 5.1-5.7**

The sentence-transformers model ('all-MiniLM-L6-v2') is expensive to load, so it
is loaded exactly once per test session and shared across all examples via a
module-level cache. Each example wires the shared model into a fresh
SemanticIndexer, keeping generated embeddings realistic (no mocking).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

# Import the modules under test. Insert both the project root (so that
# `src.*` imports inside semantic_indexer resolve) and the src directory
# (so bare `config`/`indexer` imports resolve).
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from preprocess import TextPreprocessor
from indexer import IndexingEngine
from semantic_indexer import SemanticIndexer
from search_engine import SearchEngine

EMBED_DIM = 384


# ============================================================================
# SHARED MODEL (loaded once per session, reused across all examples)
# ============================================================================

_SHARED_MODEL = None


def _shared_model():
    """Lazily load and cache the sentence-transformers model for the session."""
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        probe = SemanticIndexer(Config())
        assert probe.load_model() is True, "Failed to load all-MiniLM-L6-v2 model"
        _SHARED_MODEL = probe.model
    return _SHARED_MODEL


# ============================================================================
# STRATEGY DEFINITIONS (Smart Input Generators)
# ============================================================================

# A pool of distinct, non-stop-word terms that survive preprocessing
# (lowercase, punctuation removal, stop-word removal, lemmatization). These
# double as query terms guaranteed to be in-vocabulary for TF-IDF.
WORD_POOL = [
    "machine", "learning", "algorithm", "neural", "network", "vector",
    "database", "retrieval", "document", "semantic", "lexical", "search",
    "python", "language", "processing", "embedding", "cluster", "gradient",
    "transformer", "kernel", "feature", "matrix", "corpus", "ranking",
]


def _build_documents(specs):
    """
    Build document dicts from (unique_index, shared_words) specs.

    Each document carries a globally-unique token so preprocessing always
    yields at least one token per document (BM25 scorable) and the TF-IDF
    vocabulary survives the min_df/max_df pruning.
    """
    documents = []
    for i, (unique_idx, shared) in enumerate(specs):
        unique_word = f"uniqueterm{unique_idx}x{i}"
        words = [unique_word] + list(shared)
        documents.append(
            {
                "doc_id": f"doc_{i}",
                "content": " ".join(words),
                "metadata": {"pos": i},
            }
        )
    return documents


_doc_spec = st.tuples(
    st.integers(min_value=0, max_value=1_000_000),
    st.lists(st.sampled_from(WORD_POOL), min_size=1, max_size=6),
)

# Document corpora valid for BM25/semantic (n >= 1) and TF-IDF (n >= 2 avoids
# the degenerate max_df=0.95 vs min_df=1 single-document case).
documents_1_plus = st.lists(_doc_spec, min_size=1, max_size=10).map(_build_documents)
documents_2_plus = st.lists(_doc_spec, min_size=2, max_size=10).map(_build_documents)

# Queries always contain at least one in-vocabulary term so results are
# non-trivial for lexical methods.
queries = st.lists(st.sampled_from(WORD_POOL), min_size=1, max_size=4).map(" ".join)

top_ks = st.integers(min_value=1, max_value=15)


# ============================================================================
# ENGINE BUILDERS
# ============================================================================

def _build_lexical_engine(documents):
    """Build an IndexingEngine with BM25 + TF-IDF indices over documents."""
    indexer = IndexingEngine(Config(), TextPreprocessor())
    indexer.build_bm25_index(documents)
    indexer.build_tfidf_index(documents)
    return indexer


def _build_semantic_indexer(documents):
    """Build a SemanticIndexer (using the shared model) over documents."""
    semantic = SemanticIndexer(Config())
    semantic.model = _shared_model()
    semantic.build_faiss_index(documents)
    return semantic


def _build_search_engine(documents, with_semantic=True):
    """Construct a SearchEngine wired to freshly built indices."""
    preprocessor = TextPreprocessor()
    indexer = IndexingEngine(Config(), preprocessor)
    indexer.build_bm25_index(documents)
    indexer.build_tfidf_index(documents)
    semantic = _build_semantic_indexer(documents) if with_semantic else None
    return SearchEngine(Config(), preprocessor, indexer, semantic, db_manager=None)


# ============================================================================
# PROPERTY 15: QUERY PREPROCESSING MATCHES DOCUMENT PREPROCESSING
# ============================================================================

class TestProperty15QueryPreprocessingMatchesDocument:
    """
    Property 15: Query preprocessing matches document preprocessing

    For any identical text processed as both a query and a document, the
    preprocessing pipeline SHALL produce identical token lists.

    **Validates: Requirements 4.1**
    """

    @given(text=st.lists(st.sampled_from(WORD_POOL), min_size=1, max_size=8).map(" ".join))
    @settings(max_examples=20, deadline=None)
    def test_query_and_document_preprocessing_identical(self, text):
        """
        Given identical text,
        When preprocessed as a query and as a document (same pipeline),
        Then the token lists are identical.

        **Validates: Requirement 4.1**
        """
        engine = _build_search_engine(_build_documents([(1, ["machine"])]),
                                      with_semantic=False)
        # The engine's query preprocessing uses the same preprocessor instance
        # that the indexer used for documents.
        query_tokens = engine._preprocess_query(text)
        document_tokens = engine.preprocessor.preprocess(text)
        assert query_tokens == document_tokens


# ============================================================================
# PROPERTY 16: BM25 SCORES ALL DOCUMENTS
# ============================================================================

class TestProperty16Bm25ScoresAllDocuments:
    """
    Property 16: BM25 scores all documents

    For any query and document corpus, BM25 search SHALL return a score for
    every document in the corpus (when top_k covers the whole corpus).

    **Validates: Requirements 4.2**
    """

    @given(documents=documents_1_plus, query=queries)
    @settings(max_examples=20, deadline=None)
    def test_bm25_returns_score_for_every_document(self, documents, query):
        """
        Given any corpus and query,
        When BM25 search is run with top_k = corpus size,
        Then exactly one scored result is returned per document.

        **Validates: Requirement 4.2**
        """
        engine = _build_search_engine(documents, with_semantic=False)
        results = engine.search_bm25(query, top_k=len(documents))

        assert len(results) == len(documents)
        returned_ids = {doc_id for doc_id, _ in results}
        expected_ids = {d["doc_id"] for d in documents}
        assert returned_ids == expected_ids


# ============================================================================
# PROPERTY 17: BM25 RESULTS SORTED DESCENDING
# ============================================================================

class TestProperty17Bm25ResultsSortedDescending:
    """
    Property 17: BM25 results sorted descending

    For any search query, BM25 results SHALL be sorted by score in descending
    order (scores[i] >= scores[i+1] for all i).

    **Validates: Requirements 4.3, 4.4**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_bm25_scores_non_increasing(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When BM25 search is run,
        Then result scores are in non-increasing order.

        **Validates: Requirements 4.3, 4.4**
        """
        engine = _build_search_engine(documents, with_semantic=False)
        results = engine.search_bm25(query, top_k=top_k)

        scores = [score for _, score in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ============================================================================
# PROPERTY 18: BM25 RETURNS TOP-K DOCUMENTS
# ============================================================================

class TestProperty18Bm25ReturnsTopK:
    """
    Property 18: BM25 returns top-K documents

    For any top_k parameter, BM25 search SHALL return at most top_k documents,
    sorted by score descending.

    **Validates: Requirements 4.4**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_bm25_returns_at_most_top_k(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When BM25 search is run,
        Then at most top_k (and at most corpus size) results are returned.

        **Validates: Requirement 4.4**
        """
        engine = _build_search_engine(documents, with_semantic=False)
        results = engine.search_bm25(query, top_k=top_k)

        assert len(results) <= top_k
        assert len(results) <= len(documents)


# ============================================================================
# PROPERTY 19: BM25 RESULT FORMAT CORRECT
# ============================================================================

class TestProperty19Bm25ResultFormatCorrect:
    """
    Property 19: BM25 result format correct

    For any BM25 search result, each item SHALL be a tuple of
    (document_id: str, bm25_score: float).

    **Validates: Requirements 4.6**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_bm25_result_items_are_str_float_tuples(self, documents, query, top_k):
        """
        Given any BM25 result,
        When items are inspected,
        Then each is a (str, float) tuple.

        **Validates: Requirement 4.6**
        """
        engine = _build_search_engine(documents, with_semantic=False)
        results = engine.search_bm25(query, top_k=top_k)

        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            doc_id, score = item
            assert isinstance(doc_id, str)
            assert isinstance(score, float)


# ============================================================================
# PROPERTY 20: TF-IDF SCORES IN VALID RANGE
# ============================================================================

class TestProperty20TfidfScoresInValidRange:
    """
    Property 20: TF-IDF scores in valid range

    For any TF-IDF search result, all scores SHALL be in the range [0, 1]
    (cosine similarity of non-negative TF-IDF vectors).

    **Validates: Requirements 4.7**
    """

    @given(documents=documents_2_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_tfidf_scores_within_zero_one(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When TF-IDF search is run,
        Then every score lies within [0, 1] and results are sorted descending.

        **Validates: Requirement 4.7**
        """
        engine = _build_search_engine(documents, with_semantic=False)
        results = engine.search_tfidf(query, top_k=top_k)

        scores = [score for _, score in results]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        assert len(results) <= top_k


# ============================================================================
# PROPERTY 21: QUERY EMBEDDINGS HAVE DIMENSION 384
# ============================================================================

class TestProperty21QueryEmbeddingsHaveDimension384:
    """
    Property 21: Query embeddings have dimension 384

    For any query text, semantic encoding SHALL produce a single embedding of
    dimension 384.

    **Validates: Requirements 5.1**
    """

    @given(query=queries)
    @settings(max_examples=20, deadline=None)
    def test_query_embedding_is_384_dimensional(self, query):
        """
        Given any query text,
        When it is encoded,
        Then the embedding is a single 384-dimensional vector.

        **Validates: Requirement 5.1**
        """
        engine = _build_search_engine(_build_documents([(1, ["machine"])]))
        embedding = engine.encode_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (EMBED_DIM,)


# ============================================================================
# PROPERTY 22: QUERY EMBEDDINGS ARE UNIT LENGTH
# ============================================================================

class TestProperty22QueryEmbeddingsAreUnitLength:
    """
    Property 22: Query embeddings are unit length

    For any query embedding, the L2 norm SHALL equal 1.0 (within a
    floating-point tolerance of ±0.001).

    **Validates: Requirements 5.2**
    """

    @given(query=queries)
    @settings(max_examples=20, deadline=None)
    def test_query_embedding_is_unit_length(self, query):
        """
        Given any query text,
        When it is encoded,
        Then its L2 norm equals 1.0 within ±0.001.

        **Validates: Requirement 5.2**
        """
        engine = _build_search_engine(_build_documents([(1, ["machine"])]))
        embedding = engine.encode_query(query)

        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) <= 1e-3


# ============================================================================
# PROPERTY 23: SEMANTIC SEARCH SCORES IN [0, 1]
# ============================================================================

class TestProperty23SemanticSearchScoresInZeroOne:
    """
    Property 23: Semantic search scores in [0, 1]

    For any semantic search result, all cosine similarity scores SHALL be in
    the range [0, 1].

    **Validates: Requirements 5.3, 5.7**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_semantic_scores_within_zero_one(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When semantic search is run,
        Then every similarity score lies within [0, 1].

        **Validates: Requirements 5.3, 5.7**
        """
        engine = _build_search_engine(documents)
        results = engine.search_semantic(query, top_k=top_k)

        scores = [score for _, score in results]
        assert all(0.0 <= s <= 1.0 for s in scores)


# ============================================================================
# PROPERTY 24: SEMANTIC RESULTS SORTED DESCENDING
# ============================================================================

class TestProperty24SemanticResultsSortedDescending:
    """
    Property 24: Semantic results sorted descending

    For any semantic search query, results SHALL be sorted by similarity score
    in descending order.

    **Validates: Requirements 5.4, 5.5**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_semantic_scores_non_increasing(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When semantic search is run,
        Then result scores are in non-increasing order.

        **Validates: Requirements 5.4, 5.5**
        """
        engine = _build_search_engine(documents)
        results = engine.search_semantic(query, top_k=top_k)

        scores = [score for _, score in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ============================================================================
# PROPERTY 25: SEMANTIC SEARCH RETURNS TOP-K
# ============================================================================

class TestProperty25SemanticSearchReturnsTopK:
    """
    Property 25: Semantic search returns top-K

    For any top_k parameter, semantic search SHALL return at most top_k
    documents, sorted by score descending.

    **Validates: Requirements 5.5**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_semantic_returns_at_most_top_k(self, documents, query, top_k):
        """
        Given any corpus, query, and top_k,
        When semantic search is run,
        Then at most top_k (and at most corpus size) results are returned.

        **Validates: Requirement 5.5**
        """
        engine = _build_search_engine(documents)
        results = engine.search_semantic(query, top_k=top_k)

        assert len(results) <= top_k
        assert len(results) <= len(documents)


# ============================================================================
# PROPERTY 26: SEMANTIC RESULT FORMAT CORRECT
# ============================================================================

class TestProperty26SemanticResultFormatCorrect:
    """
    Property 26: Semantic result format correct

    For any semantic search result, each item SHALL be a tuple of
    (document_id: str, cosine_similarity_score: float).

    **Validates: Requirements 5.6**
    """

    @given(documents=documents_1_plus, query=queries, top_k=top_ks)
    @settings(max_examples=20, deadline=None)
    def test_semantic_result_items_are_str_float_tuples(self, documents, query, top_k):
        """
        Given any semantic result,
        When items are inspected,
        Then each is a (str, float) tuple.

        **Validates: Requirement 5.6**
        """
        engine = _build_search_engine(documents)
        results = engine.search_semantic(query, top_k=top_k)

        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            doc_id, score = item
            assert isinstance(doc_id, str)
            assert isinstance(score, float)


# ============================================================================
# TARGETED UNIT TESTS (specific examples & edge cases)
# ============================================================================

SAMPLE_DOCS = [
    {"doc_id": "d1", "content": "machine learning algorithm basics",
     "metadata": {"title": "one"}},
    {"doc_id": "d2", "content": "deep neural network training guide",
     "metadata": {"title": "two"}},
    {"doc_id": "d3", "content": "natural language processing and semantic search",
     "metadata": {"title": "three"}},
]


class TestBm25UnitCases:
    """Concrete unit tests for BM25 search."""

    def test_empty_query_returns_empty(self):
        engine = _build_search_engine(SAMPLE_DOCS, with_semantic=False)
        assert engine.search_bm25("", top_k=5) == []
        assert engine.search_bm25("   ", top_k=5) == []

    def test_results_match_top_k(self):
        engine = _build_search_engine(SAMPLE_DOCS, with_semantic=False)
        results = engine.search_bm25("machine learning", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(d, str) and isinstance(s, float) for d, s in results)


class TestTfidfUnitCases:
    """Concrete unit tests for TF-IDF search."""

    def test_scores_in_range(self):
        engine = _build_search_engine(SAMPLE_DOCS, with_semantic=False)
        results = engine.search_tfidf("neural network", top_k=3)
        assert all(0.0 <= s <= 1.0 for _, s in results)

    def test_empty_query_returns_empty(self):
        engine = _build_search_engine(SAMPLE_DOCS, with_semantic=False)
        assert engine.search_tfidf("", top_k=3) == []


class TestSemanticUnitCases:
    """Concrete unit tests for semantic search."""

    def test_query_embedding_shape_and_norm(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        emb = engine.encode_query("semantic search engine")
        assert emb.shape == (EMBED_DIM,)
        assert abs(float(np.linalg.norm(emb)) - 1.0) <= 1e-3

    def test_scores_in_range_and_sorted(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        results = engine.search_semantic("language processing", top_k=3)
        scores = [s for _, s in results]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_empty_query_returns_empty(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        assert engine.search_semantic("", top_k=3) == []


class TestUnifiedSearchUnitCases:
    """Concrete unit tests for the unified search interface."""

    def test_search_returns_expected_keys(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        response = engine.search("machine learning", top_k=3, fusion_method="bm25")
        assert set(response.keys()) == {"results", "execution_time_ms", "query_processed"}
        assert isinstance(response["execution_time_ms"], float)
        assert isinstance(response["query_processed"], str)

    def test_search_empty_query_returns_empty_results(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        response = engine.search("", top_k=3, fusion_method="rrf")
        assert response["results"] == []

    def test_search_invalid_method_raises(self):
        engine = _build_search_engine(SAMPLE_DOCS)
        with pytest.raises(ValueError):
            engine.search("machine", top_k=3, fusion_method="bogus")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
