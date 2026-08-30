"""
Property-Based Tests for BM25 and TF-IDF Indexing (IndexingEngine)

This module tests the IndexingEngine class from src/indexer.py using
property-based testing with Hypothesis, complemented by targeted unit tests.

Properties covered:
- Property 7: BM25 index covers all documents
- Property 8: TF-IDF matrix dimensions correct
- Property 9: TF-IDF vocabulary is unique

**Validates: Requirements 2.1-2.6**
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

# Import the modules under test from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from preprocess import TextPreprocessor
from indexer import IndexingEngine


# ============================================================================
# HELPERS & STRATEGY DEFINITIONS (Smart Input Generators)
# ============================================================================

# A pool of distinct, non-stop-word terms that survive preprocessing
# (lowercase, punctuation removal, stop-word removal, lemmatization).
SHARED_WORDS = [
    "machine", "learning", "algorithm", "neural", "network", "vector",
    "database", "retrieval", "document", "semantic", "lexical", "search",
    "python", "language", "processing", "embedding", "cluster", "gradient",
    "transformer", "kernel", "feature", "matrix", "corpus", "ranking",
]


def _build_documents(specs):
    """
    Build a list of document dicts from a list of (unique_index, shared_words)
    specs.

    Each document is guaranteed to contain a globally-unique token, ensuring:
      - preprocessing yields at least one token per document (scorable BM25),
      - the TF-IDF vocabulary is non-empty and survives max_df pruning
        (a unique term appears in exactly one document).
    """
    documents = []
    for i, (unique_idx, shared) in enumerate(specs):
        unique_word = f"uniqueterm{unique_idx}x{i}"
        words = [unique_word] + list(shared)
        content = " ".join(words)
        documents.append(
            {
                "doc_id": f"doc_{i}",
                "content": content,
                "metadata": {"pos": i},
            }
        )
    return documents


# A single document spec: a unique-word seed plus a random subset of shared words.
_doc_spec = st.tuples(
    st.integers(min_value=0, max_value=1_000_000),
    st.lists(st.sampled_from(SHARED_WORDS), min_size=0, max_size=6),
)

# Document sets valid for BM25 (n >= 1) and TF-IDF (n >= 2, avoids the
# max_df=0.95 vs min_df=1 degenerate case for a single document).
documents_bm25 = st.lists(_doc_spec, min_size=1, max_size=15).map(_build_documents)
documents_tfidf = st.lists(_doc_spec, min_size=2, max_size=15).map(_build_documents)


def _make_engine():
    """Construct a fresh IndexingEngine with default config and preprocessor."""
    return IndexingEngine(Config(), TextPreprocessor())


# ============================================================================
# PROPERTY 7: BM25 INDEX COVERS ALL DOCUMENTS
# ============================================================================

class TestProperty7Bm25IndexCoversAllDocuments:
    """
    Property 7: BM25 index covers all documents

    For any set of documents, after building a BM25Okapi index, the index
    SHALL be scorable for every document in the input set.

    **Validates: Requirement 2.1**
    """

    @given(documents=documents_bm25)
    @settings(max_examples=20, deadline=None)
    def test_bm25_scores_every_document(self, documents):
        """
        Given any set of documents,
        When a BM25 index is built and scored with a query,
        Then the score array covers exactly every input document.

        **Validates: Requirement 2.1**
        """
        engine = _make_engine()
        assert engine.build_bm25_index(documents) is True

        bm25 = engine.get_bm25_index()
        assert bm25 is not None

        # Score against a query drawn from the shared vocabulary; the returned
        # score vector must have one entry per indexed document.
        scores = bm25.get_scores(["machine", "search"])
        assert len(scores) == len(documents)

        # Document ID mapping must also cover every input document, in order.
        doc_ids = engine.get_document_ids()
        assert doc_ids == [d["doc_id"] for d in documents]

    @given(documents=documents_bm25)
    @settings(max_examples=20, deadline=None)
    def test_bm25_scorable_for_each_document_content(self, documents):
        """
        Given any set of documents,
        When the index is queried with each document's own tokens,
        Then a finite score is produced for every document.

        **Validates: Requirement 2.1**
        """
        engine = _make_engine()
        assert engine.build_bm25_index(documents) is True
        bm25 = engine.get_bm25_index()

        preprocessor = engine.preprocessor
        for doc in documents:
            query_tokens = preprocessor.preprocess(doc["content"])
            scores = bm25.get_scores(query_tokens)
            assert len(scores) == len(documents)
            assert np.all(np.isfinite(scores))


# ============================================================================
# PROPERTY 8: TF-IDF MATRIX DIMENSIONS CORRECT
# ============================================================================

class TestProperty8TfidfMatrixDimensionsCorrect:
    """
    Property 8: TF-IDF matrix dimensions correct

    For any set of n documents, building a TF-IDF matrix with max_features=5000
    SHALL produce a matrix of shape (n, m) where m <= 5000.

    **Validates: Requirements 2.2, 2.4**
    """

    @given(documents=documents_tfidf)
    @settings(max_examples=20, deadline=None)
    def test_tfidf_matrix_shape(self, documents):
        """
        Given any set of n documents,
        When a TF-IDF matrix is built,
        Then the matrix shape is (n, m) with m <= max_features.

        **Validates: Requirements 2.2, 2.4**
        """
        engine = _make_engine()
        assert engine.build_tfidf_index(documents) is True

        matrix = engine.get_tfidf_matrix()
        assert matrix is not None

        n_docs, n_features = matrix.shape
        assert n_docs == len(documents)
        assert n_features <= engine.config.max_features_tfidf
        assert n_features >= 1

    @given(documents=documents_tfidf)
    @settings(max_examples=20, deadline=None)
    def test_tfidf_matrix_columns_match_vocabulary(self, documents):
        """
        Given any set of documents,
        When a TF-IDF matrix and vocabulary are built,
        Then the number of matrix columns equals the vocabulary size.

        **Validates: Requirements 2.2, 2.4, 2.6**
        """
        engine = _make_engine()
        assert engine.build_tfidf_index(documents) is True

        matrix = engine.get_tfidf_matrix()
        vocabulary = engine.get_tfidf_vocabulary()

        assert matrix.shape[1] == len(vocabulary)


# ============================================================================
# PROPERTY 9: TF-IDF VOCABULARY IS UNIQUE
# ============================================================================

class TestProperty9TfidfVocabularyIsUnique:
    """
    Property 9: TF-IDF vocabulary is unique

    For any TF-IDF index, the vocabulary (feature names) SHALL be a non-empty
    list of unique strings with no duplicates.

    **Validates: Requirement 2.6**
    """

    @given(documents=documents_tfidf)
    @settings(max_examples=20, deadline=None)
    def test_tfidf_vocabulary_is_unique_and_non_empty(self, documents):
        """
        Given any set of documents,
        When a TF-IDF index is built,
        Then the vocabulary is a non-empty list of unique strings.

        **Validates: Requirement 2.6**
        """
        engine = _make_engine()
        assert engine.build_tfidf_index(documents) is True

        vocabulary = engine.get_tfidf_vocabulary()

        # Non-empty list of strings
        assert isinstance(vocabulary, list)
        assert len(vocabulary) > 0
        assert all(isinstance(term, str) for term in vocabulary)

        # No duplicates
        assert len(vocabulary) == len(set(vocabulary))


# ============================================================================
# TARGETED UNIT TESTS (specific examples & edge cases)
# ============================================================================

SAMPLE_DOCS = [
    {"doc_id": "d1", "content": "the quick brown fox jumps over the lazy dog",
     "metadata": {"title": "one"}},
    {"doc_id": "d2", "content": "machine learning and artificial intelligence transform technology",
     "metadata": {"title": "two"}},
    {"doc_id": "d3", "content": "natural language processing is a subfield of artificial intelligence",
     "metadata": {"title": "three"}},
]


class TestBm25UnitCases:
    """Concrete unit tests for BM25 index building."""

    def test_build_bm25_returns_true_and_maps_ids(self):
        engine = _make_engine()
        assert engine.build_bm25_index(SAMPLE_DOCS) is True
        assert engine.get_document_ids() == ["d1", "d2", "d3"]

    def test_bm25_scores_length_matches_documents(self):
        engine = _make_engine()
        engine.build_bm25_index(SAMPLE_DOCS)
        scores = engine.get_bm25_index().get_scores(["intelligence"])
        assert len(scores) == 3

    def test_build_bm25_empty_list_returns_false(self):
        engine = _make_engine()
        assert engine.build_bm25_index([]) is False
        assert engine.get_bm25_index() is None

    def test_build_bm25_missing_fields_returns_false(self):
        engine = _make_engine()
        assert engine.build_bm25_index([{"doc_id": "x"}]) is False


class TestTfidfUnitCases:
    """Concrete unit tests for TF-IDF index building."""

    def test_build_tfidf_returns_true(self):
        engine = _make_engine()
        assert engine.build_tfidf_index(SAMPLE_DOCS) is True

    def test_tfidf_matrix_shape_matches_documents(self):
        engine = _make_engine()
        engine.build_tfidf_index(SAMPLE_DOCS)
        matrix = engine.get_tfidf_matrix()
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 5000

    def test_tfidf_vocabulary_unique(self):
        engine = _make_engine()
        engine.build_tfidf_index(SAMPLE_DOCS)
        vocab = engine.get_tfidf_vocabulary()
        assert len(vocab) > 0
        assert len(vocab) == len(set(vocab))

    def test_build_tfidf_empty_list_returns_false(self):
        engine = _make_engine()
        assert engine.build_tfidf_index([]) is False

    def test_tfidf_respects_max_features(self):
        engine = IndexingEngine(Config(max_features_tfidf=3), TextPreprocessor())
        engine.build_tfidf_index(SAMPLE_DOCS)
        assert engine.get_tfidf_matrix().shape[1] <= 3


class TestSaveLoadRoundTrip:
    """Round-trip persistence keeps index coverage intact."""

    def test_save_and_load_preserves_indices(self, tmp_path):
        engine = _make_engine()
        engine.build_bm25_index(SAMPLE_DOCS)
        engine.build_tfidf_index(SAMPLE_DOCS)

        save_path = str(tmp_path / "indices")
        assert engine.save_indices(save_path) is True

        loaded = _make_engine()
        assert loaded.load_indices(save_path) is True
        assert loaded.get_document_ids() == ["d1", "d2", "d3"]
        assert loaded.get_tfidf_matrix().shape[0] == 3
        assert len(loaded.get_bm25_index().get_scores(["intelligence"])) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
