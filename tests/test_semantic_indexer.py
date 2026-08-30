"""
Property-Based Tests for Semantic Indexing (SemanticIndexer)

This module tests the SemanticIndexer class from src/semantic_indexer.py using
property-based testing with Hypothesis, complemented by targeted unit tests.

Properties covered:
- Property 10: Embeddings have correct dimension (384)
- Property 11: One embedding per document
- Property 12: Embeddings are normalized (L2 norm = 1.0)
- Property 13: FAISS index covers all embeddings
- Property 14: FAISS append extends index
- Property 48: FAISS corruption detection works

**Validates: Requirements 3.1-3.7, 13.6**

The sentence-transformers model ('all-MiniLM-L6-v2') is expensive to load, so it
is loaded exactly once per test session and shared across all examples via a
module-scoped fixture. Each example attaches the shared model to a fresh
SemanticIndexer, keeping generated embeddings realistic (no mocking).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

# Import the modules under test from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.semantic_indexer import SemanticIndexer

EMBED_DIM = 384

# ============================================================================
# SHARED MODEL FIXTURE (loaded once per session, reused across all examples)
# ============================================================================

# A single loaded model shared across every SemanticIndexer created in this
# module. Loading 'all-MiniLM-L6-v2' takes several seconds, so we do it once.
_SHARED_MODEL = None


def _shared_model():
    """Lazily load and cache the sentence-transformers model for the session."""
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        probe = SemanticIndexer(Config())
        assert probe.load_model() is True, "Failed to load all-MiniLM-L6-v2 model"
        _SHARED_MODEL = probe.model
    return _SHARED_MODEL


def _make_indexer():
    """Construct a fresh SemanticIndexer wired to the shared (already loaded) model."""
    indexer = SemanticIndexer(Config())
    indexer.model = _shared_model()
    return indexer


# ============================================================================
# STRATEGY DEFINITIONS (Smart Input Generators)
# ============================================================================

# A pool of natural, non-empty words so generated content is realistic text
# that always survives the "content must be truthy" guard in the indexer.
WORD_POOL = [
    "machine", "learning", "algorithm", "neural", "network", "vector",
    "database", "retrieval", "document", "semantic", "lexical", "search",
    "python", "language", "processing", "embedding", "cluster", "gradient",
    "transformer", "kernel", "feature", "matrix", "corpus", "ranking",
    "knowledge", "graph", "inference", "training", "dataset", "similarity",
]

# A single document's content: 1-8 words drawn from the pool, always non-empty.
_content = st.lists(st.sampled_from(WORD_POOL), min_size=1, max_size=8).map(" ".join)


def _documents(min_size, max_size):
    """
    Build a strategy producing lists of document dicts with unique doc_ids.

    Each document has a globally-unique doc_id (within the list) and non-empty
    content, satisfying build_faiss_index / add_embeddings input requirements.
    """
    return st.lists(_content, min_size=min_size, max_size=max_size).map(
        lambda contents: [
            {"doc_id": f"doc_{i}", "content": c, "metadata": {"pos": i}}
            for i, c in enumerate(contents)
        ]
    )


documents_1_plus = _documents(1, 12)
documents_1_plus_b = _documents(1, 8)


# ============================================================================
# PROPERTY 10: EMBEDDINGS HAVE CORRECT DIMENSION
# ============================================================================

class TestProperty10EmbeddingsHaveCorrectDimension:
    """
    Property 10: Embeddings have correct dimension

    For any set of documents, semantic indexing with 'all-MiniLM-L6-v2' SHALL
    produce embeddings of shape (n_documents, 384).

    **Validates: Requirements 3.1, 3.7**
    """

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_embeddings_shape_is_n_by_384(self, documents):
        """
        Given any set of n documents,
        When embeddings are generated,
        Then the resulting array has shape (n, 384).

        **Validates: Requirements 3.1, 3.7**
        """
        indexer = _make_indexer()
        contents = [d["content"] for d in documents]

        embeddings = indexer.generate_embeddings(contents)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape == (len(documents), EMBED_DIM)

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_get_embeddings_shape_after_build(self, documents):
        """
        Given a built FAISS index,
        When embeddings are retrieved via get_embeddings(),
        Then the array has shape (n, 384).

        **Validates: Requirements 3.1, 3.7**
        """
        indexer = _make_indexer()
        assert indexer.build_faiss_index(documents) is True

        embeddings = indexer.get_embeddings()
        assert embeddings.shape == (len(documents), EMBED_DIM)


# ============================================================================
# PROPERTY 11: ONE EMBEDDING PER DOCUMENT
# ============================================================================

class TestProperty11OneEmbeddingPerDocument:
    """
    Property 11: One embedding per document

    For any set of n documents, the semantic indexer SHALL generate exactly n
    embeddings (one-to-one mapping).

    **Validates: Requirements 3.2**
    """

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_exactly_one_embedding_per_document(self, documents):
        """
        Given any set of n documents,
        When the FAISS index is built,
        Then there are exactly n embeddings, n index entries, and n doc_ids.

        **Validates: Requirements 3.2**
        """
        indexer = _make_indexer()
        assert indexer.build_faiss_index(documents) is True

        n = len(documents)
        assert indexer.get_embeddings().shape[0] == n
        assert indexer.get_faiss_index().ntotal == n
        assert len(indexer.document_ids) == n
        # One-to-one mapping preserved in order.
        assert indexer.document_ids == [d["doc_id"] for d in documents]


# ============================================================================
# PROPERTY 12: EMBEDDINGS ARE NORMALIZED
# ============================================================================

class TestProperty12EmbeddingsAreNormalized:
    """
    Property 12: Embeddings are normalized

    For any generated embedding, the L2 norm SHALL equal 1.0 (within a
    floating-point tolerance of ±0.001), indicating unit-length normalization.

    **Validates: Requirements 3.3**
    """

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_all_embeddings_have_unit_l2_norm(self, documents):
        """
        Given any set of documents,
        When embeddings are generated,
        Then every embedding has an L2 norm of 1.0 within ±0.001.

        **Validates: Requirements 3.3**
        """
        indexer = _make_indexer()
        contents = [d["content"] for d in documents]

        embeddings = indexer.generate_embeddings(contents)
        norms = np.linalg.norm(embeddings, axis=1)

        assert np.allclose(norms, 1.0, atol=1e-3)


# ============================================================================
# PROPERTY 13: FAISS INDEX COVERS ALL EMBEDDINGS
# ============================================================================

class TestProperty13FaissIndexCoversAllEmbeddings:
    """
    Property 13: FAISS index covers all embeddings

    For any set of embeddings, after building a FAISS IndexFlatIP, the index
    SHALL store all embeddings (ntotal == n_embeddings) and support similarity
    search.

    **Validates: Requirements 3.4, 3.5**
    """

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_index_stores_all_and_is_searchable(self, documents):
        """
        Given any set of documents,
        When the FAISS index is built,
        Then ntotal equals the number of embeddings and search returns results.

        **Validates: Requirements 3.4, 3.5**
        """
        indexer = _make_indexer()
        assert indexer.build_faiss_index(documents) is True

        n = len(documents)
        index = indexer.get_faiss_index()
        assert index is not None
        assert index.ntotal == n

        # The index must support similarity search over the stored embeddings.
        query = indexer.get_embeddings()[:1].astype(np.float32)
        k = min(n, 5)
        distances, indices = index.search(query, k)

        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        # Returned neighbour ids are valid positions within the index.
        assert np.all(indices[0] >= 0)
        assert np.all(indices[0] < n)


# ============================================================================
# PROPERTY 14: FAISS APPEND EXTENDS INDEX
# ============================================================================

class TestProperty14FaissAppendExtendsIndex:
    """
    Property 14: FAISS append extends index

    For any existing FAISS index with n embeddings, appending m new embeddings
    SHALL result in an index containing n+m total embeddings, with all
    embeddings remaining searchable.

    **Validates: Requirements 3.6**
    """

    @given(initial=documents_1_plus_b, extra=documents_1_plus_b)
    @settings(max_examples=20, deadline=None)
    def test_append_yields_n_plus_m_and_searchable(self, initial, extra):
        """
        Given an index built from n documents,
        When m more documents are appended,
        Then the index has n+m entries and all remain searchable.

        **Validates: Requirements 3.6**
        """
        indexer = _make_indexer()
        assert indexer.build_faiss_index(initial) is True
        n = len(initial)
        assert indexer.get_faiss_index().ntotal == n

        # Give appended documents distinct ids to avoid collisions.
        extra_docs = [
            {"doc_id": f"extra_{i}", "content": d["content"]}
            for i, d in enumerate(extra)
        ]
        m = len(extra_docs)

        assert indexer.add_embeddings(extra_docs) is True

        index = indexer.get_faiss_index()
        assert index.ntotal == n + m
        assert indexer.get_embeddings().shape[0] == n + m
        assert len(indexer.document_ids) == n + m

        # Every stored embedding (including appended ones) is searchable.
        all_embeddings = indexer.get_embeddings().astype(np.float32)
        distances, indices = index.search(all_embeddings, 1)
        assert distances.shape == (n + m, 1)
        assert np.all(indices[:, 0] >= 0)
        assert np.all(indices[:, 0] < n + m)


# ============================================================================
# PROPERTY 48: FAISS CORRUPTION DETECTION WORKS
# ============================================================================

class TestProperty48FaissCorruptionDetectionWorks:
    """
    Property 48: FAISS corruption detection works

    For any corrupted FAISS index (missing/invalid data), the system SHALL
    detect the corruption on validation, rebuild the index from saved
    embeddings, and successfully perform subsequent searches.

    **Validates: Requirement 13.6**
    """

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_corruption_is_detected_and_rebuilt(self, documents):
        """
        Given a valid index whose FAISS structure is then corrupted
        (size no longer matches the saved embeddings),
        When validate_index() runs,
        Then corruption is detected, the index is rebuilt from embeddings,
        and subsequent search works with the correct size.

        **Validates: Requirement 13.6**
        """
        import faiss

        indexer = _make_indexer()
        assert indexer.build_faiss_index(documents) is True
        n = len(documents)

        # Sanity: a healthy index validates without change.
        assert indexer.validate_index() is True
        assert indexer.get_faiss_index().ntotal == n

        # Corrupt the FAISS index: replace it with an empty index so its size
        # no longer matches the saved embeddings (simulating on-disk corruption
        # where the vectors are lost but embeddings.npy survives).
        indexer.faiss_index = faiss.IndexFlatIP(EMBED_DIM)
        assert indexer.get_faiss_index().ntotal == 0

        # Validation must detect the mismatch and rebuild from embeddings.
        assert indexer.validate_index() is True
        assert indexer.get_faiss_index().ntotal == n

        # Subsequent search works against the rebuilt index.
        query = indexer.get_embeddings()[:1].astype(np.float32)
        distances, indices = indexer.get_faiss_index().search(query, 1)
        assert indices.shape == (1, 1)
        assert 0 <= indices[0, 0] < n

    @given(documents=documents_1_plus)
    @settings(max_examples=20, deadline=None)
    def test_doc_id_mismatch_is_detected_and_rebuilt(self, documents):
        """
        Given a valid index whose doc_id mapping is corrupted,
        When validate_index() runs,
        Then the corruption is detected and the index is rebuilt successfully.

        **Validates: Requirement 13.6**
        """
        indexer = _make_indexer()
        assert indexer.build_faiss_index(documents) is True
        n = len(documents)

        # Corrupt doc_id mapping (drop the tail) to break the count invariant.
        indexer.document_ids = indexer.document_ids[:-1] if n > 0 else []

        assert indexer.validate_index() is True
        assert indexer.get_faiss_index().ntotal == n


# ============================================================================
# TARGETED UNIT TESTS (specific examples & edge cases)
# ============================================================================

SAMPLE_DOCS = [
    {"doc_id": "d1", "content": "machine learning basics", "metadata": {"title": "one"}},
    {"doc_id": "d2", "content": "deep neural network guide", "metadata": {"title": "two"}},
    {"doc_id": "d3", "content": "natural language processing overview", "metadata": {"title": "three"}},
]


class TestGenerateEmbeddingsUnit:
    """Concrete unit tests for embedding generation."""

    def test_shape_matches_input_count(self):
        indexer = _make_indexer()
        emb = indexer.generate_embeddings(["hello world", "machine learning"])
        assert emb.shape == (2, EMBED_DIM)

    def test_embeddings_are_unit_normalized(self):
        indexer = _make_indexer()
        emb = indexer.generate_embeddings(["semantic search engine"])
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)

    def test_generate_without_model_raises(self):
        indexer = SemanticIndexer(Config())  # model not loaded
        with pytest.raises(RuntimeError):
            indexer.generate_embeddings(["text"])


class TestBuildFaissIndexUnit:
    """Concrete unit tests for FAISS index building."""

    def test_build_returns_true_and_maps_ids(self):
        indexer = _make_indexer()
        assert indexer.build_faiss_index(SAMPLE_DOCS) is True
        assert indexer.get_faiss_index().ntotal == 3
        assert indexer.document_ids == ["d1", "d2", "d3"]

    def test_build_empty_returns_false(self):
        indexer = _make_indexer()
        assert indexer.build_faiss_index([]) is False
        assert indexer.get_faiss_index() is None

    def test_build_missing_fields_returns_false(self):
        indexer = _make_indexer()
        assert indexer.build_faiss_index([{"doc_id": "x"}]) is False


class TestAddEmbeddingsUnit:
    """Concrete unit tests for appending embeddings."""

    def test_append_extends_index(self):
        indexer = _make_indexer()
        indexer.build_faiss_index(SAMPLE_DOCS[:1])
        assert indexer.get_faiss_index().ntotal == 1

        assert indexer.add_embeddings(SAMPLE_DOCS[1:]) is True
        assert indexer.get_faiss_index().ntotal == 3
        assert indexer.document_ids == ["d1", "d2", "d3"]

    def test_append_without_existing_index_returns_false(self):
        indexer = _make_indexer()
        assert indexer.add_embeddings(SAMPLE_DOCS) is False


class TestValidateAndCorruptionUnit:
    """Concrete unit tests for index validation and corruption recovery."""

    def test_validate_healthy_index_true(self):
        indexer = _make_indexer()
        indexer.build_faiss_index(SAMPLE_DOCS)
        assert indexer.validate_index() is True

    def test_validate_uninitialized_index_false(self):
        indexer = _make_indexer()
        assert indexer.validate_index() is False

    def test_corrupted_size_triggers_rebuild(self):
        import faiss
        indexer = _make_indexer()
        indexer.build_faiss_index(SAMPLE_DOCS)

        indexer.faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # corrupt: empty
        assert indexer.validate_index() is True
        assert indexer.get_faiss_index().ntotal == 3

    def test_save_load_roundtrip_then_validate(self, tmp_path):
        indexer = _make_indexer()
        indexer.build_faiss_index(SAMPLE_DOCS)
        save_path = str(tmp_path / "indices")
        assert indexer.save_index(save_path) is True

        loaded = _make_indexer()
        assert loaded.load_index(save_path) is True
        assert loaded.get_faiss_index().ntotal == 3
        assert loaded.validate_index() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
