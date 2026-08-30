"""
Search Engine Module for Hybrid Information Retrieval System.

This module provides the SearchEngine class, a unified interface combining
lexical (BM25, TF-IDF) and semantic (dense embedding + FAISS) retrieval, and
fuses the three ranked lists using Reciprocal Rank Fusion (RRF).

The engine is wired with already-built indices:
- lexical_indices: an IndexingEngine exposing BM25 and TF-IDF indices
- semantic_indexer: a SemanticIndexer exposing a FAISS index and embeddings
- db_manager: a DatabaseManager for fetching document content/metadata

Reciprocal Rank Fusion formula (Requirement 6.1-6.7):

    RRF(d) = Σ_i ( 1 / (k + rank_i(d)) )

where:
- i ranges over the ranking lists (BM25, TF-IDF, semantic),
- rank_i(d) is the 1-indexed position of document d in list i,
- k is the RRF k-constant (default 60, configurable),
- if d is absent from list i, rank_i(d) = K + 1 (K = top_k parameter).

Requirements: 4.1-4.7, 5.1-5.7, 6.1-6.7
"""

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A ranking list is a sequence of (document_id, score) tuples.
RankedList = Sequence[Tuple[str, float]]

DEFAULT_RRF_K = 60


class SearchEngine:
    """Unified search interface combining lexical and semantic search + RRF."""

    def __init__(self, config, preprocessor, indexer, semantic_indexer,
                 database_manager=None, **legacy_dependencies):
        """Initialize the engine with the system's indexing dependencies.

        ``db_manager`` remains accepted as a compatibility alias for callers
        created before the dependency was named ``database_manager``.
        """
        if database_manager is None:
            database_manager = legacy_dependencies.pop("db_manager", None)
        if legacy_dependencies:
            unexpected = ", ".join(sorted(legacy_dependencies))
            raise TypeError(f"Unexpected SearchEngine dependencies: {unexpected}")

        self.config = config
        self.preprocessor = preprocessor
        # Keep the indexer itself so rebuilt indices are visible immediately.
        self.indexer = indexer
        self.lexical_indices = indexer  # Backwards-compatible public alias.
        self.semantic_indexer = semantic_indexer
        self.database_manager = database_manager
        self.db_manager = database_manager  # Backwards-compatible alias.
        logger.info("SearchEngine initialized")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rrf_k(self) -> int:
        """Return the configured RRF k-constant, defaulting to 60."""
        k = getattr(self.config, "rrf_k_constant", DEFAULT_RRF_K)
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = DEFAULT_RRF_K
        return k if k >= 1 else DEFAULT_RRF_K

    def _lexical_document_ids(self) -> List[str]:
        """Document IDs aligned with BM25 scores and TF-IDF matrix rows."""
        return list(self.indexer.get_document_ids())

    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess a query through the document indexing pipeline."""
        return self.preprocessor.preprocess(query or "")

    def encode_query(self, query: str) -> np.ndarray:
        """Create and L2-normalize one configured-dimension query embedding.

        Normalization is enforced here even when a semantic indexer/model also
        normalizes its output, keeping query-time behavior explicit and robust.

        Requirements: 5.1, 5.2
        """
        if self.semantic_indexer is None:
            raise RuntimeError("Semantic indexer is not configured")

        embeddings = np.asarray(
            self.semantic_indexer.generate_embeddings([query]),
            dtype=np.float32,
        )
        expected_dimension = int(getattr(self.config, "embedding_dimension", 384))
        if embeddings.shape != (1, expected_dimension):
            raise ValueError(
                "Query encoder returned shape "
                f"{embeddings.shape}; expected (1, {expected_dimension})"
            )

        embedding = embeddings[0]
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("Query encoder returned a non-normalizable embedding")
        return (embedding / norm).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # BM25 search
    # ------------------------------------------------------------------ #
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 lexical search.

        Returns a list of (document_id, bm25_score) sorted by score descending,
        truncated to top_k. Returns an empty list for empty queries.

        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
        """
        tokens = self._preprocess_query(query)
        if not tokens:
            return []

        bm25_index = self.lexical_indices.get_bm25_index()
        doc_ids = self._lexical_document_ids()
        if bm25_index is None or not doc_ids:
            return []

        scores = np.asarray(bm25_index.get_scores(tokens), dtype=float)
        results = list(zip(doc_ids, scores.tolist()))
        # Stable sort by score descending; zero scores fall to the bottom.
        results.sort(key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in results[: max(top_k, 0)]]

    # ------------------------------------------------------------------ #
    # TF-IDF search
    # ------------------------------------------------------------------ #
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        TF-IDF lexical search using cosine similarity.

        Returns a list of (document_id, tfidf_score) sorted by score descending,
        truncated to top_k. Returns an empty list for empty queries.

        Requirements: 4.7
        """
        tokens = self._preprocess_query(query)
        if not tokens:
            return []

        vectorizer = self.lexical_indices.get_tfidf_vectorizer()
        matrix = self.lexical_indices.get_tfidf_matrix()
        doc_ids = self._lexical_document_ids()
        if vectorizer is None or matrix is None or not doc_ids:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = vectorizer.transform([" ".join(tokens)])
        sims = cosine_similarity(query_vec, matrix)[0]
        results = list(zip(doc_ids, sims.tolist()))
        results.sort(key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in results[: max(top_k, 0)]]

    # ------------------------------------------------------------------ #
    # Semantic search
    # ------------------------------------------------------------------ #
    def search_semantic(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Semantic search using dense embeddings and FAISS inner product.

        Returns a list of (document_id, cosine_similarity_score) sorted by score
        descending, truncated to top_k. Scores are clamped to [0, 1].

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
        """
        if not query or not query.strip():
            return []

        faiss_index = self.semantic_indexer.get_faiss_index()
        doc_ids = list(getattr(self.semantic_indexer, "document_ids", []))
        if faiss_index is None or faiss_index.ntotal == 0 or not doc_ids:
            return []

        query_embedding = self.encode_query(query).reshape(1, -1).astype(np.float32)

        k = min(max(top_k, 0), faiss_index.ntotal)
        if k == 0:
            return []

        distances, indices = faiss_index.search(query_embedding, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(doc_ids):
                continue
            # Inner product of unit vectors is cosine in [-1, 1]; clamp to [0, 1].
            similarity = float(min(1.0, max(0.0, score)))
            results.append((doc_ids[idx], similarity))
        # FAISS already returns descending order, but sort defensively.
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------ #
    @staticmethod
    def reciprocal_rank_fusion(ranking_lists: Sequence[RankedList],
                               top_k: int,
                               k: int = DEFAULT_RRF_K) -> List[Tuple[str, float]]:
        """
        Pure Reciprocal Rank Fusion of one or more ranked lists.

        For each document appearing in at least one list, compute:

            RRF(d) = Σ_i ( 1 / (k + rank_i(d)) )

        where rank_i(d) is the 1-indexed position of d in list i, and documents
        absent from a list are treated as rank (top_k + 1).

        Args:
            ranking_lists: Sequence of ranked lists, each a sequence of
                (document_id, score) tuples ordered best-first.
            top_k: Number of fused results to return (also defines the
                missing-document rank as top_k + 1).
            k: RRF k-constant (default 60).

        Returns:
            List of (document_id, fused_rrf_score) sorted by score descending,
            truncated to top_k.

        Requirements: 6.1, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        if top_k <= 0:
            return []

        missing_rank = top_k + 1
        rank_maps: List[Dict[str, int]] = []
        ordered_ids: List[str] = []
        seen = set()

        for ranked in ranking_lists:
            rank_map: Dict[str, int] = {}
            for position, item in enumerate(ranked, start=1):
                doc_id = item[0]
                # Keep the best (earliest) rank if a doc_id repeats in a list.
                if doc_id not in rank_map:
                    rank_map[doc_id] = position
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered_ids.append(doc_id)
            rank_maps.append(rank_map)

        fused: List[Tuple[str, float]] = []
        for doc_id in ordered_ids:
            score = 0.0
            for rank_map in rank_maps:
                rank = rank_map.get(doc_id, missing_rank)
                denom = k + rank
                if denom != 0:
                    score += 1.0 / denom
            fused.append((doc_id, float(score)))

        fused.sort(key=lambda item: item[1], reverse=True)
        return fused[:top_k]

    def search_rrf(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion of BM25, TF-IDF, and semantic search.

        Gathers full ranked lists from each retrieval method and fuses them
        using the RRF formula with the configured k-constant (default 60).

        Returns a list of (document_id, fused_rrf_score) sorted by score
        descending, truncated to top_k. Returns an empty list for empty queries.

        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        if not query or not query.strip():
            return []

        # Retrieve comprehensive ranked lists (all documents) from each method
        # so fusion sees each document's true rank where available.
        lexical_count = len(self._lexical_document_ids())
        semantic_count = len(getattr(self.semantic_indexer, "document_ids", []))
        full_k = max(lexical_count, semantic_count, top_k)

        bm25_results = self.search_bm25(query, top_k=full_k)
        tfidf_results = self.search_tfidf(query, top_k=full_k)
        semantic_results = self.search_semantic(query, top_k=full_k)

        return self.reciprocal_rank_fusion(
            [bm25_results, tfidf_results, semantic_results],
            top_k=top_k,
            k=self._rrf_k(),
        )

    # ------------------------------------------------------------------ #
    # Unified search interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 10,
               fusion_method: str = "rrf") -> Dict:
        """
        Unified search interface.

        Routes to the requested retrieval method, records execution time, and
        returns a dict with combined and individual scores per result.

        Returns:
            {
                "results": List[Tuple[doc_id, combined_score, individual_scores]],
                "execution_time_ms": float,
                "query_processed": str,
            }
        """
        valid_methods = {"rrf", "bm25", "tfidf", "semantic"}
        method = (fusion_method or "rrf").lower()
        if method not in valid_methods:
            raise ValueError(
                f"fusion_method must be one of {sorted(valid_methods)}, got {fusion_method!r}"
            )

        start = time.perf_counter()
        processed_tokens = self._preprocess_query(query)
        query_processed = " ".join(processed_tokens)

        results: List[Tuple[str, float, Dict[str, float]]] = []

        if query and query.strip():
            if method == "bm25":
                ranked = self.search_bm25(query, top_k)
                results = [(d, s, {"bm25_score": s}) for d, s in ranked]
            elif method == "tfidf":
                ranked = self.search_tfidf(query, top_k)
                results = [(d, s, {"tfidf_score": s}) for d, s in ranked]
            elif method == "semantic":
                ranked = self.search_semantic(query, top_k)
                results = [(d, s, {"semantic_score": s}) for d, s in ranked]
            else:  # rrf
                ranked = self.search_rrf(query, top_k)
                # Attach individual scores for transparency.
                bm25_map = dict(self.search_bm25(query, top_k=max(top_k, 1)))
                tfidf_map = dict(self.search_tfidf(query, top_k=max(top_k, 1)))
                semantic_map = dict(self.search_semantic(query, top_k=max(top_k, 1)))
                for doc_id, fused_score in ranked:
                    results.append((
                        doc_id,
                        fused_score,
                        {
                            "bm25_score": float(bm25_map.get(doc_id, 0.0)),
                            "tfidf_score": float(tfidf_map.get(doc_id, 0.0)),
                            "semantic_score": float(semantic_map.get(doc_id, 0.0)),
                            "rrf_score": float(fused_score),
                        },
                    ))

        execution_time_ms = (time.perf_counter() - start) * 1000.0
        return {
            "results": results,
            "execution_time_ms": execution_time_ms,
            "query_processed": query_processed,
        }
