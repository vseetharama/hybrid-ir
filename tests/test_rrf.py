"""Property and unit tests for Reciprocal Rank Fusion.

Properties 27-30. **Validates: Requirements 6.1-6.7**
"""

import math
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from search_engine import SearchEngine


@st.composite
def ranking_cases(draw):
    """Create three valid rankings with overlap and possible missing entries."""
    count = draw(st.integers(min_value=1, max_value=12))
    document_ids = [f"doc_{index}" for index in range(count)]

    first_order = list(draw(st.permutations(document_ids)))
    rankings = [[(doc_id, float(count - rank))
                 for rank, doc_id in enumerate(first_order)]]
    for _ in range(2):
        order = list(draw(st.permutations(document_ids)))
        size = draw(st.integers(min_value=0, max_value=count))
        rankings.append([(doc_id, float(size - rank))
                         for rank, doc_id in enumerate(order[:size])])
    return document_ids, rankings


def expected_rrf(document_ids, rankings, top_k, k):
    """Independent implementation of the specified one-indexed RRF formula."""
    missing_rank = top_k + 1
    expected = {}
    for doc_id in document_ids:
        total = 0.0
        for ranking in rankings:
            rank_by_id = {item[0]: rank for rank, item in enumerate(ranking, 1)}
            total += 1.0 / (k + rank_by_id.get(doc_id, missing_rank))
        expected[doc_id] = total
    return expected


class TestProperty27RrfFormula:
    """Property 27: RRF applies the fusion formula correctly."""

    @given(case=ranking_cases(), k=st.integers(min_value=1, max_value=120))
    @settings(max_examples=100, deadline=None)
    def test_rrf_scores_match_formula(self, case, k):
        """**Validates: Requirements 6.1, 6.3, 6.4**"""
        document_ids, rankings = case
        top_k = len(document_ids)
        results = SearchEngine.reciprocal_rank_fusion(rankings, top_k, k)
        expected = expected_rrf(document_ids, rankings, top_k, k)

        assert {doc_id for doc_id, _ in results} == set(document_ids)
        for doc_id, score in results:
            assert math.isclose(score, expected[doc_id], rel_tol=1e-12)


class TestProperty28RrfDescendingOrder:
    """Property 28: RRF results are sorted in descending score order."""

    @given(case=ranking_cases(), top_k=st.integers(min_value=1, max_value=15))
    @settings(max_examples=100, deadline=None)
    def test_rrf_scores_are_non_increasing(self, case, top_k):
        """**Validates: Requirement 6.5**"""
        _, rankings = case
        results = SearchEngine.reciprocal_rank_fusion(rankings, top_k)
        scores = [score for _, score in results]
        assert all(left >= right for left, right in zip(scores, scores[1:]))


class TestProperty29RrfTopK:
    """Property 29: RRF returns no more than the requested top-K."""

    @given(case=ranking_cases(), top_k=st.integers(min_value=1, max_value=15))
    @settings(max_examples=100, deadline=None)
    def test_rrf_respects_top_k(self, case, top_k):
        """**Validates: Requirement 6.6**"""
        document_ids, rankings = case
        results = SearchEngine.reciprocal_rank_fusion(rankings, top_k)
        assert len(results) == min(top_k, len(document_ids))


class TestProperty30RrfTupleFormat:
    """Property 30: Every RRF result has the specified tuple format."""

    @given(case=ranking_cases(), top_k=st.integers(min_value=1, max_value=15))
    @settings(max_examples=100, deadline=None)
    def test_rrf_items_are_string_float_tuples(self, case, top_k):
        """**Validates: Requirement 6.7**"""
        _, rankings = case
        results = SearchEngine.reciprocal_rank_fusion(rankings, top_k)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


class TestRrfUnitCases:
    """Concrete examples for missing ranks, top-K, and boundary behavior."""

    def test_known_formula_includes_missing_rank(self):
        rankings = [
            [("a", 9.0), ("b", 8.0)],
            [("b", 7.0)],
            [("a", 6.0)],
        ]
        results = dict(SearchEngine.reciprocal_rank_fusion(rankings, top_k=2, k=60))
        assert math.isclose(results["a"], 2 / 61 + 1 / 63, rel_tol=1e-12)
        assert math.isclose(results["b"], 1 / 61 + 1 / 62 + 1 / 63,
                            rel_tol=1e-12)

    def test_zero_top_k_returns_empty(self):
        assert SearchEngine.reciprocal_rank_fusion([[('a', 1.0)]], 0) == []

    def test_duplicate_document_uses_earliest_rank(self):
        ranking = [[("a", 2.0), ("a", 1.0), ("b", 0.0)]]
        results = dict(SearchEngine.reciprocal_rank_fusion(ranking, top_k=2, k=60))
        assert math.isclose(results["a"], 1 / 61, rel_tol=1e-12)
        assert math.isclose(results["b"], 1 / 63, rel_tol=1e-12)
