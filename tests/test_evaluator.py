"""Property and unit tests for evaluation metrics.

Properties 39-41. **Validates: Requirements 10.1-10.3**
"""

import math
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluator import EvaluationMetrics


@st.composite
def evaluation_cases(draw):
    """Generate unique ranked results and an arbitrary relevance judgment set."""
    corpus_size = draw(st.integers(min_value=0, max_value=30))
    corpus_ids = [f"doc_{index}" for index in range(corpus_size)]
    result_ids = list(draw(st.permutations(corpus_ids))) if corpus_ids else []

    outside_size = draw(st.integers(min_value=0, max_value=5))
    candidates = corpus_ids + [f"outside_{index}" for index in range(outside_size)]
    relevant_ids = draw(st.sets(st.sampled_from(candidates), max_size=len(candidates))) \
        if candidates else set()
    k = draw(st.integers(min_value=1, max_value=40))
    return relevant_ids, result_ids, k


class TestProperty39PrecisionAtKFormula:
    """Property 39: Precision@K formula correctness."""

    @given(case=evaluation_cases())
    @settings(max_examples=100, deadline=None)
    def test_precision_matches_relevant_top_k_divided_by_k(self, case):
        """**Validates: Requirements 10.1**"""
        relevant_ids, result_ids, k = case
        hits = len(relevant_ids.intersection(result_ids[:k]))
        expected = hits / k

        actual = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k)

        assert isinstance(actual, float)
        assert 0.0 <= actual <= 1.0
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


class TestProperty40RecallAtKFormula:
    """Property 40: Recall@K formula correctness."""
    @given(case=evaluation_cases())
    @settings(max_examples=100, deadline=None)
    def test_recall_matches_relevant_top_k_divided_by_all_relevant(self, case):
        """**Validates: Requirements 10.2**"""
        relevant_ids, result_ids, k = case
        hits = len(relevant_ids.intersection(result_ids[:k]))
        expected = hits / len(relevant_ids) if relevant_ids else 0.0

        actual = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k)

        assert isinstance(actual, float)
        assert 0.0 <= actual <= 1.0
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


class TestProperty41MrrFormula:
    """Property 41: MRR formula correctness."""

    @given(case=evaluation_cases())
    @settings(max_examples=100, deadline=None)
    def test_mrr_is_reciprocal_rank_of_first_relevant_result(self, case):
        """**Validates: Requirements 10.3**"""
        relevant_ids, result_ids, _ = case
        first_rank = next(
            (rank for rank, doc_id in enumerate(result_ids, start=1)
             if doc_id in relevant_ids),
            None,
        )
        expected = 1.0 / first_rank if first_rank is not None else 0.0

        actual = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)

        assert isinstance(actual, float)
        assert 0.0 <= actual <= 1.0
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


class TestEvaluationMetricsUnitCases:
    """Concrete examples and important boundary cases for all three metrics."""

    def test_known_ranking(self):
        relevant = {"b", "d"}
        results = ["a", "b", "c", "d"]

        assert EvaluationMetrics.precision_at_k(relevant, results, 3) == 1 / 3
        assert EvaluationMetrics.recall_at_k(relevant, results, 3) == 0.5
        assert EvaluationMetrics.mean_reciprocal_rank(relevant, results) == 0.5

    def test_k_larger_than_results_keeps_k_as_precision_denominator(self):
        relevant = {"a"}
        results = ["a", "b"]

        assert EvaluationMetrics.precision_at_k(relevant, results, 5) == 0.2
        assert EvaluationMetrics.recall_at_k(relevant, results, 5) == 1.0

    def test_empty_relevance_and_results_return_zero(self):
        assert EvaluationMetrics.precision_at_k(set(), [], 10) == 0.0
        assert EvaluationMetrics.recall_at_k(set(), [], 10) == 0.0
        assert EvaluationMetrics.mean_reciprocal_rank(set(), []) == 0.0

    def test_no_relevant_result_returns_zero(self):
        relevant = {"missing"}
        results = ["a", "b"]

        assert EvaluationMetrics.precision_at_k(relevant, results, 2) == 0.0
        assert EvaluationMetrics.recall_at_k(relevant, results, 2) == 0.0
        assert EvaluationMetrics.mean_reciprocal_rank(relevant, results) == 0.0

    def test_non_positive_k_returns_zero(self):
        relevant = {"a"}
        results = ["a"]

        assert EvaluationMetrics.precision_at_k(relevant, results, 0) == 0.0
        assert EvaluationMetrics.recall_at_k(relevant, results, -1) == 0.0

    def test_duplicate_results_count_each_document_once(self):
        relevant = {"a"}
        results = ["a", "a"]

        assert EvaluationMetrics.precision_at_k(relevant, results, 2) == 0.5
        assert EvaluationMetrics.recall_at_k(relevant, results, 2) == 1.0
