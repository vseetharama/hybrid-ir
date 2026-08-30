"""Search quality evaluation metrics.

Requirements: 10.1, 10.2, 10.3
"""

from typing import List, Set


class EvaluationMetrics:
    """Stateless calculations for Precision@K, Recall@K, and MRR."""

    @staticmethod
    def _relevant_in_top_k(
        relevant_ids: Set[str], result_ids: List[str], k: int
    ) -> int:
        """Count distinct relevant documents among the first ``k`` results."""
        if k <= 0 or not relevant_ids or not result_ids:
            return 0
        return len(set(result_ids[:k]).intersection(relevant_ids))

    @staticmethod
    def precision_at_k(
        relevant_ids: Set[str], result_ids: List[str], k: int = 10
    ) -> float:
        """Return relevant documents in the top-k divided by ``k``.

        A non-positive ``k`` or an empty relevant set produces ``0.0``. When
        fewer than ``k`` results are available, the specified ``k`` remains
        the denominator.
        """
        if k <= 0:
            return 0.0
        hits = EvaluationMetrics._relevant_in_top_k(relevant_ids, result_ids, k)
        return float(hits / k)

    @staticmethod
    def recall_at_k(
        relevant_ids: Set[str], result_ids: List[str], k: int = 10
    ) -> float:
        """Return relevant documents in the top-k divided by all relevant docs."""
        if not relevant_ids or k <= 0:
            return 0.0
        hits = EvaluationMetrics._relevant_in_top_k(relevant_ids, result_ids, k)
        return float(hits / len(relevant_ids))

    @staticmethod
    def mean_reciprocal_rank(
        relevant_ids: Set[str], result_ids: List[str]
    ) -> float:
        """Return the reciprocal rank of the first relevant result, or 0.0."""
        if not relevant_ids:
            return 0.0
        for rank, document_id in enumerate(result_ids, start=1):
            if document_id in relevant_ids:
                return float(1.0 / rank)
        return 0.0
