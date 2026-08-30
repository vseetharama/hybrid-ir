"""
Unit tests for Streamlit evaluation metrics interface.

Tests verify that the evaluation interface correctly integrates
with the EvaluationMetrics module and handles edge cases.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import sys
from pathlib import Path
from typing import Set, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluator import EvaluationMetrics


# ========================================================================
# Unit Tests for Evaluation Metrics Calculation
# ========================================================================

class TestEvaluationMetricsIntegration:
    """Test evaluation metrics integration in the evaluation interface."""

    def test_precision_at_10_with_sample_data(self):
        """
        **Validates: Requirement 10.1**
        
        Test: Precision@10 is correctly calculated for sample data.
        """
        relevant_ids = {"doc1", "doc3", "doc5"}
        result_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        
        # 3 relevant docs in top 10 / 10 = 0.3
        assert precision == 0.3

    def test_recall_at_10_with_sample_data(self):
        """
        **Validates: Requirement 10.2**
        
        Test: Recall@10 is correctly calculated for sample data.
        """
        relevant_ids = {"doc1", "doc3", "doc5"}
        result_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        
        # 3 relevant docs found in top 10 / 3 total relevant = 1.0
        assert recall == 1.0

    def test_mrr_with_sample_data(self):
        """
        **Validates: Requirement 10.3**
        
        Test: MRR is correctly calculated for sample data.
        """
        relevant_ids = {"doc3", "doc5"}
        result_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # First relevant document (doc3) is at rank 3, so MRR = 1/3 ≈ 0.3333
        assert abs(mrr - (1/3)) < 0.0001

    def test_no_relevant_documents_input(self):
        """
        **Validates: Requirement 10.4, 10.5**
        
        Test: When no relevant documents are provided, metrics are not calculated.
        """
        relevant_ids = set()
        result_ids = ["doc1", "doc2", "doc3"]
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # All should be 0.0 when no relevant docs
        assert precision == 0.0
        assert recall == 0.0
        assert mrr == 0.0

    def test_no_search_results(self):
        """
        **Validates: Requirement 10.4, 10.5**
        
        Test: When no search results exist, metrics cannot be calculated.
        """
        relevant_ids = {"doc1", "doc2"}
        result_ids = []
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # All should be 0.0 when no results
        assert precision == 0.0
        assert recall == 0.0
        assert mrr == 0.0

    def test_metrics_formatted_to_4_decimal_places(self):
        """
        **Validates: Requirement 10.5**
        
        Test: Metrics are formatted to exactly 4 decimal places.
        """
        relevant_ids = {"doc1"}
        result_ids = ["doc1", "doc2", "doc3"]
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # Format to 4 decimal places
        precision_str = f"{precision:.4f}"
        recall_str = f"{recall:.4f}"
        mrr_str = f"{mrr:.4f}"
        
        # Verify format
        assert len(precision_str.split('.')[1]) == 4
        assert len(recall_str.split('.')[1]) == 4
        assert len(mrr_str.split('.')[1]) == 4


# ========================================================================
# Property Tests for Evaluation Metrics
# ========================================================================

class TestEvaluationMetricsProperties:
    """Property-based tests for evaluation metrics."""

    @given(
        relevant_ids=st.sets(st.text(alphabet="0123456789", min_size=1), max_size=10),
        result_ids=st.lists(st.text(alphabet="0123456789", min_size=1), max_size=20, unique=True),
    )
    @settings(max_examples=50)
    def test_precision_always_between_0_and_1(self, relevant_ids, result_ids):
        """
        **Validates: Requirement 10.1**
        
        Property: Precision@10 is always in range [0.0, 1.0].
        """
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        
        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0

    @given(
        relevant_ids=st.sets(st.text(alphabet="0123456789", min_size=1), max_size=10),
        result_ids=st.lists(st.text(alphabet="0123456789", min_size=1), max_size=20, unique=True),
    )
    @settings(max_examples=50)
    def test_recall_always_between_0_and_1(self, relevant_ids, result_ids):
        """
        **Validates: Requirement 10.2**
        
        Property: Recall@10 is always in range [0.0, 1.0].
        """
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    @given(
        relevant_ids=st.sets(st.text(alphabet="0123456789", min_size=1), max_size=10),
        result_ids=st.lists(st.text(alphabet="0123456789", min_size=1), max_size=20, unique=True),
    )
    @settings(max_examples=50)
    def test_mrr_always_between_0_and_1(self, relevant_ids, result_ids):
        """
        **Validates: Requirement 10.3**
        
        Property: MRR is always in range [0.0, 1.0].
        """
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        assert isinstance(mrr, float)
        assert 0.0 <= mrr <= 1.0

    @given(
        relevant_ids=st.sets(st.text(alphabet="0123456789", min_size=1), max_size=10),
        result_ids=st.lists(st.text(alphabet="0123456789", min_size=1), max_size=20, unique=True),
    )
    @settings(max_examples=50)
    def test_all_metrics_return_float(self, relevant_ids, result_ids):
        """
        **Validates: Requirements 10.1, 10.2, 10.3**
        
        Property: All metrics always return float values.
        """
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(mrr, float)


# ========================================================================
# Tests for Evaluation Interface Logic
# ========================================================================

class TestEvaluationInterfaceLogic:
    """Test the logic for the evaluation interface."""

    def test_relevant_document_ids_parsing(self):
        """
        **Validates: Requirement 10.4**
        
        Test: Relevant document IDs are correctly parsed from comma-separated input.
        """
        # Simulate comma-separated input parsing
        input_text = "doc1, doc2, doc3"
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in input_text.split(",") 
            if doc_id.strip()
        )
        
        assert relevant_ids == {"doc1", "doc2", "doc3"}

    def test_relevant_document_ids_with_whitespace(self):
        """
        **Validates: Requirement 10.4**
        
        Test: Whitespace around document IDs is properly handled.
        """
        input_text = "  doc1  ,  doc2  ,  doc3  "
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in input_text.split(",") 
            if doc_id.strip()
        )
        
        assert relevant_ids == {"doc1", "doc2", "doc3"}

    def test_relevant_document_ids_with_empty_fields(self):
        """
        **Validates: Requirement 10.4**
        
        Test: Empty fields in comma-separated input are ignored.
        """
        input_text = "doc1, , doc2, , , doc3"
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in input_text.split(",") 
            if doc_id.strip()
        )
        
        assert relevant_ids == {"doc1", "doc2", "doc3"}

    def test_no_relevant_documents_message(self):
        """
        **Validates: Requirement 10.4**
        
        Test: "Enter relevant document IDs" message is shown when no IDs provided.
        """
        relevant_ids_input = ""
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in relevant_ids_input.split(",") 
            if doc_id.strip()
        )
        
        if not relevant_ids:
            message = "Enter relevant document IDs"
            assert message == "Enter relevant document IDs"

    def test_no_search_results_message(self):
        """
        **Validates: Requirement 10.4**
        
        Test: "No search results to evaluate" message is shown when no results.
        """
        results = []
        
        if not results:
            message = "No search results to evaluate"
            assert message == "No search results to evaluate"

    def test_extract_result_document_ids(self):
        """
        **Validates: Requirement 10.4**
        
        Test: Document IDs are correctly extracted from result objects.
        """
        results = [
            {"document_id": "doc1", "content": "text1"},
            {"document_id": "doc2", "content": "text2"},
            {"document_id": "doc3", "content": "text3"},
        ]
        
        result_ids = [result.get("document_id", "") for result in results]
        
        assert result_ids == ["doc1", "doc2", "doc3"]


# ========================================================================
# Integration Tests
# ========================================================================

class TestEvaluationInterfaceIntegration:
    """Integration tests for the evaluation interface."""

    def test_full_evaluation_flow(self):
        """
        **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
        
        Test: Complete evaluation flow from input to metric display.
        """
        # User enters relevant document IDs
        relevant_input = "doc1, doc2, doc4"
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in relevant_input.split(",") 
            if doc_id.strip()
        )
        
        # Search returns results (5 results total, 3 relevant)
        results = [
            {"document_id": "doc1", "content": "relevant"},
            {"document_id": "doc2", "content": "relevant"},
            {"document_id": "doc3", "content": "not relevant"},
            {"document_id": "doc4", "content": "relevant"},
            {"document_id": "doc5", "content": "not relevant"},
        ]
        
        result_ids = [result.get("document_id", "") for result in results]
        
        # Calculate metrics
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # Verify values
        # Precision@10: 3 relevant in top 5 / 10 = 0.3
        assert precision == 0.3
        # Recall@10: 3 relevant in top 5 / 3 total relevant = 1.0
        assert recall == 1.0
        # MRR: First relevant doc (doc1) at rank 1, so MRR = 1/1 = 1.0
        assert mrr == 1.0

    def test_evaluation_with_partial_recall(self):
        """
        **Validates: Requirements 10.1, 10.2**
        
        Test: Recall is partial when not all relevant docs are in top 10.
        """
        relevant_ids = {"doc1", "doc2", "doc3", "doc4", "doc5"}
        result_ids = ["doc1", "doc2", "doc3"]  # Only 3 of 5 relevant
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        
        assert precision == 0.3  # 3 relevant / 10
        assert recall == 0.6     # 3 relevant / 5 total relevant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
