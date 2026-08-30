"""
Streamlit Frontend for Hybrid Information Retrieval System
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, List, Any, Set
import time
from src.evaluator import EvaluationMetrics


# Configure page settings (Requirement 9.1)
st.set_page_config(
    page_title="Hybrid IR System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = "http://localhost:8000"


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state for query, results, top_k, and fusion_method.
    This ensures state persists across Streamlit reruns.
    """
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    if "results" not in st.session_state:
        st.session_state.results = []
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = 10
    
    if "fusion_method" not in st.session_state:
        st.session_state.fusion_method = "rrf"
    
    if "execution_time" not in st.session_state:
        st.session_state.execution_time = 0.0
    
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False


def format_score(score: float, decimal_places: int = 4) -> str:
    """
    Format a score to specified decimal places.
    
    Args:
        score: The numerical score to format
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted score string
    """
    return f"{score:.{decimal_places}f}"


def display_result(result: Dict[str, Any], index: int) -> None:
    """
    Display a single search result with formatted scores and metadata.
    
    Args:
        result: Result dict with document_id, content, metadata, combined_score, individual_scores
        index: Result index for display purposes
    """
    # Display result header with rank
    st.subheader(f"Result {index + 1}")
    
    # Extract fields from result
    document_id = result.get("document_id", "Unknown")
    content = result.get("content", "")
    metadata = result.get("metadata", {})
    combined_score = result.get("combined_score", 0.0)
    individual_scores = result.get("individual_scores", {})
    
    # Display content snippet (200 character max) - Requirement 9.3
    if content:
        # Truncate to 200 characters
        snippet = content[:200]
        if len(content) > 200:
            snippet += "..."
        st.write(f"**Content:** {snippet}")
    
    # Display metadata - Requirement 9.3
    if metadata:
        st.write(f"**Metadata:** {json.dumps(metadata, indent=2)}")
    
    # Display scores section
    st.write("**Scores:**")
    
    # Create columns for score display
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    
    # Individual scores - Requirement 9.7 (4 decimal places)
    with score_col1:
        bm25_score = individual_scores.get("bm25_score", 0.0)
        st.metric("BM25", format_score(bm25_score, 4))
    
    with score_col2:
        tfidf_score = individual_scores.get("tfidf_score", 0.0)
        st.metric("TF-IDF", format_score(tfidf_score, 4))
    
    with score_col3:
        semantic_score = individual_scores.get("semantic_score", 0.0)
        st.metric("Semantic", format_score(semantic_score, 4))
    
    with score_col4:
        # Combined score - Requirement 9.7 (4 decimal places)
        st.metric("Combined", format_score(combined_score, 4))
    
    # Display divider between results - Requirement 9.6
    st.divider()


def display_results() -> None:
    """
    Display all search results with scores, metadata, and execution time.
    Validates Requirements 9.2, 9.3, 9.6, 9.7, 9.8
    """
    
    st.subheader("Results")
    
    # Display execution time in milliseconds - Requirement 9.8
    if st.session_state.search_performed:
        st.info(f"⏱️ Execution time: {st.session_state.execution_time:.2f}ms")
    
    # Check if results exist
    if not st.session_state.results:
        # Handle empty state - Requirement 9.8
        st.warning("No results found")
        return
    
    # Display each result - Requirements 9.2, 9.3, 9.6, 9.7
    for idx, result in enumerate(st.session_state.results):
        display_result(result, idx)


def search_interface() -> None:
    """
    Render search interface with search bar and sidebar controls.
    Validates Requirements 9.1, 9.2, 9.4
    """
    
    # Sidebar controls (Requirement 9.2, 9.4)
    st.sidebar.header("Search Controls")
    
    # top_k slider: min 1, max 100, default 10 (Requirement 9.2)
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=100,
        value=st.session_state.top_k,
        help="Number of results to return"
    )
    st.session_state.top_k = top_k
    
    # fusion_method selectbox (Requirement 9.2)
    fusion_method = st.sidebar.selectbox(
        "Fusion Method",
        options=["rrf", "bm25", "tfidf", "semantic"],
        index=["rrf", "bm25", "tfidf", "semantic"].index(st.session_state.fusion_method),
        help="Select ranking fusion method"
    )
    st.session_state.fusion_method = fusion_method
    
    # Search bar (Requirement 9.1)
    st.subheader("Search")
    query = st.text_input(
        "Search Query",
        placeholder="Enter your search query...",
        value=st.session_state.query,
        label_visibility="collapsed"
    )
    st.session_state.query = query
    
    # Search button (Requirement 9.4)
    if st.button("Search", type="primary"):
        if not query.strip():
            st.error("Please enter a search query")
        else:
            # Call backend /search endpoint
            try:
                with st.spinner("Searching..."):
                    response = requests.post(
                        f"{BACKEND_URL}/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "fusion_method": fusion_method
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.results = result.get("results", [])
                        st.session_state.execution_time = result.get("execution_time_ms", 0.0)
                        st.session_state.search_performed = True
                        st.success("Search completed!")
                    elif response.status_code == 400:
                        error_data = response.json()
                        st.error(f"Invalid request: {error_data.get('error_message', 'Unknown error')}")
                    else:
                        error_data = response.json()
                        st.error(f"Search failed: {error_data.get('error_message', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure the backend API is running on http://localhost:8000")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The search took too long.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display results (Task 17) - Requirements 9.2, 9.3, 9.6, 9.7, 9.8
    display_results()


def main() -> None:
    """Main application entry point with tab-based interface"""
    
    # Initialize session state
    initialize_session_state()
    
    # Main title (Requirement 9.1)
    st.title("Hybrid Information Retrieval System")
    
    # Create tab structure (Requirement 9.1, 9.9)
    search_tab, evaluation_tab = st.tabs(["Search Tab", "Evaluation Tab"])
    
    with search_tab:
        search_interface()
    
    with evaluation_tab:
        evaluation_interface()


def evaluation_interface() -> None:
    """
    Render evaluation metrics interface.
    Validates Requirements 10.1, 10.2, 10.3, 10.4, 10.5
    """
    
    st.subheader("Evaluation Metrics")
    st.write("Enter relevant document IDs to calculate evaluation metrics.")
    
    # Text area for relevant document IDs input (comma-separated) - Requirement 10.4
    relevant_docs_input = st.text_area(
        "Relevant Document IDs (comma-separated)",
        placeholder="e.g., doc1, doc2, doc3",
        height=100,
        label_visibility="collapsed"
    )
    
    # Parse relevant document IDs
    relevant_ids = set()
    if relevant_docs_input.strip():
        # Split by comma and strip whitespace from each ID
        relevant_ids = set(
            doc_id.strip() 
            for doc_id in relevant_docs_input.split(",") 
            if doc_id.strip()
        )
    
    # Check if we have both relevant docs and search results
    has_relevant_docs = len(relevant_ids) > 0
    has_search_results = len(st.session_state.results) > 0
    
    # Display appropriate content based on state
    if not has_relevant_docs:
        # Handle empty relevant docs - Requirement 10.4, 10.7
        st.info("Enter relevant document IDs")
    elif not has_search_results:
        # Handle no search results - Requirement 10.4
        st.warning("No search results to evaluate")
    else:
        # Both conditions met: calculate and display metrics - Requirements 10.1, 10.2, 10.3, 10.5
        
        # Extract result document IDs in order
        result_ids = [result.get("document_id", "") for result in st.session_state.results]
        
        # Calculate Precision@10 - Requirement 10.1, 10.5
        precision_at_10 = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        
        # Calculate Recall@10 - Requirement 10.2, 10.5
        recall_at_10 = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        
        # Calculate MRR - Requirement 10.3, 10.5
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        # Display metrics formatted to 4 decimal places - Requirement 10.5
        st.subheader("Search Quality Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Precision@10",
                format_score(precision_at_10, 4)
            )
        
        with metric_col2:
            st.metric(
                "Recall@10",
                format_score(recall_at_10, 4)
            )
        
        with metric_col3:
            st.metric(
                "MRR (Mean Reciprocal Rank)",
                format_score(mrr, 4)
            )
        
        # Display interpretation guidance
        st.divider()
        st.write("**Metric Interpretations:**")
        st.write(f"- **Precision@10**: {format_score(precision_at_10, 4)} - Proportion of relevant documents in top 10 results")
        st.write(f"- **Recall@10**: {format_score(recall_at_10, 4)} - Proportion of all relevant documents found in top 10 results")
        st.write(f"- **MRR**: {format_score(mrr, 4)} - Position of first relevant document (1.0 if at rank 1, 0.5 if at rank 2, etc.)")



if __name__ == "__main__":
    main()
