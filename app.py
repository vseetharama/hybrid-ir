"""
Streamlit Frontend for Hybrid Information Retrieval System - User Friendly Version
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, List, Any, Set
from src.evaluator import EvaluationMetrics

# Configure page settings
st.set_page_config(
    page_title="🔍 Search Hub - Hybrid IR System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .title-main {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .help-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-left: 4px solid #28a745;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"


def initialize_session_state() -> None:
    """Initialize session state variables"""
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
    """Format score to decimal places"""
    return f"{score:.{decimal_places}f}"


def load_sample_documents() -> None:
    """Load sample documents into the system"""
    
    sample_docs = [
        {
            "doc_id": "1",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "metadata": {"title": "Machine Learning Basics", "year": 2023}
        },
        {
            "doc_id": "2",
            "content": "Deep learning uses neural networks with multiple layers to process complex data patterns and achieve state-of-the-art results in computer vision and NLP.",
            "metadata": {"title": "Deep Learning Guide", "year": 2023}
        },
        {
            "doc_id": "3",
            "content": "Natural language processing enables computers to understand, interpret, and generate human language in meaningful ways.",
            "metadata": {"title": "NLP Introduction", "year": 2022}
        },
        {
            "doc_id": "4",
            "content": "Semantic search uses embeddings and neural networks to find documents based on meaning rather than just keyword matching.",
            "metadata": {"title": "Semantic Search", "year": 2023}
        },
        {
            "doc_id": "5",
            "content": "Information retrieval systems combine multiple ranking algorithms like BM25, TF-IDF, and semantic search to deliver better results.",
            "metadata": {"title": "Information Retrieval Systems", "year": 2022}
        },
        {
            "doc_id": "6",
            "content": "Reciprocal Rank Fusion is an algorithm that combines rankings from multiple information retrieval systems to produce a better overall ranking.",
            "metadata": {"title": "Reciprocal Rank Fusion", "year": 2021}
        },
        {
            "doc_id": "7",
            "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "metadata": {"title": "Data Science Overview", "year": 2023}
        },
        {
            "doc_id": "8",
            "content": "Neural networks are computational models inspired by biological neurons that can learn complex patterns in data.",
            "metadata": {"title": "Neural Networks", "year": 2023}
        },
    ]
    
    try:
        with st.spinner("📥 Indexing sample documents..."):
            response = requests.post(
                f"{BACKEND_URL}/index",
                json={"documents": sample_docs},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"""
                ✅ **Documents loaded successfully!**
                
                - 📊 Indexed: {result['indexed_count']} documents
                - ❌ Failed: {result['failed_count']}
                
                Now try searching for something like:
                - "machine learning"
                - "neural networks"
                - "semantic search"
                """)
            else:
                st.error(f"❌ Failed to load documents: {response.json().get('error_message', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure the backend is running!")


def format_score(score: float, decimal_places: int = 4) -> str:
    """Format score to decimal places"""
    return f"{score:.{decimal_places}f}"


def search_interface() -> None:
    """Render search interface"""
    
    st.markdown("## 🎯 Search for Documents")
    
    # Quick setup guide
    with st.expander("⚡ Quick Start - Index Sample Documents First", expanded=True):
        st.markdown("""
        **Before searching, you need to index some documents!**
        
        Click the button below to add sample documents to the system.
        """)
        
        if st.button("📥 Load Sample Documents", key="load_samples"):
            load_sample_documents()
    
    st.markdown("---")
    
    # Help section
    with st.expander("📖 How to Search", expanded=False):
        st.markdown("""
        1. **Enter keywords** - Type what you're looking for
        2. **Choose settings** - Adjust results count and search method
        3. **Click Search** - System finds and ranks results instantly
        4. **View results** - Click to expand and see details
        
        **Examples:**
        - "machine learning"
        - "natural language processing"
        - "deep neural networks"
        """)
    
    st.markdown("---")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "🔍 What would you like to find?",
            placeholder="e.g., machine learning, data science, AI",
            value=st.session_state.query,
            label_visibility="collapsed"
        )
        st.session_state.query = query
    
    with col2:
        search_clicked = st.button("🔍 SEARCH", use_container_width=True, type="primary", key="search_btn")
    
    st.markdown("---")
    
    # Sidebar settings
    st.sidebar.markdown("## ⚙️ SETTINGS")
    
    with st.sidebar:
        st.markdown("### 📊 Results to Show")
        top_k = st.slider(
            "Number of results:",
            min_value=1,
            max_value=100,
            value=st.session_state.top_k,
            help="Higher = more results but slower search"
        )
        st.session_state.top_k = top_k
        
        st.markdown("### 🎯 Search Method")
        fusion_method = st.radio(
            "Choose algorithm:",
            options=["rrf", "bm25", "tfidf", "semantic"],
            format_func=lambda x: {
                "rrf": "🔀 HYBRID (Best)",
                "bm25": "🔑 Keywords",
                "tfidf": "📊 Important Terms",
                "semantic": "🧠 Meaning-Based"
            }[x],
            captions=[
                "Combines all methods",
                "Exact word matching",
                "Term importance ranking",
                "Concept similarity"
            ]
        )
        st.session_state.fusion_method = fusion_method
    
    # Search execution
    if search_clicked:
        if not query.strip():
            st.error("❌ Please enter a search query!")
            return
        
        try:
            with st.spinner("🔍 Searching... (usually <1 second)"):
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
                    st.balloons()
                else:
                    error_data = response.json()
                    st.error(f"❌ Error: {error_data.get('error_message', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("""
            ❌ **Cannot connect to backend!**
            
            Make sure the backend API is running:
            ```bash
            python -m uvicorn src.main:app --reload
            ```
            """)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if st.session_state.search_performed:
        display_results()


def display_results() -> None:
    """Display search results with nice formatting"""
    
    st.markdown("---")
    st.markdown("## 📋 RESULTS")
    
    if not st.session_state.results:
        st.info("🔍 No results found. Try different keywords!")
        return
    
    # Results stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Speed", f"{st.session_state.execution_time:.2f}ms")
    with col2:
        st.metric("📊 Found", f"{len(st.session_state.results)} documents")
    with col3:
        st.metric("🎯 Method", st.session_state.fusion_method.upper())
    
    st.markdown("---")
    
    # Display each result
    for idx, result in enumerate(st.session_state.results, 1):
        display_result_card(result, idx)


def display_result_card(result: Dict[str, Any], index: int) -> None:
    """Display individual result card"""
    
    doc_id = result.get("document_id", "Unknown")
    content = result.get("content", "No content")
    metadata = result.get("metadata", {})
    combined_score = result.get("combined_score", 0.0)
    individual_scores = result.get("individual_scores", {})
    
    # Relevance indicator
    relevance_pct = int(combined_score * 100)
    
    with st.expander(
        f"**#{index}** | 📄 {doc_id} | Relevance: {relevance_pct}% {'🌟🌟🌟' if relevance_pct > 80 else '🌟🌟' if relevance_pct > 60 else '🌟'}",
        expanded=(index == 1)
    ):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Document Information:**")
            if metadata:
                for key, value in metadata.items():
                    st.write(f"- **{key}**: {value}")
            else:
                st.write("- No metadata")
            
            st.markdown("**Content Preview:**")
            preview = content[:400] if content else "No content available"
            if len(content) > 400:
                preview += "..."
            st.info(preview)
        
        with col2:
            st.markdown("### Score")
            st.metric("Combined", format_score(combined_score, 3))
        
        st.markdown("---")
        
        st.markdown("### 📊 Score Breakdown")
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)
        
        with score_col1:
            st.metric(
                "🔑 BM25",
                format_score(individual_scores.get("bm25_score", 0.0), 3)
            )
        
        with score_col2:
            st.metric(
                "📈 TF-IDF",
                format_score(individual_scores.get("tfidf_score", 0.0), 3)
            )
        
        with score_col3:
            st.metric(
                "🧠 Semantic",
                format_score(individual_scores.get("semantic_score", 0.0), 3)
            )
        
        with score_col4:
            st.metric(
                "🎯 Combined",
                format_score(combined_score, 3)
            )


def evaluation_interface() -> None:
    """Render evaluation metrics interface"""
    
    st.markdown("## 📊 Evaluate Search Results")
    
    with st.expander("📖 How to Evaluate", expanded=True):
        st.markdown("""
        **Evaluate the quality of search results:**
        
        1. Run a search first
        2. Note which results are correct
        3. Enter their document IDs below
        4. Get quality metrics
        
        **Metrics explained:**
        - **Precision@10**: Out of top 10 results, how many are relevant?
        - **Recall@10**: Out of all relevant docs, how many were found?
        - **MRR**: Position of first relevant result (higher = better)
        """)
    
    st.markdown("---")
    
    st.markdown("### Enter Relevant Document IDs")
    
    relevant_docs_input = st.text_area(
        "Which results are correct? (comma-separated IDs)",
        placeholder="e.g., doc1, doc2, doc5",
        height=80,
        label_visibility="collapsed"
    )
    
    # Parse relevant docs
    relevant_ids = set()
    if relevant_docs_input.strip():
        relevant_ids = set(
            doc_id.strip()
            for doc_id in relevant_docs_input.split(",")
            if doc_id.strip()
        )
    
    # Calculate metrics
    if not st.session_state.results:
        st.warning("⚠️ Run a search first!")
    elif not relevant_ids:
        st.info("📝 Enter document IDs to evaluate")
    else:
        result_ids = [r.get("document_id", "") for r in st.session_state.results]
        
        precision = EvaluationMetrics.precision_at_k(relevant_ids, result_ids, k=10)
        recall = EvaluationMetrics.recall_at_k(relevant_ids, result_ids, k=10)
        mrr = EvaluationMetrics.mean_reciprocal_rank(relevant_ids, result_ids)
        
        st.markdown("### 📈 Search Quality Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Precision@10", format_score(precision, 3))
            st.caption("✓ Correctness of results")
        
        with metric_col2:
            st.metric("Recall@10", format_score(recall, 3))
            st.caption("✓ Coverage of relevant docs")
        
        with metric_col3:
            st.metric("MRR", format_score(mrr, 3))
            st.caption("✓ Position of first match")
        
        st.markdown("---")
        
        st.markdown("### 📊 Interpretation")
        st.markdown(f"""
        - **Precision@10 = {format_score(precision, 3)}**: 
          {int(precision*10)} out of 10 results are relevant
        
        - **Recall@10 = {format_score(recall, 3)}**: 
          Found {int(recall*100)}% of all relevant documents
        
        - **MRR = {format_score(mrr, 3)}**: 
          First relevant result at rank {int(1/mrr) if mrr > 0 else 'none found'}
        """)


def about_interface() -> None:
    """Render about/info page"""
    
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### 🔍 Hybrid Information Retrieval System
    
    This system finds documents using **three different search methods** combined intelligently:
    
    #### Search Methods:
    1. **🔑 BM25** - Keyword matching (like Google)
    2. **📊 TF-IDF** - Term importance ranking
    3. **🧠 Semantic** - AI-powered meaning search
    4. **🔀 RRF** - Hybrid combination of all three
    
    #### Why Hybrid?
    - Different methods find different results
    - Combining them gives better coverage
    - More likely to find what you're looking for
    
    #### How It Works:
    ```
    Query Input
        ↓
    [BM25] [TF-IDF] [Semantic]
        ↓        ↓        ↓
    Rank separately
        ↓
    RRF Fusion
        ↓
    Final Ranked Results
    ```
    
    #### Technology Stack:
    - **Backend**: FastAPI (Python)
    - **Frontend**: Streamlit
    - **Database**: MongoDB
    - **Search**: BM25, TF-IDF, FAISS
    - **AI**: Sentence-Transformers
    
    #### API Endpoints:
    - `POST /search` - Search documents
    - `POST /index` - Index new documents
    - `GET /health` - Health check
    
    #### Performance:
    - Typical search: < 150ms
    - Supports 100+ results
    - Multiple fusion methods
    
    ---
    
    **Made with ❤️ for information retrieval**
    """)


def main() -> None:
    """Main application"""
    
    initialize_session_state()
    
    # Header
    st.markdown("""
    # 🔍 Search Hub
    ## Hybrid Information Retrieval System
    
    **Find documents using multiple search methods combined intelligently**
    """)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Evaluate", "ℹ️ About"])
    
    with tab1:
        search_interface()
    
    with tab2:
        evaluation_interface()
    
    with tab3:
        about_interface()


if __name__ == "__main__":
    main()
