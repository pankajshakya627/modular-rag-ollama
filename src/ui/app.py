import streamlit as st
import os
import tempfile
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.WARNING)

from src.core.config import get_config
from src.components.retrieval.document_processor import DocumentProcessor, ChunkingStrategy
from src.components.retrieval.vector_store import VectorStoreManager, LangChainChromaVectorStore
from src.core.embedding import get_embedding
from src.components.orchestration.rag_graph import ModularRAGWorkflow

# Initialize Streamlit config
st.set_page_config(
    page_title="Modular RAG - AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom dynamic CSS implementation
st.markdown("""
<style>
/* Modern styling matching the app context */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #1e1e1e;
}
.source-box {
    background: #262730;
    border-left: 4px solid #4CAF50;
    padding: 10px;
    margin: 5px 0;
    border-radius: 4px;
    font-size: 0.9em;
}
.confidence-box {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
    color: white;
}
.confidence-high { background-color: #4CAF50; }
.confidence-medium { background-color: #ff9800; }
.confidence-low { background-color: #f44336; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    
    with st.expander("Document Processing", expanded=True):
        chunk_size = st.slider("Chunk Size", min_value=128, max_value=2048, value=1024, step=128)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=512, value=100, step=10)
    
    with st.expander("Retrieval Parameters", expanded=True):
        alpha = st.slider("Hybrid Search Alpha (Dense Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5, step=1)
    
    with st.expander("Advanced Pipeline Features", expanded=True):
        enable_hyde = st.toggle("Enable HyDE (Hypothetical Doc Embs)", value=True)
        enable_reranking = st.toggle("Enable Cross-Encoder Reranking", value=True)
        enable_raptor = st.toggle("Enable RAPTOR (Hierarchical Retrieval)", value=False)
        enable_query_decomp = st.toggle("Enable Query Decomposition", value=True)
        enable_stepback = st.toggle("Enable Step-Back Prompting", value=True)

    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload files to reference (.pdf, .txt, .docx)", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx']
    )
    
    if st.button("Index Documents", type="primary") and uploaded_files:
        with st.spinner("Processing & Indexing..."):
            try:
                # Setup processor
                embedding = get_embedding()
                processor = DocumentProcessor(
                    chunking_strategy=ChunkingStrategy.RECURSIVE,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                vector_store = LangChainChromaVectorStore(embedding_wrapper=embedding)
                vsm = VectorStoreManager(vector_store=vector_store, embedding_wrapper=embedding)
                
                documents = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Process
                    doc = processor.process_file(tmp_path)
                    documents.append(doc)
                    os.unlink(tmp_path)
                
                vsm.index_documents(documents)
                st.success(f"Successfully indexed {len(documents)} documents!")
            except Exception as e:
                st.error(f"Error during indexing: {str(e)}")

# Main Interface
st.title("🤖 Modular RAG Assistant")
st.markdown("Ask questions based on your indexed documents using advance LangGraph workflows.")

# Initialize the workflow
@st.cache_resource
def get_workflow():
    return ModularRAGWorkflow()

def update_workflow_config():
    """Update workflow configuration based on sidebar inputs."""
    workflow = get_workflow()
    # Ensure it's fully initialized
    if getattr(workflow, 'rag_graph', None) is None:
        workflow._init_workflow()
        
    # Update global config
    from src.core.config import get_config
    config = get_config()
    if hasattr(config, 'hybrid_search'):
        config.hybrid_search.alpha = alpha
        config.hybrid_search.top_k = top_k
        
    # Re-initialize hybrid searcher if alpha changes
    if hasattr(workflow.rag_graph, 'hybrid_searcher'):
        workflow.rag_graph.hybrid_searcher.dense_weight = alpha
        workflow.rag_graph.hybrid_searcher.sparse_weight = 1.0 - alpha
        
    return workflow

workflow = update_workflow_config()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-box"><strong>Doc:</strong> {source["doc_id"]} <br/><em>{source["content"]}</em></div>', unsafe_allow_html=True)
        if "confidence" in message:
            conf = message['confidence']
            color_class = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.5 else "confidence-low"
            st.markdown(f'<span class="confidence-box {color_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("What is Project Orion?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response setup
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with st.status("Thinking... navigating LangGraph workflows", expanded=True) as status:
            try:
                # Custom overrides based on UI toggles
                graph_config = {
                    "configurable": {
                        "enable_hyde": enable_hyde,
                        "enable_reranking": enable_reranking,
                        "enable_raptor": enable_raptor,
                        "enable_query_decomposition": enable_query_decomp,
                        "enable_step_back": enable_stepback,
                    }
                }
                
                st.write("Executing Query...")
                # Unfortunately workflow.query doesn't take config yet, so we pass state manually or modify workflow config
                # Actually, our workflow.query method accepts query string directly.
                result = workflow.query(prompt)
                
                answer = result.get('answer', "I couldn't generate an answer.")
                confidence = result.get('confidence', 0.0)
                sources = result.get('sources', [])
                workflow_stage = result.get('workflow_stage', 'Unknown')
                decomposed = result.get('decomposed_queries', [])
                
                # Update status breakdown
                st.write(f"✓ Reached Stage: {workflow_stage}")
                if decomposed:
                    st.write(f"✓ Deconstructed into {len(decomposed)} sub-queries")
                st.write(f"✓ Retrieved {len(sources)} context chunks")
                status.update(label="Query execution complete", state="complete", expanded=False)
                
                # Display Answer
                message_placeholder.markdown(answer)
                
                # Provide confidence badge
                color_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                st.markdown(f'<span class="confidence-box {color_class}">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
                
                # Display Sources inside expander
                formatted_sources = []
                if sources:
                    with st.expander(f"📚 View Cited Sources ({len(sources[:5])})"):
                        for source in sources[:5]: # Top 5
                            content_snippet = source.get('content', '').replace('\n', ' ')[:200] + "..."
                            doc_id = source.get('metadata', {}).get('document_id', 'Unknown')
                            formatted_sources.append({"doc_id": doc_id, "content": content_snippet})
                            st.markdown(f'<div class="source-box"><strong>Score:</strong> {source.get("score", 0.0):.2f} <br/><em>{content_snippet}</em></div>', unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "confidence": confidence,
                    "sources": formatted_sources
                })
                
            except Exception as e:
                status.update(label="Error executing workflow", state="error", expanded=True)
                st.error(f"Something went wrong: {str(e)}")
