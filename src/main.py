"""Main entry point for Modular RAG using LangChain and LangGraph."""
import uvicorn
import logging
import argparse
import os
import sys

# Add project root to PYTHONPATH so `python src/main.py` works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point with LangChain/LangGraph support."""
    parser = argparse.ArgumentParser(description="Modular RAG System (LangChain + LangGraph)")
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "index", "ui"],
        default="api",
        help="Run mode: api (default), cli, index, or ui"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    # Index mode arguments
    parser.add_argument(
        "--path",
        help="File or directory to index (for --mode index)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively index directory"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for indexing"
    )
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api(args.host, args.port, args.reload)
    elif args.mode == "cli":
        run_cli()
    elif args.mode == "index":
        if not args.path:
            parser.error("--mode index requires --path argument")
        run_index(args.path, args.recursive, args.chunk_size)
    elif args.mode == "ui":
        run_ui(args.port)


def run_api(host: str, port: int, reload: bool):
    """Run the FastAPI server with LangChain and LangGraph."""
    from src.api.main import app
    from langchain_core.callbacks import BaseCallbackHandler
    
    # Add LangChain logging callback
    class APICallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            logger.info("LangChain LLM started processing")
        
        def on_llm_end(self, response, **kwargs):
            logger.info("LangChain LLM finished processing")
    
    logger.info(f"Starting LangChain/LangGraph API server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def run_cli():
    """Run CLI mode with LangChain chains."""
    from src.components.orchestration.rag_graph import ModularRAGWorkflow
    
    workflow = ModularRAGWorkflow()
    
    print("\n=== Modular RAG CLI (LangChain + LangGraph) ===")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Run query through LangGraph workflow
            result = workflow.query(query)
            
            print(f"\n📝 Answer: {result['answer']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            
            if result.get('sources'):
                print("\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    doc_id = source.get('document_id', 'Unknown')
                    chunk_idx = source.get('metadata', {}).get('chunk_index', '?')
                    content_preview = source.get('content', '')[:60].replace('\n', ' ')
                    print(f"  {i}. [{doc_id}] Chunk {chunk_idx}: \"{content_preview}...\"")
            
            if result.get('decomposed_queries'):
                print(f"\n🔍 Sub-queries analyzed: {len(result['decomposed_queries'])}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_index(path: str, recursive: bool, chunk_size: int):
    """Run indexing mode with LangChain vector store."""
    from src.components.retrieval.document_processor import DocumentProcessor, ChunkingStrategy
    from src.components.retrieval.vector_store import VectorStoreManager, LangChainChromaVectorStore
    from src.core.embedding import get_embedding
    
    embedding = get_embedding()
    vector_store = LangChainChromaVectorStore(embedding_wrapper=embedding)
    vsm = VectorStoreManager(vector_store=vector_store, embedding_wrapper=embedding)
    
    processor = DocumentProcessor(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=chunk_size,
        embedding_wrapper=embedding,
    )
    
    if os.path.isfile(path):
        print(f"📄 Processing file: {path}")
        documents = [processor.process_file(path)]
    elif os.path.isdir(path):
        print(f"📁 Processing directory: {path} (recursive={recursive})")
        documents = processor.process_directory(path, recursive=recursive)
    else:
        print(f"❌ Error: {path} is not a valid file or directory")
        return
    
    print(f"📊 Found {len(documents)} documents")
    
    total_chunks = sum(len(doc.chunks) for doc in documents if doc.chunks)
    print(f"📑 Total chunks: {total_chunks}")
    
    print("🔄 Indexing with LangChain ChromaDB...")
    vsm.index_documents(documents)
    print(f"✅ Indexed {len(documents)} documents successfully!")


def run_ui(port: int):
    """Run the Streamlit UI."""
    import subprocess
    logger.info(f"Starting Streamlit UI on port {port}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", str(port)])
    except KeyboardInterrupt:
        logger.info("Streamlit UI stopped.")


if __name__ == "__main__":
    main()
