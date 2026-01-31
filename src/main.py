"""Main entry point for Modular RAG using LangChain and LangGraph."""
import uvicorn
import logging
from src.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point with LangChain/LangGraph support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular RAG System (LangChain + LangGraph)")
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "index"],
        default="api",
        help="Run mode: api (default), cli, or index"
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
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api(args.host, args.port, args.reload)
    elif args.mode == "cli":
        run_cli()
    elif args.mode == "index":
        run_index()


def run_api(host: str, port: int, reload: bool):
    """Run the FastAPI server with LangChain and LangGraph."""
    from src.api.main import app
    from langchain_core.callbacks import BaseCallbackHandler
    
    # Add LangChain logging callback
    class APICallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            logger.info("LangChain LLM started processing")
        
        def on_llm_end(self, response, **kwargs):
            logger.info("LangChain LLM completed processing")
        
        def on_retriever_start(self, query, **kwargs):
            logger.info(f"LangChain retriever started: {query[:50]}...")
        
        def on_retriever_end(self, documents, **kwargs):
            logger.info(f"LangChain retriever completed with {len(documents)} documents")
    
    config = get_config()
    
    logger.info(f"Starting Modular RAG API with LangChain {config.app.version}")
    logger.info(f"LLM Model: {config.llm.model}")
    logger.info(f"Embedding Model: {config.embedding.model}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_cli():
    """Run CLI mode with LangChain chains."""
    from src.components.orchestration.rag_graph import ModularRAGWorkflow
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    workflow = ModularRAGWorkflow()
    
    print("Modular RAG CLI (Powered by LangChain + LangGraph)")
    print("=" * 50)
    print("Type 'exit' or 'quit' to exit")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nThinking (using LangGraph workflow)...")
            result = workflow.query(query)
            
            print(f"\nAnswer: {result.get('answer', 'No answer')}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            
            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for i, src in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {src.get('content', '')[:100]}...")
            
            if result.get('decomposed_queries'):
                print(f"\nSub-queries analyzed: {len(result['decomposed_queries'])}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_index():
    """Run indexing mode with LangChain vector store."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents using LangChain")
    parser.add_argument(
        "path",
        help="File or directory to index"
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
        help="Chunk size"
    )
    
    args = parser.parse_args()
    
    from src.components.retrieval.document_processor import DocumentProcessor, ChunkingStrategy
    from src.components.retrieval.vector_store import VectorStoreManager, LangChainChromaVectorStore
    from src.core.embedding import get_embedding
    
    embedding = get_embedding()
    vector_store = LangChainChromaVectorStore(embedding_wrapper=embedding)
    vsm = VectorStoreManager(vector_store=vector_store, embedding_wrapper=embedding)
    
    processor = DocumentProcessor(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=args.chunk_size,
        embedding_wrapper=embedding,
    )
    
    import os
    
    if os.path.isfile(args.path):
        documents = [processor.process_file(args.path)]
    elif os.path.isdir(args.path):
        documents = processor.process_directory(
            args.path,
            recursive=args.recursive,
        )
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return
    
    print(f"Processing {len(documents)} documents with LangChain...")
    vsm.index_documents(documents)
    print(f"Indexed {len(documents)} documents successfully using LangChain ChromaDB!")


if __name__ == "__main__":
    main()
