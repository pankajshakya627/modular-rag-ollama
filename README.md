# Modular RAG - Production Ready RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system built with Ollama for LLM and embeddings. Features HyDE, RAPTOR, hybrid search, and LangGraph orchestration.

![Modular RAG](https://img.shields.io/badge/version-1.0.0-blue) ![Python 3.10+](https://img.shields.io/badge/python-3.10+-green) ![License MIT](https://img.shields.io/badge/license-MIT-yellow)

## Features

### Query Enhancement

- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve retrieval quality
- **Query Decomposition**: Breaks complex queries into simpler sub-questions
- **Step-back Prompting**: Generates broader queries for better context

### Retrieval Optimization

- **Hybrid Search**: Combines BM25 (sparse) and dense vector retrieval
- **RAPTOR**: Hierarchical document clustering and summarization
- **Granularity-Aware Retrieval**: Long-context chunking with span filtering

### Reranking Methods

- **ColBERT**: Late interaction for token-level matching
- **Cross-Encoder**: Transformer-based query-document scoring
- **Instruction-Following Rerankers**: SOTA reranking capabilities

### Production Ready

- **LangGraph Orchestration**: Graph-based workflow with checkpointing
- **FastAPI Endpoints**: RESTful API for easy integration
- **Comprehensive Evaluation**: Precision, recall, faithfulness, and more

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Modular RAG Architecture                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Query      │    │  Retrieval   │    │   Answer     │   │
│  │ Enhancement  │───▶│  Pipeline    │───▶│  Generation  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ • HyDE       │    │ • Hybrid     │    │ • Response   │   │
│  │ • Decompos.  │    │   Search     │    │   Synthesis  │   │
│  │ • Step-back  │    │ • RAPTOR     │    │ • Confidence │   │
│  └──────────────┘    │ • BM25       │    └──────────────┘   │
│                      └──────────────┘                       │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │  Reranking   │                       │
│                      │ • ColBERT    │                       │
│                      │ • Cross-Enc. │                       │
│                      └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

# Clone the repository

git clone https://github.com/pankajshakya627/modular-rag-ollama.git
cd modular-rag-ollama

# Install dependencies

pip install -r requirements.txt

# Install the package

pip install -e .

````

### Configuration

Edit `config/config.yaml` to configure:

```yaml
llm:
  model: "rnj-1:8b"
  base_url: "http://localhost:11434"

embedding:
  model: "nomic-embed-text-v2-moe"
  base_url: "http://localhost:11434"
````

### Start Ollama

```bash
# Pull required models
ollama pull rnj-1:8b
ollama pull nomic-embed-text-v2-moe

# Start Ollama server
ollama serve
```

### Run the API

```bash
# Start the FastAPI server
python -m src.main --mode api --port 8000
```

### Use the CLI

```bash
# Interactive CLI mode
python -m src.main --mode cli
```

### Index Documents

```bash
# Index a single file
python -m src.main --mode index /path/to/document.pdf

# Index a directory
python -m src.main --mode index /path/to/documents/
```

## API Endpoints

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Modular RAG?"}'
```

### Index Document

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Programmatic Usage

```python
from modular_rag import ModularRAGWorkflow

# Initialize workflow
workflow = ModularRAGWorkflow()

# Query
result = workflow.query("What is RAG?")
print(result["answer"])
print(f"Confidence: {result['confidence']}")
```

## Project Structure

```
modular_rag/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   └── models.py        # Pydantic models
│   ├── components/
│   │   ├── generation/
│   │   │   └── answer_generator.py
│   │   ├── orchestration/
│   │   │   └── rag_graph.py # LangGraph workflow
│   │   ├── reranking/
│   │   │   ├── base.py
│   │   │   ├── colbert.py
│   │   │   └── cross_encoder.py
│   │   └── retrieval/
│   │       ├── document_processor.py
│   │       ├── hybrid_search.py
│   │       ├── hyde.py
│   │       ├── raptor.py
│   │       └── vector_store.py
│   ├── core/
│   │   ├── config.py        # Configuration loader
│   │   ├── embedding.py     # Ollama embeddings
│   │   └── llm.py           # Ollama LLM wrapper
│   ├── utils/
│   │   └── evaluation.py    # Evaluation metrics
│   └── main.py              # CLI entry point
├── tests/
│   └── test_rag.py          # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Components

### Document Processing

- **RecursiveChunker**: Hierarchical text splitting
- **SemanticChunker**: Embedding-based semantic chunking
- **FixedSizeChunker**: Simple fixed-size chunks
- **SentenceChunker**: Sentence-based splitting

### Retrieval

- **HybridSearcher**: BM25 + Dense vector fusion
- **HyDERetriever**: Hypothetical document embeddings
- **RAPTORRetriever**: Hierarchical clustering and summarization

### Reranking

- **ColBERTReranker**: Late interaction scoring
- **CrossEncoderReranker**: Transformer-based scoring
- **LLMEnsembleReranker**: LLM-based relevance scoring

### Evaluation

- **RetrievalEvaluator**: Precision, recall, MRR, NDCG
- **GenerationEvaluator**: Answer relevancy, faithfulness
- **RAGEvaluator**: End-to-end RAG evaluation

## Configuration Options

### HyDE

```yaml
hyde:
  enabled: true
  temperature: 0.3
  max_tokens: 512
```

### RAPTOR

```yaml
raptor:
  enabled: true
  num_clusters: 5
  max_summary_length: 512
```

### Hybrid Search

```yaml
hybrid_search:
  enabled: true
  fusion_method: "rrf" # rrf, weighted, score_normalized
  alpha: 0.5 # Weight for dense retrieval
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=Modular RAG API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/modular_rag
ExecStart=/usr/bin/python -m src.main --mode api --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for RAG abstractions
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestrationllama](https
- [O://ollama.com/) for local LLM serving
