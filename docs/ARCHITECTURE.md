# System Architecture

This document provides visual architecture diagrams and component interaction flows for the **modular-rag-ollama** framework.

> [!TIP]
> The SVG diagrams below are **animated**. Open them in a browser to see data flow animations.

---

## 🏛️ Layered Architecture

![Layered Architecture Diagram](assets/architecture-layers.svg)

<details>
<summary>📌 ASCII Fallback (for terminals)</summary>

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                  PRESENTATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           FastAPI Application                               │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │   │ POST /query │  │POST /docs   │  │ GET /health │  │ WS /ws/query│        │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬───────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                ORCHESTRATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        LangGraph State Machine                              │   │
│  │   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐        │   │
│  │   │ Query  │───▶│Retrieve│───▶│ Fuse   │───▶│Rerank  │───▶│Generate│        │   │
│  │   │Analysis│    │  ×3   │    │Results │    │Top-K   │    │ Answer │         │   │
│  │   └────────┘    └────────┘    └────────┘    └────────┘    └────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬───────────────────────────────────────────┘
                                         │
      ┌──────────────────────────────────┼──────────────────────────────────┐
      │                                  │                                  │
      ▼                                  ▼                                  ▼
┌───────────────┐              ┌───────────────────┐              ┌───────────────┐
│  RETRIEVAL    │              │    RERANKING      │              │  GENERATION   │
├───────────────┤              ├───────────────────┤              ├───────────────┤
│ VectorStore   │              │ CrossEncoder      │              │ AnswerGen     │
│ BM25 Searcher │              │ ColBERT           │              │ Synthesizer   │
│ HyDE          │              │                   │              │               │
│ RAPTOR        │              │                   │              │               │
└───────────────┘              └───────────────────┘              └───────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    CORE LAYER                                       │
│   LLMWrapper (ChatOllama)  |  EmbeddingWrapper (OllamaEmbeddings)  |  Config        │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               INFRASTRUCTURE LAYER                                  │
│       Ollama Server       |        ChromaDB         |       File System             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

</details>

---

## 🔄 Query Processing Flow

![Query Processing Flow](assets/query-flow.svg)

### Flow Description

| Stage                     | Components               | Output              |
| ------------------------- | ------------------------ | ------------------- |
| **1. Query Analysis**     | Decomposition, Step-back | Sub-queries, intent |
| **2. Parallel Retrieval** | Dense, Sparse, HyDE      | 3× result sets      |
| **3. Fusion**             | Reciprocal Rank Fusion   | Merged rankings     |
| **4. Reranking**          | CrossEncoder/ColBERT     | Precision-sorted    |
| **5. Generation**         | ChatOllama + context     | Answer + sources    |

---

## 🔗 Component Interaction

![Component Interaction Diagram](assets/component-interaction.svg)

### Key Interactions

```
FastAPI ──▶ RAGGraph ──┬──▶ Retrieval Components ──▶ VectorStoreManager
                       │
                       ├──▶ Reranking Components
                       │
                       └──▶ Generation Components ──▶ LLMWrapper
```

---

## 📁 Module Dependency Graph

```mermaid
graph TD
    subgraph API["API Layer"]
        A[api/main.py]
    end

    subgraph Orch["Orchestration"]
        B[rag_graph.py]
    end

    subgraph Retrieval["Retrieval"]
        C[vector_store.py]
        D[hybrid_search.py]
        E[hyde.py]
        F[raptor.py]
    end

    subgraph Rerank["Reranking"]
        G[cross_encoder.py]
        H[colbert.py]
    end

    subgraph Gen["Generation"]
        I[answer_generator.py]
    end

    subgraph Core["Core Layer"]
        J[llm.py]
        K[embedding.py]
        L[config.py]
    end

    A --> B
    B --> C & D & E & F
    B --> G & H
    B --> I
    C & D & E & F --> K
    G & H --> J
    I --> J
    J & K --> L
```

---

## 🗄️ Data Storage

```
data/
├── documents/           # Raw input documents (PDF, DOCX, TXT)
└── vector_store/        # ChromaDB persistence (SQLite + embeddings)

config/
└── config.yaml          # Runtime configuration

.env                     # Secrets (OLLAMA_BASE_URL, etc.)
```

---

## 🔧 Initialization Sequence

```mermaid
sequenceDiagram
    participant App as Application
    participant Config as Configuration
    participant Core as Core Layer
    participant Components as Components
    participant Server as FastAPI

    App->>Config: Load config.yaml + .env
    Config-->>App: LLMConfig, EmbeddingConfig

    App->>Core: Initialize LLMWrapper
    Core-->>App: ChatOllama connected

    App->>Core: Initialize EmbeddingWrapper
    Core-->>App: OllamaEmbeddings connected

    App->>Components: Initialize VectorStore, BM25, HyDE, RAPTOR
    Components-->>App: Retrieval ready

    App->>Components: Initialize CrossEncoder, ColBERT
    Components-->>App: Reranking ready

    App->>Components: Initialize AnswerGenerator
    Components-->>App: Generation ready

    App->>Components: Build RAGGraph
    Components-->>App: LangGraph workflow compiled

    App->>Server: uvicorn.run(port=8000)
    Server-->>App: Serving at http://localhost:8000
```

---

## 🚀 Deployment

### Local Development

```bash
ollama serve              # Start Ollama
python -m src.main --mode api --port 8000
```

### Docker (Recommended)

```yaml
services:
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]

  rag-api:
    build: .
    depends_on: [ollama]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    ports: ["8000:8000"]
    volumes: [./data:/app/data]
```

---

## ⚡ Performance & Security

| Aspect            | Consideration    | Mitigation    |
| ----------------- | ---------------- | ------------- |
| **Embedding**     | O(N) per doc     | Batch, async  |
| **Vector Search** | Index size       | HNSW tuning   |
| **Reranking**     | O(K) per query   | ColBERT/GPU   |
| **LLM**           | Token latency    | Streaming     |
| **Security**      | Prompt injection | Sanitization  |
| **Secrets**       | API keys         | .env, not Git |
