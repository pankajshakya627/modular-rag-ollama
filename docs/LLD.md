# Low-Level Design (LLD)

This document provides detailed technical specifications for each component in the **modular-rag-ollama** framework.

---

## 📦 Core Layer

### 1. LLMWrapper (`src/core/llm.py`)

**Purpose:** Provides a unified interface for LLM operations using Ollama.

#### Class: `LLMWrapper`

```python
class LLMWrapper:
    def __init__(self, config: Optional[LLMConfig] = None)
    def generate(self, prompt: str, **kwargs) -> str
    def generate_with_history(self, messages: List[Dict[str, str]], **kwargs) -> str
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict
    def create_prompt_template(self, template: str, input_variables: List[str])
    def create_chain(self, prompt, output_parser=None)
```

| Method                  | Input                  | Output | Notes                  |
| ----------------------- | ---------------------- | ------ | ---------------------- |
| `generate`              | `prompt: str`          | `str`  | Single-turn generation |
| `generate_with_history` | `messages: List[Dict]` | `str`  | Multi-turn chat        |
| `generate_structured`   | `prompt, schema`       | `Dict` | JSON output parsing    |

**Implementation Details:**

- Uses `langchain_ollama.ChatOllama` for chat-style interactions
- Converts message roles (`user`, `assistant`, `system`) to LangChain message types
- Extracts `.content` from `AIMessage` response objects
- Supports temperature and max_tokens overrides per call

**Why ChatOllama over Ollama:**
| Feature | `Ollama` (legacy) | `ChatOllama` |
|---------|-------------------|--------------|
| Message history | ❌ | ✅ |
| System prompts | ❌ | ✅ |
| Role-based API | ❌ | ✅ |
| LCEL compatible | Partial | Full |

---

### 2. EmbeddingWrapper (`src/core/embedding.py`)

**Purpose:** Generates vector embeddings for documents and queries.

```python
class EmbeddingWrapper:
    def __init__(self, config: Optional[EmbeddingConfig] = None)
    def embed_query(self, text: str) -> List[float]
    def embed_documents(self, texts: List[str]) -> List[List[float]]
    def embed_with_metadata(self, text: str, metadata: Dict) -> Dict
```

**Implementation Details:**

- Uses `langchain_ollama.OllamaEmbeddings`
- Default model: `nomic-embed-text` (768 dimensions)
- Query vs. document embeddings are identical (symmetric model)

---

### 3. Configuration (`src/core/config.py`)

**Purpose:** Centralized configuration management.

```python
@dataclass
class LLMConfig:
    model: str = "llama3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048

@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
```

**Configuration Sources (priority order):**

1. Constructor arguments
2. Environment variables (`.env`)
3. YAML config file (`config/config.yaml`)
4. Defaults

---

## 📄 Document Processing (`src/components/retrieval/document_processor.py`)

### Data Structures

```python
@dataclass
class DocumentChunk:
    id: str                           # Unique chunk ID
    content: str                      # Text content
    metadata: Dict[str, Any]          # Source info, positions
    embedding: Optional[List[float]]  # Vector representation
    start_char_idx: int               # Position in original doc
    end_char_idx: int                 # Position in original doc
    chunk_index: int                  # Sequence number

@dataclass
class Document:
    id: str
    content: str                      # Full document text
    chunks: List[DocumentChunk]       # Processed chunks
    metadata: Dict[str, Any]          # Title, source, etc.
    source_type: str                  # "pdf", "text", "docx"
    source_path: Optional[str]        # File path
```

---

### Chunking Strategies

#### 1. RecursiveChunker

**How it works:**

1. Try to split by `\n\n` (paragraphs)
2. If chunks too big, split by `\n` (lines)
3. If still too big, split by `. ` (sentences)
4. If still too big, split by ` ` (words)
5. If still too big, split by character

**Parameters:**

```python
RecursiveChunker(
    chunk_size: int = 1024,       # Target chunk size
    chunk_overlap: int = 128,     # Overlap between chunks
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""],
    length_function: str = "len"  # Character count
)
```

**Issue Solved:** Prevents breaking mid-sentence/mid-paragraph.

---

#### 2. SemanticChunker

**How it works:**

1. Split text into sentences
2. Embed each sentence
3. Calculate cosine similarity between adjacent sentences
4. Split where similarity drops below threshold
5. Merge small chunks up to `chunk_size`

**Parameters:**

```python
SemanticChunker(
    embedding_wrapper: EmbeddingWrapper,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    similarity_threshold: float = 0.7  # Cutoff for splits
)
```

**Issue Solved:** Keeps semantically related content together even across paragraph boundaries.

---

#### 3. FixedSizeChunker

**How it works:**

1. Split at exactly `chunk_size` characters
2. Keep `chunk_overlap` characters from previous chunk

```python
FixedSizeChunker(
    chunk_size: int = 1024,
    chunk_overlap: int = 128
)
```

**When to use:** Predictable token budgets, simple use cases.

---

## 🔍 Retrieval Components

### 1. Vector Store (`src/components/retrieval/vector_store.py`)

#### Class: `LangChainChromaVectorStore`

```python
class LangChainChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_wrapper: Optional[EmbeddingWrapper] = None,
        collection_name: str = "modular_rag_documents",
        persist_directory: Optional[str] = None
    )

    def add_documents(self, documents: List[Document]) -> List[str]
    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]
    def delete(self, ids: List[str]) -> None
    def clear(self) -> None
```

**Search Result:**

```python
@dataclass
class SearchResult:
    id: str
    content: str
    score: float           # Similarity score (0-1)
    metadata: Dict
    document_id: str
    chunk_index: int
```

**Why ChromaDB:**

- Embedded mode (no server needed)
- SQLite persistence
- Metadata filtering support
- HNSW index for fast ANN search

---

### 2. Hybrid Search (`src/components/retrieval/hybrid_search.py`)

**Purpose:** Combines sparse (BM25) and dense (vector) retrieval.

```python
class HybridSearcher:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        bm25_searcher: Optional[BM25Searcher] = None,
        fusion_method: str = "rrf",    # rrf, weighted, score_normalized
        alpha: float = 0.5,            # Weight for dense retrieval
        top_k: int = 50
    )

    def search(self, query: str, top_k: int = None, filters: Dict = None) -> List[SearchResult]
```

#### Fusion Methods:

| Method                           | Formula                           | Best For                 |
| -------------------------------- | --------------------------------- | ------------------------ |
| **RRF** (Reciprocal Rank Fusion) | `1/(k + rank_d) + 1/(k + rank_s)` | General use, robust      |
| **Weighted**                     | `α * score_d + (1-α) * score_s`   | Tunable blending         |
| **Score Normalized**             | Min-max normalize then blend      | When score ranges differ |

**Issue Solved:**

- Dense retrieval misses exact keyword matches
- BM25 misses semantic similarity
- Hybrid captures both

---

### 3. HyDE (`src/components/retrieval/hyde.py`)

**Purpose:** Bridges vocabulary gap between queries and documents.

```python
class HyDERetriever:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_wrapper: LLMWrapper,
        hypothesis_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512
    )

    def generate_hypothetical_document(self, query: str) -> str
    def retrieve(self, query: str, top_k: int = 10, use_hyde: bool = True) -> List[SearchResult]
    def retrieve_with_comparison(self, query: str, top_k: int = 10) -> Dict[str, List[SearchResult]]
```

**How it works:**

1. User asks: "What causes diabetes?"
2. LLM generates hypothetical answer: "Diabetes is caused by insulin resistance..."
3. Embed the hypothetical answer (not the query)
4. Search for similar documents

**Why this works:** Hypothetical answer is in "document language", matching stored docs better than terse queries.

---

### 4. RAPTOR (`src/components/retrieval/raptor.py`)

**Purpose:** Hierarchical document understanding for multi-hop reasoning.

```python
class RAPTORRetriever:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_wrapper: LLMWrapper,
        num_clusters: int = 5,
        max_summary_length: int = 512,
        threshold: float = 0.7,
        embedding_wrapper: EmbeddingWrapper = None
    )

    def build_hierarchy(self, documents: List[Document], build_summaries: bool = True) -> List[ClusterSummary]
    def retrieve(self, query: str, top_k: int = 10, use_hierarchy: bool = True) -> List[SearchResult]
    def retrieve_with_summaries(self, query: str, top_k: int = 10) -> Dict
```

**How it works:**

1. Cluster document chunks by embedding similarity (K-Means)
2. Generate LLM summary for each cluster
3. Store summaries as additional searchable nodes
4. At query time, check cluster centroids first, then dive into relevant chunks

**Issue Solved:** Questions that require synthesizing info across multiple document sections.

---

## 🎯 Reranking Components

### 1. Base Reranker (`src/components/reranking/base.py`)

```python
@dataclass
class RerankedResult:
    id: str
    content: str
    original_score: float    # From initial retrieval
    reranked_score: float    # From reranker
    metadata: Dict
    document_id: str
    chunk_index: int

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_k: int = None) -> List[RerankedResult]
```

---

### 2. CrossEncoderReranker (`src/components/reranking/cross_encoder.py`)

**Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`

**How it works:**

1. Concatenate query + document: `[CLS] query [SEP] document [SEP]`
2. Pass through BERT-style transformer
3. Output: single relevance score

```python
class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
        device: Optional[str] = None  # auto-detect CUDA
    )
```

**Trade-off:** High accuracy, but O(N) inference for N candidates.

---

### 3. ColBERTReranker (`src/components/reranking/colbert.py`)

**Approach:** Late interaction with MaxSim.

**How it works:**

1. Encode query tokens: `Q = [q1, q2, ..., qn]`
2. Encode document tokens: `D = [d1, d2, ..., dm]`
3. Score = `Σ max_j(qi · dj)` for all i

**Trade-off:** Faster than Cross-Encoder (precompute document embeddings), slightly less accurate.

---

## ✍️ Generation Components

### 1. AnswerGenerator (`src/components/generation/answer_generator.py`)

```python
class AnswerGenerator:
    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        system_prompt: Optional[str] = None
    )

    def generate_answer(
        self,
        query: str,
        context: str,
        sources: Optional[List[SearchResult]] = None,
        max_length: Optional[int] = None
    ) -> AnswerResult

    def generate_answer_from_results(
        self,
        query: str,
        results: List[SearchResult],
        max_context_length: int = 4000
    ) -> AnswerResult
```

**Answer Result:**

```python
@dataclass
class AnswerResult:
    answer: str
    sources: List[Dict[str, Any]]  # Referenced documents
    confidence: float              # Estimated confidence
    metadata: Dict[str, Any]
```

**Confidence Estimation:**

- Based on answer length, hedge word presence, context overlap
- Heuristic, not calibrated probability

---

### 2. ResponseSynthesizer

**Purpose:** Merge outputs from multiple retrieval methods.

```python
class ResponseSynthesizer:
    def synthesize(
        self,
        query: str,
        retrieval_results: Dict[str, List[SearchResult]],
        synthesis_strategy: str = "concatenate"  # merge, selective
    ) -> AnswerResult
```

**Strategies:**
| Strategy | Description |
|----------|-------------|
| `concatenate` | Combine all results, deduplicate, generate |
| `merge` | Generate answer per method, then synthesize |
| `selective` | Pick top results from each, generate once |

---

## 🔄 Orchestration (`src/components/orchestration/rag_graph.py`)

### LangGraph State Machine

```python
class GraphState(TypedDict):
    query: str
    original_query: str
    decomposed_queries: List[str]
    stepback_query: str
    dense_results: List[SearchResult]
    sparse_results: List[SearchResult]
    hyde_results: List[SearchResult]
    fused_results: List[SearchResult]
    reranked_results: List[RerankedResult]
    context: str
    answer: str
    sources: List[Dict]
    confidence: float
    workflow_stage: str
    errors: List[str]

class WorkflowStage(Enum):
    QUERY_ANALYSIS = "query_analysis"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    RERANKING = "reranking"
    GENERATION = "generation"
```

### Pipeline Nodes

| Node                | Input State      | Output State                       | Purpose          |
| ------------------- | ---------------- | ---------------------------------- | ---------------- |
| `_query_analysis`   | query            | decomposed_queries, stepback_query | Query expansion  |
| `_dense_retrieval`  | query            | dense_results                      | Vector search    |
| `_sparse_retrieval` | query            | sparse_results                     | BM25 search      |
| `_hyde_retrieval`   | query            | hyde_results                       | HyDE search      |
| `_fusion`           | all results      | fused_results                      | Combine rankings |
| `_reranking`        | fused_results    | reranked_results                   | Precision boost  |
| `_generation`       | reranked_results | answer, sources                    | LLM answer       |

### LCEL Chains

```python
# Decomposition chain
self.decomposition_chain = self.decomposition_prompt | self.llm_wrapper.llm | StrOutputParser()

# Stepback chain
self.stepback_chain = self.stepback_prompt | self.llm_wrapper.llm | StrOutputParser()

# HyDE chain
self.hyde_chain = self.hyde_prompt | self.llm_wrapper.llm | StrOutputParser()
```

**Why LCEL over LLMChain:**

- LLMChain is deprecated in LangChain v1.x
- LCEL is more composable
- Direct string output (no `.get("text")`)

---

## 🌐 API Layer (`src/api/main.py`)

### Endpoints

| Method | Path              | Purpose             |
| ------ | ----------------- | ------------------- |
| POST   | `/query`          | Process a RAG query |
| POST   | `/documents`      | Add documents       |
| GET    | `/documents`      | List documents      |
| DELETE | `/documents/{id}` | Delete document     |
| GET    | `/health`         | Health check        |
| WS     | `/ws/query`       | Streaming query     |

### Query Request

```python
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_hyde: bool = True
    use_reranking: bool = True
    retrieval_method: str = "hybrid"  # dense, sparse, hybrid
```

### Query Response

```python
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: float
    metadata: Dict[str, Any]
```
