# Comprehensive RAG Techniques Guide

## Modular RAG System - Technical Deep Dive

This guide explains each advanced technique used in our Modular RAG system, including mathematical foundations, Python/LangChain implementations, common issues, and solutions.

---

## Table of Contents

1. [Dense Retrieval (Vector Search)](#1-dense-retrieval-vector-search)
2. [Sparse Retrieval (BM25)](#2-sparse-retrieval-bm25)
3. [Hybrid Search & Reciprocal Rank Fusion](#3-hybrid-search--reciprocal-rank-fusion-rrf)
4. [HyDE (Hypothetical Document Embeddings)](#4-hyde-hypothetical-document-embeddings)
5. [RAPTOR (Recursive Abstractive Processing)](#5-raptor-recursive-abstractive-processing)
6. [ColBERT Reranking](#6-colbert-reranking)
7. [Cross-Encoder Reranking](#7-cross-encoder-reranking)
8. [Query Decomposition](#8-query-decomposition)
9. [Document Chunking Strategies](#9-document-chunking-strategies)
10. [Temporal-Aware Retrieval](#10-temporal-aware-retrieval)
11. [Metadata Extraction](#11-metadata-extraction)

---

## 1. Dense Retrieval (Vector Search)

### What Is It?

Dense retrieval represents text as continuous, low-dimensional vectors (embeddings) that capture semantic meaning. Unlike keyword matching, dense retrieval finds semantically similar content even without exact word matches.

### How It Works

1. **Embedding Generation**: Text is passed through a neural network (e.g., Sentence Transformers, BERT) to produce a fixed-length vector
2. **Vector Storage**: Embeddings are stored in a vector database (ChromaDB, Pinecone, FAISS)
3. **Similarity Search**: Query is embedded and compared against stored vectors using distance metrics

### Mathematical Foundation

**Cosine Similarity** measures the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = Σ(aᵢ × bᵢ)  (dot product)
- ||A|| = √(Σaᵢ²)     (magnitude/L2 norm)
```

**Score Range**: -1 to 1 (higher = more similar)

**Example Calculation**:

```
A = [0.8, 0.2, 0.5]
B = [0.7, 0.3, 0.6]

A · B = (0.8×0.7) + (0.2×0.3) + (0.5×0.6) = 0.56 + 0.06 + 0.30 = 0.92
||A|| = √(0.64 + 0.04 + 0.25) = √0.93 ≈ 0.964
||B|| = √(0.49 + 0.09 + 0.36) = √0.94 ≈ 0.970

cosine_similarity = 0.92 / (0.964 × 0.970) ≈ 0.984
```

### Python/LangChain Implementation

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434"
)

# Create vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./data/vector_store"
)

# Add documents
vector_store.add_texts(
    texts=["Uber Q2 2025 revenue was $11.2 billion"],
    metadatas=[{"source": "earnings_report", "quarter": "Q2 2025"}]
)

# Similarity search
results = vector_store.similarity_search_with_score(
    query="What was Uber's revenue in Q2?",
    k=5
)
```

### Why It's Better Than Keyword Search

| Aspect      | Keyword Search          | Dense Retrieval                 |
| ----------- | ----------------------- | ------------------------------- |
| Synonyms    | ❌ "car" ≠ "automobile" | ✅ Captures semantic similarity |
| Typos       | ❌ Exact match required | ✅ Robust to minor variations   |
| Context     | ❌ Ignores meaning      | ✅ Understands context          |
| Scalability | ✅ Very fast            | ⚠️ Requires ANN indexing        |

### Common Issues & Solutions

| Issue                            | Cause                            | Solution                                       |
| -------------------------------- | -------------------------------- | ---------------------------------------------- |
| **Out-of-vocabulary words**      | Embedding model hasn't seen term | Use domain-adapted embeddings                  |
| **Short queries perform poorly** | Less semantic signal             | Use HyDE to expand queries                     |
| **Memory issues**                | Large embedding dimensions       | Use dimensionality reduction or smaller models |
| **Slow indexing**                | Large corpus                     | Batch processing, GPU acceleration             |

---

## 2. Sparse Retrieval (BM25)

### What Is It?

BM25 (Best Matching 25) is a probabilistic ranking algorithm based on term frequency and document length. It excels at exact keyword matching and remains a cornerstone of modern search systems.

### How It Works

BM25 improves upon TF-IDF by:

1. **Term Frequency Saturation**: Diminishing returns for repeated terms
2. **Document Length Normalization**: Prevents bias toward longer documents
3. **IDF Weighting**: Rare terms get higher importance

### Mathematical Foundation

**BM25 Scoring Formula**:

```
Score(Q, D) = Σ IDF(qᵢ) × [f(qᵢ, D) × (k₁ + 1)] / [f(qᵢ, D) + k₁ × (1 - b + b × |D|/avgdl)]

Where:
- Q = Query terms {q₁, q₂, ..., qₙ}
- D = Document
- f(qᵢ, D) = Term frequency of qᵢ in D
- |D| = Document length
- avgdl = Average document length in corpus
- k₁ = Term saturation parameter (typically 1.2-2.0)
- b = Length normalization parameter (typically 0.75)
```

**IDF (Inverse Document Frequency)**:

```
IDF(qᵢ) = log[(N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1]

Where:
- N = Total number of documents
- n(qᵢ) = Number of documents containing term qᵢ
```

### Python Implementation

```python
from rank_bm25 import BM25Okapi
from typing import List

class BM25Searcher:
    def __init__(self, documents: List[str]):
        # Tokenize documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.documents = documents

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=1.5,  # Term saturation
            b=0.75   # Length normalization
        )

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results with scores
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_indices]

# Usage
documents = [
    "Uber Q2 2025 Gross Bookings reached $46.8 billion",
    "Tesla Q2 revenue increased by 15 percent",
    "Amazon Web Services grew significantly in Q2"
]
searcher = BM25Searcher(documents)
results = searcher.search("Uber Q2 2025 bookings")
```

### Why We Use BM25

1. **Precision for exact terms**: When users search "Q2 2025", BM25 finds exact matches
2. **Speed**: O(query_length) complexity, no neural inference
3. **No training required**: Works out-of-the-box
4. **Complementary to dense**: Catches cases where dense retrieval fails

### Common Issues & Solutions

| Issue                          | Cause                    | Solution                              |
| ------------------------------ | ------------------------ | ------------------------------------- |
| **No semantic understanding**  | Term-based matching only | Combine with dense retrieval (hybrid) |
| **Vocabulary mismatch**        | Query uses synonyms      | Expand query with synonyms            |
| **Stopword matching**          | Common words dominate    | Apply stopword removal                |
| **Poor ranking for long docs** | Document length bias     | Tune `b` parameter (0.5-0.9)          |

---

## 3. Hybrid Search & Reciprocal Rank Fusion (RRF)

### What Is It?

Hybrid search combines dense (semantic) and sparse (keyword) retrieval to leverage the strengths of both. RRF is a rank fusion algorithm that merges results from multiple retrieval methods without requiring score normalization.

### How It Works

1. **Parallel Retrieval**: Run both dense and sparse searches
2. **Rank-Based Fusion**: Combine results based on their ranks, not raw scores
3. **Final Ranking**: Higher RRF scores indicate documents ranked highly by multiple methods

### Mathematical Foundation

**Reciprocal Rank Fusion Formula**:

```
RRF(d) = Σᵣ∈R [1 / (k + rankᵣ(d))]

Where:
- d = Document
- R = Set of rankers (e.g., {dense, sparse})
- k = Smoothing constant (typically 60)
- rankᵣ(d) = Rank of document d in ranker r (1-indexed)
```

**Example Calculation**:

```
Document "Uber Q2 Report":
- Dense retrieval rank: 3
- Sparse retrieval rank: 1

RRF = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.0159 + 0.0164 = 0.0323

Document "Tesla Q2 Report":
- Dense retrieval rank: 1
- Sparse retrieval rank: 8

RRF = 1/(60+1) + 1/(60+8) = 1/61 + 1/68 ≈ 0.0164 + 0.0147 = 0.0311

Result: "Uber Q2 Report" ranks higher (0.0323 > 0.0311)
```

### Python Implementation

```python
from typing import Dict, List
from collections import defaultdict

def reciprocal_rank_fusion(
    rankings: Dict[str, List[str]],  # ranker_name -> [doc_ids in rank order]
    k: int = 60
) -> List[tuple]:
    """
    Combine multiple rankings using RRF.

    Args:
        rankings: Dict mapping ranker name to ordered list of doc IDs
        k: Smoothing constant (default 60)

    Returns:
        List of (doc_id, rrf_score) sorted by score descending
    """
    rrf_scores = defaultdict(float)

    for ranker_name, doc_ids in rankings.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_results

# Usage
rankings = {
    "dense": ["doc_a", "doc_b", "doc_c", "doc_d"],
    "sparse": ["doc_c", "doc_a", "doc_d", "doc_b"]
}
fused_results = reciprocal_rank_fusion(rankings, k=60)
# Result: doc_a and doc_c score highest (appear in top ranks of both)
```

### Why RRF is Better Than Score Normalization

| Approach              | Issue                       | RRF Advantage                  |
| --------------------- | --------------------------- | ------------------------------ |
| Min-Max Normalization | Outliers skew range         | Uses ranks, immune to outliers |
| Z-Score Normalization | Assumes normal distribution | No distribution assumption     |
| Linear Combination    | Scores on different scales  | Ranks are comparable           |

### Common Issues & Solutions

| Issue                       | Cause                     | Solution                                |
| --------------------------- | ------------------------- | --------------------------------------- |
| **One retriever dominates** | Imbalanced contribution   | Use Weighted RRF                        |
| **Slow for large results**  | Processing many documents | Limit to top-100 per retriever          |
| **No score calibration**    | Different score meanings  | RRF uses ranks, inherently handles this |

---

## 4. HyDE (Hypothetical Document Embeddings)

### What Is It?

HyDE (Hypothetical Document Embeddings) improves retrieval by generating a hypothetical answer first, then using that answer's embedding to search for real documents. This bridges the gap between short queries and long documents.

### How It Works

1. **Generate Hypothesis**: LLM creates a hypothetical answer to the query
2. **Embed Hypothesis**: Convert hypothetical document to vector
3. **Search**: Find real documents similar to the hypothesis

### Why It's Needed

The "query-document gap": Queries are short ("Q2 2025 revenue?") but documents are long and detailed. Dense retrieval struggles because:

- Query embedding is in a different region of vector space
- Short text has less semantic signal

HyDE solves this by creating a "bridge" document.

### Python/LangChain Implementation

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

class HyDERetriever:
    def __init__(self, llm, embeddings, vector_store):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store

        self.prompt = PromptTemplate(
            template="""Write a detailed hypothetical answer to this question.
Include specific facts, figures, and context that would appear in a relevant document.

Question: {query}

Hypothetical Answer:""",
            input_variables=["query"]
        )

    def retrieve(self, query: str, k: int = 5):
        # Step 1: Generate hypothetical document
        hypothesis = self.llm.invoke(
            self.prompt.format(query=query)
        )

        # Step 2: Embed the hypothesis
        hypothesis_embedding = self.embeddings.embed_query(hypothesis)

        # Step 3: Search with hypothesis embedding
        results = self.vector_store.similarity_search_by_vector(
            hypothesis_embedding,
            k=k
        )
        return results

# Example
query = "What was Uber's gross bookings in Q2 2025?"

# Without HyDE: Short query embedding may not match detailed docs
# With HyDE:
# Hypothesis: "In Q2 2025, Uber reported gross bookings of approximately
#             $46.8 billion, representing a 21% year-over-year increase..."
# This detailed text has better embedding alignment with actual documents
```

### Example Transformation

| Original Query     | Hypothetical Document                                                                                                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Q2 2025 revenue?" | "In Q2 2025, the company reported total revenue of $11.2 billion, representing a 17% increase compared to Q2 2024. Key drivers included mobility segment growth and improved delivery margins..." |

### Common Issues & Solutions

| Issue                         | Cause                        | Solution                                |
| ----------------------------- | ---------------------------- | --------------------------------------- |
| **Hallucinated facts**        | LLM generates incorrect info | Focus on structure, not facts           |
| **Slow latency**              | LLM generation overhead      | Cache common queries, use faster models |
| **Wrong domain language**     | Generic hypothesis           | Fine-tune prompt for domain             |
| **Overly verbose hypothesis** | Too long for embedding       | Limit hypothesis length                 |

---

## 5. RAPTOR (Recursive Abstractive Processing)

### What Is It?

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical tree of document summaries, enabling retrieval at multiple levels of abstraction.

### How It Works

1. **Leaf Nodes**: Original document chunks
2. **Clustering**: Group semantically similar chunks
3. **Summarization**: LLM summarizes each cluster
4. **Recursion**: Repeat clustering/summarization on summaries
5. **Tree Structure**: Multi-level hierarchy from details to high-level concepts

```
                    [Root Summary]
                   /              \
          [Summary A]          [Summary B]
          /    |    \          /    |    \
      [S1]  [S2]  [S3]     [S4]  [S5]  [S6]
       |     |     |        |     |     |
    [Chunk1-3]  [Chunk4-6] [Chunk7-9] [Chunk10-12]
```

### Mathematical Foundation: Clustering

**Gaussian Mixture Models (GMM)** for soft clustering:

```
P(x|θ) = Σₖ πₖ × N(x|μₖ, Σₖ)

Where:
- x = Document embedding
- πₖ = Mixing coefficient for cluster k
- N(x|μₖ, Σₖ) = Gaussian distribution with mean μₖ, covariance Σₖ
```

**Optimal Cluster Count** (Bayesian Information Criterion):

```
BIC = -2 × log(L) + k × log(n)

Where:
- L = Likelihood of the model
- k = Number of parameters
- n = Number of data points
```

### Python Implementation

```python
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class RAPTORIndexer:
    def __init__(self, llm, embeddings, num_clusters=5, max_levels=3):
        self.llm = llm
        self.embeddings = embeddings
        self.num_clusters = num_clusters
        self.max_levels = max_levels

    def build_tree(self, chunks: List[str]) -> dict:
        """Build hierarchical tree from document chunks."""
        tree = {"level_0": chunks}
        current_level = chunks

        for level in range(1, self.max_levels + 1):
            if len(current_level) <= self.num_clusters:
                break

            # Embed current level texts
            embeddings = self.embeddings.embed_documents(current_level)

            # Cluster
            clusters = self._cluster(embeddings, min(self.num_clusters, len(current_level)))

            # Summarize each cluster
            summaries = []
            for cluster_texts in self._group_by_cluster(current_level, clusters):
                summary = self._summarize(cluster_texts)
                summaries.append(summary)

            tree[f"level_{level}"] = summaries
            current_level = summaries

        return tree

    def _cluster(self, embeddings: List[List[float]], n_clusters: int):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(embeddings)

    def _summarize(self, texts: List[str]) -> str:
        combined = "\n\n".join(texts)
        prompt = f"Summarize the following:\n{combined}\n\nSummary:"
        return self.llm.invoke(prompt)
```

### Why We Use RAPTOR

1. **Multi-hop queries**: Questions requiring information from multiple sections
2. **Holistic understanding**: High-level summaries capture document themes
3. **Efficiency**: Retrieve summaries instead of many individual chunks
4. **Complex reasoning**: Tree structure supports hierarchical reasoning

### Common Issues & Solutions

| Issue                  | Cause              | Solution                               |
| ---------------------- | ------------------ | -------------------------------------- |
| **Information loss**   | Over-summarization | Limit tree depth, preserve key details |
| **Expensive to build** | Many LLM calls     | Build offline, cache tree              |
| **Cluster quality**    | Poor grouping      | Fine-tune embedding model for domain   |
| **Stale summaries**    | Documents updated  | Implement incremental updates          |

---

## 6. ColBERT Reranking

### What Is It?

ColBERT (Contextualized Late Interaction over BERT) is a reranking model that uses token-level embeddings and "late interaction" to achieve high accuracy while remaining efficient.

### How It Works

1. **Independent Encoding**: Query and document are encoded separately
2. **Token Embeddings**: Each token gets its own embedding vector
3. **Late Interaction**: Query-document interaction happens during scoring, not encoding
4. **MaxSim**: For each query token, find max similarity with any document token

### Mathematical Foundation

**MaxSim Scoring**:

```
Score(Q, D) = Σᵢ max_j cos(qᵢ, dⱼ)

Where:
- Q = {q₁, q₂, ..., qₙ} = Query token embeddings
- D = {d₁, d₂, ..., dₘ} = Document token embeddings
- cos(qᵢ, dⱼ) = Cosine similarity between tokens
```

**Example**:

```
Query: "Uber Q2 revenue"
Q = [E("Uber"), E("Q2"), E("revenue")]

Document: "Q2 2025 Uber reported $11B revenue growth"
D = [E("Q2"), E("2025"), E("Uber"), E("reported"), E("$11B"), E("revenue"), E("growth")]

MaxSim calculation:
- max(cos(E("Uber"), D)) = cos(E("Uber"), E("Uber")) ≈ 0.95
- max(cos(E("Q2"), D))   = cos(E("Q2"), E("Q2"))     ≈ 0.98
- max(cos(E("revenue"), D)) = cos(E("revenue"), E("revenue")) ≈ 0.97

Score = 0.95 + 0.98 + 0.97 = 2.90
```

### Python Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModel

class ColBERTReranker:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token-level embeddings."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

    def maxsim(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """Compute MaxSim score between query and document."""
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_emb, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_emb, dim=-1)

        # Compute similarity matrix [query_len, doc_len]
        similarity = torch.matmul(query_norm, doc_norm.T)

        # MaxSim: for each query token, take max similarity
        max_similarities = similarity.max(dim=1).values

        # Sum over query tokens (excluding special tokens)
        return max_similarities[1:-1].sum().item()  # Exclude [CLS] and [SEP]

    def rerank(self, query: str, documents: List[str]) -> List[tuple]:
        query_emb = self.encode(query)

        scored_docs = []
        for doc in documents:
            doc_emb = self.encode(doc)
            score = self.maxsim(query_emb, doc_emb)
            scored_docs.append((doc, score))

        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
```

### Why ColBERT is Better

| Aspect      | Bi-Encoder    | Cross-Encoder  | ColBERT                    |
| ----------- | ------------- | -------------- | -------------------------- |
| Speed       | ⚡ Fast       | 🐢 Slow        | ⚡ Fast (pre-compute docs) |
| Accuracy    | Medium        | High           | High                       |
| Granularity | Single vector | Joint encoding | Token-level                |
| Storage     | Low           | N/A            | Higher (token embeddings)  |

### Common Issues & Solutions

| Issue                  | Cause                    | Solution                              |
| ---------------------- | ------------------------ | ------------------------------------- |
| **Memory usage**       | Storing token embeddings | Compress embeddings, limit doc length |
| **Slow for many docs** | Many maxsim computations | Batch processing, GPU acceleration    |
| **Low scores overall** | Unnormalized embeddings  | Apply L2 normalization                |

---

## 7. Cross-Encoder Reranking

### What Is It?

Cross-Encoders jointly encode the query and document together, allowing deep interaction between them. This produces highly accurate relevance scores but is computationally expensive.

### How It Works

1. **Concatenation**: Query and document are joined: `[CLS] query [SEP] document [SEP]`
2. **Joint Encoding**: BERT processes the combined sequence
3. **Classification**: [CLS] token output is passed through a classifier for relevance score

### Mathematical Foundation

**Sequence Classification**:

```
Input: "[CLS] What is Q2 revenue? [SEP] Uber Q2 2025 revenue was $11.2B [SEP]"

BERT Encoding: h = BERT(input)
[CLS] Representation: h_cls = h[0]

Relevance Score: s = sigmoid(W × h_cls + b)

Where:
- W = Learned weight matrix
- b = Bias term
- s ∈ [0, 1] = Relevance probability
```

### Python Implementation

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        inputs = self.tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).squeeze().item()

        return score

    def rerank(self, query: str, documents: List[str]) -> List[tuple]:
        """Rerank documents by relevance to query."""
        scored = [(doc, self.score_pair(query, doc)) for doc in documents]
        return sorted(scored, key=lambda x: x[1], reverse=True)

# Usage
reranker = CrossEncoderReranker()
query = "What was Uber's revenue in Q2 2025?"
documents = [
    "Uber Q2 2025 revenue reached $11.2 billion",
    "Tesla delivered 500,000 vehicles in Q2",
    "Apple announced new iPhone features"
]
ranked = reranker.rerank(query, documents)
# Result: Uber doc ranked first with highest score
```

### ColBERT vs Cross-Encoder Comparison

```
Query: "Q2 2025 Uber revenue"
Documents: 1000 candidates

ColBERT:
- Encode query: 1 forward pass
- Pre-computed doc embeddings
- Score: 1000 MaxSim operations (efficient)
- Total: ~100ms

Cross-Encoder:
- Encode each (query, doc) pair: 1000 forward passes
- No pre-computation possible
- Score: 1000 full BERT inferences
- Total: ~10 seconds

Solution: Use ColBERT for top-100, Cross-Encoder for top-10
```

### Common Issues & Solutions

| Issue          | Cause                   | Solution                                      |
| -------------- | ----------------------- | --------------------------------------------- |
| **Very slow**  | Full inference per pair | Limit to top-K from initial retrieval         |
| **Truncation** | Max 512 tokens          | Chunk long documents, score each              |
| **Memory OOM** | Large batch sizes       | Reduce batch size, use gradient checkpointing |

---

## 8. Query Decomposition

### What Is It?

Query decomposition breaks complex, multi-faceted questions into simpler sub-queries that can be answered independently, then combines the results.

### How It Works

1. **Analyze Query**: Identify multiple information needs
2. **Generate Sub-Queries**: Create focused questions for each aspect
3. **Parallel Retrieval**: Retrieve for each sub-query
4. **Merge Results**: Combine and deduplicate retrieved chunks

### Example Transformation

```
Original: "Compare Uber and Lyft Q2 2025 revenue, profit margins, and user growth"

Decomposed:
1. "What was Uber's Q2 2025 revenue?"
2. "What was Lyft's Q2 2025 revenue?"
3. "What was Uber's Q2 2025 profit margin?"
4. "What was Lyft's Q2 2025 profit margin?"
5. "What was Uber's Q2 2025 user growth?"
6. "What was Lyft's Q2 2025 user growth?"
```

### Python Implementation

```python
from langchain_core.prompts import PromptTemplate

class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""Decompose this complex question into simpler sub-questions.
Each sub-question should target a specific piece of information.

Complex Question: {query}

Sub-questions (one per line):""",
            input_variables=["query"]
        )

    def decompose(self, query: str) -> List[str]:
        response = self.llm.invoke(self.prompt.format(query=query))

        # Parse sub-questions from response
        sub_queries = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return sub_queries

    def retrieve_decomposed(self, query: str, retriever) -> List:
        sub_queries = self.decompose(query)

        all_results = []
        seen_ids = set()

        for sub_query in sub_queries:
            results = retriever.retrieve(sub_query)
            for result in results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    all_results.append(result)

        return all_results
```

### Why We Use Query Decomposition

1. **Multi-hop reasoning**: Answers requiring multiple fact gathering
2. **Comparison queries**: "Compare A and B" needs both A and B information
3. **Improved coverage**: Multiple sub-queries retrieve more relevant docs
4. **Reduce complexity**: Simpler queries are easier to match

### Common Issues & Solutions

| Issue                     | Cause                             | Solution                           |
| ------------------------- | --------------------------------- | ---------------------------------- |
| **Too many sub-queries**  | Over-decomposition                | Limit to 5-7 sub-queries           |
| **Redundant sub-queries** | Similar questions generated       | Deduplicate by semantic similarity |
| **Lost context**          | Sub-queries lack original context | Include context in sub-query       |
| **Slow processing**       | Many retrieval calls              | Parallel execution                 |

---

## 9. Document Chunking Strategies

### What Is It?

Chunking divides large documents into smaller pieces suitable for embedding and retrieval. The strategy significantly impacts retrieval quality.

### Chunking Strategies Comparison

| Strategy           | Description                        | Best For                        |
| ------------------ | ---------------------------------- | ------------------------------- |
| **Fixed-Size**     | Split by character/token count     | Simple prototyping              |
| **Recursive**      | Split by separators hierarchically | General purpose                 |
| **Semantic**       | Group by embedding similarity      | Technical documents             |
| **Sentence**       | Keep complete sentences            | Conversational content          |
| **Document-Based** | Use document structure (headers)   | Structured documents (MD, HTML) |

### Mathematical Foundation: Semantic Chunking

**Breakpoint Detection using Cosine Distance**:

```
Given consecutive sentence embeddings: e₁, e₂, ..., eₙ

Similarity(eᵢ, eᵢ₊₁) = cosine(eᵢ, eᵢ₊₁)
Distance(eᵢ, eᵢ₊₁) = 1 - Similarity(eᵢ, eᵢ₊₁)

Breakpoint if: Distance(eᵢ, eᵢ₊₁) > threshold

Percentile-based threshold:
threshold = percentile(all_distances, 95)
```

### Python Implementation

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    index: int
    metadata: dict

class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> List[Chunk]:
        splits = self._recursive_split(text, self.separators)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        index=chunk_index,
                        metadata={"char_length": len(current_chunk)}
                    ))
                    chunk_index += 1

                    # Add overlap from end of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                index=chunk_index,
                metadata={"char_length": len(current_chunk)}
            ))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        splits = text.split(separator)

        result = []
        for split in splits:
            if len(split) > self.chunk_size and len(separators) > 1:
                result.extend(self._recursive_split(split, separators[1:]))
            else:
                result.append(split + separator if separator else split)

        return result
```

### Best Practices

| Practice           | Recommendation                        |
| ------------------ | ------------------------------------- |
| **Chunk Size**     | 256-512 tokens for dense retrieval    |
| **Overlap**        | 10-20% of chunk size                  |
| **Preserve Units** | Keep sentences/paragraphs intact      |
| **Add Context**    | Include document title in each chunk  |
| **Measure Impact** | Test different strategies empirically |

---

## 10. Temporal-Aware Retrieval

### What Is It?

Temporal-aware retrieval extracts and uses date/time information from queries and documents to prioritize temporally relevant results.

### How It Works

1. **Entity Extraction**: Identify dates, quarters, years in text
2. **Normalization**: Convert to standard format (Q2Y2025)
3. **Scoring**: Boost matches, penalize mismatches

### Python Implementation

```python
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TemporalEntity:
    entity_type: str  # quarter, year, month, date
    normalized: str   # Q2Y2025, Y2025, etc.
    year: Optional[int] = None
    quarter: Optional[int] = None

def extract_temporal_entities(text: str) -> List[TemporalEntity]:
    entities = []

    # Quarter patterns: Q1 2025, Q2-2025, 1Q25
    quarter_patterns = [
        (r'Q([1-4])\s*[/-]?\s*(20[0-9]{2})', lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(20[0-9]{2})', _quarter_name_to_num),
    ]

    for pattern, parser in quarter_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            q, y = parser(match)
            entities.append(TemporalEntity(
                entity_type="quarter",
                normalized=f"Q{q}Y{y}",
                year=y,
                quarter=q
            ))

    # Year patterns
    for match in re.finditer(r'\b(20[0-9]{2})\b', text):
        year = int(match.group(1))
        entities.append(TemporalEntity(
            entity_type="year",
            normalized=f"Y{year}",
            year=year
        ))

    return entities

def calculate_temporal_score(
    query_entities: List[TemporalEntity],
    doc_entities: List[TemporalEntity],
    match_boost: float = 1.5,
    mismatch_penalty: float = 0.7
) -> float:
    if not query_entities:
        return 1.0  # No temporal constraint

    query_normalized = {e.normalized for e in query_entities}
    doc_normalized = {e.normalized for e in doc_entities}

    if query_normalized & doc_normalized:  # Intersection exists
        return match_boost
    elif doc_normalized:  # Doc has different time period
        return mismatch_penalty
    else:
        return 1.0  # Doc has no temporal info
```

### Integration with Reranking

```python
class TemporalColBERTReranker(ColBERTReranker):
    def rerank(self, query: str, documents: List[str]) -> List[tuple]:
        # Extract query temporal entities
        query_temporal = extract_temporal_entities(query)

        scored_docs = []
        for doc in documents:
            # Base ColBERT score
            base_score = self.score(query, doc)

            # Temporal adjustment
            doc_temporal = extract_temporal_entities(doc)
            temporal_multiplier = calculate_temporal_score(
                query_temporal, doc_temporal,
                match_boost=1.5,
                mismatch_penalty=0.7
            )

            final_score = base_score * temporal_multiplier
            scored_docs.append((doc, final_score))

        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
```

---

## 11. Metadata Extraction

### What Is It?

Automatic extraction of structured metadata from documents during ingestion, enabling filtering and improved retrieval.

### Metadata Types

| Type          | Examples                | Use Case                  |
| ------------- | ----------------------- | ------------------------- |
| **Temporal**  | Q2 2025, 2024, January  | Time-based filtering      |
| **Financial** | $46.8B, 21%, EBITDA     | Financial document search |
| **Entity**    | Uber, Tesla, California | Entity-based filtering    |
| **Semantic**  | Topics, summary         | Categorization            |

### Python Implementation

```python
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class DocumentMetadata:
    temporal: Dict[str, Any] = field(default_factory=dict)
    financial: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "temporal": self.temporal,
            "financial": self.financial,
            "entities": self.entities,
            # Flattened for filtering
            "quarters": self.temporal.get("quarters", []),
            "years": self.temporal.get("years", []),
            "companies": self.entities.get("companies", []),
        }

class MetadataExtractor:
    CURRENCY_PATTERN = r'\$[\d,.]+\s*(?:billion|million|B|M)'
    PERCENTAGE_PATTERN = r'[-+]?\d+(?:\.\d+)?%'
    COMPANY_NAMES = ['Uber', 'Tesla', 'Apple', 'Google', 'Amazon']

    def extract(self, text: str) -> DocumentMetadata:
        metadata = DocumentMetadata()

        # Temporal
        temporal_entities = extract_temporal_entities(text)
        metadata.temporal = {
            "quarters": list(set(e.normalized for e in temporal_entities
                                if e.entity_type == "quarter")),
            "years": list(set(e.year for e in temporal_entities if e.year)),
        }

        # Financial
        currencies = re.findall(self.CURRENCY_PATTERN, text, re.IGNORECASE)
        percentages = re.findall(self.PERCENTAGE_PATTERN, text)
        metadata.financial = {
            "currency_mentions": currencies[:10],
            "percentages": percentages[:10],
            "is_financial": len(currencies) > 2 or len(percentages) > 2
        }

        # Entities
        text_lower = text.lower()
        companies = [c for c in self.COMPANY_NAMES if c.lower() in text_lower]
        metadata.entities = {"companies": companies}

        return metadata
```

### Integration with Vector Store

```python
# During document processing
extractor = MetadataExtractor()

for chunk in document.chunks:
    # Extract metadata
    extracted = extractor.extract(chunk.content)

    # Merge with chunk metadata
    chunk.metadata.update(extracted.to_dict())

# During retrieval with filtering
results = vector_store.similarity_search(
    query="Q2 2025 revenue",
    filter={"quarters": {"$in": ["Q2Y2025"]}}  # ChromaDB filter syntax
)
```

---

## Summary: Issue Resolution Guide

| Issue                               | Technique             | Solution                       |
| ----------------------------------- | --------------------- | ------------------------------ |
| Short queries perform poorly        | HyDE                  | Generate hypothetical document |
| Missing semantic matches            | Dense Retrieval       | Use embedding-based search     |
| Missing exact keywords              | BM25                  | Use sparse retrieval           |
| Neither works alone                 | Hybrid + RRF          | Combine both retrieval methods |
| Wrong time periods returned         | Temporal Scoring      | Boost/penalize based on dates  |
| Complex multi-part questions        | Query Decomposition   | Break into sub-queries         |
| Need holistic understanding         | RAPTOR                | Build summary tree             |
| Initial results not accurate enough | ColBERT/Cross-Encoder | Rerank top results             |
| Chunks break mid-sentence           | Semantic Chunking     | Use overlap and boundaries     |
| No filtering capability             | Metadata Extraction   | Extract structured fields      |

---

_This guide is part of the Modular RAG System documentation._
_Last updated: February 2026_
