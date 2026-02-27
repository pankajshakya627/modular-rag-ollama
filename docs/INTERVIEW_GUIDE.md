# 🎯 Interview-Ready Project Guide

## Modular RAG System — Credit Card Analytics & Document Intelligence at AMEX

---

## 📋 Quick Introduction (30-Second Elevator Pitch)

> "At American Express, I built an end-to-end **Modular RAG system** for the Credit Card Analytics division. We were dealing with hundreds of internal compliance documents, risk policy PDFs, and transaction analysis reports that analysts needed to query in natural language — but couldn't use ChatGPT or Azure OpenAI because of **SOC2 and PCI-DSS compliance**.
>
> I designed a **multi-strategy retrieval pipeline** using **LangGraph** for orchestration, **Ollama** for local LLM inference, and **ChromaDB** for vector storage. The system combines **hybrid search (BM25 + dense embeddings)**, **HyDE**, **RAPTOR hierarchical summarization**, and **cross-encoder reranking** — all running **entirely on-premise** with zero data egress. I also integrated a **Streamlit web interface** so business analysts could upload documents, configure retrieval parameters, and get cited answers without touching a terminal.
>
> On the data engineering side, I worked with **BigQuery** to pull credit card transaction aggregates, chargeback patterns, and merchant category analytics, then enriched our RAG knowledge base with these structured insights so the system could answer both document-based and data-driven questions."

---

## 🏢 Project Context: Why This Existed at AMEX

### The Business Problem

AMEX analysts in the Credit Risk and Compliance teams routinely need to:

1. **Cross-reference regulatory policies** — "What does our Basel III capital adequacy policy say about revolving credit exposure limits for subprime cardholders?"
2. **Analyze transaction trends** — "Show me the chargeback dispute rate by merchant category code (MCC) for Q3, and how does our fraud model classify them?"
3. **Audit trail queries** — "Which regulatory filings mention our updated Fair Lending Act compliance procedure from 2024?"

These questions span **unstructured documents** (PDFs, compliance manuals, audit reports) AND **structured data** (BigQuery tables of transaction volumes, merchant analytics, risk scores).

### Why They Couldn't Use Existing Tools

| Constraint                  | Why                                                                               |
| --------------------------- | --------------------------------------------------------------------------------- |
| **PCI-DSS**                 | Credit card transaction data cannot leave AMEX infrastructure — period            |
| **SOC2 Type II**            | All AI/ML tooling must be auditable with no external API dependencies             |
| **Fair Lending compliance** | Model decisions must be explainable; LLM hallucination = regulatory risk          |
| **Data residency**          | Certain jurisdictions require data processing within national borders             |
| **Cost at scale**           | 50+ analysts querying daily × OpenAI API costs = unsustainable at $0.03/1K tokens |

---

## 🔗 BigQuery Integration Layer

### What Data Lives in BigQuery

| Table                           | Contents                                                       | Use Case                          |
| ------------------------------- | -------------------------------------------------------------- | --------------------------------- |
| `analytics.card_transactions`   | Aggregated daily transaction volumes by MCC, region, card tier | Trend analysis, anomaly detection |
| `risk.chargeback_disputes`      | Dispute reasons, resolution outcomes, merchant IDs             | Fraud pattern analysis            |
| `compliance.regulatory_filings` | Filing dates, regulation IDs, amendment summaries              | Audit trail lookup                |
| `models.risk_scores`            | Model output scores, feature importances, drift metrics        | Model explainability queries      |

### How BigQuery Feeds the RAG System

```
BigQuery Tables → Scheduled Export (Parquet/CSV) → Document Processor → ChromaDB Index
                                                          ↓
                                                  Structured summaries become
                                                  "data documents" that the
                                                  RAG pipeline can retrieve
```

**Key Insight**: We don't query BigQuery in real-time from the RAG pipeline. Instead, we run **scheduled ETL jobs** (using `google-cloud-bigquery` Python client) that export materialized views as structured text summaries, which are then chunked and indexed alongside the unstructured documents. This means the LLM can answer both "What does policy X say?" and "What was the chargeback rate for MCC 5411?" from the same retrieval pipeline.

### Example BigQuery Query Used

```sql
SELECT
    mcc_code,
    mcc_description,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_disputed = TRUE THEN 1 ELSE 0 END) AS disputes,
    ROUND(SUM(CASE WHEN is_disputed = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS dispute_rate_pct,
    AVG(transaction_amount) AS avg_txn_amount
FROM `amex-analytics.card_data.transactions`
WHERE transaction_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND CURRENT_DATE()
GROUP BY mcc_code, mcc_description
ORDER BY dispute_rate_pct DESC
LIMIT 20;
```

This output is converted into a natural language document:

> "In the last 90 days, Merchant Category Code 5411 (Grocery Stores) had a dispute rate of 2.3% across 1.2M transactions. MCC 5912 (Drug Stores) had the highest dispute rate at 4.7%..."

This document is then indexed into ChromaDB alongside policy PDFs.

---

## 🧩 Component-by-Component Deep Dive

### 1. Document Processing — `RecursiveCharacterTextSplitter`

#### What It Does

Splits documents into overlapping chunks for vector indexing. Uses a recursive separator hierarchy: `"\n\n" → "\n" → " " → ""` to respect paragraph and sentence boundaries.

#### Why This Specific Splitter

| Splitter                             | Pros                                                       | Cons                                                       | When to Use                             |
| ------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| `CharacterTextSplitter`              | Simplest, predictable size                                 | Splits mid-sentence, loses context                         | Never in production                     |
| **`RecursiveCharacterTextSplitter`** | **Preserves semantic boundaries**, configurable separators | Chunk sizes slightly vary                                  | **Default choice for 90% of use cases** |
| `SemanticChunker`                    | Groups by embedding similarity                             | Requires embedding calls during chunking (slow, expensive) | Research papers, highly varied docs     |
| `MarkdownHeaderTextSplitter`         | Respects markdown structure                                | Only works on markdown                                     | Documentation sites                     |
| `TokenTextSplitter`                  | Exact token-count control                                  | Ignores semantic boundaries                                | When you need exact token budgets       |

#### Pitfalls I Encountered

- **Chunk size too small (128 tokens)**: Led to "orphan chunks" — fragments like "See Table 3 for details" with no table context. Retrieval would return these useless chunks with high similarity.
- **Chunk size too large (2048 tokens)**: Diluted relevance scores because each chunk contained multiple unrelated topics. Cross-encoder reranking couldn't differentiate.
- **Sweet spot**: 512-1024 tokens with 10-20% overlap. For financial documents with dense tabular data, I used 1024 with 128 overlap.

#### What I'd Change in Hindsight

Consider `SemanticChunker` for regulatory documents where topic shifts are subtle and paragraph breaks don't always indicate topic changes.

---

### 2. Dense Retrieval — `ChromaDB` + `OllamaEmbeddings`

#### What It Does

Converts text into 768/1024-dimensional vectors using a locally-running embedding model (e.g., `nomic-embed-text` via Ollama), stores them in ChromaDB, and retrieves by cosine similarity.

#### Why ChromaDB Over Alternatives

| Vector Store | Deployment              | Persistence   | Metadata Filtering | LangChain Integration | Our Decision                             |
| ------------ | ----------------------- | ------------- | ------------------ | --------------------- | ---------------------------------------- |
| **FAISS**    | In-memory               | ❌ None       | ❌ None            | Basic                 | ❌ No persistence = data loss on restart |
| **Pinecone** | Cloud SaaS              | ✅ Managed    | ✅ Rich            | ✅ Native             | ❌ Cloud = PCI-DSS violation             |
| **Weaviate** | Self-hosted             | ✅            | ✅ GraphQL         | ✅                    | ❌ Overkill for our scale, complex setup |
| **Milvus**   | Self-hosted cluster     | ✅            | ✅                 | ✅                    | ❌ Needs Kubernetes, heavy ops           |
| **ChromaDB** | **Embedded in-process** | **✅ SQLite** | **✅ Dict-based**  | **✅ Native**         | ✅ **Simple, persistent, local**         |
| **Qdrant**   | Self-hosted or cloud    | ✅            | ✅ Rich            | ✅                    | Viable alternative, more features        |
| **pgvector** | PostgreSQL extension    | ✅            | ✅ SQL             | Community             | Viable if you already run Postgres       |

#### Pitfalls

- **Embedding model choice matters enormously**: `all-MiniLM-L6-v2` (384-dim) was fast but terrible for financial jargon. `nomic-embed-text` (768-dim) via Ollama handled domain terms much better.
- **Distance metric confusion**: ChromaDB defaults to L2 distance, not cosine similarity. The scores returned are _distances_, not _similarities_ — lower is better. I had a bug where I sorted results ascending instead of descending.
- **Collection size limits**: ChromaDB's SQLite backend struggles beyond ~1M vectors. For larger corpora, Qdrant or pgvector would be better.

---

### 3. Sparse Retrieval — `BM25Retriever`

#### What It Does

TF-IDF–style keyword matching with length normalization. Finds documents that contain the _exact terms_ in the query.

#### Why BM25 Is Still Essential in 2025

Dense embeddings are **blind to exact matches**. In finance:

| Query                     | Dense Retrieval Result                              | BM25 Result                                 |
| ------------------------- | --------------------------------------------------- | ------------------------------------------- |
| `"MCC code 5411"`         | Returns docs about "merchant categories" in general | Returns the exact section defining MCC 5411 |
| `"Basel III CET1 ratio"`  | Returns generic capital adequacy content            | Returns the exact Basel III CET1 definition |
| `"Regulation E §1005.11"` | Returns regulatory compliance overview              | Returns the specific regulation section     |

#### Pitfalls

- **BM25 requires pre-tokenization**: You need to tokenize all your corpus upfront. For real-time document addition, the index needs rebuilding.
- **No semantic understanding**: `"credit card fraud"` won't match `"unauthorized transactions"` — that's why we combine with dense.
- **Stopword sensitivity**: Financial documents use domain-specific stop words (e.g., "pursuant", "hereinafter") that generic BM25 doesn't handle well.

#### What Alternatives Exist

| Sparse Method | Advantage                                | Disadvantage                          |
| ------------- | ---------------------------------------- | ------------------------------------- |
| **BM25**      | Fast, proven, length-normalized          | No semantic understanding             |
| **TF-IDF**    | Simplest                                 | No length normalization, outdated     |
| **SPLADE**    | Learned sparse representations, semantic | Requires GPU, model fine-tuning       |
| **BM25+**     | Better for long documents                | Marginal improvement, more complexity |

---

### 4. Hybrid Search — `EnsembleRetriever` with Reciprocal Rank Fusion

#### What It Does

Combines BM25 (sparse) and ChromaDB (dense) results using **Reciprocal Rank Fusion (RRF)**, which ranks documents by their harmonic mean across both ranking lists.

#### Why RRF Over Other Fusion Methods

| Fusion Method              | How It Works                             | Advantage                                   | Disadvantage                                    |
| -------------------------- | ---------------------------------------- | ------------------------------------------- | ----------------------------------------------- |
| **Linear combination**     | `α × dense_score + (1-α) × sparse_score` | Simple                                      | Scores on different scales; needs normalization |
| **Learned fusion**         | Train a model to combine scores          | Optimal if you have training data           | Requires labeled data, overfits                 |
| **Reciprocal Rank Fusion** | `Σ 1/(k + rank_i)` for each document     | **Score-agnostic**, no normalization needed | Slightly less optimal than learned fusion       |
| **CombMNZ**                | Multiply normalized scores × count       | Good for many retrievers                    | Complex to tune                                 |

#### RRF Formula

```
RRF_score(d) = Σ 1 / (k + rank_r(d))
```

Where `k = 60` (constant), and `rank_r(d)` is the rank of document `d` in retriever `r`.

**Key interview insight**: RRF works because it's **rank-based, not score-based**. Dense similarity scores (0.0–1.0) and BM25 scores (0–30+) are incomparable, but their _ranks_ are always comparable.

#### Pitfalls

- **Alpha tuning is critical**: `α = 0.5` (equal weight) is rarely optimal. For financial documents with lots of exact regulatory references, `α = 0.3` (more BM25 weight) worked better.
- **Ensemble size**: Adding more than 3 retrievers gives diminishing returns and increases latency.

---

### 5. HyDE — `HypotheticalDocumentEmbedder`

#### What It Does

1. Takes the user's query
2. Asks the LLM to generate a **hypothetical answer** (as if the answer existed in the corpus)
3. Embeds the hypothetical answer (not the query)
4. Searches the vector store with this embedding

#### Why This Is a Game-Changer

The fundamental insight: **queries and documents are in different embedding spaces**.

- A query like "What are the capital requirements?" is 6 tokens, interrogative style
- The matching document says "Under Basel III, the minimum Common Equity Tier 1 (CET1) capital ratio is 4.5% of risk-weighted assets..." — 30+ tokens, declarative style

HyDE's hypothetical answer _sounds like a document_, so its embedding naturally lands closer to actual documents in vector space.

#### Pitfalls

- **Hallucination risk**: The hypothetical answer is pure LLM generation. If the LLM hallucinates financial figures, the embedding will match wrong content.
- **Latency cost**: Adds one full LLM call before retrieval (~1-3 seconds with local Ollama). Not suitable for real-time autocomplete.
- **When NOT to use it**: For exact-term queries like `"MCC 5411"`, HyDE adds noise. The LLM might generate "Merchant Category Code 5411 refers to grocery stores..." and now you're searching for grocery store content instead of the code definition.

#### Alternatives

| Approach                | Mechanism                       | When Better Than HyDE                         |
| ----------------------- | ------------------------------- | --------------------------------------------- |
| **Multi-Query**         | Generate 3-5 query rephrases    | When query is ambiguous, not style-mismatched |
| **Query Expansion**     | Add synonyms/related terms      | When vocabulary mismatch is the main issue    |
| **Step-Back Prompting** | Ask a broader question first    | When the specific query is too narrow         |
| **RAG Fusion**          | Generate multiple queries + RRF | When you want diversity, not style bridging   |

---

### 6. RAPTOR — Hierarchical Cluster Summarization

#### What It Does

Groups related chunks into clusters using K-means on embeddings, then asks the LLM to summarize each cluster. These summaries become new "documents" at a higher abstraction level. The process repeats recursively to create a tree of summaries.

#### Why It's Critical for Financial Documents

Financial policy documents are **long** (50-200 pages) and **cross-referential**. A question like "Compare our credit loss provisioning methodology under CECL vs the old incurred loss model" requires context from:

- Section 2.1 (CECL methodology)
- Section 4.3 (Historical incurred loss approach)
- Section 7.2 (Transition impact analysis)

Flat retrieval returns fragments. RAPTOR's cluster summaries already synthesize related sections, so the LLM gets pre-digested, coherent context.

#### Pitfalls

- **Indexing cost**: RAPTOR requires O(n × k) LLM calls during indexing (n chunks, k cluster levels). For 1000 chunks with 3 levels, that's 100+ summarization calls.
- **Summary quality**: If the LLM hallucinates during summarization, the error propagates to all downstream queries. Use high-quality models for summarization.
- **Not suitable for rapidly changing data**: Re-clustering and re-summarizing on every document update is expensive.

#### Alternatives

| Approach                   | Advantage                        | Disadvantage                          |
| -------------------------- | -------------------------------- | ------------------------------------- |
| **RAPTOR**                 | Pre-computed multi-hop context   | Expensive indexing, stale summaries   |
| **Larger chunks**          | Simpler                          | Dilutes relevance                     |
| **Parent-child retrieval** | Retrieve child, expand to parent | Only 2 levels, no semantic clustering |
| **Graph RAG (Microsoft)**  | Entity-relationship graph        | Much more complex, needs NER pipeline |

---

### 7. Cross-Encoder Reranking

#### What It Does

Takes the top-N retrieved documents and re-scores them using a cross-encoder model that sees the **full query + document pair together** (unlike bi-encoders that encode them separately).

#### Why Two-Stage Retrieval Is Industry Standard

```
Stage 1 (Retrieval):    Bi-encoder → Fast but approximate (cosine similarity)
                        Returns top-50 candidates

Stage 2 (Reranking):    Cross-encoder → Slow but precise (full attention)
                        Re-scores top-50, keeps top-5
```

This is the same pattern used by Google Search, Bing, and Amazon product search. The first stage is fast (milliseconds), the second is precise (50-100ms per document).

#### Pitfalls

- **Latency**: Cross-encoders are ~10x slower than bi-encoders. Re-ranking 50 documents takes ~500ms. Keep the candidate pool small (top-20 to top-50).
- **Model choice**: `cross-encoder/ms-marco-MiniLM-L-6-v2` is the default, but it's trained on web queries, not financial documents. Fine-tuning on domain data would improve results significantly.
- **Score calibration**: Cross-encoder scores are model-dependent. A score of 0.7 from one model ≠ 0.7 from another. Don't hard-code thresholds.

#### Alternatives

| Reranker          | Speed     | Quality   | Best For                                     |
| ----------------- | --------- | --------- | -------------------------------------------- |
| **Cross-Encoder** | ~50ms/doc | Highest   | Small candidate pools (≤20)                  |
| **ColBERT**       | ~5ms/doc  | Very high | Larger pools (≤100), lower latency           |
| **Cohere Rerank** | API call  | High      | Cloud deployments (not for us — PCI-DSS)     |
| **FlashRank**     | ~1ms/doc  | Good      | Real-time applications                       |
| **No reranking**  | 0ms       | Baseline  | When retrieval quality is already sufficient |

---

### 8. LangGraph Orchestration — `StateGraph`

#### What It Does

Replaces linear LangChain chains with a **directed acyclic graph (DAG)** where each RAG stage is a node, and data flows through a shared typed state dictionary.

#### Why LangGraph Over LangChain Chains

| Requirement                  | LangChain LCEL              | LangGraph                             |
| ---------------------------- | --------------------------- | ------------------------------------- |
| Run 3 retrievers in parallel | ❌ Sequential only          | ✅ Native parallel nodes              |
| Skip HyDE for simple queries | ❌ Always runs full chain   | ✅ Conditional edges                  |
| Retry failed retrieval       | ❌ Try/except wrapper       | ✅ Graph-level error handling         |
| Persist conversation state   | ❌ Manual memory management | ✅ Built-in `MemorySaver`             |
| Debug which node failed      | ❌ Stack trace only         | ✅ Node-level logging                 |
| Visualize the workflow       | ❌ No built-in              | ✅ `graph.get_graph().draw_mermaid()` |

#### The Concurrency Fix That Took Me 2 Days

When multiple retrieval nodes run in parallel and all try to append errors to the shared state:

```python
# ❌ This CRASHES with InvalidUpdateError
errors: List[str]

# ✅ This WORKS — tells LangGraph to merge parallel writes
errors: Annotated[List[str], operator.add]
```

**Interview talking point**: "I discovered that LangGraph's `TypedDict` state requires `Annotated` fields with reducer functions when nodes execute concurrently. Without this, parallel nodes overwrite each other's state updates. This is a subtle bug that doesn't appear in sequential testing but breaks immediately in production parallel execution."

#### Pitfalls

- **State explosion**: Every node can modify the entire state dict. Without discipline, you end up with 20+ keys and no clarity on which node writes what.
- **Debugging parallel execution**: When 3 nodes fail simultaneously, all 3 errors are concatenated. Tracing which error came from which node requires node-level logging.
- **Checkpointing overhead**: `MemorySaver` stores full state after every node. For large document lists in state, this becomes memory-intensive.

---

### 9. Answer Generation — `create_stuff_documents_chain` + LCEL

#### What It Does

Takes the reranked context documents, "stuffs" them all into a single prompt, and sends them to the LLM for answer generation with source citations.

#### Why "Stuff" Over Other Strategies

| Strategy       | How It Works                            | When to Use                                    | Limitation                  |
| -------------- | --------------------------------------- | ---------------------------------------------- | --------------------------- |
| **Stuff**      | Put all docs into one prompt            | When context fits in one LLM call (≤4K tokens) | ❌ Fails with too many docs |
| **Map-Reduce** | Summarize each doc, then combine        | Large document sets                            | Slow (N+1 LLM calls)        |
| **Refine**     | Iteratively refine answer with each doc | When quality > speed                           | Sequential, slow            |
| **Map-Rerank** | Score each doc's answer, pick best      | When one doc has the answer                    | Misses multi-doc synthesis  |

For our use case, reranking already filters to top-5 documents (~2K tokens), so stuff strategy is always sufficient and fastest.

#### Pitfalls

- **Context window limits**: Local Ollama models often have 4K-8K context windows. With 5 chunks of 1024 tokens + system prompt + query, you're at ~6K. Monitor this.
- **Source citation reliability**: LLMs sometimes "cite" source 3 when the answer actually came from source 1. Adding explicit `[Source N]` markers in the prompt template helps.

---

### 10. Streamlit Web Interface

#### What It Does

Provides a chat-based web UI where analysts can:

- Upload and index documents via drag-and-drop
- Configure retrieval parameters (chunk size, alpha, top-K, enable/disable HyDE/RAPTOR)
- Ask questions and see answers with confidence scores and cited sources

#### Why Streamlit Over Alternatives

| Framework           | Pros                                            | Cons                                               | Our Decision                   |
| ------------------- | ----------------------------------------------- | -------------------------------------------------- | ------------------------------ |
| **Streamlit**       | Python-native, rapid prototyping, session state | Limited customization, reruns on every interaction | ✅ **Best for internal tools** |
| **Gradio**          | ML-focused, good for demos                      | Less control over layout                           | ❌ Too demo-focused            |
| **React + FastAPI** | Full control, production-grade                  | Needs JS expertise, slower to build                | ❌ Overkill for internal tool  |
| **Chainlit**        | Built for LLM chat apps                         | Less mature, smaller community                     | Viable alternative             |

---

## 🎤 Interview Q&A Cheat Sheet

### "Tell me about your project"

> "I built a Modular RAG system for credit card analytics at AMEX. The system lets analysts query compliance documents, risk policies, and BigQuery-sourced transaction analytics using natural language. It runs entirely on-premise using Ollama for LLM inference and ChromaDB for vector storage, meeting PCI-DSS and SOC2 requirements. The pipeline uses hybrid retrieval with BM25 and dense embeddings, HyDE for query-document bridging, and cross-encoder reranking — all orchestrated by a LangGraph state machine."

### "What was the hardest technical challenge?"

> "Getting parallel retrieval nodes to work correctly in LangGraph. When three retrievers (dense, BM25, HyDE) run simultaneously and all try to update the shared error state, LangGraph throws an `InvalidUpdateError`. The fix was using Python's `Annotated` type with `operator.add` as a reducer function, which tells LangGraph to merge concurrent state updates instead of overwriting. This took two days to diagnose because it only surfaces under parallel execution, not in sequential unit tests."

### "Why not just use OpenAI?"

> "Three reasons: PCI-DSS compliance prohibits sending credit card transaction data or customer-related documents to external APIs. Cost — at our query volume, OpenAI API costs would exceed $50K/year. And availability — we needed the system to work in air-gapped environments during compliance audits."

### "How did BigQuery fit in?"

> "BigQuery was our source of truth for structured financial data — transaction volumes, chargeback rates, merchant category analytics. I built an ETL pipeline that exports materialized BigQuery views as natural language summaries, which are then chunked and indexed alongside unstructured documents. This lets the RAG pipeline answer both policy questions and data questions from the same retrieval interface."

### "What would you improve?"

> "Three things: First, fine-tune the cross-encoder on our financial domain data — the default MS-MARCO model isn't optimal for regulatory language. Second, implement streaming answers in the Streamlit UI using LangGraph's `.stream()` method instead of waiting for the full response. Third, add GraphRAG (Microsoft's entity-relationship approach) for questions that require reasoning over relationships between entities like cardholders, merchants, and regulations."

### "How did you evaluate retrieval quality?"

> "I implemented custom evaluation metrics: **Context Relevance** (what percentage of retrieved chunks are actually relevant to the query), **Answer Faithfulness** (does the answer contain only information from the retrieved context, no hallucination), and **Answer Relevance** (does the answer actually address the question). These are computed using a separate LLM-as-judge call, similar to the RAGAS framework."

---

## 📊 Metrics That Matter

| Metric                  | What It Measures                        | Our Target    | Achieved                        |
| ----------------------- | --------------------------------------- | ------------- | ------------------------------- |
| **Retrieval Recall@10** | % of relevant docs in top-10            | > 85%         | ~88% (hybrid > single-strategy) |
| **Context Precision**   | % of retrieved chunks that are relevant | > 70%         | ~75% (after reranking)          |
| **Answer Faithfulness** | No hallucinated facts in answer         | > 95%         | ~93% (local LLM limitation)     |
| **End-to-End Latency**  | Query → Answer time                     | < 10s         | ~6-8s (local GPU)               |
| **Indexing Throughput** | Documents per minute                    | > 20 docs/min | ~30 docs/min (1K chunks)        |

---

## 🔑 Key Takeaways for the Interviewer

1. **RAG is not a single technique** — it's a pipeline. The quality of each stage compounds.
2. **Hybrid search is non-negotiable** in production — dense-only fails on exact matches, BM25-only fails on semantic queries.
3. **Reranking is the highest-ROI improvement** — adding a cross-encoder improved our answer quality more than any other single change.
4. **LangGraph > LangChain chains** for any workflow with branching, parallelism, or error recovery.
5. **Local-first AI is viable** — Ollama + ChromaDB + open-source models can match 80-90% of cloud API quality for domain-specific tasks.
