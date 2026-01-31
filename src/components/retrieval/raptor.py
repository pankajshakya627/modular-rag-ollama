"""RAPTOR (Recursive Abstractive Processing) implementation."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

# Direct import to avoid circular dependency
from src.core.uuid_utils import generate_uuid  # UUID v7

import numpy as np
from sklearn.cluster import KMeans

from .document_processor import Document, DocumentChunk
from .vector_store import (
    BaseVectorStore,
    LangChainChromaVectorStore,
    SearchResult,
    VectorStoreManager,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Represents a cluster of documents with its summary."""
    id: str
    chunk_ids: List[str]
    summary: str
    centroid_embedding: List[float]
    metadata: Dict[str, Any] = None


class RAPTORRetriever:
    """RAPTOR retriever with hierarchical document clustering and summarization."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_wrapper,
        num_clusters: int = 5,
        max_summary_length: int = 512,
        threshold: float = 0.7,
        embedding_wrapper=None,
    ):
        """Initialize the RAPTOR retriever."""
        self.vsm = vector_store_manager
        self.llm_wrapper = llm_wrapper
        self.num_clusters = num_clusters
        self.max_summary_length = max_summary_length
        self.threshold = threshold
        self.embedding_wrapper = embedding_wrapper
        
        # Storage for cluster summaries
        self.cluster_summaries: List[ClusterSummary] = []
        
        # Store original documents
        self._documents: List[Document] = []
    
    def build_hierarchy(
        self,
        documents: List[Document],
        build_summaries: bool = True,
    ) -> List[ClusterSummary]:
        """Build hierarchical structure over documents."""
        self._documents = documents
        
        # Collect all chunks and their embeddings
        all_chunks = []
        all_embeddings = []
        chunk_to_doc = {}
        
        for doc in documents:
            for chunk in doc.chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")
                all_chunks.append(chunk)
                all_embeddings.append(chunk.embedding)
                chunk_to_doc[chunk.id] = doc.id
        
        if not all_embeddings:
            logger.warning("No embeddings found for clustering")
            return []
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        # Hierarchical clustering
        if len(embeddings_array) <= self.num_clusters:
            # Not enough documents for clustering
            logger.info("Not enough documents for clustering, using all as single cluster")
            cluster_assignments = [0] * len(all_chunks)
            num_actual_clusters = 1
        else:
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                n_init=10,
            )
            cluster_assignments = kmeans.fit_predict(embeddings_array).tolist()
            num_actual_clusters = self.num_clusters
        
        # Group chunks by cluster
        clusters: Dict[int, List[Tuple[DocumentChunk, int, str]]] = {}
        for i, (chunk, cluster_id) in enumerate(zip(all_chunks, cluster_assignments)):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((chunk, i, chunk_to_doc.get(chunk.id, "")))
        
        # Generate summaries for each cluster
        self.cluster_summaries = []
        
        for cluster_id in range(num_actual_clusters):
            if cluster_id not in clusters:
                continue
            
            cluster_items = clusters[cluster_id]
            cluster_chunks = [item[0] for item in cluster_items]
            cluster_embeddings = [all_embeddings[item[1]] for item in cluster_items]
            chunk_ids = [chunk.id for chunk in cluster_chunks]
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            # Generate summary
            summary = ""
            if build_summaries:
                summary = self._summarize_cluster(cluster_chunks)
            
            cluster_summary = ClusterSummary(
                id=generate_uuid(),
                chunk_ids=chunk_ids,
                summary=summary,
                centroid_embedding=centroid,
                metadata={
                    "cluster_id": cluster_id,
                    "num_chunks": len(chunk_ids),
                    "documents": list(set(item[2] for item in cluster_items)),
                },
            )
            self.cluster_summaries.append(cluster_summary)
        
        logger.info(f"Built hierarchy with {len(self.cluster_summaries)} clusters")
        return self.cluster_summaries
    
    def _summarize_cluster(self, chunks: List[DocumentChunk]) -> str:
        """Generate a summary for a cluster of chunks."""
        # Combine chunk contents
        combined_content = "\n\n".join([chunk.content for chunk in chunks])
        
        # Truncate if too long
        if len(combined_content) > 2000:
            combined_content = combined_content[:2000] + "..."
        
        prompt = f"""Generate a concise summary of the following document excerpts.
Focus on the key themes and main points.

Document Excerpts:
{combined_content}

Summary (max {self.max_summary_length} tokens):"""
        
        try:
            summary = self.llm_wrapper.llm.generate(
                prompt,
                max_tokens=self.max_summary_length,
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating cluster summary: {e}")
            # Return first chunk content as fallback
            return chunks[0].content[:500] if chunks else ""
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hierarchy: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieve documents using RAPTOR hierarchy."""
        # Get query embedding
        if self.embedding_wrapper is None:
            raise ValueError("Embedding wrapper required for RAPTOR retrieval")
        
        query_embedding = self.embedding_wrapper.embed_query(query)
        
        if not use_hierarchy or not self.cluster_summaries:
            # Fallback to standard retrieval
            return self.vsm.search(query, top_k, filters)
        
        # Find relevant clusters
        cluster_scores = []
        for cluster in self.cluster_summaries:
            # Calculate similarity to cluster centroid
            centroid = np.array(cluster.centroid_embedding)
            query_emb = np.array(query_embedding)
            
            similarity = np.dot(centroid, query_emb) / (
                np.linalg.norm(centroid) * np.linalg.norm(query_emb) + 1e-8
            )
            cluster_scores.append((cluster, similarity))
        
        # Sort clusters by similarity
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get chunks from top clusters
        selected_chunks = []
        seen_chunk_ids = set()
        
        for cluster, score in cluster_scores:
            if score < self.threshold and len(seen_chunk_ids) > 0:
                break
            
            for chunk_id in cluster.chunk_ids:
                if chunk_id not in seen_chunk_ids:
                    selected_chunks.append(chunk_id)
                    seen_chunk_ids.add(chunk_id)
                
                if len(selected_chunks) >= top_k:
                    break
            
            if len(selected_chunks) >= top_k:
                break
        
        # Retrieve selected chunks from vector store
        all_results = self.vsm.search(query, top_k * 2, filters)
        
        # Filter to selected chunks
        selected_results = [
            r for r in all_results 
            if any(chunk_id in r.id for chunk_id in selected_chunks[:top_k])
        ]
        
        # If not enough results, add from all results
        if len(selected_results) < top_k:
            for r in all_results:
                if r.id not in [sr.id for sr in selected_results]:
                    selected_results.append(r)
                    if len(selected_results) >= top_k:
                        break
        
        return selected_results[:top_k]
    
    def retrieve_with_summaries(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[str, List[SearchResult]]:
        """Retrieve documents and include relevant cluster summaries."""
        # Get relevant clusters
        if self.embedding_wrapper is None:
            raise ValueError("Embedding wrapper required for RAPTOR retrieval")
        
        query_embedding = self.embedding_wrapper.embed_query(query)
        
        cluster_scores = []
        for cluster in self.cluster_summaries:
            centroid = np.array(cluster.centroid_embedding)
            query_emb = np.array(query_embedding)
            
            similarity = np.dot(centroid, query_emb) / (
                np.linalg.norm(centroid) * np.linalg.norm(query_emb) + 1e-8
            )
            cluster_scores.append((cluster, similarity))
        
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get relevant summaries
        relevant_summaries = [
            {"summary": cluster.summary, "score": score, **cluster.metadata}
            for cluster, score in cluster_scores[:3]
            if score > 0.3
        ]
        
        # Get document chunks
        results = self.retrieve(query, top_k)
        
        return {
            "documents": results,
            "summaries": relevant_summaries,
        }
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAPTOR hierarchy."""
        return {
            "num_clusters": len(self.cluster_summaries),
            "total_chunks_covered": sum(
                len(cluster.chunk_ids) for cluster in self.cluster_summaries
            ),
            "cluster_details": [
                {
                    "id": cluster.id,
                    "num_chunks": len(cluster.chunk_ids),
                    "summary_length": len(cluster.summary),
                    "documents": cluster.metadata.get("documents", []),
                }
                for cluster in self.cluster_summaries
            ],
        }
    
    def clear_hierarchy(self) -> None:
        """Clear the RAPTOR hierarchy."""
        self.cluster_summaries = []
        self._documents = []
