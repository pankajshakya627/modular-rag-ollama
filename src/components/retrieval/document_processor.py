"""Document processing and chunking strategies for Modular RAG."""
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging

# Direct import to avoid circular dependency
from src.core.uuid_utils import generate_uuid  # UUID v7 for time-sorted IDs

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    start_char_idx: int = 0
    end_char_idx: int = 0
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "start_char_idx": self.start_char_idx,
            "end_char_idx": self.end_char_idx,
            "chunk_index": self.chunk_index,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            start_char_idx=data.get("start_char_idx", 0),
            end_char_idx=data.get("end_char_idx", 0),
            chunk_index=data.get("chunk_index", 0),
        )


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    content: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "text"
    source_path: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = generate_uuid()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata,
            "source_type": self.source_type,
            "source_path": self.source_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            chunks=[DocumentChunk.from_dict(c) for c in data.get("chunks", [])],
            metadata=data.get("metadata", {}),
            source_type=data.get("source_type", "text"),
            source_path=data.get("source_path"),
        )


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk text into smaller pieces."""
        pass


class RecursiveChunker(BaseChunker):
    """Recursive text chunker using separators."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separators: Optional[List[str]] = None,
        length_function: str = "len",
    ):
        """Initialize the recursive chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        self.length_function = length_function
        
        if length_function == "len":
            self._get_length = len
        else:
            self._get_length = len
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text using hierarchical separators."""
        if not separators or separators[0] == "":
            return [text] if text else []
        
        current_separators = separators[1:]
        recursive_texts = []
        
        for separator in separators[:1]:
            if separator:
                texts = text.split(separator)
            else:
                texts = [text]
            
            for text_fragment in texts:
                if text_fragment:
                    if current_separators:
                        recursive_texts.extend(
                            self._split_text(text_fragment.strip(), current_separators)
                        )
                    else:
                        recursive_texts.append(text_fragment.strip())
        
        return recursive_texts
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk text using recursive splitting."""
        if not text:
            return []
        
        # Clean the text
        text = text.strip()
        
        # Split text recursively
        segments = self._split_text(text, self.separators)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for segment in segments:
            segment_length = self._get_length(segment)
            
            if current_length + segment_length <= self.chunk_size:
                current_chunk += ("" if not current_chunk else " ") + segment
                current_length = self._get_length(current_chunk)
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, chunk_index, metadata, len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                if segment_length > self.chunk_size:
                    # Segment is too big, split it further
                    sub_chunks = self._split_fixed(
                        segment, self.chunk_size, self.chunk_overlap
                    )
                    for sub_chunk in sub_chunks:
                        chunk = self._create_chunk(
                            sub_chunk, chunk_index, metadata, len(sub_chunk)
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = segment
                    current_length = segment_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, chunk_index, metadata, len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_fixed(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        if len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]],
        length: int,
    ) -> DocumentChunk:
        """Create a document chunk."""
        chunk_id = generate_uuid()
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata or {},
            start_char_idx=0,
            end_char_idx=length,
            chunk_index=chunk_index,
        )


class SemanticChunker(BaseChunker):
    """Semantic text chunker using sentence embeddings."""
    
    def __init__(
        self,
        embedding_wrapper,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the semantic chunker."""
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_wrapper = embedding_wrapper
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk text semantically based on embedding similarity."""
        if not text:
            return []
        
        # First, create initial chunks using recursive chunking
        initial_chunks = self.chunker.chunk(text, metadata)
        
        if len(initial_chunks) <= 1:
            return initial_chunks
        
        # Get embeddings for each chunk
        contents = [chunk.content for chunk in initial_chunks]
        embeddings = self.embedding_wrapper.embed_texts(contents)
        
        # Merge chunks that are semantically similar
        final_chunks = []
        current_chunk = initial_chunks[0]
        current_embedding = embeddings[0]
        current_start = 0
        
        for i in range(1, len(initial_chunks)):
            chunk = initial_chunks[i]
            embedding = embeddings[i]
            
            # Calculate similarity with current chunk
            similarity = self.embedding_wrapper.embedding.compute_similarity(
                current_embedding, embedding
            )
            
            if similarity >= self.similarity_threshold:
                # Merge chunks
                current_chunk.content += " " + chunk.content
                current_embedding = self.embedding_wrapper.embedding.compute_centroid(
                    [current_embedding, embedding]
                )
            else:
                # Save current chunk and start new one
                current_chunk.end_char_idx = chunk.start_char_idx
                final_chunks.append(current_chunk)
                
                current_chunk = chunk
                current_embedding = embedding
                current_start = i
        
        # Don't forget the last chunk
        final_chunks.append(current_chunk)
        
        # Re-index chunks
        for idx, chunk in enumerate(final_chunks):
            chunk.chunk_index = idx
        
        return final_chunks


class FixedSizeChunker(BaseChunker):
    """Fixed-size text chunker."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        """Initialize the fixed-size chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk text into fixed-size pieces."""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end]
            
            chunk = DocumentChunk(
                id=generate_uuid(),
                content=content,
                metadata=metadata or {},
                start_char_idx=start,
                end_char_idx=end,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            chunk_index += 1
            
            if start >= len(text):
                break
        
        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-based text chunker."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        sentence_endings: str = ".!?",
    ):
        """Initialize the sentence chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_endings = sentence_endings
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentence_pattern = f"[{re.escape(self.sentence_endings)}]+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Chunk text by sentences."""
        if not text:
            return []
        
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk += ("" if not current_chunk else " ") + sentence + "."
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        id=generate_uuid(),
                        content=current_chunk.strip(),
                        metadata=metadata or {},
                        start_char_idx=0,
                        end_char_idx=current_length,
                        chunk_index=chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence + "."
                current_length = sentence_length + 1
        
        if current_chunk:
            chunk = DocumentChunk(
                id=generate_uuid(),
                content=current_chunk.strip(),
                metadata=metadata or {},
                start_char_idx=0,
                end_char_idx=current_length,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)
        
        return chunks


class DocumentProcessor:
    """Main document processor for handling various document types."""
    
    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        embedding_wrapper=None,
    ):
        """Initialize the document processor."""
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_wrapper = embedding_wrapper
        
        self._init_chunker()
    
    def _init_chunker(self):
        """Initialize the appropriate chunker."""
        if self.chunking_strategy == ChunkingStrategy.RECURSIVE:
            self.chunker = RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            if self.embedding_wrapper is None:
                raise ValueError("Embedding wrapper required for semantic chunking")
            self.chunker = SemanticChunker(
                embedding_wrapper=self.embedding_wrapper,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            self.chunker = FixedSizeChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE:
            self.chunker = SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            self.chunker = RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Process plain text into a document with chunks."""
        doc_id = doc_id or generate_uuid()
        
        # Clean the text
        text = self._clean_text(text)
        
        # Chunk the text
        chunks = self.chunker.chunk(text, metadata)
        
        # Create document
        document = Document(
            id=doc_id,
            content=text,
            chunks=chunks,
            metadata=metadata or {},
            source_type="text",
        )
        
        logger.info(f"Processed text into {len(chunks)} chunks")
        return document
    
    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Process a file and create a document."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and extract text
        content = self._extract_text_from_file(file_path)
        
        # Generate document ID from file hash
        doc_id = self._generate_file_hash(file_path)
        
        # Add file metadata
        if metadata is None:
            metadata = {}
        metadata["file_name"] = path.name
        metadata["file_size"] = path.stat().st_size
        metadata["file_type"] = path.suffix
        
        # Process the text
        document = self.process_text(
            text=content,
            metadata=metadata,
            doc_id=doc_id,
        )
        document.source_path = str(path.absolute())
        document.source_type = path.suffix
        
        return document
    
    def process_directory(
        self,
        directory_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_files: Optional[int] = None,
    ) -> List[Document]:
        """Process all files in a directory."""
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        extensions = extensions or [".txt", ".pdf", ".docx", ".md", ".html"]
        
        files = []
        if recursive:
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(path.glob(f"*{ext}"))
        
        if max_files:
            files = files[:max_files]
        
        documents = []
        for file_path in files:
            try:
                doc = self.process_file(str(file_path))
                documents.append(doc)
                logger.info(f"Processed: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize unicode
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text.strip()
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats."""
        import pypdf
        from docx import Document as DocxDocument
        
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == ".txt" or ext == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == ".pdf":
            text = []
            with open(file_path, 'rb') as f:
                pdf = pypdf.PdfReader(f)
                for page in pdf.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        
        elif ext == ".html":
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate a hash for file identification."""
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    
    def set_chunking_strategy(
        self,
        strategy: ChunkingStrategy,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Change the chunking strategy at runtime."""
        self.chunking_strategy = strategy
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        self._init_chunker()
