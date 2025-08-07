from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

class DocumentType(Enum):
    CONTRACT = "contract"
    STATUTE = "statute"
    CASE_LAW = "case_law"
    REGULATION = "regulation"
    UNKNOWN = "unknown"

@dataclass
class Document:
    content: str
    filename: str
    document_type: DocumentType = DocumentType.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):

        if 'filename' not in self.metadata:
            self.metadata['filename'] = self.filename
        
        if self.document_type == DocumentType.UNKNOWN:
            self._infer_document_type()
    
    def _infer_document_type(self):
        filename_lower = self.filename.lower()
        if 'contract' in filename_lower or 'agreement' in filename_lower:
            self.document_type = DocumentType.CONTRACT
        elif 'statute' in filename_lower or 'code' in filename_lower:
            self.document_type = DocumentType.STATUTE
        elif 'case' in filename_lower or 'v.' in filename_lower:
            self.document_type = DocumentType.CASE_LAW

@dataclass
class Chunk:
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    
    def __str__(self):
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Chunk {self.chunk_index}: {preview}"

@dataclass
class Query:
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedChunk:
    chunk: Chunk
    relevance_score: float
    rank: int
    
    def __str__(self):
        return f"Rank {self.rank} (Score: {self.relevance_score:.3f}): {self.chunk}"

@dataclass
class Citation:
    text: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    
    def format(self) -> str:
        if self.page_number:
            return f"[{self.document_name}, Page {self.page_number}]"
        return f"[{self.document_name}]"

@dataclass
class Answer:
    query_id: str
    text: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    confidence_score: float = 0.0
    generation_time_seconds: float = 0.0
    
    def format_with_citations(self) -> str:
        return self.text