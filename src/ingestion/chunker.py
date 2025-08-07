from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

class LegalDocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_sections: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_sections = preserve_sections
        
        self.separators = [
            "\n\nARTICLE",    
            "\n\nSection",     
            "\n\n§",          
            "\n\n",           
            "\n",             
            ". ",             
            "; ",             
            ", ",             
            " "               
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            cleaned_text = self._clean_legal_text(document.page_content)
            
            chunks = self.splitter.split_text(cleaned_text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        'chunk_index': chunk_idx,
                        'chunk_total': len(chunks),
                        'original_doc_index': doc_idx,
                        'chunk_size': len(chunk_text)
                    }
                )
                all_chunks.append(chunk_doc)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        self._print_chunk_statistics(all_chunks)
        
        return all_chunks
    
    def _clean_legal_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Section\s+(\d+)', r'Section \1', text)
        
        text = text.replace('\f', '\n\n')
        
        return text.strip()
    
    def _print_chunk_statistics(self, chunks: List[Document]):
        if not chunks:
            return
            
        chunk_sizes = [len(c.page_content) for c in chunks]
        
        print(f"\nChunk Statistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Average size: {sum(chunk_sizes) / len(chunks):.0f} chars")
        print(f"  Min size: {min(chunk_sizes)} chars")
        print(f"  Max size: {max(chunk_sizes)} chars")

        if chunks:
            print(f"\nFirst chunk preview:")
            print(f"  {chunks[0].page_content[:150]}...")