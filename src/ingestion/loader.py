import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.models import Document as LegalDocument, DocumentType

class LegalDocumentLoader:    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_documents = []
        
    def load_single_pdf(self, file_path: Path) -> List[Document]:
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            for i, page in enumerate(pages):
                page.metadata.update({
                    'source': file_path.name,
                    'page': i + 1,
                    'total_pages': len(pages),
                    'file_path': str(file_path),
                    'file_hash': self._get_file_hash(file_path)
                })
            
            print(f"Loaded {file_path.name}: {len(pages)} pages")
            return pages
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return []
    
    def load_all_pdfs(self) -> List[Document]:
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            print(f"Add PDF files to: {self.data_dir.absolute()}")
            return []
        
        all_documents = []
        
        print(f"Found {len(pdf_files)} PDF files to load")
        
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            docs = self.load_single_pdf(pdf_file)
            all_documents.extend(docs)
        
        self.loaded_documents = all_documents
        print(f" Loaded {len(all_documents)} pages from {len(pdf_files)} files")
        
        return all_documents
    
    def _get_file_hash(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_document_stats(self) -> Dict[str, Any]:
        if not self.loaded_documents:
            return {"error": "No documents loaded"}
        
        total_chars = sum(len(doc.page_content) for doc in self.loaded_documents)
        
        files = {}
        for doc in self.loaded_documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in files:
                files[source] = {
                    'pages': 0,
                    'total_chars': 0
                }
            files[source]['pages'] += 1
            files[source]['total_chars'] += len(doc.page_content)
        
        return {
            'total_documents': len(self.loaded_documents),
            'total_characters': total_chars,
            'average_page_length': total_chars / len(self.loaded_documents),
            'files': files
        }
    
    def save_processed_documents(self, output_dir: str = "data/processed"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for doc in self.loaded_documents:
            source = doc.metadata.get('source', 'unknown').replace('.pdf', '')
            page = doc.metadata.get('page', 0)
            
            output_file = output_path / f"{source}_page_{page}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {doc.metadata.get('source')}\n")
                f.write(f"Page: {page}\n")
                f.write(f"{'='*50}\n\n")
                f.write(doc.page_content)
        
        print(f" Saved processed documents to {output_path}")