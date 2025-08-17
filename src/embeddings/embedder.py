import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import pickle
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class LegalDocumentEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.embeddings_model = OpenAIEmbeddings(
            model=model_name
        )
        self.embeddings_cache = {}
        self.metadata_cache = []
        
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Initialized embedder with model: {model_name}")
    
    def generate_embeddings(self, chunks: List[Document], batch_size: int = 100) -> Tuple[List[List[float]], List[Dict]]:
        if not chunks:
            print(" No chunks provided for embedding generation")
            return [], []
        
        print(f"\n Generating embeddings for {len(chunks)} chunks \n Model: {self.model_name} \n")
        
        all_embeddings = []
        all_metadata = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i + batch_size]
            
            texts = [chunk.page_content for chunk in batch]
            metadatas = [self._prepare_metadata(chunk, idx + i) for idx, chunk in enumerate(batch)]
            
            try:
                batch_embeddings = self.embeddings_model.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
                all_metadata.extend(metadatas)
                
            except Exception as e:
                print(f" Error generating embeddings for batch {i//batch_size + 1}: {e}")
                all_embeddings.extend([[0.0] * 1536] * len(batch))
                all_metadata.extend(metadatas)
        
        self.embeddings_cache = all_embeddings
        self.metadata_cache = all_metadata
        
        self._print_embedding_stats(all_embeddings, all_metadata)
        
        return all_embeddings, all_metadata
    
    def _prepare_metadata(self, chunk: Document, chunk_id: int) -> Dict[str, Any]:
        metadata = {
            'chunk_id': chunk_id,
            'text': chunk.page_content[:500],
            'text_length': len(chunk.page_content),
            'source': chunk.metadata.get('source', 'unknown'),
            'page': chunk.metadata.get('page', 0),
            'chunk_index': chunk.metadata.get('chunk_index', 0),
            'start_index': chunk.metadata.get('start_index', 0),
            'embedding_model': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        
        for key, value in chunk.metadata.items():
            if key not in metadata:
                metadata[key] = value
        
        return metadata
    
    def _print_embedding_stats(self, embeddings: List[List[float]], metadata: List[Dict]):
        if not embeddings:
            print("No embeddings generated")
            return
        
        embedding_dims = len(embeddings[0]) if embeddings else 0
        total_chunks = len(embeddings)
        
        norms = [np.linalg.norm(emb) for emb in embeddings]
        avg_norm = np.mean(norms)
        
        print(f"\n Embedding Statistics: \n Total embeddings: {total_chunks} \n Embedding dimensions: {embedding_dims} \n Average embedding norm: {avg_norm:.4f} \n Memory usage: ~{(total_chunks * embedding_dims * 4) / (1024*1024):.2f} MB")
        
        sources = {}
        for meta in metadata:
            source = meta.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n   Embeddings by source:")
        for source, count in sources.items():
            print(f"     - {source}: {count} chunks")
    
    def save_embeddings(self, filename: Optional[str] = None) -> str:
        if not self.embeddings_cache:
            print(" No embeddings to save")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embeddings_{self.model_name}_{timestamp}.pkl"
        
        filepath = self.embeddings_dir / filename
        
        data = {
            'embeddings': self.embeddings_cache,
            'metadata': self.metadata_cache,
            'model': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': self.metadata_cache,
                'model': self.model_name,
                'total_embeddings': len(self.embeddings_cache),
                'embedding_dimensions': len(self.embeddings_cache[0]) if self.embeddings_cache else 0,
                'created_at': data['created_at']
            }, f, indent=2)
        
        print(f" Saved embeddings to: {filepath}")
        print(f" Saved metadata to: {metadata_file}")
        
        return str(filepath)
    
    def load_embeddings(self, filepath: str) -> Tuple[List[List[float]], List[Dict]]:
        filepath_obj = Path(filepath)
        
        if not filepath_obj.exists():
            print(f" File not found: {filepath}")
            return [], []
        
        with open(filepath_obj, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings_cache = data['embeddings']
        self.metadata_cache = data['metadata']
        
        print(f" Loaded {len(self.embeddings_cache)} embeddings from: {filepath}")
        
        return self.embeddings_cache, self.metadata_cache
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        if not self.embeddings_cache:
            print(" No embeddings available for search")
            return []
        
        query_embedding = self.embeddings_model.embed_query(query)
        
        similarities = []
        for idx, embedding in enumerate(self.embeddings_cache):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((idx, similarity, self.metadata_cache[idx]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


def main():
    from src.ingestion.loader import LegalDocumentLoader
    from src.ingestion.chunker import LegalDocumentChunker
    
    print("Legal Document Embedding Pipeline")
    
    print("\n Loading documents...")
    loader = LegalDocumentLoader()
    documents = loader.load_all_pdfs()
    
    if not documents:
        print(" No documents found. Please add PDFs to data/raw/")
        return
    
    print("\nChunking documents...")
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_documents(documents)
    
    if not chunks:
        print("No chunks created")
        return
    
    print("\n Generating embeddings")
    embedder = LegalDocumentEmbedder()
    embeddings, metadata = embedder.generate_embeddings(chunks)
    
    print("\n Saving embeddings")
    filepath = embedder.save_embeddings()
    
    print("\n Testing similarity search")
    test_query = "legal contract terms"
    results = embedder.search_similar(test_query, top_k=3)
    
    print(f"\nTop 3 results for query: '{test_query}'")
    for idx, (chunk_id, score, meta) in enumerate(results, 1):
        print(f"\n{idx}. Chunk {chunk_id} (similarity: {score:.4f}) \n Source: {meta['source']}, Page: {meta['page']} \n Preview: {meta['text'][:100]}")
    
    print("Embedding pipeline ")
    print(f"   Embeddings saved to: {filepath}")


if __name__ == "__main__":
    main()