import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
from datetime import datetime
import hashlib

sys.path.append(str(Path(__file__).parent.parent.parent))

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from tqdm import tqdm

class LegalChromaStore:
    def __init__(self, 
                 persist_directory: str = "data/chroma_db",
                 collection_name: str = "legal_documents"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        self.collection = None
        
        print(f" Initialized ChromaDB")
        print(f"   Storage: {self.persist_directory}")
        print(f"   Collection: {collection_name}")
    
    def create_or_get_collection(self, 
                                 embedding_dimension: int = 1536,
                                 reset_if_exists: bool = False) -> chromadb.Collection:

        try:
            if reset_if_exists:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f" Deleted existing collection: {self.collection_name}")
                except:
                    pass
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": embedding_dimension}
            )
            
            count = self.collection.count()
            if count > 0:
                print(f"Retrieved existing collection with {count} documents")
            else:
                print(f"Created new empty collection")
            
            return self.collection
            
        except Exception as e:
            print(f"Error creating/getting collection: {e}")
            raise
    
    def add_embeddings(self, 
                      embeddings: List[List[float]], 
                      metadata: List[Dict],
                      batch_size: int = 100) -> int:
        if not self.collection:
            self.create_or_get_collection()
        
        if not self.collection:
            print("Failed to create/get collection")
            return 0
        
        if not embeddings or not metadata:
            print("No embeddings to add")
            return 0
        
        print(f"\nAdding {len(embeddings)} embeddings to ChromaDB...")
        
        total_added = 0
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding batches"):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            
            ids = []
            documents = []
            metadatas = []
            
            for j, meta in enumerate(batch_metadata):
                doc_id = f"{meta.get('source', 'unknown')}_{meta.get('page', 0)}_{meta.get('chunk_index', j+i)}"
                ids.append(doc_id)
                
                documents.append(meta.get('text', ''))
                
                clean_meta = {k: v for k, v in meta.items() if k != 'text'}
                metadatas.append(clean_meta)
            
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=[emb for emb in batch_embeddings],
                    documents=documents,
                    metadatas=metadatas
                )
                total_added += len(batch_embeddings)
                
            except Exception as e:
                print(f"Error adding batch {i//batch_size + 1}: {e}")
        
        print(f"Successfully added {total_added} documents to ChromaDB")
        return total_added
    
    def load_and_store_embeddings(self, embeddings_file: str) -> int:
        embeddings_path = Path(embeddings_file)
        
        if not embeddings_path.exists():
            print(f"Embeddings file not found: {embeddings_file}")
            return 0
        
        print(f"\n Loading embeddings from: {embeddings_file}")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data.get('embeddings', [])
        metadata = data.get('metadata', [])
        
        print(f"   Loaded {len(embeddings)} embeddings")
        
        return self.add_embeddings(embeddings, metadata)
    
    def search(self, 
              query_embedding: List[float],
              n_results: int = 5,
              filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.collection:
            print("  No collection available for search")
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}
        
        try:
            if filter_metadata:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=filter_metadata
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}
    
    def search_by_text(self,
                      query_text: str,
                      embedder,
                      n_results: int = 5,
                      filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        query_embedding = embedder.embeddings_model.embed_query(query_text)
        
        results = self.search(query_embedding, n_results, filter_metadata)
        
        formatted_results = []
        documents = results.get("documents", [])
        distances = results.get("distances", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        
        for i in range(len(documents)):
            formatted_results.append({
                "document": documents[i] if i < len(documents) else "",
                "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                "distance": distances[i] if i < len(distances) else 1.0,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "id": ids[i] if i < len(ids) else ""
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {"error": "No collection initialized"}
        
        try:
            count = self.collection.count()
            
            sample = self.collection.get(limit=min(100, count))
            
            sources = {}
            if sample and sample.get("metadatas"):
                metadatas = sample.get("metadatas")
                if metadatas:
                    for meta in metadatas:
                        source = meta.get("source", "unknown") if meta else "unknown"
                        sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "sample_sources": sources
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        if self.collection_name:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
                self.collection = None
            except Exception as e:
                print(f"Error deleting collection: {e}")
    
    def list_collections(self) -> List[str]:
        collections = self.client.list_collections()
        return [col.name for col in collections]


def main():    
    print("ChromaDB Vector Store Setup")
    
    chroma_store = LegalChromaStore()
    
    chroma_store.create_or_get_collection(reset_if_exists=True)
    
    embeddings_dir = Path("data/embeddings")
    embedding_files = list(embeddings_dir.glob("embeddings_*.pkl"))
    
    if not embedding_files:
        print("No embedding files found in data/embeddings/")
        return
    
    latest_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
    print(f"\n Using embeddings file: {latest_file.name}")
    
    num_stored = chroma_store.load_and_store_embeddings(str(latest_file))
    
    if num_stored > 0:
        print("\n Testing search functionality...")
        
        from src.embeddings.embedder import LegalDocumentEmbedder
        embedder = LegalDocumentEmbedder()
        
        test_queries = [
            "project setup and structure",
            "real-time functionality",
            "error handling"
        ]
        
        for query in test_queries:
            print(f"\n Query: '{query}'")
            results = chroma_store.search_by_text(query, embedder, n_results=2)
            
            for idx, result in enumerate(results, 1):
                print(f"\n  {idx}. Score: {result['score']:.4f}")
                print(f"     Source: {result['metadata'].get('source', 'unknown')}")
                print(f"     Page: {result['metadata'].get('page', 'unknown')}")
                print(f"     Preview: {result['document'][:100]}...")
    
    stats = chroma_store.get_collection_stats()
    print("Collection Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("ChromaDB setup complete!")


if __name__ == "__main__":
    main()