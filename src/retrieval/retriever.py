import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vectorstore.chroma_store import LegalChromaStore
from src.embeddings.embedder import LegalDocumentEmbedder
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class RetrievalStrategy(Enum):
    SIMILARITY = "similarity"
    MMR = "maximum_marginal_relevance"
    THRESHOLD = "similarity_threshold"
    CONTEXTUAL = "contextual"


@dataclass
class RetrievalConfig:
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    top_k: int = 5
    similarity_threshold: float = 0.7
    mmr_lambda: float = 0.5
    rerank: bool = True
    max_context_length: int = 4000


@dataclass
class RetrievedDocument:
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    retrieval_method: str


class LegalDocumentRetriever:
    def __init__(self, 
                 chroma_store: Optional[LegalChromaStore] = None,
                 embedder: Optional[LegalDocumentEmbedder] = None,
                 config: Optional[RetrievalConfig] = None):
        self.chroma_store = chroma_store or LegalChromaStore()
        self.embedder = embedder or LegalDocumentEmbedder()
        self.config = config or RetrievalConfig()
        
        if not self.chroma_store.collection:
            self.chroma_store.create_or_get_collection()
        
        self.retrieval_history = []
        
        print(f"   Strategy: {self.config.strategy.value}")
        print(f"   Top-K: {self.config.top_k}")
        print(f"   Reranking: {'Enabled' if self.config.rerank else 'Disabled'}")
    
    def retrieve(self, 
                query: str,
                strategy: Optional[RetrievalStrategy] = None,
                top_k: Optional[int] = None,
                filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.top_k
        
        print(f"   Query: '{query[:50]}...' " if len(query) > 50 else f"   Query: '{query}'")
        print(f"   Strategy: {strategy.value}")
        
        if strategy == RetrievalStrategy.SIMILARITY:
            results = self._similarity_search(query, top_k * 2, filter_metadata)
        elif strategy == RetrievalStrategy.MMR:
            results = self._mmr_search(query, top_k, filter_metadata)
        elif strategy == RetrievalStrategy.THRESHOLD:
            results = self._threshold_search(query, filter_metadata)
        elif strategy == RetrievalStrategy.CONTEXTUAL:
            results = self._contextual_search(query, top_k, filter_metadata)
        else:
            results = self._similarity_search(query, top_k, filter_metadata)
        
        if self.config.rerank and len(results) > 0:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]
        
        results = self._trim_to_context_limit(results)
        
        self._store_retrieval_history(query, results)
        
        print(f" Retrieved {len(results)} documents")
        
        return results
    
    def _similarity_search(self, 
                          query: str, 
                          top_k: int,
                          filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        results = self.chroma_store.search_by_text(
            query_text=query,
            embedder=self.embedder,
            n_results=top_k,
            filter_metadata=filter_metadata
        )
        
        retrieved_docs = []
        for result in results:
            doc = RetrievedDocument(
                content=result["document"],
                metadata=result["metadata"],
                score=result["score"],
                chunk_id=result["id"],
                retrieval_method="similarity"
            )
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def _mmr_search(self, 
                   query: str, 
                   top_k: int,
                   filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        candidates = self._similarity_search(query, top_k * 3, filter_metadata)
        
        if not candidates:
            return []
        
        query_embedding = np.array(self.embedder.embeddings_model.embed_query(query))
        
        selected = []
        remaining = candidates.copy()
        
        if remaining:
            selected.append(remaining[0])
            remaining.pop(0)
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                relevance = candidate.score
                
                max_sim_to_selected = 0
                for sel_doc in selected:
                    content_sim = self._calculate_content_similarity(
                        candidate.content, 
                        sel_doc.content
                    )
                    max_sim_to_selected = max(max_sim_to_selected, content_sim)
                
                mmr_score = (self.config.mmr_lambda * relevance - 
                           (1 - self.config.mmr_lambda) * max_sim_to_selected)
                mmr_scores.append((mmr_score, candidate))
            
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                selected.append(mmr_scores[0][1])
                remaining.remove(mmr_scores[0][1])
        
        for doc in selected:
            doc.retrieval_method = "mmr"
        
        return selected
    
    def _threshold_search(self, 
                         query: str,
                         filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        candidates = self._similarity_search(query, 100, filter_metadata)
        
        results = [
            doc for doc in candidates 
            if doc.score >= self.config.similarity_threshold
        ]
        
        for doc in results:
            doc.retrieval_method = "threshold"
        
        return results
    
    def _contextual_search(self, 
                          query: str, 
                          top_k: int,
                          filter_metadata: Optional[Dict] = None) -> List[RetrievedDocument]:
        initial_results = self._similarity_search(query, top_k, filter_metadata)
        
        if not initial_results:
            return []
        
        expanded_results = []
        seen_ids = set()
        
        for doc in initial_results:
            if doc.chunk_id not in seen_ids:
                expanded_results.append(doc)
                seen_ids.add(doc.chunk_id)
            
            doc.retrieval_method = "contextual"
        
        return expanded_results
    
    def _rerank_results(self, 
                       query: str, 
                       results: List[RetrievedDocument],
                       top_k: int) -> List[RetrievedDocument]:
        print("Reranking results...")
        
        for doc in results:
            base_score = doc.score
            
            query_terms = query.lower().split()
            term_matches = sum(1 for term in query_terms if term in doc.content.lower())
            term_boost = term_matches / len(query_terms) if query_terms else 0
            
            ideal_length = 500
            length_penalty = 1.0 - abs(len(doc.content) - ideal_length) / ideal_length * 0.2
            length_penalty = max(0, length_penalty)
            
            doc.score = base_score * 0.7 + term_boost * 0.2 + length_penalty * 0.1
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def _trim_to_context_limit(self, 
                               results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        total_length = 0
        trimmed_results = []
        
        for doc in results:
            doc_length = len(doc.content)
            if total_length + doc_length <= self.config.max_context_length:
                trimmed_results.append(doc)
                total_length += doc_length
            else:
                remaining = self.config.max_context_length - total_length
                if remaining > 100:
                    doc.content = doc.content[:remaining] + "..."
                    trimmed_results.append(doc)
                break
        
        return trimmed_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _store_retrieval_history(self, query: str, results: List[RetrievedDocument]):
        """Store retrieval history for analysis."""
        self.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": len(results),
            "scores": [doc.score for doc in results],
            "strategy": self.config.strategy.value
        })
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        if not self.retrieval_history:
            return {"message": "No retrieval history yet"}
        
        scores = []
        for entry in self.retrieval_history:
            scores.extend(entry["scores"])
        
        return {
            "total_queries": len(self.retrieval_history),
            "average_results_per_query": np.mean([h["num_results"] for h in self.retrieval_history]),
            "average_score": np.mean(scores) if scores else 0,
            "score_std": np.std(scores) if scores else 0,
            "strategies_used": list(set(h["strategy"] for h in self.retrieval_history))
        }
    
    def multi_query_retrieve(self, queries: List[str], aggregate: bool = True) -> List[RetrievedDocument]:
        print(f"\n Multi-query retrieval with {len(queries)} queries")
        
        all_results = {}
        
        for query in queries:
            results = self.retrieve(query)
            for doc in results:
                if doc.chunk_id not in all_results:
                    all_results[doc.chunk_id] = doc
                else:
                    all_results[doc.chunk_id].score = (
                        all_results[doc.chunk_id].score + doc.score
                    ) / 2
        
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:self.config.top_k]


def main():
    
    
    print("Legal Document Retrieval System")
   
    
    chroma_store = LegalChromaStore()
    embedder = LegalDocumentEmbedder()
    
    configs = [
        RetrievalConfig(strategy=RetrievalStrategy.SIMILARITY, top_k=3),
        RetrievalConfig(strategy=RetrievalStrategy.MMR, top_k=3, mmr_lambda=0.7),
        RetrievalConfig(strategy=RetrievalStrategy.THRESHOLD, similarity_threshold=0.0)
    ]
    
    test_query = "project phases and implementation steps"
    
    for config in configs:
        print(f"\n{'='*40}")
        print(f"Testing {config.strategy.value} strategy")
        print(f"{'='*40}")
        
        retriever = LegalDocumentRetriever(
            chroma_store=chroma_store,
            embedder=embedder,
            config=config
        )
        
        results = retriever.retrieve(test_query)
        
        print(f"\nResults for: '{test_query}'")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Score: {doc.score:.4f}")
            print(f"   Method: {doc.retrieval_method}")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'unknown')}")
            print(f"   Preview: {doc.content[:100]}...")
    
    print("Testing Multi-Query Retrieval")
    
    retriever = LegalDocumentRetriever(chroma_store=chroma_store, embedder=embedder)
    
    multi_queries = [
        "project setup and structure",
        "implementation phases",
        "development steps"
    ]
    
    results = retriever.multi_query_retrieve(multi_queries)
    
    print(f"\nAggregated results from {len(multi_queries)} queries:")
    for i, doc in enumerate(results[:3], 1):
        print(f"\n{i}. Aggregated Score: {doc.score:.4f}")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Preview: {doc.content[:100]}...")
    
    print("Retrieval Statistics")
    stats = retriever.get_retrieval_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("Retrieval system demonstration complete!")


if __name__ == "__main__":
    main()