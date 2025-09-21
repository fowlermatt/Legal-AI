import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class QueryIntent(Enum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    ANALYSIS = "analysis"
    SEARCH = "search"
    EXTRACTION = "extraction"
    SUMMARY = "summary"
    QUESTION = "question"


@dataclass
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    keywords: List[str]
    entities: List[str]
    expanded_queries: List[str]
    metadata: Dict[str, Any]


class QueryProcessor:
    def __init__(self, use_llm: bool = True, model_name: str = "gpt-3.5-turbo"):
        self.use_llm = use_llm
        
        if use_llm:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0
            )
        else:
            self.llm = None
        
        self.legal_synonyms = {
            "contract": ["agreement", "covenant", "compact", "deal"],
            "clause": ["provision", "article", "section", "term"],
            "liability": ["responsibility", "obligation", "accountability"],
            "breach": ["violation", "infringement", "default", "non-compliance"],
            "party": ["participant", "signatory", "contractor"],
            "termination": ["cancellation", "dissolution", "expiration"],
            "indemnity": ["compensation", "reimbursement", "protection"],
            "warranty": ["guarantee", "assurance", "promise"],
            "negligence": ["carelessness", "failure of care", "tort"],
            "damages": ["compensation", "reparation", "remedy"]
        }
        
        self.query_templates = {
            QueryIntent.DEFINITION: [
                "What is {term}?",
                "Define {term}",
                "Meaning of {term}",
                "{term} definition"
            ],
            QueryIntent.PROCEDURE: [
                "How to {action}",
                "Steps for {action}",
                "Process of {action}",
                "{action} procedure"
            ],
            QueryIntent.COMPARISON: [
                "Difference between {term1} and {term2}",
                "{term1} vs {term2}",
                "Compare {term1} with {term2}"
            ]
        }
        
        print(f"   LLM: {'Enabled' if use_llm else 'Disabled'}")
    
    def process(self, query: str) -> ProcessedQuery:
        print(f"\n Processing query: '{query}'")
        
        cleaned_query = self._clean_query(query)
        
        intent = self._classify_intent(cleaned_query)
        
        keywords = self._extract_keywords(cleaned_query)
        entities = self._extract_entities(cleaned_query)
        
        expanded_queries = self._expand_query(cleaned_query, intent, keywords)
        
        processed = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            keywords=keywords,
            entities=entities,
            expanded_queries=expanded_queries,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "processing_method": "llm" if self.use_llm else "rule_based"
            }
        )
        
        print(f"   Intent: {intent.value}")
        print(f"   Keywords: {keywords[:5]}")
        print(f"   Expanded queries: {len(expanded_queries)}")
        
        return processed
    
    def _clean_query(self, query: str) -> str:
        cleaned = " ".join(query.split())
        cleaned = re.sub(r'[^\w\s\?]', '', cleaned)
        
        return cleaned
    
    def _classify_intent(self, query: str) -> QueryIntent:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "meaning of", "definition"]):
            return QueryIntent.DEFINITION
        elif any(word in query_lower for word in ["how to", "steps", "procedure", "process"]):
            return QueryIntent.PROCEDURE
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return QueryIntent.COMPARISON
        elif any(word in query_lower for word in ["analyze", "analysis", "evaluate"]):
            return QueryIntent.ANALYSIS
        elif any(word in query_lower for word in ["find", "search", "locate", "show"]):
            return QueryIntent.SEARCH
        elif any(word in query_lower for word in ["extract", "get", "retrieve"]):
            return QueryIntent.EXTRACTION
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return QueryIntent.SUMMARY
        else:
            return QueryIntent.QUESTION
    
    def _extract_keywords(self, query: str) -> List[str]:
        stopwords = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
            "in", "with", "to", "for", "of", "as", "from", "by", "that", "this",
            "what", "how", "when", "where", "why", "who", "can", "could", "would",
            "should", "shall", "will", "may", "might", "must", "do", "does", "did"
        }
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        for word in words:
            for legal_term, synonyms in self.legal_synonyms.items():
                if word in synonyms or word == legal_term:
                    keywords.append(legal_term)
        
        return list(set(keywords))
    
    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        
        words = query.split()
        for word in words:
            if word[0].isupper() and word.lower() not in ["i", "what", "how", "when", "where"]:
                entities.append(word)
        
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        return entities
    
    def _expand_query(self, query: str, intent: QueryIntent, keywords: List[str]) -> List[str]:
        expanded = [query]
        
        if self.use_llm:
            expanded.extend(self._llm_expand_query(query, intent))
        else:
            expanded.extend(self._rule_based_expand(query, intent, keywords))
        
        for keyword in keywords:
            if keyword in self.legal_synonyms:
                for synonym in self.legal_synonyms[keyword][:2]:
                    expanded.append(query.replace(keyword, synonym))
        
        seen = set()
        unique_expanded = []
        for q in expanded:
            if q not in seen:
                seen.add(q)
                unique_expanded.append(q)
        
        return unique_expanded[:5]
    
    def _rule_based_expand(self, query: str, intent: QueryIntent, keywords: List[str]) -> List[str]:
        expansions = []
        
        if intent == QueryIntent.DEFINITION:
            for keyword in keywords[:2]:
                expansions.append(f"What is {keyword}")
                expansions.append(f"Define {keyword} in legal context")
        elif intent == QueryIntent.PROCEDURE:
            expansions.append(f"Steps for {' '.join(keywords[:3])}")
            expansions.append(f"Process of {' '.join(keywords[:3])}")
        elif intent == QueryIntent.COMPARISON and len(keywords) >= 2:
            expansions.append(f"Difference between {keywords[0]} and {keywords[1]}")
            expansions.append(f"Compare {keywords[0]} with {keywords[1]}")
        
        return expansions
    
    def _llm_expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        if not self.llm:
            return []
        
        prompt = f"""Generate 3 alternative phrasings for this query that would help find relevant information:

Original query: "{query}"
Query intent: {intent.value}

Provide only the rephrased queries, one per line, without numbering or explanation.
Focus on legal document context."""
        
        try:
            response = self.llm.predict(prompt)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:3]
        except Exception as e:
            print(f"LLM expansion failed: {e}")
            return []
    
    def generate_hypothetical_answer(self, query: str) -> str:
        if not self.llm:
            return ""
        
        prompt = f"""Write a brief, factual answer to this question as it might appear in a legal document:

Question: {query}

Provide only the answer text, as if it were an excerpt from a legal document."""
        
        try:
            hypothetical = self.llm.predict(prompt)
            return hypothetical
        except Exception as e:
            print(f"Failed to generate hypothetical answer: {e}")
            return ""
    
    def decompose_complex_query(self, query: str) -> List[str]:
        sub_queries = []
        
        if any(conj in query.lower() for conj in [" and ", " as well as ", " also ", " plus "]):
            parts = re.split(r'\s+and\s+|\s+as well as\s+|\s+also\s+|\s+plus\s+', query, flags=re.IGNORECASE)
            sub_queries.extend(parts)
        
        if query.count("?") > 1:
            parts = [q.strip() + "?" for q in query.split("?") if q.strip()]
            sub_queries.extend(parts)
        
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def get_query_suggestions(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        suggestions = []
        
        processed = self.process(query)
        
        if processed.intent == QueryIntent.DEFINITION:
            main_keyword = processed.keywords[0] if processed.keywords else "this"
            suggestions.extend([
                f"What are the legal implications of {main_keyword}?",
                f"Examples of {main_keyword} in contracts",
                f"How is {main_keyword} enforced?"
            ])
        elif processed.intent == QueryIntent.PROCEDURE:
            suggestions.extend([
                "What are the requirements?",
                "What are the potential risks?",
                "What is the timeline?"
            ])
        
        return suggestions[:3]


def main():
    test_queries = [
        "What is a breach of contract?",
        "How to terminate a lease agreement",
        "Compare warranty vs indemnity clauses",
        "Find all references to liability limitations",
        "breach contract damages compensation",
        "Analyze the termination clause in employment contracts"
    ]
    
    processor_rules = QueryProcessor(use_llm=False)
    
    for query in test_queries[:3]:
        processed = processor_rules.process(query)
        
        print(f"\nOriginal: {query}")
        print(f"Intent: {processed.intent.value}")
        print(f"Keywords: {', '.join(processed.keywords)}")
        print(f"Expanded queries:")
        for i, exp in enumerate(processed.expanded_queries, 1):
            print(f"  {i}. {exp}")
    
    try:
        processor_llm = QueryProcessor(use_llm=True)
        
        test_query = "What are the key differences between indemnification and limitation of liability clauses?"
        processed = processor_llm.process(test_query)
        
        print(f"\nOriginal: {test_query}")
        print(f"Intent: {processed.intent.value}")
        print(f"Keywords: {', '.join(processed.keywords)}")
        print(f"Expanded queries:")
        for i, exp in enumerate(processed.expanded_queries, 1):
            print(f"  {i}. {exp}")
        
        print("\n🔮 Hypothetical Answer (HyDE):")
        hypothetical = processor_llm.generate_hypothetical_answer(test_query)
        print(f"  {hypothetical[:200]}..." if len(hypothetical) > 200 else f"  {hypothetical}")
        
    except Exception as e:
        print(f"LLM processing skipped (API issue): {e}")
    
    complex_query = "What is a breach of contract and how can I claim damages?"
    sub_queries = processor_rules.decompose_complex_query(complex_query)
    print(f"Complex query: {complex_query}")
    print("Sub-queries:")
    for i, sub in enumerate(sub_queries, 1):
        print(f"  {i}. {sub}")
    
    suggestions = processor_rules.get_query_suggestions("What is a warranty?")
    print("Follow-up suggestions:")
    for i, sug in enumerate(suggestions, 1):
        print(f"  {i}. {sug}")
    
if __name__ == "__main__":
    main()