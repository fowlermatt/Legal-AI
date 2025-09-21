import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from dotenv import load_dotenv

from src.retrieval.retriever import RetrievedDocument

load_dotenv()


class ResponseFormat(Enum):
    STANDARD = "standard"
    DETAILED = "detailed"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    LEGAL_MEMO = "legal_memo"


@dataclass
class GeneratedResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    format: ResponseFormat
    metadata: Dict[str, Any]


class LegalResponseGenerator:
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        self.prompts = self._initialize_prompts()
        
        print(f"Initialized generator with model: {model_name}")
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        prompts = {}
        
        prompts["standard"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question accurately and concisely.
            
Context:
{context}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain enough information, say so."""
        )
        
        prompts["detailed"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

Provide a detailed answer that:
1. Directly addresses the question
2. Includes relevant details from the context
3. Explains any legal concepts mentioned
4. Notes any limitations or caveats

Answer based only on the provided context."""
        )
        
        prompts["summary"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, provide a brief summary answering the question.

Context:
{context}

Question: {question}

Provide a concise summary (2-3 sentences) that directly answers the question."""
        )
        
        prompts["bullet_points"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question using bullet points.

Context:
{context}

Question: {question}

Provide your answer as bullet points, with each point being a key piece of information."""
        )
        
        prompts["legal_memo"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, draft a brief legal memo addressing the question.

Context:
{context}

Question: {question}

Structure your response as:
ISSUE: [State the legal question]
BRIEF ANSWER: [Provide a direct answer]
ANALYSIS: [Explain the reasoning based on the context]
CONCLUSION: [Summarize the findings]"""
        )
        
        prompts["citation"] = PromptTemplate(
            input_variables=["answer", "sources"],
            template="""Add citations to the following answer using the provided sources.

Answer: {answer}

Sources:
{sources}

Rewrite the answer with inline citations [Source: page X] where appropriate."""
        )
        
        return prompts