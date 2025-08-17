import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    def validate(self):
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        return True

@dataclass
class ChromaConfig:
    persist_directory: Path
    collection_name: str = "legal_docs"
    
    def validate(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        return True

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def validate(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return True

class Settings:
    
    def __init__(self):
        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        self.chroma = ChromaConfig(
            persist_directory=PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "legal_docs")
        )
        
        self.chunking = ChunkingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
    def validate_all(self):
        self.openai.validate()
        self.chroma.validate()
        self.chunking.validate()
        print("All configurations validated")
        return True
    
    def print_config(self):
        print("\n=== Configuration ===")
        print(f"OpenAI Model: {self.openai.model}")
        print(f"Embedding Model: {self.openai.embedding_model}")
        print(f"ChromaDB Directory: {self.chroma.persist_directory}")
        print(f"Collection Name: {self.chroma.collection_name}")
        print(f"Chunk Size: {self.chunking.chunk_size}")
        print(f"Chunk Overlap: {self.chunking.chunk_overlap}")
        print(f"Debug Mode: {self.debug}")
        print(f"Log Level: {self.log_level}")
        print("\n")

settings = Settings()