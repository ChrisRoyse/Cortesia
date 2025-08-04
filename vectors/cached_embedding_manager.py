#!/usr/bin/env python3
"""
Cached Embedding Manager - Singleton for Fast Model Loading
===========================================================

Ensures the embedding model is loaded ONCE and reused across all instances.
This solves the performance issue where ChromaDB was re-initializing the model
every time, causing 20-30 second delays.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import os
from chromadb.utils import embedding_functions
from pathlib import Path
import time

# Set cache directory for sentence transformers
cache_dir = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)


class CachedEmbeddingManager:
    """Singleton manager for cached embedding function"""
    
    _instance = None
    _embedding_function = None
    _load_time = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embedding_function(self):
        """Get or create the cached embedding function"""
        if self._embedding_function is None:
            print("Loading embedding model (this should only happen ONCE)...")
            start = time.time()
            
            # Create the embedding function - this will use the cached model
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            
            self._load_time = time.time() - start
            print(f"Embedding model loaded in {self._load_time:.3f}s")
        else:
            print(f"Using cached embedding model (originally loaded in {self._load_time:.3f}s)")
            
        return self._embedding_function


# Global singleton instance
_embedding_manager = CachedEmbeddingManager()


def get_cached_embedding_function():
    """Get the cached embedding function (loads only once)"""
    return _embedding_manager.get_embedding_function()