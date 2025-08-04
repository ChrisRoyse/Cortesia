#!/usr/bin/env python3
"""
Check and ensure embedding model is properly cached
"""

import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import time

# Check where models are cached
cache_dir = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
print(f"Model cache directory: {cache_dir}")
print(f"Cache exists: {cache_dir.exists()}")

if cache_dir.exists():
    # List cached models
    models = list(cache_dir.glob("*"))
    print(f"\nCached models found: {len(models)}")
    for model_path in models[:5]:
        print(f"  - {model_path.name}")

# Set environment variable to ensure model caching
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)

print("\nLoading model with explicit caching...")
start = time.time()

# Load the model - should use cache if available
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                            cache_folder=str(cache_dir),
                            device='cpu')

load_time = time.time() - start
print(f"Model loaded in {load_time:.3f}s")

# Test the model
test_text = "This is a test sentence."
print(f"\nTesting embedding generation...")
start = time.time()
embedding = model.encode(test_text)
encode_time = time.time() - start
print(f"Embedding generated in {encode_time:.3f}s")
print(f"Embedding shape: {embedding.shape}")

# Now test loading again (should be instant)
print("\nLoading model again (should be cached)...")
start = time.time()
model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                             cache_folder=str(cache_dir),
                             device='cpu')
load_time2 = time.time() - start
print(f"Second load time: {load_time2:.3f}s")

if load_time2 < 1.0:
    print("✅ Model is properly cached!")
else:
    print("⚠️ Model cache might not be working correctly")