#!/usr/bin/env python3
"""Simple direct test of indexing simulation files"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from indexer_universal import UniversalIndexer, UniversalCodeParser
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# Setup paths
sim1_path = Path("1_multi_language")
db_path = Path("test_db_sim1")

# Clean database
if db_path.exists():
    shutil.rmtree(db_path)

# Create parser and embeddings
parser = UniversalCodeParser()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Testing Universal RAG Indexer on Simulation 1")
print("=" * 60)

# Process each file directly
all_documents = []
languages_detected = {}
chunk_types = {}

for file_path in sim1_path.rglob('*'):
    if not file_path.is_file():
        continue
        
    print(f"\nProcessing: {file_path.relative_to(sim1_path)}")
    
    try:
        # Detect language
        lang = parser.detect_language(file_path)
        languages_detected[lang] = languages_detected.get(lang, 0) + 1
        print(f"  Language detected: {lang}")
        
        # Read content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract code blocks if it's code
        if file_path.suffix in ['.py', '.js', '.tsx', '.rs']:
            chunks = parser.extract_code_blocks(content, lang)
            print(f"  Code chunks extracted: {len(chunks)}")
            
            for chunk in chunks:
                chunk_types[chunk['type']] = chunk_types.get(chunk['type'], 0) + 1
                
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'source': str(file_path),
                        'language': lang,
                        'chunk_type': chunk['type'],
                        'chunk_name': chunk.get('name', 'unknown')
                    }
                )
                all_documents.append(doc)
        else:
            # Simple document for config/docs
            doc = Document(
                page_content=content[:1000],  # Limit size for testing
                metadata={
                    'source': str(file_path),
                    'file_type': file_path.suffix[1:],
                    'chunk_type': 'document'
                }
            )
            all_documents.append(doc)
            chunk_types['document'] = chunk_types.get('document', 0) + 1
            
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("INDEXING RESULTS")
print(f"{'='*60}")
print(f"Total documents created: {len(all_documents)}")
print(f"Languages detected: {languages_detected}")
print(f"Chunk types: {chunk_types}")

if all_documents:
    print("\nCreating vector database...")
    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    print(f"Database created at: {db_path}")
    
    # Test a query
    print("\nTesting query system...")
    results = vector_db.similarity_search("UserAuthentication", k=3)
    print(f"Query 'UserAuthentication' returned {len(results)} results")
    
    if results:
        print(f"Top result: {results[0].metadata.get('source', 'Unknown')}")
        print(f"Content preview: {results[0].page_content[:100]}...")

print("\nTest complete!")