# LLMKG Vector Database System

## Overview
Advanced RAG (Retrieval-Augmented Generation) system implementing 2025 best practices for semantic search across documentation and codebase.

## Current System Files

### Core Components
- **`indexer_advanced.py`** - Advanced indexer with semantic chunking and AST parsing
- **`query_advanced.py`** - Query interface with reranking and type filtering
- **`chroma_db_advanced/`** - Active vector database (5,066 chunks)

### Utilities
- **`kill_processes.py`** - Process management utility for handling stuck processes

### Documentation
- **`ADVANCED_INDEXING_SUMMARY.md`** - Complete system implementation details
- **`RESOURCE_MANAGEMENT.md`** - Resource usage and troubleshooting guide

## Quick Start

### Querying the Database
```bash
# Basic search
python query_advanced.py -q "your search query"

# Search with more results
python query_advanced.py -q "temporal memory" -k 10

# Search only code
python query_advanced.py -q "impl SpikingColumn" -t code

# Search only documentation
python query_advanced.py -q "allocation engine" -t docs

# Show full content
python query_advanced.py -q "TTFS encoding" -f

# Show additional context
python query_advanced.py -q "neural pathways" -c

# Disable reranking for faster results
python query_advanced.py -q "quick search" --no-rerank
```

### Re-indexing (if needed)
```bash
# Re-index entire codebase and docs
python indexer_advanced.py --root-dir .. --db-dir chroma_db_advanced

# Index from a different root
python indexer_advanced.py --root-dir /path/to/project --db-dir my_database
```

### Process Management
```bash
# Check for stuck processes
python kill_processes.py
```

## System Statistics

### Current Database
- **Total Files Indexed**: 760
  - Documentation: 709 files
  - Code: 51 files (Python + Rust)
- **Total Chunks**: 5,066
  - Semantic chunks: 4,198
  - AST-parsed chunks: 868
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Query Speed**: <0.1 seconds

## Features

### Indexing
- **Semantic Chunking**: Groups text by meaning similarity
- **AST Code Parsing**: Extracts complete functions/classes
- **Context Preservation**: Includes imports and dependencies
- **Gitignore Awareness**: Respects project structure
- **Hybrid Strategies**: Different approaches for different content

### Querying
- **Semantic Search**: Deep understanding of concepts
- **Automatic Reranking**: Multi-factor relevance scoring
- **Type Filtering**: Search specific content types
- **Fast Performance**: Sub-100ms query times
- **Context Display**: Show additional metadata

## Architecture

```
vectors/
├── indexer_advanced.py      # Indexing engine
├── query_advanced.py         # Query interface
├── chroma_db_advanced/       # Vector database
│   ├── *.bin                 # Vector indices
│   ├── chroma.sqlite3        # Metadata store
│   └── metadata.json         # Index metadata
├── kill_processes.py         # Process manager
└── README.md                 # This file
```

## Best Practices Implemented

1. **Semantic Coherence**: Chunks maintain meaning
2. **Context Preservation**: Code includes necessary imports
3. **Optimal Sizing**: 200-800 char chunks for balance
4. **Performance**: Fast indexing and queries
5. **Resource Management**: Proper cleanup and monitoring
6. **Gitignore Respect**: Follows project conventions

## Troubleshooting

### High CPU Usage
```bash
python kill_processes.py
```

### Re-index After Major Changes
```bash
python indexer_advanced.py --root-dir .. --db-dir chroma_db_advanced
```

### Query Not Finding Expected Results
- Try different search terms
- Use natural language for concepts
- Use technical terms for code
- Check if content was indexed (not in .gitignore)

## Requirements

- Python 3.8+
- Dependencies:
  - langchain
  - langchain-huggingface
  - sentence-transformers
  - chromadb
  - scikit-learn
  - numpy
  - click

## Notes

- The system excludes the `vectors/` directory from indexing
- Respects `.gitignore` patterns
- Automatically cleans up resources after operations
- Optimized for both accuracy and speed