#!/usr/bin/env python3
"""
Debug Boolean Query Processing
==============================

Let's see what's actually happening with boolean queries.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from integrated_indexing_system import IntegratedIndexingSystem
from enterprise_query_parser import create_enterprise_query_parser
from multi_level_indexer import IndexType


def debug_boolean_queries():
    """Debug what's happening with boolean queries"""
    
    print("DEBUGGING BOOLEAN QUERY PROCESSING")
    print("="*60)
    
    system = IntegratedIndexingSystem("./debug_db")
    parser = create_enterprise_query_parser()
    project_root = Path("C:/code/LLMKG")
    
    # Index a file
    lib_file = project_root / "crates/neuromorphic-core/src/lib.rs"
    if lib_file.exists():
        system._index_single_file(lib_file, project_root)
        print(f"Indexed {lib_file.name}\n")
    
    # Test queries
    test_queries = [
        "pub AND fn",
        "pub",
        "fn",
        "struct NOT test",
        "struct",
        "test"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Parse the query
        parsed = parser.parse(query)
        print(f"  Query Type: {parsed.query_type.value}")
        print(f"  FTS Query: {parsed.fts_query}")
        print(f"  Tokens: {parsed.tokens}")
        print(f"  Operators: {parsed.operators}")
        
        # Try exact search
        try:
            results = system.search(parsed.fts_query, IndexType.EXACT, limit=3)
            print(f"  Exact Results: {len(results)}")
            if results:
                for r in results[:1]:
                    preview = r.content[:100].replace('\n', ' ')
                    print(f"    → {preview}...")
        except Exception as e:
            print(f"  Exact Error: {e}")
        
        # Try semantic search
        try:
            results = system.search(query, IndexType.SEMANTIC, limit=3)
            print(f"  Semantic Results: {len(results)}")
        except Exception as e:
            print(f"  Semantic Error: {e}")


if __name__ == "__main__":
    debug_boolean_queries()