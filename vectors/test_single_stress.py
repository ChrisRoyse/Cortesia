#!/usr/bin/env python3
"""
Quick single stress test for debugging
"""

import sys
import io
from pathlib import Path

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from enterprise_indexing_system import create_enterprise_indexing_system

def test_special_characters():
    """Test special character handling in real queries"""
    print("=" * 70)
    print("STRESS TEST: Special Characters in Real Code")
    print("=" * 70)
    
    system = create_enterprise_indexing_system(
        db_path="./test_special_chars",
        use_redis=False  # Disable Redis for testing
    )
    
    # Index a single file first
    project_root = Path("C:/code/LLMKG")
    cargo_file = project_root / "Cargo.toml"
    
    if cargo_file.exists():
        print(f"\n1. Indexing {cargo_file.name}...")
        
        # Read the file content to verify it has what we're looking for
        with open(cargo_file, 'r') as f:
            content = f.read()
            print(f"   File size: {len(content)} bytes")
            if "[dependencies]" in content:
                print("   ✓ Contains [dependencies]")
            if "[workspace]" in content:
                print("   ✓ Contains [workspace]")
        
        # Index it
        try:
            result = system.base_system._index_single_file(cargo_file, project_root)
            print(f"   Indexing result: {result}")
            
            # Get stats
            stats = system.base_system.get_index_statistics()
            print(f"   Index stats: {stats}")
            
        except Exception as e:
            print(f"   Error indexing: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n2. Testing special character search...")
    
    # Test search for [dependencies]
    print("\n   Searching for: [dependencies]")
    try:
        results = system.search_enterprise("[dependencies]", limit=5)
        print(f"   Found: {len(results)} results")
        for r in results[:2]:
            print(f"     - {r.relative_path}: {r.context[:50]}...")
    except Exception as e:
        print(f"   Search error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")

if __name__ == "__main__":
    test_special_characters()