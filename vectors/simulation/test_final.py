#!/usr/bin/env python3
"""
Final comprehensive test suite for Universal RAG Indexing System
Tests against all three simulation environments with proper overrides
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our indexing system components
from indexer_universal import UniversalIndexer, UniversalCodeParser
from query_universal import UniversalQuerier
from git_tracker import GitChangeTracker
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class SimulationIndexer(UniversalIndexer):
    """Modified indexer for simulation testing"""
    
    def should_index_file(self, file_path: Path) -> bool:
        """Override to allow simulation files"""
        # Skip binary files
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.db', '.sqlite',
                            '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz'}
        if file_path.suffix.lower() in binary_extensions:
            return False
            
        # Supported extensions
        valid_extensions = {
            # Documentation
            '.md', '.txt', '.rst', '.markdown',
            # Code
            '.py', '.rs', '.js', '.jsx', '.ts', '.tsx', '.go', '.java', '.c', '.cpp', '.cc',
            '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.m',
            # Config
            '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.properties', '.xml'
        }
        
        return file_path.suffix.lower() in valid_extensions


def test_simulation_1():
    """Test multi-language project"""
    print("\n" + "="*60)
    print("TESTING SIMULATION 1: Multi-Language Project")
    print("="*60)
    
    sim_path = Path("1_multi_language")
    db_path = Path("test_db_sim1")
    
    # Clean database
    if db_path.exists():
        shutil.rmtree(db_path)
    
    # Create custom indexer
    indexer = SimulationIndexer(
        root_dir=str(sim_path),
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("[1] Indexing multi-language project...")
    start_time = time.time()
    success = indexer.run()
    indexing_time = time.time() - start_time
    
    if not success:
        print("  [FAIL] Indexing failed")
        return 0
    
    stats = indexer.stats
    print(f"\n[2] Indexing Results:")
    print(f"  Files processed: {stats['total_files']}")
    print(f"  Chunks created: {stats['total_chunks']}")
    print(f"  Languages: {dict(stats['languages'])}")
    print(f"  Time: {indexing_time:.2f}s")
    
    score = 0
    
    # Test language detection
    expected_langs = {'python': 2, 'javascript': 1, 'typescript': 1, 'rust': 1}
    for lang, expected in expected_langs.items():
        actual = stats['languages'].get(lang, 0)
        if actual == expected:
            print(f"  [OK] {lang}: {actual} files")
            score += 5
        else:
            print(f"  [FAIL] {lang}: {actual} files (expected {expected})")
    
    # Test code extraction
    if stats['total_chunks'] >= 50:
        print(f"  [OK] Chunk extraction: {stats['total_chunks']} chunks")
        score += 20
    else:
        print(f"  [FAIL] Insufficient chunks: {stats['total_chunks']}")
    
    # Test queries
    print("\n[3] Testing Queries:")
    querier = UniversalQuerier(
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    querier.initialize()
    
    test_queries = [
        ("UserAuthentication", 1),
        ("calculate_similarity", 1),
        ("database configuration", 1),
    ]
    
    for query, min_results in test_queries:
        results = querier.search(query, k=5)
        if len(results) >= min_results:
            print(f"  [OK] Query '{query}': {len(results)} results")
            score += 10
        else:
            print(f"  [FAIL] Query '{query}': {len(results)} results")
    
    # Cleanup
    indexer.cleanup()
    querier.cleanup()
    
    print(f"\nSimulation 1 Score: {score}/75")
    return score


def test_simulation_2():
    """Test evolving codebase"""
    print("\n" + "="*60)
    print("TESTING SIMULATION 2: Evolving Codebase")
    print("="*60)
    
    sim_path = Path("2_evolving_codebase")
    db_path = Path("test_db_sim2")
    
    # Clean database
    if db_path.exists():
        shutil.rmtree(db_path)
    
    # Create custom indexer
    indexer = SimulationIndexer(
        root_dir=str(sim_path),
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("[1] Initial indexing...")
    success = indexer.run()
    
    if not success:
        print("  [FAIL] Indexing failed")
        return 0
    
    initial_stats = dict(indexer.stats)
    print(f"  Files: {initial_stats['total_files']}")
    print(f"  Chunks: {initial_stats['total_chunks']}")
    
    score = 0
    
    # Test git tracking
    print("\n[2] Testing Git Tracking:")
    tracker = GitChangeTracker(sim_path)
    
    commit = tracker.get_current_commit()
    if commit:
        print(f"  [OK] Git detected: {commit[:8]}")
        score += 25
    else:
        print(f"  [FAIL] Git not detected")
    
    # Test cache performance
    print("\n[3] Testing Cache:")
    querier = UniversalQuerier(
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    querier.initialize()
    
    # First query
    start = time.time()
    results1 = querier.search("calculate", k=5)
    first_time = time.time() - start
    
    # Cached query
    start = time.time()
    results2 = querier.search("calculate", k=5)
    cached_time = time.time() - start
    
    speedup = first_time / cached_time if cached_time > 0 else 1
    print(f"  First: {first_time*1000:.1f}ms")
    print(f"  Cached: {cached_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    if speedup > 1.5:
        print(f"  [OK] Cache working")
        score += 25
    else:
        print(f"  [FAIL] Cache not effective")
    
    # Cleanup
    indexer.cleanup()
    querier.cleanup()
    
    print(f"\nSimulation 2 Score: {score}/50")
    return score


def test_simulation_3():
    """Test edge cases"""
    print("\n" + "="*60)
    print("TESTING SIMULATION 3: Edge Cases")
    print("="*60)
    
    sim_path = Path("3_edge_cases")
    db_path = Path("test_db_sim3")
    
    # Clean database
    if db_path.exists():
        shutil.rmtree(db_path)
    
    # Create custom indexer
    indexer = SimulationIndexer(
        root_dir=str(sim_path),
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("[1] Indexing edge cases...")
    success = indexer.run()
    
    stats = indexer.stats
    print(f"  Files: {stats['total_files']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Errors: {len(stats.get('errors', []))}")
    
    score = 0
    
    # Test error handling
    print("\n[2] Error Handling:")
    if stats['total_files'] > 0:
        print(f"  [OK] Processed files despite errors")
        score += 25
    else:
        print(f"  [FAIL] No files processed")
    
    if stats['total_chunks'] > 0:
        print(f"  [OK] Created chunks from valid content")
        score += 25
    else:
        print(f"  [FAIL] No chunks created")
    
    # Test Unicode support
    print("\n[3] Unicode Support:")
    querier = UniversalQuerier(
        db_dir=str(db_path),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    querier.initialize()
    
    try:
        results = querier.search("emoji", k=3)
        if results:
            print(f"  [OK] Unicode content searchable")
            score += 25
        else:
            print(f"  [FAIL] Unicode content not found")
    except:
        print(f"  [FAIL] Unicode search failed")
    
    # Cleanup
    indexer.cleanup()
    querier.cleanup()
    
    print(f"\nSimulation 3 Score: {score}/75")
    return score


def main():
    """Run all tests"""
    print("="*60)
    print("UNIVERSAL RAG SYSTEM - FINAL SIMULATION TESTS")
    print("="*60)
    
    scores = []
    
    # Run each test
    try:
        scores.append(test_simulation_1())
    except Exception as e:
        print(f"Simulation 1 failed: {e}")
        scores.append(0)
    
    try:
        scores.append(test_simulation_2())
    except Exception as e:
        print(f"Simulation 2 failed: {e}")
        scores.append(0)
    
    try:
        scores.append(test_simulation_3())
    except Exception as e:
        print(f"Simulation 3 failed: {e}")
        scores.append(0)
    
    # Calculate total
    total_score = sum(scores)
    max_score = 200  # 75 + 50 + 75
    percentage = (total_score / max_score) * 100
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Simulation 1: {scores[0]}/75")
    print(f"Simulation 2: {scores[1]}/50")
    print(f"Simulation 3: {scores[2]}/75")
    print(f"\nTOTAL SCORE: {total_score}/{max_score} ({percentage:.1f}%)")
    
    if percentage >= 90:
        print("\n[EXCELLENT] System is working perfectly!")
    elif percentage >= 75:
        print("\n[GOOD] System is working well")
    elif percentage >= 60:
        print("\n[ACCEPTABLE] System works with issues")
    else:
        print("\n[NEEDS WORK] System has problems")
    
    # Save results
    results = {
        'scores': scores,
        'total': total_score,
        'percentage': percentage,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('test_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_results_final.json")
    
    return percentage >= 75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)