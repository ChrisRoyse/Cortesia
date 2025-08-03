#!/usr/bin/env python3
"""
Comprehensive test suite for Universal RAG Indexing System
Tests against all three simulation environments
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our indexing system
from indexer_universal import UniversalIndexer
from query_universal import UniversalQuerier
from git_tracker import GitChangeTracker


class SimulationTester:
    """Test the indexing system against simulation environments"""
    
    def __init__(self):
        self.simulation_dir = Path(__file__).parent
        self.results = {
            'sim1_multi_language': {},
            'sim2_evolving': {},
            'sim3_edge_cases': {},
            'overall_score': 0
        }
        
    def setup_test_db(self, sim_name: str) -> Path:
        """Setup test database for simulation"""
        db_path = self.simulation_dir / f"test_db_{sim_name}"
        if db_path.exists():
            shutil.rmtree(db_path)
        return db_path
        
    def test_simulation_1_multi_language(self) -> Dict:
        """Test multi-language project indexing"""
        print("\n" + "="*60)
        print("TESTING SIMULATION 1: Multi-Language Project")
        print("="*60)
        
        sim_path = self.simulation_dir / "1_multi_language"
        db_path = self.setup_test_db("sim1")
        
        results = {
            'language_detection': {},
            'code_extraction': {},
            'chunk_counts': {},
            'query_results': {},
            'errors': [],
            'score': 0
        }
        
        # Index the simulation
        print("\n[1] Indexing multi-language project...")
        indexer = UniversalIndexer(
            root_dir=str(sim_path),
            db_dir=str(db_path),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Disable gitignore filtering for simulation tests
        indexer.gitignore_parser.patterns = []
        
        try:
            start_time = time.time()
            success = indexer.run()
            indexing_time = time.time() - start_time
            
            if not success:
                results['errors'].append("Indexing failed")
                return results
                
            # Analyze results
            stats = indexer.stats
            results['indexing_time'] = indexing_time
            results['total_files'] = stats['total_files']
            results['total_chunks'] = stats['total_chunks']
            results['languages'] = dict(stats['languages'])
            results['chunk_types'] = dict(stats['chunk_types'])
            
            print(f"\n[2] Indexing Results:")
            print(f"  - Files processed: {stats['total_files']}")
            print(f"  - Chunks created: {stats['total_chunks']}")
            print(f"  - Languages detected: {list(stats['languages'].keys())}")
            print(f"  - Time taken: {indexing_time:.2f}s")
            
            # Test language detection accuracy
            expected_languages = {
                'python': 2,      # app.py, models.py
                'javascript': 1,  # app.js
                'typescript': 1,  # components.tsx
                'rust': 1,        # main.rs
                'unknown': 0      # Should detect all languages
            }
            
            correct_detections = 0
            for lang, expected_count in expected_languages.items():
                actual_count = stats['languages'].get(lang, 0)
                if lang != 'unknown':
                    if actual_count == expected_count:
                        correct_detections += 1
                        print(f"  [OK] {lang}: {actual_count} files (expected {expected_count})")
                    else:
                        print(f"  [FAIL] {lang}: {actual_count} files (expected {expected_count})")
                        results['errors'].append(f"Language detection mismatch for {lang}")
                        
            results['language_detection']['accuracy'] = correct_detections / len(expected_languages)
            
            # Test code extraction
            print("\n[3] Testing Code Extraction:")
            expected_extractions = {
                'function': 20,  # Minimum expected functions
                'class': 10,     # Minimum expected classes
                'method': 15,    # Minimum expected methods
            }
            
            extraction_score = 0
            for chunk_type, min_expected in expected_extractions.items():
                actual = stats['chunk_types'].get(chunk_type, 0)
                if actual >= min_expected:
                    extraction_score += 1
                    print(f"  [OK] {chunk_type}: {actual} (>= {min_expected})")
                else:
                    print(f"  [FAIL] {chunk_type}: {actual} (expected >= {min_expected})")
                    results['errors'].append(f"Insufficient {chunk_type} extraction")
                    
            results['code_extraction']['score'] = extraction_score / len(expected_extractions)
            
            # Test query functionality
            print("\n[4] Testing Query System:")
            querier = UniversalQuerier(
                db_dir=str(db_path),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            querier.initialize()
            
            test_queries = [
                ("UserAuthentication class", "code", 1),
                ("database configuration", "config", 1),
                ("calculate_similarity function", "code", 1),
                ("React component", "code", 1),
                ("installation instructions", "docs", 1),
            ]
            
            query_success = 0
            for query, filter_type, min_results in test_queries:
                results_found = querier.search(
                    query=query,
                    k=5,
                    filter_type=filter_type if filter_type != "all" else None,
                    rerank=True
                )
                
                if len(results_found) >= min_results:
                    query_success += 1
                    print(f"  [OK] Query '{query}': {len(results_found)} results")
                else:
                    print(f"  [FAIL] Query '{query}': {len(results_found)} results (expected >= {min_results})")
                    results['errors'].append(f"Query failed: {query}")
                    
            results['query_results']['accuracy'] = query_success / len(test_queries)
            
            # Calculate overall score
            scores = [
                results['language_detection']['accuracy'] * 30,  # 30% weight
                results['code_extraction']['score'] * 30,        # 30% weight
                results['query_results']['accuracy'] * 40,       # 40% weight
            ]
            results['score'] = sum(scores)
            
            print(f"\n[5] Simulation 1 Score: {results['score']:.1f}/100")
            
        except Exception as e:
            results['errors'].append(f"Test failed: {str(e)}")
            print(f"ERROR: {e}")
            
        finally:
            # Cleanup
            if hasattr(indexer, 'cleanup'):
                indexer.cleanup()
            if 'querier' in locals():
                querier.cleanup()
                
        return results
        
    def test_simulation_2_evolving(self) -> Dict:
        """Test evolving codebase with git tracking"""
        print("\n" + "="*60)
        print("TESTING SIMULATION 2: Evolving Codebase")
        print("="*60)
        
        sim_path = self.simulation_dir / "2_evolving_codebase"
        db_path = self.setup_test_db("sim2")
        
        results = {
            'initial_index': {},
            'incremental_updates': {},
            'git_tracking': {},
            'cache_performance': {},
            'errors': [],
            'score': 0
        }
        
        try:
            # Test initial indexing
            print("\n[1] Initial indexing of evolving codebase...")
            indexer = UniversalIndexer(
                root_dir=str(sim_path),
                db_dir=str(db_path),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Disable gitignore filtering for simulation tests
            indexer.gitignore_parser.patterns = []
            
            start_time = time.time()
            success = indexer.run()
            initial_time = time.time() - start_time
            
            if not success:
                results['errors'].append("Initial indexing failed")
                return results
                
            initial_stats = dict(indexer.stats)
            results['initial_index'] = {
                'files': initial_stats['total_files'],
                'chunks': initial_stats['total_chunks'],
                'time': initial_time
            }
            
            print(f"  - Initial files: {initial_stats['total_files']}")
            print(f"  - Initial chunks: {initial_stats['total_chunks']}")
            print(f"  - Time: {initial_time:.2f}s")
            
            # Test git tracking
            print("\n[2] Testing Git Change Detection...")
            tracker = GitChangeTracker(sim_path)
            
            # Check if it's a git repo
            current_commit = tracker.get_current_commit()
            if current_commit:
                print(f"  [OK] Git repository detected")
                print(f"  - Current commit: {current_commit[:8]}")
                results['git_tracking']['detected'] = True
            else:
                print(f"  [FAIL] Git repository not detected")
                results['git_tracking']['detected'] = False
                
            # Get changed files
            changes = tracker.get_changed_files()
            print(f"  - Changed files detected: {len(changes)}")
            results['git_tracking']['changes'] = len(changes)
            
            # Simulate file modification
            print("\n[3] Simulating file changes...")
            test_file = sim_path / "calculator.py"
            if test_file.exists():
                original_content = test_file.read_text()
                # Add a comment to trigger change
                modified_content = original_content + "\n# Test modification"
                test_file.write_text(modified_content)
                
                # Check if change is detected
                new_changes = tracker.get_changed_files()
                if len(new_changes) > len(changes):
                    print(f"  [OK] File modification detected")
                    results['incremental_updates']['detection'] = True
                else:
                    print(f"  [FAIL] File modification not detected")
                    results['incremental_updates']['detection'] = False
                    
                # Restore original content
                test_file.write_text(original_content)
            
            # Test query caching
            print("\n[4] Testing Query Cache Performance...")
            querier = UniversalQuerier(
                db_dir=str(db_path),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            querier.initialize()
            
            test_query = "calculate function"
            
            # First query (cache miss)
            start_time = time.time()
            results1 = querier.search(test_query, k=5)
            first_query_time = time.time() - start_time
            
            # Second query (cache hit)
            start_time = time.time()
            results2 = querier.search(test_query, k=5)
            cached_query_time = time.time() - start_time
            
            cache_speedup = first_query_time / cached_query_time if cached_query_time > 0 else 1
            
            print(f"  - First query: {first_query_time*1000:.1f}ms")
            print(f"  - Cached query: {cached_query_time*1000:.1f}ms")
            print(f"  - Cache speedup: {cache_speedup:.1f}x")
            
            results['cache_performance'] = {
                'first_query_ms': first_query_time * 1000,
                'cached_query_ms': cached_query_time * 1000,
                'speedup': cache_speedup
            }
            
            if cache_speedup > 2:
                print(f"  [OK] Cache performance excellent")
                cache_score = 100
            elif cache_speedup > 1.5:
                print(f"  [OK] Cache performance good")
                cache_score = 75
            else:
                print(f"  [FAIL] Cache performance needs improvement")
                cache_score = 50
                
            # Calculate score
            scores = [
                (1 if results['git_tracking'].get('detected', False) else 0) * 25,
                (1 if results['incremental_updates'].get('detection', False) else 0) * 25,
                (cache_score / 100) * 50
            ]
            results['score'] = sum(scores)
            
            print(f"\n[5] Simulation 2 Score: {results['score']:.1f}/100")
            
        except Exception as e:
            results['errors'].append(f"Test failed: {str(e)}")
            print(f"ERROR: {e}")
            
        finally:
            # Cleanup
            if 'indexer' in locals() and hasattr(indexer, 'cleanup'):
                indexer.cleanup()
            if 'querier' in locals():
                querier.cleanup()
                
        return results
        
    def test_simulation_3_edge_cases(self) -> Dict:
        """Test edge cases and error handling"""
        print("\n" + "="*60)
        print("TESTING SIMULATION 3: Edge Cases")
        print("="*60)
        
        sim_path = self.simulation_dir / "3_edge_cases"
        db_path = self.setup_test_db("sim3")
        
        results = {
            'error_handling': {},
            'unicode_support': {},
            'deduplication': {},
            'performance': {},
            'errors': [],
            'score': 0
        }
        
        try:
            # Index edge case files
            print("\n[1] Indexing edge case files...")
            indexer = UniversalIndexer(
                root_dir=str(sim_path),
                db_dir=str(db_path),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Disable gitignore filtering for simulation tests
            indexer.gitignore_parser.patterns = []
            
            start_time = time.time()
            success = indexer.run()
            indexing_time = time.time() - start_time
            
            stats = indexer.stats
            
            print(f"  - Files processed: {stats['total_files']}")
            print(f"  - Files failed: {len(stats.get('files_failed', []))}")
            print(f"  - Errors encountered: {len(stats.get('errors', []))}")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Time: {indexing_time:.2f}s")
            
            # Test error handling
            print("\n[2] Testing Error Handling:")
            error_handling_score = 0
            
            # Check if syntax error file was processed
            if stats['total_files'] > 0:
                print(f"  [OK] Continued processing despite errors")
                error_handling_score += 50
            else:
                print(f"  [FAIL] Failed to process any files")
                
            # Check if some chunks were created despite errors
            if stats['total_chunks'] > 0:
                print(f"  [OK] Created chunks from valid content")
                error_handling_score += 50
            else:
                print(f"  [FAIL] No chunks created")
                
            results['error_handling']['score'] = error_handling_score
            
            # Test Unicode support
            print("\n[3] Testing Unicode Support:")
            querier = UniversalQuerier(
                db_dir=str(db_path),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            querier.initialize()
            
            # Search for Unicode content
            unicode_queries = [
                "こんにちは",  # Japanese
                "中文内容",     # Chinese
                "مرحبا",       # Arabic
                "emoji",       # Emoji content
            ]
            
            unicode_found = 0
            for query in unicode_queries:
                try:
                    results_found = querier.search(query, k=3)
                    if results_found:
                        unicode_found += 1
                        print(f"  [OK] Found content for: {query}")
                except:
                    print(f"  [FAIL] Failed to search: {query}")
                    
            results['unicode_support']['score'] = (unicode_found / len(unicode_queries)) * 100
            
            # Test deduplication
            print("\n[4] Testing Deduplication:")
            
            # Check for duplicate detection in chunks
            duplicate_file_processed = False
            for chunk_type in stats.get('chunk_types', {}).keys():
                if 'duplicate' in chunk_type.lower() or stats['total_chunks'] < stats['total_files'] * 5:
                    duplicate_file_processed = True
                    break
                    
            if duplicate_file_processed:
                print(f"  [OK] Deduplication appears to be working")
                results['deduplication']['working'] = True
                dedup_score = 100
            else:
                print(f"  ? Deduplication status unclear")
                results['deduplication']['working'] = False
                dedup_score = 50
                
            # Test performance with large file
            print("\n[5] Testing Performance with Large Files:")
            
            # The massive_function.py should have been processed
            massive_file_chunks = 0
            for doc_type in stats.get('chunk_types', {}).keys():
                if 'fallback' in doc_type or 'code_block' in doc_type:
                    massive_file_chunks = stats['chunk_types'][doc_type]
                    break
                    
            if massive_file_chunks > 0:
                print(f"  [OK] Large file processed: {massive_file_chunks} chunks created")
                performance_score = 100
            else:
                print(f"  [FAIL] Large file processing unclear")
                performance_score = 50
                
            results['performance']['large_file_handling'] = performance_score
            
            # Calculate overall score
            scores = [
                results['error_handling']['score'] * 0.3,        # 30% weight
                results['unicode_support']['score'] * 0.2,       # 20% weight
                dedup_score * 0.25,                             # 25% weight
                performance_score * 0.25                        # 25% weight
            ]
            results['score'] = sum(scores) / 100 * 100
            
            print(f"\n[6] Simulation 3 Score: {results['score']:.1f}/100")
            
        except Exception as e:
            results['errors'].append(f"Test failed: {str(e)}")
            print(f"ERROR: {e}")
            
        finally:
            # Cleanup
            if 'indexer' in locals() and hasattr(indexer, 'cleanup'):
                indexer.cleanup()
            if 'querier' in locals():
                querier.cleanup()
                
        return results
        
    def run_all_tests(self) -> Dict:
        """Run all simulation tests"""
        print("="*60)
        print("UNIVERSAL RAG INDEXING SYSTEM - SIMULATION TESTS")
        print("="*60)
        
        # Run each simulation test
        self.results['sim1_multi_language'] = self.test_simulation_1_multi_language()
        self.results['sim2_evolving'] = self.test_simulation_2_evolving()
        self.results['sim3_edge_cases'] = self.test_simulation_3_edge_cases()
        
        # Calculate overall score
        sim_scores = [
            self.results['sim1_multi_language'].get('score', 0),
            self.results['sim2_evolving'].get('score', 0),
            self.results['sim3_edge_cases'].get('score', 0)
        ]
        
        self.results['overall_score'] = sum(sim_scores) / len(sim_scores)
        
        # Print summary
        print("\n" + "="*60)
        print("SIMULATION TEST SUMMARY")
        print("="*60)
        
        print("\nIndividual Scores:")
        print(f"  Simulation 1 (Multi-Language): {sim_scores[0]:.1f}/100")
        print(f"  Simulation 2 (Evolving Code):  {sim_scores[1]:.1f}/100")
        print(f"  Simulation 3 (Edge Cases):     {sim_scores[2]:.1f}/100")
        
        print(f"\nOVERALL SYSTEM SCORE: {self.results['overall_score']:.1f}/100")
        
        if self.results['overall_score'] >= 90:
            print("\n[EXCELLENT] System is working perfectly!")
        elif self.results['overall_score'] >= 75:
            print("\n[GOOD] System is working well with minor issues")
        elif self.results['overall_score'] >= 60:
            print("\n[ACCEPTABLE] System works but needs improvements")
        else:
            print("\n[NEEDS WORK] System has significant issues")
            
        # List any errors
        all_errors = []
        for sim_name, sim_results in self.results.items():
            if isinstance(sim_results, dict) and 'errors' in sim_results:
                all_errors.extend(sim_results['errors'])
                
        if all_errors:
            print("\nIssues Found:")
            for error in all_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
                
        # Save detailed results
        results_file = self.simulation_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        return self.results


def main():
    """Main test runner"""
    tester = SimulationTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['overall_score'] >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Needs improvement


if __name__ == "__main__":
    main()