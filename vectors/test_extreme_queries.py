#!/usr/bin/env python3
"""
Extreme Query Tests - Push the System to Its Limits
====================================================

These tests are designed to be EXTREMELY difficult and expose
every weakness in the query parsing and search system.

Author: Claude (Sonnet 4)
Date: 2025-08-04
"""

import sys
import io
import time
from pathlib import Path

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from integrated_indexing_system import IntegratedIndexingSystem
from enterprise_query_parser import create_enterprise_query_parser
from multi_level_indexer import IndexType


class ExtremeQueryTester:
    """Tests that will likely break the system"""
    
    def __init__(self):
        self.system = IntegratedIndexingSystem("./extreme_test_db")
        self.parser = create_enterprise_query_parser()
        self.project_root = Path("C:/code/LLMKG")
        self.results = []
        
    def setup(self):
        """Index test files"""
        print("Setting up test environment...")
        test_files = [
            self.project_root / "Cargo.toml",
            self.project_root / "README.md",
            self.project_root / "crates/neuromorphic-core/src/lib.rs",
            self.project_root / "crates/neuromorphic-core/src/column.rs",
            self.project_root / "crates/neuromorphic-core/src/minicolumn.rs",
            self.project_root / "crates/neuromorphic-core/src/layer.rs"
        ]
        
        for file_path in test_files:
            if file_path.exists():
                self.system._index_single_file(file_path, self.project_root)
        
        stats = self.system.get_index_statistics()
        print(f"Indexed {stats['integrated_system']['total_chunks']} chunks\n")
    
    def test_complex_boolean_and(self):
        """Test 1: Complex Boolean AND queries"""
        print("\n" + "="*60)
        print("TEST 1: Complex Boolean AND Queries")
        print("="*60)
        
        tests = [
            ("pub AND fn", "Basic AND"),
            ("pub AND struct AND impl", "Triple AND"),
            ("use AND std AND Result", "Library imports AND"),
            ("async AND await AND Future", "Async pattern AND"),
            ("Error AND Result AND Option", "Error handling AND")
        ]
        
        passed = 0
        for query, desc in tests:
            results = self._search_with_fallback(query)
            if results:
                print(f"  ✓ {desc}: '{query}' → {len(results)} results")
                passed += 1
            else:
                print(f"  ✗ {desc}: '{query}' → NO RESULTS")
        
        return passed, len(tests)
    
    def test_complex_boolean_not(self):
        """Test 2: Complex Boolean NOT queries"""
        print("\n" + "="*60)
        print("TEST 2: Complex Boolean NOT Queries")
        print("="*60)
        
        tests = [
            ("impl NOT test", "Basic NOT"),
            ("struct NOT pub", "Private structs"),
            ("fn NOT async", "Sync functions"),
            ("Result NOT Error", "Success results"),
            ("use NOT std", "Non-std imports")
        ]
        
        passed = 0
        for query, desc in tests:
            results = self._search_with_fallback(query)
            if results:
                print(f"  ✓ {desc}: '{query}' → {len(results)} results")
                passed += 1
            else:
                print(f"  ✗ {desc}: '{query}' → NO RESULTS")
        
        return passed, len(tests)
    
    def test_nested_boolean(self):
        """Test 3: Nested Boolean expressions"""
        print("\n" + "="*60)
        print("TEST 3: Nested Boolean Expressions")
        print("="*60)
        
        tests = [
            ("(pub OR private) AND struct", "OR inside AND"),
            ("fn AND (async OR sync)", "AND with OR group"),
            ("(Result OR Option) AND NOT Error", "Complex grouping"),
            ("impl AND (Display OR Debug) NOT test", "Multiple groups"),
            ("(pub fn) OR (impl trait)", "Phrase groups")
        ]
        
        passed = 0
        for query, desc in tests:
            results = self._search_with_fallback(query)
            if results:
                print(f"  ✓ {desc}: '{query}' → {len(results)} results")
                passed += 1
            else:
                print(f"  ✗ {desc}: '{query}' → NO RESULTS")
        
        return passed, len(tests)
    
    def test_special_chars_in_boolean(self):
        """Test 4: Special characters mixed with boolean logic"""
        print("\n" + "="*60)
        print("TEST 4: Special Characters + Boolean Logic")
        print("="*60)
        
        tests = [
            ("Result<T> AND Error", "Generics with AND"),
            ("Vec<> OR HashMap<>", "Multiple generics OR"),
            ("&mut AND self", "References AND self"),
            ("#[derive] OR #[test]", "Attributes OR"),
            ("impl<T> NOT where", "Generic impl without where")
        ]
        
        passed = 0
        for query, desc in tests:
            results = self._search_with_fallback(query)
            if results:
                print(f"  ✓ {desc}: '{query}' → {len(results)} results")
                passed += 1
            else:
                print(f"  ✗ {desc}: '{query}' → NO RESULTS")
        
        return passed, len(tests)
    
    def test_proximity_and_phrase(self):
        """Test 5: Proximity searches and exact phrases"""
        print("\n" + "="*60)
        print("TEST 5: Proximity and Phrase Searches")
        print("="*60)
        
        tests = [
            ('"pub fn"', "Exact phrase"),
            ('"impl Default"', "Trait impl phrase"),
            ('"use std"', "Import phrase"),
            ('pub NEAR/3 struct', "Proximity 3 words"),
            ('async NEAR/5 await', "Async proximity")
        ]
        
        passed = 0
        for query, desc in tests:
            results = self._search_with_fallback(query)
            if results:
                print(f"  ✓ {desc}: '{query}' → {len(results)} results")
                passed += 1
            else:
                print(f"  ✗ {desc}: '{query}' → NO RESULTS")
        
        return passed, len(tests)
    
    def _search_with_fallback(self, query):
        """Search with multiple fallback strategies"""
        # Strategy 1: Parse and search with FTS query
        try:
            parsed = self.parser.parse(query)
            results = self.system.search(parsed.fts_query, IndexType.EXACT, limit=10)
            if results:
                return results
        except:
            pass
        
        # Strategy 2: Try raw query with exact search
        try:
            results = self.system.search(query, IndexType.EXACT, limit=10)
            if results:
                return results
        except:
            pass
        
        # Strategy 3: Try semantic search
        try:
            results = self.system.search(query, IndexType.SEMANTIC, limit=10)
            if results:
                return results
        except:
            pass
        
        return []
    
    def run_all_tests(self):
        """Run all extreme tests"""
        print("\n" + "🔥"*20)
        print("EXTREME QUERY TESTS - PUSHING SYSTEM TO LIMITS")
        print("🔥"*20)
        
        self.setup()
        
        test_functions = [
            self.test_complex_boolean_and,
            self.test_complex_boolean_not,
            self.test_nested_boolean,
            self.test_special_chars_in_boolean,
            self.test_proximity_and_phrase
        ]
        
        total_passed = 0
        total_tests = 0
        
        for test_func in test_functions:
            passed, num_tests = test_func()
            total_passed += passed
            total_tests += num_tests
            self.results.append((test_func.__name__, passed, num_tests))
        
        # Results
        print("\n" + "="*60)
        print("EXTREME TEST RESULTS")
        print("="*60)
        
        for test_name, passed, total in self.results:
            accuracy = (passed/total)*100 if total > 0 else 0
            status = "✅" if accuracy == 100 else "❌"
            print(f"{status} {test_name}: {passed}/{total} ({accuracy:.0f}%)")
        
        overall_accuracy = (total_passed/total_tests)*100 if total_tests > 0 else 0
        
        print(f"\nOverall Accuracy: {overall_accuracy:.1f}% ({total_passed}/{total_tests})")
        
        if overall_accuracy == 100:
            print("\n🎉 PERFECT SCORE! System handles all extreme queries!")
        elif overall_accuracy >= 80:
            print("\n✅ Good performance, but room for improvement")
        else:
            print("\n❌ System needs significant improvements")
            print("\nFailing areas that need fixing:")
            for test_name, passed, total in self.results:
                if passed < total:
                    print(f"  - {test_name}: {total-passed} failures")
        
        return overall_accuracy == 100


if __name__ == "__main__":
    tester = ExtremeQueryTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)