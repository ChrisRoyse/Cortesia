#!/usr/bin/env python3
"""
Validation script to check if embedding dimension fixes are properly applied.
This script verifies that all test files use 96-dimensional embeddings consistently.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

class EmbeddingValidationReport:
    def __init__(self):
        self.issues = defaultdict(list)
        self.fixed_files = set()
        self.problem_files = set()
        self.total_files_checked = 0
        self.patterns_found = Counter()
        
    def add_issue(self, file_path: str, line_num: int, issue: str, line_content: str):
        self.issues[file_path].append({
            'line': line_num,
            'issue': issue,
            'content': line_content.strip()
        })
        self.problem_files.add(file_path)
        
    def add_fixed_usage(self, file_path: str):
        self.fixed_files.add(file_path)
        
    def get_summary(self) -> Dict:
        return {
            'total_files_checked': self.total_files_checked,
            'files_with_issues': len(self.problem_files),
            'files_properly_fixed': len(self.fixed_files),
            'issues_by_type': dict(self.patterns_found),
            'detailed_issues': dict(self.issues)
        }

def find_embedding_patterns(file_path: str, content: str) -> List[Tuple[int, str, str]]:
    """Find embedding dimension patterns in file content."""
    patterns = []
    lines = content.split('\n')
    
    # Patterns to check for
    dimension_patterns = [
        (r'vec!\[0\.0;\s*(\d+)\]', 'zero_vector'),
        (r'vec!\[0\.1;\s*(\d+)\]', 'point_one_vector'),  
        (r'vec!\[1\.0;\s*(\d+)\]', 'one_vector'),
        (r'vec!\[.*?;\s*(\d+)\]', 'general_vector'),
        (r'Vec::with_capacity\((\d+)\)', 'vec_capacity'),
        (r'embedding.*?(\d+)', 'embedding_comment'),
        (r'dimension.*?(\d+)', 'dimension_comment'),
        (r'(\d+).*?dimension', 'dimension_mention'),
    ]
    
    for line_num, line in enumerate(lines, 1):
        for pattern, pattern_type in dimension_patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                dimension = match.group(1)
                if dimension.isdigit():
                    patterns.append((line_num, pattern_type, line, int(dimension)))
    
    return patterns

def validate_brain_enhanced_graph_tests(report: EmbeddingValidationReport):
    """Validate brain_enhanced_graph test files."""
    test_dir = Path("tests/core")
    brain_test_files = [
        "test_brain_enhanced_graph_mod.rs",
        "test_brain_entity_manager.rs", 
        "test_brain_graph_core.rs",
        "test_brain_graph_types.rs",
        "test_brain_query_engine.rs",
        "test_brain_relationship_manager.rs",
        "test_brain_analytics.rs",
        "test_brain_advanced_ops.rs"
    ]
    
    print("[BRAIN] Validating brain_enhanced_graph test files...")
    
    for test_file in brain_test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            report.total_files_checked += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            patterns = find_embedding_patterns(str(file_path), content)
            has_issues = False
            has_96d = False
            
            for line_num, pattern_type, line, dimension in patterns:
                if dimension == 96:
                    has_96d = True
                elif dimension in [64, 128, 384]:
                    has_issues = True
                    report.add_issue(
                        str(file_path), 
                        line_num, 
                        f"Non-96D embedding found: {dimension}D", 
                        line
                    )
                    report.patterns_found[f"{dimension}D_vector"] += 1
            
            if has_96d and not has_issues:
                report.add_fixed_usage(str(file_path))
                print(f"  [OK] {test_file} - Using 96D embeddings correctly")
            elif has_issues:
                print(f"  [ERROR] {test_file} - Has dimension issues")
            else:
                print(f"  [WARN] {test_file} - No embedding patterns found")

def validate_cognitive_tests(report: EmbeddingValidationReport):
    """Validate cognitive test files."""
    test_dir = Path("tests/cognitive")
    
    print("\n[COGNITIVE] Validating cognitive test files...")
    
    cognitive_files = list(test_dir.glob("test_*.rs")) + list(test_dir.glob("*_tests.rs"))
    
    for file_path in cognitive_files:
        if file_path.is_file():
            report.total_files_checked += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            patterns = find_embedding_patterns(str(file_path), content)
            has_issues = False
            has_96d = False
            
            for line_num, pattern_type, line, dimension in patterns:
                if dimension == 96:
                    has_96d = True
                elif dimension in [64, 128, 384]:
                    has_issues = True
                    report.add_issue(
                        str(file_path), 
                        line_num, 
                        f"Non-96D embedding found: {dimension}D", 
                        line
                    )
                    report.patterns_found[f"{dimension}D_vector"] += 1
            
            if has_96d and not has_issues:
                report.add_fixed_usage(str(file_path))
                print(f"  [OK] {file_path.name} - Using 96D embeddings correctly")
            elif has_issues:
                print(f"  [ERROR] {file_path.name} - Has dimension issues")

def validate_core_tests(report: EmbeddingValidationReport):
    """Validate core test files."""
    test_dir = Path("tests/core")
    
    print("\n[CORE] Validating core test files...")
    
    core_files = [f for f in test_dir.glob("test_*.rs") if not f.name.startswith("test_brain")]
    
    for file_path in core_files:
        if file_path.is_file():
            report.total_files_checked += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            patterns = find_embedding_patterns(str(file_path), content)
            has_issues = False
            has_96d = False
            
            for line_num, pattern_type, line, dimension in patterns:
                if dimension == 96:
                    has_96d = True
                elif dimension in [64, 128, 384]:
                    has_issues = True
                    report.add_issue(
                        str(file_path), 
                        line_num, 
                        f"Non-96D embedding found: {dimension}D", 
                        line
                    )
                    report.patterns_found[f"{dimension}D_vector"] += 1
            
            if has_96d and not has_issues:
                report.add_fixed_usage(str(file_path))
                print(f"  [OK] {file_path.name} - Using 96D embeddings correctly")
            elif has_issues:
                print(f"  [ERROR] {file_path.name} - Has dimension issues")

def check_missing_methods(report: EmbeddingValidationReport):
    """Check for missing methods that should have been added."""
    print("\n[METHODS] Checking for missing method implementations...")
    
    # Check specific files for required methods
    methods_to_check = [
        ("src/core/brain_enhanced_graph/brain_graph_types.rs", ["with_entities", "entity_count"]),
        ("src/core/brain_enhanced_graph/brain_entity_manager.rs", ["batch_add_entities"]),
        ("src/core/brain_enhanced_graph/brain_query_engine.rs", ["similarity_search_with_filter"]),
    ]
    
    for file_path, required_methods in methods_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for method in required_methods:
                if f"fn {method}" not in content and f"pub fn {method}" not in content:
                    report.add_issue(
                        file_path,
                        0,
                        f"Missing method implementation: {method}",
                        f"Method '{method}' should be implemented"
                    )
                    report.patterns_found["missing_method"] += 1
                else:
                    print(f"  [OK] {method} found in {file_path}")

def validate_specific_problematic_files(report: EmbeddingValidationReport):
    """Check specific files that were identified as having issues."""
    print("\n[SPECIFIC] Validating specific problematic files...")
    
    problematic_files = [
        "tests/cognitive/test_divergent.rs",
        "tests/cognitive/test_lateral.rs", 
        "tests/cognitive/test_orchestrator.rs",
        "tests/core/test_graph_mod.rs",
        "tests/core/test_graph_entity_operations.rs",
        "tests/core/test_semantic_summary.rs"
    ]
    
    for file_path in problematic_files:
        if os.path.exists(file_path):
            report.total_files_checked += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for specific bad patterns
            bad_patterns = [
                r'vec!\[0\.1;\s*(64|128|384)\]',
                r'vec!\[0\.0;\s*(64|128|384)\]', 
                r'vec!\[1\.0;\s*(64|128|384)\]'
            ]
            
            has_issues = False
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern in bad_patterns:
                    match = re.search(pattern, line)
                    if match:
                        has_issues = True
                        dimension = match.group(1)
                        report.add_issue(
                            file_path,
                            line_num,
                            f"Hardcoded {dimension}D vector found",
                            line
                        )
                        report.patterns_found[f"hardcoded_{dimension}D"] += 1
            
            if not has_issues:
                print(f"  [OK] {file_path} - No hardcoded non-96D vectors found")
            else:
                print(f"  [ERROR] {file_path} - Still has hardcoded dimension issues")

def generate_report(report: EmbeddingValidationReport):
    """Generate a comprehensive validation report."""
    print("\n" + "="*80)
    print("EMBEDDING DIMENSION VALIDATION REPORT")
    print("="*80)
    
    summary = report.get_summary()
    
    print(f"\nSUMMARY:")
    print(f"  * Total files checked: {summary['total_files_checked']}")
    print(f"  * Files with issues: {summary['files_with_issues']}")
    print(f"  * Files properly fixed: {summary['files_properly_fixed']}")
    
    if summary['issues_by_type']:
        print(f"\nISSUES BY TYPE:")
        for issue_type, count in summary['issues_by_type'].items():
            print(f"  * {issue_type}: {count}")
    
    if summary['detailed_issues']:
        print(f"\nDETAILED ISSUES:")
        for file_path, issues in summary['detailed_issues'].items():
            print(f"\n  FILE: {file_path}:")
            for issue in issues:
                print(f"    Line {issue['line']}: {issue['issue']}")
                print(f"    Code: {issue['content']}")
    
    # Overall status
    if summary['files_with_issues'] == 0:
        print(f"\n[SUCCESS] All embedding dimensions have been properly fixed to 96D!")
        return True
    else:
        print(f"\n[WARNING] {summary['files_with_issues']} files still have dimension issues.")
        return False

def main():
    """Main validation function."""
    print("Starting embedding dimension validation...")
    print("This script will check for proper 96D embedding usage across all test files.\n")
    
    # Change to the project root directory
    if os.path.exists("Cargo.toml"):
        print("[OK] Found Cargo.toml - running from project root")
    else:
        print("[ERROR] Cargo.toml not found - please run from project root")
        return False
    
    report = EmbeddingValidationReport()
    
    # Run all validation checks
    validate_brain_enhanced_graph_tests(report)
    validate_cognitive_tests(report)
    validate_core_tests(report)
    check_missing_methods(report)
    validate_specific_problematic_files(report)
    
    # Generate final report
    success = generate_report(report)
    
    # Save detailed report to file
    with open("embedding_validation_report.json", 'w') as f:
        json.dump(report.get_summary(), f, indent=2)
    
    print(f"\n[SAVE] Detailed report saved to: embedding_validation_report.json")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)