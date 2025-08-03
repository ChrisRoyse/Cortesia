#!/usr/bin/env python3
"""
Debug script to validate ground truth detection logic
"""

import os
from final_accuracy_validation import RealWorldValidator

def test_specific_cases():
    """Test specific cases where ground truth detection might be failing"""
    
    llmkg_root = os.path.dirname(os.path.abspath(__file__))
    llmkg_root = os.path.dirname(llmkg_root)  # Go up to LLMKG root
    
    validator = RealWorldValidator(llmkg_root)
    
    # Test cases that should have documentation
    test_cases = [
        ("vectors/benchmark_optimized_chunker.py", 49, "python"),  # BenchmarkResult class
        ("crates/neuromorphic-core/src/neural_branch.rs", 107, "rust"),  # NeuromorphicMemoryBranch
    ]
    
    for file_path, line_num, language in test_cases:
        full_path = os.path.join(llmkg_root, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Test ground truth detection
            has_docs = validator._manual_check_for_documentation(lines, line_num, language)
            
            print(f"\nFile: {file_path}")
            print(f"Line {line_num}: {lines[line_num].strip()}")
            print(f"Ground truth has docs: {has_docs}")
            
            # Show context
            print("Context:")
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 8)
            for i in range(start, end):
                marker = ">>>" if i == line_num else "   "
                print(f"{marker} {i:3d}: {lines[i].rstrip()}")
            
        except Exception as e:
            print(f"Error testing {file_path}: {str(e)}")

if __name__ == "__main__":
    test_specific_cases()