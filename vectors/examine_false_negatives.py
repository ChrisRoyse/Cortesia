#!/usr/bin/env python3
"""
Examine specific false negatives to understand the ground truth issues
"""

import os
import json
from final_accuracy_validation import RealWorldValidator

def examine_specific_cases():
    """Examine specific files to debug false negatives"""
    
    # Initialize validator
    llmkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validator = RealWorldValidator(llmkg_root)
    
    # Test specific files that should have documentation
    test_files = [
        "vectors/smart_chunker_optimized.py",
        "crates/neuromorphic-core/src/neural_branch.rs",
        "crates/neuromorphic-core/src/ttfs_concept.rs"
    ]
    
    for file_path in test_files:
        full_path = os.path.join(llmkg_root, file_path)
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Examining: {file_path}")
        print(f"{'='*60}")
        
        # Get language from extension
        if file_path.endswith('.py'):
            language = 'python'
        elif file_path.endswith('.rs'):
            language = 'rust'
        else:
            language = 'unknown'
        
        # Analyze with our system
        items = validator.analyze_file_for_documentation(full_path, language)
        
        print(f"Found {len(items)} documentable items:")
        
        for i, item in enumerate(items[:5]):  # Show first 5 items
            print(f"\nItem {i+1}:")
            print(f"  Type: {item.item_type}")
            print(f"  Name: {item.item_name}")
            print(f"  Line: {item.line_number}")
            print(f"  Detected docs: {item.has_documentation}")
            print(f"  Confidence: {item.confidence:.2f}")
            print(f"  Doc lines: {item.documentation_lines}")
            
            # Check ground truth manually
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                ground_truth = validator._manual_check_for_documentation(lines, item.line_number, language)
                print(f"  Ground truth: {ground_truth}")
                
                # Show context around the item
                print(f"  Context:")
                start = max(0, item.line_number - 2)
                end = min(len(lines), item.line_number + 8)
                for j in range(start, end):
                    marker = ">>>" if j == item.line_number else "   "
                    print(f"    {marker} {j:3d}: {lines[j].rstrip()}")
                
                # Determine accuracy
                is_correct = (item.has_documentation == ground_truth)
                accuracy_status = "CORRECT" if is_correct else "INCORRECT"
                print(f"  Accuracy: {accuracy_status}")
                
                if not is_correct:
                    if item.has_documentation and not ground_truth:
                        print(f"    -> FALSE POSITIVE: detected docs where none exist")
                    elif not item.has_documentation and ground_truth:
                        print(f"    -> FALSE NEGATIVE: missed existing documentation")
                
            except Exception as e:
                print(f"  Error checking ground truth: {str(e)}")

if __name__ == "__main__":
    examine_specific_cases()