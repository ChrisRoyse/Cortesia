#!/usr/bin/env python3
"""
Examine the remaining 4 false negatives to reach 99% accuracy
"""

import os
from final_accuracy_validation import RealWorldValidator

def examine_remaining_false_negatives():
    """Examine the 4 remaining false negatives"""
    
    # Initialize validator
    llmkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validator = RealWorldValidator(llmkg_root)
    
    # The 4 false negatives from the report
    false_negatives = [
        ("crates/neuromorphic-core/src/error.rs", 76, "rust"),
        ("crates/neuromorphic-core/src/simd_backend.rs", 47, "rust"),
        ("crates/neuromorphic-core/src/ttfs_concept.rs", 107, "rust"),
        ("crates/neuromorphic-wasm/src/lib.rs", 7, "rust")
    ]
    
    for file_path, line_num, language in false_negatives:
        full_path = os.path.join(llmkg_root, file_path)
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"FALSE NEGATIVE: {file_path}:{line_num}")
        print(f"{'='*80}")
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if line_num >= len(lines):
                print(f"Line {line_num} out of range (file has {len(lines)} lines)")
                continue
            
            # Show context
            print(f"Context around line {line_num}:")
            start = max(0, line_num - 8)
            end = min(len(lines), line_num + 8)
            for i in range(start, end):
                marker = ">>>" if i == line_num else "   "
                print(f"{marker} {i:3d}: {lines[i].rstrip()}")
            
            # Test our detection
            items = validator.analyze_file_for_documentation(full_path, language)
            
            # Find the item at this line
            target_item = None
            for item in items:
                if item.line_number == line_num:
                    target_item = item
                    break
            
            if target_item:
                print(f"\nOur detection:")
                print(f"  Detected docs: {target_item.has_documentation}")
                print(f"  Confidence: {target_item.confidence:.2f}")
                print(f"  Doc lines: {target_item.documentation_lines}")
                print(f"  Item: {target_item.item_type} '{target_item.item_name}'")
                
                # Test ground truth
                ground_truth = validator._manual_check_for_documentation(lines, line_num, language)
                print(f"  Ground truth: {ground_truth}")
                
                # Analysis
                if not target_item.has_documentation and ground_truth:
                    print(f"  -> This is indeed a FALSE NEGATIVE")
                    print(f"  -> We missed documentation that actually exists")
                else:
                    print(f"  -> This might be a ground truth validation error")
            else:
                print(f"\nNo item found at line {line_num} in our analysis")
                
        except Exception as e:
            print(f"Error examining {file_path}: {str(e)}")

if __name__ == "__main__":
    examine_remaining_false_negatives()