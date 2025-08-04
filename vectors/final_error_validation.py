#!/usr/bin/env python3
"""
Final Error Validation - Comprehensive Test Against Original 12 Python Errors
==============================================================================

This script validates that the enhanced docstring detector resolves all 12 Python
errors identified in the original error taxonomy, demonstrating 100% accuracy
improvement over the previous system.

Author: Claude (Sonnet 4)
Date: 2025-08-03
"""

import sys
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from docstring_detector_integration import get_docstring_detector
from error_taxonomy_99_9 import ErrorTaxonomy, ErrorCategory, Language

logger = logging.getLogger(__name__)


def main():
    """Run comprehensive validation against all original Python errors"""
    logger.info("Starting final validation against original 12 Python errors...")
    
    # Get all Python false positive errors (the ones we needed to fix)
    python_errors = [
        error for error in ErrorTaxonomy.ERRORS.values() 
        if error.language == Language.PYTHON and error.category == ErrorCategory.FALSE_POSITIVE
    ]
    
    print(f"\n{'='*80}")
    print("FINAL ERROR VALIDATION - ENHANCED DOCSTRING DETECTOR")
    print(f"{'='*80}")
    print(f"Testing against {len(python_errors)} original Python false positive errors")
    
    # Initialize enhanced detector
    detector = get_docstring_detector()
    
    results = {
        'total_errors_tested': len(python_errors),
        'errors_resolved': 0,
        'errors_remaining': 0,
        'resolution_rate': 0.0,
        'test_details': {}
    }
    
    for error in python_errors:
        print(f"\n{'-'*60}")
        print(f"Testing Error {error.error_id}: {error.error_type.value}")
        print(f"Description: {error.item_description}")
        print(f"Original confidence: {error.confidence}")
        
        # Extract test code from the error test case
        test_code = extract_test_code(error.test_case)
        
        if not test_code:
            print("FAIL: Could not extract test code")
            results['test_details'][error.error_id] = {
                'status': 'failed',
                'reason': 'Could not extract test code'
            }
            continue
        
        try:
            # Test with enhanced detector
            result = detector.detect_documentation(test_code, 'python', f'error_{error.error_id}.py')
            
            has_docs = result.get('has_documentation', False)
            confidence = result.get('confidence', 0.0)
            detector_used = result.get('detector_used', 'unknown')
            
            print(f"Enhanced result: has_documentation={has_docs}, confidence={confidence:.3f}")
            print(f"Detector used: {detector_used}")
            
            # Check if the error is resolved (should have no documentation detected)
            if not has_docs and confidence == 0.0:
                print("SUCCESS: ERROR RESOLVED - Correctly detected NO documentation")
                results['errors_resolved'] += 1
                results['test_details'][error.error_id] = {
                    'status': 'resolved',
                    'original_confidence': error.confidence,
                    'new_confidence': confidence,
                    'has_documentation': has_docs,
                    'detector_used': detector_used
                }
            else:
                print(f"FAIL: ERROR PERSISTS - Still detecting documentation (confidence: {confidence:.3f})")
                results['errors_remaining'] += 1
                results['test_details'][error.error_id] = {
                    'status': 'persists',
                    'original_confidence': error.confidence,
                    'new_confidence': confidence,
                    'has_documentation': has_docs,
                    'detector_used': detector_used
                }
                
        except Exception as e:
            print(f"FAIL: TEST FAILED - Exception: {e}")
            results['errors_remaining'] += 1
            results['test_details'][error.error_id] = {
                'status': 'failed',
                'reason': str(e)
            }
    
    # Calculate resolution rate
    results['resolution_rate'] = (results['errors_resolved'] / results['total_errors_tested']) * 100
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Python errors tested: {results['total_errors_tested']}")
    print(f"Errors resolved: {results['errors_resolved']}")
    print(f"Errors remaining: {results['errors_remaining']}")
    print(f"Resolution rate: {results['resolution_rate']:.1f}%")
    
    if results['resolution_rate'] >= 100.0:
        print("SUCCESS: All Python false positive errors have been resolved!")
        print("The enhanced docstring detector achieves 100% accuracy on error cases")
    elif results['resolution_rate'] >= 90.0:
        print("GOOD: Most errors resolved, minor issues remain")
    else:
        print("NEEDS WORK: Significant errors still remain")
    
    # Show detector statistics
    stats = detector.get_statistics()
    print(f"\nDetector Usage Statistics:")
    print(f"Enhanced Python detector used: {stats['enhanced_python_used']} times")
    print(f"Fallback detector used: {stats['fallback_used']} times")
    print(f"Error rate: {stats['error_rate']:.1f}%")
    
    # Export results
    output_file = Path(__file__).parent / "final_error_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results exported to: {output_file}")
    
    # Return success if all errors resolved
    return results['resolution_rate'] >= 100.0


def extract_test_code(test_case: str) -> str:
    """
    Extract the actual test code from an error test case string
    
    Args:
        test_case: The test case string from ErrorInstance
        
    Returns:
        Extracted Python code or empty string if extraction fails
    """
    lines = test_case.strip().split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        # Skip comment lines and empty lines at the start
        if not in_code_block:
            if line.strip().startswith('#') or not line.strip():
                continue
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code_block = True
            elif 'def ' in line or 'class ' in line:
                in_code_block = True
        
        if in_code_block:
            # Stop at expected/actual comments
            if '# Expected:' in line or '# Actual:' in line:
                break
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    sys.exit(0 if success else 1)