#!/usr/bin/env python3
"""
Test the Rust module documentation fix
"""

from ultra_reliable_core import UniversalDocumentationDetector

def test_rust_module_fix():
    """Test the fix for Rust module documentation"""
    
    # Test case from lib.rs:7
    rust_code = '''//! WASM bindings for neuromorphic processing
//!
//! Provides web-compatible interface with SIMD acceleration for
//! browser-based neuromorphic computation.

use wasm_bindgen::prelude::*;

pub mod simd_bindings;
pub mod snn_wasm;
pub mod ttfs_wasm;'''
    
    print("Testing Rust module documentation:")
    print("=" * 50)
    
    lines = rust_code.split('\n')
    for i, line in enumerate(lines):
        print(f"{i:2d}: {line}")
    
    # Test the detection
    detector = UniversalDocumentationDetector()
    
    # Test detection for the module declaration at line 7
    result = detector.detect_documentation_multi_pass(rust_code, 'rust', 7)
    
    print(f"\nDetection results for line 7 (pub mod simd_bindings;):")
    print(f"  Has documentation: {result['has_documentation']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Documentation lines: {result['documentation_lines']}")
    print(f"  Detection methods: {result['detection_methods']}")
    print(f"  Patterns found: {result['patterns_found'][:3]}...")  # First 3

if __name__ == "__main__":
    test_rust_module_fix()