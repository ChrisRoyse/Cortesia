#!/usr/bin/env python3
"""
Debug why AppConfig documentation isn't being detected
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker
from ultra_reliable_core import UniversalDocumentationDetector

def debug_appconfig_docs():
    rust_code = '''/// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost:5432/db".to_string(),
            server_host: "0.0.0.0".to_string(),
            server_port: 8001,
        }
    }
}
'''
    
    lines = rust_code.split('\n')
    detector = UniversalDocumentationDetector()
    
    print("=== AppConfig Documentation Debug ===")
    print(f"Looking for documentation around line 2 (AppConfig struct)")
    
    # Test documentation detection for AppConfig specifically
    doc_detection = detector.detect_documentation_multi_pass(rust_code, 'rust', 2)  # Line 2 is the AppConfig struct
    
    print(f"Has documentation: {doc_detection['has_documentation']}")
    print(f"Confidence: {doc_detection['confidence']}")
    print(f"Documentation lines: {doc_detection['documentation_lines']}")
    print(f"Patterns found: {doc_detection['patterns_found']}")
    print(f"Detection methods: {doc_detection['detection_methods']}")
    
    print("\n=== Line by line analysis ===")
    for i, line in enumerate(lines):
        print(f"Line {i}: {repr(line)}")
        if line.strip().startswith('///'):
            print(f"  -> This is Rust documentation!")

if __name__ == "__main__":
    debug_appconfig_docs()