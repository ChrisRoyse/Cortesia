#!/usr/bin/env python3
"""
Debug documentation filtering in chunk creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker import SmartChunker

def debug_doc_filtering():
    chunker = SmartChunker()
    
    rust_code = '''/// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
}'''
    
    lines = rust_code.split('\n')
    
    # Find the AppConfig declaration
    declarations = chunker.find_declarations(lines, 'rust')
    appconfig_decl = None
    for decl in declarations:
        if decl.name == 'AppConfig':
            appconfig_decl = decl
            break
    
    if not appconfig_decl:
        print("AppConfig declaration not found!")
        return
    
    print(f"=== AppConfig Declaration Debug ===")
    print(f"Declaration line: {appconfig_decl.line_number}")
    print(f"Declaration scope: {appconfig_decl.scope_start} -> {appconfig_decl.scope_end}")
    
    # Simulate the doc filtering logic
    doc_detection = chunker.doc_detector.detect_documentation_multi_pass(
        '\n'.join(lines), 'rust', appconfig_decl.line_number
    )
    
    print(f"\nDocumentation detection:")
    print(f"  Has documentation: {doc_detection['has_documentation']}")
    print(f"  Documentation lines: {doc_detection['documentation_lines']}")
    
    # Apply the filtering logic from create_declaration_chunk
    doc_search_start = max(0, appconfig_decl.line_number - 15)
    associated_doc_lines = []
    
    print(f"\nFiltering logic:")
    print(f"  Declaration line: {appconfig_decl.line_number}")
    print(f"  Doc search start: {doc_search_start}")
    
    if doc_detection['has_documentation'] and doc_detection['documentation_lines']:
        for doc_line in doc_detection['documentation_lines']:
            print(f"  Checking doc line {doc_line}:")
            
            # Only include documentation that's close to and before this declaration
            if (doc_search_start <= doc_line < appconfig_decl.line_number):
                print(f"    -> INCLUDED (before declaration in range)")
                associated_doc_lines.append(doc_line)
            # Also include documentation that's within the first few lines after declaration start
            elif (appconfig_decl.line_number <= doc_line <= appconfig_decl.line_number + 2):
                print(f"    -> INCLUDED (after declaration within 2 lines)")
                associated_doc_lines.append(doc_line)
            else:
                print(f"    -> EXCLUDED (out of range)")
    
    print(f"\nAssociated doc lines: {associated_doc_lines}")
    
    # Check gap detection
    if associated_doc_lines:
        print(f"\nGap detection:")
        gap_found = False
        for line_num in range(max(associated_doc_lines) + 1, appconfig_decl.line_number):
            line_content = lines[line_num] if line_num < len(lines) else ""
            is_doc = chunker._is_documentation_line(line_content, 'rust')
            is_annotation = chunker._is_declaration_annotation(line_content, 'rust')
            print(f"  Line {line_num}: {repr(line_content)} -> empty: {not line_content.strip()}, is_doc: {is_doc}, is_annotation: {is_annotation}")
            
            if line_content.strip() and not is_doc:
                if is_annotation:
                    print(f"    -> SKIPPED (annotation, part of declaration)")
                    continue
                
                print(f"    -> GAP FOUND! Non-empty, non-doc line between doc and declaration")
                gap_found = True
                break
        
        print(f"  Gap found: {gap_found}")

if __name__ == "__main__":
    debug_doc_filtering()