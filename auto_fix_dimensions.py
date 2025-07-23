#!/usr/bin/env python3
"""
Auto-fix script for common embedding dimension issues.
This script automatically fixes the most common patterns identified by the validation script.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class DimensionFixer:
    def __init__(self):
        self.files_modified = set()
        self.fixes_applied = 0
        self.backup_dir = Path("embedding_fixes_backup")
        
    def create_backup(self, file_path: str):
        """Create a backup of the file before modifying."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir()
            
        backup_path = self.backup_dir / Path(file_path).name
        counter = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{Path(file_path).stem}_{counter}{Path(file_path).suffix}"
            counter += 1
            
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        
        print(f"  [BACKUP] Created backup: {backup_path}")
        
    def fix_hardcoded_vectors(self, content: str) -> Tuple[str, int]:
        """Fix hardcoded vector dimensions."""
        fixes = 0
        
        # Pattern 1: vec![value; dimension]
        patterns = [
            (r'vec!\[([^;]+);\s*(64|128|384)\]', r'vec![\1; 96]'),
            (r'Vec::with_capacity\((64|128|384)\)', r'Vec::with_capacity(96)'),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content))
            content = re.sub(pattern, replacement, content)
            fixes += len(matches)
            
        return content, fixes
    
    def fix_dimension_variables(self, content: str) -> Tuple[str, int]:
        """Fix embedding dimension variable assignments."""
        fixes = 0
        
        patterns = [
            (r'let\s+embedding_dim\s*=\s*(64|128|384);', r'let embedding_dim = 96;'),
            (r'embedding_dim:\s*(64|128|384),', r'embedding_dim: 96,'),
            (r'const\s+EMBEDDING_DIM[^=]*=\s*(64|128|384);', r'const EMBEDDING_DIM = 96;'),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content))
            content = re.sub(pattern, replacement, content)
            fixes += len(matches)
            
        return content, fixes
    
    def fix_function_calls(self, content: str) -> Tuple[str, int]:
        """Fix function calls with wrong dimensions."""
        fixes = 0
        
        patterns = [
            (r'create_test_embedding\((64|128|384)\)', r'create_test_embedding(96)'),
            (r'create_embedding\([^,)]+,\s*(64|128|384)\)', r'create_embedding(\1, 96)'),
            (r'generate_simple_embedding\([^,)]+,\s*(64|128|384)\)', r'generate_simple_embedding(\1, 96)'),
        ]
        
        for pattern, replacement in patterns:
            # Handle the capture group reference properly
            if '\\1' in replacement:
                def replace_func(match):
                    if len(match.groups()) >= 2:
                        return replacement.replace('\\1', match.group(1))
                    return replacement
                
                matches = list(re.finditer(pattern, content))
                content = re.sub(pattern, replace_func, content)
                fixes += len(matches)
            else:
                matches = list(re.finditer(pattern, content))
                content = re.sub(pattern, replacement, content)
                fixes += len(matches)
                
        return content, fixes
    
    def fix_assertions(self, content: str) -> Tuple[str, int]:
        """Fix assertion statements with wrong dimensions."""
        fixes = 0
        
        patterns = [
            (r'assert_eq!\([^,]+\.len\(\),\s*(64|128|384)\)', r'assert_eq!(\1.len(), 96)'),
            (r'assert_eq!\([^,]+,\s*&(64|128|384)\.0\)', r'assert_eq!(\1, &96.0)'),
            (r'assert_eq!\([^,]+\.embedding_dimension\(\),\s*(64|128|384)\)', r'assert_eq!(\1.embedding_dimension(), 96)'),
        ]
        
        for pattern, replacement in patterns:
            # Handle capture group references
            def replace_func(match):
                # Find the part before .len(), .embedding_dimension(), etc.
                if 'assert_eq!' in match.group(0):
                    parts = match.group(0).split(',')
                    if len(parts) >= 2:
                        first_part = parts[0].replace('assert_eq!(', '')
                        return f"assert_eq!({first_part}, 96)"
                return match.group(0).replace(match.group(1), '96')
            
            matches = list(re.finditer(pattern, content))
            content = re.sub(pattern, replace_func, content)
            fixes += len(matches)
                
        return content, fixes
    
    def fix_helper_functions(self, content: str) -> Tuple[str, int]:
        """Fix helper function implementations."""
        fixes = 0
        
        # Fix function signatures and implementations
        patterns = [
            # Function that creates embeddings with wrong capacity
            (r'(fn\s+create_\w*embedding[^{]*\{[^}]*Vec::with_capacity\()(64|128|384)(\)[^}]*\})', 
             r'\g<1>96\g<3>'),
            # Loop ranges for embeddings  
            (r'(for\s+\w+\s+in\s+0\.\.)(64|128|384)', r'\g<1>96'),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
            fixes += len(matches)
                
        return content, fixes
    
    def fix_file(self, file_path: str) -> bool:
        """Fix a single file and return True if modifications were made."""
        if not os.path.exists(file_path):
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        content = original_content
        total_fixes = 0
        
        # Apply all fix types
        content, fixes = self.fix_hardcoded_vectors(content)
        total_fixes += fixes
        
        content, fixes = self.fix_dimension_variables(content)  
        total_fixes += fixes
        
        content, fixes = self.fix_function_calls(content)
        total_fixes += fixes
        
        content, fixes = self.fix_assertions(content)
        total_fixes += fixes
        
        content, fixes = self.fix_helper_functions(content)
        total_fixes += fixes
        
        if total_fixes > 0:
            self.create_backup(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.files_modified.add(file_path)
            self.fixes_applied += total_fixes
            
            print(f"  [FIXED] {file_path}: {total_fixes} fixes applied")
            return True
        else:
            print(f"  [SKIP] {file_path}: No fixes needed")
            return False
    
    def fix_high_priority_files(self):
        """Fix the highest priority files first."""
        print("[PRIORITY] Fixing high-priority files...")
        
        high_priority_files = [
            "tests/core/test_graph_mod.rs",
            "tests/core/test_graph_entity_operations.rs", 
            "tests/core/test_brain_enhanced_graph_mod.rs",
            "tests/core/test_brain_analytics.rs",
            "tests/cognitive/test_divergent.rs",
            "tests/cognitive/test_lateral.rs",
            "tests/cognitive/test_orchestrator.rs",
        ]
        
        for file_path in high_priority_files:
            self.fix_file(file_path)
    
    def fix_all_problematic_files(self):
        """Fix all files identified as having issues."""
        print("\n[ALL] Fixing all problematic files...")
        
        # Load the validation report to get the list of problematic files
        if os.path.exists("embedding_validation_report.json"):
            with open("embedding_validation_report.json", 'r') as f:
                report = json.load(f)
                
            problematic_files = list(report.get("detailed_issues", {}).keys())
            
            for file_path in problematic_files:
                # Convert Windows path separators to Unix for consistency
                normalized_path = file_path.replace('\\', '/')
                self.fix_file(normalized_path)
        else:
            print("  [WARN] No validation report found. Run validate_embedding_fixes.py first.")
    
    def generate_summary(self):
        """Generate a summary of changes made."""
        print("\n" + "="*60)
        print("AUTO-FIX SUMMARY")
        print("="*60)
        print(f"Files modified: {len(self.files_modified)}")
        print(f"Total fixes applied: {self.fixes_applied}")
        
        if self.files_modified:
            print(f"\nModified files:")
            for file_path in sorted(self.files_modified):
                print(f"  - {file_path}")
                
        if self.backup_dir.exists():
            print(f"\nBackups created in: {self.backup_dir}")
            
        print(f"\nNext steps:")
        print(f"1. Run 'python validate_embedding_fixes.py' to check remaining issues")  
        print(f"2. Run 'cargo test' to verify tests pass")
        print(f"3. Review changes in modified files")

def main():
    """Main function."""
    print("Starting automatic embedding dimension fixes...")
    
    if not os.path.exists("Cargo.toml"):
        print("[ERROR] Not in project root. Please run from the directory containing Cargo.toml")
        return False
        
    fixer = DimensionFixer()
    
    # Fix high priority files first
    fixer.fix_high_priority_files()
    
    # Then fix all other problematic files
    fixer.fix_all_problematic_files()
    
    # Generate summary
    fixer.generate_summary()
    
    return len(fixer.files_modified) > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)