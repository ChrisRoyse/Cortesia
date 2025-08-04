# Micro-Task 083: Setup File Generation Templates

## Objective
Create templates for generating consistent test data files across all pattern categories.

## Context
Templates ensure that all generated test files follow consistent formatting, include proper headers, and contain the necessary metadata for systematic testing. This standardization simplifies validation and debugging.

## Prerequisites
- Task 082 completed (Encoding validation tools created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create templates directory: `mkdir templates`
3. Navigate to templates: `cd templates`
4. Create `rust_file_template.rs`:
   ```rust
   //! Test file: {filename}
   //! Category: {category}
   //! Pattern focus: {pattern_focus}
   //! Generated: {timestamp}
   //! Windows compatible: true
   //! UTF-8 encoding: true
   
   // Test patterns for vector search system validation
   // This file contains {pattern_focus} patterns for comprehensive testing
   
   {file_content}
   
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_file_compiles() {
           // Ensures the test file is syntactically valid
           assert!(true, "File compiles successfully");
       }
   }
   ```
5. Create `toml_file_template.toml`:
   ```toml
   # Test file: {filename}
   # Category: {category}  
   # Pattern focus: {pattern_focus}
   # Generated: {timestamp}
   # Windows compatible: true
   # UTF-8 encoding: true
   
   # Test configuration for vector search system validation
   # This file contains {pattern_focus} patterns for comprehensive testing
   
   {file_content}
   ```
6. Create `text_file_template.txt`:
   ```
   Test file: {filename}
   Category: {category}
   Pattern focus: {pattern_focus}
   Generated: {timestamp}
   Windows compatible: true
   UTF-8 encoding: true
   
   Test content for vector search system validation
   This file contains {pattern_focus} patterns for comprehensive testing
   
   {file_content}
   
   --- End of test content ---
   ```
7. Create `json_file_template.json`:
   ```json
   {
     "_metadata": {
       "filename": "{filename}",
       "category": "{category}",
       "pattern_focus": "{pattern_focus}",
       "generated": "{timestamp}",
       "windows_compatible": true,
       "utf8_encoding": true,
       "description": "Test data for vector search system validation"
     },
     "content": {file_content}
   }
   ```
8. Create `template_generator.py`:
   ```python
   #!/usr/bin/env python3
   """
   Template-based file generator for test data creation.
   """
   
   import os
   import json
   from datetime import datetime
   from pathlib import Path
   
   class TestFileGenerator:
       def __init__(self, templates_dir="templates"):
           self.templates_dir = Path(templates_dir)
           self.templates = self._load_templates()
       
       def _load_templates(self):
           """Load all template files."""
           templates = {}
           
           template_files = {
               'rust': 'rust_file_template.rs',
               'toml': 'toml_file_template.toml', 
               'txt': 'text_file_template.txt',
               'json': 'json_file_template.json'
           }
           
           for template_type, template_file in template_files.items():
               template_path = self.templates_dir / template_file
               if template_path.exists():
                   with open(template_path, 'r', encoding='utf-8') as f:
                       templates[template_type] = f.read()
           
           return templates
       
       def generate_file(self, template_type, filename, category, pattern_focus, content, output_dir="."):
           """Generate a test file from template."""
           if template_type not in self.templates:
               raise ValueError(f"Template type '{template_type}' not found")
           
           template = self.templates[template_type]
           
           # Prepare template variables
           variables = {
               'filename': filename,
               'category': category,
               'pattern_focus': pattern_focus,
               'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'file_content': content
           }
           
           # Replace template variables
           generated_content = template.format(**variables)
           
           # Write to output file
           output_path = Path(output_dir) / filename
           output_path.parent.mkdir(parents=True, exist_ok=True)
           
           with open(output_path, 'w', encoding='utf-8') as f:
               f.write(generated_content)
           
           return str(output_path)
       
       def generate_rust_file(self, filename, category, pattern_focus, rust_code, output_dir="."):
           """Generate a Rust test file."""
           return self.generate_file('rust', filename, category, pattern_focus, rust_code, output_dir)
       
       def generate_toml_file(self, filename, category, pattern_focus, toml_content, output_dir="."):
           """Generate a TOML test file."""
           return self.generate_file('toml', filename, category, pattern_focus, toml_content, output_dir)
       
       def generate_text_file(self, filename, category, pattern_focus, text_content, output_dir="."):
           """Generate a text test file."""
           return self.generate_file('txt', filename, category, pattern_focus, text_content, output_dir)
       
       def generate_json_file(self, filename, category, pattern_focus, json_content, output_dir="."):
           """Generate a JSON test file."""
           # JSON content should be a dict or string
           if isinstance(json_content, dict):
               json_content = json.dumps(json_content, indent=2)
           return self.generate_file('json', filename, category, pattern_focus, json_content, output_dir)
   
   def main():
       """Test the template generator."""
       generator = TestFileGenerator()
       
       # Test Rust file generation
       rust_code = '''use std::collections::HashMap;
   
   fn main() {
       let result: Result<String, Box<dyn std::error::Error>> = Ok("test".to_string());
       println!("{:?}", result);
   }'''
   
       output_file = generator.generate_rust_file(
           "test_generics.rs",
           "rust_patterns", 
           "Generic types and Result patterns",
           rust_code,
           "../rust_patterns"
       )
       
       print(f"Generated test file: {output_file}")
   
   if __name__ == "__main__":
       main()
   ```
9. Create `generate_test.bat`:
   ```batch
   @echo off
   echo Testing template generator...
   
   REM Test template generator
   python template_generator.py
   
   if errorlevel 1 (
       echo ERROR: Template generator test failed
       exit /b 1
   )
   
   echo Template generator test completed successfully
   
   REM Verify generated file
   if exist "..\rust_patterns\test_generics.rs" (
       echo ✓ Test file generated successfully
       type "..\rust_patterns\test_generics.rs"
   ) else (
       echo ✗ Test file was not generated
       exit /b 1
   )
   ```
10. Return to root: `cd ..\..\..`
11. Commit: `git add data\test_files\templates && git commit -m "task_083: Create file generation templates and generator"`

## Expected Output
- Templates for Rust, TOML, text, and JSON files
- Python-based template generator
- Consistent file headers with metadata
- Windows batch test script

## Success Criteria
- [ ] All 4 template files created
- [ ] Template generator script created
- [ ] Test batch script created
- [ ] Template variables system implemented
- [ ] UTF-8 encoding and Windows compatibility ensured

## Validation Commands
```cmd
cd data\test_files\templates
python template_generator.py
generate_test.bat
```

## Next Task
task_084_generate_basic_text_samples.md

## Notes
- Templates ensure consistency across all generated test files
- Metadata headers help with debugging and validation
- Generator supports multiple file types with unified interface
- All templates include Windows compatibility markers