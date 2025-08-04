# Micro-Task 093: Validate Code Generation

## Objective
Create comprehensive validation for all code files with special characters to ensure proper generation and vector search compatibility.

## Context
Before proceeding to Rust-specific patterns, validate that all code generation tasks produced correct files with proper special character handling and template formatting.

## Prerequisites
- Task 092 completed (Regex patterns generated)

## Time Estimate
8 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create comprehensive code validation script `validate_code_generation.py`:
   ```python
   #!/usr/bin/env python3
   """
   Comprehensive validation for code generation tasks 089-092.
   """
   
   import os
   import sys
   import json
   from pathlib import Path
   from datetime import datetime
   
   def validate_code_files():
       """Validate all generated code files."""
       code_samples_dir = Path("code_samples")
       
       if not code_samples_dir.exists():
           print("❌ Error: code_samples directory not found")
           return False
       
       # Expected files from tasks 089-092
       expected_files = {
           # Task 089 - Code with special characters
           "python_special_chars.py": {
               "type": "python",
               "min_size": 5000,
               "required_patterns": ["Result<T, E>", "HashMap<K, V>", "->", "lambda", "import"]
           },
           "javascript_special_chars.js": {
               "type": "javascript", 
               "min_size": 4000,
               "required_patterns": ["<T>", "=>", "async", "import", "interface"]
           },
           "config_special_chars.json": {
               "type": "json",
               "min_size": 2000,
               "required_patterns": ["{", "}", "[", "]", ":", ","]
           },
           
           # Task 090 - Shell script patterns
           "build_script.bat": {
               "type": "batch",
               "min_size": 2000,
               "required_patterns": ["@echo off", "%", "if errorlevel", "goto", "set"]
           },
           "build_script.ps1": {
               "type": "powershell",
               "min_size": 3000,
               "required_patterns": ["param(", "[CmdletBinding()]", "$", "function", "try"]
           },
           
           # Task 091 - Markup patterns  
           "test_document.html": {
               "type": "html",
               "min_size": 3000,
               "required_patterns": ["<html>", "&lt;", "&gt;", "&amp;", "<code>"]
           },
           "project_config.xml": {
               "type": "xml",
               "min_size": 2500,
               "required_patterns": ["<?xml", "xmlns", "<![CDATA[", "</", "Result<T, E>"]
           },
           "documentation.md": {
               "type": "markdown",
               "min_size": 4000,
               "required_patterns": ["# ", "## ", "```", "`Result<T, E>`", "| "]
           },
           
           # Task 092 - Regex patterns
           "regex_patterns.txt": {
               "type": "text",
               "min_size": 3000,
               "required_patterns": ["\\d", "\\w", "\\s", "[a-z]", "Result<\\w+"]
           }
       }
       
       print(f"🔍 Validating {len(expected_files)} code files...")
       print("=" * 80)
       
       validation_results = []
       total_passed = 0
       total_failed = 0
       
       for filename, expectations in expected_files.items():
           file_path = code_samples_dir / filename
           
           print(f"\n📄 Validating {filename} ({expectations['type']}):")
           
           # Test 1: File exists and readable
           if not file_path.exists():
               print(f"  ❌ File not found")
               validation_results.append({
                   "filename": filename,
                   "passed": False,
                   "errors": ["File not found"]
               })
               total_failed += 1
               continue
           
           try:
               with open(file_path, 'r', encoding='utf-8') as f:
                   content = f.read()
           except Exception as e:
               print(f"  ❌ Cannot read file: {e}")
               validation_results.append({
                   "filename": filename,
                   "passed": False,
                   "errors": [f"Read error: {e}"]
               })
               total_failed += 1
               continue
           
           print(f"  ✅ Readable: {len(content)} characters")
           
           errors = []
           
           # Test 2: Minimum file size
           if len(content) < expectations["min_size"]:
               error_msg = f"File too small: {len(content)} < {expectations['min_size']}"
               print(f"  ❌ {error_msg}")
               errors.append(error_msg)
           else:
               print(f"  ✅ Size check: {len(content)} >= {expectations['min_size']}")
           
           # Test 3: Template format (for non-code files)
           if expectations['type'] in ['text']:
               if "Test file:" not in content:
                   error_msg = "Missing template header"
                   print(f"  ❌ {error_msg}")
                   errors.append(error_msg)
               else:
                   print(f"  ✅ Template format: Header found")
           
           # Test 4: Required patterns
           missing_patterns = []
           for pattern in expectations["required_patterns"]:
               if pattern not in content:
                   missing_patterns.append(pattern)
           
           if missing_patterns:
               error_msg = f"Missing patterns: {missing_patterns}"
               print(f"  ❌ {error_msg}")
               errors.append(error_msg)
           else:
               print(f"  ✅ Required patterns: All {len(expectations['required_patterns'])} found")
           
           # Test 5: UTF-8 encoding
           try:
               content.encode('utf-8')
               print(f"  ✅ Encoding: Valid UTF-8")
           except UnicodeEncodeError as e:
               error_msg = f"UTF-8 encoding error: {e}"
               print(f"  ❌ {error_msg}")
               errors.append(error_msg)
           
           # Test 6: Special character density (should be high for code files)
           special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
           total_chars = len(content)
           density = (special_chars / total_chars * 100) if total_chars > 0 else 0
           
           expected_density = {
               'python': 10,
               'javascript': 10,
               'json': 15,
               'batch': 8,
               'powershell': 12,
               'html': 8,
               'xml': 10,
               'markdown': 5,
               'text': 5
           }
           
           min_density = expected_density.get(expectations['type'], 5)
           if density < min_density:
               error_msg = f"Low special char density: {density:.1f}% < {min_density}%"
               print(f"  ⚠️  {error_msg}")
               # This is a warning, not a failure
           else:
               print(f"  ✅ Special char density: {density:.1f}% >= {min_density}%")
           
           # Overall result
           if errors:
               validation_results.append({
                   "filename": filename,
                   "passed": False,
                   "errors": errors,
                   "size": len(content),
                   "special_char_density": density
               })
               total_failed += 1
               print(f"  💥 Overall: FAILED ({len(errors)} issues)")
           else:
               validation_results.append({
                   "filename": filename,
                   "passed": True,
                   "errors": [],
                   "size": len(content),
                   "special_char_density": density
               })
               total_passed += 1
               print(f"  🎉 Overall: PASSED")
       
       # Summary
       print("\n" + "=" * 80)
       print(f"📊 CODE GENERATION VALIDATION SUMMARY")
       print(f"   Total files checked: {len(expected_files)}")
       print(f"   ✅ Passed: {total_passed}")
       print(f"   ❌ Failed: {total_failed}")
       print(f"   📈 Success rate: {(total_passed/len(expected_files)*100):.1f}%")
       
       # Detailed analysis
       if validation_results:
           total_size = sum(r['size'] for r in validation_results)
           avg_density = sum(r['special_char_density'] for r in validation_results) / len(validation_results)
           
           print(f"\n📈 CONTENT ANALYSIS")
           print(f"   Total content: {total_size:,} characters")
           print(f"   Average file size: {total_size//len(validation_results):,} characters")
           print(f"   Average special char density: {avg_density:.1f}%")
       
       # Save detailed results
       report = {
           "validation_date": datetime.now().isoformat(),
           "total_files": len(expected_files),
           "passed": total_passed,
           "failed": total_failed,
           "success_rate": total_passed/len(expected_files)*100,
           "results": validation_results
       }
       
       with open("code_generation_validation_report.json", 'w', encoding='utf-8') as f:
           json.dump(report, f, indent=2)
       
       print(f"\n📋 Detailed report saved to: code_generation_validation_report.json")
       
       if total_failed > 0:
           print(f"\n⚠️  {total_failed} files failed validation. Check errors above.")
           return False
       
       print(f"\n🎉 All {total_passed} code files passed validation successfully!")
       return True
   
   def analyze_special_character_coverage():
       """Analyze special character coverage across all code files."""
       print("\n🔍 Analyzing special character coverage...")
       
       code_samples_dir = Path("code_samples")
       if not code_samples_dir.exists():
           print("❌ code_samples directory not found")
           return
       
       # Special character categories to track
       char_categories = {
           'brackets': ['[', ']', '{', '}', '(', ')'],
           'operators': ['=', '+', '-', '*', '/', '%', '&', '|'],
           'comparisons': ['<', '>', '!'],
           'arrows': ['->', '=>', '::'],
           'quotes': ['"', "'", '`'],
           'punctuation': [',', ';', ':', '.'],
           'special': ['#', '@', '$', '^', '~', '\\', '?']
       }
       
       coverage_stats = {}
       
       for file_path in code_samples_dir.glob("*.txt") | code_samples_dir.glob("*.py") | code_samples_dir.glob("*.js") | code_samples_dir.glob("*.html") | code_samples_dir.glob("*.xml") | code_samples_dir.glob("*.md") | code_samples_dir.glob("*.bat") | code_samples_dir.glob("*.ps1") | code_samples_dir.glob("*.json"):
           try:
               with open(file_path, 'r', encoding='utf-8') as f:
                   content = f.read()
               
               file_stats = {}
               for category, chars in char_categories.items():
                   count = sum(content.count(char) for char in chars)
                   file_stats[category] = count
               
               coverage_stats[file_path.name] = file_stats
               
           except Exception as e:
               print(f"  ⚠️  Error analyzing {file_path.name}: {e}")
       
       # Print coverage summary
       print("\n📊 Special Character Coverage by Category:")
       print("=" * 60)
       
       for category in char_categories.keys():
           total_count = sum(stats.get(category, 0) for stats in coverage_stats.values())
           files_with_category = sum(1 for stats in coverage_stats.values() if stats.get(category, 0) > 0)
           
           print(f"{category:12}: {total_count:6} occurrences in {files_with_category}/{len(coverage_stats)} files")
       
       print("\n📋 Per-file breakdown saved in validation report.")
   
   def main():
       """Main validation function."""
       print("🚀 Starting comprehensive code generation validation...")
       
       success = validate_code_files()
       
       if success:
           analyze_special_character_coverage()
           print("\n✨ All code files are ready for Rust-specific pattern generation!")
           return 0
       else:
           print("\n❌ Code generation validation failed. Please review errors and regenerate failed files.")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Create quick validation batch script `validate_code.bat`:
   ```batch
   @echo off
   echo 🚀 Code Generation Validation Suite
   echo ===================================
   
   echo.
   echo 🔍 Running code generation validation...
   python validate_code_generation.py
   
   if errorlevel 1 (
       echo.
       echo ❌ Code generation validation failed
       echo 📋 Check code_generation_validation_report.json for details
       exit /b 1
   ) else (
       echo.
       echo ✅ All code generation validation passed!
   )
   
   echo.
   echo 📋 Validation report available in code_generation_validation_report.json
   echo 🎉 Code generation phase completed successfully!
   ```
4. Run validation: `validate_code.bat`
5. Return to root: `cd ..\..`
6. Commit: `git add data\test_files\validate_code_generation.py data\test_files\validate_code.bat data\test_files\code_generation_validation_report.json && git commit -m "task_093: Comprehensive validation of code generation"`

## Expected Output
- Comprehensive validation of all code files from tasks 089-092
- Special character density analysis
- Pattern coverage verification
- JSON validation report with detailed results
- Windows batch automation script

## Success Criteria
- [ ] All expected code files validated successfully
- [ ] Template format verification completed
- [ ] Required patterns confirmed in all files
- [ ] UTF-8 encoding verified
- [ ] Special character coverage analyzed
- [ ] Validation report generated

## Validation Commands
```cmd
cd data\test_files
validate_code.bat
type code_generation_validation_report.json
```

## Next Task
task_094_generate_rust_struct_patterns.md

## Notes
- Comprehensive validation ensures quality before proceeding
- Special character analysis validates pattern complexity
- Coverage verification confirms comprehensive test data
- Success here enables confident progression to Rust-specific patterns