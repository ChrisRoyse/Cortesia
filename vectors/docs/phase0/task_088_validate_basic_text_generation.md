# Micro-Task 088: Validate Basic Text Generation

## Objective
Create comprehensive validation tools to verify that all basic text files are properly generated, encoded, and suitable for vector search testing.

## Context
Before proceeding to more complex pattern generation, we need to ensure all basic text files meet quality standards and can be processed correctly by the vector search system.

## Prerequisites
- Task 087 completed (Content type variations generated)

## Time Estimate
10 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create comprehensive validation script `validate_all_basic_text.py`:
   ```python
   #!/usr/bin/env python3
   """
   Comprehensive validation for all basic text generation tasks.
   """
   
   import os
   import sys
   import json
   from pathlib import Path
   from datetime import datetime
   
   def validate_file_exists_and_readable(file_path):
       """Validate that file exists and is readable."""
       if not file_path.exists():
           return False, "File does not exist"
       
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               content = f.read()
           return True, f"Readable, {len(content)} characters"
       except Exception as e:
           return False, f"Read error: {e}"
   
   def validate_template_format(file_path):
       """Validate that file follows template format."""
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               content = f.read()
           
           required_headers = [
               "Test file:",
               "Category:",
               "Pattern focus:",
               "Generated:",
               "Windows compatible: true",
               "UTF-8 encoding: true"
           ]
           
           missing_headers = []
           for header in required_headers:
               if header not in content:
                   missing_headers.append(header)
           
           if missing_headers:
               return False, f"Missing headers: {missing_headers}"
           
           return True, "All required headers present"
       
       except Exception as e:
           return False, f"Validation error: {e}"
   
   def validate_content_quality(file_path):
       """Validate content quality and characteristics."""
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               content = f.read()
           
           # Extract actual content (after headers, before footer)
           lines = content.split('\n')
           content_lines = []
           in_content = False
           
           for line in lines:
               if "--- End of test content ---" in line:
                   break
               if in_content or ("Test content for vector search" in line):
                   if line.strip():
                       content_lines.append(line)
                   in_content = True
           
           actual_content = '\n'.join(content_lines[1:])  # Skip the "Test content..." line
           
           # Quality checks
           if len(actual_content) < 100:
               return False, f"Content too short: {len(actual_content)} chars"
           
           words = actual_content.split()
           if len(words) < 20:
               return False, f"Too few words: {len(words)}"
           
           # Check for reasonable text structure
           sentences = actual_content.count('.') + actual_content.count('!') + actual_content.count('?')
           if sentences < 2:
               return False, f"Too few sentences: {sentences}"
           
           return True, f"{len(words)} words, {sentences} sentences, {len(actual_content)} chars"
       
       except Exception as e:
           return False, f"Content analysis error: {e}"
   
   def validate_encoding(file_path):
       """Validate UTF-8 encoding without BOM."""
       try:
           # Check for UTF-8 BOM
           with open(file_path, 'rb') as f:
               first_bytes = f.read(3)
           
           if first_bytes == b'\xef\xbb\xbf':
               return False, "File has UTF-8 BOM (should not have BOM)"
           
           # Validate UTF-8 encoding
           with open(file_path, 'r', encoding='utf-8') as f:
               f.read()
           
           return True, "Valid UTF-8 without BOM"
       
       except UnicodeDecodeError as e:
           return False, f"UTF-8 decode error: {e}"
       except Exception as e:
           return False, f"Encoding check error: {e}"
   
   def run_comprehensive_validation():
       """Run all validation checks on basic text files."""
       basic_text_dir = Path("basic_text")
       
       if not basic_text_dir.exists():
           print("❌ Error: basic_text directory not found")
           return False
       
       # Expected files from all tasks
       expected_files = [
           # Task 084 - Basic text samples
           "simple_sentences.txt",
           "technical_documentation.txt",
           "lists_and_enumerations.txt", 
           "numerical_data.txt",
           "question_answer.txt",
           
           # Task 085 - Paragraph variations
           "short_paragraphs.txt",
           "long_paragraphs.txt",
           "mixed_paragraphs.txt",
           "nested_structure.txt",
           "list_structure.txt",
           
           # Task 086 - Formatting variations
           "case_emphasis.txt",
           "spacing_alignment.txt",
           "line_length.txt",
           "punctuation_symbols.txt",
           "numbers_dates.txt",
           
           # Task 087 - Content type variations
           "technical_documentation.txt",  # Note: may overlap with task 084
           "business_communication.txt",
           "academic_research.txt",
           "creative_narrative.txt",
           "conversational_informal.txt"
       ]
       
       # Remove duplicates
       expected_files = list(set(expected_files))
       
       print(f"🔍 Validating {len(expected_files)} expected basic text files...")
       print("=" * 80)
       
       validation_results = []
       total_passed = 0
       total_failed = 0
       
       for filename in expected_files:
           file_path = basic_text_dir / filename
           
           print(f"\n📄 Validating {filename}:")
           
           # Test 1: File exists and readable
           exists_ok, exists_msg = validate_file_exists_and_readable(file_path)
           print(f"  {'✅' if exists_ok else '❌'} Exists/Readable: {exists_msg}")
           
           if not exists_ok:
               validation_results.append({
                   "filename": filename,
                   "passed": False,
                   "errors": [exists_msg]
               })
               total_failed += 1
               continue
           
           # Test 2: Template format
           template_ok, template_msg = validate_template_format(file_path)
           print(f"  {'✅' if template_ok else '❌'} Template Format: {template_msg}")
           
           # Test 3: Content quality
           content_ok, content_msg = validate_content_quality(file_path)
           print(f"  {'✅' if content_ok else '❌'} Content Quality: {content_msg}")
           
           # Test 4: Encoding
           encoding_ok, encoding_msg = validate_encoding(file_path)
           print(f"  {'✅' if encoding_ok else '❌'} Encoding: {encoding_msg}")
           
           # Overall result
           all_passed = exists_ok and template_ok and content_ok and encoding_ok
           errors = []
           if not template_ok: errors.append(template_msg)
           if not content_ok: errors.append(content_msg)
           if not encoding_ok: errors.append(encoding_msg)
           
           validation_results.append({
               "filename": filename,
               "passed": all_passed,
               "errors": errors
           })
           
           if all_passed:
               total_passed += 1
               print(f"  🎉 Overall: PASSED")
           else:
               total_failed += 1
               print(f"  💥 Overall: FAILED ({len(errors)} issues)")
       
       # Summary
       print("\n" + "=" * 80)
       print(f"📊 VALIDATION SUMMARY")
       print(f"   Total files checked: {len(expected_files)}")
       print(f"   ✅ Passed: {total_passed}")
       print(f"   ❌ Failed: {total_failed}")
       print(f"   📈 Success rate: {(total_passed/len(expected_files)*100):.1f}%")
       
       # Save detailed results
       report = {
           "validation_date": datetime.now().isoformat(),
           "total_files": len(expected_files),
           "passed": total_passed,
           "failed": total_failed,
           "success_rate": total_passed/len(expected_files)*100,
           "results": validation_results
       }
       
       with open("basic_text_validation_report.json", 'w', encoding='utf-8') as f:
           json.dump(report, f, indent=2)
       
       print(f"\n📋 Detailed report saved to: basic_text_validation_report.json")
       
       if total_failed > 0:
           print(f"\n⚠️  {total_failed} files failed validation. Check errors above.")
           return False
       
       print(f"\n🎉 All {total_passed} files passed validation successfully!")
       return True
   
   def fix_common_issues():
       """Attempt to fix common validation issues."""
       print("🔧 Attempting to fix common issues...")
       
       # Run encoding fix if available
       if Path("fix_encoding.py").exists():
           print("  Running encoding fix...")
           os.system("python fix_encoding.py basic_text")
       
       print("  Fix attempt completed.")
   
   def main():
       """Main validation function."""
       print("🚀 Starting comprehensive basic text validation...")
       
       success = run_comprehensive_validation()
       
       if not success:
           print("\n🔧 Attempting automatic fixes...")
           fix_common_issues()
           
           print("\n🔄 Re-running validation after fixes...")
           success = run_comprehensive_validation()
       
       if success:
           print("\n✨ All basic text files are ready for vector search testing!")
           return 0
       else:
           print("\n❌ Validation failed. Please review errors and fix manually.")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Create quick check script `quick_check_basic.py`:
   ```python
   #!/usr/bin/env python3
   """
   Quick check for basic text file generation status.
   """
   
   import os
   from pathlib import Path
   
   def quick_check():
       """Quick status check of basic text generation."""
       basic_text_dir = Path("basic_text")
       
       if not basic_text_dir.exists():
           print("❌ basic_text directory not found")
           return False
       
       files = list(basic_text_dir.glob("*.txt"))
       print(f"📁 Found {len(files)} text files in basic_text/")
       
       # Check total size
       total_size = sum(f.stat().st_size for f in files)
       print(f"💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
       
       # List files with sizes
       print("\n📄 Files:")
       for file_path in sorted(files):
           size = file_path.stat().st_size
           print(f"  {file_path.name}: {size:,} bytes")
       
       # Basic content check
       print(f"\n🔍 Quick content validation:")
       issues = 0
       for file_path in files:
           try:
               with open(file_path, 'r', encoding='utf-8') as f:
                   content = f.read()
               
               if len(content) < 50:
                   print(f"  ⚠️  {file_path.name}: Very short content ({len(content)} chars)")
                   issues += 1
               elif "Test file:" not in content:
                   print(f"  ⚠️  {file_path.name}: Missing template header")
                   issues += 1
               else:
                   words = len(content.split())
                   print(f"  ✅ {file_path.name}: {words} words")
           
           except Exception as e:
               print(f"  ❌ {file_path.name}: Error reading file - {e}")
               issues += 1
       
       print(f"\n📊 Summary: {len(files)} files, {issues} issues")
       return issues == 0
   
   def main():
       """Main function."""
       success = quick_check()
       return 0 if success else 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
4. Create validation batch script `validate_basic_text.bat`:
   ```batch
   @echo off
   echo 🚀 Basic Text Validation Suite
   echo ================================
   
   REM Quick check first
   echo.
   echo 🔍 Running quick check...
   python quick_check_basic.py
   if errorlevel 1 (
       echo ⚠️  Quick check found issues
   ) else (
       echo ✅ Quick check passed
   )
   
   echo.
   echo 🧪 Running comprehensive validation...
   python validate_all_basic_text.py
   
   if errorlevel 1 (
       echo.
       echo ❌ Comprehensive validation failed
       echo 📋 Check basic_text_validation_report.json for details
       exit /b 1
   ) else (
       echo.
       echo ✅ All validations passed successfully!
   )
   
   echo.
   echo 📋 Validation report available in basic_text_validation_report.json
   echo 🎉 Basic text generation phase completed!
   ```
5. Run validation: `validate_basic_text.bat`
6. Return to root: `cd ..\..`
7. Commit: `git add data\test_files\validate_all_basic_text.py data\test_files\quick_check_basic.py data\test_files\validate_basic_text.bat data\test_files\basic_text_validation_report.json && git commit -m "task_088: Comprehensive validation of basic text generation"`

## Expected Output
- Comprehensive validation script for all basic text files
- Quick check script for status overview
- Windows batch automation script
- JSON validation report with detailed results
- Automatic issue detection and fixing

## Success Criteria
- [ ] Comprehensive validation script created
- [ ] Quick check script created
- [ ] Batch automation script created
- [ ] All basic text files validated successfully
- [ ] Validation report generated
- [ ] No encoding or format issues detected

## Validation Commands
```cmd
cd data\test_files
validate_basic_text.bat
type basic_text_validation_report.json
```

## Next Task
task_089_generate_code_with_special_characters.md

## Notes
- Validation ensures quality before proceeding to complex patterns
- Automated fixing reduces manual intervention requirements
- JSON report provides detailed debugging information
- Batch script simplifies validation workflow for Windows users
- Success here validates the foundation for all subsequent test data generation