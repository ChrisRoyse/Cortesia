# Micro-Task 080: Create Test Data Manifest

## Objective
Create a comprehensive manifest file that tracks all test data files and their purposes for systematic validation.

## Context
A manifest file will help track what test data exists, what patterns it contains, and how it should be used in testing. This enables automated validation and ensures comprehensive test coverage.

## Prerequisites
- Task 079 completed (Test data directory structure created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create `test_data_manifest.json`:
   ```json
   {
     "manifest_version": "1.0",
     "created_date": "2024-08-04",
     "description": "Test data manifest for vector search system validation",
     "windows_compatibility": true,
     "encoding": "UTF-8",
     "categories": {
       "basic_text": {
         "description": "Plain text files for baseline testing",
         "file_count": 0,
         "max_file_size": "1MB",
         "patterns": ["plain_text", "simple_sentences", "paragraphs"]
       },
       "code_samples": {
         "description": "Code files with special characters",
         "file_count": 0,
         "max_file_size": "512KB",
         "patterns": ["functions", "classes", "imports", "comments"]
       },
       "rust_patterns": {
         "description": "Rust language-specific patterns",
         "file_count": 0,
         "max_file_size": "256KB",
         "patterns": ["structs", "enums", "traits", "impls", "modules"]
       },
       "generic_types": {
         "description": "Generic type patterns",
         "file_count": 0,
         "max_file_size": "128KB",
         "patterns": ["Result<T,E>", "Option<T>", "Vec<String>", "HashMap<K,V>"]
       },
       "bracket_patterns": {
         "description": "Various bracket combinations",
         "file_count": 0,
         "max_file_size": "64KB",
         "patterns": ["[workspace]", "[dependencies]", "[dev-dependencies]", "[]"]
       },
       "operator_patterns": {
         "description": "Operator sequences",
         "file_count": 0,
         "max_file_size": "64KB",
         "patterns": ["->", ">>", "<<", "::", "=>", "..=", ".."]
       },
       "macro_patterns": {
         "description": "Rust macro patterns",
         "file_count": 0,
         "max_file_size": "128KB",
         "patterns": ["#[derive]", "#[cfg]", "#[test]", "#[allow]", "macro_rules!"]
       },
       "mixed_patterns": {
         "description": "Complex combinations",
         "file_count": 0,
         "max_file_size": "256KB",
         "patterns": ["multiple_special_chars", "nested_patterns", "complex_syntax"]
       },
       "large_files": {
         "description": "Performance testing files",
         "file_count": 0,
         "max_file_size": "10MB",
         "patterns": ["repeated_patterns", "large_datasets", "performance_test"]
       },
       "edge_cases": {
         "description": "Boundary conditions and error scenarios",
         "file_count": 0,
         "max_file_size": "32KB",
         "patterns": ["empty_files", "single_chars", "unicode", "malformed"]
       }
     },
     "validation_rules": {
       "encoding_check": "All files must be UTF-8 encoded",
       "path_check": "All paths must use Windows backslash separators",
       "size_check": "Files must not exceed category max_file_size",
       "pattern_check": "Files must contain documented patterns"
     }
   }
   ```
3. Create validation script `validate_manifest.bat`:
   ```batch
   @echo off
   echo Validating test data manifest...
   echo Checking directory structure...
   if not exist "basic_text" echo ERROR: basic_text directory missing
   if not exist "code_samples" echo ERROR: code_samples directory missing
   if not exist "rust_patterns" echo ERROR: rust_patterns directory missing
   if not exist "generic_types" echo ERROR: generic_types directory missing
   if not exist "bracket_patterns" echo ERROR: bracket_patterns directory missing
   if not exist "operator_patterns" echo ERROR: operator_patterns directory missing
   if not exist "macro_patterns" echo ERROR: macro_patterns directory missing
   if not exist "mixed_patterns" echo ERROR: mixed_patterns directory missing
   if not exist "large_files" echo ERROR: large_files directory missing
   if not exist "edge_cases" echo ERROR: edge_cases directory missing
   echo Validation complete.
   ```
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\test_data_manifest.json data\test_files\validate_manifest.bat && git commit -m "task_080: Create test data manifest and validation script"`

## Expected Output
- Comprehensive manifest file tracking all test data categories
- Validation script for directory structure
- Documentation of patterns and size limits

## Success Criteria
- [ ] Manifest file created with all 10 categories
- [ ] Validation script created and executable
- [ ] All patterns and limits documented
- [ ] Files committed to Git

## Validation Commands
```cmd
type data\test_files\test_data_manifest.json
data\test_files\validate_manifest.bat
```

## Next Task
task_081_create_pattern_reference_guide.md

## Notes
- Manifest will be updated as files are generated
- Validation script helps ensure directory integrity
- JSON format enables programmatic access