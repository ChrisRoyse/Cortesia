# Task 054: Regex Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates regex test cases that validates pattern matching with regular expressions.

## Project Structure
tests/
  regex_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for regular expression pattern matching including basic patterns, character classes, quantifiers, and complex regex constructs.

## Requirements
1. Create comprehensive integration test
2. Test regex pattern matching
3. Validate complex regex constructs
4. Handle regex edge cases safely
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_basic_regex_patterns() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let regex_test_files = test_generator.generate_regex_test_files().await?;
    let validator = CorrectnessValidator::new(&regex_test_files.index_path, &regex_test_files.vector_path).await?;
    
    let basic_regex_test_cases = vec![
        // Basic patterns
        ("regex:/function/", vec!["functions.rs", "function_defs.py"], "function pattern"),
        ("regex:/test\\d+/", vec!["test_files.rs", "numbered_tests.py"], "test with numbers"),
        ("regex:/error.*/", vec!["error_handling.rs", "error_types.py"], "error patterns"),
        
        // Character classes
        ("regex:/[Tt]est/", vec!["Test.rs", "test.py", "testing.js"], "test case insensitive"),
        ("regex:/\\d{3}/", vec!["numbers.rs", "codes.py", "ids.js"], "three digits"),
        ("regex:/\\w+_test/", vec!["unit_test.rs", "integration_test.py"], "word_test pattern"),
        
        // Quantifiers
        ("regex:/test+/", vec!["test.rs", "testt.py", "testing.js"], "test with plus"),
        ("regex:/test*/", vec!["tes.rs", "test.py", "testttt.js"], "test with star"),
        ("regex:/test?/", vec!["tes.rs", "test.py"], "optional test"),
        
        // Anchors
        ("regex:/^function/", vec!["function_start.rs"], "function at start"),
        ("regex:/test$/", vec!["end_test.rs"], "test at end"),
        
        // Groups and alternation
        ("regex:/(test|spec)/", vec!["test.rs", "spec.py", "testing.js"], "test or spec"),
        ("regex:/func(tion)?/", vec!["func.rs", "function.py"], "optional tion"),
    ];
    
    for (query, expected_files, description) in basic_regex_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::RegexSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed basic regex test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Basic regex patterns test completed successfully");
    Ok(())
}
```

## Success Criteria
- Basic regex patterns work correctly
- Character classes and quantifiers function properly
- Anchors (^, $) work for line boundaries
- Groups and alternation work correctly
- Complex regex patterns are handled safely
- Regex timeout protection prevents catastrophic backtracking
- Precision >= 85% for regex searches
- Performance acceptable for complex patterns

## Time Limit
10 minutes maximum