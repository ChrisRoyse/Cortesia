# Task 052: Wildcard Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates wildcard test cases that validates the system correctly handles *, ?, and range pattern matching.

## Project Structure
tests/
  wildcard_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for wildcard pattern matching including asterisk (*), question mark (?), and character range patterns.

## Requirements
1. Create comprehensive integration test
2. Test all wildcard pattern types
3. Validate pattern matching accuracy
4. Handle complex wildcard combinations
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_asterisk_wildcards() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let wildcard_test_files = test_generator.generate_wildcard_test_files().await?;
    let validator = CorrectnessValidator::new(&wildcard_test_files.index_path, &wildcard_test_files.vector_path).await?;
    
    let asterisk_test_cases = vec![
        // Beginning wildcards
        ("*function", vec!["my_function.rs", "test_function.rs", "main_function.rs"], "ending with function"),
        ("*test", vec!["unit_test.rs", "integration_test.rs", "stress_test.rs"], "ending with test"),
        ("*error", vec!["parse_error.rs", "runtime_error.rs", "system_error.rs"], "ending with error"),
        
        // Ending wildcards
        ("function*", vec!["function_def.rs", "function_call.rs", "function_impl.rs"], "starting with function"),
        ("test*", vec!["test_case.rs", "test_data.rs", "test_utils.rs"], "starting with test"),
        ("error*", vec!["error_handling.rs", "error_types.rs", "error_codes.rs"], "starting with error"),
        
        // Middle wildcards
        ("func*ion", vec!["function.rs", "funktion.rs"], "func...ion pattern"),
        ("test*case", vec!["test_case.rs", "testcase.rs", "test_best_case.rs"], "test...case pattern"),
        ("struct*impl", vec!["struct_impl.rs", "struct_with_impl.rs"], "struct...impl pattern"),
        
        // Multiple wildcards
        ("*test*case*", vec!["unit_test_case_runner.rs", "my_test_case_example.rs"], "multiple wildcards"),
        ("*func*impl*", vec!["some_func_impl_detail.rs", "test_func_impl_code.rs"], "func with impl wildcards"),
        
        // File extension wildcards
        ("*.rs", vec!["main.rs", "lib.rs", "test.rs", "utils.rs"], "Rust files"),
        ("*.py", vec!["script.py", "test.py", "utils.py", "main.py"], "Python files"),
        ("*.js", vec!["app.js", "test.js", "utils.js", "main.js"], "JavaScript files"),
        
        // Complex patterns
        ("test_*_case.rs", vec!["test_unit_case.rs", "test_integration_case.rs"], "complex test pattern"),
        ("*_function_*.py", vec!["my_function_impl.py", "test_function_call.py"], "function with wildcards"),
    ];
    
    for (query, expected_files, description) in asterisk_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::WildcardSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed asterisk wildcard test {}: {} - {}", description, query, result.summary());
        
        // Verify wildcard pattern matching
        for file_result in &result.actual_files {
            assert!(matches_wildcard_pattern(&file_result.filename, query),
                   "File {} doesn't match wildcard pattern {}", file_result.filename, query);
        }
    }
    
    println!("Asterisk wildcards test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_question_mark_wildcards() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let question_test_files = test_generator.generate_question_wildcard_test_files().await?;
    let validator = CorrectnessValidator::new(&question_test_files.index_path, &question_test_files.vector_path).await?;
    
    let question_test_cases = vec![
        // Single character wildcards
        ("test?", vec!["test1", "test2", "testA", "testB"], "test with single char"),
        ("func?", vec!["func1", "func2", "funcA", "funcB"], "func with single char"),
        ("?est", vec!["test", "best", "nest", "rest"], "single char before est"),
        
        // Multiple question marks
        ("test??", vec!["test01", "test02", "testAB", "testXY"], "test with two chars"),
        ("??st", vec!["test", "best", "nest", "fast"], "two chars before st"),
        ("f?n?", vec!["func", "find", "font", "fang"], "f_n_ pattern"),
        
        // Mixed with literals
        ("test_?", vec!["test_1", "test_2", "test_A", "test_B"], "test_ with single char"),
        ("func_??.rs", vec!["func_01.rs", "func_02.rs", "func_AB.rs"], "func with two chars and extension"),
        
        // Version numbers
        ("version_?.?", vec!["version_1.0", "version_2.1", "version_3.5"], "version patterns"),
        ("v?.?.?", vec!["v1.0.0", "v2.1.3", "v3.5.7"], "semantic version patterns"),
        
        // File naming patterns
        ("test_case_??.rs", vec!["test_case_01.rs", "test_case_02.rs", "test_case_AB.rs"], "numbered test cases"),
        ("module_?.py", vec!["module_1.py", "module_2.py", "module_A.py"], "module patterns"),
    ];
    
    for (query, expected_files, description) in question_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::WildcardSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.90,
            recall_threshold: 0.85,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed question mark wildcard test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Question mark wildcards test completed successfully");
    Ok(())
}

fn matches_wildcard_pattern(text: &str, pattern: &str) -> bool {
    // Simple wildcard matching implementation
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    
    wildcard_match(&text_chars, &pattern_chars, 0, 0)
}

fn wildcard_match(text: &[char], pattern: &[char], text_idx: usize, pattern_idx: usize) -> bool {
    if pattern_idx == pattern.len() {
        return text_idx == text.len();
    }
    
    if pattern_idx < pattern.len() && pattern[pattern_idx] == '*' {
        // Try matching zero or more characters
        for i in text_idx..=text.len() {
            if wildcard_match(text, pattern, i, pattern_idx + 1) {
                return true;
            }
        }
        false
    } else if text_idx < text.len() && 
              (pattern[pattern_idx] == '?' || pattern[pattern_idx] == text[text_idx]) {
        wildcard_match(text, pattern, text_idx + 1, pattern_idx + 1)
    } else {
        false
    }
}
```

## Success Criteria
- Asterisk (*) wildcards match zero or more characters correctly
- Question mark (?) wildcards match exactly one character
- Complex wildcard patterns with mixed * and ? work
- File extension wildcards work properly
- Version number patterns match correctly
- Performance acceptable for wildcard searches
- Precision >= 85% for asterisk patterns
- Precision >= 90% for question mark patterns
- All wildcard combinations tested successfully

## Time Limit
10 minutes maximum