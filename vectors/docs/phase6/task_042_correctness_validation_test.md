# Task 042: Correctness Validation Test

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates correctness validation tests that validates the search engine accuracy meets production requirements with real data and edge cases.

## Project Structure
tests/
  correctness_validation_test.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive integration test that validates search result correctness across all query types, file formats, and edge cases using ground truth datasets.

## Requirements
1. Create comprehensive integration test
2. Test search result accuracy with ground truth data
3. Validate precision, recall, and F1 scores meet thresholds
4. Handle error conditions gracefully
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::path::PathBuf;

#[tokio::test]
async fn test_correctness_validation_comprehensive() -> Result<()> {
    // Setup test environment
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate ground truth test cases
    let ground_truth_cases = test_generator.generate_correctness_test_cases()?;
    
    // Initialize validation system
    let validator = CorrectnessValidator::new(&index_path, &vector_path).await?;
    
    // Execute validation for each test case
    let mut results = Vec::new();
    for case in ground_truth_cases {
        let result = validator.validate(&case).await?;
        results.push(result);
        
        // Assert minimum accuracy thresholds
        assert!(result.precision >= 0.90, "Precision below threshold: {}", result.precision);
        assert!(result.recall >= 0.85, "Recall below threshold: {}", result.recall);
        assert!(result.f1_score >= 0.87, "F1 score below threshold: {}", result.f1_score);
    }
    
    // Generate comprehensive report
    let report = ValidationReport::from_results(results)?;
    println!("Correctness validation report: {}", report);
    
    Ok(())
}

#[tokio::test]
async fn test_special_characters_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    let test_files = test_generator.generate_special_characters_tests()?;
    
    // Initialize search system
    let validator = CorrectnessValidator::new(&index_path, &vector_path).await?;
    
    // Test each special character case
    let test_cases = vec![
        ("[workspace]", vec!["Cargo.toml"], true, "workspace configuration"),
        ("Result<T, E>", vec!["lib.rs"], true, "generic type syntax"),
        ("fn main() {", vec!["main.rs"], true, "function definition"),
        ("\"hello world\"", vec!["strings.rs"], true, "quoted strings"),
        ("regex: /\\d+/", vec!["regex.rs"], true, "regex patterns"),
        ("C:\\Windows\\System32", vec!["paths.rs"], true, "Windows paths"),
        ("π ≈ 3.14159", vec!["math.rs"], true, "Unicode symbols"),
        ("NULL && true", vec!["logic.rs"], true, "boolean operators"),
        ("array[0] = value", vec!["arrays.rs"], true, "array indexing"),
        ("@annotation", vec!["annotations.rs"], true, "annotation syntax"),
    ];
    
    for (query, expected_files, should_pass, description) in test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.90,
            recall_threshold: 0.85,
        };
        
        let result = validator.validate(&case).await?;
        
        if should_pass {
            assert!(result.is_correct, "Failed {}: {}", description, result.summary());
            assert!(result.precision >= 0.90, "Low precision for {}: {}", description, result.precision);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_edge_case_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let validator = CorrectnessValidator::new(&temp_dir.path(), &temp_dir.path()).await?;
    
    // Test empty query
    let empty_case = GroundTruthCase {
        query: "".to_string(),
        expected_files: vec![],
        query_type: QueryType::TextSearch,
        expected_count: 0,
        precision_threshold: 1.0,
        recall_threshold: 1.0,
    };
    
    let result = validator.validate(&empty_case).await?;
    assert!(result.is_correct, "Empty query should be handled correctly");
    
    // Test very long query
    let long_query = "a".repeat(10000);
    let long_case = GroundTruthCase {
        query: long_query,
        expected_files: vec![],
        query_type: QueryType::TextSearch,
        expected_count: 0,
        precision_threshold: 1.0,
        recall_threshold: 1.0,
    };
    
    let result = validator.validate(&long_case).await?;
    assert!(result.is_correct, "Long query should be handled correctly");
    
    // Test query with only special characters
    let special_case = GroundTruthCase {
        query: "!@#$%^&*()_+-=[]{}|;':\",./<>?".to_string(),
        expected_files: vec![],
        query_type: QueryType::TextSearch,
        expected_count: 0,
        precision_threshold: 1.0,
        recall_threshold: 1.0,
    };
    
    let result = validator.validate(&special_case).await?;
    assert!(result.is_correct, "Special characters query should be handled correctly");
    
    Ok(())
}

#[tokio::test]
async fn test_multi_language_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate multilingual test files
    let test_files = test_generator.generate_multilingual_tests()?;
    let validator = CorrectnessValidator::new(&test_files.index_path, &test_files.vector_path).await?;
    
    let test_cases = vec![
        ("print", vec!["python_file.py", "rust_file.rs"], "print function"),
        ("function", vec!["javascript_file.js", "python_file.py"], "function keyword"),
        ("class", vec!["java_file.java", "python_file.py"], "class definition"),
        ("import", vec!["python_file.py", "java_file.java"], "import statement"),
        ("struct", vec!["rust_file.rs", "c_file.c"], "struct definition"),
    ];
    
    for (query, expected_files, description) in test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed multilingual test {}: {}", description, result.summary());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_file_format_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate files in different formats
    let test_files = test_generator.generate_format_tests()?;
    let validator = CorrectnessValidator::new(&test_files.index_path, &test_files.vector_path).await?;
    
    let format_tests = vec![
        ("TODO", vec!["comments.rs", "readme.md", "notes.txt"], "comment search"),
        ("function", vec!["code.rs", "script.py", "program.js"], "code search"),
        ("configuration", vec!["config.toml", "settings.json", "env.yaml"], "config search"),
        ("documentation", vec!["readme.md", "docs.txt", "help.html"], "docs search"),
    ];
    
    for (query, expected_files, description) in format_tests {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.90,
            recall_threshold: 0.85,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed format test {}: {}", description, result.summary());
        assert!(result.precision >= 0.90, "Low precision for {}: {}", description, result.precision);
    }
    
    Ok(())
}
```

## Success Criteria
- All correctness validation tests pass with 100% success rate
- Precision scores >= 90% for all test categories
- Recall scores >= 85% for all test categories
- F1 scores >= 87% for all test categories
- Edge cases handled gracefully without panics
- Multi-language and multi-format searches work correctly
- Comprehensive validation report generated

## Time Limit
10 minutes maximum