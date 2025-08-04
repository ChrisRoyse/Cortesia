# Task 041: Create Ground Truth Validation Integration Test

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates the main integration test that validates the entire ground truth validation pipeline end-to-end.

## Project Structure
```
tests/
  integration_validation.rs  <- Create this file
Cargo.toml
```

## Task Description
Create a comprehensive integration test that validates the ground truth validation system works correctly with real test data and produces accurate results.

## Requirements
1. Create `tests/integration_validation.rs`
2. Test ground truth dataset loading and validation
3. Test correctness validator with real data
4. Verify accuracy metrics are calculated correctly
5. Test edge cases and error conditions

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::path::PathBuf;

// Import validation modules (adjust paths as needed)
use llmkg::validation::{
    ground_truth::{GroundTruthDataset, GroundTruthCase, QueryType},
    correctness::CorrectnessValidator,
    test_data::TestDataGenerator,
};

#[tokio::test]
async fn test_ground_truth_validation_pipeline() -> Result<()> {
    // Setup test environment
    let temp_dir = TempDir::new()?;
    let test_data_dir = temp_dir.path().join("test_data");
    let text_index_path = temp_dir.path().join("text_index");
    let vector_db_path = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
    
    // Generate test data
    let test_generator = TestDataGenerator::new(&test_data_dir)?;
    let test_set = test_generator.generate_comprehensive_test_set()?;
    
    assert!(test_set.total_files > 0, "No test files generated");
    assert!(test_set.total_size_bytes > 0, "Test files have no content");
    
    // Create ground truth dataset
    let mut dataset = GroundTruthDataset::new();
    
    // Add test cases for special characters
    dataset.add_test(GroundTruthCase {
        query: "[workspace]".to_string(),
        expected_files: vec!["workspace_test.toml".to_string()],
        expected_count: 1,
        must_contain: vec!["[workspace]".to_string()],
        must_not_contain: vec!["[dependencies]".to_string()],
        query_type: QueryType::SpecialCharacters,
    });
    
    // Add test cases for boolean logic
    dataset.add_test(GroundTruthCase {
        query: "pub AND struct".to_string(),
        expected_files: vec!["boolean_and.rs".to_string()],
        expected_count: 1,
        must_contain: vec!["pub".to_string(), "struct".to_string()],
        must_not_contain: vec![],
        query_type: QueryType::BooleanAnd,
    });
    
    // Add test case for wildcards
    dataset.add_test(GroundTruthCase {
        query: "Status::*".to_string(),
        expected_files: vec!["boolean_or.rs".to_string()],
        expected_count: 1,
        must_contain: vec!["Status::".to_string()],
        must_not_contain: vec![],
        query_type: QueryType::Wildcard,
    });
    
    // Save and reload dataset to test serialization
    let dataset_path = temp_dir.path().join("ground_truth.json");
    dataset.save_to_file(&dataset_path)?;
    let reloaded_dataset = GroundTruthDataset::load_from_file(&dataset_path)?;
    
    assert_eq!(dataset.test_cases.len(), reloaded_dataset.test_cases.len());
    
    // Initialize search system and validator
    // Note: This assumes the main search system exists - may need mocking
    let validator = CorrectnessValidator::new(&text_index_path, &vector_db_path).await?;
    
    // Run validation on each test case
    let mut all_passed = true;
    let mut results = Vec::new();
    
    for (i, test_case) in reloaded_dataset.test_cases.iter().enumerate() {
        println!("Running validation test {}/{}: {}", i + 1, reloaded_dataset.test_cases.len(), test_case.query);
        
        let validation_result = validator.validate(test_case).await?;
        
        println!("  Result: {}", validation_result.summary());
        
        // Check if validation passed
        if !validation_result.is_correct {
            all_passed = false;
            println!("  FAILED: {}", validation_result.error_details.join("; "));
        }
        
        results.push(validation_result);
    }
    
    // Calculate overall accuracy
    let passed_count = results.iter().filter(|r| r.is_correct).count();
    let overall_accuracy = (passed_count as f64 / results.len() as f64) * 100.0;
    
    println!("\n=== Ground Truth Validation Results ===");
    println!("Total test cases: {}", results.len());
    println!("Passed: {}", passed_count);
    println!("Failed: {}", results.len() - passed_count);
    println!("Overall accuracy: {:.1}%", overall_accuracy);
    
    // Verify specific metrics
    assert!(results.len() >= 3, "Should have at least 3 test cases");
    
    // For a comprehensive validation system, we want high accuracy
    // In development, we might accept lower accuracy, but log the issues
    if overall_accuracy < 100.0 {
        println!("WARNING: Not all validation tests passed. Check implementation.");
        for (i, result) in results.iter().enumerate() {
            if !result.is_correct {
                println!("Failed test {}: {}", i + 1, result.error_details.join(", "));
            }
        }
    }
    
    // Test error handling
    let invalid_case = GroundTruthCase {
        query: "".to_string(), // Empty query
        expected_files: vec![],
        expected_count: 0,
        must_contain: vec![],
        must_not_contain: vec![],
        query_type: QueryType::SpecialCharacters,
    };
    
    let error_result = validator.validate(&invalid_case).await?;
    // Empty query should either fail or return empty results - both are valid
    
    Ok(())
}

#[tokio::test]
async fn test_validation_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let text_index_path = temp_dir.path().join("text_index");
    let vector_db_path = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
    
    let validator = CorrectnessValidator::new(&text_index_path, &vector_db_path).await?;
    
    // Test case with no expected files
    let no_results_case = GroundTruthCase {
        query: "nonexistent_pattern_xyz123".to_string(),
        expected_files: vec![],
        expected_count: 0,
        must_contain: vec![],
        must_not_contain: vec![],
        query_type: QueryType::SpecialCharacters,
    };
    
    let result = validator.validate(&no_results_case).await?;
    assert_eq!(result.expected_count, 0);
    
    // Test case with content requirements
    let content_case = GroundTruthCase {
        query: "test".to_string(),
        expected_files: vec!["test.rs".to_string()],
        expected_count: 1,
        must_contain: vec!["required_content".to_string()],
        must_not_contain: vec!["forbidden_content".to_string()],
        query_type: QueryType::SpecialCharacters,
    };
    
    let content_result = validator.validate(&content_case).await?;
    // This will likely fail unless we have matching test data, which is expected
    
    Ok(())
}
```

## Success Criteria
- Integration test compiles and runs
- Ground truth dataset serialization/deserialization works
- Validation pipeline processes multiple test cases
- Accuracy calculations are correct
- Edge cases are handled properly
- Test provides meaningful output and error reporting

## Time Limit
10 minutes maximum