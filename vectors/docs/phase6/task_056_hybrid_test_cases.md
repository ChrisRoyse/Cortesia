# Task 056: Hybrid Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates hybrid test cases that validates combined text and vector search with result fusion.

## Project Structure
tests/
  hybrid_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for hybrid search combining text search and vector similarity with reciprocal rank fusion (RRF).

## Requirements
1. Create comprehensive integration test
2. Test hybrid search combining text and vector
3. Validate result fusion accuracy
4. Handle ranking and scoring correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_hybrid_text_vector_search() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let hybrid_test_files = test_generator.generate_hybrid_test_files().await?;
    let validator = CorrectnessValidator::new(&hybrid_test_files.index_path, &hybrid_test_files.vector_path).await?;
    
    let hybrid_test_cases = vec![
        // Text + Vector combination
        ("hybrid:function AND vector:implementation", 
         vec!["function_impl.rs", "method_implementations.py"], "function with implementation semantics"),
        ("hybrid:error OR vector:exception handling", 
         vec!["error_handling.rs", "exception_management.py"], "error or exception semantics"),
        ("hybrid:\"test case\" AND vector:validation", 
         vec!["test_validation.rs", "validation_tests.py"], "test case with validation semantics"),
        
        // Complex hybrid queries
        ("hybrid:(database AND connection) OR vector:persistence storage", 
         vec!["db_connections.rs", "storage_systems.py"], "database or storage concepts"),
        ("hybrid:async AND vector:concurrency NOT test", 
         vec!["async_production.rs", "concurrent_systems.py"], "async concurrency not test"),
        
        // Weighted hybrid search
        ("hybrid:function[0.7] + vector:implementation[0.3]", 
         vec!["function_impl.rs"], "weighted function implementation"),
        ("hybrid:error[0.4] + vector:handling exception[0.6]", 
         vec!["error_handling.rs"], "weighted error handling"),
    ];
    
    for (query, expected_files, description) in hybrid_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::HybridSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed hybrid test {}: {} - {}", description, query, result.summary());
        
        // Verify hybrid scoring
        for file_result in &result.actual_files {
            assert!(file_result.hybrid_score > 0.0,
                   "Hybrid score should be positive for {}", file_result.filename);
            assert!(file_result.hybrid_score <= 1.0,
                   "Hybrid score should be <= 1.0 for {}", file_result.filename);
        }
    }
    
    println!("Hybrid text-vector search test completed successfully");
    Ok(())
}
```

## Success Criteria
- Hybrid search combines text and vector results effectively
- RRF algorithm produces meaningful rankings
- Weighted queries respect weight parameters
- Complex hybrid queries work correctly
- Precision >= 80% for hybrid searches
- Recall >= 75% for combined results
- Hybrid scores are properly calculated
- Performance acceptable for hybrid operations

## Time Limit
10 minutes maximum