# Task 055: Vector Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates vector test cases that validates semantic similarity search using vector embeddings.

## Project Structure
tests/
  vector_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for vector similarity search including semantic matching, similarity thresholds, and vector space operations.

## Requirements
1. Create comprehensive integration test
2. Test vector similarity search
3. Validate semantic matching accuracy
4. Handle similarity thresholds correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_semantic_similarity_search() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let vector_test_files = test_generator.generate_vector_test_files().await?;
    let validator = CorrectnessValidator::new(&vector_test_files.index_path, &vector_test_files.vector_path).await?;
    
    let semantic_test_cases = vec![
        // Semantic similarity
        ("vector:function implementation", vec!["function_impl.rs", "method_definitions.py"], "function implementation semantics"),
        ("vector:error handling", vec!["error_handling.rs", "exception_management.py"], "error handling semantics"),
        ("vector:database query", vec!["db_queries.rs", "sql_operations.py"], "database query semantics"),
        ("vector:file processing", vec!["file_io.rs", "document_processing.py"], "file processing semantics"),
        
        // Synonyms and related terms
        ("vector:test validation", vec!["test_cases.rs", "validation_tests.py"], "testing concepts"),
        ("vector:authentication login", vec!["auth.rs", "login_system.py"], "authentication concepts"),
        ("vector:serialize encode", vec!["serialization.rs", "encoding.py"], "data transformation"),
        ("vector:thread concurrency", vec!["threading.rs", "concurrent_processing.py"], "concurrency concepts"),
        
        // Technical concepts
        ("vector:machine learning AI", vec!["ml_models.py", "ai_training.rs"], "AI/ML concepts"),
        ("vector:blockchain cryptocurrency", vec!["blockchain.rs", "crypto_wallet.py"], "blockchain concepts"),
        ("vector:web server HTTP", vec!["web_server.rs", "http_handlers.py"], "web server concepts"),
        ("vector:microservices distributed", vec!["microservices.rs", "distributed_systems.py"], "distributed systems"),
    ];
    
    for (query, expected_files, description) in semantic_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::VectorSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.75,
            recall_threshold: 0.70,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed semantic similarity test {}: {} - {}", description, query, result.summary());
        
        // Verify semantic relevance
        for file_result in &result.actual_files {
            assert!(file_result.similarity_score >= 0.6,
                   "Similarity score too low: {} for file {}", file_result.similarity_score, file_result.filename);
        }
    }
    
    println!("Semantic similarity search test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_similarity_thresholds() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let threshold_test_files = test_generator.generate_threshold_test_files().await?;
    let validator = CorrectnessValidator::new(&threshold_test_files.index_path, &threshold_test_files.vector_path).await?;
    
    let threshold_test_cases = vec![
        // High similarity threshold
        ("vector:function@0.9", vec!["exact_function_match.rs"], "high similarity threshold"),
        ("vector:error@0.8", vec!["close_error_match.rs", "error_handling.rs"], "medium-high threshold"),
        
        // Medium similarity threshold  
        ("vector:test@0.6", vec!["test_cases.rs", "testing.py", "validation.js"], "medium threshold"),
        ("vector:data@0.5", vec!["data_processing.rs", "data_structures.py"], "medium-low threshold"),
        
        // Low similarity threshold
        ("vector:code@0.3", vec!["code_files.rs", "programming.py", "scripts.js"], "low threshold"),
    ];
    
    for (query, expected_files, description) in threshold_test_cases {
        let parts: Vec<&str> = query.split('@').collect();
        let base_query = parts[0];
        let threshold: f32 = parts[1].parse().unwrap();
        
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::VectorSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed threshold test {}: {} - {}", description, query, result.summary());
        
        // Verify all results meet the threshold
        for file_result in &result.actual_files {
            assert!(file_result.similarity_score >= threshold,
                   "Result {} has similarity {} below threshold {}", 
                   file_result.filename, file_result.similarity_score, threshold);
        }
    }
    
    println!("Similarity thresholds test completed successfully");
    Ok(())
}
```

## Success Criteria
- Semantic similarity search finds related concepts
- Similarity thresholds are respected correctly
- Vector embeddings provide meaningful results
- Related terms and synonyms are matched
- Technical concepts are grouped semantically
- Precision >= 75% for vector searches
- Recall >= 70% for semantic matching
- Similarity scores are within expected ranges
- Performance acceptable for vector operations

## Time Limit
10 minutes maximum