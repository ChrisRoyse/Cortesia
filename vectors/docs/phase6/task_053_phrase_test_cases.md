# Task 053: Phrase Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates phrase test cases that validates exact phrase matching with proper word order and boundaries.

## Project Structure
tests/
  phrase_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for exact phrase matching including quoted phrases, multi-word expressions, and phrase boundary detection.

## Requirements
1. Create comprehensive integration test
2. Test exact phrase matching
3. Validate word order and boundaries
4. Handle quoted and unquoted phrases
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_quoted_phrase_search() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let phrase_test_files = test_generator.generate_phrase_test_files().await?;
    let validator = CorrectnessValidator::new(&phrase_test_files.index_path, &phrase_test_files.vector_path).await?;
    
    let quoted_phrase_test_cases = vec![
        // Basic quoted phrases
        ("\"hello world\"", vec!["greetings.rs", "examples.py"], "hello world phrase"),
        ("\"function main\"", vec!["main_functions.rs", "entry_points.py"], "function main phrase"),
        ("\"error handling\"", vec!["error_handling.rs", "exception_handling.py"], "error handling phrase"),
        ("\"test case\"", vec!["test_cases.rs", "unit_tests.py"], "test case phrase"),
        
        // Programming phrases
        ("\"pub fn main\"", vec!["main_functions.rs"], "public main function"),
        ("\"let mut variable\"", vec!["mutable_variables.rs"], "mutable variable declaration"),
        ("\"async fn test\"", vec!["async_tests.rs"], "async test function"),
        ("\"struct implementation\"", vec!["struct_impl.rs"], "struct implementation"),
        
        // Technical phrases
        ("\"database connection\"", vec!["db_connection.rs", "database.py"], "database connection phrase"),
        ("\"http request\"", vec!["http_client.rs", "web_requests.py"], "http request phrase"),
        ("\"json parsing\"", vec!["json_parser.rs", "json_utils.py"], "json parsing phrase"),
        ("\"thread safety\"", vec!["thread_safe.rs", "concurrency.py"], "thread safety phrase"),
        
        // Long phrases
        ("\"this is a long phrase for testing\"", vec!["long_phrases.txt"], "long test phrase"),
        ("\"complex multi word phrase with many terms\"", vec!["complex_phrases.txt"], "complex phrase"),
        
        // Phrases with special characters
        ("\"function() -> Result<T, E>\"", vec!["result_functions.rs"], "function signature phrase"),
        ("\"std::collections::HashMap\"", vec!["hashmap_usage.rs"], "HashMap path phrase"),
        ("\"#[derive(Debug, Clone)]\"", vec!["derive_macros.rs"], "derive attribute phrase"),
    ];
    
    for (query, expected_files, description) in quoted_phrase_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::PhraseSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.95,
            recall_threshold: 0.90,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed quoted phrase test {}: {} - {}", description, query, result.summary());
        
        // Verify exact phrase matching
        let phrase = query.trim_matches('"');
        for file_result in &result.actual_files {
            assert!(file_result.content.contains(phrase),
                   "Exact phrase '{}' not found in {}", phrase, file_result.filename);
        }
    }
    
    println!("Quoted phrase search test completed successfully");
    Ok(())
}
```

## Success Criteria
- Exact phrase matching with correct word order
- Quoted phrases handled properly
- Word boundaries respected
- Special characters in phrases work
- Precision >= 95% for phrase searches
- Recall >= 90% for phrase matching
- Long phrases matched correctly
- Case sensitivity handled appropriately

## Time Limit
10 minutes maximum