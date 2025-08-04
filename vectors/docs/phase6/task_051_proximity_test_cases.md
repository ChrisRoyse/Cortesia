# Task 051: Proximity Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates proximity test cases that validates the system correctly handles NEAR and WITHIN proximity search operators.

## Project Structure
tests/
  proximity_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for proximity search including NEAR operators, WITHIN distance specifications, and complex proximity combinations.

## Requirements
1. Create comprehensive integration test
2. Test all proximity search scenarios
3. Validate search accuracy with proximity operators
4. Handle distance specifications correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_near_operator() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let proximity_test_files = test_generator.generate_proximity_test_files().await?;
    let validator = CorrectnessValidator::new(&proximity_test_files.index_path, &proximity_test_files.vector_path).await?;
    
    let near_test_cases = vec![
        // Basic NEAR operations
        ("function NEAR main", vec!["main_functions.rs"], "function near main"),
        ("struct NEAR impl", vec!["struct_implementations.rs"], "struct near impl"),
        ("error NEAR handling", vec!["error_handling.rs"], "error near handling"),
        ("async NEAR await", vec!["async_code.rs"], "async near await"),
        ("test NEAR case", vec!["test_cases.rs"], "test near case"),
        
        // Programming constructs NEAR
        ("pub NEAR fn", vec!["public_functions.rs"], "pub near fn"),
        ("let NEAR mut", vec!["mutable_variables.rs"], "let near mut"),
        ("match NEAR arm", vec!["pattern_matching.rs"], "match near arm"),
        ("use NEAR std", vec!["std_usage.rs"], "use near std"),
        ("impl NEAR trait", vec!["trait_implementations.rs"], "impl near trait"),
        
        // Technical terms NEAR
        ("http NEAR request", vec!["http_requests.rs"], "http near request"),
        ("json NEAR parse", vec!["json_parsing.rs"], "json near parse"),
        ("database NEAR connection", vec!["db_connections.rs"], "database near connection"),
        ("thread NEAR safe", vec!["thread_safety.rs"], "thread near safe"),
        ("regex NEAR pattern", vec!["regex_patterns.rs"], "regex near pattern"),
    ];
    
    for (query, expected_files, description) in near_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::ProximitySearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed NEAR test {}: {} - {}", description, query, result.summary());
        
        // Verify proximity - terms should be close to each other
        let terms: Vec<&str> = query.split(" NEAR ").collect();
        for file_result in &result.actual_files {
            let positions = find_term_positions(&file_result.content, &terms);
            assert!(are_terms_close(&positions, 10), // Default NEAR distance
                   "Terms not close enough in {} for NEAR query {}", file_result.filename, query);
        }
    }
    
    println!("NEAR operator test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_within_distance_operator() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let within_test_files = test_generator.generate_within_test_files().await?;
    let validator = CorrectnessValidator::new(&within_test_files.index_path, &within_test_files.vector_path).await?;
    
    let within_test_cases = vec![
        // WITHIN with specific distances
        ("function WITHIN 5 main", vec!["close_functions.rs"], "function within 5 words of main"),
        ("struct WITHIN 3 field", vec!["struct_fields.rs"], "struct within 3 words of field"),
        ("error WITHIN 2 handling", vec!["error_handling.rs"], "error within 2 words of handling"),
        ("async WITHIN 1 await", vec!["async_await.rs"], "async within 1 word of await"),
        
        // Larger distances
        ("function WITHIN 20 implementation", vec!["function_impl.rs"], "function within 20 words of implementation"),
        ("class WITHIN 15 method", vec!["class_methods.py"], "class within 15 words of method"),
        ("database WITHIN 10 query", vec!["db_queries.rs"], "database within 10 words of query"),
        
        // Very close proximity
        ("pub WITHIN 1 fn", vec!["public_functions.rs"], "pub within 1 word of fn"),
        ("let WITHIN 1 mut", vec!["mutable_vars.rs"], "let within 1 word of mut"),
        ("use WITHIN 2 std", vec!["std_imports.rs"], "use within 2 words of std"),
        
        // Technical terms with distance
        ("http WITHIN 5 client", vec!["http_clients.rs"], "http within 5 words of client"),
        ("json WITHIN 3 serialize", vec!["json_serialization.rs"], "json within 3 words of serialize"),
        ("thread WITHIN 4 spawn", vec!["thread_spawning.rs"], "thread within 4 words of spawn"),
    ];
    
    for (query, expected_files, description) in within_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::ProximitySearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed WITHIN test {}: {} - {}", description, query, result.summary());
        
        // Verify distance constraint
        let parts: Vec<&str> = query.split(" WITHIN ").collect();
        let first_term = parts[0];
        let distance_and_term: Vec<&str> = parts[1].split(' ').collect();
        let distance: usize = distance_and_term[0].parse().unwrap();
        let second_term = distance_and_term[1];
        
        for file_result in &result.actual_files {
            let positions = find_term_positions(&file_result.content, &[first_term, second_term]);
            assert!(are_terms_within_distance(&positions, distance),
                   "Terms not within distance {} in {} for WITHIN query {}", distance, file_result.filename, query);
        }
    }
    
    println!("WITHIN distance operator test completed successfully");
    Ok(())
}

fn find_term_positions(content: &str, terms: &[&str]) -> Vec<Vec<usize>> {
    let mut positions = Vec::new();
    let words: Vec<&str> = content.split_whitespace().collect();
    
    for term in terms {
        let mut term_positions = Vec::new();
        for (i, word) in words.iter().enumerate() {
            if word.to_lowercase().contains(&term.to_lowercase()) {
                term_positions.push(i);
            }
        }
        positions.push(term_positions);
    }
    
    positions
}

fn are_terms_close(positions: &[Vec<usize>], max_distance: usize) -> bool {
    if positions.len() < 2 || positions.iter().any(|p| p.is_empty()) {
        return false;
    }
    
    for pos1 in &positions[0] {
        for pos2 in &positions[1] {
            if pos1.abs_diff(*pos2) <= max_distance {
                return true;
            }
        }
    }
    false
}

fn are_terms_within_distance(positions: &[Vec<usize>], max_distance: usize) -> bool {
    are_terms_close(positions, max_distance)
}
```

## Success Criteria
- NEAR operator finds terms in close proximity (default 10 words)
- WITHIN operator respects specified distance constraints  
- Proximity calculations work correctly for all distances
- Complex proximity queries with multiple terms work
- Performance acceptable for proximity searches
- Edge cases with very close/far distances handled
- Precision >= 85% for proximity operations
- Distance verification works for all test cases

## Time Limit
10 minutes maximum