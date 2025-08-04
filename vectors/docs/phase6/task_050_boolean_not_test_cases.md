# Task 050: Boolean NOT Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates boolean NOT test cases that validates the system correctly handles NOT logic for exclusion operations.

## Project Structure
tests/
  boolean_not_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for boolean NOT operations including simple NOT queries, exclusion logic, and complex NOT combinations with AND/OR operations.

## Requirements
1. Create comprehensive integration test
2. Test all boolean NOT scenarios  
3. Validate search accuracy with NOT logic (exclusion)
4. Handle precedence and grouping correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::collections::HashSet;

#[tokio::test]
async fn test_simple_not_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let not_test_files = test_generator.generate_boolean_not_test_files().await?;
    let validator = CorrectnessValidator::new(&not_test_files.index_path, &not_test_files.vector_path).await?;
    
    let simple_not_test_cases = vec![
        // Basic NOT operations
        ("function NOT test", vec!["functions.rs", "function_impl.rs"], "function but not test"),
        ("struct NOT impl", vec!["struct_definitions.rs", "data_types.rs"], "struct but not impl"),
        ("error NOT handling", vec!["error_types.rs", "error_definitions.rs"], "error but not handling"),
        ("async NOT sync", vec!["async_operations.rs", "futures.rs"], "async but not sync"),
        ("public NOT private", vec!["public_api.rs", "public_functions.rs"], "public but not private"),
        
        // Programming terms NOT
        ("class NOT interface", vec!["classes.py", "class_definitions.java"], "class but not interface"),
        ("method NOT function", vec!["methods.py", "class_methods.java"], "method but not function"),
        ("import NOT export", vec!["imports.py", "import_statements.rs"], "import but not export"),
        ("variable NOT constant", vec!["variables.rs", "var_declarations.py"], "variable but not constant"),
        
        // Technical NOT operations
        ("database NOT cache", vec!["database.rs", "db_operations.py"], "database but not cache"),
        ("client NOT server", vec!["client.rs", "client_code.py"], "client but not server"),
        ("read NOT write", vec!["read_operations.rs", "file_reading.py"], "read but not write"),
        ("GET NOT POST", vec!["http_get.rs", "get_requests.py"], "GET but not POST"),
    ];
    
    for (query, expected_files, description) in simple_not_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed simple NOT test {}: {} - {}", description, query, result.summary());
        
        // Verify NOT exclusion behavior
        let parts: Vec<&str> = query.split(" NOT ").collect();
        let include_term = parts[0];
        let exclude_term = parts[1];
        
        for file_result in &result.actual_files {
            assert!(file_result.content.to_lowercase().contains(&include_term.to_lowercase()),
                   "Include term '{}' missing in {} for NOT query", include_term, file_result.filename);
            assert!(!file_result.content.to_lowercase().contains(&exclude_term.to_lowercase()),
                   "Exclude term '{}' found in {} for NOT query", exclude_term, file_result.filename);
        }
    }
    
    println!("Simple NOT operations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_not_with_and_or_combinations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let complex_not_test_files = test_generator.generate_complex_not_test_files().await?;
    let validator = CorrectnessValidator::new(&complex_not_test_files.index_path, &complex_not_test_files.vector_path).await?;
    
    let complex_not_test_cases = vec![
        // NOT with AND
        ("function AND async NOT test", vec!["async_functions.rs"], "async function but not test"),
        ("struct AND field NOT method", vec!["struct_fields.rs"], "struct field but not method"),
        ("error AND handling NOT panic", vec!["error_handling.rs"], "error handling but not panic"),
        
        // NOT with OR
        ("(function OR method) NOT test", vec!["functions.rs", "methods.py"], "function or method but not test"),
        ("(read OR write) NOT cache", vec!["file_io.rs", "io_operations.py"], "read or write but not cache"),
        ("(GET OR POST) NOT auth", vec!["http_methods.rs", "api_endpoints.py"], "GET or POST but not auth"),
        
        // Complex combinations
        ("function AND (async OR sync) NOT test", vec!["production_functions.rs"], "function and async/sync but not test"),
        ("(struct OR class) AND field NOT method", vec!["data_structures.rs"], "struct/class with field but not method"),
        ("database AND (read OR write) NOT transaction", vec!["db_operations.rs"], "database read/write but not transaction"),
        
        // Multiple NOT terms
        ("function NOT test NOT deprecated", vec!["current_functions.rs"], "function but not test or deprecated"),
        ("struct NOT interface NOT abstract", vec!["concrete_structs.rs"], "struct but not interface or abstract"),
        
        // Parenthesized NOT
        ("function NOT (test OR spec)", vec!["production_functions.rs"], "function but not test or spec"),
        ("struct NOT (impl OR trait)", vec!["plain_structs.rs"], "struct but not impl or trait"),
        ("async NOT (test OR mock)", vec!["production_async.rs"], "async but not test or mock"),
    ];
    
    for (query, expected_files, description) in complex_not_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed complex NOT test {}: {} - {}", description, query, result.summary());
    }
    
    println!("NOT with AND/OR combinations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_not_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let edge_case_test_files = test_generator.generate_not_edge_case_test_files().await?;
    let validator = CorrectnessValidator::new(&edge_case_test_files.index_path, &edge_case_test_files.vector_path).await?;
    
    let edge_case_test_cases = vec![
        // NOT with special characters
        ("function() NOT test()", vec!["functions.rs"], "function calls but not test calls"),
        ("struct{} NOT class{}", vec!["structs.rs"], "struct blocks but not class blocks"),
        
        // NOT with numbers
        ("version NOT 1.0", vec!["version_2.rs", "latest_version.py"], "version but not 1.0"),
        ("test NOT 123", vec!["test_abc.rs", "test_xyz.py"], "test but not 123"),
        
        // Case sensitivity NOT
        ("Function NOT TEST", vec!["functions.rs"], "Function but not TEST"),
        ("function NOT Function", vec!["function_impl.rs"], "function but not Function"),
        
        // NOT with very common terms
        ("code NOT the", vec!["code_samples.rs"], "code but not 'the'"),
        ("function NOT a", vec!["functions.rs"], "function but not 'a'"),
        
        // NOT with non-existent terms  
        ("function NOT nonexistent_xyz123", vec!["functions.rs", "function_impl.rs"], "function but not nonexistent"),
        ("existing_code NOT impossible_term", vec!["existing_code.rs"], "existing but not impossible"),
        
        // Empty NOT results
        ("nonexistent_term NOT anything", vec![], "nonexistent term should return empty"),
        
        // NOT with Unicode
        ("código NOT test", vec!["spanish_code.py"], "Spanish code but not test"),
        ("函数 NOT 测试", vec!["chinese_functions.py"], "Chinese function but not test"),
    ];
    
    for (query, expected_files, description) in edge_case_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: if expected_files.is_empty() { 1.0 } else { 0.75 },
            recall_threshold: if expected_files.is_empty() { 1.0 } else { 0.70 },
        };
        
        let result = validator.validate(&case).await?;
        
        if expected_files.is_empty() {
            assert!(result.actual_files.is_empty(),
                   "Expected no results for edge case {}: {}", description, query);
        } else {
            assert!(result.is_correct, "Failed edge case NOT test {}: {} - {}", description, query, result.summary());
        }
    }
    
    println!("NOT edge cases test completed successfully");
    Ok(())
}
```

## Success Criteria
- Simple NOT queries correctly exclude specified terms
- Complex NOT with AND/OR combinations work properly
- NOT exclusion behavior is verified in all results
- Edge cases with special characters and Unicode work
- Empty results handled properly for impossible queries
- Performance acceptable for NOT operations
- Case sensitivity handled correctly
- Parenthesized NOT expressions work properly

## Time Limit
10 minutes maximum