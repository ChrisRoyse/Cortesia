# Task 048: Boolean AND Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates boolean AND test cases that validates the system correctly handles AND logic with proper precedence, grouping, and edge cases.

## Project Structure
tests/
  boolean_and_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for boolean AND operations including simple AND queries, precedence handling, grouping with parentheses, and complex multi-term AND combinations.

## Requirements
1. Create comprehensive integration test
2. Test all boolean AND scenarios
3. Validate search accuracy with AND logic
4. Handle precedence and grouping correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_simple_and_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate test files with overlapping content for AND operations
    let and_test_files = test_generator.generate_boolean_and_test_files().await?;
    let validator = CorrectnessValidator::new(&and_test_files.index_path, &and_test_files.vector_path).await?;
    
    let simple_and_test_cases = vec![
        // Basic two-term AND
        ("function AND main", vec!["main.rs", "lib.rs"], "function and main together"),
        ("struct AND impl", vec!["structs.rs", "implementations.rs"], "struct with implementation"),
        ("error AND handling", vec!["error_handling.rs", "result_types.rs"], "error handling code"),
        ("test AND case", vec!["test_files.rs", "unit_tests.rs"], "test case code"),
        ("async AND await", vec!["async_code.rs", "futures.rs"], "async/await patterns"),
        
        // Case insensitive AND
        ("Function AND Main", vec!["main.rs", "lib.rs"], "case insensitive function and main"),
        ("STRUCT AND IMPL", vec!["structs.rs", "implementations.rs"], "uppercase struct and impl"),
        ("Error and Handling", vec!["error_handling.rs", "result_types.rs"], "mixed case error handling"),
        
        // Programming language specific AND
        ("pub AND fn", vec!["public_functions.rs", "library.rs"], "public functions"),
        ("mut AND ref", vec!["mutable_refs.rs", "borrowing.rs"], "mutable references"),
        ("match AND arm", vec!["pattern_matching.rs", "match_expressions.rs"], "match arms"),
        ("use AND mod", vec!["modules.rs", "imports.rs"], "module usage"),
        ("let AND mut", vec!["variables.rs", "mutable_vars.rs"], "mutable variables"),
        
        // Content-specific AND
        ("TODO AND FIXME", vec!["todos.rs", "code_issues.py"], "code annotations"),
        ("import AND export", vec!["modules.js", "imports.py"], "module operations"),
        ("class AND method", vec!["classes.py", "methods.java"], "class methods"),
        ("interface AND implementation", vec!["interfaces.java", "implementations.cpp"], "interface implementations"),
        
        // Technical terms AND
        ("database AND connection", vec!["db_connection.rs", "database.py"], "database connections"),
        ("http AND request", vec!["http_client.rs", "web_requests.py"], "HTTP requests"),
        ("json AND parse", vec!["json_parser.rs", "json_handling.py"], "JSON parsing"),
        ("regex AND pattern", vec!["regex_utils.rs", "pattern_matching.py"], "regex patterns"),
        ("thread AND safety", vec!["thread_safety.rs", "concurrency.cpp"], "thread safety"),
    ];
    
    for (query, expected_files, description) in simple_and_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.90,
            recall_threshold: 0.85,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed simple AND test {}: {} - {}", description, query, result.summary());
        assert!(result.precision >= 0.90, "Low precision for {}: {:.2}", description, result.precision);
        
        // Verify that ALL terms are present in results
        for file_result in &result.actual_files {
            assert!(file_result.content.to_lowercase().contains(&query.split(" AND ").next().unwrap().to_lowercase()),
                   "First AND term missing in {}", file_result.filename);
            assert!(file_result.content.to_lowercase().contains(&query.split(" AND ").last().unwrap().to_lowercase()),
                   "Second AND term missing in {}", file_result.filename);
        }
    }
    
    println!("Simple AND operations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_multi_term_and_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let multi_and_test_files = test_generator.generate_multi_term_and_test_files().await?;
    let validator = CorrectnessValidator::new(&multi_and_test_files.index_path, &multi_and_test_files.vector_path).await?;
    
    let multi_term_and_test_cases = vec![
        // Three-term AND
        ("function AND test AND case", vec!["test_functions.rs", "unit_tests.rs"], "three term function test case"),
        ("struct AND field AND access", vec!["struct_access.rs", "field_operations.rs"], "struct field access"),
        ("error AND result AND handling", vec!["error_results.rs", "error_handling.rs"], "error result handling"),
        ("async AND fn AND await", vec!["async_functions.rs", "futures.rs"], "async function await"),
        ("pub AND static AND ref", vec!["static_refs.rs", "public_statics.rs"], "public static references"),
        
        // Four-term AND
        ("impl AND trait AND for AND struct", vec!["trait_implementations.rs"], "trait impl for struct"),
        ("use AND std AND collections AND HashMap", vec!["hashmap_usage.rs", "collections.rs"], "HashMap usage"),
        ("match AND pattern AND guard AND arm", vec!["pattern_guards.rs", "match_arms.rs"], "match pattern guards"),
        ("fn AND main AND args AND env", vec!["main_args.rs", "command_line.rs"], "main function args"),
        
        // Five-term AND
        ("pub AND fn AND new AND self AND mut", vec!["constructors.rs", "builder_pattern.rs"], "constructor methods"),
        ("async AND fn AND test AND tokio AND main", vec!["async_tests.rs", "tokio_tests.rs"], "async tokio tests"),
        
        // Long AND chains
        ("function AND implementation AND test AND validation AND error AND handling", 
         vec!["comprehensive_tests.rs"], "comprehensive function testing"),
        ("database AND connection AND pool AND async AND transaction AND commit", 
         vec!["db_transactions.rs"], "database transaction handling"),
        
        // Mixed case multi-term AND
        ("Function AND Test AND Case", vec!["test_functions.rs", "unit_tests.rs"], "mixed case three terms"),
        ("STRUCT AND field AND ACCESS", vec!["struct_access.rs", "field_operations.rs"], "mixed case struct access"),
        
        // Technical multi-term AND
        ("http AND client AND request AND response AND json", vec!["http_json.rs", "api_client.rs"], "HTTP JSON client"),
        ("thread AND pool AND executor AND spawn AND join", vec!["thread_pools.rs", "executors.rs"], "thread pool execution"),
        ("regex AND compile AND match AND capture AND group", vec!["regex_captures.rs", "pattern_matching.rs"], "regex capture groups"),
        ("file AND system AND read AND write AND buffer", vec!["file_io.rs", "buffered_io.rs"], "buffered file IO"),
        
        // Domain-specific multi-term AND
        ("machine AND learning AND model AND training AND data", vec!["ml_training.py", "data_models.py"], "ML model training"),
        ("web AND server AND route AND handler AND middleware", vec!["web_routes.rs", "server_handlers.rs"], "web server routing"),
        ("blockchain AND transaction AND hash AND verify AND signature", vec!["blockchain.rs", "crypto_verify.rs"], "blockchain verification"),
    ];
    
    for (query, expected_files, description) in multi_term_and_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed multi-term AND test {}: {} - {}", description, query, result.summary());
        
        // Verify that ALL AND terms are present in results
        let terms: Vec<&str> = query.split(" AND ").collect();
        for file_result in &result.actual_files {
            for term in &terms {
                assert!(file_result.content.to_lowercase().contains(&term.to_lowercase()),
                       "AND term '{}' missing in {}", term, file_result.filename);
            }
        }
    }
    
    println!("Multi-term AND operations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_and_with_parentheses_grouping() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let grouped_and_test_files = test_generator.generate_grouped_and_test_files().await?;
    let validator = CorrectnessValidator::new(&grouped_and_test_files.index_path, &grouped_and_test_files.vector_path).await?;
    
    let grouped_and_test_cases = vec![
        // Simple grouping
        ("(function AND test)", vec!["test_functions.rs", "function_tests.rs"], "grouped function and test"),
        ("(struct AND impl)", vec!["struct_impl.rs", "implementations.rs"], "grouped struct and impl"),
        ("(error AND handling)", vec!["error_handling.rs", "error_types.rs"], "grouped error handling"),
        
        // Nested AND groups
        ("((function AND test) AND (case AND validation))", 
         vec!["test_validation.rs"], "nested AND groups"),
        ("((struct AND field) AND (access AND method))", 
         vec!["struct_methods.rs"], "nested struct access"),
        
        // Mixed grouping with multiple terms
        ("(function AND test AND case)", vec!["test_functions.rs"], "grouped three terms"),
        ("(async AND fn AND await)", vec!["async_functions.rs"], "grouped async function"),
        ("(pub AND static AND mut)", vec!["static_mutable.rs"], "grouped public static mut"),
        
        // Complex nested grouping
        ("(function AND (test AND validation))", 
         vec!["function_test_validation.rs"], "function with nested test validation"),
        ("(struct AND (field AND (access AND method)))", 
         vec!["struct_field_access.rs"], "deeply nested struct operations"),
        ("((database AND connection) AND (pool AND async))", 
         vec!["async_db_pool.rs"], "database connection with async pool"),
        
        // Grouping with technical terms
        ("(http AND (client AND request))", vec!["http_client.rs"], "HTTP client request grouping"),
        ("(json AND (parse AND serialize))", vec!["json_operations.rs"], "JSON parse and serialize"),
        ("(thread AND (safe AND atomic))", vec!["thread_safety.rs"], "thread safe atomic operations"),
        ("(regex AND (compile AND match))", vec!["regex_operations.rs"], "regex compile and match"),
        
        // Multiple separate AND groups
        ("(function AND test) AND (error AND handling)", 
         vec!["test_error_handling.rs"], "separate function test and error handling groups"),
        ("(struct AND impl) AND (trait AND bound)", 
         vec!["trait_bounds.rs"], "struct impl with trait bounds"),
        ("(async AND await) AND (future AND poll)", 
         vec!["future_polling.rs"], "async await with future polling"),
        
        // Deeply nested complex grouping
        ("((function AND test) AND ((validation AND error) AND handling))", 
         vec!["comprehensive_testing.rs"], "deeply nested function test validation"),
        ("(((http AND client) AND request) AND ((json AND response) AND parse))", 
         vec!["http_json_client.rs"], "deeply nested HTTP JSON operations"),
    ];
    
    for (query, expected_files, description) in grouped_and_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed grouped AND test {}: {} - {}", description, query, result.summary());
        
        // Extract all terms from grouped query and verify presence
        let terms = extract_and_terms_from_grouped_query(&query);
        for file_result in &result.actual_files {
            for term in &terms {
                assert!(file_result.content.to_lowercase().contains(&term.to_lowercase()),
                       "Grouped AND term '{}' missing in {}", term, file_result.filename);
            }
        }
    }
    
    println!("AND with parentheses grouping test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_and_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let edge_case_test_files = test_generator.generate_and_edge_case_test_files().await?;
    let validator = CorrectnessValidator::new(&edge_case_test_files.index_path, &edge_case_test_files.vector_path).await?;
    
    let edge_case_test_cases = vec![
        // AND with special characters
        ("function() AND main()", vec!["function_calls.rs"], "function calls with parentheses"),
        ("struct{} AND impl{}", vec!["struct_blocks.rs"], "struct and impl blocks"),
        ("\"string\" AND 'char'", vec!["string_literals.rs"], "string and char literals"),
        ("array[0] AND map[key]", vec!["array_map_access.rs"], "array and map access"),
        ("regex:/.+/ AND pattern:/\\d+/", vec!["regex_patterns.rs"], "regex patterns"),
        
        // AND with numbers
        ("version AND 2.0", vec!["version_2.rs", "v2_features.py"], "version with number"),
        ("test AND 123", vec!["test_123.rs", "numbered_tests.py"], "test with number"),
        ("port AND 8080", vec!["port_8080.rs", "server_config.py"], "port number"),
        
        // AND with underscores and hyphens
        ("snake_case AND camelCase", vec!["naming_conventions.rs"], "different naming cases"),
        ("kebab-case AND snake_case", vec!["style_guides.md"], "hyphenated and underscored"),
        ("multi_word_function AND single_param", vec!["function_params.rs"], "multi-word identifiers"),
        
        // AND with file extensions
        (".rs AND .py", vec!["polyglot_project.md"], "file extensions"),
        ("Cargo.toml AND package.json", vec!["multi_lang_config.md"], "config files"),
        ("README.md AND LICENSE", vec!["project_docs.md"], "documentation files"),
        
        // Empty and whitespace edge cases
        ("  function  AND  test  ", vec!["test_functions.rs"], "query with extra whitespace"),
        ("function\tAND\ttest", vec!["test_functions.rs"], "query with tabs"),
        ("function\nAND\ntest", vec!["test_functions.rs"], "query with newlines"),
        
        // Case sensitivity edge cases
        ("and AND AND", vec!["boolean_logic.rs"], "and keyword with AND operator"),
        ("Function AND function", vec!["mixed_case_functions.rs"], "same word different cases"),
        ("TEST AND test AND Test", vec!["case_variations.rs"], "multiple case variations"),
        
        // Very long AND chains
        ("a AND b AND c AND d AND e AND f AND g AND h AND i AND j", 
         vec!["alphabet_sequence.rs"], "very long AND chain"),
        
        // AND with Unicode
        ("función AND prueba", vec!["spanish_code.py"], "Spanish function and test"),
        ("функция AND тест", vec!["russian_code.py"], "Russian function and test"),
        ("π AND ∑", vec!["math_symbols.py"], "mathematical symbols"),
        
        // AND that should return no results
        ("nonexistent_term_12345 AND another_nonexistent_67890", vec![], "non-existent terms"),
        ("function AND impossible_combination_xyz", vec![], "one valid, one invalid term"),
        
        // AND with very common words
        ("the AND and", vec!["common_words.txt"], "very common English words"),
        ("a AND an", vec!["articles.txt"], "English articles"),
        ("is AND are", vec!["verb_forms.txt"], "verb forms"),
    ];
    
    for (query, expected_files, description) in edge_case_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: if expected_files.is_empty() { 1.0 } else { 0.80 },
            recall_threshold: if expected_files.is_empty() { 1.0 } else { 0.75 },
        };
        
        let result = validator.validate(&case).await?;
        
        if expected_files.is_empty() {
            // Should return no results
            assert!(result.actual_files.is_empty(), 
                   "Expected no results for edge case {}: {}", description, query);
        } else {
            assert!(result.is_correct, "Failed edge case AND test {}: {} - {}", description, query, result.summary());
        }
    }
    
    println!("AND edge cases test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_and_performance_with_large_result_sets() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate large dataset for performance testing
    let large_dataset = test_generator.generate_large_and_performance_dataset(1000).await?;
    let validator = CorrectnessValidator::new(&large_dataset.index_path, &large_dataset.vector_path).await?;
    
    let performance_test_cases = vec![
        // AND queries that should match many files
        ("function AND test", vec![], "common terms in many files"),
        ("struct AND impl", vec![], "structural programming terms"),
        ("error AND handling", vec![], "error handling patterns"),
        ("async AND await", vec![], "async programming patterns"),
        ("pub AND fn", vec![], "public function declarations"),
        
        // AND queries with decreasing selectivity
        ("very_common_term AND less_common_term", vec![], "mixed selectivity"),
        ("rare_term_12345 AND function", vec![], "rare and common terms"),
        ("specific_implementation_detail AND general_concept", vec![], "specific and general"),
    ];
    
    for (query, _, description) in performance_test_cases {
        let start_time = std::time::Instant::now();
        
        let search_result = validator.search_engine.search(&query).await?;
        
        let search_duration = start_time.elapsed();
        
        // Performance assertions
        assert!(search_duration < std::time::Duration::from_secs(5), 
               "AND search too slow for {}: {:?}", description, search_duration);
        
        // Verify result quality for non-empty results
        if !search_result.is_empty() {
            let terms: Vec<&str> = query.split(" AND ").collect();
            for result in &search_result {
                for term in &terms {
                    assert!(result.content.to_lowercase().contains(&term.to_lowercase()),
                           "Performance test: AND term '{}' missing in result", term);
                }
            }
        }
        
        println!("Performance test {}: {} results in {:?}", description, search_result.len(), search_duration);
    }
    
    println!("AND performance with large result sets test completed successfully");
    Ok(())
}

fn extract_and_terms_from_grouped_query(query: &str) -> Vec<String> {
    // Simple extraction of AND terms from grouped queries
    // In a real implementation, this would use proper parsing
    let mut terms = Vec::new();
    let cleaned = query.replace("(", "").replace(")", "");
    for part in cleaned.split(" AND ") {
        if !part.trim().is_empty() {
            terms.push(part.trim().to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}
```

## Success Criteria
- Simple two-term AND queries work correctly with 90%+ precision
- Multi-term AND queries (3+ terms) maintain 85%+ precision
- Parentheses grouping is properly handled and precedence respected
- All AND terms are verified to be present in result documents
- Edge cases with special characters, numbers, and Unicode work
- Performance remains acceptable for large result sets (< 5 seconds)
- Case-insensitive AND operations work correctly
- Empty result sets are handled properly for impossible combinations
- Very long AND chains are processed without errors
- Complex nested grouping works correctly

## Time Limit
10 minutes maximum