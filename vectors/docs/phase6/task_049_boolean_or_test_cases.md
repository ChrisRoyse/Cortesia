# Task 049: Boolean OR Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates boolean OR test cases that validates the system correctly handles OR logic with proper precedence, grouping, and union operations.

## Project Structure
tests/
  boolean_or_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for boolean OR operations including simple OR queries, precedence handling, grouping with parentheses, and complex multi-term OR combinations that return the union of matching documents.

## Requirements
1. Create comprehensive integration test
2. Test all boolean OR scenarios
3. Validate search accuracy with OR logic (union of results)
4. Handle precedence and grouping correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::collections::HashSet;

#[tokio::test]
async fn test_simple_or_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate test files with different content for OR operations
    let or_test_files = test_generator.generate_boolean_or_test_files().await?;
    let validator = CorrectnessValidator::new(&or_test_files.index_path, &or_test_files.vector_path).await?;
    
    let simple_or_test_cases = vec![
        // Basic two-term OR
        ("function OR main", vec!["functions.rs", "main.rs", "lib.rs", "app.rs"], "function or main"),
        ("struct OR class", vec!["structs.rs", "classes.py", "objects.java"], "struct or class"),
        ("error OR exception", vec!["errors.rs", "exceptions.py", "error_handling.java"], "error or exception"),
        ("test OR spec", vec!["tests.rs", "specs.py", "test_files.js"], "test or spec"),
        ("async OR sync", vec!["async_code.rs", "sync_code.rs", "concurrency.py"], "async or sync"),
        
        // Case insensitive OR
        ("Function OR Main", vec!["functions.rs", "main.rs", "lib.rs"], "case insensitive function or main"),
        ("STRUCT OR CLASS", vec!["structs.rs", "classes.py", "objects.java"], "uppercase struct or class"),
        ("Error or Exception", vec!["errors.rs", "exceptions.py", "error_handling.java"], "mixed case error or exception"),
        
        // Programming language specific OR
        ("pub OR private", vec!["public.rs", "private.rs", "visibility.java"], "visibility modifiers"),
        ("mut OR const", vec!["mutable.rs", "constants.rs", "immutable.py"], "mutability keywords"),
        ("impl OR trait", vec!["implementations.rs", "traits.rs", "interfaces.java"], "implementation keywords"),
        ("use OR import", vec!["use_statements.rs", "imports.py", "includes.cpp"], "import statements"),
        ("let OR var", vec!["variables.rs", "var_declarations.js", "local_vars.py"], "variable declarations"),
        
        // Content-specific OR
        ("TODO OR FIXME", vec!["todos.rs", "fixmes.py", "code_comments.java", "issues.js"], "code annotations"),
        ("GET OR POST", vec!["http_get.rs", "http_post.py", "web_methods.js"], "HTTP methods"),
        ("read OR write", vec!["file_read.rs", "file_write.py", "io_operations.java"], "IO operations"),
        ("push OR pull", vec!["git_push.sh", "git_pull.sh", "version_control.md"], "git operations"),
        
        // Technical terms OR
        ("database OR storage", vec!["database.rs", "storage.py", "persistence.java"], "data persistence"),
        ("client OR server", vec!["client.rs", "server.py", "networking.java"], "network architecture"),
        ("encode OR decode", vec!["encoding.rs", "decoding.py", "serialization.java"], "data transformation"),
        ("compile OR interpret", vec!["compiler.rs", "interpreter.py", "language_processing.java"], "code execution"),
        ("cache OR memory", vec!["cache.rs", "memory.py", "storage_systems.java"], "storage systems"),
    ];
    
    for (query, expected_files, description) in simple_or_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed simple OR test {}: {} - {}", description, query, result.summary());
        assert!(result.precision >= 0.85, "Low precision for {}: {:.2}", description, result.precision);
        
        // Verify that results contain at least one of the OR terms
        let terms: Vec<&str> = query.split(" OR ").collect();
        for file_result in &result.actual_files {
            let has_any_term = terms.iter().any(|term| 
                file_result.content.to_lowercase().contains(&term.to_lowercase()));
            assert!(has_any_term, "No OR terms found in {} for query {}", file_result.filename, query);
        }
        
        // Verify union behavior - should get more results than individual terms
        let first_term_results = validator.search_engine.search(terms[0]).await?;
        let second_term_results = validator.search_engine.search(terms[1]).await?;
        let or_results = validator.search_engine.search(&query).await?;
        
        assert!(or_results.len() >= first_term_results.len(), 
               "OR results should be >= first term results for {}", query);
        assert!(or_results.len() >= second_term_results.len(), 
               "OR results should be >= second term results for {}", query);
    }
    
    println!("Simple OR operations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_multi_term_or_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let multi_or_test_files = test_generator.generate_multi_term_or_test_files().await?;
    let validator = CorrectnessValidator::new(&multi_or_test_files.index_path, &multi_or_test_files.vector_path).await?;
    
    let multi_term_or_test_cases = vec![
        // Three-term OR
        ("function OR method OR procedure", 
         vec!["functions.rs", "methods.py", "procedures.sql", "callable.java"], "callable entities"),
        ("struct OR class OR object", 
         vec!["structs.rs", "classes.py", "objects.js", "data_types.java"], "data structures"),
        ("error OR exception OR failure", 
         vec!["errors.rs", "exceptions.py", "failures.java", "error_handling.cpp"], "error types"),
        ("test OR spec OR check", 
         vec!["tests.rs", "specs.py", "checks.java", "validation.js"], "validation methods"),
        ("read OR write OR access", 
         vec!["read_ops.rs", "write_ops.py", "access_control.java", "io_operations.cpp"], "data access"),
        
        // Four-term OR
        ("int OR integer OR number OR numeric", 
         vec!["integers.rs", "numbers.py", "numeric.java", "math_types.cpp"], "numeric types"),
        ("GET OR POST OR PUT OR DELETE", 
         vec!["http_get.rs", "http_post.py", "http_put.java", "http_delete.js"], "HTTP methods"),
        ("add OR insert OR create OR new", 
         vec!["add_ops.rs", "insert_ops.py", "create_ops.java", "new_ops.js"], "creation operations"),
        ("file OR document OR record OR entry", 
         vec!["files.rs", "documents.py", "records.java", "entries.js"], "data entities"),
        
        // Five-term OR
        ("red OR green OR blue OR yellow OR purple", 
         vec!["colors.rs", "rgb_values.py", "color_constants.java"], "color values"),
        ("monday OR tuesday OR wednesday OR thursday OR friday", 
         vec!["weekdays.rs", "calendar.py", "date_utils.java"], "weekdays"),
        
        // Technical multi-term OR
        ("json OR xml OR yaml OR toml OR csv", 
         vec!["json_parser.rs", "xml_parser.py", "yaml_config.java", "toml_settings.rs", "csv_reader.py"], "data formats"),
        ("mysql OR postgresql OR sqlite OR mongodb OR redis", 
         vec!["mysql_conn.rs", "postgres.py", "sqlite_db.java", "mongo_client.js", "redis_cache.py"], "databases"),
        ("rust OR python OR java OR javascript OR cpp", 
         vec!["rust_code.rs", "python_script.py", "java_class.java", "js_module.js", "cpp_source.cpp"], "programming languages"),
        
        // Domain-specific multi-term OR
        ("login OR signin OR authenticate OR authorize OR verify", 
         vec!["auth_login.rs", "signin.py", "authentication.java", "authorization.js", "verification.cpp"], "authentication"),
        ("encrypt OR decrypt OR hash OR sign OR verify", 
         vec!["encryption.rs", "decryption.py", "hashing.java", "signatures.js", "crypto_verify.cpp"], "cryptography"),
        ("compress OR decompress OR zip OR unzip OR archive", 
         vec!["compression.rs", "decompression.py", "zip_utils.java", "archive_tools.js"], "compression"),
        
        // Mixed case multi-term OR
        ("Function OR Method OR PROCEDURE", 
         vec!["functions.rs", "methods.py", "procedures.sql"], "mixed case callables"),
        ("ERROR OR exception OR Failure", 
         vec!["errors.rs", "exceptions.py", "failures.java"], "mixed case errors"),
    ];
    
    for (query, expected_files, description) in multi_term_or_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed multi-term OR test {}: {} - {}", description, query, result.summary());
        
        // Verify that results contain at least one of the OR terms
        let terms: Vec<&str> = query.split(" OR ").collect();
        for file_result in &result.actual_files {
            let has_any_term = terms.iter().any(|term| 
                file_result.content.to_lowercase().contains(&term.to_lowercase()));
            assert!(has_any_term, "No OR terms found in {} for query {}", file_result.filename, query);
        }
        
        // Verify union behavior - OR should return union of all individual term results
        let mut all_individual_files = HashSet::new();
        for term in &terms {
            let term_results = validator.search_engine.search(term).await?;
            for result in term_results {
                all_individual_files.insert(result.filename);
            }
        }
        
        let or_results = validator.search_engine.search(&query).await?;
        let or_files: HashSet<String> = or_results.iter().map(|r| r.filename.clone()).collect();
        
        // OR results should include all files that match any individual term
        for individual_file in &all_individual_files {
            assert!(or_files.contains(individual_file), 
                   "OR query missing file {} that matches individual terms", individual_file);
        }
    }
    
    println!("Multi-term OR operations test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_or_with_parentheses_grouping() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let grouped_or_test_files = test_generator.generate_grouped_or_test_files().await?;
    let validator = CorrectnessValidator::new(&grouped_or_test_files.index_path, &grouped_or_test_files.vector_path).await?;
    
    let grouped_or_test_cases = vec![
        // Simple grouping
        ("(function OR method)", vec!["functions.rs", "methods.py", "callables.java"], "grouped function or method"),
        ("(struct OR class)", vec!["structs.rs", "classes.py", "objects.java"], "grouped struct or class"),
        ("(error OR exception)", vec!["errors.rs", "exceptions.py", "error_handling.java"], "grouped error or exception"),
        
        // Nested OR groups
        ("((function OR method) OR (procedure OR routine))", 
         vec!["functions.rs", "methods.py", "procedures.sql", "routines.cpp"], "nested OR groups"),
        ("((read OR write) OR (input OR output))", 
         vec!["read_ops.rs", "write_ops.py", "input_handling.java", "output_ops.js"], "nested IO operations"),
        
        // Mixed grouping with multiple terms
        ("(function OR method OR procedure)", vec!["functions.rs", "methods.py", "procedures.sql"], "grouped three terms"),
        ("(GET OR POST OR PUT)", vec!["http_get.rs", "http_post.py", "http_put.java"], "grouped HTTP methods"),
        ("(int OR float OR double)", vec!["integers.rs", "floats.py", "doubles.java"], "grouped numeric types"),
        
        // Complex nested grouping
        ("(function OR (method OR procedure))", 
         vec!["functions.rs", "methods.py", "procedures.sql"], "function or nested method/procedure"),
        ("(struct OR (class OR (object OR type)))", 
         vec!["structs.rs", "classes.py", "objects.js", "types.java"], "deeply nested data structures"),
        ("((database OR storage) OR (cache OR memory))", 
         vec!["database.rs", "storage.py", "cache.java", "memory.cpp"], "database storage or cache memory"),
        
        // Grouping with technical terms
        ("(json OR (xml OR yaml))", vec!["json_files.rs", "xml_parser.py", "yaml_config.java"], "JSON or nested XML/YAML"),
        ("(encrypt OR (decrypt OR hash))", vec!["encryption.rs", "decryption.py", "hashing.java"], "crypto operations grouping"),
        ("(client OR (server OR proxy))", vec!["client.rs", "server.py", "proxy.java"], "network components grouping"),
        
        // Multiple separate OR groups
        ("(function OR method) OR (struct OR class)", 
         vec!["functions.rs", "methods.py", "structs.rs", "classes.py"], "separate callable and data structure groups"),
        ("(read OR write) OR (input OR output)", 
         vec!["read_ops.rs", "write_ops.py", "input.java", "output.js"], "separate read/write and input/output groups"),
        ("(GET OR POST) OR (PUT OR DELETE)", 
         vec!["http_get.rs", "http_post.py", "http_put.java", "http_delete.js"], "separate HTTP method groups"),
        
        // Deeply nested complex grouping
        ("((function OR method) OR ((validation OR test) OR check))", 
         vec!["functions.rs", "methods.py", "validation.java", "tests.py", "checks.js"], "deeply nested function or validation"),
        ("(((json OR xml) OR yaml) OR ((csv OR tsv) OR data))", 
         vec!["json_parser.rs", "xml_handler.py", "yaml_config.java", "csv_reader.js", "data_files.cpp"], "deeply nested data formats"),
    ];
    
    for (query, expected_files, description) in grouped_or_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed grouped OR test {}: {} - {}", description, query, result.summary());
        
        // Extract all terms from grouped query and verify at least one is present
        let terms = extract_or_terms_from_grouped_query(&query);
        for file_result in &result.actual_files {
            let has_any_term = terms.iter().any(|term| 
                file_result.content.to_lowercase().contains(&term.to_lowercase()));
            assert!(has_any_term, "No grouped OR terms found in {} for query {}", file_result.filename, query);
        }
    }
    
    println!("OR with parentheses grouping test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_or_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let edge_case_test_files = test_generator.generate_or_edge_case_test_files().await?;
    let validator = CorrectnessValidator::new(&edge_case_test_files.index_path, &edge_case_test_files.vector_path).await?;
    
    let edge_case_test_cases = vec![
        // OR with special characters
        ("function() OR main()", vec!["function_calls.rs", "main_calls.py"], "function calls with parentheses"),
        ("struct{} OR class{}", vec!["struct_blocks.rs", "class_blocks.py"], "struct or class blocks"),
        ("\"string\" OR 'char'", vec!["string_literals.rs", "char_literals.cpp"], "string or char literals"),
        ("array[0] OR list[0]", vec!["array_access.rs", "list_access.py"], "array or list access"),
        
        // OR with numbers
        ("version OR 2.0", vec!["version_info.rs", "v2_features.py", "version_2.txt"], "version or number"),
        ("test OR 123", vec!["test_files.rs", "numbered_items.py", "item_123.txt"], "test or number"),
        ("port OR 8080", vec!["port_config.rs", "port_8080.py", "server_config.txt"], "port or number"),
        
        // OR with underscores and hyphens
        ("snake_case OR camelCase", vec!["naming_conventions.rs", "style_guide.py"], "different naming cases"),
        ("kebab-case OR snake_case", vec!["style_guides.md", "naming_rules.txt"], "hyphenated or underscored"),
        
        // OR with file extensions
        (".rs OR .py", vec!["file_types.md", "language_extensions.txt"], "file extensions"),
        ("Cargo.toml OR package.json", vec!["config_files.md", "build_configs.txt"], "config files"),
        ("README.md OR LICENSE", vec!["project_docs.md", "documentation.txt"], "documentation files"),
        
        // Empty and whitespace edge cases
        ("  function  OR  test  ", vec!["functions.rs", "tests.py"], "query with extra whitespace"),
        ("function\tOR\ttest", vec!["functions.rs", "tests.py"], "query with tabs"),
        ("function\nOR\ntest", vec!["functions.rs", "tests.py"], "query with newlines"),
        
        // Case sensitivity edge cases
        ("or OR OR", vec!["boolean_logic.rs", "logical_operators.py"], "or keyword with OR operator"),
        ("Function OR function", vec!["functions.rs", "function_defs.py"], "same word different cases"),
        ("TEST OR test OR Test", vec!["test_files.rs", "testing.py"], "multiple case variations"),
        
        // Very long OR chains
        ("a OR b OR c OR d OR e OR f OR g OR h OR i OR j OR k OR l OR m OR n OR o OR p", 
         vec!["alphabet_items.rs", "letter_sequences.py"], "very long OR chain"),
        
        // OR with Unicode
        ("función OR prueba", vec!["spanish_code.py", "multilingual.rs"], "Spanish function or test"),
        ("函数 OR 测试", vec!["chinese_code.py", "multilingual.rs"], "Chinese function or test"),
        ("π OR ∑", vec!["math_symbols.py", "greek_letters.rs"], "mathematical symbols"),
        
        // OR with one non-existent term
        ("function OR nonexistent_term_12345", vec!["functions.rs", "function_defs.py"], "one valid, one invalid term"),
        ("existing_code OR impossible_combination_xyz", vec!["existing_code.rs"], "existing or non-existent"),
        
        // OR that should return many results
        ("a OR the", vec!["common_words.txt", "articles.txt", "text_files.md"], "very common English words"),
        ("function OR class", vec!["functions.rs", "classes.py", "code_files.java"], "common programming terms"),
        
        // OR with very specific terms
        ("specific_function_name_xyz123 OR another_specific_term_abc789", 
         vec!["specific_functions.rs"], "very specific terms"),
        
        // Single term OR (should behave like regular search)
        ("function OR", vec!["functions.rs", "function_defs.py"], "trailing OR"),
        ("OR function", vec!["functions.rs", "function_defs.py"], "leading OR"),
    ];
    
    for (query, expected_files, description) in edge_case_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::BooleanSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.70, // Lower precision for edge cases
            recall_threshold: 0.65,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed edge case OR test {}: {} - {}", description, query, result.summary());
        
        // For non-empty expected results, verify OR behavior
        if !expected_files.is_empty() {
            let valid_terms: Vec<&str> = query.split(" OR ")
                .filter(|term| !term.trim().is_empty() && !term.contains("nonexistent") && !term.contains("impossible"))
                .collect();
            
            if !valid_terms.is_empty() {
                for file_result in &result.actual_files {
                    let has_any_term = valid_terms.iter().any(|term| 
                        file_result.content.to_lowercase().contains(&term.trim().to_lowercase()));
                    assert!(has_any_term, "No valid OR terms found in {} for query {}", file_result.filename, query);
                }
            }
        }
    }
    
    println!("OR edge cases test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_or_performance_with_large_result_sets() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate large dataset for performance testing
    let large_dataset = test_generator.generate_large_or_performance_dataset(1000).await?;
    let validator = CorrectnessValidator::new(&large_dataset.index_path, &large_dataset.vector_path).await?;
    
    let performance_test_cases = vec![
        // OR queries that should match many files
        ("function OR method", vec![], "common programming terms"),
        ("error OR exception", vec![], "error handling terms"),
        ("read OR write", vec![], "IO operation terms"),
        ("public OR private", vec![], "visibility modifiers"),
        ("int OR string", vec![], "common data types"),
        
        // OR queries with very broad matches
        ("a OR the OR and OR or OR in OR on OR at OR to", vec![], "very common words"),
        ("test OR spec OR check OR validate OR verify", vec![], "validation terms"),
        
        // OR queries mixing common and rare terms
        ("function OR rare_specific_term_xyz123", vec![], "common and rare terms"),
        ("very_common_term OR moderately_common OR rare_term", vec![], "mixed frequency terms"),
    ];
    
    for (query, _, description) in performance_test_cases {
        let start_time = std::time::Instant::now();
        
        let search_result = validator.search_engine.search(&query).await?;
        
        let search_duration = start_time.elapsed();
        
        // Performance assertions
        assert!(search_duration < std::time::Duration::from_secs(10), 
               "OR search too slow for {}: {:?}", description, search_duration);
        
        // Verify result quality for non-empty results
        if !search_result.is_empty() {
            let terms: Vec<&str> = query.split(" OR ").collect();
            for result in &search_result {
                let has_any_term = terms.iter().any(|term| 
                    result.content.to_lowercase().contains(&term.trim().to_lowercase()));
                assert!(has_any_term, "Performance test: No OR terms found in result for query {}", query);
            }
            
            // Verify union behavior - should be >= individual term results
            let individual_result_counts: Vec<usize> = {
                let mut counts = Vec::new();
                for term in &terms {
                    if !term.trim().is_empty() {
                        let term_results = validator.search_engine.search(term.trim()).await?;
                        counts.push(term_results.len());
                    }
                }
                counts
            };
            
            if let Some(max_individual) = individual_result_counts.iter().max() {
                assert!(search_result.len() >= *max_individual,
                       "OR result count {} should be >= max individual term count {} for query {}", 
                       search_result.len(), max_individual, query);
            }
        }
        
        println!("Performance test {}: {} results in {:?}", description, search_result.len(), search_duration);
    }
    
    println!("OR performance with large result sets test completed successfully");
    Ok(())
}

fn extract_or_terms_from_grouped_query(query: &str) -> Vec<String> {
    // Simple extraction of OR terms from grouped queries
    // In a real implementation, this would use proper parsing
    let mut terms = Vec::new();
    let cleaned = query.replace("(", "").replace(")", "");
    for part in cleaned.split(" OR ") {
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
- Simple two-term OR queries work correctly with 85%+ precision
- Multi-term OR queries (3+ terms) maintain 80%+ precision
- Parentheses grouping is properly handled and precedence respected
- Results contain at least one of the OR terms in each document
- Union behavior verified - OR returns union of individual term results
- Edge cases with special characters, numbers, and Unicode work
- Performance remains acceptable for large result sets (< 10 seconds)
- Case-insensitive OR operations work correctly
- Very long OR chains are processed without errors
- Complex nested grouping works correctly
- OR results are always >= the largest individual term result set

## Time Limit
10 minutes maximum