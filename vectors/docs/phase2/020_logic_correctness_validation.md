# Task 020: Logic Correctness Validation

## Prerequisites
- Task 001-019 completed: All component interaction tests passing
- Full system integration verified
- All error scenarios handled properly

## Required Imports
```rust
// Add to logic validation test file
use anyhow::Result;
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
use std::collections::HashSet;
```

## Context
You have comprehensive integration testing from Tasks 017-019. Now you need focused validation specifically for boolean logic correctness to ensure 100% accuracy. This task creates targeted tests that verify AND/OR/NOT logic with zero false positives/negatives.

Critical requirements for 100% accuracy:
- AND logic: ALL terms must be present
- OR logic: ANY term must be present
- NOT logic: Excluded term must NOT be present
- Zero false positives/negatives

## Your Task (10 minutes max)
Implement focused logic correctness validation tests that verify 100% boolean logic accuracy with comprehensive test cases.

## Success Criteria
1. Create focused AND logic correctness tests
2. Implement comprehensive OR logic validation
3. Test NOT logic exclusion accuracy
4. Verify nested boolean expression correctness
5. Achieve 100% accuracy validation with zero false positives/negatives

## Implementation Steps

### 1. RED: Write failing boolean logic correctness tests
```rust
// Create tests/logic_correctness_tests.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;

use llmkg::boolean::{BooleanSearchEngine, SearchResult};
use llmkg::validator::DocumentLevelValidator;

#[test]
fn test_and_logic_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_logic_correctness_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing AND logic correctness...");
    
    // AND logic test cases with known expected results
    let and_test_cases = vec![
        ("pub AND struct", vec!["pub", "struct"], vec!["logic1.rs", "logic3.rs"]),
        ("fn AND return", vec!["fn", "return"], vec!["logic2.rs"]),
        ("impl AND Display", vec!["impl", "Display"], vec!["logic4.rs"]),
        ("pub AND fn AND return", vec!["pub", "fn", "return"], vec!["logic5.rs"]),
    ];
    
    for (query, required_terms, expected_files) in and_test_cases {
        println!("  Testing AND query: {}", query);
        
        let results = boolean_engine.search_boolean(query)?;
        
        // Validate with validator
        let validation = validator.validate_boolean_results(query, &results)?;
        assert!(validation, "Validator should confirm AND logic correctness for: {}", query);
        
        // Manual verification: every result must contain ALL required terms
        for result in &results {
            let content_lower = result.content.to_lowercase();
            for term in &required_terms {
                assert!(content_lower.contains(&term.to_lowercase()), 
                       "Result '{}' missing required AND term '{}' in query '{}'", 
                       result.file_path, term, query);
            }
        }
        
        // Verify expected files are found
        let found_files: std::collections::HashSet<_> = results.iter()
            .map(|r| r.file_path.as_str())
            .collect();
        
        for expected_file in &expected_files {
            assert!(found_files.contains(expected_file), 
                   "Expected file '{}' not found for AND query '{}'", expected_file, query);
        }
        
        println!("    ✓ AND logic correct: {} results, all terms verified", results.len());
    }
    
    Ok(())
}

fn create_logic_correctness_index(index_path: &Path) -> Result<()> {
    use tantivy::schema::{Schema, TEXT, STORED};
    use tantivy::{Index, doc};
    
    let mut schema_builder = Schema::builder();
    let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
    let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
    let schema = schema_builder.build();
    
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    // Carefully crafted documents for logic correctness testing
    let logic_docs = vec![
        ("logic1.rs", "pub struct LogicData { value: i32 }"), // HAS: pub, struct
        ("logic2.rs", "fn helper() -> return bool { true }"), // HAS: fn, return
        ("logic3.rs", "pub struct AdvancedStruct { data: String }"), // HAS: pub, struct
        ("logic4.rs", "impl Display for LogicType { fn fmt() {} }"), // HAS: impl, Display
        ("logic5.rs", "pub fn process() -> return Result<()> { Ok(()) }"), // HAS: pub, fn, return
        ("logic6.rs", "struct InternalStruct { private: bool }"), // HAS: struct (NOT pub)
        ("logic7.rs", "fn private_function() { println!(\"test\"); }"), // HAS: fn (NOT return)
        ("logic8.rs", "impl Clone for CloneType { }"), // HAS: impl (NOT Display)
        ("logic9.rs", "pub enum StatusEnum { Active, Inactive }"), // HAS: pub (NOT struct)
        ("logic10.rs", "trait MyTrait { fn method(&self); }"), // HAS: fn (NOT pub, NOT return)
    ];
    
    for (filename, content) in logic_docs {
        index_writer.add_document(doc!(
            file_path_field => filename,
            content_field => content,
            raw_content_field => content,
            chunk_index_field => 0u64
        ))?;
    }
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Add OR and NOT logic correctness tests
```rust
#[test]
fn test_or_logic_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_logic_correctness_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing OR logic correctness...");
    
    // OR logic test cases with known expected results
    let or_test_cases = vec![
        ("struct OR enum", vec!["struct", "enum"], 
         vec!["logic1.rs", "logic3.rs", "logic6.rs", "logic9.rs"]),
        ("impl OR trait", vec!["impl", "trait"], 
         vec!["logic4.rs", "logic8.rs", "logic10.rs"]),
        ("fn OR struct OR enum", vec!["fn", "struct", "enum"], 
         vec!["logic1.rs", "logic2.rs", "logic3.rs", "logic5.rs", "logic6.rs", "logic7.rs", "logic9.rs", "logic10.rs"]),
    ];
    
    for (query, possible_terms, expected_files) in or_test_cases {
        println!("  Testing OR query: {}", query);
        
        let results = boolean_engine.search_boolean(query)?;
        
        // Validate with validator
        let validation = validator.validate_boolean_results(query, &results)?;
        assert!(validation, "Validator should confirm OR logic correctness for: {}", query);
        
        // Manual verification: every result must contain AT LEAST ONE required term
        for result in &results {
            let content_lower = result.content.to_lowercase();
            let has_any_term = possible_terms.iter()
                .any(|term| content_lower.contains(&term.to_lowercase()));
            assert!(has_any_term, 
                   "Result '{}' missing all OR terms {:?} in query '{}'", 
                   result.file_path, possible_terms, query);
        }
        
        // Verify no false negatives: files with any term should be found
        let found_files: std::collections::HashSet<_> = results.iter()
            .map(|r| r.file_path.as_str())
            .collect();
        
        for expected_file in &expected_files {
            assert!(found_files.contains(expected_file), 
                   "Expected file '{}' not found for OR query '{}'", expected_file, query);
        }
        
        println!("    ✓ OR logic correct: {} results, at least one term verified", results.len());
    }
    
    Ok(())
}

#[test]
fn test_not_logic_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_logic_correctness_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing NOT logic correctness...");
    
    // NOT logic test cases with expected exclusions
    let not_test_cases = vec![
        ("pub NOT enum", "pub", "enum", 
         vec!["logic1.rs", "logic3.rs", "logic5.rs"], // Should have pub, NOT enum
         vec!["logic9.rs"]), // Should be excluded (has enum)
        ("struct NOT pub", "struct", "pub",
         vec!["logic6.rs"], // Should have struct, NOT pub  
         vec!["logic1.rs", "logic3.rs"]), // Should be excluded (have pub)
        ("fn NOT return", "fn", "return",
         vec!["logic7.rs", "logic10.rs"], // Should have fn, NOT return
         vec!["logic2.rs", "logic5.rs"]), // Should be excluded (have return)
    ];
    
    for (query, include_term, exclude_term, expected_included, expected_excluded) in not_test_cases {
        println!("  Testing NOT query: {}", query);
        
        let results = boolean_engine.search_boolean(query)?;
        
        // Validate with validator
        let validation = validator.validate_boolean_results(query, &results)?;
        assert!(validation, "Validator should confirm NOT logic correctness for: {}", query);
        
        // Manual verification: every result must contain include_term AND NOT contain exclude_term
        for result in &results {
            let content_lower = result.content.to_lowercase();
            
            assert!(content_lower.contains(include_term), 
                   "Result '{}' missing required include term '{}' in NOT query '{}'", 
                   result.file_path, include_term, query);
            
            assert!(!content_lower.contains(exclude_term), 
                   "Result '{}' contains excluded term '{}' in NOT query '{}'", 
                   result.file_path, exclude_term, query);
        }
        
        // Verify expected inclusions
        let found_files: std::collections::HashSet<_> = results.iter()
            .map(|r| r.file_path.as_str())
            .collect();
        
        for expected_file in &expected_included {
            assert!(found_files.contains(expected_file), 
                   "Expected included file '{}' not found for NOT query '{}'", expected_file, query);
        }
        
        // Verify expected exclusions are NOT in results
        for excluded_file in &expected_excluded {
            assert!(!found_files.contains(excluded_file), 
                   "Excluded file '{}' incorrectly found for NOT query '{}'", excluded_file, query);
        }
        
        println!("    ✓ NOT logic correct: {} results, exclusions verified", results.len());
    }
    
    Ok(())
}
```

### 3. REFACTOR: Add nested boolean expression correctness
```rust
#[test]
fn test_nested_boolean_correctness() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_logic_correctness_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing nested boolean expression correctness...");
    
    // Nested boolean expression test cases
    let nested_test_cases = vec![
        ("(pub AND struct) OR (impl AND Display)", 
         vec!["logic1.rs", "logic3.rs", "logic4.rs"]), // pub+struct OR impl+Display
        ("(fn AND return) OR (pub AND enum)",
         vec!["logic2.rs", "logic5.rs", "logic9.rs"]), // fn+return OR pub+enum
        ("pub AND (struct OR enum)",
         vec!["logic1.rs", "logic3.rs", "logic9.rs"]), // pub AND (struct OR enum)
        ("(pub OR impl) NOT enum",
         vec!["logic1.rs", "logic3.rs", "logic4.rs", "logic5.rs", "logic8.rs"]), // (pub OR impl) NOT enum
    ];
    
    for (query, expected_files) in nested_test_cases {
        println!("  Testing nested query: {}", query);
        
        let results = boolean_engine.search_boolean(query)?;
        
        // Validate with validator  
        let validation = validator.validate_boolean_results(query, &results)?;
        assert!(validation, "Validator should confirm nested logic correctness for: {}", query);
        
        // For nested expressions, rely on validator for correctness since manual verification is complex
        // but verify expected files are found
        let found_files: std::collections::HashSet<_> = results.iter()
            .map(|r| r.file_path.as_str())
            .collect();
        
        for expected_file in &expected_files {
            assert!(found_files.contains(expected_file), 
                   "Expected file '{}' not found for nested query '{}'", expected_file, query);
        }
        
        println!("    ✓ Nested logic correct: {} results, expected files verified", results.len());
    }
    
    Ok(())
}

#[test]
fn test_zero_false_positives_negatives() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_logic_correctness_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing zero false positives/negatives...");
    
    // Test cases designed to catch false positives/negatives
    let accuracy_test_cases = vec![
        // Should find exactly these files, no more, no less
        ("pub AND struct", vec!["logic1.rs", "logic3.rs"]),
        ("impl AND Display", vec!["logic4.rs"]),
        ("fn AND return", vec!["logic2.rs", "logic5.rs"]),
        ("struct NOT pub", vec!["logic6.rs"]),
        ("nonexistent", vec![]), // Should find nothing
    ];
    
    for (query, expected_exact_files) in accuracy_test_cases {
        println!("  Testing accuracy for: {}", query);
        
        let results = boolean_engine.search_boolean(query)?;
        
        // Validate with validator
        let validation = validator.validate_boolean_results(query, &results)?;
        
        if expected_exact_files.is_empty() {
            assert!(!validation, "Empty results should not validate for query: {}", query);
            assert!(results.is_empty(), "Should find no results for query: {}", query);
        } else {
            assert!(validation, "Non-empty results should validate for query: {}", query);
            
            // Check for exact match - no false positives or negatives
            let found_files: std::collections::HashSet<_> = results.iter()
                .map(|r| r.file_path.as_str())
                .collect();
            
            let expected_files: std::collections::HashSet<_> = expected_exact_files.iter()
                .map(|s| *s)
                .collect();
            
            assert_eq!(found_files, expected_files, 
                      "Results mismatch for query '{}'. Found: {:?}, Expected: {:?}", 
                      query, found_files, expected_files);
        }
        
        println!("    ✓ Accuracy verified: {} results match exactly", results.len());
    }
    
    println!("✓ Zero false positives/negatives confirmed");
    
    Ok(())
}
```

## Validation Checklist
- [ ] AND logic correctness verified with all terms required
- [ ] OR logic correctness verified with any term matching
- [ ] NOT logic correctness verified with proper exclusion
- [ ] Nested boolean expressions work correctly
- [ ] Zero false positives/negatives achieved

## Context for Next Task
Next task (021) will focus on performance and accuracy validation, combining the logic correctness from this task with performance measurements to ensure all targets are met.