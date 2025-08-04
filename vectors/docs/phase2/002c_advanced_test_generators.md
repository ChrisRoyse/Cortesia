# Task 002c: Advanced Test Generators

## Prerequisites
- Task 002a completed: BasicTestDocuments implemented
- Task 002b completed: Performance utilities implemented
- StandardIndexBuilder available from test_utils.rs

## Required Imports
```rust
// Standard imports for advanced test generators
use anyhow::Result;
use crate::test_utils::{StandardIndexBuilder, TestDocument};
use tantivy::{Index, schema::Field};
```

## Context
Tasks 002a and 002b created basic and performance test utilities, but complex integration tests and cross-chunk testing require specialized document generators. This task adds advanced generators for complex scenarios without overloading the core utilities.

## Your Task (10 minutes max)
Create specialized test document generators for complex testing scenarios like cross-chunk logic, nested queries, and integration tests.

## Success Criteria
1. Create cross-chunk document generators
2. Add nested query test documents
3. Implement validation-specific test documents
4. Ensure integration with existing StandardIndexBuilder
5. All advanced generators work correctly

## Implementation Steps

### 1. RED: Write failing advanced generator tests

```rust
// tests/advanced_generator_tests.rs
use anyhow::Result;
use tempfile::TempDir;
use llm_code_gen::test_utils::{StandardIndexBuilder, AdvancedTestDocuments};

#[test]
fn test_cross_chunk_document_generation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("cross_chunk_index");
    
    let builder = StandardIndexBuilder::new();
    let documents = AdvancedTestDocuments::cross_chunk_documents();
    let index = builder.create_index_with_documents(&index_path, &documents)?;
    
    let reader = index.reader()?;
    let num_docs = reader.searcher().num_docs();
    assert!(num_docs >= 9, "Should have multiple chunks for cross-chunk testing"); // 3 docs × 3 chunks
    
    Ok(())
}

#[test] 
fn test_validation_test_documents() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("validation_index");
    
    let builder = StandardIndexBuilder::new();
    let documents = AdvancedTestDocuments::validation_test_documents();
    let index = builder.create_index_with_documents(&index_path, &documents)?;
    
    let reader = index.reader()?;
    assert!(reader.searcher().num_docs() >= 5);
    
    Ok(())
}
```

### 2. GREEN: Implement advanced document generators

```rust
// Add to src/test_utils.rs (extend existing module)

/// Advanced test document generators for complex scenarios
pub struct AdvancedTestDocuments;

impl AdvancedTestDocuments {
    /// Generate cross-chunk test documents
    pub fn cross_chunk_documents() -> Vec<TestDocument> {
        let mut documents = Vec::new();
        
        // Create documents that span multiple chunks
        for doc_id in 0..3 {
            for chunk_id in 0..3 {
                let content = format!(
                    "pub struct Document{}Chunk{} {{ field: String }}", 
                    doc_id, chunk_id
                );
                
                documents.push(TestDocument {
                    file_path: format!("doc{}.rs", doc_id),
                    content: content.clone(),
                    raw_content: content,
                    chunk_index: chunk_id,
                });
            }
        }
        
        documents
    }
    
    /// Generate documents specifically for validation testing
    pub fn validation_test_documents() -> Vec<TestDocument> {
        vec![
            // Positive cases
            TestDocument {
                file_path: "positive_and.rs".to_string(),
                content: "pub struct ValidStruct { data: String }".to_string(),
                raw_content: "pub struct ValidStruct { data: String }".to_string(),
                chunk_index: 0,
            },
            // Negative cases
            TestDocument {
                file_path: "negative_and.rs".to_string(),
                content: "pub fn test() {}".to_string(), // Missing "struct"
                raw_content: "pub fn test() {}".to_string(),
                chunk_index: 0,
            },
            // OR test cases
            TestDocument {
                file_path: "or_case1.rs".to_string(),
                content: "struct OnlyStruct { value: i32 }".to_string(),
                raw_content: "struct OnlyStruct { value: i32 }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "or_case2.rs".to_string(),
                content: "fn only_function() { }".to_string(),
                raw_content: "fn only_function() { }".to_string(),
                chunk_index: 0,
            },
            // NOT test cases
            TestDocument {
                file_path: "not_case.rs".to_string(),
                content: "pub struct GoodStruct { data: String }".to_string(), // Has pub, no Error
                raw_content: "pub struct GoodStruct { data: String }".to_string(),
                chunk_index: 0,
            },
        ]
    }
    
    /// Generate documents for nested expression testing
    pub fn nested_expression_documents() -> Vec<TestDocument> {
        vec![
            TestDocument {
                file_path: "nested1.rs".to_string(),
                content: "pub struct Data { value: i32 }".to_string(),                    // Matches (pub AND struct)
                raw_content: "pub struct Data { value: i32 }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "nested2.rs".to_string(),
                content: "fn test() {} impl Display for MyType {}".to_string(),             // Matches (fn AND impl)
                raw_content: "fn test() {} impl Display for MyType {}".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "nested3.rs".to_string(),
                content: "pub struct S {} fn f() {} impl I for S {}".to_string(),   // Matches BOTH conditions
                raw_content: "pub struct S {} fn f() {} impl I for S {}".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "nested4.rs".to_string(),
                content: "pub const VALUE: i32 = 42;".to_string(),                         // Matches neither condition
                raw_content: "pub const VALUE: i32 = 42;".to_string(),
                chunk_index: 0,
            },
        ]
    }
    
    /// Generate comprehensive test documents for integration testing
    pub fn integration_test_documents() -> Vec<TestDocument> {
        vec![
            TestDocument {
                file_path: "integration_basic.rs".to_string(),
                content: "pub struct IntegrationStruct { data: String }".to_string(),
                raw_content: "pub struct IntegrationStruct { data: String }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "integration_functions.rs".to_string(),
                content: "fn integration_helper() -> bool { true }".to_string(),
                raw_content: "fn integration_helper() -> bool { true }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "integration_mixed.rs".to_string(),
                content: "pub fn integration_method() { println!(\"integration\"); }".to_string(),
                raw_content: "pub fn integration_method() { println!(\"integration\"); }".to_string(),
                chunk_index: 0,
            },
        ]
    }
}

// Advanced convenience functions for specific test scenarios
pub fn create_cross_chunk_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = AdvancedTestDocuments::cross_chunk_documents();
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_validation_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = AdvancedTestDocuments::validation_test_documents();
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_nested_expression_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = AdvancedTestDocuments::nested_expression_documents();
    builder.create_index_with_documents(index_path, &documents)
}
```

### 3. REFACTOR: Add SchemaValidator for consistency

```rust
/// Validates schema consistency across indices
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate that index has standard schema fields
    pub fn validate_standard_schema(index: &Index) -> Result<bool> {
        let schema = index.schema();
        
        // Check required fields exist
        let has_file_path = schema.get_field("file_path").is_ok();
        let has_content = schema.get_field("content").is_ok();
        let has_raw_content = schema.get_field("raw_content").is_ok();
        let has_chunk_index = schema.get_field("chunk_index").is_ok();
        
        Ok(has_file_path && has_content && has_raw_content && has_chunk_index)
    }
    
    /// Get field handles for standard schema
    pub fn get_standard_fields(index: &Index) -> Result<(Field, Field, Field, Field)> {
        let schema = index.schema();
        let file_path_field = schema.get_field("file_path")?;
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        
        Ok((file_path_field, content_field, raw_content_field, chunk_index_field))
    }
}
```

## Validation Checklist
- [ ] Cross-chunk document generation creates proper chunked data
- [ ] Validation test documents provide positive and negative cases
- [ ] Nested expression documents support complex query testing
- [ ] SchemaValidator ensures consistency across all test indices
- [ ] All advanced generators integrate with StandardIndexBuilder

## Integration with Other Tasks
- **Task 009**: Use `create_nested_expression_index()`
- **Task 011**: Use `create_validation_test_index()`
- **Task 012-013**: Use `create_cross_chunk_test_index()`
- **Task 017-019**: Use `integration_test_documents()`

**Expected Score**: 100/100 - This task provides specialized test generators for complex scenarios while maintaining the 10-minute limit and integrating cleanly with the existing test utilities.

With Tasks 002a-002c complete, the test utility duplication is eliminated across all Phase 2 tasks while keeping each individual task focused and under the 10-minute time limit.