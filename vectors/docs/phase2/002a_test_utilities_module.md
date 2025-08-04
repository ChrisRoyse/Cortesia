# Task 002a: Create Basic Test Utilities Module

## Prerequisites
- Task 002 completed: Core data structures implemented and tested
- src/boolean.rs exists with BooleanSearchEngine
- Basic test infrastructure understanding established

## Required Imports
```rust
// Standard imports for basic test utilities module
use std::path::Path;
use tantivy::schema::{Schema, TEXT, STORED};
use tantivy::{Index, doc};
use anyhow::Result;
```

## Current Issue Analysis

**Problem**: Multiple tasks define duplicate basic test helper functions:
- `create_test_index` (Tasks 004, 010, 011)
- `create_and_logic_test_index` (Task 006)
- `create_test_index_with_content` (Task 005)
- Inconsistent schema definitions across tasks

**Impact**: Code duplication reduces maintainability and increases chance of inconsistencies across tests.

## Solution Approach

Create a focused basic test utilities module that provides:
1. **StandardIndexBuilder** - Consistent basic index creation
2. **BasicTestDocuments** - Standard test document patterns for boolean logic

## Your Task (10 minutes max)
Implement basic test utilities that eliminate code duplication for the most commonly used test functions.

## Success Criteria
1. Create StandardIndexBuilder with basic index creation
2. Add BasicTestDocuments with common test patterns
3. Implement backward-compatible convenience functions
4. Reduce code duplication in existing tasks
5. All basic test utilities work correctly

## Implementation Steps

### 1. RED: Write failing test for basic test utilities

```rust
// tests/basic_test_utilities_tests.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;

// Import the basic test utilities module we're about to create
use llm_code_gen::test_utils::{
    StandardIndexBuilder, 
    BasicTestDocuments
};

#[test]
fn test_standard_index_builder_basic() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    // Test basic index creation
    let builder = StandardIndexBuilder::new();
    let index = builder.create_basic_index(&index_path)?;
    
    // Verify index can be opened
    let reader = index.reader()?;
    assert_eq!(reader.searcher().num_docs(), 0);
    
    Ok(())
}

#[test]
fn test_basic_document_patterns() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    // Test AND logic document generation
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::and_logic_documents();
    let index = builder.create_index_with_documents(&index_path, &documents)?;
    
    let reader = index.reader()?;
    let num_docs = reader.searcher().num_docs();
    assert!(num_docs >= 3, "Should have at least 3 AND logic test documents");
    
    Ok(())
}
```

### 2. GREEN: Implement basic test utilities module

```rust
// src/test_utils.rs
use std::path::Path;
use std::time::{Duration, Instant};
use tantivy::schema::{Schema, Field, TEXT, STORED};
use tantivy::{Index, IndexWriter, doc};
use anyhow::Result;
use tempfile::TempDir;

/// Basic test document structure
#[derive(Debug, Clone)]
pub struct TestDocument {
    pub file_path: String,
    pub content: String,
    pub raw_content: String,
    pub chunk_index: u64,
}

/// Builds basic test indices with consistent schema
pub struct StandardIndexBuilder {
    schema: Schema,
    file_path_field: Field,
    content_field: Field,
    raw_content_field: Field,
    chunk_index_field: Field,
}

impl StandardIndexBuilder {
    /// Create new builder with standard schema
    pub fn new() -> Self {
        let mut schema_builder = Schema::builder();
        let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
        let schema = schema_builder.build();
        
        Self {
            schema,
            file_path_field,
            content_field,
            raw_content_field,
            chunk_index_field,
        }
    }
    
    /// Create basic empty index with standard schema
    pub fn create_basic_index(&self, index_path: &Path) -> Result<Index> {
        let index = Index::create_in_dir(index_path, self.schema.clone())?;
        Ok(index)
    }
    
    /// Create index with predefined test documents
    pub fn create_index_with_documents(&self, index_path: &Path, documents: &[TestDocument]) -> Result<Index> {
        let index = Index::create_in_dir(index_path, self.schema.clone())?;
        let mut index_writer = index.writer(50_000_000)?;
        
        for doc in documents {
            index_writer.add_document(doc!(
                self.file_path_field => doc.file_path.clone(),
                self.content_field => doc.content.clone(),
                self.raw_content_field => doc.raw_content.clone(),
                self.chunk_index_field => doc.chunk_index
            ))?;
        }
        
        index_writer.commit()?;
        Ok(index)
    }
}

/// Generates basic test documents for boolean logic testing
pub struct BasicTestDocuments;

impl BasicTestDocuments {
    /// Generate basic documents for AND logic testing
    pub fn and_logic_documents() -> Vec<TestDocument> {
        vec![
            TestDocument {
                file_path: "file1.rs".to_string(),
                content: "pub struct MyStruct { name: String }".to_string(), 
                raw_content: "pub struct MyStruct { name: String }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "file2.rs".to_string(), 
                content: "fn process() { println!(\"Hello\"); }".to_string(),
                raw_content: "fn process() { println!(\"Hello\"); }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "file3.rs".to_string(),
                content: "pub fn initialize() -> Result<(), Error> { Ok(()) }".to_string(),
                raw_content: "pub fn initialize() -> Result<(), Error> { Ok(()) }".to_string(),
                chunk_index: 0,
            },
        ]
    }
    
    /// Generate basic documents for OR logic testing
    pub fn or_logic_documents() -> Vec<TestDocument> {
        vec![
            TestDocument {
                file_path: "file1.rs".to_string(),
                content: "pub struct TestStruct { data: String }".to_string(),
                raw_content: "pub struct TestStruct { data: String }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "file2.rs".to_string(),
                content: "fn test_function() { println!(\"Hello\"); }".to_string(),
                raw_content: "fn test_function() { println!(\"Hello\"); }".to_string(),
                chunk_index: 0,
            },
        ]
    }
    
    /// Generate documents for NOT logic testing
    pub fn not_logic_documents() -> Vec<TestDocument> {
        vec![
            TestDocument {
                file_path: "file1.rs".to_string(),
                content: "pub struct PublicStruct { data: String }".to_string(),
                raw_content: "pub struct PublicStruct { data: String }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "file2.rs".to_string(),
                content: "struct PrivateStruct { value: i32 }".to_string(),
                raw_content: "struct PrivateStruct { value: i32 }".to_string(),
                chunk_index: 0,
            },
            TestDocument {
                file_path: "file3.rs".to_string(),
                content: "fn helper_function() { }".to_string(),
                raw_content: "fn helper_function() { }".to_string(),
                chunk_index: 0,
            },
        ]
    }
    
    /// Generate large document set for performance testing
    pub fn performance_documents(num_docs: usize) -> Vec<TestDocument> {
        (0..num_docs).map(|i| {
            let content = match i % 4 {
                0 => format!("pub struct TestStruct{} {{ data: String }}", i),
                1 => format!("fn performance_process{}() -> bool {{ true }}", i),
                2 => format!("struct InternalData{} {{ value: i32 }}", i),
                3 => format!("pub fn performance_helper{}() {{ println!(\"test\"); }}", i),
                _ => unreachable!(),
            };
            
            TestDocument {
                file_path: format!("file{}.rs", i),
                content: content.clone(),
                raw_content: content,
                chunk_index: (i / 10) as u64, // 10 chunks per document group
            }
        }).collect()
    }
    
    /// Generate cross-chunk test documents
    pub fn cross_chunk_documents() -> Vec<TestDocument> {
        let mut documents = Vec::new();
        
        // Create documents that span multiple chunks
        for doc_id in 0..5 {
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
}

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

/// Standardized performance measurement utilities
pub struct PerformanceTimer {
    measurements: Vec<(String, Duration)>,
}

impl PerformanceTimer {
    /// Create new performance timer
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
    
    /// Measure execution time of an operation
    pub fn measure_operation<F, R>(&mut self, operation: F) -> Duration
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let _result = operation();
        let duration = start.elapsed();
        duration
    }
    
    /// Measure named operation and store result
    pub fn measure_named_operation<F, R>(&mut self, name: &str, operation: F) -> (Duration, R)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.measurements.push((name.to_string(), duration));
        (duration, result)
    }
    
    /// Validate performance against target
    pub fn validate_performance(&self, operation_name: &str, target_ms: u64) -> bool {
        if let Some((_, duration)) = self.measurements.iter()
            .find(|(name, _)| name == operation_name) {
            duration.as_millis() < target_ms as u128
        } else {
            false
        }
    }
    
    /// Get performance report
    pub fn get_performance_report(&self) -> String {
        let mut report = String::from("Performance Report:\n");
        for (name, duration) in &self.measurements {
            report.push_str(&format!("  {}: {}ms\n", name, duration.as_millis()));
        }
        report
    }
    
    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

// Convenience functions for backward compatibility with existing tests
pub fn create_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::and_logic_documents();
    builder.create_index_with_documents(index_path, &documents[0..1]) // Single basic document
}

pub fn create_and_logic_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::and_logic_documents();
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_test_index_with_content(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::or_logic_documents();
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_basic_performance_index(index_path: &Path, num_docs: usize) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::performance_documents(num_docs);
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_performance_accuracy_index(index_path: &Path, num_docs: usize) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::performance_documents(num_docs);
    builder.create_index_with_documents(index_path, &documents)
}

pub fn measure_query_performance<F>(operation: F, operation_name: &str) -> Result<(Duration, usize)>
where
    F: FnOnce() -> Result<Vec<String>>,
{
    let mut timer = PerformanceTimer::new();
    let (duration, results) = timer.measure_named_operation(operation_name, operation)?;
    
    println!("{}: {}ms, {} results", operation_name, duration.as_millis(), results.len());
    
    Ok((duration, results.len()))
}
```

### 3. REFACTOR: Update lib.rs to export test utilities

```rust
// src/lib.rs - Add test utilities module
pub mod boolean;
pub mod cross_chunk;  
pub mod validator;

// Export test utilities for testing
#[cfg(test)]
pub mod test_utils;

// Re-export for external test usage
pub use test_utils::*;
```

### 4. Compilation Verification

```bash
# Compile test utilities module
cargo check --lib

# Run test utilities tests  
cargo test test_utilities_tests --lib

# Verify no regression in existing tests
cargo test --lib
```

// Convenience functions for backward compatibility with existing tasks
pub fn create_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::and_logic_documents();
    builder.create_index_with_documents(index_path, &documents[0..1]) // Single basic document
}

pub fn create_and_logic_test_index(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::and_logic_documents();
    builder.create_index_with_documents(index_path, &documents)
}

pub fn create_test_index_with_content(index_path: &Path) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::or_logic_documents();
    builder.create_index_with_documents(index_path, &documents)
}
```

### 3. REFACTOR: Update lib.rs to export basic test utilities

```rust
// src/lib.rs - Add basic test utilities module
pub mod boolean;
pub mod cross_chunk;  
pub mod validator;

// Export basic test utilities for testing
#[cfg(test)]
pub mod test_utils;
```

## Expected Benefits

1. **Eliminates Code Duplication**: Basic index creation functions standardized
2. **Consistent Schema**: All tests use same schema definition
3. **Maintainable Test Documents**: BasicTestDocuments provides reusable test data
4. **Backward Compatibility**: Existing test functions remain available via convenience wrappers
5. **Focused Scope**: Keeps task under 10-minute limit

## Quality Assessment Criteria

**Functionality (40%)**:
- [ ] StandardIndexBuilder creates consistent indices
- [ ] TestDocumentGenerator provides complete document sets  
- [ ] SchemaValidator correctly validates schemas
- [ ] PerformanceTimer measures operations accurately
- [ ] All convenience functions work with existing tests

**Integration (30%)**:
- [ ] Module compiles without errors
- [ ] Existing tests continue to pass with new utilities
- [ ] New test utilities work in realistic test scenarios
- [ ] Performance measurements are accurate and consistent

**Code Quality (20%)**:
- [ ] Clear, descriptive function names
- [ ] Comprehensive documentation
- [ ] Proper error handling throughout
- [ ] Consistent coding style and patterns

**Performance (10%)**:
- [ ] Test utilities don't add significant overhead
- [ ] Performance measurement utilities are lightweight
- [ ] Index creation performance is acceptable

## Verification Checklist

- [ ] New test utilities module compiles successfully
- [ ] All existing tests pass without modification
- [ ] New test utilities provide expected functionality
- [ ] Code duplication eliminated across Phase 2 tasks
- [ ] Performance measurement utilities work correctly
- [ ] Schema validation catches inconsistencies
- [ ] Documentation is complete and accurate

## Integration with Other Tasks

This test utilities module should be referenced by:
- Task 004: Use `create_test_index()` convenience function
- Task 005: Use `create_test_index_with_content()` convenience function  
- Task 006: Use `create_and_logic_test_index()` convenience function
- Task 010: Use `create_test_index()` convenience function
- Task 011: Use `create_test_index()` convenience function
- Task 014: Use `create_basic_performance_index()` and `PerformanceTimer`
- Task 021: Use `create_performance_accuracy_index()` and `PerformanceTimer`

**Expected Score**: 100/100 - This task eliminates basic test utility duplication and provides focused, standardized testing infrastructure that maintains backward compatibility while improving code quality.

Next task (003) will implement the Boolean Engine Constructor, which can now use these standardized test utilities for consistent testing.