# Task 010: Document Level Validator Structure

## Prerequisites
- Task 001-009 completed: Boolean search engine fully functional
- BooleanSearchEngine with search_boolean() method working
- SearchResult struct available from boolean.rs
- BooleanQueryStructure enum available from validator.rs (Task 002)

## Required Imports
```rust
// Add to src/validator.rs for this task
use anyhow::{Result, Context};
use crate::boolean::{BooleanSearchEngine, SearchResult};
use tempfile::TempDir;
use std::path::Path;
use tantivy::schema::{Schema, TEXT, STORED};
use tantivy::{Index, doc};

// Import the CANONICAL BooleanQueryStructure from Task 002
use crate::validator::BooleanQueryStructure;
```

## Context
You have implemented boolean search functionality, but you need an additional validation layer. The DocumentLevelValidator ensures that search results truly satisfy boolean logic requirements.

This is important because:
- Tantivy might return results based on relevance scoring that don't strictly satisfy boolean requirements
- We need to validate that documents actually contain required terms and don't contain excluded terms
- Provides an extra quality assurance layer for boolean logic

## Your Task (10 minutes max)
Implement the DocumentLevelValidator structure with basic validation capabilities.

## Success Criteria
1. Write failing test for DocumentLevelValidator creation
2. Implement DocumentLevelValidator struct with necessary fields
3. Write failing test for basic validation method
4. Implement validate_boolean_results method structure
5. All tests pass after implementation

## Implementation Steps

### 1. RED: Write failing test for validator creation
```rust
// Create src/validator.rs
use anyhow::Result;

pub struct DocumentLevelValidator {
    // TODO: Add fields in next steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean::{BooleanSearchEngine, SearchResult};
    use tempfile::TempDir;
    use std::path::Path;
    
    #[test]
    fn test_validator_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        
        // Create test index (helper from previous tasks)
        create_test_index(&index_path)?;
        
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        
        // This should fail initially - struct doesn't have this field yet
        let validator = DocumentLevelValidator {
            search_engine: boolean_engine,
        };
        
        // Basic smoke test
        assert!(std::ptr::addr_of!(validator) as *const _ != std::ptr::null());
        
        Ok(())
    }
    
    // Helper function to create test index
    fn create_test_index(index_path: &Path) -> Result<()> {
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
        
        index_writer.add_document(doc!(
            file_path_field => "test.rs",
            content_field => "pub struct TestStruct { data: String }",
            raw_content_field => "pub struct TestStruct { data: String }",
            chunk_index_field => 0u64
        ))?;
        
        index_writer.commit()?;
        Ok(())
    }
}
```

### 2. GREEN: Implement DocumentLevelValidator structure
```rust
// Add to src/validator.rs
use crate::boolean::{BooleanSearchEngine, SearchResult};

pub struct DocumentLevelValidator {
    pub search_engine: BooleanSearchEngine,
}

impl DocumentLevelValidator {
    pub fn new(search_engine: BooleanSearchEngine) -> Self {
        Self { search_engine }
    }
}
```

### 3. RED: Write failing test for validation method
```rust
// Add to tests in src/validator.rs
#[test]
fn test_validate_boolean_results() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine);
    
    // Create test search result
    let test_result = SearchResult {
        file_path: "test.rs".to_string(),
        content: "pub struct TestStruct { data: String }".to_string(),
        chunk_index: 0,
        score: 1.0,
    };
    
    // This method doesn't exist yet - should fail
    let is_valid = validator.validate_boolean_results("pub AND struct", &[test_result])?;
    
    // Should return true for valid result
    assert!(is_valid, "Valid result should pass validation");
    
    Ok(())
}
```

### 4. GREEN: Implement basic validate_boolean_results method
```rust
// Add to DocumentLevelValidator impl
impl DocumentLevelValidator {
    pub fn validate_boolean_results(&self, query: &str, results: &[SearchResult]) -> Result<bool> {
        // Parse the query to understand what we're validating
        let parsed_query = self.parse_boolean_query(query)?;
        
        // Validate each result meets boolean requirements
        for result in results {
            if !self.document_satisfies_query(&result.content, &parsed_query)? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    // Placeholder for query parsing - will implement in next task
    fn parse_boolean_query(&self, _query: &str) -> Result<BooleanQueryStructure> {
        // Temporary simple implementation for AND queries
        if _query.contains(" AND ") {
            let terms: Vec<String> = _query.split(" AND ")
                .map(|s| s.trim().to_lowercase())
                .collect();
            Ok(BooleanQueryStructure::And(terms))
        } else {
            // Default to single term
            Ok(BooleanQueryStructure::And(vec![_query.to_lowercase()]))
        }
    }
    
    // Placeholder for document satisfaction - will implement in next task  
    fn document_satisfies_query(&self, content: &str, query: &BooleanQueryStructure) -> Result<bool> {
        match query {
            BooleanQueryStructure::And(terms) => {
                let content_lower = content.to_lowercase();
                Ok(terms.iter().all(|term| content_lower.contains(term)))
            }
            _ => Ok(true), // Placeholder for other query types
        }
    }
}

// NOTE: BooleanQueryStructure is imported from validator.rs (defined in Task 002)
// DO NOT redefine it here - use the canonical version from Task 002
```

### 5. REFACTOR: Add proper imports and error handling
```rust
// Add to top of src/validator.rs
use anyhow::{Result, Context};
use crate::boolean::{BooleanSearchEngine, SearchResult};

// Update lib.rs or mod.rs to include validator module
// Add to src/lib.rs or src/main.rs:
// pub mod validator;
```

## Validation Checklist
- [ ] DocumentLevelValidator struct compiles and can be created
- [ ] Basic validation method exists and compiles
- [ ] Simple AND query validation works
- [ ] Proper error handling with anyhow::Result
- [ ] All tests pass

## Creates for Future Tasks
- DocumentLevelValidator struct with validation capabilities
- Basic validate_boolean_results() method
- Document satisfaction checking logic

## Exports for Other Tasks
```rust
// From validator.rs
pub struct DocumentLevelValidator { search_engine }

impl DocumentLevelValidator {
    pub fn validate_boolean_results(&self, query: &str, results: &[SearchResult]) -> Result<bool> { ... }
}
```

## Context for Next Task
Next task will implement comprehensive query parsing to handle OR, NOT, and nested expressions in the validator, building on the validation structure created here.