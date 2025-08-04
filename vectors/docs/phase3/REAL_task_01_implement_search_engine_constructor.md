# REAL Task 01: Implement SearchEngine Constructor

**Estimated Time: 10 minutes**  
**TDD Approach: RED-GREEN-REFACTOR**  
**Prerequisites: None - extends existing stub**

## Context
The existing `SearchEngine::new()` method in `src/search/mod.rs` returns `todo!()`. This task implements the actual constructor using Tantivy's Index API to create a working search engine with a proper schema.

## Your Task - TDD RED PHASE
**WRITE THE FAILING TEST FIRST**

Create/update `src/search/mod.rs` with this test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine_new() -> Result<(), SearchError> {
        // RED: This should fail because SearchEngine::new() returns todo!()
        let engine = SearchEngine::new()?;
        
        // Basic verification that we have a working engine
        assert!(engine.get_schema().get_field("content").is_ok());
        Ok(())
    }
}
```

**Verify test fails**: `cargo test test_search_engine_new` should fail with `todo!()` panic.

## Your Task - TDD GREEN PHASE
**Write minimum code to make test pass**

Replace the existing SearchEngine struct and implementation:

```rust
use tantivy::{Index, schema::*};

/// Search engine for querying indexed documents
pub struct SearchEngine {
    index: Index,
    content_field: Field,
}

impl SearchEngine {
    pub fn new() -> Result<Self, SearchError> {
        // Create schema with content field
        let mut schema_builder = Schema::builder();
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let schema = schema_builder.build();
        
        // Create in-memory index
        let index = Index::create_in_ram(schema);
        
        Ok(SearchEngine {
            index,
            content_field,
        })
    }
    
    // Helper method for test verification
    pub fn get_schema(&self) -> &Schema {
        self.index.schema()
    }
}
```

**Verify test passes**: `cargo test test_search_engine_new` should now pass.

## Your Task - TDD REFACTOR PHASE
**Clean up the code while keeping tests green**

Add proper error handling and documentation:

```rust
impl SearchEngine {
    /// Creates a new SearchEngine with an in-memory Tantivy index
    /// 
    /// # Returns
    /// - `Ok(SearchEngine)` on successful creation
    /// - `Err(SearchError)` if index creation fails
    pub fn new() -> Result<Self, SearchError> {
        let mut schema_builder = Schema::builder();
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let schema = schema_builder.build();
        
        let index = Index::create_in_ram(schema);
        
        Ok(SearchEngine {
            index,
            content_field,
        })
    }
    
    /// Gets the Tantivy schema for this search engine
    pub fn get_schema(&self) -> &Schema {
        self.index.schema()
    }
}
```

## Validation Checklist
- [ ] Test fails initially (`cargo test test_search_engine_new` panics with `todo!()`)
- [ ] After implementation, test passes
- [ ] Code compiles without warnings (`cargo build`)
- [ ] No existing functionality broken (`cargo test`)
- [ ] SearchEngine can be constructed successfully
- [ ] Schema contains expected "content" field

## Integration Notes
- This task extends the existing SearchEngine in `src/search/mod.rs`
- Does not create duplicate types or modules
- Uses real Tantivy API (no mocks or stubs)
- Maintains compatibility with existing code structure
- Ready for Task 02 to add query parsing

## Success Criteria
**This task is complete when:**
1. `SearchEngine::new()` creates a working Tantivy index
2. Test `test_search_engine_new` passes
3. No `todo!()` remains in the implementation
4. Code follows RED-GREEN-REFACTOR cycle
5. Integration with existing codebase is clean