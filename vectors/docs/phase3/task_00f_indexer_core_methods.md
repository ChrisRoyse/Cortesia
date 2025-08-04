# Task 00f: Add DocumentIndexer Core Constructor and Basic Methods

**Estimated Time: 10 minutes**
**Lines of Code: 25**
**Prerequisites: Task 00e completed**

## Context

Phase 3 tasks need to create DocumentIndexer instances and access basic functionality. This task adds the core methods.

## Your Task

Add the `impl DocumentIndexer` block with constructor and essential methods to `src/indexer.rs`.

## Required Implementation

Add this implementation block to `src/indexer.rs`:

```rust
impl DocumentIndexer {
    /// Create a new document indexer
    pub fn new(index_path: &Path) -> Result<Self> {
        let engine = BooleanSearchEngine::new(index_path)?;
        
        Ok(Self {
            engine,
            config: IndexingConfig::default(),
        })
    }
    
    /// Update indexing configuration
    pub fn set_config(&mut self, config: IndexingConfig) {
        self.config = config;
    }
    
    /// Get current indexing configuration
    pub fn config(&self) -> &IndexingConfig {
        &self.config
    }
    
    /// Get reference to the underlying search engine
    pub fn search_engine(&self) -> &BooleanSearchEngine {
        &self.engine
    }
}
```

## Success Criteria

- [ ] `impl DocumentIndexer` block added
- [ ] `new()` constructor method accepting `&Path`
- [ ] `set_config()` method for updating configuration
- [ ] `config()` getter method returning reference
- [ ] `search_engine()` getter method returning engine reference
- [ ] All methods have proper return types
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Next Task

Task 00g will add file indexing methods to DocumentIndexer.