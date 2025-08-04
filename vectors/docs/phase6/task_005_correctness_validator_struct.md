# Task 005: Create CorrectnessValidator Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The CorrectnessValidator is the core engine that executes queries and validates results against ground truth data.

## Project Structure
```
src/
  validation/
    ground_truth.rs    <- Already exists
    correctness.rs     <- Create this file
  lib.rs
```

## Task Description
Create the `CorrectnessValidator` struct that integrates with the UnifiedSearchSystem to validate query results against ground truth test cases. This is the main validation engine.

## Requirements
1. Create `src/validation/correctness.rs`
2. Implement `CorrectnessValidator` struct
3. Add async constructor for system integration
4. Add search mode determination method
5. Create proper error handling

## Expected Code Structure
```rust
use crate::validation::ground_truth::{GroundTruthCase, QueryType};
use anyhow::{Result, Context};
use std::path::Path;

// Note: These imports assume the main search system exists
// You may need to adjust based on actual implementation
use crate::{UnifiedSearchSystem, SearchMode, UnifiedResult};

pub struct CorrectnessValidator {
    search_system: UnifiedSearchSystem,
}

impl CorrectnessValidator {
    pub async fn new<P: AsRef<Path>>(
        text_index_path: P,
        vector_db_path: &str,
    ) -> Result<Self> {
        let search_system = UnifiedSearchSystem::new(text_index_path.as_ref(), vector_db_path)
            .await
            .context("Failed to initialize search system for validation")?;
        
        Ok(Self { search_system })
    }
    
    pub fn determine_search_mode(&self, query_type: &QueryType) -> SearchMode {
        match query_type {
            QueryType::Vector => SearchMode::VectorOnly,
            QueryType::Hybrid => SearchMode::Hybrid,
            _ => SearchMode::TextOnly,
        }
    }
    
    pub async fn health_check(&self) -> Result<()> {
        // Verify the search system is working
        // Implementation needed:
        // - Test basic query execution
        // - Verify both text and vector indices are accessible
        // - Check system resources
    }
    
    pub fn get_system_info(&self) -> SystemInfo {
        // Return information about the search system state
        // Implementation needed
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub text_index_docs: usize,
    pub vector_index_docs: usize,
    pub memory_usage_mb: f64,
    pub index_version: String,
}
```

## Dependencies to Add to Cargo.toml
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
```

## Mock Implementations (if main system not ready)
If the main UnifiedSearchSystem doesn't exist yet, create mock versions:
```rust
// Temporary mock for development
pub struct UnifiedSearchSystem;
pub enum SearchMode { TextOnly, VectorOnly, Hybrid }
pub struct UnifiedResult {
    pub file_path: String,
    pub content: String,
    pub score: f64,
}

impl UnifiedSearchSystem {
    pub async fn new(_text_path: &Path, _vector_path: &str) -> Result<Self> {
        Ok(Self)
    }
    
    pub async fn search_hybrid(&self, _query: &str, _mode: SearchMode) -> Result<Vec<UnifiedResult>> {
        Ok(vec![])
    }
}
```

## Success Criteria
- CorrectnessValidator struct compiles
- Constructor properly initializes the search system
- Search mode determination works correctly
- Health check method is implemented
- Integration points are properly defined

## Time Limit
10 minutes maximum