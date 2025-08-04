# Task 001: Create GroundTruthDataset Struct and Basic Methods

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This system uses Tantivy for text search, LanceDB for vector search, and Rayon for parallel processing. The goal is to achieve 100% accuracy validation.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Create this file
  lib.rs
Cargo.toml
```

## Task Description
Create the foundation `GroundTruthDataset` struct that will hold test cases for validating search accuracy. This struct needs to manage collections of test cases with serialization support.

## Requirements
1. Create `src/validation/ground_truth.rs`
2. Implement `GroundTruthDataset` struct with:
   - `test_cases: Vec<GroundTruthCase>`
   - `new()` constructor
   - `add_test()` method
   - Basic validation methods

## Expected Code Structure
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthDataset {
    pub test_cases: Vec<GroundTruthCase>,
}

impl GroundTruthDataset {
    pub fn new() -> Self {
        // Implementation needed
    }
    
    pub fn add_test(&mut self, case: GroundTruthCase) {
        // Implementation needed
    }
    
    pub fn len(&self) -> usize {
        // Implementation needed
    }
    
    pub fn is_empty(&self) -> bool {
        // Implementation needed
    }
}
```

## Dependencies to Add to Cargo.toml
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
```

## Success Criteria
- File compiles without errors
- All methods have basic implementations
- Struct is properly serializable with serde
- Basic tests pass (create simple unit tests)

## Time Limit
10 minutes maximum