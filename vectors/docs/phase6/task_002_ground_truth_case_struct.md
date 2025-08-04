# Task 002: Create GroundTruthCase Struct with Query Types

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 001. The GroundTruthCase represents individual test cases that validate specific query behaviors and expected results.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `GroundTruthCase` struct that represents individual test cases for validation. Each case contains a query, expected results, and validation criteria.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Implement `GroundTruthCase` struct with all required fields
3. Create helper methods for case validation
4. Ensure proper serialization support

## Expected Code Structure to Add
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthCase {
    pub query: String,
    pub expected_files: Vec<String>,
    pub expected_count: usize,
    pub must_contain: Vec<String>,
    pub must_not_contain: Vec<String>,
    pub query_type: QueryType,
}

impl GroundTruthCase {
    pub fn new(
        query: String,
        expected_files: Vec<String>,
        query_type: QueryType,
    ) -> Self {
        // Implementation needed - calculate expected_count from files
    }
    
    pub fn with_content_requirements(
        mut self,
        must_contain: Vec<String>,
        must_not_contain: Vec<String>,
    ) -> Self {
        // Implementation needed
    }
    
    pub fn validate_result_count(&self, actual_count: usize) -> bool {
        // Implementation needed
    }
    
    pub fn validate_files(&self, actual_files: &[String]) -> (Vec<String>, Vec<String>) {
        // Return (missing_files, unexpected_files)
        // Implementation needed
    }
}
```

## Dependencies
Same as Task 001 - should already be in Cargo.toml

## Success Criteria
- GroundTruthCase struct compiles without errors
- All methods have working implementations
- Struct integrates properly with GroundTruthDataset
- Helper methods provide useful validation logic
- Proper error handling with clear return values

## Time Limit
10 minutes maximum