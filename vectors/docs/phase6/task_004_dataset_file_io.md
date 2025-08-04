# Task 004: Implement Dataset File I/O Methods

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-003. The dataset needs to be able to save and load ground truth test cases from JSON files.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement file I/O methods for the GroundTruthDataset to save and load test cases from JSON files. This enables persistent storage of validation test cases.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Implement `load_from_file()` method
3. Implement `save_to_file()` method
4. Add proper error handling with anyhow
5. Create example JSON structure

## Expected Code Structure to Add
```rust
use std::path::Path;
use anyhow::{Result, Context};

impl GroundTruthDataset {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read ground truth file: {}", path.display()))?;
        
        let dataset: GroundTruthDataset = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse ground truth JSON: {}", path.display()))?;
        
        // Validate loaded dataset
        // Implementation needed
        
        Ok(dataset)
    }
    
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize ground truth dataset")?;
        
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write ground truth file: {}", path.display()))?;
        
        Ok(())
    }
    
    pub fn validate(&self) -> Result<()> {
        // Validate all test cases in the dataset
        // Implementation needed:
        // - Check for duplicate queries
        // - Validate file paths exist (if checking against real files)
        // - Ensure required fields are not empty
        // - Validate query types match query content
    }
    
    pub fn merge(&mut self, other: GroundTruthDataset) {
        // Merge another dataset into this one, avoiding duplicates
        // Implementation needed
    }
}
```

## Example JSON Structure to Document
```json
{
  "test_cases": [
    {
      "query": "[workspace]",
      "expected_files": ["Cargo.toml", "src/lib.rs"],
      "expected_count": 2,
      "must_contain": ["[workspace]"],
      "must_not_contain": ["[dependencies]"],
      "query_type": "SpecialCharacters"
    }
  ]
}
```

## Success Criteria
- File I/O methods work correctly
- Proper error handling with meaningful messages
- JSON serialization/deserialization works
- Validation method catches common issues
- Example JSON structure is documented

## Time Limit
10 minutes maximum