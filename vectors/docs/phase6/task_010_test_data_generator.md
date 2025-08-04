# Task 010: Create TestDataGenerator Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The TestDataGenerator creates diverse test files for validating different query types and edge cases.

## Project Structure
```
src/
  validation/
    test_data.rs       <- Create this file
  lib.rs
```

## Task Description
Create the `TestDataGenerator` struct that generates comprehensive test datasets including special characters, boolean logic, proximity, wildcards, large files, and edge cases.

## Requirements
1. Create `src/validation/test_data.rs`
2. Implement `TestDataGenerator` struct
3. Add methods for creating different types of test files
4. Support both small targeted tests and large stress tests
5. Include Windows-specific test cases

## Expected Code Structure
```rust
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

pub struct TestDataGenerator {
    output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTestSet {
    pub files: Vec<GeneratedTestFile>,
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTestFile {
    pub path: PathBuf,
    pub content_type: TestFileType,
    pub size_bytes: u64,
    pub expected_matches: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestFileType {
    SpecialCharacters,
    BooleanLogic,
    Proximity,
    Wildcard,
    Regex,
    LargeFile,
    Empty,
    Unicode,
    WindowsPaths,
    SyntheticCode,
}

impl TestDataGenerator {
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        
        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("Failed to create test data directory: {}", output_dir.display()))?;
        
        Ok(Self { output_dir })
    }
    
    pub fn generate_comprehensive_test_set(&self) -> Result<GeneratedTestSet> {
        let start_time = std::time::Instant::now();
        let mut generated_files = Vec::new();
        
        // Generate different types of test files
        generated_files.extend(self.generate_special_characters_tests()?);
        generated_files.extend(self.generate_boolean_logic_tests()?);
        generated_files.extend(self.generate_proximity_tests()?);
        generated_files.extend(self.generate_wildcard_tests()?);
        generated_files.extend(self.generate_unicode_tests()?);
        generated_files.extend(self.generate_edge_case_tests()?);
        
        // Generate synthetic code files for stress testing
        generated_files.extend(self.generate_synthetic_rust_files(100)?);
        
        // Calculate total size
        let total_size: u64 = generated_files.iter().map(|f| f.size_bytes).sum();
        
        let test_set = GeneratedTestSet {
            total_files: generated_files.len(),
            total_size_bytes: total_size,
            generation_time_ms: start_time.elapsed().as_millis() as u64,
            files: generated_files,
        };
        
        println!("Generated {} test files ({:.2} MB) in {}ms", 
                test_set.total_files, 
                total_size as f64 / 1_048_576.0,
                test_set.generation_time_ms);
        
        Ok(test_set)
    }
    
    fn create_test_file(&self, filename: &str, content: &str, file_type: TestFileType) -> Result<GeneratedTestFile> {
        let file_path = self.output_dir.join(filename);
        
        fs::write(&file_path, content)
            .with_context(|| format!("Failed to write test file: {}", file_path.display()))?;
        
        let size_bytes = content.len() as u64;
        
        Ok(GeneratedTestFile {
            path: file_path,
            content_type: file_type,
            size_bytes,
            expected_matches: Vec::new(), // To be filled by specific generators
        })
    }
    
    pub fn cleanup(&self) -> Result<()> {
        if self.output_dir.exists() {
            fs::remove_dir_all(&self.output_dir)
                .with_context(|| format!("Failed to cleanup test directory: {}", self.output_dir.display()))?;
        }
        Ok(())
    }
    
    pub fn get_test_file_path(&self, filename: &str) -> PathBuf {
        self.output_dir.join(filename)
    }
    
    pub fn count_files(&self) -> Result<usize> {
        let mut count = 0;
        for entry in fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                count += 1;
            }
        }
        Ok(count)
    }
}
```

## Success Criteria
- TestDataGenerator struct compiles without errors
- Directory creation and file writing work correctly
- Generated test set structure captures all necessary information
- Cleanup method properly removes test files
- File counting and path methods work correctly
- Error handling is comprehensive

## Time Limit
10 minutes maximum