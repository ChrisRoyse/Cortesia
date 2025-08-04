# Task 06: Implement index_directory_parallel() Method

## Context
You are implementing Phase 4 of a vector indexing system. Now you need to implement directory traversal with parallel indexing, building on the file filtering and parallel processing capabilities implemented in previous tasks.

## Current State
- `src/parallel.rs` exists with `ParallelIndexer` struct
- `is_indexable_file()` filtering method is implemented
- `index_files_parallel()` method processes individual files
- `IndexingStats` tracks processing and filtering statistics

## Task Objective
Implement the `index_directory_parallel()` method that walks a directory tree, filters indexable files, and processes them in parallel.

## Implementation Requirements

### 1. Add walkdir dependency import
Add this import at the top of `src/parallel.rs`:
```rust
use walkdir::WalkDir;
```

### 2. Implement index_directory_parallel() method
Add this method to the `ParallelIndexer` implementation:
```rust
pub fn index_directory_parallel(&self, dir_path: &Path) -> Result<IndexingStats> {
    println!("Starting parallel indexing of directory: {}", dir_path.display());
    
    // Collect all files first
    let mut all_files = Vec::new();
    let mut total_found = 0;
    let mut total_skipped = 0;
    
    for entry in WalkDir::new(dir_path) {
        let entry = entry?;
        
        if entry.file_type().is_file() {
            total_found += 1;
            
            if self.is_indexable_file(entry.path()) {
                all_files.push(entry.path().to_path_buf());
            } else {
                total_skipped += 1;
            }
        }
    }
    
    println!("Found {} files to index ({} total, {} skipped)\", 
             all_files.len(), total_found, total_skipped);
    
    // Index files in parallel
    let mut stats = self.index_files_parallel(all_files)?;
    
    // Update the filtering statistics
    stats.files_found = total_found;
    stats.files_skipped = total_skipped;\n    
    Ok(stats)
}
```

### 3. Add directory validation method
Add this helper method to validate directories before processing:
```rust
fn validate_directory(&self, dir_path: &Path) -> Result<()> {
    if !dir_path.exists() {
        return Err(anyhow::anyhow!(\"Directory does not exist: {}\", dir_path.display()));
    }
    
    if !dir_path.is_dir() {
        return Err(anyhow::anyhow!(\"Path is not a directory: {}\", dir_path.display()));
    }
    
    // Check if we have read permissions
    let metadata = std::fs::metadata(dir_path)?;
    if metadata.permissions().readonly() {
        println!(\"Warning: Directory may be read-only: {}\", dir_path.display());
    }
    
    Ok(())
}
```

### 4. Update index_directory_parallel() to use validation
Modify the beginning of `index_directory_parallel()`:
```rust
pub fn index_directory_parallel(&self, dir_path: &Path) -> Result<IndexingStats> {
    // Validate directory first
    self.validate_directory(dir_path)?;
    
    println!(\"Starting parallel indexing of directory: {}\", dir_path.display());
    
    // ... rest of the existing implementation ...
}
```

### 5. Add progress reporting
Enhance the file collection with progress reporting:
```rust
pub fn index_directory_parallel(&self, dir_path: &Path) -> Result<IndexingStats> {
    self.validate_directory(dir_path)?;
    
    println!(\"Starting parallel indexing of directory: {}\", dir_path.display());
    
    let mut all_files = Vec::new();
    let mut total_found = 0;
    let mut total_skipped = 0;
    
    // Use iterator with progress reporting every 1000 files
    for (count, entry) in WalkDir::new(dir_path).into_iter().enumerate() {
        let entry = entry?;
        
        if count > 0 && count % 1000 == 0 {
            println!(\"  Scanned {} entries...\", count);
        }
        
        if entry.file_type().is_file() {
            total_found += 1;
            
            if self.is_indexable_file(entry.path()) {
                all_files.push(entry.path().to_path_buf());
            } else {
                total_skipped += 1;
            }
        }
    }
    
    println!(\"Found {} files to index ({} total, {} skipped)\", 
             all_files.len(), total_found, total_skipped);
    
    if all_files.is_empty() {
        println!(\"No indexable files found in directory\");
        return Ok(IndexingStats::new());
    }
    
    // Index files in parallel
    let mut stats = self.index_files_parallel(all_files)?;
    
    // Update the filtering statistics
    stats.files_found = total_found;
    stats.files_skipped = total_skipped;
    
    println!(\"Indexing complete: {}\", stats.summary());
    
    Ok(stats)
}
```

### 6. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_index_directory_parallel() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join(\"index\");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create a test directory structure
    let test_dir = temp_dir.path().join(\"test_project\");
    std::fs::create_dir_all(&test_dir)?;
    
    // Create indexable files
    let src_dir = test_dir.join(\"src\");
    std::fs::create_dir_all(&src_dir)?;
    
    std::fs::write(src_dir.join(\"main.rs\"), \"fn main() { println!(\\\"Hello\\\"); }\")?;
    std::fs::write(src_dir.join(\"lib.rs\"), \"pub mod utils;\")?;
    std::fs::write(test_dir.join(\"README.md\"), \"# Test Project\")?;
    
    // Create non-indexable files
    let target_dir = test_dir.join(\"target\");
    std::fs::create_dir_all(&target_dir)?;
    std::fs::write(target_dir.join(\"binary.exe\"), \"binary data\")?;
    
    // Index the directory
    let stats = parallel_indexer.index_directory_parallel(&test_dir)?;
    
    // Should have processed the indexable files only\n    assert_eq!(stats.files_processed, 3); // main.rs, lib.rs, README.md\n    assert!(stats.files_found >= 3); // At least the files we created\n    assert!(stats.files_skipped >= 1); // At least the binary file\n    assert!(stats.total_size > 0);\n    \n    Ok(())\n}\n\n#[test]\nfn test_directory_validation() -> Result<()> {\n    let temp_dir = TempDir::new()?;\n    let index_path = temp_dir.path().join(\"index\");\n    let parallel_indexer = ParallelIndexer::new(&index_path)?;\n    \n    // Test with existing directory\n    let existing_dir = temp_dir.path().join(\"existing\");\n    std::fs::create_dir_all(&existing_dir)?;\n    assert!(parallel_indexer.validate_directory(&existing_dir).is_ok());\n    \n    // Test with non-existent directory\n    let non_existent = temp_dir.path().join(\"does_not_exist\");\n    assert!(parallel_indexer.validate_directory(&non_existent).is_err());\n    \n    // Test with file instead of directory\n    let file_path = temp_dir.path().join(\"not_a_dir.txt\");\n    std::fs::write(&file_path, \"content\")?;\n    assert!(parallel_indexer.validate_directory(&file_path).is_err());\n    \n    Ok(())\n}\n\n#[test]\nfn test_empty_directory() -> Result<()> {\n    let temp_dir = TempDir::new()?;\n    let index_path = temp_dir.path().join(\"index\");\n    let parallel_indexer = ParallelIndexer::new(&index_path)?;\n    \n    // Create empty directory\n    let empty_dir = temp_dir.path().join(\"empty\");\n    std::fs::create_dir_all(&empty_dir)?;\n    \n    // Should handle empty directory gracefully\n    let stats = parallel_indexer.index_directory_parallel(&empty_dir)?;\n    \n    assert_eq!(stats.files_processed, 0);\n    assert_eq!(stats.files_found, 0);\n    assert_eq!(stats.files_skipped, 0);\n    \n    Ok(())\n}\n```\n\n## Success Criteria\n- [ ] `index_directory_parallel()` walks directory trees correctly\n- [ ] Directory validation prevents errors with invalid paths\n- [ ] File filtering is applied during directory traversal\n- [ ] Progress reporting works for large directories\n- [ ] Statistics accurately reflect filtering results\n- [ ] All tests pass including edge cases\n- [ ] No compilation errors or warnings\n\n## Time Limit\n10 minutes\n\n## Notes\n- Use `walkdir::WalkDir` for recursive directory traversal\n- Progress reporting helps with large directory scans\n- Validate directories before processing to provide clear error messages\n- Handle empty directories gracefully\n- Statistics should differentiate between found, processed, and skipped files