# Task 05: Add is_indexable_file() Filter Method

## Context
You are implementing Phase 4 of a vector indexing system. The parallel indexer needs to filter files to determine which ones should be indexed, excluding things like binary files, temporary files, and other non-indexable content.

## Current State
- `src/parallel.rs` exists with `ParallelIndexer` struct
- `get_index_path()` and `ensure_index_path_exists()` are implemented
- Core parallel indexing functionality is working

## Task Objective
Implement the `is_indexable_file()` method that filters files based on extension, size, and other criteria to determine if they should be indexed.

## Implementation Requirements

### 1. Add file extension constants
Add these constants at the top of `src/parallel.rs` after the imports:
```rust
// Supported file extensions for indexing
const INDEXABLE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "java", "cpp", "c", "h", "hpp",
    "go", "rb", "php", "cs", "swift", "kt", "scala", "clj", "hs", "ml",
    "txt", "md", "rst", "adoc", "tex", "org",
    "json", "yaml", "yml", "toml", "xml", "csv",
    "html", "css", "scss", "sass", "less",
    "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd"
];

// Files/directories to always skip
const SKIP_PATTERNS: &[&str] = &[
    ".git", ".svn", ".hg", ".bzr",
    "node_modules", "target", "build", "dist", "out", "bin", "obj",
    ".DS_Store", "Thumbs.db", ".tmp", ".temp",
    "__pycache__", ".pytest_cache", ".coverage",
    ".idea", ".vscode", "*.swp", "*.swo", "*~"
];

// Maximum file size to index (10 MB)
const MAX_FILE_SIZE_BYTES: u64 = 10 * 1024 * 1024;
```

### 2. Implement is_indexable_file() method
Add this method to the `ParallelIndexer` implementation:
```rust
fn is_indexable_file(&self, path: &Path) -> bool {
    // Skip if path contains any skip patterns
    let path_str = path.to_string_lossy();
    for pattern in SKIP_PATTERNS {
        if path_str.contains(pattern) {
            return false;
        }
    }
    
    // Check file extension
    if let Some(extension) = path.extension() {
        let ext_str = extension.to_string_lossy().to_lowercase();
        if !INDEXABLE_EXTENSIONS.contains(&ext_str.as_str()) {
            return false;
        }
    } else {
        // No extension - skip unless it's a known text file
        return self.is_likely_text_file(path);
    }
    
    // Check file size
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.len() > MAX_FILE_SIZE_BYTES {
            return false;
        }
    } else {
        // Can't read metadata - skip
        return false;
    }
    
    true
}

fn is_likely_text_file(&self, path: &Path) -> bool {
    // Check for common text files without extensions
    if let Some(filename) = path.file_name() {
        let filename_str = filename.to_string_lossy().to_lowercase();
        matches!(filename_str.as_str(), 
            "readme" | "license" | "changelog" | "authors" | "contributors" |
            "makefile" | "dockerfile" | "gemfile" | "rakefile" | "guardfile"
        )
    } else {
        false
    }
}
```

### 3. Add file filtering statistics
Enhance the `IndexingStats` struct to track filtering:
```rust
#[derive(Debug, Clone)]
pub struct IndexingStats {
    pub files_processed: usize,
    pub total_size: usize,
    pub start_time: Instant,
    pub files_per_second: f64,
    pub bytes_per_second: f64,
    pub files_skipped: usize,  // Add this field
    pub files_found: usize,    // Add this field
}

impl IndexingStats {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            total_size: 0,
            start_time: Instant::now(),
            files_per_second: 0.0,
            bytes_per_second: 0.0,
            files_skipped: 0,
            files_found: 0,
        }
    }
    
    // Update the summary method
    pub fn summary(&self) -> String {
        format!(
            "Found {} files, processed {} ({} skipped) - {:.2} MB in {:.2}s - Rate: {:.1} files/min, {:.2} MB/s",
            self.files_found,
            self.files_processed,
            self.files_skipped,
            self.total_size as f64 / (1024.0 * 1024.0),
            self.duration().as_secs_f64(),
            self.files_per_minute(),
            self.megabytes_per_second()
        )
    }
}
```

### 4. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_is_indexable_file() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test indexable files
    let rust_file = temp_dir.path().join("test.rs");
    let python_file = temp_dir.path().join("script.py");
    let text_file = temp_dir.path().join("README");
    
    assert!(parallel_indexer.is_indexable_file(&rust_file));
    assert!(parallel_indexer.is_indexable_file(&python_file));
    assert!(parallel_indexer.is_indexable_file(&text_file));
    
    // Test non-indexable files
    let binary_file = temp_dir.path().join("app.exe");
    let git_file = temp_dir.path().join(".git/config");
    let node_modules_file = temp_dir.path().join("node_modules/package.json");
    
    assert!(!parallel_indexer.is_indexable_file(&binary_file));
    assert!(!parallel_indexer.is_indexable_file(&git_file));
    assert!(!parallel_indexer.is_indexable_file(&node_modules_file));
    
    Ok(())
}

#[test]
fn test_file_size_filtering() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create a small file (should be indexable)
    let small_file = temp_dir.path().join("small.rs");
    std::fs::write(&small_file, "fn main() {}")?;
    assert!(parallel_indexer.is_indexable_file(&small_file));
    
    // Create a large file (should be skipped)
    let large_file = temp_dir.path().join("large.rs");
    let large_content = "a".repeat((MAX_FILE_SIZE_BYTES + 1) as usize);
    std::fs::write(&large_file, large_content)?;
    assert!(!parallel_indexer.is_indexable_file(&large_file));
    
    Ok(())
}

#[test]
fn test_extension_filtering() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test various supported extensions
    for ext in INDEXABLE_EXTENSIONS {
        let file_path = temp_dir.path().join(format!("test.{}", ext));
        assert!(parallel_indexer.is_indexable_file(&file_path), 
               "Extension {} should be indexable", ext);
    }
    
    // Test unsupported extensions
    let unsupported = ["bin", "exe", "dll", "so", "dylib", "jpg", "png", "gif", "mp4"];
    for ext in &unsupported {
        let file_path = temp_dir.path().join(format!("test.{}", ext));
        assert!(!parallel_indexer.is_indexable_file(&file_path),
               "Extension {} should not be indexable", ext);
    }
    
    Ok(())
}
```

## Success Criteria
- [ ] File extension filtering works correctly
- [ ] Skip patterns prevent indexing unwanted directories
- [ ] File size limits are enforced
- [ ] Text files without extensions are handled
- [ ] All filtering tests pass
- [ ] Statistics track skipped and found files
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- The extension list covers common programming and text file types
- Skip patterns prevent indexing build artifacts and VCS directories
- File size limit prevents memory issues with large files
- Consider expanding the extension list based on project needs