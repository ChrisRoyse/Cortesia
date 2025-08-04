# Task 03: Complete IndexingStats Implementation

## Context
You are implementing Phase 4 of a vector indexing system. The `IndexingStats` struct was created in the previous task, but it needs additional methods for comprehensive statistics tracking and reporting.

## Current State
- `src/parallel.rs` exists with `ParallelIndexer` struct
- `IndexingStats` struct exists with basic fields
- `index_files_parallel()` method is implemented

## Task Objective
Complete the `IndexingStats` implementation with additional utility methods for better statistics reporting and analysis.

## Implementation Requirements

### 1. Enhance IndexingStats with additional methods
Replace the existing `IndexingStats` implementation in `src/parallel.rs`:
```rust
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct IndexingStats {
    pub files_processed: usize,
    pub total_size: usize,
    pub start_time: Instant,
    pub files_per_second: f64,
    pub bytes_per_second: f64,
}

impl IndexingStats {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            total_size: 0,
            start_time: Instant::now(),
            files_per_second: 0.0,
            bytes_per_second: 0.0,
        }
    }
    
    pub fn duration(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    pub fn calculate_rates(&mut self) {
        let duration_secs = self.duration().as_secs_f64();
        if duration_secs > 0.0 {
            self.files_per_second = self.files_processed as f64 / duration_secs;
            self.bytes_per_second = self.total_size as f64 / duration_secs;
        }
    }
    
    pub fn files_per_minute(&self) -> f64 {
        self.files_per_second * 60.0
    }
    
    pub fn megabytes_per_second(&self) -> f64 {
        self.bytes_per_second / (1024.0 * 1024.0)
    }
    
    pub fn average_file_size(&self) -> usize {
        if self.files_processed > 0 {
            self.total_size / self.files_processed
        } else {
            0
        }
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Processed {} files ({:.2} MB) in {:.2}s - Rate: {:.1} files/min, {:.2} MB/s",
            self.files_processed,
            self.total_size as f64 / (1024.0 * 1024.0),
            self.duration().as_secs_f64(),
            self.files_per_minute(),
            self.megabytes_per_second()
        )
    }
}
```

### 2. Update index_files_parallel() to calculate rates
Modify the end of the `index_files_parallel()` method:
```rust
pub fn index_files_parallel(&self, file_paths: Vec<PathBuf>) -> Result<IndexingStats> {
    let stats = Arc::new(Mutex::new(IndexingStats::new()));
    
    // ... existing parallel processing code ...
    
    let mut final_stats = stats.lock().unwrap().clone();
    final_stats.calculate_rates(); // Add this line
    Ok(final_stats)
}
```

### 3. Add comprehensive tests
Add these tests to the test module in `src/parallel.rs`:
```rust
#[test]
fn test_indexing_stats_calculations() -> Result<()> {
    let mut stats = IndexingStats::new();
    
    // Simulate some processing
    std::thread::sleep(std::time::Duration::from_millis(100));
    stats.files_processed = 10;
    stats.total_size = 1024 * 1024; // 1 MB
    stats.calculate_rates();
    
    assert_eq!(stats.files_processed, 10);
    assert_eq!(stats.total_size, 1024 * 1024);
    assert!(stats.files_per_second > 0.0);
    assert!(stats.bytes_per_second > 0.0);
    assert_eq!(stats.average_file_size(), 1024 * 1024 / 10);
    
    let summary = stats.summary();
    assert!(summary.contains("Processed 10 files"));
    assert!(summary.contains("files/min"));
    
    Ok(())
}

#[test]
fn test_indexing_stats_edge_cases() -> Result<()> {
    let stats = IndexingStats::new();
    
    // Test with zero files
    assert_eq!(stats.files_per_second, 0.0);
    assert_eq!(stats.bytes_per_second, 0.0);
    assert_eq!(stats.average_file_size(), 0);
    
    let summary = stats.summary();
    assert!(summary.contains("Processed 0 files"));
    
    Ok(())
}
```

### 4. Update the existing test to use new methods
Update the `test_index_files_parallel_basic` test:
```rust
#[test]
fn test_index_files_parallel_basic() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create test files
    let mut test_files = Vec::new();
    for i in 0..3 {
        let file_path = temp_dir.path().join(format!("test_{}.rs", i));
        let content = format!("pub fn test_{}() {{ println!(\"Hello {}\"); }}", i, i);
        fs::write(&file_path, content)?;
        test_files.push(file_path);
    }
    
    // Index files in parallel
    let stats = parallel_indexer.index_files_parallel(test_files)?;
    
    assert_eq!(stats.files_processed, 3);
    assert!(stats.total_size > 0);
    assert!(stats.duration().as_millis() < 5000);
    
    // Test new methods
    assert!(stats.files_per_second >= 0.0);
    assert!(stats.bytes_per_second >= 0.0);
    assert!(stats.average_file_size() > 0);
    
    println!("Stats: {}", stats.summary());
    
    Ok(())
}
```

## Success Criteria
- [ ] Enhanced `IndexingStats` with rate calculations
- [ ] `calculate_rates()` method implemented correctly
- [ ] `files_per_minute()` and `megabytes_per_second()` methods work
- [ ] `average_file_size()` calculation is correct
- [ ] `summary()` method provides useful output
- [ ] All tests pass including edge cases
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Focus on making the statistics useful and accurate
- Handle edge cases like zero files or zero duration
- The summary string should be human-readable
- Statistics should help identify performance bottlenecks