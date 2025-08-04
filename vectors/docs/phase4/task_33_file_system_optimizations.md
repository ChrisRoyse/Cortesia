# Task 33: Implement File System Optimizations

## Context
You are implementing Phase 4 of a vector indexing system. Cross-platform testing has been implemented. Now you need to create comprehensive file system optimizations that leverage Windows-specific features, handle different filesystem types efficiently, and provide maximum performance for the indexing system.

## Current State
- `src/windows.rs` has cross-platform testing and compatibility layers
- Platform detection and capability assessment are working
- Need filesystem-specific optimizations for maximum indexing performance
- Must handle NTFS, FAT32, ExFAT, and ReFS filesystems efficiently

## Task Objective
Implement comprehensive file system optimizations with Windows-specific features, efficient directory traversal, metadata caching, and performance monitoring for optimal indexing speed.

## Implementation Requirements

### 1. Add filesystem detection and optimization
Add this filesystem optimization system to `src/windows.rs`:
```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum FileSystemType {
    NTFS,
    FAT32,
    ExFAT,
    ReFS,
    UDF,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct FileSystemCapabilities {
    pub fs_type: FileSystemType,
    pub supports_compression: bool,
    pub supports_encryption: bool,
    pub supports_hard_links: bool,
    pub supports_sparse_files: bool,
    pub supports_reparse_points: bool,
    pub supports_alternate_streams: bool,
    pub supports_file_ids: bool,
    pub supports_short_names: bool,
    pub max_file_size: u64,
    pub max_volume_size: u64,
    pub cluster_size: u32,
    pub case_sensitive: bool,
    pub unicode_filenames: bool,
}

impl FileSystemCapabilities {
    pub fn detect_for_path(path: &Path) -> Result<Self> {
        #[cfg(windows)]
        {
            Self::detect_windows_filesystem(path)
        }
        #[cfg(not(windows))]
        {
            Ok(Self::generic_capabilities())
        }
    }
    
    #[cfg(windows)]
    fn detect_windows_filesystem(path: &Path) -> Result<Self> {
        use std::process::Command;
        
        // Get the root path for the drive
        let root_path = if let Some(prefix) = path.components().next() {
            match prefix {
                std::path::Component::Prefix(prefix_component) => {
                    format!("{}\\", prefix_component.as_os_str().to_string_lossy())
                }
                _ => "C:\\".to_string(),
            }
        } else {
            "C:\\".to_string()
        };
        
        // Use fsutil to get filesystem information
        let output = Command::new("fsutil")
            .args(&["fsinfo", "volumeinfo", &root_path])
            .output()?;
        
        let info = String::from_utf8_lossy(&output.stdout);
        
        let fs_type = if info.contains("NTFS") {
            FileSystemType::NTFS
        } else if info.contains("FAT32") {
            FileSystemType::FAT32
        } else if info.contains("exFAT") {
            FileSystemType::ExFAT
        } else if info.contains("ReFS") {
            FileSystemType::ReFS
        } else {
            FileSystemType::Unknown("Detected".to_string())
        };
        
        Ok(Self::capabilities_for_filesystem(fs_type))
    }
    
    fn capabilities_for_filesystem(fs_type: FileSystemType) -> Self {
        match fs_type {
            FileSystemType::NTFS => Self {
                fs_type,
                supports_compression: true,
                supports_encryption: true,
                supports_hard_links: true,
                supports_sparse_files: true,
                supports_reparse_points: true,
                supports_alternate_streams: true,
                supports_file_ids: true,
                supports_short_names: true,
                max_file_size: 16_u64.pow(12), // 16TB
                max_volume_size: 256_u64.pow(4), // 256TB
                cluster_size: 4096,
                case_sensitive: false, // Can be enabled on Windows 10+
                unicode_filenames: true,
            },
            FileSystemType::FAT32 => Self {
                fs_type,
                supports_compression: false,
                supports_encryption: false,
                supports_hard_links: false,
                supports_sparse_files: false,
                supports_reparse_points: false,
                supports_alternate_streams: false,
                supports_file_ids: false,
                supports_short_names: true,
                max_file_size: 4_294_967_295, // 4GB - 1 byte
                max_volume_size: 2_199_023_255_552, // 2TB
                cluster_size: 32768,
                case_sensitive: false,
                unicode_filenames: true,
            },
            FileSystemType::ExFAT => Self {
                fs_type,
                supports_compression: false,
                supports_encryption: false,
                supports_hard_links: false,
                supports_sparse_files: false,
                supports_reparse_points: false,
                supports_alternate_streams: false,
                supports_file_ids: false,
                supports_short_names: false,
                max_file_size: 16_u64.pow(12), // 16TB
                max_volume_size: 128_u64.pow(4), // 128PB theoretical
                cluster_size: 32768,
                case_sensitive: false,
                unicode_filenames: true,
            },
            FileSystemType::ReFS => Self {
                fs_type,
                supports_compression: false, // ReFS v2+ supports compression
                supports_encryption: false,
                supports_hard_links: false,
                supports_sparse_files: true,
                supports_reparse_points: true,
                supports_alternate_streams: false,
                supports_file_ids: true,
                supports_short_names: false,
                max_file_size: 35_u64.pow(4), // 35PB
                max_volume_size: 35_u64.pow(4), // 35PB
                cluster_size: 4096,
                case_sensitive: false,
                unicode_filenames: true,
            },
            _ => Self::generic_capabilities(),
        }
    }
    
    fn generic_capabilities() -> Self {
        Self {
            fs_type: FileSystemType::Unknown("Generic".to_string()),
            supports_compression: false,
            supports_encryption: false,
            supports_hard_links: false,
            supports_sparse_files: false,
            supports_reparse_points: false,
            supports_alternate_streams: false,
            supports_file_ids: false,
            supports_short_names: false,
            max_file_size: u64::MAX,
            max_volume_size: u64::MAX,
            cluster_size: 4096,
            case_sensitive: true,
            unicode_filenames: true,
        }
    }
}

pub struct FileSystemOptimizer {
    capabilities: FileSystemCapabilities,
    cache: HashMap<PathBuf, CachedMetadata>,
    stats: OptimizationStats,
}

#[derive(Debug, Clone)]
struct CachedMetadata {
    size: u64,
    modified_time: std::time::SystemTime,
    is_directory: bool,
    cached_at: Instant,
    file_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub directory_scans: u64,
    pub files_processed: u64,
    pub total_bytes_processed: u64,
    pub optimization_time_saved: Duration,
    pub filesystem_operations: u64,
}

impl OptimizationStats {
    pub fn new() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            directory_scans: 0,
            files_processed: 0,
            total_bytes_processed: 0,
            optimization_time_saved: Duration::new(0, 0),
            filesystem_operations: 0,
        }
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl FileSystemOptimizer {
    pub fn new(path: &Path) -> Result<Self> {
        let capabilities = FileSystemCapabilities::detect_for_path(path)?;
        
        Ok(Self {
            capabilities,
            cache: HashMap::new(),
            stats: OptimizationStats::new(),
        })
    }
    
    pub fn get_capabilities(&self) -> &FileSystemCapabilities {
        &self.capabilities
    }
    
    pub fn get_stats(&self) -> &OptimizationStats {
        &self.stats
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    pub fn optimized_metadata(&mut self, path: &Path) -> Result<std::fs::Metadata> {
        let start = Instant::now();
        
        if let Some(cached) = self.check_cache(path) {
            self.stats.cache_hits += 1;
            self.stats.optimization_time_saved += start.elapsed();
            return Ok(self.metadata_from_cached(&cached));
        }
        
        self.stats.cache_misses += 1;
        self.stats.filesystem_operations += 1;
        
        let metadata = std::fs::metadata(path)?;
        
        // Cache the metadata
        let cached = CachedMetadata {
            size: metadata.len(),
            modified_time: metadata.modified().unwrap_or(std::time::UNIX_EPOCH),
            is_directory: metadata.is_dir(),
            cached_at: Instant::now(),
            file_id: self.get_file_id(path).ok(),
        };
        
        self.cache.insert(path.to_path_buf(), cached);
        
        Ok(metadata)
    }
    
    fn check_cache(&self, path: &Path) -> Option<&CachedMetadata> {
        if let Some(cached) = self.cache.get(path) {
            // Cache is valid for 30 seconds
            if cached.cached_at.elapsed() < Duration::from_secs(30) {
                return Some(cached);
            }
        }
        None
    }
    
    fn metadata_from_cached(&self, cached: &CachedMetadata) -> std::fs::Metadata {
        // This is a simplified approach - in reality, we'd need to construct
        // a proper Metadata object or use a different caching strategy
        // For now, we'll return the actual metadata but track the cache hit
        unimplemented!("This would require platform-specific metadata construction")
    }
    
    #[cfg(windows)]
    fn get_file_id(&self, path: &Path) -> Result<u64> {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        if !self.capabilities.supports_file_ids {
            return Err(anyhow::anyhow!("File IDs not supported on this filesystem"));
        }
        
        // This would use Windows API to get the file ID
        // Simplified implementation
        Ok(0)
    }
    
    #[cfg(not(windows))]
    fn get_file_id(&self, _path: &Path) -> Result<u64> {
        Err(anyhow::anyhow!("File IDs not supported on this platform"))
    }
    
    pub fn optimized_read_dir(&mut self, path: &Path) -> Result<Vec<std::fs::DirEntry>> {
        let start = Instant::now();
        self.stats.directory_scans += 1;
        self.stats.filesystem_operations += 1;
        
        let mut entries = Vec::new();
        
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            entries.push(entry);
        }
        
        // Sort entries for consistent processing order
        entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        
        // Apply filesystem-specific optimizations
        if self.capabilities.fs_type == FileSystemType::NTFS {
            self.apply_ntfs_optimizations(&mut entries)?;
        }
        
        let processing_time = start.elapsed();
        self.stats.optimization_time_saved += processing_time;
        
        Ok(entries)
    }
    
    fn apply_ntfs_optimizations(&mut self, entries: &mut Vec<std::fs::DirEntry>) -> Result<()> {
        // NTFS-specific optimizations
        
        // 1. Skip alternate data streams if they're not needed
        entries.retain(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            !name_str.contains(':') // Skip ADS entries
        });
        
        // 2. Use short names if available for performance (when supported)
        if self.capabilities.supports_short_names {
            // This would involve additional Windows API calls
        }
        
        // 3. Batch metadata operations
        self.batch_metadata_operations(entries)?;
        
        Ok(())
    }
    
    fn batch_metadata_operations(&mut self, entries: &[std::fs::DirEntry]) -> Result<()> {
        // Batch multiple metadata operations for efficiency
        let mut batch_size = 0;
        let max_batch_size = 100;
        
        for entry in entries {
            if batch_size >= max_batch_size {
                // Process batch
                batch_size = 0;
            }
            
            // Add to batch processing
            batch_size += 1;
        }
        
        Ok(())
    }
    
    pub fn optimize_for_indexing(&mut self, path: &Path) -> Result<IndexingOptimization> {
        let start = Instant::now();
        
        let mut optimization = IndexingOptimization::new();
        
        // Analyze directory structure
        optimization.directory_depth = self.calculate_directory_depth(path)?;
        optimization.estimated_file_count = self.estimate_file_count(path)?;
        optimization.recommended_batch_size = self.calculate_optimal_batch_size()?;
        
        // Filesystem-specific recommendations
        match self.capabilities.fs_type {
            FileSystemType::NTFS => {
                optimization.recommendations.push("Use file IDs for tracking".to_string());
                optimization.recommendations.push("Enable compression detection".to_string());
                if self.capabilities.supports_sparse_files {
                    optimization.recommendations.push("Skip sparse files if not indexing content".to_string());
                }
            }
            FileSystemType::FAT32 => {
                optimization.recommendations.push("Be aware of 4GB file size limit".to_string());
                optimization.recommendations.push("Use sequential access patterns".to_string());
            }
            FileSystemType::ExFAT => {
                optimization.recommendations.push("Optimize for large files".to_string());
                optimization.recommendations.push("Use larger read buffers".to_string());
            }
            FileSystemType::ReFS => {
                optimization.recommendations.push("Leverage built-in integrity checking".to_string());
                optimization.recommendations.push("Use sparse file detection".to_string());
            }
            _ => {
                optimization.recommendations.push("Use generic optimization strategies".to_string());
            }
        }
        
        optimization.analysis_time = start.elapsed();
        
        Ok(optimization)
    }
    
    fn calculate_directory_depth(&mut self, path: &Path) -> Result<u32> {
        let mut max_depth = 0;
        let mut current_depth = 0;
        
        fn visit_directory(
            optimizer: &mut FileSystemOptimizer,
            dir: &Path,
            current_depth: u32,
            max_depth: &mut u32,
        ) -> Result<()> {
            *max_depth = (*max_depth).max(current_depth);
            
            if current_depth > 20 {
                // Prevent infinite recursion
                return Ok(());
            }
            
            for entry in optimizer.optimized_read_dir(dir)? {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    visit_directory(optimizer, &entry_path, current_depth + 1, max_depth)?;
                }
            }
            
            Ok(())
        }
        
        visit_directory(self, path, 0, &mut max_depth)?;
        Ok(max_depth)
    }
    
    fn estimate_file_count(&mut self, path: &Path) -> Result<u64> {
        let mut file_count = 0;
        
        // Sample a few directories to estimate total count
        let sample_entries = self.optimized_read_dir(path)?;
        let sample_files = sample_entries.iter().filter(|e| e.path().is_file()).count();
        let sample_dirs = sample_entries.iter().filter(|e| e.path().is_dir()).count();
        
        if sample_dirs > 0 {
            // Rough estimate based on sampling
            file_count = (sample_files * (sample_dirs + 1) * 10) as u64;
        } else {
            file_count = sample_files as u64;
        }
        
        Ok(file_count)
    }
    
    fn calculate_optimal_batch_size(&self) -> usize {
        match self.capabilities.fs_type {
            FileSystemType::NTFS => 50,   // Good balance for NTFS
            FileSystemType::FAT32 => 20,  // Smaller batches for FAT32
            FileSystemType::ExFAT => 100, // Larger batches for ExFAT
            FileSystemType::ReFS => 75,   // Medium batches for ReFS
            _ => 30,
        }
    }
    
    pub fn parallel_directory_scan(&mut self, path: &Path, thread_count: usize) -> Result<Vec<PathBuf>> {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let results = Arc::new(Mutex::new(Vec::new()));
        let path = Arc::new(path.to_path_buf());
        
        let mut handles = Vec::new();
        
        for i in 0..thread_count {
            let results = Arc::clone(&results);
            let path = Arc::clone(&path);
            
            let handle = thread::spawn(move || {
                // Each thread processes a subset of directories
                // This is a simplified approach - real implementation would be more sophisticated
                let mut local_results = Vec::new();
                
                // Simulate work
                thread::sleep(Duration::from_millis(10));
                local_results.push(path.as_ref().clone());
                
                if let Ok(mut global_results) = results.lock() {
                    global_results.extend(local_results);
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().map_err(|_| anyhow::anyhow!("Thread panicked"))?;
        }
        
        let results = results.lock().unwrap();
        Ok(results.clone())
    }
    
    pub fn cleanup_cache(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.cache.retain(|_, cached| {
            now.duration_since(cached.cached_at) < max_age
        });
    }
}

#[derive(Debug, Clone)]
pub struct IndexingOptimization {
    pub directory_depth: u32,
    pub estimated_file_count: u64,
    pub recommended_batch_size: usize,
    pub recommendations: Vec<String>,
    pub analysis_time: Duration,
}

impl IndexingOptimization {
    pub fn new() -> Self {
        Self {
            directory_depth: 0,
            estimated_file_count: 0,
            recommended_batch_size: 30,
            recommendations: Vec::new(),
            analysis_time: Duration::new(0, 0),
        }
    }
}
```

### 2. Add performance monitoring and profiling
Add these monitoring capabilities:
```rust
pub struct FileSystemPerformanceMonitor {
    operation_times: HashMap<String, Vec<Duration>>,
    start_times: HashMap<String, Instant>,
    total_operations: u64,
    total_time: Duration,
}

impl FileSystemPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            start_times: HashMap::new(),
            total_operations: 0,
            total_time: Duration::new(0, 0),
        }
    }
    
    pub fn start_operation(&mut self, operation: &str) {
        self.start_times.insert(operation.to_string(), Instant::now());
    }
    
    pub fn end_operation(&mut self, operation: &str) {
        if let Some(start_time) = self.start_times.remove(operation) {
            let duration = start_time.elapsed();
            
            self.operation_times
                .entry(operation.to_string())  
                .or_insert_with(Vec::new)
                .push(duration);
            
            self.total_operations += 1;
            self.total_time += duration;
        }
    }
    
    pub fn get_average_time(&self, operation: &str) -> Option<Duration> {
        if let Some(times) = self.operation_times.get(operation) {
            if !times.is_empty() {
                let total: Duration = times.iter().sum();
                Some(total / times.len() as u32)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    pub fn get_performance_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("File System Performance Report\n");
        report.push_str("==============================\n");
        report.push_str(&format!("Total Operations: {}\n", self.total_operations));
        report.push_str(&format!("Total Time: {:?}\n", self.total_time));
        
        if self.total_operations > 0 {
            report.push_str(&format!(
                "Average Time per Operation: {:?}\n",
                self.total_time / self.total_operations as u32
            ));
        }
        
        report.push_str("\nOperation Breakdown:\n");
        report.push_str("--------------------\n");
        
        for (operation, times) in &self.operation_times {
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let min_time = times.iter().min().unwrap_or(&Duration::new(0, 0));
            let max_time = times.iter().max().unwrap_or(&Duration::new(0, 0));
            
            report.push_str(&format!(
                "{}: {} ops, avg: {:?}, min: {:?}, max: {:?}\n",
                operation, times.len(), avg_time, min_time, max_time
            ));
        }
        
        report
    }
}

// Integration with existing WindowsPathHandler
impl WindowsPathHandler {
    pub fn with_filesystem_optimizer(mut self, path: &Path) -> Result<Self> {
        // This would integrate the filesystem optimizer
        // For now, return self as we'd need to modify the struct
        Ok(self)
    }
    
    pub fn get_filesystem_recommendations(&self, path: &Path) -> Result<Vec<String>> {
        let mut optimizer = FileSystemOptimizer::new(path)?;
        let optimization = optimizer.optimize_for_indexing(path)?;
        Ok(optimization.recommendations)
    }
    
    pub fn benchmark_filesystem_operations(&self, path: &Path) -> Result<String> {
        let mut monitor = FileSystemPerformanceMonitor::new();
        
        // Benchmark various operations
        monitor.start_operation("metadata");
        let _ = std::fs::metadata(path);
        monitor.end_operation("metadata");
        
        monitor.start_operation("read_dir");
        if path.is_dir() {
            let _ = std::fs::read_dir(path);
        }
        monitor.end_operation("read_dir");
        
        monitor.start_operation("canonicalize");
        let _ = path.canonicalize();
        monitor.end_operation("canonicalize");
        
        Ok(monitor.get_performance_report())
    }
}
```

### 3. Add comprehensive tests for filesystem optimizations
Add these test modules:
```rust
#[cfg(test)]
mod filesystem_optimization_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_filesystem_detection() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let capabilities = FileSystemCapabilities::detect_for_path(temp_dir.path())?;
        
        println!("Detected filesystem: {:?}", capabilities.fs_type);
        println!("Capabilities: {:#?}", capabilities);
        
        // Should detect some filesystem type
        assert_ne!(capabilities.fs_type, FileSystemType::Unknown("Generic".to_string()));
        
        Ok(())
    }
    
    #[test]
    fn test_filesystem_optimizer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        let capabilities = optimizer.get_capabilities();
        assert!(capabilities.unicode_filenames);
        
        println!("Optimizer created for filesystem: {:?}", capabilities.fs_type);
        
        Ok(())
    }
    
    #[test]
    fn test_metadata_caching() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        // Create a test file
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, "test content")?;
        
        // First access should be a cache miss
        let _metadata1 = optimizer.optimized_metadata(&test_file)?;
        assert_eq!(optimizer.get_stats().cache_misses, 1);
        assert_eq!(optimizer.get_stats().cache_hits, 0);
        
        // Second access should be a cache hit (within 30 seconds)
        let _metadata2 = optimizer.optimized_metadata(&test_file)?;
        assert_eq!(optimizer.get_stats().cache_hits, 1);
        
        Ok(())
    }
    
    #[test]
    fn test_directory_optimization() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        // Create test directory structure
        std::fs::create_dir_all(temp_dir.path().join("subdir1"))?;
        std::fs::create_dir_all(temp_dir.path().join("subdir2"))?;
        std::fs::write(temp_dir.path().join("file1.txt"), "content1")?;
        std::fs::write(temp_dir.path().join("file2.txt"), "content2")?;
        
        let entries = optimizer.optimized_read_dir(temp_dir.path())?;
        
        // Should have found all entries
        assert_eq!(entries.len(), 4);
        
        // Should have updated statistics
        assert_eq!(optimizer.get_stats().directory_scans, 1);
        
        Ok(())
    }
    
    #[test]
    fn test_indexing_optimization_analysis() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        // Create a nested directory structure
        std::fs::create_dir_all(temp_dir.path().join("level1/level2/level3"))?;
        std::fs::write(temp_dir.path().join("level1/file1.txt"), "content")?;
        std::fs::write(temp_dir.path().join("level1/level2/file2.txt"), "content")?;
        
        let optimization = optimizer.optimize_for_indexing(temp_dir.path())?;
        
        println!("Indexing optimization analysis:");
        println!("  Directory depth: {}", optimization.directory_depth);
        println!("  Estimated file count: {}", optimization.estimated_file_count);
        println!("  Recommended batch size: {}", optimization.recommended_batch_size);
        println!("  Recommendations: {:?}", optimization.recommendations);
        println!("  Analysis time: {:?}", optimization.analysis_time);
        
        assert!(optimization.directory_depth >= 3); // Should detect depth
        assert!(!optimization.recommendations.is_empty()); // Should have recommendations
        
        Ok(())
    }
    
    #[test]
    fn test_performance_monitoring() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut monitor = FileSystemPerformanceMonitor::new();
        
        // Simulate some operations
        for i in 0..10 {
            monitor.start_operation("test_op");
            std::thread::sleep(Duration::from_millis(1));
            monitor.end_operation("test_op");
        }
        
        let avg_time = monitor.get_average_time("test_op");
        assert!(avg_time.is_some());
        
        let report = monitor.get_performance_report();
        println!("Performance report:\n{}", report);
        
        assert!(report.contains("test_op"));
        assert!(report.contains("10 ops"));
        
        Ok(())
    }
    
    #[test]
    fn test_cache_cleanup() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        // Create test file and cache metadata
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, "content")?;
        let _metadata = optimizer.optimized_metadata(&test_file)?;
        
        // Verify cache has entry
        assert_eq!(optimizer.cache.len(), 1);
        
        // Clean cache with very short max age
        optimizer.cleanup_cache(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(2));
        optimizer.cleanup_cache(Duration::from_millis(1));
        
        // Cache should be empty after cleanup
        assert_eq!(optimizer.cache.len(), 0);
        
        Ok(())
    }
    
    #[test]
    #[ignore] // Expensive test - run with --ignored
    fn test_parallel_directory_scan() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        // Create multiple directories for parallel processing
        for i in 0..10 {
            std::fs::create_dir_all(temp_dir.path().join(format!("dir_{}", i)))?;
            std::fs::write(temp_dir.path().join(format!("dir_{}/file.txt", i)), "content")?;
        }
        
        let start = Instant::now();
        let results = optimizer.parallel_directory_scan(temp_dir.path(), 4)?;
        let parallel_time = start.elapsed();
        
        println!("Parallel scan completed in {:?}", parallel_time);
        println!("Found {} paths", results.len());
        
        assert!(!results.is_empty());
        
        Ok(())
    }
    
    #[test]
    fn test_filesystem_specific_optimizations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let optimizer = FileSystemOptimizer::new(temp_dir.path())?;
        
        let capabilities = optimizer.get_capabilities();
        
        match capabilities.fs_type {
            FileSystemType::NTFS => {
                assert!(capabilities.supports_compression);
                assert!(capabilities.supports_hard_links);
                assert!(capabilities.supports_alternate_streams);
                println!("NTFS optimizations available");
            }
            FileSystemType::FAT32 => {
                assert!(!capabilities.supports_compression);
                assert!(!capabilities.supports_hard_links);
                assert_eq!(capabilities.max_file_size, 4_294_967_295);
                println!("FAT32 limitations detected");
            }
            FileSystemType::ExFAT => {  
                assert!(!capabilities.supports_hard_links);
                assert!(capabilities.max_file_size > 4_294_967_295);
                println!("ExFAT optimizations available");
            }
            FileSystemType::ReFS => {
                assert!(capabilities.supports_sparse_files);
                assert!(!capabilities.supports_short_names);
                println!("ReFS optimizations available");
            }
            _ => {
                println!("Generic filesystem handling");
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_integration_with_path_handler() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = WindowsPathHandler::new();
        
        // Test filesystem recommendations
        let recommendations = handler.get_filesystem_recommendations(temp_dir.path())?;
        assert!(!recommendations.is_empty());
        
        println!("Filesystem recommendations:");
        for rec in &recommendations {
            println!("  - {}", rec);
        }
        
        // Test performance benchmarking
        let benchmark_report = handler.benchmark_filesystem_operations(temp_dir.path())?;
        assert!(benchmark_report.contains("Performance Report"));
        
        println!("Benchmark report:\n{}", benchmark_report);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Comprehensive filesystem detection and capability assessment
- [ ] Efficient metadata caching with configurable expiration
- [ ] Filesystem-specific optimizations for NTFS, FAT32, ExFAT, and ReFS
- [ ] Performance monitoring and profiling capabilities
- [ ] Parallel directory scanning with thread safety
- [ ] Batch processing optimizations for large directories
- [ ] Cache management with automatic cleanup
- [ ] Integration with existing path validation system
- [ ] Comprehensive testing across different filesystem types
- [ ] Performance benchmarks demonstrating optimization benefits
- [ ] All tests pass with measurable performance improvements
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- NTFS supports advanced features like compression, encryption, and alternate data streams
- FAT32 has significant limitations including 4GB file size limit
- ExFAT is optimized for large files and removable media
- ReFS provides built-in integrity checking and supports very large files
- Metadata caching can significantly improve performance for repeated operations
- Parallel processing requires careful synchronization to avoid race conditions
- Different filesystems have different optimal batch sizes and access patterns
- Performance monitoring helps identify bottlenecks in the indexing process