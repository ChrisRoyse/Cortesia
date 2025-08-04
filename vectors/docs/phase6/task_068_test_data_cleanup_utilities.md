# Task 068: Create Test Data Cleanup Utilities

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates automatic cleanup functions, resource leak prevention, temporary file management, graceful shutdown handling, and error recovery cleanup for maintaining a clean test environment and preventing resource accumulation.

## Project Structure
```
src/cleanup/
├── mod.rs              <- Cleanup module entry point
├── automatic.rs        <- Automatic cleanup functions
├── resources.rs        <- Resource leak prevention
├── temporary.rs        <- Temporary file management
├── shutdown.rs         <- Graceful shutdown handling
└── recovery.rs         <- Error recovery cleanup
```

## Task Description
Create comprehensive test data cleanup utilities that automatically manage temporary files, prevent resource leaks, handle graceful shutdown scenarios, implement error recovery cleanup, and maintain system health throughout the validation process.

## Requirements
1. Implement automatic cleanup functions for test data and temporary files
2. Create resource leak prevention mechanisms
3. Build temporary file management with automatic lifecycle handling
4. Implement graceful shutdown handling with cleanup guarantees
5. Provide error recovery cleanup for failed operations

## Expected File Content/Code Structure

### Main Cleanup Module (`src/cleanup/mod.rs`)
```rust
//! Comprehensive test data cleanup utilities for LLMKG validation
//! 
//! Provides automatic cleanup, resource leak prevention, temporary file management,
//! graceful shutdown handling, and error recovery cleanup.

pub mod automatic;
pub mod resources;
pub mod temporary;
pub mod shutdown;
pub mod recovery;

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::signal;
use tracing::{info, warn, error, debug};

pub use automatic::*;
pub use resources::*;
pub use temporary::*;
pub use shutdown::*;
pub use recovery::*;

/// Global cleanup manager for coordinating all cleanup activities
pub struct CleanupManager {
    temporary_files: Arc<Mutex<TemporaryFileManager>>,
    resource_tracker: Arc<Mutex<ResourceTracker>>,
    shutdown_handler: Arc<Mutex<ShutdownHandler>>,
    cleanup_schedule: Arc<Mutex<CleanupSchedule>>,
    is_shutting_down: Arc<Mutex<bool>>,
}

impl CleanupManager {
    /// Create a new cleanup manager
    pub fn new() -> Self {
        Self {
            temporary_files: Arc::new(Mutex::new(TemporaryFileManager::new())),
            resource_tracker: Arc::new(Mutex::new(ResourceTracker::new())),
            shutdown_handler: Arc::new(Mutex::new(ShutdownHandler::new())),
            cleanup_schedule: Arc::new(Mutex::new(CleanupSchedule::new())),
            is_shutting_down: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Initialize the cleanup manager with signal handlers
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing cleanup manager");
        
        // Set up signal handlers for graceful shutdown
        self.setup_signal_handlers().await?;
        
        // Start background cleanup tasks
        self.start_background_cleanup().await?;
        
        // Register shutdown hooks
        self.register_shutdown_hooks()?;
        
        info!("Cleanup manager initialized successfully");
        Ok(())
    }
    
    /// Set up signal handlers for graceful shutdown
    async fn setup_signal_handlers(&self) -> Result<()> {
        let is_shutting_down = Arc::clone(&self.is_shutting_down);
        let cleanup_manager = self.clone_for_signal_handler();
        
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                    .expect("Failed to install SIGTERM handler");
                let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
                    .expect("Failed to install SIGINT handler");
                
                tokio::select! {
                    _ = sigterm.recv() => {
                        info!("Received SIGTERM, initiating graceful shutdown");
                    }
                    _ = sigint.recv() => {
                        info!("Received SIGINT, initiating graceful shutdown");
                    }
                }
            }
            
            #[cfg(windows)]
            {
                let _ = signal::ctrl_c().await;
                info!("Received Ctrl+C, initiating graceful shutdown");
            }
            
            // Mark as shutting down
            *is_shutting_down.lock().unwrap() = true;
            
            // Perform cleanup
            if let Err(e) = cleanup_manager.perform_shutdown_cleanup().await {
                error!("Error during shutdown cleanup: {}", e);
            }
        });
        
        Ok(())
    }
    
    /// Start background cleanup tasks
    async fn start_background_cleanup(&self) -> Result<()> {
        let temporary_files = Arc::clone(&self.temporary_files);
        let resource_tracker = Arc::clone(&self.resource_tracker);
        let is_shutting_down = Arc::clone(&self.is_shutting_down);
        
        // Periodic cleanup task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Every minute
            
            loop {
                interval.tick().await;
                
                if *is_shutting_down.lock().unwrap() {
                    break;
                }
                
                // Clean up expired temporary files
                if let Ok(mut temp_manager) = temporary_files.lock() {
                    if let Err(e) = temp_manager.cleanup_expired_files() {
                        warn!("Error cleaning up expired temporary files: {}", e);
                    }
                }
                
                // Check for resource leaks
                if let Ok(mut tracker) = resource_tracker.lock() {
                    if let Err(e) = tracker.check_for_leaks() {
                        warn!("Error checking for resource leaks: {}", e);
                    }
                }
                
                debug!("Background cleanup completed");
            }
        });
        
        Ok(())
    }
    
    /// Register shutdown hooks
    fn register_shutdown_hooks(&self) -> Result<()> {
        // Register with the process exit handler
        let cleanup_manager = self.clone_for_shutdown();
        
        // Use a global static to ensure cleanup happens on process exit
        std::panic::set_hook(Box::new(move |panic_info| {
            error!("Panic detected: {:?}", panic_info);
            
            // Perform emergency cleanup
            if let Err(e) = cleanup_manager.perform_emergency_cleanup() {
                eprintln!("Emergency cleanup failed: {}", e);
            }
        }));
        
        Ok(())
    }
    
    /// Clone manager for signal handler (simplified version)
    fn clone_for_signal_handler(&self) -> SimpleCleanupManager {
        SimpleCleanupManager {
            temporary_files: Arc::clone(&self.temporary_files),
            resource_tracker: Arc::clone(&self.resource_tracker),
        }
    }
    
    /// Clone manager for shutdown (simplified version)
    fn clone_for_shutdown(&self) -> SimpleCleanupManager {
        SimpleCleanupManager {
            temporary_files: Arc::clone(&self.temporary_files),
            resource_tracker: Arc::clone(&self.resource_tracker),
        }
    }
    
    /// Perform shutdown cleanup
    pub async fn perform_shutdown_cleanup(&self) -> Result<()> {
        info!("Performing shutdown cleanup");
        
        // Clean up temporary files
        if let Ok(mut temp_manager) = self.temporary_files.lock() {
            temp_manager.cleanup_all_files()?;
        }
        
        // Clean up resources
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.cleanup_all_resources()?;
        }
        
        // Run shutdown handlers
        if let Ok(mut handler) = self.shutdown_handler.lock() {
            handler.run_shutdown_hooks().await?;
        }
        
        info!("Shutdown cleanup completed");
        Ok(())
    }
    
    /// Perform emergency cleanup (synchronous for panic handler)
    fn perform_emergency_cleanup(&self) -> Result<()> {
        warn!("Performing emergency cleanup");
        
        // Clean up temporary files
        if let Ok(mut temp_manager) = self.temporary_files.lock() {
            temp_manager.cleanup_all_files()?;
        }
        
        // Clean up resources
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.cleanup_all_resources()?;
        }
        
        warn!("Emergency cleanup completed");
        Ok(())
    }
    
    /// Register a temporary file for cleanup
    pub fn register_temporary_file(&self, path: PathBuf, lifetime: Duration) -> Result<()> {
        if let Ok(mut temp_manager) = self.temporary_files.lock() {
            temp_manager.register_file(path, lifetime)
        } else {
            Err(anyhow::anyhow!("Failed to acquire temporary file manager lock"))
        }
    }
    
    /// Register a resource for tracking
    pub fn register_resource(&self, resource: Box<dyn CleanupResource>) -> Result<()> {
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.register_resource(resource)
        } else {
            Err(anyhow::anyhow!("Failed to acquire resource tracker lock"))
        }
    }
    
    /// Register a shutdown hook
    pub fn register_shutdown_hook(&self, hook: Box<dyn ShutdownHook>) -> Result<()> {
        if let Ok(mut handler) = self.shutdown_handler.lock() {
            handler.register_hook(hook)
        } else {
            Err(anyhow::anyhow!("Failed to acquire shutdown handler lock"))
        }
    }
    
    /// Check system health
    pub fn check_system_health(&self) -> SystemHealthReport {
        let temp_file_count = self.temporary_files.lock()
            .map(|tm| tm.get_file_count())
            .unwrap_or(0);
        
        let resource_count = self.resource_tracker.lock()
            .map(|rt| rt.get_resource_count())
            .unwrap_or(0);
        
        let memory_usage = get_current_memory_usage();
        let disk_usage = get_temp_directory_usage();
        
        SystemHealthReport {
            temporary_file_count: temp_file_count,
            tracked_resource_count: resource_count,
            memory_usage_mb: memory_usage.unwrap_or(0),
            temp_disk_usage_mb: disk_usage.unwrap_or(0),
            is_healthy: temp_file_count < 10000 && resource_count < 1000,
        }
    }
}

/// Simplified cleanup manager for signal handlers
struct SimpleCleanupManager {
    temporary_files: Arc<Mutex<TemporaryFileManager>>,
    resource_tracker: Arc<Mutex<ResourceTracker>>,
}

impl SimpleCleanupManager {
    async fn perform_shutdown_cleanup(&self) -> Result<()> {
        // Clean up temporary files
        if let Ok(mut temp_manager) = self.temporary_files.lock() {
            temp_manager.cleanup_all_files()?;
        }
        
        // Clean up resources
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.cleanup_all_resources()?;
        }
        
        Ok(())
    }
    
    fn perform_emergency_cleanup(&self) -> Result<()> {
        // Clean up temporary files
        if let Ok(mut temp_manager) = self.temporary_files.lock() {
            temp_manager.cleanup_all_files()?;
        }
        
        // Clean up resources
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.cleanup_all_resources()?;
        }
        
        Ok(())
    }
}

/// System health report
#[derive(Debug, Clone)]
pub struct SystemHealthReport {
    pub temporary_file_count: usize,
    pub tracked_resource_count: usize,
    pub memory_usage_mb: u64,
    pub temp_disk_usage_mb: u64,
    pub is_healthy: bool,
}

impl SystemHealthReport {
    pub fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if self.temporary_file_count > 1000 {
            warnings.push(format!("High temporary file count: {}", self.temporary_file_count));
        }
        
        if self.tracked_resource_count > 100 {
            warnings.push(format!("High tracked resource count: {}", self.tracked_resource_count));
        }
        
        if self.memory_usage_mb > 2048 {
            warnings.push(format!("High memory usage: {} MB", self.memory_usage_mb));
        }
        
        if self.temp_disk_usage_mb > 10240 {
            warnings.push(format!("High temp disk usage: {} MB", self.temp_disk_usage_mb));
        }
        
        warnings
    }
}

/// Cleanup schedule for managing periodic cleanup tasks
struct CleanupSchedule {
    last_full_cleanup: Instant,
    last_temp_cleanup: Instant,
    last_resource_check: Instant,
}

impl CleanupSchedule {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_full_cleanup: now,
            last_temp_cleanup: now,
            last_resource_check: now,
        }
    }
    
    fn should_run_full_cleanup(&mut self) -> bool {
        let should_run = self.last_full_cleanup.elapsed() > Duration::from_secs(3600); // Every hour
        if should_run {
            self.last_full_cleanup = Instant::now();
        }
        should_run
    }
    
    fn should_run_temp_cleanup(&mut self) -> bool {
        let should_run = self.last_temp_cleanup.elapsed() > Duration::from_secs(300); // Every 5 minutes
        if should_run {
            self.last_temp_cleanup = Instant::now();
        }
        should_run
    }
    
    fn should_run_resource_check(&mut self) -> bool {
        let should_run = self.last_resource_check.elapsed() > Duration::from_secs(60); // Every minute
        if should_run {
            self.last_resource_check = Instant::now();
        }
        should_run
    }
}

/// Get current memory usage (platform-specific implementation needed)
fn get_current_memory_usage() -> Option<u64> {
    // Placeholder implementation
    // Real implementation would use platform-specific APIs
    Some(0)
}

/// Get temporary directory disk usage
fn get_temp_directory_usage() -> Option<u64> {
    // Placeholder implementation
    // Real implementation would calculate actual disk usage
    Some(0)
}

/// Global cleanup manager instance
static CLEANUP_MANAGER: std::sync::OnceLock<CleanupManager> = std::sync::OnceLock::new();

/// Initialize global cleanup manager
pub async fn initialize_global_cleanup_manager() -> Result<()> {
    let manager = CleanupManager::new();
    manager.initialize().await?;
    
    CLEANUP_MANAGER.set(manager)
        .map_err(|_| anyhow::anyhow!("Cleanup manager already initialized"))?;
    
    Ok(())
}

/// Get global cleanup manager
pub fn get_global_cleanup_manager() -> Option<&'static CleanupManager> {
    CLEANUP_MANAGER.get()
}

/// Cleanup utility functions for common scenarios
pub fn cleanup_test_directory<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if path.exists() && path.is_dir() {
        std::fs::remove_dir_all(path)?;
        info!("Cleaned up test directory: {}", path.display());
    }
    Ok(())
}

/// Cleanup temporary files matching a pattern
pub fn cleanup_files_matching_pattern<P: AsRef<std::path::Path>>(
    directory: P, 
    pattern: &str
) -> Result<usize> {
    let directory = directory.as_ref();
    let mut cleaned_count = 0;
    
    if !directory.exists() {
        return Ok(0);
    }
    
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if filename.to_string_lossy().contains(pattern) {
                    match std::fs::remove_file(&path) {
                        Ok(_) => {
                            cleaned_count += 1;
                            debug!("Removed file matching pattern '{}': {}", pattern, path.display());
                        }
                        Err(e) => {
                            warn!("Failed to remove file {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }
    
    info!("Cleaned up {} files matching pattern '{}'", cleaned_count, pattern);
    Ok(cleaned_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_cleanup_manager_initialization() -> Result<()> {
        let manager = CleanupManager::new();
        manager.initialize().await?;
        
        let health = manager.check_system_health();
        assert!(health.temporary_file_count == 0);
        
        Ok(())
    }
    
    #[test]
    fn test_cleanup_test_directory() -> Result<()> {
        let temp_dir = tempdir()?;
        let test_dir = temp_dir.path().join("test_cleanup");
        std::fs::create_dir(&test_dir)?;
        
        assert!(test_dir.exists());
        cleanup_test_directory(&test_dir)?;
        assert!(!test_dir.exists());
        
        Ok(())
    }
    
    #[test]
    fn test_cleanup_files_matching_pattern() -> Result<()> {
        let temp_dir = tempdir()?;
        
        // Create test files
        std::fs::write(temp_dir.path().join("test_001.tmp"), "test")?;
        std::fs::write(temp_dir.path().join("test_002.tmp"), "test")?;
        std::fs::write(temp_dir.path().join("other.txt"), "other")?;
        
        let cleaned_count = cleanup_files_matching_pattern(temp_dir.path(), ".tmp")?;
        assert_eq!(cleaned_count, 2);
        
        // Check that only .tmp files were removed
        assert!(!temp_dir.path().join("test_001.tmp").exists());
        assert!(!temp_dir.path().join("test_002.tmp").exists());
        assert!(temp_dir.path().join("other.txt").exists());
        
        Ok(())
    }
}
```

### Automatic Cleanup Functions (`src/cleanup/automatic.rs`)
```rust
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::fs;
use tracing::{info, warn, debug};

/// Automatic cleanup configuration
#[derive(Debug, Clone)]
pub struct AutomaticCleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,
    
    /// Cleanup interval in seconds
    pub interval_secs: u64,
    
    /// Maximum age of temporary files before cleanup (seconds)
    pub max_file_age_secs: u64,
    
    /// Maximum total size of temporary files (bytes)
    pub max_total_size_bytes: u64,
    
    /// Directories to monitor for cleanup
    pub monitored_directories: Vec<PathBuf>,
    
    /// File patterns to clean up
    pub cleanup_patterns: Vec<String>,
    
    /// Preserve files matching these patterns
    pub preserve_patterns: Vec<String>,
}

impl Default for AutomaticCleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 300, // 5 minutes
            max_file_age_secs: 3600, // 1 hour
            max_total_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            monitored_directories: vec![
                std::env::temp_dir(),
                PathBuf::from("./tests/temp"),
                PathBuf::from("./tests/output/artifacts"),
            ],
            cleanup_patterns: vec![
                "*.tmp".to_string(),
                "*.temp".to_string(),
                "validation_*".to_string(),
                "test_index_*".to_string(),
                "benchmark_*".to_string(),
            ],
            preserve_patterns: vec![
                "*.keep".to_string(),
                "ground_truth*".to_string(),
                "*.gitkeep".to_string(),
            ],
        }
    }
}

/// Automatic cleanup engine
pub struct AutomaticCleanup {
    config: AutomaticCleanupConfig,
}

impl AutomaticCleanup {
    /// Create a new automatic cleanup instance
    pub fn new(config: AutomaticCleanupConfig) -> Self {
        Self { config }
    }
    
    /// Run automatic cleanup based on configuration
    pub fn run_cleanup(&self) -> Result<CleanupResult> {
        if !self.config.enabled {
            return Ok(CleanupResult::default());
        }
        
        info!("Starting automatic cleanup");
        let mut total_result = CleanupResult::default();
        
        for directory in &self.config.monitored_directories {
            if directory.exists() {
                let result = self.cleanup_directory(directory)?;
                total_result.merge(result);
            } else {
                debug!("Skipping non-existent directory: {}", directory.display());
            }
        }
        
        info!("Automatic cleanup completed: {:?}", total_result);
        Ok(total_result)
    }
    
    /// Clean up a specific directory
    fn cleanup_directory(&self, directory: &Path) -> Result<CleanupResult> {
        debug!("Cleaning up directory: {}", directory.display());
        let mut result = CleanupResult::default();
        
        let entries = fs::read_dir(directory)?;
        let cutoff_time = SystemTime::now() - Duration::from_secs(self.config.max_file_age_secs);
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                let file_result = self.process_file(&path, cutoff_time)?;
                result.merge(file_result);
            } else if path.is_dir() {
                // Recursively clean subdirectories
                let dir_result = self.cleanup_directory(&path)?;
                result.merge(dir_result);
                
                // Remove empty directories
                if self.is_directory_empty(&path)? {
                    match fs::remove_dir(&path) {
                        Ok(_) => {
                            result.directories_removed += 1;
                            debug!("Removed empty directory: {}", path.display());
                        }
                        Err(e) => {
                            warn!("Failed to remove empty directory {}: {}", path.display(), e);
                            result.errors.push(format!("Failed to remove directory {}: {}", path.display(), e));
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Process an individual file for cleanup
    fn process_file(&self, path: &Path, cutoff_time: SystemTime) -> Result<CleanupResult> {
        let mut result = CleanupResult::default();
        
        // Check if file should be preserved
        if self.should_preserve_file(path) {
            debug!("Preserving file: {}", path.display());
            return Ok(result);
        }
        
        // Check if file matches cleanup patterns
        if !self.matches_cleanup_pattern(path) {
            return Ok(result);
        }
        
        let metadata = fs::metadata(path)?;
        let file_size = metadata.len();
        
        // Check file age
        if let Ok(modified) = metadata.modified() {
            if modified < cutoff_time {
                match fs::remove_file(path) {
                    Ok(_) => {
                        result.files_removed += 1;
                        result.bytes_freed += file_size;
                        debug!("Removed old file: {} ({} bytes)", path.display(), file_size);
                    }
                    Err(e) => {
                        warn!("Failed to remove file {}: {}", path.display(), e);
                        result.errors.push(format!("Failed to remove file {}: {}", path.display(), e));
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Check if a file should be preserved
    fn should_preserve_file(&self, path: &Path) -> bool {
        if let Some(filename) = path.file_name() {
            let filename_str = filename.to_string_lossy();
            for pattern in &self.config.preserve_patterns {
                if glob_match(pattern, &filename_str) {
                    return true;
                }
            }
        }
        false
    }
    
    /// Check if a file matches cleanup patterns
    fn matches_cleanup_pattern(&self, path: &Path) -> bool {
        if self.config.cleanup_patterns.is_empty() {
            return true; // Clean all files if no patterns specified
        }
        
        if let Some(filename) = path.file_name() {
            let filename_str = filename.to_string_lossy();
            for pattern in &self.config.cleanup_patterns {
                if glob_match(pattern, &filename_str) {
                    return true;
                }
            }
        }
        false
    }
    
    /// Check if a directory is empty
    fn is_directory_empty(&self, path: &Path) -> Result<bool> {
        let mut entries = fs::read_dir(path)?;
        Ok(entries.next().is_none())
    }
    
    /// Clean up files based on total size limit
    pub fn cleanup_by_size_limit(&self, directory: &Path) -> Result<CleanupResult> {
        debug!("Cleaning up directory by size limit: {}", directory.display());
        let mut result = CleanupResult::default();
        
        // Collect all files with their sizes and modification times
        let mut files = Vec::new();
        self.collect_files_recursive(directory, &mut files)?;
        
        // Sort by modification time (oldest first)
        files.sort_by_key(|(_, _, modified)| *modified);
        
        let mut total_size = files.iter().map(|(_, size, _)| *size).sum::<u64>();
        
        // Remove oldest files until under size limit
        for (path, file_size, _) in files {
            if total_size <= self.config.max_total_size_bytes {
                break;
            }
            
            if !self.should_preserve_file(&path) && self.matches_cleanup_pattern(&path) {
                match fs::remove_file(&path) {
                    Ok(_) => {
                        result.files_removed += 1;
                        result.bytes_freed += file_size;
                        total_size -= file_size;
                        debug!("Removed file for size limit: {} ({} bytes)", path.display(), file_size);
                    }
                    Err(e) => {
                        warn!("Failed to remove file {}: {}", path.display(), e);
                        result.errors.push(format!("Failed to remove file {}: {}", path.display(), e));
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Collect files recursively
    fn collect_files_recursive(
        &self, 
        directory: &Path, 
        files: &mut Vec<(PathBuf, u64, SystemTime)>
    ) -> Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        files.push((path, metadata.len(), modified));
                    }
                }
            } else if path.is_dir() {
                self.collect_files_recursive(&path, files)?;
            }
        }
        Ok(())
    }
    
    /// Force cleanup of all matching files regardless of age
    pub fn force_cleanup(&self, directory: &Path) -> Result<CleanupResult> {
        info!("Force cleaning directory: {}", directory.display());
        let mut result = CleanupResult::default();
        
        if !directory.exists() {
            return Ok(result);
        }
        
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if !self.should_preserve_file(&path) && self.matches_cleanup_pattern(&path) {
                    if let Ok(metadata) = fs::metadata(&path) {
                        let file_size = metadata.len();
                        
                        match fs::remove_file(&path) {
                            Ok(_) => {
                                result.files_removed += 1;
                                result.bytes_freed += file_size;
                                debug!("Force removed file: {}", path.display());
                            }
                            Err(e) => {
                                warn!("Failed to force remove file {}: {}", path.display(), e);
                                result.errors.push(format!("Failed to remove file {}: {}", path.display(), e));
                            }
                        }
                    }
                }
            } else if path.is_dir() {
                let dir_result = self.force_cleanup(&path)?;
                result.merge(dir_result);
                
                if self.is_directory_empty(&path)? {
                    match fs::remove_dir(&path) {
                        Ok(_) => {
                            result.directories_removed += 1;
                            debug!("Force removed empty directory: {}", path.display());
                        }
                        Err(e) => {
                            warn!("Failed to force remove directory {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// Cleanup operation result
#[derive(Debug, Default, Clone)]
pub struct CleanupResult {
    pub files_removed: usize,
    pub directories_removed: usize,
    pub bytes_freed: u64,
    pub errors: Vec<String>,
}

impl CleanupResult {
    /// Merge another cleanup result into this one
    pub fn merge(&mut self, other: CleanupResult) {
        self.files_removed += other.files_removed;
        self.directories_removed += other.directories_removed;
        self.bytes_freed += other.bytes_freed;
        self.errors.extend(other.errors);
    }
    
    /// Check if cleanup was successful (no errors)
    pub fn is_successful(&self) -> bool {
        self.errors.is_empty()
    }
    
    /// Get total items removed
    pub fn total_items_removed(&self) -> usize {
        self.files_removed + self.directories_removed
    }
    
    /// Get human-readable size freed
    pub fn size_freed_human(&self) -> String {
        if self.bytes_freed >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", self.bytes_freed as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if self.bytes_freed >= 1024 * 1024 {
            format!("{:.2} MB", self.bytes_freed as f64 / (1024.0 * 1024.0))
        } else if self.bytes_freed >= 1024 {
            format!("{:.2} KB", self.bytes_freed as f64 / 1024.0)
        } else {
            format!("{} bytes", self.bytes_freed)
        }
    }
}

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
    // Simplified glob matching - in production, use a proper glob library
    if pattern == "*" {
        return true;
    }
    
    if pattern.starts_with('*') && pattern.len() > 1 {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }
    
    if pattern.ends_with('*') && pattern.len() > 1 {
        let prefix = &pattern[..pattern.len() - 1];
        return text.starts_with(prefix);
    }
    
    pattern == text
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::thread;
    
    #[test]
    fn test_automatic_cleanup() -> Result<()> {
        let temp_dir = tempdir()?;
        let config = AutomaticCleanupConfig {
            enabled: true,
            max_file_age_secs: 1, // 1 second for testing
            monitored_directories: vec![temp_dir.path().to_path_buf()],
            cleanup_patterns: vec!["*.tmp".to_string()],
            preserve_patterns: vec!["*.keep".to_string()],
            ..Default::default()
        };
        
        // Create test files
        fs::write(temp_dir.path().join("old.tmp"), "test")?;
        fs::write(temp_dir.path().join("preserve.keep"), "keep this")?;
        fs::write(temp_dir.path().join("other.txt"), "other")?;
        
        // Wait for files to age
        thread::sleep(Duration::from_secs(2));
        
        let cleanup = AutomaticCleanup::new(config);
        let result = cleanup.run_cleanup()?;
        
        // Should remove old.tmp but preserve others
        assert_eq!(result.files_removed, 1);
        assert!(!temp_dir.path().join("old.tmp").exists());
        assert!(temp_dir.path().join("preserve.keep").exists());
        assert!(temp_dir.path().join("other.txt").exists()); // Different pattern
        
        Ok(())
    }
    
    #[test]
    fn test_force_cleanup() -> Result<()> {
        let temp_dir = tempdir()?;
        let config = AutomaticCleanupConfig {
            enabled: true,
            monitored_directories: vec![temp_dir.path().to_path_buf()],
            cleanup_patterns: vec!["*.tmp".to_string()],
            preserve_patterns: vec!["*.keep".to_string()],
            ..Default::default()
        };
        
        // Create test files
        fs::write(temp_dir.path().join("file1.tmp"), "test1")?;
        fs::write(temp_dir.path().join("file2.tmp"), "test2")?;
        fs::write(temp_dir.path().join("preserve.keep"), "keep this")?;
        
        let cleanup = AutomaticCleanup::new(config);
        let result = cleanup.force_cleanup(temp_dir.path())?;
        
        // Should remove both .tmp files but preserve .keep file
        assert_eq!(result.files_removed, 2);
        assert!(!temp_dir.path().join("file1.tmp").exists());
        assert!(!temp_dir.path().join("file2.tmp").exists());
        assert!(temp_dir.path().join("preserve.keep").exists());
        
        Ok(())
    }
    
    #[test]
    fn test_glob_match() {
        assert!(glob_match("*.tmp", "file.tmp"));
        assert!(glob_match("test_*", "test_123"));
        assert!(glob_match("*", "anything"));
        assert!(!glob_match("*.tmp", "file.txt"));
        assert!(!glob_match("test_*", "other_123"));
    }
    
    #[test]
    fn test_cleanup_result_merge() {
        let mut result1 = CleanupResult {
            files_removed: 5,
            directories_removed: 1,
            bytes_freed: 1000,
            errors: vec!["error1".to_string()],
        };
        
        let result2 = CleanupResult {
            files_removed: 3,
            directories_removed: 2,
            bytes_freed: 500,
            errors: vec!["error2".to_string()],
        };
        
        result1.merge(result2);
        
        assert_eq!(result1.files_removed, 8);
        assert_eq!(result1.directories_removed, 3);
        assert_eq!(result1.bytes_freed, 1500);
        assert_eq!(result1.errors.len(), 2);
    }
}
```

## Success Criteria
- Automatic cleanup functions remove expired temporary files correctly
- Resource leak prevention tracks and cleans up system resources
- Temporary file management handles lifecycle automatically with proper cleanup
- Graceful shutdown handling ensures all cleanup tasks complete before exit
- Error recovery cleanup handles failed operations by cleaning up partial state
- Signal handlers (SIGTERM, SIGINT, Ctrl+C) trigger proper cleanup sequences
- Background cleanup tasks run periodically without impacting performance
- Memory and disk usage monitoring prevents resource exhaustion
- Cleanup patterns and preservation rules work correctly
- Cross-platform compatibility for Windows and Unix systems

## Time Limit
10 minutes maximum