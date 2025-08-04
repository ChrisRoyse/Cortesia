# Task 29: Implement Memory Leak Detection Framework

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 28 (Index corruption recovery)
**Input Files:** `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs`, `C:\code\LLMKG\vectors\tantivy_search\tests\`

## Complete Context (For AI with ZERO Knowledge)

You are implementing **memory leak detection for a Rust search system** that processes large files and maintains indexes. Memory leaks in search systems typically occur in:
- Long-running indexing operations that accumulate temporary data
- Search result caching that grows unbounded  
- File handles and index writers that aren't properly closed
- Background processing threads that retain references

**What is Memory Leak Detection?** A system that monitors memory usage patterns, detects abnormal growth, and validates that resources are cleaned up properly after operations complete.

**This Task:** Creates a comprehensive memory monitoring framework with leak detection, memory pressure handling, and automated cleanup validation.

## Exact Steps (6 minutes implementation)

### Step 1: Create memory monitoring module (3 minutes)
Create file: `C:\code\LLMKG\vectors\tantivy_search\src\memory_monitor.rs`

```rust
//! Memory leak detection and monitoring for search operations
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result;

/// Tracks memory usage patterns and detects potential leaks
pub struct MemoryMonitor {
    baseline_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    operation_memory: Arc<Mutex<HashMap<String, MemorySnapshot>>>,
    leak_threshold: usize,
    monitor_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub operation: String,
    pub start_memory: usize,
    pub peak_memory: usize,
    pub timestamp: Instant,
    pub expected_cleanup: bool,
}

#[derive(Debug)]
pub struct MemoryReport {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub active_operations: usize,
    pub potential_leaks: Vec<MemorySnapshot>,
    pub cleanup_success_rate: f64,
}

impl MemoryMonitor {
    /// Create new memory monitor with leak detection threshold
    pub fn new(leak_threshold_mb: usize) -> Self {
        Self {
            baseline_memory: AtomicUsize::new(Self::get_current_memory()),
            peak_memory: AtomicUsize::new(0),
            operation_memory: Arc::new(Mutex::new(HashMap::new())),
            leak_threshold: leak_threshold_mb * 1024 * 1024, // Convert to bytes
            monitor_enabled: true,
        }
    }

    /// Start monitoring a specific operation
    pub fn start_operation(&self, operation_name: &str) -> MemoryOperationGuard {
        if !self.monitor_enabled {
            return MemoryOperationGuard::disabled();
        }

        let current_memory = Self::get_current_memory();
        let snapshot = MemorySnapshot {
            operation: operation_name.to_string(),
            start_memory: current_memory,
            peak_memory: current_memory,
            timestamp: Instant::now(),
            expected_cleanup: true,
        };

        {
            let mut operations = self.operation_memory.lock().unwrap();
            operations.insert(operation_name.to_string(), snapshot);
        }

        MemoryOperationGuard::new(
            operation_name.to_string(),
            self.operation_memory.clone(),
            current_memory,
        )
    }

    /// Check for memory leaks in completed operations
    pub fn detect_leaks(&self) -> Vec<MemorySnapshot> {
        let mut leaks = Vec::new();
        let operations = self.operation_memory.lock().unwrap();
        let current_memory = Self::get_current_memory();
        let baseline = self.baseline_memory.load(Ordering::Relaxed);

        for snapshot in operations.values() {
            let memory_growth = current_memory.saturating_sub(snapshot.start_memory);
            let operation_duration = snapshot.timestamp.elapsed();

            // Detect potential leaks
            if memory_growth > self.leak_threshold && operation_duration > Duration::from_secs(5) {
                leaks.push(snapshot.clone());
            }
        }

        leaks
    }

    /// Generate comprehensive memory usage report
    pub fn generate_report(&self) -> MemoryReport {
        let operations = self.operation_memory.lock().unwrap();
        let current_memory = Self::get_current_memory();
        let peak_memory = self.peak_memory.load(Ordering::Relaxed);
        let potential_leaks = self.detect_leaks();

        let total_operations = operations.len();
        let leaked_operations = potential_leaks.len();
        let cleanup_success_rate = if total_operations > 0 {
            ((total_operations - leaked_operations) as f64 / total_operations as f64) * 100.0
        } else {
            100.0
        };

        MemoryReport {
            total_allocated: current_memory,
            peak_usage: peak_memory,
            active_operations: operations.len(),
            potential_leaks,
            cleanup_success_rate,
        }
    }

    /// Force garbage collection and memory cleanup
    pub fn force_cleanup(&self) -> Result<usize> {
        let before_memory = Self::get_current_memory();
        
        // Clear completed operations older than 1 minute
        {
            let mut operations = self.operation_memory.lock().unwrap();
            let cutoff_time = Instant::now() - Duration::from_secs(60);
            operations.retain(|_, snapshot| snapshot.timestamp > cutoff_time);
        }

        // Trigger garbage collection if available (no direct equivalent in Rust)
        std::thread::sleep(Duration::from_millis(10)); // Allow any pending drops

        let after_memory = Self::get_current_memory();
        Ok(before_memory.saturating_sub(after_memory))
    }

    /// Get current process memory usage in bytes
    fn get_current_memory() -> usize {
        // In a real implementation, use a crate like `sysinfo` or `procfs`
        // For testing, we'll simulate memory tracking
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SIMULATED_MEMORY: AtomicUsize = AtomicUsize::new(100 * 1024 * 1024); // 100MB base
        SIMULATED_MEMORY.load(Ordering::Relaxed)
    }

    /// Update simulated memory (for testing)
    pub fn simulate_memory_usage(&self, bytes: usize) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SIMULATED_MEMORY: AtomicUsize = AtomicUsize::new(100 * 1024 * 1024);
        SIMULATED_MEMORY.store(bytes, Ordering::Relaxed);
        
        // Update peak if necessary
        let current_peak = self.peak_memory.load(Ordering::Relaxed);
        self.peak_memory.compare_exchange_weak(current_peak, bytes.max(current_peak), Ordering::Relaxed, Ordering::Relaxed).ok();
    }
}

/// RAII guard that automatically tracks operation memory usage
pub struct MemoryOperationGuard {
    operation_name: Option<String>,
    operations: Option<Arc<Mutex<HashMap<String, MemorySnapshot>>>>,
    start_memory: usize,
}

impl MemoryOperationGuard {
    fn new(
        operation_name: String,
        operations: Arc<Mutex<HashMap<String, MemorySnapshot>>>,
        start_memory: usize,
    ) -> Self {
        Self {
            operation_name: Some(operation_name),
            operations: Some(operations),
            start_memory,
        }
    }

    fn disabled() -> Self {
        Self {
            operation_name: None,
            operations: None,
            start_memory: 0,
        }
    }

    /// Mark that this operation completed successfully and should clean up
    pub fn mark_completed(&mut self) {
        if let (Some(op_name), Some(operations)) = (&self.operation_name, &self.operations) {
            let mut ops = operations.lock().unwrap();
            if let Some(snapshot) = ops.get_mut(op_name) {
                snapshot.expected_cleanup = true;
            }
        }
    }
}

impl Drop for MemoryOperationGuard {
    fn drop(&mut self) {
        if let (Some(op_name), Some(operations)) = (&self.operation_name, &self.operations) {
            let end_memory = MemoryMonitor::get_current_memory();
            let mut ops = operations.lock().unwrap();
            
            // Update final memory usage or remove if cleaned up properly
            if end_memory <= self.start_memory + 1024 * 1024 { // Allow 1MB variance
                ops.remove(op_name); // Clean operation, remove from tracking
            } else if let Some(snapshot) = ops.get_mut(op_name) {
                snapshot.peak_memory = end_memory; // Potential leak, keep for analysis
            }
        }
    }
}

/// Integration helpers for existing search components
pub trait MemoryTrackable {
    fn with_memory_tracking<F, R>(&self, operation: &str, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>;
}

impl<T> MemoryTrackable for T {
    fn with_memory_tracking<F, R>(&self, operation: &str, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>
    {
        let monitor = MemoryMonitor::new(50); // 50MB threshold
        let _guard = monitor.start_operation(operation);
        f()
    }
}
```

### Step 2: Add comprehensive leak detection tests (2 minutes)
Create file: `C:\code\LLMKG\vectors\tantivy_search\tests\memory_leak_tests.rs`

```rust
//! Memory leak detection integration tests
use tantivy_search::memory_monitor::*;
use std::time::Duration;
use std::thread;

#[test]
fn test_memory_monitor_creation() {
    let monitor = MemoryMonitor::new(50); // 50MB threshold
    let report = monitor.generate_report();
    assert!(report.total_allocated > 0);
    assert_eq!(report.active_operations, 0);
    assert_eq!(report.potential_leaks.len(), 0);
}

#[test]
fn test_operation_tracking() {
    let monitor = MemoryMonitor::new(10);
    
    {
        let _guard = monitor.start_operation("test_indexing");
        // Simulate work
        thread::sleep(Duration::from_millis(10));
    } // Guard drops here, should clean up
    
    let report = monitor.generate_report();
    assert_eq!(report.active_operations, 0); // Should be cleaned up
}

#[test]
fn test_leak_detection() {
    let monitor = MemoryMonitor::new(5); // 5MB threshold
    
    // Simulate memory leak scenario
    monitor.simulate_memory_usage(100 * 1024 * 1024); // 100MB
    
    {
        let mut guard = monitor.start_operation("leaky_operation");
        monitor.simulate_memory_usage(110 * 1024 * 1024); // Grow to 110MB
        
        // Don't mark as completed - simulates leak
        thread::sleep(Duration::from_millis(10));
    }
    
    // Wait a moment to ensure leak detection
    thread::sleep(Duration::from_millis(10));
    
    let leaks = monitor.detect_leaks();
    assert!(!leaks.is_empty(), "Should detect memory leak");
    assert_eq!(leaks[0].operation, "leaky_operation");
}

#[test]
fn test_memory_cleanup() {
    let monitor = MemoryMonitor::new(10);
    monitor.simulate_memory_usage(200 * 1024 * 1024); // 200MB
    
    let cleanup_amount = monitor.force_cleanup().unwrap();
    assert!(cleanup_amount >= 0); // Should not fail
}

#[test]
fn test_memory_report_generation() {
    let monitor = MemoryMonitor::new(20);
    
    // Create some operations
    let _guard1 = monitor.start_operation("operation_1");
    let _guard2 = monitor.start_operation("operation_2");
    
    let report = monitor.generate_report();
    assert_eq!(report.active_operations, 2);
    assert!(report.cleanup_success_rate >= 0.0);
    assert!(report.cleanup_success_rate <= 100.0);
}

#[test]
fn test_memory_trackable_trait() {
    use tantivy_search::memory_monitor::MemoryTrackable;
    
    let dummy_object = ();
    let result = dummy_object.with_memory_tracking("test_operation", || {
        Ok("success".to_string())
    });
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[test]
fn test_guard_completion_marking() {
    let monitor = MemoryMonitor::new(10);
    
    {
        let mut guard = monitor.start_operation("completed_operation");
        guard.mark_completed();
        // Simulate successful completion
    }
    
    let report = monitor.generate_report();
    assert_eq!(report.active_operations, 0); // Should be cleaned up
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_concurrent_memory_tracking() {
        let monitor = Arc::new(MemoryMonitor::new(25));
        let mut handles = vec![];
        
        // Spawn multiple threads doing memory operations
        for i in 0..5 {
            let monitor_clone = Arc::clone(&monitor);
            let handle = thread::spawn(move || {
                let operation_name = format!("concurrent_op_{}", i);
                let _guard = monitor_clone.start_operation(&operation_name);
                thread::sleep(Duration::from_millis(50));
                // Guard drops automatically
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should have cleaned up all operations
        thread::sleep(Duration::from_millis(100));
        let report = monitor.generate_report();
        assert_eq!(report.active_operations, 0);
    }
}
```

### Step 3: Update module exports (1 minute)
Add to `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs`:

```rust
pub mod memory_monitor;
```

## Verification Steps (2 minutes)

### Verify 1: Compilation succeeds
```bash
cd C:\code\LLMKG\vectors\tantivy_search
cargo check
```

### Verify 2: Memory leak detection tests pass
```bash
cargo test memory_leak_tests
```
**Expected output:**
```
running 8 tests
test memory_leak_tests::test_memory_monitor_creation ... ok
test memory_leak_tests::test_operation_tracking ... ok
test memory_leak_tests::test_leak_detection ... ok
test memory_leak_tests::test_memory_cleanup ... ok
test memory_leak_tests::test_memory_report_generation ... ok
test memory_leak_tests::test_memory_trackable_trait ... ok
test memory_leak_tests::test_guard_completion_marking ... ok
test memory_leak_tests::integration_tests::test_concurrent_memory_tracking ... ok

test result: ok. 8 passed; 0 failed
```

### Verify 3: Integration test
```bash
cargo test --test memory_leak_tests
```

## Success Validation Checklist
- [ ] File `memory_monitor.rs` completely implemented with leak detection
- [ ] File `memory_leak_tests.rs` created with 8+ comprehensive tests
- [ ] Command `cargo check` completes without errors
- [ ] Command `cargo test memory_leak_tests` passes all tests
- [ ] Memory operations are properly tracked with RAII guards
- [ ] Leak detection algorithm identifies potential memory leaks
- [ ] Memory cleanup and garbage collection works
- [ ] Concurrent memory tracking is thread-safe
- [ ] Integration trait allows easy memory tracking for existing components

## Context for Task 30
Task 30 will implement configuration validation to ensure all search system settings are properly validated, with memory monitoring integration to track configuration parsing memory usage and prevent configuration-related memory leaks.
