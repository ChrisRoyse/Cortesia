# Task 100_01: Add ShutdownHandler Struct with Atomic Shutdown State

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 99 completed (Health checks implemented)
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` (module declarations)
- `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` (tokio dependency)

## Complete Context (For AI with ZERO Knowledge)

**What is a ShutdownHandler?** A shutdown handler is a critical system component that manages graceful application termination. In Rust applications, it coordinates cleanup tasks, ensures data integrity, and prevents resource leaks when the application receives termination signals (SIGTERM, Ctrl+C).

**What are Atomic Operations?** Atomic operations are thread-safe operations that complete without interruption. `AtomicBool` provides lock-free boolean values that can be safely accessed from multiple threads without data races.

**Why Use Arc<AtomicBool>?** Arc (Atomically Reference Counted) allows sharing the shutdown state across multiple threads, while AtomicBool provides thread-safe boolean operations. This combination enables any thread to check or set shutdown status safely.

**Context in Phase 1:** This is part of the graceful shutdown system that ensures the Tantivy search engine can properly close indexes, flush data, and clean up resources before termination.

## Exact Steps (6 minutes implementation)

### Step 1: Create shutdown.rs module (2 minutes)
Create file `C:/code/LLMKG/vectors/tantivy_search/src/shutdown.rs`:
```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error};

/// Core shutdown handler for graceful application termination
/// 
/// Manages shutdown state using atomic operations for thread-safe access
/// and coordinates cleanup tasks across the application.
pub struct ShutdownHandler {
    /// Atomic boolean indicating if shutdown has been requested
    /// Uses Relaxed ordering for performance in status checks
    shutdown_requested: Arc<AtomicBool>,
    
    /// Thread-safe storage for cleanup tasks to execute during shutdown
    /// RwLock allows multiple readers (status checks) with exclusive writers (registration)
    cleanup_tasks: Arc<RwLock<Vec<CleanupTask>>>,
}

/// Represents a cleanup task to be executed during shutdown
#[derive(Clone)]
pub struct CleanupTask {
    /// Human-readable name for logging and debugging
    name: String,
    
    /// Priority level (lower numbers execute first)
    /// Allows critical tasks like index flushing to run before less critical cleanup
    priority: u8,
    
    /// The actual cleanup function to execute
    /// Uses Arc to allow sharing across threads
    cleanup_fn: Arc<dyn Fn() -> Result<()> + Send + Sync>,
}

impl ShutdownHandler {
    /// Create a new shutdown handler with empty task list
    pub fn new() -> Self {
        Self {
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            cleanup_tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Check if shutdown has been requested (thread-safe, non-blocking)
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Relaxed)
    }
    
    /// Request shutdown (thread-safe, can be called from any thread)
    pub fn request_shutdown(&self) {
        info!("Shutdown requested");
        self.shutdown_requested.store(true, Ordering::Relaxed);
    }
    
    /// Get a clone of the shutdown flag for sharing with other components
    pub fn get_shutdown_flag(&self) -> Arc<AtomicBool> {
        self.shutdown_requested.clone()
    }
}

impl Default for ShutdownHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shutdown_handler_creation() {
        let handler = ShutdownHandler::new();
        assert!(!handler.is_shutdown_requested());
    }
    
    #[test]
    fn test_shutdown_request() {
        let handler = ShutdownHandler::new();
        
        // Initially not requested
        assert!(!handler.is_shutdown_requested());
        
        // Request shutdown
        handler.request_shutdown();
        assert!(handler.is_shutdown_requested());
    }
    
    #[test]
    fn test_shutdown_flag_sharing() {
        let handler = ShutdownHandler::new();
        let flag = handler.get_shutdown_flag();
        
        // Both should show same state
        assert!(!handler.is_shutdown_requested());
        assert!(!flag.load(Ordering::Relaxed));
        
        // Update through handler
        handler.request_shutdown();
        assert!(handler.is_shutdown_requested());
        assert!(flag.load(Ordering::Relaxed));
    }
    
    #[test]
    fn test_default_implementation() {
        let handler = ShutdownHandler::default();
        assert!(!handler.is_shutdown_requested());
    }
}
```

### Step 2: Update lib.rs to include module (1 minute)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:
```rust
pub mod shutdown;
pub use shutdown::{ShutdownHandler, CleanupTask};
```

### Step 3: Create basic integration test (3 minutes)
Create file `C:/code/LLMKG/vectors/tantivy_search/tests/shutdown_integration.rs`:
```rust
use tantivy_search::ShutdownHandler;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_shutdown_from_multiple_threads() {
    let handler = Arc::new(ShutdownHandler::new());
    
    // Clone for thread
    let handler_clone = handler.clone();
    
    // Spawn thread that checks shutdown status
    let check_thread = thread::spawn(move || {
        for _ in 0..10 {
            if handler_clone.is_shutdown_requested() {
                return true;
            }
            thread::sleep(Duration::from_millis(10));
        }
        false
    });
    
    // Wait a bit, then request shutdown
    thread::sleep(Duration::from_millis(20));
    handler.request_shutdown();
    
    // Thread should detect shutdown
    let detected = check_thread.join().unwrap();
    assert!(detected, "Thread should have detected shutdown request");
}
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test shutdown
```

**Expected output:**
```
running 4 tests
test shutdown::tests::test_shutdown_handler_creation ... ok
test shutdown::tests::test_shutdown_request ... ok
test shutdown::tests::test_shutdown_flag_sharing ... ok
test shutdown::tests::test_default_implementation ... ok
test result: ok. 4 passed; 0 failed; 0 ignored
```

## If This Task Fails

**Error 1: "error[E0433]: failed to resolve: use of undeclared crate or module"**
```bash
# Solution: Missing dependencies
cargo add tokio --features full
cargo add tracing
cargo add anyhow
cargo build
```

**Error 2: "Permission denied" when creating files**
```bash
# Solution (Windows): Fix permissions
icacls C:/code/LLMKG/vectors/tantivy_search /grant Users:F /T
# Solution (Unix): Fix permissions  
chmod -R 755 C:/code/LLMKG/vectors/tantivy_search
```

**Error 3: "could not compile due to previous error"**
```bash
# Solution: Version conflicts or cache issues
cargo clean
cargo update
cargo build --release
```

## Troubleshooting Checklist
- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] Tokio dependency with "full" features in Cargo.toml
- [ ] tracing and anyhow dependencies present
- [ ] File permissions allow read/write operations
- [ ] No conflicting processes locking files
- [ ] src/lib.rs properly declares the shutdown module

## Recovery Procedures

### Network Failure During Crate Download
If crates.io is unreachable:
1. Check internet: `ping crates.io`
2. Try proxy: `export HTTPS_PROXY=http://proxy:8080`
3. Use vendored deps: `cargo vendor && cargo build --offline`

### Compilation Failure  
If build fails:
1. Clear cache: `cargo clean`
2. Update deps: `cargo update`
3. Check target: `cargo build --target x86_64-pc-windows-msvc`
4. Verify syntax: `cargo check`

### Test Failure
If tests fail:
1. Run single test: `cargo test test_shutdown_handler_creation`
2. Run with output: `cargo test shutdown -- --nocapture`
3. Check logs: `RUST_LOG=debug cargo test`

## Success Validation Checklist
- [ ] File `src/shutdown.rs` exists with exactly 4 test functions
- [ ] Command `cargo test shutdown` shows 4 tests passing
- [ ] ShutdownHandler struct has AtomicBool and RwLock fields
- [ ] Integration test passes demonstrating thread safety
- [ ] Module properly exported in lib.rs

## Files Created For Next Task
1. **C:/code/LLMKG/vectors/tantivy_search/src/shutdown.rs** - Complete ShutdownHandler implementation
2. **C:/code/LLMKG/vectors/tantivy_search/tests/shutdown_integration.rs** - Thread safety integration test
3. **Updated lib.rs** - Module export declarations

## Context for Task 100_02
Task 100_02 will build upon this ShutdownHandler to add the CleanupTask registration system, allowing components to register cleanup functions that execute during graceful shutdown.