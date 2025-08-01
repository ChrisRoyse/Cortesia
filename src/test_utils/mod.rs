/// Test utilities for cross-platform compatibility
use std::sync::Once;
use std::process::Command;

static CLEANUP: Once = Once::new();

/// Windows-specific cleanup for test processes
#[cfg(target_os = "windows")]
pub fn cleanup_test_processes() {
    // Kill any lingering test processes before starting
    let _ = Command::new("taskkill")
        .args(["/F", "/IM", "llmkg*.exe"])
        .output();
}

#[cfg(not(target_os = "windows"))]
pub fn cleanup_test_processes() {
    // No-op on Unix systems
}

/// Initialize test environment with proper cleanup
pub fn init_test_env() {
    CLEANUP.call_once(|| {
        cleanup_test_processes();
        
        // Register cleanup on panic
        std::panic::set_hook(Box::new(|_| {
            cleanup_test_processes();
        }));
    });
}

/// Test guard that ensures cleanup on drop
pub struct TestGuard;

impl Drop for TestGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            // Force cleanup of any processes created during test
            std::thread::sleep(std::time::Duration::from_millis(100));
            cleanup_test_processes();
        }
    }
}

/// Create a test guard for automatic cleanup
pub fn test_guard() -> TestGuard {
    init_test_env();
    TestGuard
}