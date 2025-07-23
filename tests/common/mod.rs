/// Common test utilities and setup for all integration tests
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize test environment once
pub fn init() {
    INIT.call_once(|| {
        // Set up test environment
        std::env::set_var("RUST_TEST_THREADS", "1");
        
        // Windows-specific initialization
        #[cfg(target_os = "windows")]
        {
            // Ensure test binaries don't conflict
            unsafe {
                // Enable process cleanup on Windows
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                std::env::set_var("_CREATE_NO_WINDOW", CREATE_NO_WINDOW.to_string());
            }
        }
        
        // Register panic handler to ensure cleanup
        let default_panic = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            cleanup_on_panic();
            default_panic(info);
        }));
    });
}

/// Cleanup function called on panic
fn cleanup_on_panic() {
    #[cfg(target_os = "windows")]
    {
        // Give Windows time to release file handles
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Test guard that ensures proper cleanup
pub struct TestEnvironment;

impl TestEnvironment {
    pub fn new() -> Self {
        init();
        TestEnvironment
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        // Ensure cleanup on test completion
        #[cfg(target_os = "windows")]
        {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

/// Macro to setup test environment
#[macro_export]
macro_rules! setup_test {
    () => {
        let _env = crate::common::TestEnvironment::new();
    };
}