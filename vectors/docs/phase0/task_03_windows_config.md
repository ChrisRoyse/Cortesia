# Task 03: Setup Windows-Specific Configuration

## Context
You are continuing Phase 0 of a Rust-based vector search system. Tasks 01-02 created the basic project structure and Rust modules. Now you need to create Windows-specific configuration files to ensure optimal performance and compatibility.

## Objective
Create Windows-specific configuration files (.cargo/config.toml, rust-toolchain.toml) and implement Windows environment setup functions.

## Requirements
1. Create .cargo/config.toml for Windows-specific build settings
2. Create rust-toolchain.toml for version pinning
3. Implement Windows environment setup in setup.rs
4. Add Windows-specific path handling
5. Test directory creation and path resolution on Windows

## Configuration Files to Create

### .cargo/config.toml
```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-cpu=native"]

[build]
target = "x86_64-pc-windows-msvc"

[env]
# Windows-specific environment variables
RUST_LOG = "debug"
```

### rust-toolchain.toml
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-pc-windows-msvc"]
```

## Implementation for setup.rs
```rust
use std::path::{Path, PathBuf};
use anyhow::Result;
use tracing::{info, debug};

#[cfg(windows)]
use windows_sys::Win32::System::Console::AllocConsole;

pub struct WindowsSetup;

impl WindowsSetup {
    /// Setup Windows environment for optimal performance
    pub async fn setup_environment() -> Result<()> {
        Self::setup_console()?;
        Self::create_directories()?;
        Self::setup_logging()?;
        Self::verify_permissions()?;
        
        info!("Windows environment setup completed successfully");
        Ok(())
    }
    
    #[cfg(windows)]
    fn setup_console() -> Result<()> {
        // Windows console setup for proper Unicode support
        unsafe { AllocConsole(); }
        Ok(())
    }
    
    #[cfg(not(windows))]
    fn setup_console() -> Result<()> {
        Ok(())
    }
    
    fn create_directories() -> Result<()> {
        let dirs = vec![
            "src", "test_data", "indexes", 
            "indexes/tantivy", "indexes/lancedb"
        ];
        
        for dir in dirs {
            let path = PathBuf::from(dir);
            std::fs::create_dir_all(&path)?;
            debug!("Created directory: {}", path.display());
        }
        
        Ok(())
    }
}
```

## Implementation Steps
1. Create .cargo/ directory and config.toml
2. Create rust-toolchain.toml in project root
3. Implement WindowsSetup struct in setup.rs
4. Add Windows-specific console setup
5. Add directory creation with proper path handling
6. Test environment setup function
7. Verify all paths work correctly on Windows

## Success Criteria
- [ ] .cargo/config.toml exists with Windows-specific settings
- [ ] rust-toolchain.toml exists with stable toolchain pinning
- [ ] WindowsSetup struct implemented in setup.rs
- [ ] Windows console setup works correctly
- [ ] Directory creation works with Windows paths
- [ ] All configuration compiles without errors
- [ ] Environment setup function runs successfully

## Test Command
```bash
cargo check
cargo run --bin setup_test  # If you create a test binary
```

## Time Estimate
10 minutes

## Next Task
Task 04: Setup logging and tracing configuration for debugging and monitoring.