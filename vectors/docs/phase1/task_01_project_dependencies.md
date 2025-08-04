# Task 01: Create Initial Project Structure with Complete Dependencies

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** None - First task in Phase 1
**Input Files:** None (creating from scratch)

## Complete Context (For AI with ZERO Knowledge)

**What is Cargo.toml?** Cargo.toml is Rust's package manifest file, similar to package.json in Node.js or requirements.txt in Python. It defines metadata about your Rust project and specifies all external dependencies (crates) your project needs.

**What is Tantivy?** Tantivy is a fast, full-text search engine library written in Rust, similar to Apache Lucene but designed specifically for Rust applications. It provides indexing, searching, and scoring capabilities for text documents with high performance and memory safety.

**Why These Specific Dependencies?**
- **tantivy = "0.22.0"** - Core search engine providing indexing, querying, and text analysis
- **tree-sitter = "0.20.10"** - Parser generator for creating Abstract Syntax Trees (AST) to understand code structure
- **tree-sitter-rust = "0.20.4"** - Language-specific parser for Rust code syntax analysis
- **tree-sitter-python = "0.20.4"** - Language-specific parser for Python code syntax analysis
- **tokio = { version = "1.41.0", features = ["full"] }** - Async runtime for handling concurrent operations like file I/O and search requests
- **anyhow = "1.0.94"** - Ergonomic error handling library providing rich error context and chaining
- **serde = { version = "1.0.217", features = ["derive"] }** - Serialization framework for converting Rust data structures to/from JSON, TOML, etc.
- **serde_json = "1.0.134"** - JSON-specific serialization support for configuration and data exchange
- **tempfile = "3.14.0"** - Testing utility for creating temporary files and directories safely
- **winapi = { version = "0.3.55", features = ["winuser"] }** - Windows-specific APIs for file system operations and path handling

**Why This Configuration?** The project uses a multi-crate workspace architecture where this initial crate serves as the foundation for the Tantivy-based search system. The dev profile with opt-level = 1 provides reasonable performance during development while maintaining fast compilation times.

## Exact Steps (6 minutes implementation)

### Step 1: Create project directory structure (1 minute)
```bash
# Navigate to vectors directory and create project
cd C:\code\LLMKG\vectors
mkdir tantivy_search
cd tantivy_search

# Initialize basic Rust project structure
mkdir src
mkdir tests
mkdir benches
```

### Step 2: Create Cargo.toml with complete dependency specification (3 minutes)
Create file `C:\code\LLMKG\vectors\tantivy_search\Cargo.toml` with EXACT content:
```toml
[package]
name = "tantivy_search"
version = "0.1.0"
edition = "2021"
author = "LLMKG Vector Search Team"
description = "High-performance text search engine with AST-aware chunking"
license = "MIT"
repository = "https://github.com/llmkg/vector-search"
readme = "README.md"
keywords = ["search", "tantivy", "ast", "indexing"]
categories = ["text-processing", "database"]

[dependencies]
# Core search engine - provides indexing, querying, text analysis
tantivy = "0.22.0"

# AST parsing for semantic chunking
tree-sitter = "0.20.10"
tree-sitter-rust = "0.20.4"
tree-sitter-python = "0.20.4"

# Async runtime for concurrent operations
tokio = { version = "1.41.0", features = ["full"] }

# Error handling with rich context
anyhow = "1.0.94"

# Serialization for configuration and data exchange
serde = { version = "1.0.217", features = ["derive"] }
serde_json = "1.0.134"

# Logging for debugging and monitoring
log = "0.4.22"
env_logger = "0.11.5"

# Command line argument parsing
clap = { version = "4.5.20", features = ["derive"] }

# File system utilities
walkdir = "2.5.0"

[dev-dependencies]
# Testing utilities
tempfile = "3.14.0"
criterion = { version = "0.5.1", features = ["html_reports"] }

# Windows-specific dependencies for path handling
[target.'cfg(windows)'.dependencies]
winapi = { version = "0.3.55", features = ["winuser", "fileapi", "handleapi"] }

# Optimization profiles
[profile.dev]
opt-level = 1
overflow-checks = true
debug-assertions = true

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
debug = true

[profile.test]
opt-level = 1
debug = true

[[bench]]
name = "search_performance"
harness = false
path = "benches/search_performance.rs"
```

### Step 3: Create initial source structure (1 minute)
Create file `C:\code\LLMKG\vectors\tantivy_search\src\main.rs`:
```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub index_path: String,
    pub max_memory: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            index_path: "indexes/main".to_string(),
            max_memory: 50_000_000, // 50MB
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Tantivy Search Engine - Phase 1 Foundation");
    println!("📁 Project structure initialized");
    println!("📦 Dependencies configured");
    println!("✅ Ready for Phase 1 implementation");
    
    let config = SearchConfig::default();
    println!("🔧 Default config: {:?}", config);
    
    Ok(())
}
```

And create `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs`:
```rust
//! Tantivy Search Engine - Phase 1 Foundation
//! 
//! This crate provides a high-performance text search engine built on Tantivy
//! with AST-aware chunking capabilities for code search.

use anyhow::Result;

/// Version information for the search engine
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Initialize the search system
pub fn init() -> Result<()> {
    println!("Initializing Tantivy Search Engine v{}", version());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
}
```

### Step 4: Create basic documentation (30 seconds)
```bash
echo '# Tantivy Search System' > README.md
echo '## Phase 1: Core Search Engine with AST-aware Chunking' >> README.md
echo '' >> README.md
echo 'High-performance text search using Tantivy with semantic code chunking.' >> README.md
```

### Step 5: Verify project compilation (30 seconds)
```bash
cargo check
```

## Verification Steps (2 minutes)

### Verify 1: Project structure exists
```bash
dir C:\code\LLMKG\vectors\tantivy_search
```
**Expected output:**
```
Cargo.toml  README.md  src/
```

### Verify 2: Dependencies compile successfully
```bash
cargo check
```
**Expected output:**
```
    Updating crates.io index
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in 15.23s
```

### Verify 3: Application runs successfully
```bash
cargo run
```
**Expected output:**
```
Tantivy Search System v0.1.0 - Phase 1 Initialized
```

### Verify 4: All dependencies resolve without conflicts
```bash
cargo tree
```
**Expected output:** Dependency tree showing tantivy 0.22.0 and all transitive dependencies without errors

## If This Task Fails

**Error 1: "cargo: command not found"**
```bash
# Solution: Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustc --version  # Should show 1.70+
cargo --version  # Should show 1.70+
```

**Error 2: "failed to resolve dependencies"**
```bash
# Solution: Network/registry issues
cargo clean
rm -f Cargo.lock
cargo update
cargo check --verbose
# If still fails, check proxy settings:
export HTTPS_PROXY=http://your-proxy:8080
```

**Error 3: "permission denied" when creating directories**
```bash
# Solution (Windows): Fix permissions
icacls C:\code\LLMKG /grant Users:F /T
# Or run as administrator:
# Right-click Command Prompt -> "Run as Administrator"

# Solution (Unix): Fix permissions
sudo mkdir -p C:/code/LLMKG/vectors
sudo chown -R $USER:$USER C:/code/LLMKG
chmod -R 755 C:/code/LLMKG
```

**Error 4: "linker `link.exe` not found"**
```bash
# Solution: Install Visual Studio Build Tools for Windows
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Or install full Visual Studio Community
# Then restart terminal and try again
rustup default stable-msvc  # Use MSVC toolchain on Windows
```

**Error 5: "error: Microsoft Visual C++ 14.0 is required"**
```bash
# Solution: Install Microsoft C++ Build Tools
# Method 1: Visual Studio Installer
winget install Microsoft.VisualStudio.2022.BuildTools
# Method 2: Direct download
# https://aka.ms/vs/17/release/vs_buildtools.exe
# Select "C++ build tools" workload
```

## Troubleshooting Checklist
- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] Cargo version 1.70+ installed (`cargo --version`)
- [ ] Internet connection available (`ping crates.io`)
- [ ] Directory permissions allow read/write (`ls -la C:/code/LLMKG`)
- [ ] No antivirus blocking cargo operations
- [ ] Sufficient disk space (>2GB free for dependencies)
- [ ] Windows: Visual Studio Build Tools installed
- [ ] Proxy configured if behind corporate firewall

## Recovery Procedures

### Complete Environment Reset
If multiple errors persist:
1. **Uninstall Rust**: `rustup self uninstall`
2. **Remove cargo cache**: `rm -rf ~/.cargo`
3. **Reinstall Rust**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
4. **Update to latest**: `rustup update stable`
5. **Verify installation**: `cargo --version`
6. **Retry task from Step 1**

### Dependency Resolution Issues
If specific crates fail to download:
1. **Check crate exists**: Visit https://crates.io/crates/tantivy
2. **Try alternative registry**: Add to `~/.cargo/config.toml`:
   ```toml
   [source.crates-io]
   replace-with = "tuna"
   [source.tuna]
   registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
   ```
3. **Use offline mode**: `cargo fetch` then `cargo build --offline`
4. **Manual dependency fixing**: Remove problematic deps, build, then re-add

### Windows-Specific Issues
For Windows path or compiler problems:
1. **Use Windows Subsystem for Linux (WSL2)**:
   ```bash
   wsl --install
   # Then run all commands in WSL2 Ubuntu environment
   ```
2. **Use MSVC toolchain**: `rustup default stable-x86_64-pc-windows-msvc`
3. **Use GNU toolchain**: `rustup default stable-x86_64-pc-windows-gnu`
4. **Check PATH**: Ensure `C:\Program Files\Microsoft Visual Studio\...\bin` is in PATH

## Success Validation Checklist
- [ ] Directory `C:\code\LLMKG\vectors\tantivy_search` exists
- [ ] File `Cargo.toml` contains exactly 45 dependencies/configs
- [ ] File `src/main.rs` contains version print statement
- [ ] File `src/lib.rs` contains version function
- [ ] File `README.md` exists with project description
- [ ] Command `cargo check` completes in <30 seconds
- [ ] Command `cargo run` prints initialization message
- [ ] Command `cargo tree` shows no dependency conflicts
- [ ] No compilation warnings or errors

## Files Created For Next Task

Task 02 expects these EXACT files to exist:
1. **C:\code\LLMKG\vectors\tantivy_search\Cargo.toml** - Complete manifest with all dependencies
2. **C:\code\LLMKG\vectors\tantivy_search\src\main.rs** - Main binary entry point
3. **C:\code\LLMKG\vectors\tantivy_search\src\lib.rs** - Library entry point with version function
4. **C:\code\LLMKG\vectors\tantivy_search\README.md** - Basic project documentation

## Context for Task 02

Task 02 will create the modular architecture by implementing the core module structure (schema.rs, chunker.rs, indexer.rs, search.rs, utils.rs). The dependency foundation established here provides all the necessary crates for AST parsing, async operations, error handling, and search functionality that will be used throughout Phase 1.