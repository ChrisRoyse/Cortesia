# Task 01: Create Project Structure and Cargo.toml

## Context
You are setting up the foundation for a Rust-based vector search system that combines Tantivy (text search) and LanceDB (vector search) with Windows compatibility. This is Phase 0 Task 1 of a larger project to build a production-ready search system.

## Objective
Create the basic Rust project structure with a Windows-compatible Cargo.toml file containing all necessary dependencies.

## Requirements
1. Create a new Rust library project structure
2. Setup Cargo.toml with all Windows-compatible dependencies
3. Create basic directory structure for the project
4. Ensure all dependencies work on Windows

## Expected Directory Structure After Completion
```
src/
├── lib.rs                 // Main library interface with explicit module declarations
├── setup.rs              // Environment setup functions
├── test_data.rs           // Test data generation utilities  
├── benchmark.rs           // Performance benchmarking functions
├── validation.rs          // Component validation functions
└── test_utils.rs          // Standardized test utilities module
test_data/                 // Test data storage directory
indexes/                   // Index storage directory
├── tantivy/              // Tantivy-specific index files
└── lancedb/              // LanceDB-specific index files
```

## Explicit Module Import Paths
All modules must use these exact import paths:

### lib.rs Module Structure
```rust
//! LLMKG Vector Search System - Phase 0 Prerequisites
//! 
//! This library provides the foundation for a hybrid search system
//! combining Tantivy (text search) and LanceDB (vector search).

// Standard library imports
use std::path::Path;
use std::result::Result as StdResult;

// Third-party crate imports  
use anyhow::Result;
use tracing::info;

// Internal module declarations - EXACT PATH REFERENCES
pub mod setup;              // Located at src/setup.rs
pub mod test_data;          // Located at src/test_data.rs  
pub mod benchmark;          // Located at src/benchmark.rs
pub mod validation;         // Located at src/validation.rs

// Test utilities - conditionally compiled
#[cfg(test)]
pub mod test_utils;         // Located at src/test_utils.rs

// Re-exports for external usage
pub use setup::*;
pub use test_data::*;
pub use benchmark::*;
pub use validation::*;

// Test utilities re-export for external test crates
#[cfg(test)]
pub use test_utils::*;

/// Main result type for the library
pub type Result<T> = anyhow::Result<T>;

/// Main error type for the library
pub type Error = anyhow::Error;
```

### Inter-Module Import Patterns
Each module must follow these explicit import patterns:

#### setup.rs Import Structure
```rust
// External crate imports
use anyhow::Result;
use tracing::{info, warn, error};
use std::path::{Path, PathBuf};
use std::env;

// Internal crate imports - EXPLICIT PATHS
use crate::validation::ArchitectureValidator;  // From src/validation.rs
```

#### test_data.rs Import Structure  
```rust
// External crate imports
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

// Internal crate imports - EXPLICIT PATHS  
use crate::setup::WindowsSetup;               // From src/setup.rs
```

#### benchmark.rs Import Structure
```rust  
// External crate imports
use anyhow::Result;
use std::time::{Duration, Instant};

// Internal crate imports - EXPLICIT PATHS
use crate::test_data::TestDataGenerator;      // From src/test_data.rs
use crate::validation::ArchitectureValidator; // From src/validation.rs
```

#### validation.rs Import Structure
```rust
// External crate imports  
use anyhow::Result;
use std::process::Command;

// Internal crate imports - EXPLICIT PATHS
use crate::setup::WindowsSetup;               // From src/setup.rs
```

#### test_utils.rs Import Structure (Test-Only)
```rust
// External crate imports
use std::path::Path;
use std::time::{Duration, Instant};
use tantivy::schema::{Schema, Field, TEXT, STORED};
use tantivy::{Index, IndexWriter, doc};
use anyhow::Result;
use tempfile::TempDir;

// Internal crate imports - EXPLICIT PATHS
use crate::test_data::TestDataGenerator;      // From src/test_data.rs
use crate::benchmark::BaselineBenchmark;      // From src/benchmark.rs
```

## Dependencies Required (Windows-Compatible)
```toml
[dependencies]
# Text search and parsing - EXACT VERSIONS
tantivy = { version = "0.24.0", features = ["mmap"] }
tree-sitter = { version = "0.20.10" }
tree-sitter-rust = { version = "0.20.4" }
tree-sitter-python = { version = "0.20.4" }

# Async runtime and concurrency - EXACT VERSIONS
tokio = { version = "1.35.1", features = ["full", "tracing"] }
rayon = { version = "1.8.0" }

# Serialization and data handling - EXACT VERSIONS
serde = { version = "1.0.193", features = ["derive"] }
serde_json = { version = "1.0.108" }

# Error handling and logging - EXACT VERSIONS
anyhow = { version = "1.0.78" }
thiserror = { version = "1.0.51" }
tracing = { version = "0.1.40" }
tracing-subscriber = { version = "0.3.18", features = ["env-filter"] }

# File system operations - EXACT VERSIONS
walkdir = { version = "2.4.0" }
regex = { version = "1.10.2" }

# Testing and temporary files - EXACT VERSIONS
tempfile = { version = "3.8.1" }

[target.'cfg(windows)'.dependencies]  
# Windows-specific dependencies - EXACT VERSIONS
windows-sys = { version = "0.52.0", features = [
    "Win32_Foundation",
    "Win32_System_Environment", 
    "Win32_Storage_FileSystem"
] }

[dev-dependencies]
# Development and testing dependencies - EXACT VERSIONS
criterion = { version = "0.5.1", features = ["html_reports"] }
proptest = { version = "1.4.0" }
tokio-test = { version = "0.4.3" }
```

## Implementation Steps
1. Run `cargo init --lib` in the project root
2. Replace Cargo.toml with the Windows-compatible dependency list above
3. Create the directory structure shown above
4. Create placeholder Rust files (lib.rs, setup.rs, etc.) with basic module declarations
5. Run `cargo check` to verify all dependencies resolve on Windows

## Success Criteria
- [ ] Cargo.toml exists with correct Windows-compatible dependencies
- [ ] All required directories exist (src/, test_data/, indexes/)
- [ ] Basic Rust module files exist with proper module structure
- [ ] `cargo check` runs without errors
- [ ] All dependencies download and compile successfully on Windows

## Test Command
```bash
cargo check
```

## Time Estimate
10 minutes

## Next Task
Task 02: Create basic Rust module structure with proper exports and imports.