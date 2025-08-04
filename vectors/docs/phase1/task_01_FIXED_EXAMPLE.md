# Task 01: Create Initial Cargo.toml and Project Foundation [FIXED VERSION]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** None - This is the first task
**Required Tools:** Rust toolchain, file system access

## Complete Context (For AI with ZERO Knowledge)

You are creating the foundation for a **Tantivy-based text search system** that will handle special characters like `[workspace]`, `Result<T,E>`, `#[derive]` etc. 

**What is Tantivy?** A full-text search engine library for Rust, similar to Elasticsearch but as a Rust crate.

**Project Goal:** Build a search system that indexes code files using AST-based chunking and allows searching with full special character support.

**This Task:** Creates the Cargo.toml manifest file that defines all dependencies and the basic project structure.

## Pre-Task Environment Check
Run these commands first:
```bash
rustc --version  # Should show Rust 1.70+ 
cargo --version  # Should show cargo 1.70+
```
If either fails, install Rust from https://rustup.rs/

## Exact Steps (6 minutes implementation)

### Step 1: Create Project Directory (1 minute)
```bash
# Create the project directory structure
mkdir -p C:/code/LLMKG/vectors/tantivy_search
cd C:/code/LLMKG/vectors/tantivy_search
```

**Expected result:** You are now in `C:/code/LLMKG/vectors/tantivy_search/`

### Step 2: Create Cargo.toml (3 minutes)
Create the file `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` with EXACTLY this content:

```toml
[package]
name = "tantivy_search"
version = "0.1.0" 
edition = "2021"
description = "Phase 1: Tantivy-based text search with special character support"

[dependencies]
# Core search engine
tantivy = "0.22.0"

# Async runtime for concurrent operations
tokio = { version = "1.41.0", features = ["full"] }

# Error handling
anyhow = "1.0.94"

# Serialization for config/results
serde = { version = "1.0.217", features = ["derive"] }
serde_json = "1.0.134"

# AST parsing for smart chunking
tree-sitter = "0.20.10"
tree-sitter-rust = "0.20.4"

[dev-dependencies]
# For creating temporary directories in tests
tempfile = "3.14.0"

# Windows-specific optimizations
[profile.dev]
opt-level = 1  # Faster compilation on Windows

[profile.release] 
debug = true   # Keep debug symbols for troubleshooting
lto = true     # Link-time optimization
```

**Why these dependencies?**
- `tantivy = "0.22.0"` - The search engine (specific version for stability)
- `tokio` - Async runtime for performance  
- `anyhow` - Easy error handling
- `tree-sitter-*` - For AST-based code chunking

### Step 3: Create Basic Source Structure (1 minute)
```bash
# Create source directory
mkdir src

# Create main.rs placeholder
echo 'fn main() { println!("Phase 1: Tantivy Search System Starting..."); }' > src/main.rs

# Create lib.rs for library functions
echo '//! Phase 1: Tantivy Text Search with Smart Chunking' > src/lib.rs
```

### Step 4: Create .gitignore (1 minute)
Create `C:/code/LLMKG/vectors/tantivy_search/.gitignore`:
```
/target
Cargo.lock
*.pdb
/indexes/*
!indexes/.gitkeep
```

## Verification Steps (2 minutes)

### Verify 1: Files exist
```bash
ls -la
# Should show: Cargo.toml, src/, .gitignore
ls src/
# Should show: main.rs, lib.rs
```

### Verify 2: Dependencies resolve
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 3: Basic run works
```bash
cargo run
```
**Expected output:**
```
Phase 1: Tantivy Search System Starting...
```

## Success Validation Checklist
- [ ] Directory exists: `C:/code/LLMKG/vectors/tantivy_search/`
- [ ] File exists: `Cargo.toml` with exact dependencies listed above
- [ ] File exists: `src/main.rs` and `src/lib.rs`
- [ ] Command `cargo check` completes without errors
- [ ] Command `cargo run` prints the expected message
- [ ] All dependency versions exactly match the specification

## If This Task Fails

**Error: "command not found: cargo"**
- Solution: Install Rust toolchain from https://rustup.rs/

**Error: "failed to resolve dependencies"**  
- Solution: Check internet connection, verify Cargo.toml syntax

**Error: "permission denied"**
- Solution: Ensure write permissions to C:/code/ directory

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/Cargo.toml** - Project manifest with all Phase 1 dependencies
2. **C:/code/LLMKG/vectors/tantivy_search/src/main.rs** - Basic main function  
3. **C:/code/LLMKG/vectors/tantivy_search/src/lib.rs** - Library entry point

**Next Task (Task 02)** will use these files to create the module structure (schema.rs, chunker.rs, indexer.rs, search.rs).

## Context for Task 02
Task 02 will create the core module files that implement the Tantivy schema, smart chunker, document indexer, and search engine. The Cargo.toml created here provides all necessary dependencies for those implementations.