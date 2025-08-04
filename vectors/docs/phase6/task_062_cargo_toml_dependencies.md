# Task 062: Create Cargo.toml with Dependencies

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates a production-ready Cargo.toml configuration with all necessary dependencies for the validation system including search engines, testing frameworks, and Windows-specific libraries.

## Project Structure
```
Cargo.toml  <- Update/create this file (in project root)
```

## Task Description
Create or update the Cargo.toml file with all dependencies required for the validation system. Include production dependencies for search engines, development dependencies for testing, and Windows-specific configurations.

## Requirements
1. Include all core search engine dependencies (Tantivy, LanceDB)
2. Add comprehensive testing and benchmarking dependencies
3. Configure Windows-specific dependencies and features
4. Set up development dependencies for validation tools
5. Include proper feature flags for optional functionality

## Expected File Content/Code Structure
```toml
# LLMKG Vector Indexing System - Production Configuration
# 
# This Cargo.toml includes all dependencies for:
# - Core vector indexing (Tantivy, LanceDB)
# - Validation and testing framework
# - Performance benchmarking
# - Windows-specific functionality
# - Development and debugging tools

[package]
name = "llmkg-validation"
version = "1.0.0"
edition = "2021"
description = "High-performance vector indexing system with comprehensive validation"
license = "MIT OR Apache-2.0"
repository = "https://github.com/your-org/llmkg"
readme = "README.md"
keywords = ["search", "vector", "indexing", "validation", "rust"]
categories = ["text-processing", "database-implementations"]

[dependencies]
# Core search engines
tantivy = { version = "0.19", features = ["mmap", "quickwit"] }
lancedb = { version = "0.4", features = ["s3"] }

# Async runtime and utilities
tokio = { version = "1.35", features = [
    "full",
    "rt-multi-thread",
    "macros",
    "fs",
    "process",
    "signal",
    "time"
] }
futures = "0.3"
async-trait = "0.1"

# Serialization and data handling
serde = { version = "1.0", features = ["derive", "rc"] }
serde_json = "1.0"
toml = "0.8"
csv = "1.3"

# Text processing and NLP
regex = "1.10"
unicode-normalization = "0.1"
unicode-segmentation = "1.10"

# Vector operations and ML
ndarray = { version = "0.15", features = ["serde"] }
candle-core = "0.3"
candle-nn = "0.3"
hf-hub = "0.3"
tokenizers = "0.15"

# Performance and concurrency
rayon = "1.8"
crossbeam = "0.8"
dashmap = "5.5"
parking_lot = "0.12"

# Error handling and logging
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json", "fmt"] }
tracing-appender = "0.2"

# System utilities
clap = { version = "4.4", features = ["derive", "env"] }
dirs = "5.0"
walkdir = "2.4"
notify = "6.1"
tempfile = "3.8"

# Network and HTTP
reqwest = { version = "0.11", features = ["json", "stream"] }
url = "2.5"

# Compression and encoding
flate2 = "1.0"
tar = "0.4"
base64 = "0.21"

# Time and date handling
chrono = { version = "0.4", features = ["serde"] }

# Memory mapping and file I/O
memmap2 = "0.9"

# Windows-specific dependencies
[target.'cfg(windows)'.dependencies]
winapi = { version = "0.3", features = [
    "winbase",
    "fileapi",
    "handleapi",
    "processenv",
    "synchapi",
    "winnt",
    "winerror",
    "consoleapi",
    "wincon",
    "profileapi"
] }
windows = { version = "0.52", features = [
    "Win32_Foundation",
    "Win32_Storage_FileSystem",
    "Win32_System_Performance",
    "Win32_System_SystemInformation",
    "Win32_System_Diagnostics_ToolHelp",
    "Win32_System_Threading",
    "Win32_Security"
] }

# Unix-specific dependencies (for cross-platform compatibility)
[target.'cfg(unix)'.dependencies]
libc = "0.2"

[dev-dependencies]
# Testing frameworks
criterion = { version = "0.5", features = ["html_reports", "cargo_bench_support"] }
proptest = "1.4"
quickcheck = "1.0"
quickcheck_macros = "1.0"

# Test utilities
pretty_assertions = "1.4"
serial_test = "3.0"
rstest = "0.18"
mockall = "0.12"

# Benchmarking and profiling
pprof = { version = "0.13", features = ["criterion", "protobuf-codec"] }
dhat = "0.3"

# Development tools
cargo-nextest = "0.9"
tracing-test = "0.2"

[features]
default = ["validation", "benchmarks"]

# Core features
validation = []
benchmarks = ["criterion"]
profiling = ["pprof", "dhat"]

# Search engine features
tantivy-full = ["tantivy/mmap", "tantivy/quickwit"]
lancedb-full = ["lancedb/s3"]

# Platform-specific features
windows-full = []
unix-full = []

# Development features
dev-tools = ["cargo-nextest", "tracing-test"]
testing = ["proptest", "quickcheck", "mockall"]

# Experimental features
experimental = []

[profile.release]
# Production optimizations
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = "symbols"
debug = false

[profile.bench]
# Benchmark optimizations
opt-level = 3
lto = true
codegen-units = 1
debug = true
panic = "abort"

[profile.dev]
# Development configuration
opt-level = 0
debug = true
panic = "unwind"
overflow-checks = true

[profile.test]
# Test configuration
opt-level = 1
debug = true
panic = "unwind"
overflow-checks = true

# Workspace configuration
[workspace]
members = [
    ".",
    "validation",
    "benchmarks"
]

# Binary targets
[[bin]]
name = "llmkg-validate"
path = "src/bin/validate.rs"
required-features = ["validation"]

[[bin]]
name = "llmkg-benchmark"
path = "src/bin/benchmark.rs"
required-features = ["benchmarks"]

# Benchmark targets
[[bench]]
name = "search_performance"
harness = false
required-features = ["benchmarks"]

[[bench]]
name = "indexing_performance"
harness = false
required-features = ["benchmarks"]

[[bench]]
name = "validation_performance"
harness = false
required-features = ["benchmarks"]

# Example targets
[[example]]
name = "basic_validation"
required-features = ["validation"]

[[example]]
name = "performance_benchmark"
required-features = ["benchmarks"]

[[example]]
name = "windows_compatibility"
required-features = ["windows-full"]

# Package metadata
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[package.metadata.release]
pre-release-replacements = [
    { file = "CHANGELOG.md", search = "Unreleased", replace = "{{version}}" },
    { file = "CHANGELOG.md", search = "\\.\\.\\.HEAD", replace = "...{{tag_name}}" },
    { file = "CHANGELOG.md", search = "ReleaseDate", replace = "{{date}}" },
]
```

## Additional Configuration Notes

### Dependency Rationale:
- **tantivy**: Full-text search engine with mmap support for performance
- **lancedb**: Vector database with S3 support for cloud storage
- **tokio**: Async runtime with full feature set for concurrent operations
- **rayon**: Data parallelism for CPU-intensive operations
- **tracing**: Structured logging with JSON output for production monitoring
- **criterion**: Statistical benchmarking framework
- **winapi/windows**: Windows-specific system APIs for file handling and performance monitoring

### Feature Flags:
- `validation`: Core validation functionality (default)
- `benchmarks`: Performance benchmarking tools (default)
- `profiling`: Memory and CPU profiling tools
- `windows-full`: All Windows-specific features
- `experimental`: Cutting-edge features under development

### Build Profiles:
- **release**: Maximum optimization for production deployment
- **bench**: Optimized with debug symbols for accurate benchmarking
- **dev**: Fast compilation for development iteration
- **test**: Balanced optimization for reliable test execution

## Success Criteria
- Cargo.toml compiles successfully with `cargo check`
- All dependencies resolve without conflicts
- Windows-specific dependencies are properly configured
- Feature flags work correctly with `cargo build --features <flag>`
- Benchmark targets compile with `cargo bench --no-run`
- Release profile produces optimized binaries
- Development dependencies support comprehensive testing

## Time Limit
10 minutes maximum