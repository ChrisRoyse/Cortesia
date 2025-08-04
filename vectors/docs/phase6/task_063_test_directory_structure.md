# Task 063: Set up Test Directory Structure

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates a standardized directory structure for organizing test files, temporary directories, output directories, and establishing naming conventions for the validation system.

## Project Structure
```
tests/                          <- Create this directory structure
├── fixtures/                   <- Test data files
│   ├── ground_truth/          <- Ground truth datasets
│   ├── synthetic/             <- Generated test files
│   ├── large_files/           <- Large file test cases
│   └── unicode/               <- Unicode test files
├── integration/               <- Integration tests
├── performance/               <- Performance benchmarks
├── security/                  <- Security validation tests
├── temp/                      <- Temporary test files (auto-cleanup)
├── output/                    <- Test output and reports
└── utils/                     <- Test utilities
```

## Task Description
Create a comprehensive directory structure for organizing all validation test files, data, outputs, and utilities. Establish clear naming conventions and organization patterns that support both automated testing and manual validation.

## Requirements
1. Create hierarchical directory structure for different test types
2. Establish naming conventions for test files and data
3. Set up temporary directories with auto-cleanup
4. Configure output directories for reports and results
5. Create utility directories for shared test code

## Expected File Content/Code Structure

### Directory Structure Setup Script
Create this as `setup_test_directories.rs` in the project root:

```rust
use std::fs;
use std::path::Path;
use anyhow::Result;

/// Creates the complete test directory structure for LLMKG validation
pub fn setup_test_directories() -> Result<()> {
    let directories = [
        // Root test directories
        "tests",
        "tests/fixtures",
        "tests/integration", 
        "tests/performance",
        "tests/security",
        "tests/temp",
        "tests/output",
        "tests/utils",
        
        // Fixture subdirectories
        "tests/fixtures/ground_truth",
        "tests/fixtures/synthetic", 
        "tests/fixtures/large_files",
        "tests/fixtures/unicode",
        "tests/fixtures/special_chars",
        "tests/fixtures/boolean_logic",
        "tests/fixtures/proximity",
        "tests/fixtures/wildcards",
        "tests/fixtures/regex",
        "tests/fixtures/vectors",
        "tests/fixtures/hybrid",
        
        // Integration test categories
        "tests/integration/correctness",
        "tests/integration/search_modes",
        "tests/integration/query_types",
        "tests/integration/error_handling",
        "tests/integration/windows_specific",
        
        // Performance test categories
        "tests/performance/latency",
        "tests/performance/throughput", 
        "tests/performance/concurrent",
        "tests/performance/stress",
        "tests/performance/memory",
        "tests/performance/baseline",
        
        // Security test categories
        "tests/security/malicious_queries",
        "tests/security/injection_attacks",
        "tests/security/path_traversal",
        "tests/security/resource_exhaustion",
        
        // Output directories
        "tests/output/reports",
        "tests/output/benchmarks",
        "tests/output/logs",
        "tests/output/artifacts",
        "tests/output/screenshots", // For debugging
        
        // Temporary directories (will be cleaned up)
        "tests/temp/indexing",
        "tests/temp/search",
        "tests/temp/validation",
        "tests/temp/downloads",
        
        // Utility directories
        "tests/utils/generators",
        "tests/utils/validators",
        "tests/utils/helpers",
        "tests/utils/mocks",
    ];
    
    for dir in &directories {
        fs::create_dir_all(dir)?;
        println!("Created directory: {}", dir);
    }
    
    // Create .gitkeep files for empty directories
    let gitkeep_dirs = [
        "tests/temp",
        "tests/output/logs",
        "tests/fixtures/large_files",
    ];
    
    for dir in &gitkeep_dirs {
        let gitkeep_path = Path::new(dir).join(".gitkeep");
        fs::write(gitkeep_path, "")?;
    }
    
    // Create .gitignore for temp directories
    let gitignore_content = r#"# Temporary test files
/tests/temp/*
!/tests/temp/.gitkeep

# Test outputs (keep reports but ignore temporary artifacts)
/tests/output/logs/*
!/tests/output/logs/.gitkeep
/tests/output/artifacts/*
!/tests/output/artifacts/.gitkeep

# Large test files (download on demand)
/tests/fixtures/large_files/*
!/tests/fixtures/large_files/.gitkeep
!/tests/fixtures/large_files/README.md

# OS-specific files
.DS_Store
Thumbs.db
desktop.ini
"#;
    
    fs::write("tests/.gitignore", gitignore_content)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    
    #[test]
    fn test_directory_structure_creation() -> Result<()> {
        setup_test_directories()?;
        
        // Verify key directories exist
        assert!(Path::new("tests/fixtures/ground_truth").exists());
        assert!(Path::new("tests/integration/correctness").exists());
        assert!(Path::new("tests/performance/latency").exists());
        assert!(Path::new("tests/security/malicious_queries").exists());
        assert!(Path::new("tests/output/reports").exists());
        assert!(Path::new("tests/temp/validation").exists());
        assert!(Path::new("tests/utils/generators").exists());
        
        Ok(())
    }
}
```

### Test File Naming Conventions (`tests/utils/naming_conventions.md`)
```markdown
# LLMKG Test File Naming Conventions

## Directory Organization

### tests/fixtures/
Test data files organized by query type and complexity:

**Pattern**: `{category}/{difficulty}_{id}_{description}.{ext}`

Examples:
- `ground_truth/basic_001_simple_keyword.json`
- `boolean_logic/advanced_042_nested_and_or.json`
- `unicode/extreme_007_mixed_scripts.txt`
- `large_files/stress_001_10mb_rust_code.rs`

### tests/integration/
Integration test files:

**Pattern**: `test_{category}_{specific_feature}.rs`

Examples:
- `test_correctness_special_characters.rs`
- `test_search_modes_boolean_logic.rs` 
- `test_query_types_proximity_search.rs`
- `test_error_handling_malformed_queries.rs`

### tests/performance/
Performance benchmark files:

**Pattern**: `bench_{metric}_{scenario}.rs`

Examples:
- `bench_latency_single_query.rs`
- `bench_throughput_concurrent_users.rs`
- `bench_memory_large_index.rs`
- `bench_stress_sustained_load.rs`

### tests/security/
Security validation files:

**Pattern**: `sec_{attack_type}_{vector}.rs`

Examples:
- `sec_injection_sql_like.rs`
- `sec_traversal_path_escape.rs`
- `sec_exhaustion_memory_bomb.rs`
- `sec_malicious_regex_patterns.rs`

## File Content Standards

### Test Data Files (.json, .txt, .rs)
- Use descriptive filenames indicating content and purpose
- Include metadata header with test case description
- Keep file sizes reasonable (<10MB unless testing large files)
- Use UTF-8 encoding consistently

### Test Code Files (.rs)
- Start with comprehensive module documentation
- Include test case descriptions and expected outcomes
- Use descriptive test function names
- Group related tests in the same file

### Ground Truth Files (.json)
- Follow schema: `{query_type}_{difficulty}_{unique_id}.json`
- Include expected results and validation criteria
- Document edge cases and expected behaviors
- Maintain backwards compatibility

## Directory Cleanup Rules

### Automatic Cleanup (tests/temp/)
- All files deleted after each test run
- Subdirectories recreated as needed
- No persistent data storage

### Manual Cleanup (tests/output/)
- Log files rotated weekly
- Benchmark results archived monthly
- Report files kept indefinitely
- Screenshots deleted after 30 days

### Version Control (.gitignore)
- Track test structure and fixtures
- Ignore temporary files and outputs
- Include essential test data
- Exclude generated large files
```

### Test Directory README (`tests/README.md`)
```markdown
# LLMKG Validation Test Suite

This directory contains the comprehensive validation test suite for the LLMKG Vector Indexing System.

## Quick Start

1. **Setup directories**: Run `cargo run --bin setup-test-dirs`
2. **Generate test data**: Run `cargo test --test generate_test_data`
3. **Run validation**: Run `cargo test --test validation_suite`
4. **View reports**: Check `tests/output/reports/`

## Directory Structure

### `fixtures/` - Test Data
- **ground_truth/**: Curated test cases with known correct results
- **synthetic/**: Generated test files for specific scenarios
- **large_files/**: Large files for scale testing (10MB+ each)
- **unicode/**: International text samples and edge cases
- **special_chars/**: Special character and symbol test cases

### `integration/` - Integration Tests
- **correctness/**: Accuracy validation tests
- **search_modes/**: Different search mode validation
- **query_types/**: Specific query type testing
- **error_handling/**: Error condition testing
- **windows_specific/**: Windows compatibility tests

### `performance/` - Performance Benchmarks  
- **latency/**: Response time measurements
- **throughput/**: Queries per second testing
- **concurrent/**: Multi-user load testing
- **stress/**: System limit testing
- **memory/**: Memory usage profiling

### `security/` - Security Validation
- **malicious_queries/**: Attack vector testing
- **injection_attacks/**: Code injection prevention
- **path_traversal/**: File system security
- **resource_exhaustion/**: DoS protection

### `temp/` - Temporary Files
- Automatically cleaned up after tests
- Used for intermediate processing
- Never committed to version control

### `output/` - Test Results
- **reports/**: Validation reports (HTML, JSON, Markdown)
- **benchmarks/**: Performance benchmark results
- **logs/**: Detailed test execution logs
- **artifacts/**: Debug artifacts and screenshots

### `utils/` - Test Utilities
- **generators/**: Test data generation utilities
- **validators/**: Custom validation functions
- **helpers/**: Common test helper functions
- **mocks/**: Mock implementations for testing

## Running Tests

### Full Validation Suite
```bash
cargo test --test validation_suite --release
```

### Performance Benchmarks
```bash
cargo bench --bench search_performance
```

### Security Tests
```bash
cargo test --test security_validation
```

### Windows-Specific Tests
```bash
cargo test --test windows_compatibility
```

## Test Data Management

### Generating Test Data
```bash
cargo run --bin generate-test-data -- --category all --count 1000
```

### Cleaning Test Data
```bash
cargo run --bin cleanup-test-data -- --temp-only
```

### Validating Test Data
```bash
cargo test --test validate_test_data
```

## Contributing

1. Follow naming conventions in `utils/naming_conventions.md`
2. Add new test categories to the appropriate subdirectory
3. Update this README when adding new test types
4. Ensure all tests pass on Windows and Unix systems
5. Include comprehensive test documentation

## Maintenance

- Test data is regenerated weekly via CI/CD
- Performance baselines updated monthly
- Security test vectors updated when new threats identified
- Directory structure validated on every commit
```

### Automated Cleanup Utility (`tests/utils/cleanup.rs`)
```rust
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};
use anyhow::Result;
use tracing::{info, warn, error};

/// Automated cleanup utility for test directories
pub struct TestCleanup {
    temp_dir: String,
    output_dir: String,
    max_log_age_days: u64,
    max_artifact_age_days: u64,
}

impl TestCleanup {
    pub fn new() -> Self {
        Self {
            temp_dir: "tests/temp".to_string(),
            output_dir: "tests/output".to_string(),
            max_log_age_days: 7,
            max_artifact_age_days: 30,
        }
    }
    
    /// Clean all temporary test files
    pub fn clean_temp_files(&self) -> Result<()> {
        info!("Cleaning temporary test files in {}", self.temp_dir);
        
        if Path::new(&self.temp_dir).exists() {
            fs::remove_dir_all(&self.temp_dir)?;
            fs::create_dir_all(&self.temp_dir)?;
            info!("Cleaned and recreated temp directory");
        }
        
        Ok(())
    }
    
    /// Clean old log files
    pub fn clean_old_logs(&self) -> Result<()> {
        let log_dir = format!("{}/logs", self.output_dir);
        self.clean_old_files(&log_dir, self.max_log_age_days)
    }
    
    /// Clean old artifacts
    pub fn clean_old_artifacts(&self) -> Result<()> {
        let artifact_dir = format!("{}/artifacts", self.output_dir);
        self.clean_old_files(&artifact_dir, self.max_artifact_age_days)
    }
    
    /// Clean files older than specified days
    fn clean_old_files(&self, dir: &str, max_age_days: u64) -> Result<()> {
        if !Path::new(dir).exists() {
            return Ok(());
        }
        
        let cutoff_time = SystemTime::now() - Duration::from_secs(max_age_days * 24 * 60 * 60);
        let mut cleaned_count = 0;
        
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                let metadata = fs::metadata(&path)?;
                if let Ok(modified) = metadata.modified() {
                    if modified < cutoff_time {
                        match fs::remove_file(&path) {
                            Ok(_) => {
                                cleaned_count += 1;
                                info!("Removed old file: {}", path.display());
                            }
                            Err(e) => {
                                warn!("Failed to remove file {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }
        
        info!("Cleaned {} old files from {}", cleaned_count, dir);
        Ok(())
    }
    
    /// Full cleanup routine
    pub fn full_cleanup(&self) -> Result<()> {
        info!("Starting full test directory cleanup");
        
        self.clean_temp_files()?;
        self.clean_old_logs()?;
        self.clean_old_artifacts()?;
        
        info!("Test directory cleanup completed");
        Ok(())
    }
}

/// Cleanup hook for test teardown
pub fn cleanup_after_tests() -> Result<()> {
    let cleanup = TestCleanup::new();
    cleanup.clean_temp_files()
}

/// Cleanup hook for CI/CD
pub fn cleanup_for_ci() -> Result<()> {
    let cleanup = TestCleanup::new();
    cleanup.full_cleanup()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_cleanup_temp_files() -> Result<()> {
        let cleanup = TestCleanup::new();
        cleanup.clean_temp_files()?;
        
        // Verify temp directory exists and is empty
        let temp_path = Path::new("tests/temp");
        assert!(temp_path.exists());
        assert!(temp_path.is_dir());
        
        Ok(())
    }
    
    #[test]
    fn test_cleanup_old_files() -> Result<()> {
        let temp_dir = tempdir()?;
        let cleanup = TestCleanup::new();
        
        // This would test file age-based cleanup
        // Implementation depends on creating test files with different ages
        
        Ok(())
    }
}
```

## Success Criteria
- All directories are created successfully with proper permissions
- Naming conventions are documented and followed consistently  
- Temporary directories are properly isolated and auto-cleaned
- Output directories preserve important results while managing disk space
- Test utilities support the complete validation workflow
- Directory structure supports parallel test execution
- Windows path handling works correctly for all directory types

## Time Limit
10 minutes maximum