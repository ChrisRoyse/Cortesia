# Micro-Task 020: Setup Testing Framework

## Objective
Configure the testing framework and create test utilities for consistent testing across all crates.

## Context
A robust testing framework is essential for ensuring code quality and preventing regressions. This task sets up testing conventions, utilities, and configuration that will be used across all crates.

## Prerequisites
- Task 019 completed (Documentation structure created)
- Development tools installed
- Testing directories created

## Time Estimate
9 minutes

## Instructions
1. Create `tests/common/mod.rs` for shared test utilities:
   ```rust
   //! Common test utilities for the vector search system
   
   use std::path::PathBuf;
   use std::env;
   
   /// Get test data directory path
   pub fn test_data_dir() -> PathBuf {
       let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
       path.push("data");
       path.push("test_files");
       path
   }
   
   /// Create temporary test directory
   pub fn create_temp_dir(name: &str) -> PathBuf {
       let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
       path.push("data");
       path.push("temp");
       path.push(name);
       std::fs::create_dir_all(&path).expect("Failed to create temp dir");
       path
   }
   
   /// Clean up test directory
   pub fn cleanup_test_dir(path: &PathBuf) {
       if path.exists() {
           std::fs::remove_dir_all(path).ok();
       }
   }
   ```
2. Create integration test template `tests/integration_test_template.rs`:
   ```rust
   //! Template for integration tests
   
   mod common;
   
   #[cfg(test)]
   mod tests {
       use super::common::*;
       
       #[test]
       fn test_template() {
           // This is a template - replace with actual tests
           let test_dir = create_temp_dir("template_test");
           
           // Test logic here
           assert!(test_dir.exists());
           
           // Cleanup
           cleanup_test_dir(&test_dir);
       }
   }
   ```
3. Test the framework: `cargo test --test integration_test_template`
4. Commit testing framework: `git add tests/ && git commit -m "Setup testing framework"`

## Expected Output
- Common test utilities available for all crates
- Integration test template created
- Testing framework validated
- Framework committed to version control

## Success Criteria
- [ ] `tests/common/mod.rs` created with utilities
- [ ] Integration test template works
- [ ] Test framework compiles and runs
- [ ] Testing framework committed to Git

## Next Task
task_021_configure_benchmark_framework.md