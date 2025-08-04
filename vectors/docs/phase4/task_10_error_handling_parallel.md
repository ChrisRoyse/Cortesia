# Task 10: Error Handling Parallel

## Context
You are implementing Phase 4 of a vector indexing system. This task focuses on implementing robust error handling for parallel operations. After validating thread safety, you now need to ensure that the parallel indexer handles errors gracefully across multiple threads, provides meaningful error messages, and maintains system stability even when individual operations fail.

## Current State
- `src/parallel.rs` exists with complete parallel indexing implementation
- Thread safety validation tests are passing
- Basic error handling exists but needs comprehensive parallel-specific handling
- Rayon parallel processing is working correctly

## Task Objective
Implement comprehensive error handling for parallel processing that ensures graceful degradation, proper error propagation, meaningful error messages, and system stability when operations fail across multiple threads.

## Implementation Requirements

### 1. Add comprehensive parallel error handling test
Add this test to the test module in `src/parallel.rs`:
```rust
#[test]
fn test_parallel_error_handling_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("error_handling_test");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test 1: Non-existent directory
    let non_existent = temp_dir.path().join("does_not_exist");
    match parallel_indexer.index_directory_parallel(&non_existent) {
        Err(e) => {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("does_not_exist") || error_msg.contains("No such file"),
                   "Error message should mention the missing directory: {}", error_msg);
            println!("✓ Non-existent directory error handled: {}", error_msg);
        }
        Ok(_) => panic!("Expected error for non-existent directory"),
    }
    
    // Test 2: File instead of directory
    let file_path = temp_dir.path().join("not_a_directory.txt");
    std::fs::write(&file_path, "This is a file, not a directory")?;
    match parallel_indexer.index_directory_parallel(&file_path) {
        Err(e) => {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("not_a_directory") || error_msg.contains("not a directory"),
                   "Error message should indicate it's not a directory: {}", error_msg);
            println!("✓ File-as-directory error handled: {}", error_msg);
        }
        Ok(_) => panic!("Expected error when treating file as directory"),
    }
    
    // Test 3: Empty directory (should succeed but process no files)
    let empty_dir = temp_dir.path().join("empty");
    std::fs::create_dir_all(&empty_dir)?;
    let stats = parallel_indexer.index_directory_parallel(&empty_dir)?;
    assert_eq!(stats.files_processed, 0, "Empty directory should process 0 files");
    println!("✓ Empty directory handled gracefully");
    
    // Test 4: Directory with mixed accessible/inaccessible content
    let mixed_dir = temp_dir.path().join("mixed_access");
    std::fs::create_dir_all(&mixed_dir)?;
    
    // Create some normal files
    std::fs::write(mixed_dir.join("normal1.txt"), "Normal content 1")?;
    std::fs::write(mixed_dir.join("normal2.rs"), "fn main() { println!(\"Hello\"); }")?;
    
    // Create a subdirectory with restricted permissions (Unix only)
    #[cfg(unix)]
    {
        let restricted_subdir = mixed_dir.join("restricted");
        std::fs::create_dir_all(&restricted_subdir)?;
        std::fs::write(restricted_subdir.join("hidden.txt"), "Hidden content")?;
        
        // Remove read permissions
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&restricted_subdir)?.permissions();
        perms.set_mode(0o000);
        std::fs::set_permissions(&restricted_subdir, perms)?;
        
        // This should process the accessible files and handle the restricted directory gracefully
        let stats = parallel_indexer.index_directory_parallel(&mixed_dir)?;
        assert!(stats.files_processed >= 2, "Should process at least the accessible files");
        println!("✓ Mixed accessibility handled: {} files processed", stats.files_processed);
        
        // Restore permissions for cleanup
        let mut perms = std::fs::metadata(&restricted_subdir)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&restricted_subdir, perms)?;
    }
    
    #[cfg(not(unix))]
    {
        // On non-Unix systems, just test normal processing
        let stats = parallel_indexer.index_directory_parallel(&mixed_dir)?;
        assert!(stats.files_processed >= 2, "Should process the normal files");
        println!("✓ Normal processing on non-Unix system: {} files", stats.files_processed);
    }
    
    Ok(())
}
```

### 2. Add concurrent error handling test
Add this test to validate error handling across multiple threads:
```rust
#[test]
fn test_concurrent_error_scenarios() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("concurrent_errors");
    let parallel_indexer = Arc::new(ParallelIndexer::new(&index_path)?);
    
    // Create a mix of valid and invalid paths
    let mut test_paths = Vec::new();
    
    // Valid paths
    for i in 0..3 {
        let valid_path = temp_dir.path().join(format!("valid_{}", i));
        create_test_project(&valid_path, 10)?;
        test_paths.push((valid_path, true)); // (path, should_succeed)
    }
    
    // Invalid paths
    test_paths.push((temp_dir.path().join("nonexistent_1"), false));
    test_paths.push((temp_dir.path().join("nonexistent_2"), false));
    
    // File paths (should fail)
    let file_path1 = temp_dir.path().join("file1.txt");
    let file_path2 = temp_dir.path().join("file2.txt");
    std::fs::write(&file_path1, "content1")?;
    std::fs::write(&file_path2, "content2")?;
    test_paths.push((file_path1, false));
    test_paths.push((file_path2, false));
    
    // Launch concurrent operations with mixed success/failure scenarios
    let handles: Vec<_> = test_paths.into_iter().enumerate()
        .map(|(i, (path, should_succeed))| {
            let indexer = Arc::clone(&parallel_indexer);
            std::thread::spawn(move || -> (usize, bool, Result<IndexingStats>) {
                let result = indexer.index_directory_parallel(&path);
                (i, should_succeed, result)
            })
        })
        .collect();
    
    // Collect results and validate error handling
    let mut success_count = 0;
    let mut expected_success_count = 0;
    let mut error_count = 0;
    
    for handle in handles {
        let (thread_id, should_succeed, result) = handle.join().unwrap();
        
        if should_succeed {
            expected_success_count += 1;
            match result {
                Ok(stats) => {
                    success_count += 1;
                    println!("Thread {} succeeded: {} files", thread_id, stats.files_processed);
                }
                Err(e) => {
                    panic!("Thread {} was expected to succeed but failed: {}", thread_id, e);
                }
            }
        } else {
            match result {
                Ok(_) => {
                    panic!("Thread {} was expected to fail but succeeded", thread_id);
                }
                Err(e) => {
                    error_count += 1;
                    println!("Thread {} failed as expected: {}", thread_id, e);
                }
            }
        }
    }
    
    // Validate that successes and failures occurred as expected
    assert_eq!(success_count, expected_success_count, 
              "Expected {} successes, got {}", expected_success_count, success_count);
    assert!(error_count >= 4, "Expected at least 4 errors, got {}", error_count);
    
    println!("Concurrent error handling test: {} successes, {} errors", 
            success_count, error_count);
    
    Ok(())
}
```

### 3. Add error recovery and continuation test
Add this test to validate error recovery behavior:
```rust
#[test]
fn test_error_recovery_and_continuation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("recovery_test");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create a directory with mixed content including problematic files
    let test_project = temp_dir.path().join("recovery_project");
    std::fs::create_dir_all(&test_project)?;
    
    // Create normal, indexable files
    for i in 0..10 {
        let file_path = test_project.join(format!("normal_{}.rs", i));
        let content = format!("// Normal Rust file {}\nfn function_{}() {{}}\n", i, i);
        std::fs::write(file_path, content)?;
    }
    
    // Create some problematic subdirectories and files
    let problematic_subdir = test_project.join("problematic");
    std::fs::create_dir_all(&problematic_subdir)?;
    
    // Add normal files in the problematic subdirectory
    for i in 0..5 {
        let file_path = problematic_subdir.join(format!("subfile_{}.md", i));
        std::fs::write(file_path, format!("# Document {}\nContent here.", i))?;
    }
    
    // Create files with unusual characteristics
    std::fs::write(test_project.join("empty.txt"), "")?; // Empty file
    std::fs::write(test_project.join("large.log"), "x".repeat(1024 * 100))?; // Large file
    std::fs::write(test_project.join("unicode_文件.txt"), "Unicode content")?; // Unicode filename
    
    // Create binary-like files that should be filtered out
    std::fs::write(test_project.join("binary.exe"), b"\x00\x01\x02\x03")?;
    std::fs::write(test_project.join("image.jpg"), b"\xFF\xD8\xFF")?;
    
    // Process the directory - should continue despite individual file issues
    let stats = parallel_indexer.index_directory_parallel(&test_project)?;
    
    // Validate that processing continued and handled errors gracefully
    assert!(stats.files_processed > 0, "Should have processed some files despite errors");
    
    // Should process at least the Rust and Markdown files
    assert!(stats.files_processed >= 10, 
           "Should process at least 10 files, got {}", stats.files_processed);
    
    assert!(stats.total_size > 0, "Should have processed some content");
    
    println!("Error recovery test: {} files processed, {} bytes total", 
            stats.files_processed, stats.total_size);
    
    // Verify that the indexer can still function after encountering problems
    let recovery_dir = temp_dir.path().join("after_recovery");
    create_test_project(&recovery_dir, 5)?;
    
    let recovery_stats = parallel_indexer.index_directory_parallel(&recovery_dir)?;
    assert!(recovery_stats.files_processed >= 5, 
           "Indexer should still work after error recovery");
    
    println!("Post-recovery test: {} files processed", recovery_stats.files_processed);
    
    Ok(())
}
```

### 4. Add error propagation and aggregation test
Add this test to validate proper error reporting:
```rust
#[test]
fn test_error_propagation_and_reporting() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("error_propagation");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test detailed error messages for different failure modes
    struct ErrorTest {
        name: &'static str,
        setup: Box<dyn Fn(&Path) -> Result<PathBuf>>,
        expected_error_contains: &'static str,
    }
    
    let error_tests = vec![
        ErrorTest {
            name: "Invalid index path",
            setup: Box::new(|_| {
                // Try to create indexer with invalid path
                Ok(Path::new("/dev/null/invalid/path").to_path_buf())
            }),
            expected_error_contains: "path",
        },
        
        ErrorTest {
            name: "Permission denied directory",
            setup: Box::new(|base_path| {
                let restricted_path = base_path.join("permission_denied");
                std::fs::create_dir_all(&restricted_path)?;
                
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = std::fs::metadata(&restricted_path)?.permissions();
                    perms.set_mode(0o000);
                    std::fs::set_permissions(&restricted_path, perms)?;
                }
                
                Ok(restricted_path)
            }),
            expected_error_contains: "permission",
        },
    ];
    
    for test in error_tests {
        println!("Testing error case: {}", test.name);
        
        match test.setup(temp_dir.path()) {
            Ok(test_path) => {
                let result = parallel_indexer.index_directory_parallel(&test_path);
                
                match result {
                    Err(e) => {
                        let error_msg = format!("{:?}", e); // Use Debug format for full error context
                        println!("  Error message: {}", error_msg);
                        
                        // Verify error message contains expected information
                        let error_msg_lower = error_msg.to_lowercase();
                        if !error_msg_lower.contains(test.expected_error_contains) {
                            println!("  Warning: Error message may not contain expected info '{}': {}", 
                                   test.expected_error_contains, error_msg);
                        }
                    }
                    Ok(_) => {
                        println!("  Warning: Expected error for {} but operation succeeded", test.name);
                    }
                }
                
                // Clean up permissions if needed
                #[cfg(unix)]
                if test.name.contains("permission") {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(metadata) = std::fs::metadata(&test_path) {
                        let mut perms = metadata.permissions();
                        perms.set_mode(0o755);
                        let _ = std::fs::set_permissions(&test_path, perms);
                    }
                }
            }
            Err(setup_error) => {
                println!("  Setup failed for {}: {}", test.name, setup_error);
            }
        }
    }
    
    // Test error context preservation
    let context_test_path = temp_dir.path().join("context_test_nonexistent");
    let result = parallel_indexer.index_directory_parallel(&context_test_path);
    
    match result {
        Err(e) => {
            let error_chain = format!("{:?}", e);
            println!("Error context test - full error chain: {}", error_chain);
            
            // Error should contain contextual information
            assert!(error_chain.contains("context_test_nonexistent"), 
                   "Error should contain the problematic path");
        }
        Ok(_) => {
            panic!("Expected error for non-existent path in context test");
        }
    }
    
    println!("Error propagation and reporting test completed");
    
    Ok(())
}
```

### 5. Add partial failure handling test
Add this test to validate graceful handling of partial failures:
```rust
#[test]
fn test_partial_failure_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("partial_failure");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create a complex directory structure with mixed success/failure scenarios
    let test_project = temp_dir.path().join("partial_test");
    std::fs::create_dir_all(&test_project)?;
    
    // Create successful processing candidates
    let success_files = [
        ("src/main.rs", "fn main() { println!(\"Hello\"); }"),
        ("src/lib.rs", "pub mod utils;"),
        ("docs/README.md", "# Project Documentation"),
        ("config/settings.toml", "[database]\nurl = \"localhost\""),
    ];
    
    let mut expected_processable = 0;
    for (path, content) in &success_files {
        let file_path = test_project.join(path);
        std::fs::create_dir_all(file_path.parent().unwrap())?;
        std::fs::write(file_path, content)?;
        expected_processable += 1;
    }
    
    // Create files that should be filtered out but not cause errors
    let filtered_files = [
        ("target/debug/binary", "binary content"),
        (".git/config", "git config"),
        ("node_modules/package/index.js", "module.exports = {};"),
        ("build/output.o", "object file content"),
    ];
    
    for (path, content) in &filtered_files {
        let file_path = test_project.join(path);
        std::fs::create_dir_all(file_path.parent().unwrap())?;
        std::fs::write(file_path, content)?;
    }
    
    // Create some problematic but not fatal scenarios
    let problematic_subdir = test_project.join("temp");
    std::fs::create_dir_all(&problematic_subdir)?;
    std::fs::write(problematic_subdir.join("temp.tmp"), "temporary content")?;
    
    // Process the directory - should handle partial failures gracefully
    let stats = parallel_indexer.index_directory_parallel(&test_project)?;
    
    println!("Partial failure test results:");
    println!("  Files processed: {}", stats.files_processed);
    println!("  Total size: {} bytes", stats.total_size);
    println!("  Duration: {:.2}s", stats.duration().as_secs_f64());
    
    // Should successfully process the indexable files
    assert!(stats.files_processed >= expected_processable, 
           "Should process at least {} indexable files, got {}", 
           expected_processable, stats.files_processed);
    
    // Should have reasonable performance despite mixed content
    assert!(stats.total_size > 0, "Should have processed some content");
    assert!(stats.duration().as_secs() < 30, "Should complete in reasonable time");
    
    // Verify the indexer remains functional after partial failures
    let validation_dir = temp_dir.path().join("post_partial_validation");
    create_test_project(&validation_dir, 3)?;
    
    let validation_stats = parallel_indexer.index_directory_parallel(&validation_dir)?;
    assert!(validation_stats.files_processed >= 3, 
           "Indexer should remain functional after partial failures");
    
    println!("Post-partial-failure validation: {} files processed", 
            validation_stats.files_processed);
    
    Ok(())
}
```

## Success Criteria
- [ ] Comprehensive error handling test validates all error scenarios
- [ ] Concurrent error scenarios test handles mixed success/failure cases
- [ ] Error recovery test ensures continued operation after problems
- [ ] Error propagation test provides meaningful error messages
- [ ] Partial failure test handles mixed content gracefully
- [ ] All error conditions produce appropriate error messages
- [ ] System remains stable and functional after encountering errors
- [ ] No panics or crashes during error conditions
- [ ] All tests pass consistently

## Time Limit
10 minutes

## Notes
- Error handling is crucial for production reliability
- Parallel operations complicate error scenarios significantly
- Graceful degradation is better than complete failure
- Error messages should be actionable and informative
- The system should continue operating after recoverable errors
- Thread safety must be maintained even during error conditions
- Test both expected and unexpected error scenarios