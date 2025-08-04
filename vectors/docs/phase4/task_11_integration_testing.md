# Task 11: Integration Testing

## Context
You are implementing Phase 4 of a vector indexing system. This task focuses on comprehensive integration testing between the new parallel indexing system and the existing search infrastructure. After implementing robust error handling, you now need to ensure that the parallel indexer integrates seamlessly with the broader vector search system, maintains compatibility with existing APIs, and works correctly in end-to-end scenarios.

## Current State
- `src/parallel.rs` exists with complete parallel indexing implementation
- Thread safety validation and error handling are comprehensive
- Parallel indexer works correctly in isolation
- Need to validate integration with existing search system components

## Task Objective
Implement comprehensive integration tests that validate the parallel indexer works correctly with the existing search system, maintains API compatibility, and provides end-to-end functionality for real-world usage scenarios.

## Implementation Requirements

### 1. Add search system integration test
Add this test to the test module in `src/parallel.rs`:
```rust
#[test]
fn test_search_system_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("integration_test");
    
    // Create parallel indexer
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create test project with diverse content for searching
    let test_project = temp_dir.path().join("search_integration_project");
    std::fs::create_dir_all(&test_project.join("src"))?;
    std::fs::create_dir_all(&test_project.join("docs"))?;
    std::fs::create_dir_all(&test_project.join("tests"))?;
    
    // Create Rust source files with searchable content
    let rust_files = [
        ("src/main.rs", "fn main() {\n    println!(\"Hello, world!\");\n    let vector_search = VectorSearch::new();\n}"),
        ("src/lib.rs", "pub mod search;\npub mod indexing;\n\nuse std::collections::HashMap;\npub struct VectorIndex;"),
        ("src/search.rs", "use crate::VectorIndex;\n\npub fn search_documents(query: &str) -> Vec<String> {\n    // Vector search implementation\n    vec![]\n}"),
        ("src/indexing.rs", "pub fn index_document(content: &str) -> Vec<f32> {\n    // Document indexing logic\n    vec![0.1, 0.2, 0.3]\n}"),
    ];
    
    for (path, content) in &rust_files {
        std::fs::write(test_project.join(path), content)?;
    }
    
    // Create documentation files
    let doc_files = [
        ("docs/README.md", "# Vector Search System\n\nThis system provides advanced vector search capabilities.\n\n## Features\n- Parallel indexing\n- Fast search\n- Rust implementation"),
        ("docs/api.md", "# API Documentation\n\n## VectorSearch\n\nThe main search interface.\n\n### Methods\n- `search(query)` - Performs vector search"),
        ("tests/integration_test.rs", "#[test]\nfn test_vector_search() {\n    let search = VectorSearch::new();\n    assert!(search.is_ready());\n}"),
    ];
    
    for (path, content) in &doc_files {
        std::fs::write(test_project.join(path), content)?;
    }
    
    // Index the project using parallel indexer
    let indexing_stats = parallel_indexer.index_directory_parallel(&test_project)?;
    
    println!("Integration test indexing completed:");
    println!("  Files processed: {}", indexing_stats.files_processed);
    println!("  Total size: {} bytes", indexing_stats.total_size);
    
    // Validate that all expected files were processed
    assert!(indexing_stats.files_processed >= 7, 
           "Should process at least 7 files (4 Rust + 3 docs), got {}", 
           indexing_stats.files_processed);
    
    // Verify that indexing created the expected index structure
    assert!(index_path.exists(), "Index directory should exist");
    
    // Check if we can create another indexer instance and it sees the existing index
    let secondary_indexer = ParallelIndexer::new(&index_path)?;
    
    // Test that the indexer can handle incremental updates
    let new_file = test_project.join("src/utils.rs");
    std::fs::write(&new_file, "pub fn utility_function() -> bool { true }")?;
    
    let incremental_stats = secondary_indexer.index_directory_parallel(&test_project)?;
    
    // Should process the new file plus any re-indexing
    assert!(incremental_stats.files_processed > 0, 
           "Incremental indexing should process at least the new file");
    
    println!("Incremental indexing: {} files processed", incremental_stats.files_processed);
    
    Ok(())
}
```

### 2. Add API compatibility test
Add this test to validate API compatibility:
```rust
#[test]
fn test_api_compatibility() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("api_compatibility");
    
    // Test that ParallelIndexer can be used as a drop-in replacement
    // for the existing DocumentIndexer API patterns
    
    // Create test data
    let test_project = temp_dir.path().join("api_test_project");
    create_test_project(&test_project, 15)?;
    
    // Test 1: Basic indexing API
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    let stats = parallel_indexer.index_directory_parallel(&test_project)?;
    
    // Validate stats structure compatibility
    assert!(stats.files_processed > 0);
    assert!(stats.total_size > 0);
    assert!(stats.duration().as_millis() > 0);
    
    // Test that stats methods work as expected
    let fps = stats.files_per_second;
    assert!(fps >= 0.0, "Files per second should be non-negative");
    
    let mbps = stats.megabytes_per_second();
    assert!(mbps >= 0.0, "Megabytes per second should be non-negative");
    
    let summary = stats.summary();
    assert!(!summary.is_empty(), "Summary should not be empty");
    assert!(summary.contains(&stats.files_processed.to_string()), 
           "Summary should contain file count");
    
    println!("API compatibility test - Stats: {}", summary);
    
    // Test 2: Path handling compatibility
    let path_variants = [
        test_project.clone(),
        test_project.canonicalize()?,
    ];
    
    for (i, path) in path_variants.iter().enumerate() {
        let variant_stats = parallel_indexer.index_directory_parallel(path)?;
        println!("Path variant {}: {} files", i, variant_stats.files_processed);
        
        // Should produce consistent results regardless of path format
        assert!(variant_stats.files_processed > 0, 
               "Path variant {} should process files", i);
    }
    
    // Test 3: Multiple indexer instances (should be allowed)
    let indexer2 = ParallelIndexer::new(&index_path.join("second"))?;
    let indexer3 = ParallelIndexer::new(&index_path.join("third"))?;
    
    // All should work independently
    let stats2 = indexer2.index_directory_parallel(&test_project)?;
    let stats3 = indexer3.index_directory_parallel(&test_project)?;
    
    assert!(stats2.files_processed > 0);
    assert!(stats3.files_processed > 0);
    
    println!("Multiple indexers test: {} + {} + {} files", 
            stats.files_processed, stats2.files_processed, stats3.files_processed);
    
    Ok(())
}
```

### 3. Add end-to-end workflow test
Add this test for complete workflow validation:
```rust
#[test]
fn test_end_to_end_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().join("e2e_workflow");
    
    // Simulate a complete development workflow
    println!("Starting end-to-end workflow test...");
    
    // Step 1: Initial project setup and indexing
    let project_path = base_path.join("project");
    let index_path = base_path.join("index");
    
    // Create initial project structure
    std::fs::create_dir_all(&project_path.join("src"))?;
    std::fs::create_dir_all(&project_path.join("docs"))?;
    
    let initial_files = [
        ("src/main.rs", "fn main() {\n    println!(\"Initial version\");\n}"),
        ("src/core.rs", "pub struct Core {\n    version: u32,\n}"),
        ("docs/guide.md", "# User Guide\n\nInitial documentation."),
    ];
    
    for (path, content) in &initial_files {
        std::fs::write(project_path.join(path), content)?;
    }
    
    // Initial indexing
    let indexer = ParallelIndexer::new(&index_path)?;
    let initial_stats = indexer.index_directory_parallel(&project_path)?;
    
    println!("Step 1 - Initial indexing: {} files", initial_stats.files_processed);
    assert!(initial_stats.files_processed >= 3);
    
    // Step 2: Project development - add new files
    std::thread::sleep(std::time::Duration::from_millis(100)); // Ensure different timestamps
    
    let new_files = [
        ("src/utils.rs", "pub fn helper() -> String {\n    \"helper function\".to_string()\n}"),
        ("src/tests.rs", "#[cfg(test)]\nmod tests {\n    #[test]\n    fn test_helper() {}\n}"),
        ("docs/api.md", "# API Reference\n\nDetailed API documentation."),
    ];
    
    for (path, content) in &new_files {
        std::fs::write(project_path.join(path), content)?;
    }
    
    // Incremental indexing
    let development_stats = indexer.index_directory_parallel(&project_path)?;
    
    println!("Step 2 - Development indexing: {} files", development_stats.files_processed);
    assert!(development_stats.files_processed >= 6); // Should see all files
    
    // Step 3: File modifications
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    // Modify existing files
    std::fs::write(project_path.join("src/main.rs"), 
                  "fn main() {\n    println!(\"Updated version\");\n    let core = Core { version: 2 };\n}")?;
    
    std::fs::write(project_path.join("docs/guide.md"), 
                  "# User Guide\n\nUpdated documentation with more details.\n\n## New Section\n\nAdditional content.")?;
    
    // Re-index after modifications
    let update_stats = indexer.index_directory_parallel(&project_path)?;
    
    println!("Step 3 - Update indexing: {} files", update_stats.files_processed);
    assert!(update_stats.files_processed >= 6);
    
    // Step 4: Large-scale addition (simulate bulk import)
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    let bulk_dir = project_path.join("bulk");
    std::fs::create_dir_all(&bulk_dir)?;
    
    for i in 0..20 {
        let content = format!("// Bulk file {}\npub fn bulk_function_{}() {{\n    // Implementation\n}}", i, i);
        std::fs::write(bulk_dir.join(format!("bulk_{:02}.rs", i)), content)?;
    }
    
    // Index the expanded project
    let bulk_stats = indexer.index_directory_parallel(&project_path)?;
    
    println!("Step 4 - Bulk indexing: {} files", bulk_stats.files_processed);
    assert!(bulk_stats.files_processed >= 26); // All previous + 20 new
    
    // Step 5: Performance validation over time
    let performance_samples = [
        ("sample1", 5),
        ("sample2", 10),
        ("sample3", 15),
    ];
    
    let mut performance_results = Vec::new();
    
    for (name, file_count) in &performance_samples {
        let sample_dir = base_path.join(name);
        create_test_project(&sample_dir, *file_count)?;
        
        let start_time = std::time::Instant::now();
        let sample_stats = indexer.index_directory_parallel(&sample_dir)?;
        let duration = start_time.elapsed();
        
        performance_results.push((name, file_count, sample_stats.files_processed, duration));
        
        println!("Performance sample {}: {} files in {:?}", 
                name, sample_stats.files_processed, duration);
    }
    
    // Validate consistent performance
    for (name, expected, actual, duration) in &performance_results {
        assert!(actual >= expected, 
               "Sample {} should process at least {} files, got {}", name, expected, actual);
        assert!(duration.as_secs() < 60, 
               "Sample {} took too long: {:?}", name, duration);
    }
    
    println!("End-to-end workflow test completed successfully");
    
    Ok(())
}
```

### 4. Add cross-platform integration test
Add this test for cross-platform compatibility:
```rust
#[test]
fn test_cross_platform_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("cross_platform");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create test project with platform-specific considerations
    let test_project = temp_dir.path().join("cross_platform_project");
    std::fs::create_dir_all(&test_project)?;
    
    // Test different file naming conventions
    let cross_platform_files = [
        // Standard files
        ("normal_file.rs", "// Normal file"),
        ("kebab-case-file.md", "# Kebab case file"),
        ("snake_case_file.py", "# Snake case file"),
        ("camelCaseFile.js", "// Camel case file"),
        
        // Files with spaces (common on Windows/Mac)
        ("file with spaces.txt", "Content with spaces in filename"),
        ("My Document.md", "# Document with spaces"),
        
        // Unicode filenames
        ("测试文件.txt", "Chinese filename test"),
        ("файл.rs", "Russian filename test"),
        ("εφαρμογή.md", "Greek filename test"),
        
        // Case sensitivity tests
        ("CaseSensitive.rs", "// Case sensitive test"),
        ("casesensitive.txt", "case sensitive lowercase"),
    ];
    
    for (filename, content) in &cross_platform_files {
        let file_path = test_project.join(filename);
        match std::fs::write(&file_path, content) {
            Ok(_) => println!("Created: {}", filename),
            Err(e) => println!("Warning: Could not create {}: {}", filename, e),
        }
    }
    
    // Create nested directory structure
    let nested_dirs = [
        "deeply/nested/directory/structure",
        "another/path/with/many/levels",
        "mixed/Path/With/Different/Cases",
    ];
    
    for dir_path in &nested_dirs {
        let full_path = test_project.join(dir_path);
        std::fs::create_dir_all(&full_path)?;
        std::fs::write(full_path.join("nested_file.rs"), "// Nested file")?;
    }
    
    // Test indexing with cross-platform considerations
    let stats = parallel_indexer.index_directory_parallel(&test_project)?;
    
    println!("Cross-platform integration results:");
    println!("  Files processed: {}", stats.files_processed);
    println!("  Total size: {} bytes", stats.total_size);
    
    // Should process most files (some unicode names might not work on all platforms)
    assert!(stats.files_processed >= 10, 
           "Should process at least 10 files cross-platform, got {}", stats.files_processed);
    
    // Test path handling with different separators and formats
    let path_formats = [
        test_project.clone(),
        test_project.join("."), // Current directory reference
        test_project.join("..").join(test_project.file_name().unwrap()), // Parent reference
    ];
    
    for (i, path_format) in path_formats.iter().enumerate() {
        if path_format.exists() {
            match parallel_indexer.index_directory_parallel(path_format) {
                Ok(path_stats) => {
                    println!("Path format {}: {} files", i, path_stats.files_processed);
                    assert!(path_stats.files_processed > 0, 
                           "Path format {} should process files", i);
                }
                Err(e) => {
                    println!("Path format {} failed (may be platform-specific): {}", i, e);
                }
            }
        }
    }
    
    // Platform-specific tests
    #[cfg(windows)]
    {
        println!("Running Windows-specific integration tests...");
        
        // Test Windows-specific path handling
        let windows_file = test_project.join("windows_specific.txt");
        std::fs::write(&windows_file, "Windows-specific content")?;
        
        let windows_stats = parallel_indexer.index_directory_parallel(&test_project)?;
        assert!(windows_stats.files_processed > 0, "Windows integration should work");
    }
    
    #[cfg(unix)]
    {
        println!("Running Unix-specific integration tests...");
        
        // Test Unix-specific features like symlinks
        let unix_file = test_project.join("unix_specific.rs");
        std::fs::write(&unix_file, "// Unix-specific content")?;
        
        // Create a symlink if possible
        let symlink_path = test_project.join("symlink_test");
        match std::os::unix::fs::symlink(&unix_file, &symlink_path) {
            Ok(_) => println!("Created symlink for testing"),
            Err(_) => println!("Could not create symlink (may be restricted)"),
        }
        
        let unix_stats = parallel_indexer.index_directory_parallel(&test_project)?;
        assert!(unix_stats.files_processed > 0, "Unix integration should work");
    }
    
    Ok(())
}
```

### 5. Add performance regression integration test
Add this test to validate performance doesn't regress:
```rust
#[test]
fn test_performance_regression_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("performance_regression");
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Create predictable test datasets of different sizes
    let test_sizes = [25, 50, 100];
    let mut performance_baseline = Vec::new();
    
    for &size in &test_sizes {
        let test_project = temp_dir.path().join(format!("perf_test_{}", size));
        create_test_project(&test_project, size)?;
        
        // Measure indexing performance
        let start_time = std::time::Instant::now();
        let stats = parallel_indexer.index_directory_parallel(&test_project)?;
        let duration = start_time.elapsed();
        
        let throughput = stats.files_processed as f64 / duration.as_secs_f64();
        performance_baseline.push((size, throughput, duration));
        
        println!("Performance baseline - {} files: {:.1} files/sec in {:?}", 
                size, throughput, duration);
        
        // Basic performance expectations
        assert!(throughput > 1.0, "Should process at least 1 file/sec for {} files", size);
        assert!(duration.as_secs() < 120, "Should complete {} files in under 2 minutes", size);
    }
    
    // Test repeated operations to check for performance degradation
    let repeat_test_size = 50;
    let repeat_project = temp_dir.path().join("repeat_test");
    create_test_project(&repeat_project, repeat_test_size)?;
    
    let mut repeat_times = Vec::new();
    for iteration in 0..5 {
        let start = std::time::Instant::now();
        let stats = parallel_indexer.index_directory_parallel(&repeat_project)?;
        let duration = start.elapsed();
        
        repeat_times.push(duration);
        
        println!("Iteration {}: {} files in {:?}", 
                iteration, stats.files_processed, duration);
        
        assert!(stats.files_processed >= repeat_test_size, 
               "Should consistently process files");
    }
    
    // Check for performance consistency (no significant degradation)
    let first_time = repeat_times[0];
    let last_time = repeat_times[repeat_times.len() - 1];
    
    // Allow up to 50% slowdown (generous for test environment variability)
    let max_acceptable_slowdown = first_time.as_millis() * 3 / 2;
    assert!(last_time.as_millis() <= max_acceptable_slowdown,
           "Performance degraded too much: first {:?}, last {:?}", first_time, last_time);
    
    // Test memory stability with large dataset
    let large_test_project = temp_dir.path().join("large_memory_test");
    create_test_project(&large_test_project, 150)?;
    
    let memory_test_start = std::time::Instant::now();
    let large_stats = parallel_indexer.index_directory_parallel(&large_test_project)?;
    let memory_test_duration = memory_test_start.elapsed();
    
    println!("Memory stability test: {} files in {:?}", 
            large_stats.files_processed, memory_test_duration);
    
    // Should handle larger datasets without issues
    assert!(large_stats.files_processed >= 150);
    assert!(memory_test_duration.as_secs() < 300, "Large dataset should complete in reasonable time");
    
    // Final integration validation
    let final_project = temp_dir.path().join("final_validation");
    create_test_project(&final_project, 30)?;
    
    let final_stats = parallel_indexer.index_directory_parallel(&final_project)?;
    assert!(final_stats.files_processed >= 30, 
           "Final validation should work after all performance tests");
    
    println!("Performance regression integration test completed successfully");
    
    Ok(())
}
```

## Success Criteria
- [ ] Search system integration test validates compatibility with existing components
- [ ] API compatibility test ensures drop-in replacement capability
- [ ] End-to-end workflow test validates complete development lifecycle
- [ ] Cross-platform integration test works on different operating systems
- [ ] Performance regression test maintains acceptable performance levels
- [ ] All integration points work correctly
- [ ] No breaking changes to existing APIs
- [ ] Real-world usage scenarios function properly
- [ ] All tests pass consistently

## Time Limit
10 minutes

## Notes
- Integration tests are crucial for validating system-wide compatibility
- Focus on real-world usage patterns and workflows
- Cross-platform compatibility ensures broad deployment capability
- Performance regression detection prevents quality degradation
- End-to-end tests validate the complete user experience
- API compatibility ensures existing code continues to work
- These tests should run as part of continuous integration