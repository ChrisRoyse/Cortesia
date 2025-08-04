# Task 057: Large File Handling Tests

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates large file handling tests that validates the system efficiently processes files ranging from 1MB to 50MB.

## Project Structure
tests/
  large_file_handling_tests.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive tests for large file processing including memory efficiency, streaming, chunking, and performance with large documents.

## Requirements
1. Create comprehensive integration test
2. Test files from 1MB to 50MB in size
3. Validate memory efficiency during processing
4. Handle streaming and chunking correctly
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::time::Instant;

#[tokio::test]
async fn test_large_file_indexing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let large_file_generator = LargeFileGenerator::new(temp_dir.path())?;
    
    // Generate files of different sizes
    let file_sizes = vec![
        (1_000_000, "1MB"),      // 1MB
        (5_000_000, "5MB"),      // 5MB
        (10_000_000, "10MB"),    // 10MB
        (25_000_000, "25MB"),    // 25MB
        (50_000_000, "50MB"),    // 50MB
    ];
    
    let mut test_files = Vec::new();
    for (size, label) in file_sizes {
        let file_path = large_file_generator.generate_large_text_file(size, label).await?;
        test_files.push((file_path, size, label));
    }
    
    let validator = CorrectnessValidator::new(&temp_dir.path().join("index"), 
                                           &temp_dir.path().join("vectors")).await?;
    
    for (file_path, size, label) in test_files {
        let start_time = Instant::now();
        let initial_memory = get_memory_usage()?;
        
        // Index the large file
        let indexing_result = validator.index_large_file(&file_path).await?;
        
        let indexing_duration = start_time.elapsed();
        let peak_memory = get_memory_usage()?;
        let memory_used = peak_memory.saturating_sub(initial_memory);
        
        // Performance assertions
        assert!(indexing_duration.as_secs() < 120, 
               "Large file {} indexing too slow: {:?}", label, indexing_duration);
        
        // Memory efficiency assertions - should not use more than 5x file size
        assert!(memory_used < size * 5,
               "Excessive memory usage for {}: {}MB used for {}MB file", 
               label, memory_used / 1_000_000, size / 1_000_000);
        
        // Verify indexing success
        assert!(indexing_result.chunks_processed > 0,
               "No chunks processed for large file {}", label);
        assert!(indexing_result.words_indexed > 1000,
               "Too few words indexed for large file {}: {}", label, indexing_result.words_indexed);
        
        println!("Large file {} ({} bytes): indexed in {:?}, {}MB memory, {} chunks", 
                label, size, indexing_duration, memory_used / 1_000_000, indexing_result.chunks_processed);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_large_file_search_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let large_file_generator = LargeFileGenerator::new(temp_dir.path())?;
    
    // Generate a 20MB file with known content patterns
    let large_file = large_file_generator.generate_structured_large_file(20_000_000).await?;
    let validator = CorrectnessValidator::new(&temp_dir.path().join("index"), 
                                           &temp_dir.path().join("vectors")).await?;
    
    // Index the large file
    validator.index_large_file(&large_file).await?;
    
    let search_test_cases = vec![
        ("function", "common term search"),
        ("specific_pattern_12345", "rare term search"),
        ("function AND implementation", "boolean search"),
        ("\"exact phrase match\"", "phrase search"),
        ("function*", "wildcard search"),
        ("vector:programming concepts", "vector search"),
    ];
    
    for (query, description) in search_test_cases {
        let start_time = Instant::now();
        let initial_memory = get_memory_usage()?;
        
        let search_results = validator.search_engine.search(query).await?;
        
        let search_duration = start_time.elapsed();
        let search_memory = get_memory_usage()?;
        let memory_used = search_memory.saturating_sub(initial_memory);
        
        // Performance assertions for large file searches
        assert!(search_duration.as_secs() < 30,
               "Large file search too slow for {}: {:?}", description, search_duration);
        
        // Memory should not spike during search
        assert!(memory_used < 100_000_000, // 100MB limit
               "Excessive memory usage during search {}: {}MB", description, memory_used / 1_000_000);
        
        // Results should be reasonable
        if !search_results.is_empty() {
            assert!(search_results.len() < 10000,
                   "Too many results for {}: {}", description, search_results.len());
            
            for result in search_results.iter().take(10) {
                assert!(!result.content.is_empty(),
                       "Empty content in search result for {}", description);
                assert!(result.content.len() < 10000,
                       "Result content too large for {}: {} chars", description, result.content.len());
            }
        }
        
        println!("Large file search {}: {} results in {:?}, {}MB memory", 
                description, search_results.len(), search_duration, memory_used / 1_000_000);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_streaming_large_file_processing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let large_file_generator = LargeFileGenerator::new(temp_dir.path())?;
    
    // Generate a very large file (30MB)
    let very_large_file = large_file_generator.generate_streaming_test_file(30_000_000).await?;
    let validator = CorrectnessValidator::new(&temp_dir.path().join("index"), 
                                           &temp_dir.path().join("vectors")).await?;
    
    let start_time = Instant::now();
    let initial_memory = get_memory_usage()?;
    
    // Process file in streaming mode
    let streaming_result = validator.process_file_streaming(&very_large_file).await?;
    
    let processing_duration = start_time.elapsed();
    let peak_memory = get_memory_usage()?;
    let memory_used = peak_memory.saturating_sub(initial_memory);
    
    // Streaming should be memory efficient
    assert!(memory_used < 200_000_000, // 200MB limit for 30MB file
           "Streaming processing used too much memory: {}MB", memory_used / 1_000_000);
    
    // Should complete in reasonable time
    assert!(processing_duration.as_secs() < 300, // 5 minutes
           "Streaming processing too slow: {:?}", processing_duration);
    
    // Verify streaming results
    assert!(streaming_result.chunks_processed > 100,
           "Too few chunks processed in streaming: {}", streaming_result.chunks_processed);
    assert!(streaming_result.bytes_processed > 25_000_000,
           "Not enough bytes processed: {}", streaming_result.bytes_processed);
    
    // Test search on streamed content
    let search_results = validator.search_engine.search("streaming_test_content").await?;
    assert!(!search_results.is_empty(),
           "No search results found in streamed content");
    
    println!("Streaming processing: {}MB in {:?}, {}MB memory peak, {} chunks", 
            streaming_result.bytes_processed / 1_000_000, processing_duration, 
            memory_used / 1_000_000, streaming_result.chunks_processed);
    
    Ok(())
}

#[tokio::test]
async fn test_large_file_error_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let validator = CorrectnessValidator::new(&temp_dir.path().join("index"), 
                                           &temp_dir.path().join("vectors")).await?;
    
    // Test with non-existent large file
    let non_existent_file = temp_dir.path().join("non_existent_large_file.txt");
    let result = validator.index_large_file(&non_existent_file).await;
    assert!(result.is_err(), "Should fail for non-existent file");
    
    // Test with corrupted large file
    let corrupted_file = temp_dir.path().join("corrupted_large_file.txt");
    std::fs::write(&corrupted_file, vec![0u8; 1000000])?; // 1MB of null bytes
    
    let result = validator.index_large_file(&corrupted_file).await;
    // Should handle gracefully, not crash
    match result {
        Ok(result) => {
            assert!(result.chunks_processed >= 0, "Should handle corrupted file gracefully");
        },
        Err(_) => {
            println!("Corrupted file handling failed gracefully as expected");
        }
    }
    
    // Test with extremely large file (simulate)
    let huge_file_path = temp_dir.path().join("huge_file.txt");
    // Don't actually create a huge file, just test the size check
    let result = validator.check_file_size_limits(&huge_file_path, 100_000_000_000).await; // 100GB
    assert!(result.is_err(), "Should reject extremely large files");
    
    println!("Large file error handling test completed successfully");
    Ok(())
}

fn get_memory_usage() -> Result<u64> {
    #[cfg(unix)]
    {
        use std::fs;
        let contents = fs::read_to_string("/proc/self/status")?;
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Ok(parts[1].parse::<u64>()? * 1024);
                }
            }
        }
        Ok(0)
    }
    
    #[cfg(windows)]
    {
        // Simplified for Windows
        Ok(0)
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        Ok(0)
    }
}
```

## Success Criteria
- Files up to 50MB are indexed efficiently
- Memory usage stays within 5x file size during indexing
- Search performance remains acceptable for large files (< 30 seconds)
- Streaming processing uses < 200MB memory for 30MB files
- Large file indexing completes within 2 minutes per file
- Error handling works for corrupted and missing files
- Chunking produces reasonable chunk counts
- Search results are properly limited and formatted

## Time Limit
10 minutes maximum