# Task 028: Generate Large File Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-012. Large file handling is critical for production systems that must index substantial codebases and handle memory pressure scenarios.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_large_file_tests()` method that creates test files of varying sizes (1MB, 10MB, 50MB) to validate performance, memory usage, and chunking boundary behavior.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files at 1MB, 10MB, and 50MB sizes
3. Include files that test chunking boundaries and buffer limits
4. Create memory pressure scenarios with sparse matches
5. Add performance degradation detection patterns
6. Include realistic large file content (not just repeated data)
7. Generate files with different content densities and structures

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_large_file_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // 1MB file with realistic Rust code structure
        let medium_file_content = self.generate_rust_code_content(1_048_576)?; // 1MB
        let mut medium_file = self.create_test_file("large_1mb_rust.rs", &medium_file_content, TestFileType::LargeFile)?;
        medium_file.expected_matches = vec![
            "pub fn".to_string(),
            "impl".to_string(),
            "Result<".to_string(),
            "use std::".to_string(),
            "// TODO:".to_string(),
        ];
        files.push(medium_file);
        
        // 10MB file with mixed content types
        let large_file_content = self.generate_mixed_content_file(10_485_760)?; // 10MB
        let mut large_file = self.create_test_file("large_10mb_mixed.txt", &large_file_content, TestFileType::LargeFile)?;
        large_file.expected_matches = vec![
            "SECTION_MARKER".to_string(),
            "performance_test_".to_string(),
            "data_chunk_".to_string(),
            "ERROR:".to_string(),
            "SUCCESS:".to_string(),
        ];
        files.push(large_file);
        
        // 50MB file for stress testing
        let huge_file_content = self.generate_stress_test_content(52_428_800)?; // 50MB
        let mut huge_file = self.create_test_file("large_50mb_stress.log", &huge_file_content, TestFileType::LargeFile)?;
        huge_file.expected_matches = vec![
            "BENCHMARK_START".to_string(),
            "BENCHMARK_END".to_string(),
            "MEMORY_USAGE:".to_string(),
            "CHUNK_BOUNDARY_".to_string(),
            "PERFORMANCE_METRIC".to_string(),
        ];
        files.push(huge_file);
        
        // Sparse matches file - large file with few matches
        let sparse_content = self.generate_sparse_matches_content(5_242_880)?; // 5MB
        let mut sparse_file = self.create_test_file("large_sparse_matches.rs", &sparse_content, TestFileType::LargeFile)?;
        sparse_file.expected_matches = vec![
            "RARE_PATTERN_001".to_string(),
            "RARE_PATTERN_002".to_string(),
            "RARE_PATTERN_003".to_string(),
            "NEEDLE_IN_HAYSTACK".to_string(),
        ];
        files.push(sparse_file);
        
        // Chunking boundary test file
        let boundary_content = self.generate_chunking_boundary_content(2_097_152)?; // 2MB
        let mut boundary_file = self.create_test_file("large_chunking_test.rs", &boundary_content, TestFileType::LargeFile)?;
        boundary_file.expected_matches = vec![
            "CHUNK_BOUNDARY_MARKER".to_string(),
            "SPLIT_POINT_TEST".to_string(),
            "BUFFER_OVERFLOW_TEST".to_string(),
            "CONTINUATION_MARKER".to_string(),
        ];
        files.push(boundary_file);
        
        Ok(files)
    }
    
    /// Generate realistic Rust code content of specified size
    fn generate_rust_code_content(&self, target_size: usize) -> Result<String> {
        let mut content = String::with_capacity(target_size + 1000);
        
        // File header
        content.push_str(r#"
//! Large Rust file for performance testing
//! This file contains realistic Rust code patterns
//! Generated for vector indexing system validation

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, Error};

"#);
        
        let mut counter = 0;
        while content.len() < target_size {
            let module_content = format!(r#"
/// Module for handling operation batch {}
pub mod operation_batch_{} {{
    use super::*;
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BatchProcessor {{
        pub id: usize,
        pub capacity: usize,
        pub processed_items: Vec<ProcessedItem>,
        pub status: BatchStatus,
    }}
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum BatchStatus {{
        Pending,
        Processing,
        Completed,
        Failed(String),
    }}
    
    impl BatchProcessor {{
        pub fn new(id: usize, capacity: usize) -> Self {{
            Self {{
                id,
                capacity,
                processed_items: Vec::with_capacity(capacity),
                status: BatchStatus::Pending,
            }}
        }}
        
        pub async fn process_items(&mut self, items: Vec<InputItem>) -> Result<()> {{
            self.status = BatchStatus::Processing;
            
            for (index, item) in items.into_iter().enumerate() {{
                // TODO: Implement proper error handling for item processing
                let processed = self.process_single_item(item, index).await?;
                self.processed_items.push(processed);
                
                if self.processed_items.len() >= self.capacity {{
                    break;
                }}
            }}
            
            self.status = BatchStatus::Completed;
            Ok(())
        }}
        
        async fn process_single_item(&self, item: InputItem, index: usize) -> Result<ProcessedItem> {{
            let start_time = Instant::now();
            
            // Simulate complex processing
            tokio::time::sleep(Duration::from_millis(1)).await;
            
            let processing_time = start_time.elapsed();
            
            Ok(ProcessedItem {{
                original_index: index,
                data: item.data,
                processing_time,
                metadata: generate_metadata(),
            }})
        }}
    }}
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputItem {{
        pub data: String,
        pub priority: u8,
        pub timestamp: u64,
    }}
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProcessedItem {{
        pub original_index: usize,
        pub data: String,
        pub processing_time: Duration,
        pub metadata: HashMap<String, String>,
    }}
    
    fn generate_metadata() -> HashMap<String, String> {{
        let mut metadata = HashMap::new();
        metadata.insert("processor_version".to_string(), "1.0.0".to_string());
        metadata.insert("processing_node".to_string(), "node-001".to_string());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata
    }}
}}

"#, counter, counter);
            
            content.push_str(&module_content);
            counter += 1;
            
            // Add some variety every 10 modules
            if counter % 10 == 0 {
                content.push_str(&format!(r#"
// Performance checkpoint {}
const CHECKPOINT_{}: &str = "Performance measurement point";

pub fn benchmark_checkpoint_{}() -> Duration {{
    let start = Instant::now();
    // Simulate work
    std::thread::sleep(Duration::from_nanos(100));
    start.elapsed()
}}

"#, counter, counter, counter));
            }
        }
        
        Ok(content)
    }
    
    /// Generate mixed content file with different sections
    fn generate_mixed_content_file(&self, target_size: usize) -> Result<String> {
        let mut content = String::with_capacity(target_size + 1000);
        
        content.push_str("=== MIXED CONTENT FILE FOR PERFORMANCE TESTING ===\n\n");
        
        let mut section = 0;
        while content.len() < target_size {
            match section % 4 {
                0 => {
                    content.push_str(&format!("SECTION_MARKER: Configuration Section {}\n", section));
                    content.push_str("# Configuration data with various formats\n");
                    for i in 0..50 {
                        content.push_str(&format!("config_option_{}_{} = value_{}\n", section, i, i));
                    }
                },
                1 => {
                    content.push_str(&format!("SECTION_MARKER: Log Section {}\n", section));
                    for i in 0..30 {
                        content.push_str(&format!("INFO: performance_test_{}_{}; SUCCESS: operation completed\n", section, i));
                        content.push_str(&format!("DEBUG: data_chunk_{}_processed; size: {} bytes\n", i, i * 1024));
                    }
                },
                2 => {
                    content.push_str(&format!("SECTION_MARKER: Data Section {}\n", section));
                    for i in 0..40 {
                        content.push_str(&format!("DATA_RECORD_{}: {{\"id\": {}, \"value\": \"data_chunk_{}\", \"timestamp\": {}}}\n", 
                                              i, i, i, 1700000000 + i));
                    }
                },
                3 => {
                    content.push_str(&format!("SECTION_MARKER: Error Section {}\n", section));
                    for i in 0..20 {
                        if i % 3 == 0 {
                            content.push_str(&format!("ERROR: Failed to process item {}: Connection timeout\n", i));
                        } else {
                            content.push_str(&format!("SUCCESS: Item {} processed successfully\n", i));
                        }
                    }
                },
                _ => unreachable!(),
            }
            section += 1;
        }
        
        Ok(content)
    }
    
    /// Generate content specifically for stress testing
    fn generate_stress_test_content(&self, target_size: usize) -> Result<String> {
        let mut content = String::with_capacity(target_size + 1000);
        
        content.push_str("BENCHMARK_START: Large file stress test initiated\n");
        content.push_str("MEMORY_USAGE: Initial allocation complete\n\n");
        
        let mut chunk_id = 0;
        while content.len() < target_size {
            content.push_str(&format!("CHUNK_BOUNDARY_{}: Starting chunk processing\n", chunk_id));
            
            // Generate repetitive but not identical content
            for i in 0..100 {
                content.push_str(&format!("PERFORMANCE_METRIC: chunk_{}_item_{} processed in {}ms\n", 
                                        chunk_id, i, (i * 7) % 100));
                
                if i % 10 == 0 {
                    content.push_str(&format!("MEMORY_USAGE: Heap size: {}MB, Stack: {}KB\n", 
                                            (chunk_id * 10 + i) % 1000, (i * 4) % 100));
                }
            }
            
            content.push_str(&format!("CHUNK_BOUNDARY_{}: Chunk processing complete\n\n", chunk_id));
            chunk_id += 1;
        }
        
        content.push_str("BENCHMARK_END: Stress test completed successfully\n");
        Ok(content)
    }
    
    /// Generate content with sparse matching patterns
    fn generate_sparse_matches_content(&self, target_size: usize) -> Result<String> {
        let mut content = String::with_capacity(target_size + 1000);
        
        // Fill most of the file with non-matching content
        let filler_line = "This is filler content that should not match any search patterns. ";
        let rare_pattern_interval = target_size / 20; // Insert rare patterns every ~250KB
        
        let mut current_size = 0;
        let mut rare_pattern_count = 1;
        
        while current_size < target_size {
            // Add filler content
            for _ in 0..1000 {
                content.push_str(filler_line);
                current_size += filler_line.len();
                
                if current_size >= rare_pattern_interval * rare_pattern_count {
                    // Insert a rare pattern
                    match rare_pattern_count % 4 {
                        1 => content.push_str(&format!("RARE_PATTERN_001: Special marker at position {}\n", current_size)),
                        2 => content.push_str(&format!("RARE_PATTERN_002: Another marker at position {}\n", current_size)),
                        3 => content.push_str(&format!("RARE_PATTERN_003: Third marker at position {}\n", current_size)),
                        0 => content.push_str(&format!("NEEDLE_IN_HAYSTACK: Found at position {}\n", current_size)),
                        _ => unreachable!(),
                    }
                    rare_pattern_count += 1;
                    current_size += 100; // Approximate size of pattern line
                }
                
                if current_size >= target_size {
                    break;
                }
            }
        }
        
        Ok(content)
    }
    
    /// Generate content that tests chunking boundaries
    fn generate_chunking_boundary_content(&self, target_size: usize) -> Result<String> {
        let mut content = String::with_capacity(target_size + 1000);
        
        // Assume typical chunk size is 64KB
        let chunk_size = 65536;
        let num_chunks = target_size / chunk_size;
        
        for chunk_num in 0..num_chunks {
            let chunk_start = chunk_num * chunk_size;
            
            content.push_str(&format!("CHUNK_BOUNDARY_MARKER: Chunk {} starts at byte {}\n", 
                                    chunk_num, chunk_start));
            
            // Fill the chunk, placing test patterns at strategic positions
            let mut current_chunk_size = 100; // Account for header
            
            while current_chunk_size < chunk_size - 200 {
                content.push_str("Regular content line that fills the chunk with predictable data. ");
                current_chunk_size += 63;
                
                // Place patterns near chunk boundaries
                if current_chunk_size > chunk_size - 500 && current_chunk_size < chunk_size - 300 {
                    content.push_str("SPLIT_POINT_TEST: Pattern near chunk boundary\n");
                    current_chunk_size += 50;
                }
                
                if current_chunk_size > chunk_size - 200 {
                    content.push_str("BUFFER_OVERFLOW_TEST: Testing buffer limits\n");
                    current_chunk_size += 45;
                    break;
                }
            }
            
            // End of chunk marker
            content.push_str(&format!("CONTINUATION_MARKER: Chunk {} ends, continues to chunk {}\n\n", 
                                    chunk_num, chunk_num + 1));
        }
        
        Ok(content)
    }
}
```

## Success Criteria
- Method generates 5+ test files of varying sizes (1MB to 50MB)
- Each file includes expected_matches for validation testing
- Files test different scenarios: realistic code, mixed content, stress testing
- Chunking boundary behavior is properly tested
- Sparse match scenarios validate search efficiency
- Memory pressure scenarios are included
- Content is diverse and realistic, not just repeated patterns

## Time Limit
10 minutes maximum