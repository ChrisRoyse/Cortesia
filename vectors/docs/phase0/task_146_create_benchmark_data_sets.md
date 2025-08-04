# Micro-Task 146: Create Benchmark Data Sets

## Objective
Generate standardized data sets for consistent benchmarking across all performance tests.

## Context
Standardized benchmark data ensures consistent and comparable results across different benchmark runs and validates performance under various data loads.

## Prerequisites
- Task 145 completed (Memory profiling setup)
- Test data generation framework established
- Profiling tools configured

## Time Estimate
10 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Create benchmark data generation script `generate_benchmark_data.rs`:
   ```rust
   use std::fs;
   use serde_json::json;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       fs::create_dir_all("benchmark_data")?;
       
       // Small dataset - 1K documents
       generate_dataset("small", 1_000, 100)?;
       
       // Medium dataset - 10K documents  
       generate_dataset("medium", 10_000, 500)?;
       
       // Large dataset - 100K documents
       generate_dataset("large", 100_000, 1000)?;
       
       println!("Benchmark datasets generated successfully");
       Ok(())
   }
   
   fn generate_dataset(name: &str, count: usize, avg_words: usize) -> Result<(), Box<dyn std::error::Error>> {
       let filename = format!("benchmark_data/{}_dataset.json", name);
       let mut documents = Vec::new();
       
       for i in 0..count {
           documents.push(json!({
               "id": i,
               "title": format!("Document {}", i),
               "content": generate_content(avg_words),
               "category": format!("category_{}", i % 10),
               "timestamp": chrono::Utc::now().timestamp() + i as i64
           }));
       }
       
       fs::write(filename, serde_json::to_string_pretty(&documents)?)?;
       Ok(())
   }
   
   fn generate_content(word_count: usize) -> String {
       (0..word_count).map(|i| format!("word{}", i % 1000)).collect::<Vec<_>>().join(" ")
   }
   ```
3. Create benchmark data configuration `benchmark_datasets.toml`:
   ```toml
   [datasets.small]
   document_count = 1000
   average_words = 100
   categories = 10
   file_size_mb = 1
   
   [datasets.medium] 
   document_count = 10000
   average_words = 500
   categories = 50
   file_size_mb = 50
   
   [datasets.large]
   document_count = 100000
   average_words = 1000
   categories = 100
   file_size_mb = 500
   
   [performance_targets]
   small_dataset_ms = 1
   medium_dataset_ms = 3
   large_dataset_ms = 5
   ```
4. Generate datasets: `cargo run --bin generate_benchmark_data`
5. Validate data: `dir benchmark_data\*.json`
6. Commit datasets: `git add src/bin/generate_benchmark_data.rs benchmark_datasets.toml benchmark_data/ && git commit -m "Create standardized benchmark datasets"`

## Expected Output
- Three standardized benchmark datasets (small, medium, large)
- Benchmark dataset configuration
- Data generation utility
- Performance targets for each dataset

## Success Criteria
- [ ] Benchmark data generation script created
- [ ] Three datasets generated successfully
- [ ] Dataset configuration file created
- [ ] Data files validated and committed
- [ ] Performance targets established

## Validation Commands
```batch
# Verify datasets generated
dir benchmark_data

# Check file sizes
dir benchmark_data\*.json

# Validate JSON format
type benchmark_data\small_dataset.json | head -10
```

## Next Task
task_147_establish_performance_baseline.md

## Notes
- Consistent data ensures reproducible benchmarks
- Different dataset sizes test scalability
- JSON format allows easy inspection and validation
- Performance targets guide optimization efforts