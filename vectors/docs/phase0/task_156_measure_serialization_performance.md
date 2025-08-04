# Micro-Task 156: Measure Serialization Performance

## Objective
Benchmark JSON serialization/deserialization performance for data exchange operations.

## Prerequisites
- Task 155 completed (I/O operations benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create serialization benchmark `bench_serialization.rs`:
   ```rust
   use serde::{Deserialize, Serialize};
   use std::time::Instant;
   
   #[derive(Serialize, Deserialize, Clone)]
   struct Document {
       id: u64,
       title: String,
       content: String,
       category: String,
       tags: Vec<String>,
       metadata: std::collections::HashMap<String, String>,
   }
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking serialization performance...");
       
       let small_docs = generate_documents(100);
       let medium_docs = generate_documents(1000);
       let large_docs = generate_documents(10000);
       
       benchmark_serialization("Small", &small_docs)?;
       benchmark_serialization("Medium", &medium_docs)?;
       benchmark_serialization("Large", &large_docs)?;
       
       Ok(())
   }
   
   fn generate_documents(count: usize) -> Vec<Document> {
       (0..count).map(|i| Document {
           id: i as u64,
           title: format!("Document {}", i),
           content: format!("Content for document {} with lots of text", i),
           category: format!("category_{}", i % 5),
           tags: vec![format!("tag_{}", i % 10), format!("tag_{}", (i + 1) % 10)],
           metadata: {
               let mut map = std::collections::HashMap::new();
               map.insert("author".to_string(), format!("author_{}", i % 3));
               map.insert("created".to_string(), "2024-01-01".to_string());
               map
           },
       }).collect()
   }
   
   fn benchmark_serialization(size: &str, docs: &[Document]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Testing {} dataset: {} documents", size, docs.len());
       
       // Serialize benchmark
       let start = Instant::now();
       let json = serde_json::to_string(docs)?;
       let serialize_duration = start.elapsed();
       
       // Deserialize benchmark
       let start = Instant::now();
       let _deserialized: Vec<Document> = serde_json::from_str(&json)?;
       let deserialize_duration = start.elapsed();
       
       let json_size_mb = json.len() as f64 / 1024.0 / 1024.0;
       let serialize_throughput = json_size_mb / serialize_duration.as_secs_f64();
       let deserialize_throughput = json_size_mb / deserialize_duration.as_secs_f64();
       
       println!("  Serialize: {:.3}ms ({:.1} MB/s)", serialize_duration.as_millis(), serialize_throughput);
       println!("  Deserialize: {:.3}ms ({:.1} MB/s)", deserialize_duration.as_millis(), deserialize_throughput);
       println!("  JSON size: {:.2} MB", json_size_mb);
       
       Ok(())
   }
   ```
2. Run: `cargo run --release --bin bench_serialization`
3. Commit: `git add src/bin/bench_serialization.rs && git commit -m "Measure JSON serialization performance"`

## Success Criteria
- [ ] Serialization benchmark created
- [ ] Serialize/deserialize performance measured
- [ ] Throughput calculated
- [ ] Results committed

## Next Task
task_157_benchmark_text_processing.md