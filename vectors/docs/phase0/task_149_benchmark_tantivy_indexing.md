# Micro-Task 149: Benchmark Tantivy Indexing

## Objective
Measure Tantivy indexing performance across different document sizes and validate indexing efficiency.

## Prerequisites
- Task 148 completed (Allocation latency measured)
- Tantivy integration functional
- Benchmark datasets available

## Time Estimate
8 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Create Tantivy indexing benchmark `bench_tantivy_indexing.rs`:
   ```rust
   use tantivy::{Index, IndexWriter, doc, schema::*};
   use std::time::{Duration, Instant};
   use serde_json::Value;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking Tantivy indexing...");
       
       // Load datasets
       let small = load_dataset("benchmark_data/small_dataset.json")?;
       let medium = load_dataset("benchmark_data/medium_dataset.json")?;
       let large = load_dataset("benchmark_data/large_dataset.json")?;
       
       // Benchmark indexing
       benchmark_indexing("small", &small)?;
       benchmark_indexing("medium", &medium)?;
       benchmark_indexing("large", &large)?;
       
       println!("Tantivy indexing benchmark complete");
       Ok(())
   }
   
   fn benchmark_indexing(size: &str, documents: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
       let mut schema_builder = Schema::builder();
       let title = schema_builder.add_text_field("title", TEXT | STORED);
       let content = schema_builder.add_text_field("content", TEXT);
       let schema = schema_builder.build();
       
       let index = Index::create_in_ram(schema.clone());
       let mut index_writer: IndexWriter = index.writer(50_000_000)?;
       
       let start = Instant::now();
       
       for doc in documents {
           let doc_title = doc["title"].as_str().unwrap_or("");
           let doc_content = doc["content"].as_str().unwrap_or("");
           
           index_writer.add_document(doc!(
               title => doc_title,
               content => doc_content
           ))?;
       }
       
       index_writer.commit()?;
       let duration = start.elapsed();
       
       let docs_per_sec = documents.len() as f64 / duration.as_secs_f64();
       
       println!("{} dataset: {} docs in {:.3}ms ({:.1} docs/sec)", 
               size, documents.len(), duration.as_millis(), docs_per_sec);
       
       Ok(())
   }
   
   fn load_dataset(path: &str) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
       let content = std::fs::read_to_string(path)?;
       Ok(serde_json::from_str(&content)?)
   }
   ```
3. Run benchmark: `cargo run --release --bin bench_tantivy_indexing`
4. Commit: `git add src/bin/bench_tantivy_indexing.rs && git commit -m "Benchmark Tantivy indexing performance"`

## Success Criteria
- [ ] Tantivy indexing benchmark created
- [ ] Performance measured for all datasets
- [ ] Indexing rates calculated
- [ ] Results logged and committed

## Next Task
task_150_benchmark_lancedb_operations.md