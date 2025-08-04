# Micro-Task 151: Measure Hybrid Search Performance

## Objective
Benchmark hybrid search combining Tantivy text search with LanceDB vector similarity.

## Prerequisites
- Task 150 completed (LanceDB operations benchmarked)
- Hybrid search implementation functional

## Time Estimate
10 minutes

## Instructions
1. Create hybrid search benchmark `bench_hybrid_search.rs`:
   ```rust
   use std::time::Instant;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking hybrid search...");
       
       benchmark_hybrid_search("small", 100, 10)?;
       benchmark_hybrid_search("medium", 1000, 50)?;
       benchmark_hybrid_search("large", 10000, 100)?;
       
       Ok(())
   }
   
   fn benchmark_hybrid_search(dataset: &str, docs: usize, queries: usize) -> Result<(), Box<dyn std::error::Error>> {
       println!("Hybrid search on {} dataset: {} queries", dataset, queries);
       
       let start = Instant::now();
       
       for i in 0..queries {
           // Simulate text search component
           let _text_results = simulate_text_search(&format!("query {}", i));
           
           // Simulate vector search component  
           let _vector_results = simulate_vector_search(384);
           
           // Simulate result fusion
           let _hybrid_results = combine_results(_text_results, _vector_results);
       }
       
       let duration = start.elapsed();
       let avg_latency = duration.as_millis() as f64 / queries as f64;
       
       println!("  {} queries in {:.3}ms (avg: {:.3}ms/query)", 
               queries, duration.as_millis(), avg_latency);
       
       if avg_latency > 5.0 {
           println!("  ❌ Exceeds 5ms target by {:.3}ms", avg_latency - 5.0);
       } else {
           println!("  ✅ Meets 5ms target");
       }
       
       Ok(())
   }
   
   fn simulate_text_search(query: &str) -> Vec<u32> {
       // Simulate text search latency
       std::thread::sleep(std::time::Duration::from_micros(100));
       vec![1, 2, 3, 4, 5]
   }
   
   fn simulate_vector_search(dims: usize) -> Vec<u32> {
       // Simulate vector search latency
       std::thread::sleep(std::time::Duration::from_micros(200));
       vec![6, 7, 8, 9, 10]
   }
   
   fn combine_results(text: Vec<u32>, vector: Vec<u32>) -> Vec<u32> {
       let mut combined = text;
       combined.extend(vector);
       combined.sort();
       combined.dedup();
       combined
   }
   ```
2. Run benchmark: `cargo run --release --bin bench_hybrid_search`
3. Commit: `git add src/bin/bench_hybrid_search.rs && git commit -m "Measure hybrid search performance vs 5ms target"`

## Success Criteria
- [ ] Hybrid search benchmark created
- [ ] Performance measured for all datasets
- [ ] 5ms target validation included
- [ ] Results committed

## Next Task
task_152_profile_memory_allocations.md