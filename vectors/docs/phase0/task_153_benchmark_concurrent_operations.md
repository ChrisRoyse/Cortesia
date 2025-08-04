# Micro-Task 153: Benchmark Concurrent Operations

## Objective
Measure performance under concurrent search operations to validate thread safety and scalability.

## Prerequisites
- Task 152 completed (Memory allocations profiled)

## Time Estimate
9 minutes

## Instructions
1. Create concurrent benchmark `bench_concurrent.rs`:
   ```rust
   use std::sync::Arc;
   use std::thread;
   use std::time::{Duration, Instant};
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking concurrent operations...");
       
       benchmark_concurrent_searches(2, 100)?;
       benchmark_concurrent_searches(4, 100)?;
       benchmark_concurrent_searches(8, 100)?;
       benchmark_concurrent_searches(16, 100)?;
       
       Ok(())
   }
   
   fn benchmark_concurrent_searches(thread_count: usize, queries_per_thread: usize) -> Result<(), Box<dyn std::error::Error>> {
       println!("Testing {} threads, {} queries each", thread_count, queries_per_thread);
       
       let start = Instant::now();
       let mut handles = vec![];
       
       for thread_id in 0..thread_count {
           let handle = thread::spawn(move || {
               for query_id in 0..queries_per_thread {
                   simulate_search(thread_id, query_id);
               }
           });
           handles.push(handle);
       }
       
       for handle in handles {
           handle.join().unwrap();
       }
       
       let duration = start.elapsed();
       let total_queries = thread_count * queries_per_thread;
       let queries_per_sec = total_queries as f64 / duration.as_secs_f64();
       let avg_latency = duration.as_millis() as f64 / total_queries as f64;
       
       println!("  {} total queries in {:.3}ms", total_queries, duration.as_millis());
       println!("  {:.1} queries/sec, {:.3}ms avg latency", queries_per_sec, avg_latency);
       
       if avg_latency > 5.0 {
           println!("  ❌ Exceeds 5ms target");
       } else {
           println!("  ✅ Meets 5ms target");
       }
       
       Ok(())
   }
   
   fn simulate_search(thread_id: usize, query_id: usize) {
       // Simulate search work
       let _result: Vec<f32> = (0..100).map(|i| (thread_id + query_id + i) as f32).collect();
       thread::sleep(Duration::from_micros(50)); // Simulate computation
   }
   ```
2. Run: `cargo run --release --bin bench_concurrent`
3. Commit: `git add src/bin/bench_concurrent.rs && git commit -m "Benchmark concurrent search operations"`

## Success Criteria
- [ ] Concurrent benchmark created
- [ ] Multi-thread performance measured
- [ ] Scalability assessed
- [ ] Results committed

## Next Task
task_154_measure_cache_performance.md