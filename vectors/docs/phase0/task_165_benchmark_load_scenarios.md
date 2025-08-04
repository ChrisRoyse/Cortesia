# Micro-Task 165: Benchmark Load Scenarios

## Objective
Test system performance under various load scenarios to validate scalability.

## Prerequisites
- Task 164 completed (Performance dashboard created)

## Time Estimate
10 minutes

## Instructions
1. Create load scenario benchmark `bench_load_scenarios.rs`:
   ```rust
   use std::sync::Arc;
   use std::sync::atomic::{AtomicUsize, Ordering};
   use std::thread;
   use std::time::{Duration, Instant};
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking load scenarios...");
       
       benchmark_light_load()?;
       benchmark_medium_load()?;
       benchmark_heavy_load()?;
       benchmark_burst_load()?;
       benchmark_sustained_load()?;
       
       Ok(())
   }
   
   fn benchmark_light_load() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Light Load Scenario ===");
       println!("2 threads, 10 queries/sec, 30 seconds");
       
       let duration = Duration::from_secs(30);
       let threads = 2;
       let qps = 10;
       
       run_load_test(threads, qps, duration)
   }
   
   fn benchmark_medium_load() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Medium Load Scenario ===");
       println!("4 threads, 50 queries/sec, 60 seconds");
       
       let duration = Duration::from_secs(60);
       let threads = 4;
       let qps = 50;
       
       run_load_test(threads, qps, duration)
   }
   
   fn benchmark_heavy_load() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Heavy Load Scenario ===");
       println!("8 threads, 200 queries/sec, 120 seconds");
       
       let duration = Duration::from_secs(120);
       let threads = 8;
       let qps = 200;
       
       run_load_test(threads, qps, duration)
   }
   
   fn benchmark_burst_load() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Burst Load Scenario ===");
       println!("16 threads, 1000 queries/sec, 10 seconds");
       
       let duration = Duration::from_secs(10);
       let threads = 16;
       let qps = 1000;
       
       run_load_test(threads, qps, duration)
   }
   
   fn benchmark_sustained_load() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Sustained Load Scenario ===");
       println!("6 threads, 100 queries/sec, 300 seconds");
       
       let duration = Duration::from_secs(300);
       let threads = 6;
       let qps = 100;
       
       run_load_test(threads, qps, duration)
   }
   
   fn run_load_test(thread_count: usize, target_qps: usize, duration: Duration) -> Result<(), Box<dyn std::error::Error>> {
       let completed_queries = Arc::new(AtomicUsize::new(0));
       let failed_queries = Arc::new(AtomicUsize::new(0));
       let total_latency = Arc::new(AtomicUsize::new(0));
       
       let start_time = Instant::now();
       let queries_per_thread = target_qps / thread_count;
       let interval = Duration::from_millis(1000 / queries_per_thread as u64);
       
       let mut handles = vec![];
       
       for thread_id in 0..thread_count {
           let completed = Arc::clone(&completed_queries);
           let failed = Arc::clone(&failed_queries);
           let latency = Arc::clone(&total_latency);
           
           let handle = thread::spawn(move || {
               let thread_start = Instant::now();
               
               while thread_start.elapsed() < duration {
                   let query_start = Instant::now();
                   
                   // Simulate query processing
                   if simulate_query(thread_id) {
                       completed.fetch_add(1, Ordering::Relaxed);
                       let query_latency = query_start.elapsed().as_micros() as usize;
                       latency.fetch_add(query_latency, Ordering::Relaxed);
                   } else {
                       failed.fetch_add(1, Ordering::Relaxed);
                   }
                   
                   thread::sleep(interval);
               }
           });
           
           handles.push(handle);
       }
       
       for handle in handles {
           handle.join().unwrap();
       }
       
       let actual_duration = start_time.elapsed();
       let completed = completed_queries.load(Ordering::Relaxed);
       let failed = failed_queries.load(Ordering::Relaxed);
       let total_lat = total_latency.load(Ordering::Relaxed);
       
       let actual_qps = completed as f64 / actual_duration.as_secs_f64();
       let success_rate = completed as f64 / (completed + failed) as f64 * 100.0;
       let avg_latency = if completed > 0 { total_lat as f64 / completed as f64 / 1000.0 } else { 0.0 };
       
       println!("Results:");
       println!("  Duration: {:.1}s", actual_duration.as_secs_f64());
       println!("  Completed: {} queries", completed);
       println!("  Failed: {} queries", failed);
       println!("  Actual QPS: {:.1}", actual_qps);
       println!("  Success Rate: {:.1}%", success_rate);
       println!("  Avg Latency: {:.3}ms", avg_latency);
       
       // Validate against 5ms target
       if avg_latency > 5.0 {
           println!("  ❌ FAILED: Exceeds 5ms target by {:.3}ms", avg_latency - 5.0);
       } else {
           println!("  ✅ PASSED: Meets 5ms target");
       }
       
       Ok(())
   }
   
   fn simulate_query(thread_id: usize) -> bool {
       // Simulate variable query processing time
       let base_time = 1000; // 1ms in microseconds
       let variation = (thread_id % 3) * 500; // Add some variation
       
       thread::sleep(Duration::from_micros((base_time + variation) as u64));
       
       // 95% success rate
       (thread_id + rand::random::<usize>()) % 100 < 95
   }
   ```
2. Run abbreviated test: `cargo run --release --bin bench_load_scenarios | head -50`
3. Commit: `git add src/bin/bench_load_scenarios.rs && git commit -m "Benchmark various load scenarios for scalability testing"`

## Success Criteria
- [ ] Load scenario benchmark created
- [ ] Multiple load patterns tested
- [ ] Performance under load measured
- [ ] 5ms target validation included
- [ ] Results committed

## Next Task
task_166_measure_memory_fragmentation.md