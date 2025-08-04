# Micro-Task 025: Setup Performance Monitoring

## Objective
Configure performance monitoring and profiling tools for development and testing.

## Context
Performance monitoring is crucial for a vector search system. This task sets up tools and configurations for tracking performance metrics, memory usage, and identifying bottlenecks during development.

## Prerequisites
- Task 024 completed (Environment documentation created)
- Benchmark framework configured
- Development tools installed

## Time Estimate
9 minutes

## Instructions
1. Create `perf_monitor.toml` configuration:
   ```toml
   [monitoring]
   enabled = true
   sample_rate = 1000  # samples per second
   memory_tracking = true
   cpu_profiling = true
   
   [thresholds]
   max_memory_mb = 1024
   max_cpu_percent = 80
   max_response_time_ms = 100
   
   [output]
   directory = "data/logs"
   metrics_file = "performance_metrics.json"
   profiling_file = "profiling_data.prof"
   ```
2. Create performance monitoring utility `src/perf_utils.rs`:
   ```rust
   //! Performance monitoring utilities
   
   use std::time::{Duration, Instant};
   use std::sync::atomic::{AtomicU64, Ordering};
   
   /// Simple performance counter
   pub struct PerfCounter {
       start_time: Instant,
       operation_count: AtomicU64,
       total_duration: AtomicU64,
   }
   
   impl PerfCounter {
       pub fn new() -> Self {
           Self {
               start_time: Instant::now(),
               operation_count: AtomicU64::new(0),
               total_duration: AtomicU64::new(0),
           }
       }
       
       pub fn record_operation(&self, duration: Duration) {
           self.operation_count.fetch_add(1, Ordering::Relaxed);
           self.total_duration.fetch_add(
               duration.as_nanos() as u64,
               Ordering::Relaxed
           );
       }
       
       pub fn average_duration(&self) -> Duration {
           let count = self.operation_count.load(Ordering::Relaxed);
           let total = self.total_duration.load(Ordering::Relaxed);
           
           if count == 0 {
               Duration::from_nanos(0)
           } else {
               Duration::from_nanos(total / count)
           }
       }
       
       pub fn operations_per_second(&self) -> f64 {
           let elapsed = self.start_time.elapsed().as_secs_f64();
           let count = self.operation_count.load(Ordering::Relaxed) as f64;
           
           if elapsed == 0.0 {
               0.0
           } else {
               count / elapsed
           }
       }
   }
   
   #[cfg(test)]
   mod tests {
       use super::*;
       use std::thread;
       
       #[test]
       fn test_perf_counter() {
           let counter = PerfCounter::new();
           
           // Simulate some operations
           for _ in 0..5 {
               let start = Instant::now();
               thread::sleep(Duration::from_millis(1));
               counter.record_operation(start.elapsed());
           }
           
           assert!(counter.average_duration() > Duration::from_nanos(0));
           assert!(counter.operations_per_second() > 0.0);
       }
   }
   ```
3. Test performance utilities:
   - `rustc --test src/perf_utils.rs`
   - `./perf_utils.exe` (runs tests)
4. Clean up test: `del perf_utils.exe`
5. Commit performance monitoring: `git add perf_monitor.toml src/perf_utils.rs && git commit -m "Setup performance monitoring"`

## Expected Output
- Performance monitoring configuration created
- Performance utilities implemented and tested
- Monitoring tools ready for development use
- Configuration committed to repository

## Success Criteria
- [ ] `perf_monitor.toml` configuration file created
- [ ] Performance utility functions implemented
- [ ] Performance utilities compile and test successfully
- [ ] Monitoring setup committed to Git

## Next Task
task_026_configure_memory_profiling.md