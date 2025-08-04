# Micro-Task 138: Create Baseline Measurement Framework

## Objective
Implement the core measurement framework for capturing performance baselines with focus on the <5ms neuromorphic concept allocation target.

## Context
This framework provides standardized measurement collection for all benchmarking tasks. It ensures consistent metrics collection and statistical analysis for performance validation.

## Prerequisites
- Task 137 completed (Benchmark environment setup)
- Criterion crate available in workspace
- Benchmark configuration established

## Time Estimate
10 minutes

## Instructions
1. Navigate to benchmarks directory: `cd C:\code\LLMKG\vectors\benches`
2. Create baseline measurement framework `baseline_framework.rs`:
   ```rust
   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
   use std::time::{Duration, Instant};
   use serde::{Serialize, Deserialize};
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct PerformanceMetrics {
       pub allocation_time_ms: f64,
       pub memory_usage_mb: f64,
       pub cpu_usage_percent: f64,
       pub concurrent_operations: u32,
   }
   
   #[derive(Debug)]
   pub struct BenchmarkConfig {
       pub target_allocation_ms: f64,
       pub memory_limit_mb: u64,
       pub max_cpu_percent: f64,
   }
   
   impl Default for BenchmarkConfig {
       fn default() -> Self {
           Self {
               target_allocation_ms: 5.0, // <5ms target
               memory_limit_mb: 1024,
               max_cpu_percent: 80.0,
           }
       }
   }
   
   pub fn measure_allocation_time<F>(operation: F) -> Duration 
   where F: FnOnce() -> () {
       let start = Instant::now();
       operation();
       start.elapsed()
   }
   
   pub fn validate_allocation_target(duration: Duration, config: &BenchmarkConfig) -> bool {
       duration.as_millis() as f64 <= config.target_allocation_ms
   }
   
   pub fn baseline_allocation_benchmark(c: &mut Criterion) {
       let config = BenchmarkConfig::default();
       
       c.bench_function("neuromorphic_concept_allocation", |b| {
           b.iter(|| {
               // Mock neuromorphic concept allocation
               let duration = measure_allocation_time(|| {
                   std::hint::black_box(Vec::<u8>::with_capacity(1024));
               });
               assert!(validate_allocation_target(duration, &config));
           })
       });
   }
   
   criterion_group!(benches, baseline_allocation_benchmark);
   criterion_main!(benches);
   ```
3. Create metrics collection module `metrics_collector.rs`:
   ```rust
   use std::time::{Duration, SystemTime};
   use sysinfo::{System, SystemExt, ProcessExt};
   
   pub struct MetricsCollector {
       system: System,
   }
   
   impl MetricsCollector {
       pub fn new() -> Self {
           let mut system = System::new_all();
           system.refresh_all();
           Self { system }
       }
       
       pub fn collect_memory_usage(&mut self) -> f64 {
           self.system.refresh_memory();
           self.system.used_memory() as f64 / 1024.0 / 1024.0 // MB
       }
       
       pub fn collect_cpu_usage(&mut self) -> f64 {
           self.system.refresh_cpu();
           self.system.global_cpu_info().cpu_usage() as f64
       }
   }
   ```
4. Update `Cargo.toml` to include benchmark dependencies:
   ```toml
   [dev-dependencies]
   criterion = { version = "0.5", features = ["html_reports"] }
   sysinfo = "0.29"
   serde = { version = "1.0", features = ["derive"] }
   ```
5. Test framework: `cargo bench --bench baseline_framework`
6. Commit framework: `git add benches/ && git commit -m "Add baseline measurement framework with 5ms allocation validation"`

## Expected Output
- Baseline measurement framework implemented
- Metrics collection system created
- <5ms allocation validation function
- Statistical measurement utilities

## Success Criteria
- [ ] Framework compiles without errors
- [ ] Allocation time measurement works
- [ ] <5ms validation function implemented
- [ ] Metrics collection operational
- [ ] Benchmark execution completes successfully
- [ ] Framework committed to git

## Validation Commands
```batch
# Test framework compilation
cargo check --benches

# Run baseline benchmark
cargo bench --bench baseline_framework

# Verify HTML reports generation
dir target\criterion\
```

## Next Task
task_139_configure_windows_performance_monitoring.md

## Notes
- Framework focuses on the critical <5ms allocation target
- Windows-specific performance monitoring included
- Statistical significance built into measurement
- HTML reports provide visual performance analysis