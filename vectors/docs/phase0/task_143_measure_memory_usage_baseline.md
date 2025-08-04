# Micro-Task 143: Measure Memory Usage Baseline

## Objective
Establish comprehensive memory usage baselines for vector search operations, ensuring system stays within the 1GB memory limit while maintaining <5ms allocation performance.

## Context
Memory usage directly impacts allocation performance. This task measures memory consumption patterns to establish baselines that validate both the 1GB memory target and <5ms allocation performance remain achievable under various memory loads.

## Prerequisites
- Task 142 completed (Basic performance baseline established)
- System monitoring tools operational
- Memory profiling capabilities configured
- Baseline measurement framework available

## Time Estimate
10 minutes

## Instructions
1. Create memory usage benchmark `memory_baseline.rs`:
   ```rust
   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
   use std::hint::black_box;
   use std::collections::HashMap;
   use sysinfo::{System, SystemExt, ProcessExt};
   
   struct MemoryTracker {
       system: System,
       initial_memory: u64,
   }
   
   impl MemoryTracker {
       fn new() -> Self {
           let mut system = System::new_all();
           system.refresh_all();
           
           let initial_memory = if let Some(process) = system.process(sysinfo::get_current_pid().unwrap()) {
               process.memory()
           } else {
               0
           };
           
           Self { system, initial_memory }
       }
       
       fn current_memory_mb(&mut self) -> f64 {
           self.system.refresh_process(sysinfo::get_current_pid().unwrap());
           if let Some(process) = self.system.process(sysinfo::get_current_pid().unwrap()) {
               process.memory() as f64 / 1024.0 / 1024.0 // Convert to MB
           } else {
               0.0
           }
       }
       
       fn memory_delta_mb(&mut self) -> f64 {
           let current = self.current_memory_mb();
           let initial_mb = self.initial_memory as f64 / 1024.0 / 1024.0;
           current - initial_mb
       }
   }
   
   fn memory_baseline_single_vector(c: &mut Criterion) {
       let mut group = c.benchmark_group("memory_baseline_single_vector");
       
       for vector_size in [128, 256, 512, 1024, 2048].iter() {
           group.bench_with_input(
               BenchmarkId::new("vector_memory_usage", vector_size),
               vector_size,
               |b, &size| {
                   let mut tracker = MemoryTracker::new();
                   let initial_memory = tracker.current_memory_mb();
                   
                   b.iter(|| {
                       let start = std::time::Instant::now();
                       
                       // Allocate vector and measure memory
                       let vector = black_box(vec![1.0f32; size]);
                       let allocation_time = start.elapsed();
                       
                       // Measure memory after allocation
                       let memory_used = tracker.current_memory_mb();
                       let memory_delta = memory_used - initial_memory;
                       
                       // Validate allocation time still meets 5ms target
                       assert!(allocation_time.as_millis() < 5,
                               "Vector allocation exceeded 5ms: {}μs for {} elements",
                               allocation_time.as_micros(), size);
                       
                       // Validate memory usage is reasonable
                       let expected_mb = (size * 4) as f64 / 1024.0 / 1024.0; // 4 bytes per f32
                       assert!(memory_delta <= expected_mb * 2.0, // Allow 2x overhead
                               "Memory usage too high: {:.2}MB for expected {:.2}MB",
                               memory_delta, expected_mb);
                       
                       // Ensure we're well under 1GB limit
                       assert!(memory_used < 1024.0,
                               "Memory usage exceeded 1GB limit: {:.2}MB", memory_used);
                       
                       vector
                   });
               }
           );
       }
       
       group.finish();
   }
   
   fn memory_baseline_multiple_concepts(c: &mut Criterion) {
       c.bench_function("memory_baseline_multiple_neuromorphic_concepts", |b| {
           let mut tracker = MemoryTracker::new();
           
           b.iter(|| {
               let start = std::time::Instant::now();
               let initial_memory = tracker.current_memory_mb();
               
               // Simulate storing multiple neuromorphic concepts
               let mut concepts = Vec::new();
               for i in 0..100 {
                   let concept = black_box(vec![i as f32; 512]);
                   let metadata = black_box(format!("concept_{}", i));
                   concepts.push((concept, metadata));
               }
               
               let allocation_time = start.elapsed();
               let final_memory = tracker.current_memory_mb();
               let memory_delta = final_memory - initial_memory;
               
               // Validate timing
               assert!(allocation_time.as_millis() < 50, // Allow 50ms for 100 concepts
                       "Multiple concept allocation too slow: {}ms", allocation_time.as_millis());
               
               // Validate memory usage
               let expected_mb = (100 * 512 * 4) as f64 / 1024.0 / 1024.0; // ~200MB expected
               assert!(memory_delta <= expected_mb * 2.0,
                       "Memory delta too high: {:.2}MB vs expected {:.2}MB",
                       memory_delta, expected_mb);
               
               // Ensure total memory under limit
               assert!(final_memory < 1024.0,
                       "Total memory exceeded 1GB: {:.2}MB", final_memory);
               
               black_box(concepts)
           });
       });
   }
   
   fn memory_baseline_fragmentation_test(c: &mut Criterion) {
       c.bench_function("memory_baseline_fragmentation_resistance", |b| {
           let mut tracker = MemoryTracker::new();
           
           b.iter(|| {
               let start = std::time::Instant::now();
               let initial_memory = tracker.current_memory_mb();
               
               // Create fragmentation pattern: allocate, deallocate, allocate
               let mut allocations = Vec::new();
               
               // Phase 1: Initial allocations
               for i in 0..50 {
                   allocations.push(black_box(vec![i as f32; 256]));
               }
               
               // Phase 2: Deallocate every other allocation
               for i in (0..allocations.len()).step_by(2) {
                   allocations[i].clear();
                   allocations[i].shrink_to_fit();
               }
               
               // Phase 3: New allocations in gaps
               for i in 0..25 {
                   allocations.push(black_box(vec![(i + 100) as f32; 256]));
               }
               
               let total_time = start.elapsed();
               let final_memory = tracker.current_memory_mb();
               
               // Validate fragmentation doesn't kill performance
               assert!(total_time.as_millis() < 20,
                       "Fragmentation test too slow: {}ms", total_time.as_millis());
               
               // Memory should not grow excessively due to fragmentation
               let memory_delta = final_memory - initial_memory;
               assert!(memory_delta < 100.0, // Max 100MB delta allowed
                       "Memory fragmentation too high: {:.2}MB delta", memory_delta);
               
               black_box(allocations)
           });
       });
   }
   
   fn memory_baseline_concurrent_usage(c: &mut Criterion) {
       use std::sync::Arc;
       use std::thread;
       
       c.bench_function("memory_baseline_concurrent_access", |b| {
           let mut tracker = MemoryTracker::new();
           
           b.iter(|| {
               let start = std::time::Instant::now();
               let initial_memory = tracker.current_memory_mb();
               
               // Create shared data structure
               let shared_concepts = Arc::new(std::sync::Mutex::new(Vec::<Vec<f32>>::new()));
               
               let handles: Vec<_> = (0..4).map(|thread_id| {
                   let concepts = Arc::clone(&shared_concepts);
                   thread::spawn(move || {
                       // Each thread adds concepts
                       for i in 0..25 {
                           let concept = vec![(thread_id * 100 + i) as f32; 256];
                           let mut guard = concepts.lock().unwrap();
                           guard.push(concept);
                       }
                   })
               }).collect();
               
               for handle in handles {
                   handle.join().unwrap();
               }
               
               let total_time = start.elapsed();
               let final_memory = tracker.current_memory_mb();
               let memory_delta = final_memory - initial_memory;
               
               // Validate concurrent access doesn't degrade significantly
               assert!(total_time.as_millis() < 30,
                       "Concurrent memory access too slow: {}ms", total_time.as_millis());
               
               // Validate memory usage for concurrent scenario
               let expected_mb = (4 * 25 * 256 * 4) as f64 / 1024.0 / 1024.0; // ~100MB
               assert!(memory_delta <= expected_mb * 2.0,
                       "Concurrent memory usage too high: {:.2}MB vs expected {:.2}MB",
                       memory_delta, expected_mb);
               
               black_box(shared_concepts)
           });
       });
   }
   
   criterion_group!(
       memory_benches,
       memory_baseline_single_vector,
       memory_baseline_multiple_concepts,
       memory_baseline_fragmentation_test,
       memory_baseline_concurrent_usage
   );
   criterion_main!(memory_benches);
   ```
2. Create memory monitoring utility `memory_monitor.rs`:
   ```rust
   use sysinfo::{System, SystemExt, ProcessExt, Pid};
   use std::time::{Duration, Instant};
   use serde::{Serialize, Deserialize};
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct MemorySnapshot {
       pub timestamp: u64,
       pub process_memory_mb: f64,
       pub system_memory_mb: f64,
       pub available_memory_mb: f64,
       pub memory_utilization_percent: f64,
   }
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct MemoryProfile {
       pub snapshots: Vec<MemorySnapshot>,
       pub peak_memory_mb: f64,
       pub average_memory_mb: f64,
       pub memory_growth_rate_mb_per_sec: f64,
       pub meets_1gb_limit: bool,
   }
   
   pub struct MemoryMonitor {
       system: System,
       start_time: Instant,
       snapshots: Vec<MemorySnapshot>,
   }
   
   impl MemoryMonitor {
       pub fn new() -> Self {
           let mut system = System::new_all();
           system.refresh_all();
           
           Self {
               system,
               start_time: Instant::now(),
               snapshots: Vec::new(),
           }
       }
       
       pub fn take_snapshot(&mut self) -> MemorySnapshot {
           self.system.refresh_all();
           
           let process_memory = if let Some(process) = self.system.process(sysinfo::get_current_pid().unwrap()) {
               process.memory() as f64 / 1024.0 / 1024.0
           } else {
               0.0
           };
           
           let total_memory = self.system.total_memory() as f64 / 1024.0 / 1024.0;
           let available_memory = self.system.available_memory() as f64 / 1024.0 / 1024.0;
           let utilization = ((total_memory - available_memory) / total_memory) * 100.0;
           
           let snapshot = MemorySnapshot {
               timestamp: self.start_time.elapsed().as_millis() as u64,
               process_memory_mb: process_memory,
               system_memory_mb: total_memory,
               available_memory_mb: available_memory,
               memory_utilization_percent: utilization,
           };
           
           self.snapshots.push(snapshot.clone());
           snapshot
       }
       
       pub fn continuous_monitoring<F>(&mut self, duration: Duration, operation: F) -> MemoryProfile
       where F: FnOnce() {
           let monitoring_start = Instant::now();
           
           // Take initial snapshot
           self.take_snapshot();
           
           // Start background monitoring
           let handle = {
               let mut monitor_clone = Self::new();
               std::thread::spawn(move || {
                   while monitoring_start.elapsed() < duration {
                       monitor_clone.take_snapshot();
                       std::thread::sleep(Duration::from_millis(100)); // Sample every 100ms
                   }
                   monitor_clone.snapshots
               })
           };
           
           // Execute the operation
           operation();
           
           // Wait for monitoring to complete and collect results
           let background_snapshots = handle.join().unwrap_or_default();
           self.snapshots.extend(background_snapshots);
           
           // Take final snapshot
           self.take_snapshot();
           
           self.generate_profile()
       }
       
       pub fn generate_profile(&self) -> MemoryProfile {
           if self.snapshots.is_empty() {
               return MemoryProfile {
                   snapshots: Vec::new(),
                   peak_memory_mb: 0.0,
                   average_memory_mb: 0.0,
                   memory_growth_rate_mb_per_sec: 0.0,
                   meets_1gb_limit: true,
               };
           }
           
           let peak_memory = self.snapshots.iter()
               .map(|s| s.process_memory_mb)
               .fold(0.0f64, f64::max);
           
           let average_memory = self.snapshots.iter()
               .map(|s| s.process_memory_mb)
               .sum::<f64>() / self.snapshots.len() as f64;
           
           let growth_rate = if self.snapshots.len() > 1 {
               let first = &self.snapshots[0];
               let last = &self.snapshots[self.snapshots.len() - 1];
               let time_diff_sec = (last.timestamp - first.timestamp) as f64 / 1000.0;
               let memory_diff = last.process_memory_mb - first.process_memory_mb;
               
               if time_diff_sec > 0.0 {
                   memory_diff / time_diff_sec
               } else {
                   0.0
               }
           } else {
               0.0
           };
           
           MemoryProfile {
               snapshots: self.snapshots.clone(),
               peak_memory_mb: peak_memory,
               average_memory_mb: average_memory,  
               memory_growth_rate_mb_per_sec: growth_rate,
               meets_1gb_limit: peak_memory <= 1024.0,
           }
       }
   }
   ```
3. Create memory baseline execution script `run_memory_baseline.bat`:
   ```batch
   @echo off
   echo ========================================
   echo Vector Search Memory Usage Baseline
   echo ========================================
   
   echo Setting up memory monitoring environment...
   echo Target: Stay within 1GB memory limit
   echo Allocation target: Still maintain ^<5ms performance
   echo.
   
   echo Running memory baseline benchmarks...
   cargo bench --bench memory_baseline -- --output-format=json > memory_baseline_results.json
   if %ERRORLEVEL% EQU 0 (
       echo ✓ Memory baseline benchmarks completed
   ) else (
       echo ✗ Memory baseline benchmarks failed
       exit /b 1
   )
   
   echo.
   echo Checking memory compliance...
   findstr "exceeded 1GB" memory_baseline_results.json >nul
   if %ERRORLEVEL% EQU 0 (
       echo ⚠ WARNING: Some tests exceeded 1GB memory limit
   ) else (
       echo ✓ All tests stayed within 1GB memory limit
   )
   
   echo.
   echo Checking allocation performance...
   findstr "exceeded 5ms" memory_baseline_results.json >nul
   if %ERRORLEVEL% EQU 0 (
       echo ⚠ WARNING: Some allocations exceeded 5ms target
   ) else (
       echo ✓ All allocations met 5ms performance target
   )
   
   echo.
   echo Memory baseline measurement complete!
   echo Results: memory_baseline_results.json
   echo HTML reports: target\criterion\
   echo ========================================
   ```
4. Test memory baseline: `run_memory_baseline.bat`
5. Analyze memory usage patterns in generated reports
6. Commit memory baseline: `git add . && git commit -m "Establish memory usage baseline with 1GB limit validation"`

## Expected Output
- Comprehensive memory usage benchmark suite
- Memory monitoring and profiling utilities
- Validation that memory usage stays within 1GB limit
- Confirmation that <5ms allocation performance maintained under memory load

## Success Criteria
- [ ] Memory baseline benchmarks execute successfully
- [ ] Memory usage tracking functional
- [ ] 1GB memory limit validation working
- [ ] <5ms allocation performance maintained
- [ ] Fragmentation resistance measured
- [ ] Concurrent memory access profiled
- [ ] Memory growth rates calculated
- [ ] Baseline committed to git

## Validation Commands
```batch
# Run memory baseline
run_memory_baseline.bat

# Check specific memory benchmark
cargo bench --bench memory_baseline memory_baseline_single_vector

# Analyze memory reports
type memory_baseline_results.json | findstr "memory"

# Check HTML memory reports
dir target\criterion\memory_baseline*
```

## Next Task
task_144_measure_cpu_utilization_baseline.md

## Notes
- Memory and performance are interconnected - both must be validated together
- 1GB limit ensures system scalability
- Memory fragmentation can impact allocation performance
- Concurrent access patterns reveal real-world memory behavior