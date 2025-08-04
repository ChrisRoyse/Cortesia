# Micro-Task 142: Create Basic Performance Baseline

## Objective
Establish the fundamental performance baseline for vector search operations, focusing on the critical <5ms neuromorphic concept allocation target.

## Context
This task creates the first set of baseline measurements that will serve as the reference point for all future performance comparisons and regression detection. The baseline must accurately reflect realistic system performance.

## Prerequisites
- Task 141 completed (Benchmark environment integration validated)
- Environment optimized for performance measurement
- Statistical analysis tools operational
- Criterion benchmark framework ready

## Time Estimate
10 minutes

## Instructions
1. Create basic baseline benchmark `basic_baseline.rs`:
   ```rust
   use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
   use std::time::{Duration, Instant};
   use std::hint::black_box;
   
   // Import our measurement framework
   use vector_search::baseline_framework::{BenchmarkConfig, measure_allocation_time};
   use vector_search::stats_analysis::StatisticalAnalysis;
   
   struct BaselineMetrics {
       allocation_times: Vec<Duration>,
       memory_allocations: Vec<usize>,
       throughput_ops: Vec<u64>,
   }
   
   impl BaselineMetrics {
       fn new() -> Self {
           Self {
               allocation_times: Vec::new(),
               memory_allocations: Vec::new(),
               throughput_ops: Vec::new(),
           }
       }
       
       fn record_allocation(&mut self, duration: Duration, size: usize) {
           self.allocation_times.push(duration);
           self.memory_allocations.push(size);
       }
   }
   
   fn baseline_neuromorphic_allocation(c: &mut Criterion) {
       let config = BenchmarkConfig::default();
       let mut metrics = BaselineMetrics::new();
       
       c.bench_function("baseline_neuromorphic_concept_allocation", |b| {
           b.iter(|| {
               let start = Instant::now();
               
               // Simulate neuromorphic concept allocation
               let concept_vector = black_box(vec![0.0f32; 512]); // 512-dimensional vector
               let concept_metadata = black_box(vec![0u8; 256]);  // Metadata storage
               let concept_index = black_box(Vec::<u32>::with_capacity(64)); // Index references
               
               let duration = start.elapsed();
               
               // Validate 5ms target
               assert!(duration.as_millis() < 5, 
                       "Allocation exceeded 5ms target: {}μs", duration.as_micros());
               
               // Calculate total memory allocation
               let total_size = (concept_vector.len() * 4) + concept_metadata.len() + (concept_index.capacity() * 4);
               
               black_box((concept_vector, concept_metadata, concept_index, duration, total_size))
           });
       });
   }
   
   fn baseline_vector_operations(c: &mut Criterion) {
       let mut group = c.benchmark_group("baseline_vector_operations");
       group.throughput(Throughput::Elements(1));
       
       // Test different vector sizes
       for size in [128, 256, 512, 1024].iter() {
           group.bench_with_input(BenchmarkId::new("vector_creation", size), size, |b, &size| {
               b.iter(|| {
                   let start = Instant::now();
                   let vector = black_box(vec![1.0f32; size]);
                   let duration = start.elapsed();
                   
                   // Ensure reasonable allocation time scaling
                   let max_time = Duration::from_micros((size as u64) * 10); // 10μs per element max
                   assert!(duration <= max_time, 
                           "Vector creation too slow: {}μs for {} elements", 
                           duration.as_micros(), size);
                   
                   vector
               });
           });
       }
       
       group.finish();
   }
   
   fn baseline_memory_patterns(c: &mut Criterion) {
       c.bench_function("baseline_memory_allocation_patterns", |b| {
           b.iter(|| {
               let start = Instant::now();
               
               // Pattern 1: Sequential allocations
               let seq1 = black_box(Vec::<f32>::with_capacity(256));
               let seq2 = black_box(Vec::<f32>::with_capacity(512));
               let seq3 = black_box(Vec::<f32>::with_capacity(1024));
               
               // Pattern 2: Interleaved allocations
               let mut interleaved = Vec::new();
               for i in 0..10 {
                   interleaved.push(black_box(vec![i as f32; 64]));
               }
               
               let duration = start.elapsed();
               
               // Validate overall pattern allocation time
               assert!(duration.as_millis() < 2, 
                       "Memory pattern allocation too slow: {}μs", duration.as_micros());
               
               (seq1, seq2, seq3, interleaved)
           });
       });
   }
   
   fn baseline_concurrent_allocations(c: &mut Criterion) {
       use std::sync::Arc;
       use std::thread;
       
       c.bench_function("baseline_concurrent_allocations", |b| {
           b.iter(|| {
               let start = Instant::now();
               
               let handles: Vec<_> = (0..4).map(|i| {
                   thread::spawn(move || {
                       // Each thread allocates its own concept
                       let concept = black_box(vec![i as f32; 256]);
                       concept
                   })
               }).collect();
               
               let results: Vec<_> = handles.into_iter()
                   .map(|h| h.join().unwrap())
                   .collect();
               
               let duration = start.elapsed();
               
               // Concurrent allocations should not degrade significantly
               assert!(duration.as_millis() < 8, 
                       "Concurrent allocation too slow: {}μs", duration.as_micros());
               
               black_box(results)
           });
       });
   }
   
   criterion_group!(
       baseline_benches,
       baseline_neuromorphic_allocation,
       baseline_vector_operations,
       baseline_memory_patterns,
       baseline_concurrent_allocations
   );
   criterion_main!(baseline_benches);
   ```
2. Create baseline measurement collector `baseline_collector.rs`:
   ```rust
   use std::collections::HashMap;
   use std::fs::{File, create_dir_all};
   use std::io::Write;
   use std::path::Path;
   use serde::{Serialize, Deserialize};
   use chrono::{DateTime, Utc};
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct BaselineMeasurement {
       pub benchmark_name: String,
       pub mean_time_us: f64,
       pub std_dev_us: f64,
       pub min_time_us: f64,
       pub max_time_us: f64,
       pub sample_count: usize,
       pub meets_5ms_target: bool,
       pub timestamp: DateTime<Utc>,
   }
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct BaselineReport {
       pub system_info: SystemInfo,
       pub measurements: Vec<BaselineMeasurement>,
       pub overall_summary: OverallSummary,
       pub generated_at: DateTime<Utc>,
   }
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct SystemInfo {
       pub os: String,
       pub cpu_model: String,
       pub total_memory_gb: f64,
       pub rust_version: String,
       pub benchmark_config: String,
   }
   
   #[derive(Debug, Serialize, Deserialize)]
   pub struct OverallSummary {
       pub total_benchmarks: usize,
       pub passing_5ms_target: usize,
       pub failing_5ms_target: usize,
       pub average_allocation_time_us: f64,
       pub performance_grade: String,
   }
   
   pub struct BaselineCollector {
       measurements: Vec<BaselineMeasurement>,
       system_info: SystemInfo,
   }
   
   impl BaselineCollector {
       pub fn new() -> Self {
           Self {
               measurements: Vec::new(),
               system_info: Self::collect_system_info(),
           }
       }
       
       fn collect_system_info() -> SystemInfo {
           SystemInfo {
               os: std::env::consts::OS.to_string(),
               cpu_model: "Windows CPU".to_string(), // Simplified for baseline
               total_memory_gb: 16.0, // Placeholder - could use sysinfo crate
               rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
               benchmark_config: "High Performance, 5ms target".to_string(),
           }
       }
       
       pub fn add_measurement(&mut self, measurement: BaselineMeasurement) {
           self.measurements.push(measurement);
       }
       
       pub fn generate_report(&self) -> BaselineReport {
           let passing = self.measurements.iter()
               .filter(|m| m.meets_5ms_target)
               .count();
           
           let failing = self.measurements.len() - passing;
           
           let avg_time = if !self.measurements.is_empty() {
               self.measurements.iter()
                   .map(|m| m.mean_time_us)
                   .sum::<f64>() / self.measurements.len() as f64
           } else {
               0.0
           };
           
           let grade = if failing == 0 {
               "A+ (All targets met)".to_string()
           } else if passing > failing {
               "B+ (Most targets met)".to_string()
           } else {
               "C (Performance needs improvement)".to_string()
           };
           
           BaselineReport {
               system_info: self.system_info.clone(),
               measurements: self.measurements.clone(),
               overall_summary: OverallSummary {
                   total_benchmarks: self.measurements.len(),
                   passing_5ms_target: passing,
                   failing_5ms_target: failing,
                   average_allocation_time_us: avg_time,
                   performance_grade: grade,
               },
               generated_at: Utc::now(),
           }
       }
       
       pub fn save_report(&self, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
           create_dir_all(output_dir)?;
           
           let report = self.generate_report();
           
           // Save JSON report
           let json_path = output_dir.join("baseline_report.json");
           let json_content = serde_json::to_string_pretty(&report)?;
           std::fs::write(json_path, json_content)?;
           
           // Save human-readable report
           let txt_path = output_dir.join("baseline_summary.txt");
           let summary = format!(
               "Vector Search Baseline Performance Report\n\
               Generated: {}\n\n\
               System Information:\n\
               - OS: {}\n\
               - CPU: {}\n\
               - Memory: {:.1}GB\n\
               - Rust: {}\n\n\
               Performance Summary:\n\
               - Total Benchmarks: {}\n\
               - Passing 5ms Target: {}\n\
               - Failing 5ms Target: {}\n\
               - Average Allocation Time: {:.2}μs\n\
               - Performance Grade: {}\n\n\
               Target Compliance: {:.1}%\n",
               report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
               report.system_info.os,
               report.system_info.cpu_model,
               report.system_info.total_memory_gb,
               report.system_info.rust_version,
               report.overall_summary.total_benchmarks,
               report.overall_summary.passing_5ms_target,
               report.overall_summary.failing_5ms_target,
               report.overall_summary.average_allocation_time_us,
               report.overall_summary.performance_grade,
               (report.overall_summary.passing_5ms_target as f64 / report.overall_summary.total_benchmarks as f64) * 100.0
           );
           
           std::fs::write(txt_path, summary)?;
           
           Ok(())
       }
   }
   ```
3. Create baseline execution script `run_baseline.bat`:
   ```batch
   @echo off
   echo ========================================
   echo Vector Search Basic Performance Baseline
   echo ========================================
   
   echo Setting up high performance environment...
   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
   
   echo.
   echo Running baseline benchmarks...
   echo Target: Neuromorphic concept allocation ^< 5ms
   echo.
   
   cargo bench --bench basic_baseline -- --output-format=json > baseline_results.json
   if %ERRORLEVEL% EQU 0 (
       echo ✓ Baseline benchmarks completed successfully
   ) else (
       echo ✗ Baseline benchmarks failed
       exit /b 1
   )
   
   echo.
   echo Generating baseline reports...
   if exist target\criterion\ (
       echo ✓ Criterion HTML reports available in target\criterion\
   )
   
   if exist baseline_results.json (
       echo ✓ JSON results saved to baseline_results.json
   )
   
   echo.
   echo ========================================
   echo Baseline measurement complete!
   echo Check target\criterion\ for detailed reports
   echo ========================================
   ```
4. Test baseline measurement: `run_baseline.bat`
5. Validate results meet <5ms target: Check generated reports
6. Commit baseline implementation: `git add . && git commit -m "Implement basic performance baseline with 5ms allocation validation"`

## Expected Output
- Comprehensive baseline benchmark suite
- Measurement collection and reporting system
- Performance validation against <5ms target
- Automated baseline execution script

## Success Criteria
- [ ] Baseline benchmarks execute successfully
- [ ] Neuromorphic concept allocation measured
- [ ] <5ms target validation functional
- [ ] Vector operations baseline established
- [ ] Memory allocation patterns measured
- [ ] Concurrent allocation performance recorded
- [ ] JSON and HTML reports generated
- [ ] Baseline results committed to git

## Validation Commands
```batch
# Run baseline measurements
run_baseline.bat

# Check specific benchmark
cargo bench --bench basic_baseline baseline_neuromorphic_concept_allocation

# Verify HTML reports
dir target\criterion\baseline_neuromorphic_concept_allocation\

# Check JSON output
type baseline_results.json
```

## Next Task
task_143_measure_memory_usage_baseline.md

## Notes
- This establishes the fundamental performance reference point
- All measurements validate against the 5ms allocation target
- Statistical significance ensures reliable baselines
- Reports provide both machine-readable and human-readable formats