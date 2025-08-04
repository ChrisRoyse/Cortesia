# Micro-Task 148: Measure Allocation Latency

## Objective
Measure detailed allocation latency to validate the <5ms allocation target across different scenarios.

## Context
Allocation latency is the critical metric for validating system performance. This task measures precise timing for allocation operations.

## Prerequisites
- Task 147 completed (Performance baseline established)
- Profiling tools configured
- Benchmark datasets available

## Time Estimate
9 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Create allocation latency measurement `measure_allocation_latency.rs`:
   ```rust
   use std::time::{Duration, Instant};
   use std::collections::HashMap;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Measuring allocation latency...");
       
       let mut results = HashMap::new();
       
       // Test different allocation scenarios
       results.insert("small_vector", measure_vector_allocation(100)?);
       results.insert("medium_vector", measure_vector_allocation(1000)?);
       results.insert("large_vector", measure_vector_allocation(10000)?);
       
       results.insert("hashmap_small", measure_hashmap_allocation(100)?);
       results.insert("hashmap_medium", measure_hashmap_allocation(1000)?);
       results.insert("hashmap_large", measure_hashmap_allocation(10000)?);
       
       results.insert("string_small", measure_string_allocation(100)?);
       results.insert("string_medium", measure_string_allocation(1000)?);
       results.insert("string_large", measure_string_allocation(10000)?);
       
       // Generate report
       generate_latency_report(&results)?;
       
       // Validate 5ms target
       validate_allocation_target(&results)?;
       
       Ok(())
   }
   
   fn measure_vector_allocation(size: usize) -> Result<Duration, Box<dyn std::error::Error>> {
       let iterations = 1000;
       let mut total = Duration::new(0, 0);
       
       for _ in 0..iterations {
           let start = Instant::now();
           let _vec: Vec<u64> = Vec::with_capacity(size);
           total += start.elapsed();
       }
       
       Ok(total / iterations as u32)
   }
   
   fn measure_hashmap_allocation(size: usize) -> Result<Duration, Box<dyn std::error::Error>> {
       let iterations = 1000;
       let mut total = Duration::new(0, 0);
       
       for _ in 0..iterations {
           let start = Instant::now();
           let _map: HashMap<usize, String> = HashMap::with_capacity(size);
           total += start.elapsed();
       }
       
       Ok(total / iterations as u32)
   }
   
   fn measure_string_allocation(size: usize) -> Result<Duration, Box<dyn std::error::Error>> {
       let iterations = 1000;
       let mut total = Duration::new(0, 0);
       
       for _ in 0..iterations {
           let start = Instant::now();
           let _string = String::with_capacity(size);
           total += start.elapsed();
       }
       
       Ok(total / iterations as u32)
   }
   
   fn generate_latency_report(results: &HashMap<&str, Duration>) -> Result<(), Box<dyn std::error::Error>> {
       use std::fs;
       
       let mut report = String::new();
       report.push_str(&format!("# Allocation Latency Report\n"));
       report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now()));
       
       for (test, latency) in results {
           let micros = latency.as_micros();
           let millis = latency.as_millis();
           report.push_str(&format!("{}: {}μs ({:.3}ms)\n", test, micros, millis as f64 / 1000.0));
       }
       
       fs::create_dir_all("latency_results")?;
       fs::write("latency_results/allocation_latency_report.md", report)?;
       Ok(())
   }
   
   fn validate_allocation_target(results: &HashMap<&str, Duration>) -> Result<(), Box<dyn std::error::Error>> {
       let target = Duration::from_millis(5);
       let mut violations = Vec::new();
       
       for (test, latency) in results {
           if *latency > target {
               violations.push((test, latency));
           }
       }
       
       if violations.is_empty() {
           println!("✅ All allocations meet 5ms target");
       } else {
           println!("❌ {} allocation(s) exceed 5ms target:", violations.len());
           for (test, latency) in violations {
               println!("  - {}: {:.3}ms", test, latency.as_millis() as f64 / 1000.0);
           }
       }
       
       Ok(())
   }
   ```
3. Create latency measurement script `measure_latency.bat`:
   ```batch
   @echo off
   echo Measuring allocation latency...
   if not exist latency_results mkdir latency_results
   
   echo Running latency measurements...
   cargo run --release --bin measure_allocation_latency
   
   echo Results summary:
   type latency_results\allocation_latency_report.md
   
   echo Latency measurement complete.
   ```
4. Run latency measurement: `measure_latency.bat`
5. Review results: `type latency_results\allocation_latency_report.md`
6. Commit measurements: `git add src/bin/measure_allocation_latency.rs measure_latency.bat latency_results/ && git commit -m "Measure allocation latency for 5ms target validation"`

## Expected Output
- Detailed allocation latency measurements
- Latency report for different allocation types
- 5ms target validation results
- Performance baseline for allocations

## Success Criteria
- [ ] Allocation latency measurement tool created
- [ ] Measurements completed for different scenarios
- [ ] Latency report generated
- [ ] 5ms target validation performed
- [ ] Results committed to repository

## Validation Commands
```batch
# Review latency results
type latency_results\allocation_latency_report.md

# Check measurement completion
dir latency_results

# Verify target validation
cargo run --bin measure_allocation_latency | findstr "target"
```

## Next Task
task_149_benchmark_tantivy_indexing.md

## Notes
- Precise timing requires release mode compilation
- Multiple iterations provide statistical accuracy
- Different allocation types reveal performance patterns
- Target validation ensures performance requirements