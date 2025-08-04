# Micro-Task 155: Benchmark I/O Operations

## Objective
Measure file I/O performance for index loading and data persistence operations.

## Prerequisites
- Task 154 completed (Cache performance measured)

## Time Estimate
7 minutes

## Instructions
1. Create I/O benchmark `bench_io.rs`:
   ```rust
   use std::fs;
   use std::time::Instant;
   use std::io::Write;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking I/O operations...");
       
       benchmark_file_write(1024, "1KB")?;
       benchmark_file_write(1024 * 1024, "1MB")?;
       benchmark_file_write(10 * 1024 * 1024, "10MB")?;
       
       benchmark_file_read("1KB")?;
       benchmark_file_read("1MB")?;
       benchmark_file_read("10MB")?;
       
       Ok(())
   }
   
   fn benchmark_file_write(size: usize, label: &str) -> Result<(), Box<dyn std::error::Error>> {
       let data = vec![0u8; size];
       let filename = format!("temp_{}.dat", label);
       
       let start = Instant::now();
       fs::write(&filename, data)?;
       let duration = start.elapsed();
       
       let throughput_mbps = (size as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
       
       println!("Write {}: {:.3}ms ({:.1} MB/s)", label, duration.as_millis(), throughput_mbps);
       
       Ok(())
   }
   
   fn benchmark_file_read(label: &str) -> Result<(), Box<dyn std::error::Error>> {
       let filename = format!("temp_{}.dat", label);
       
       let start = Instant::now();
       let data = fs::read(&filename)?;
       let duration = start.elapsed();
       
       let throughput_mbps = (data.len() as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
       
       println!("Read {}: {:.3}ms ({:.1} MB/s)", label, duration.as_millis(), throughput_mbps);
       
       // Cleanup
       fs::remove_file(&filename).ok();
       
       Ok(())
   }
   ```
2. Run: `cargo run --release --bin bench_io`
3. Commit: `git add src/bin/bench_io.rs && git commit -m "Benchmark I/O operations for index persistence"`

## Success Criteria
- [ ] I/O benchmark created
- [ ] Read/write performance measured
- [ ] Throughput calculated
- [ ] Results committed

## Next Task
task_156_measure_serialization_performance.md