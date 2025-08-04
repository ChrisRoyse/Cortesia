# Micro-Task 166: Measure Memory Fragmentation

## Objective
Measure memory fragmentation patterns and their impact on allocation performance.

## Prerequisites
- Task 165 completed (Load scenarios benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create memory fragmentation measurement `measure_fragmentation.rs`:
   ```rust
   use std::collections::HashMap;
   use std::time::{Duration, Instant};
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Measuring memory fragmentation impact...");
       
       measure_allocation_patterns()?;
       measure_fragmentation_impact()?;
       measure_cleanup_effectiveness()?;
       
       Ok(())
   }
   
   fn measure_allocation_patterns() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Allocation Pattern Analysis ===");
       
       // Test different allocation patterns
       test_sequential_allocation("Sequential", 1000, 1024)?;
       test_random_allocation("Random", 1000, 1024)?;
       test_mixed_allocation("Mixed", 1000, 1024)?;
       
       Ok(())
   }
   
   fn test_sequential_allocation(pattern: &str, count: usize, size: usize) -> Result<(), Box<dyn std::error::Error>> {
       let start = Instant::now();
       let mut allocations = Vec::new();
       
       for _ in 0..count {
           let vec: Vec<u8> = vec![0; size];
           allocations.push(vec);
       }
       
       let duration = start.elapsed();
       let avg_alloc_time = duration.as_micros() as f64 / count as f64;
       
       println!("{} allocation: {:.3}μs avg ({} x {}B)", 
               pattern, avg_alloc_time, count, size);
       
       Ok(())
   }
   
   fn test_random_allocation(pattern: &str, count: usize, base_size: usize) -> Result<(), Box<dyn std::error::Error>> {
       let start = Instant::now();
       let mut allocations = Vec::new();
       
       for i in 0..count {
           // Random size between base_size/2 and base_size*2
           let size = base_size / 2 + (i * 17) % (base_size * 3 / 2);
           let vec: Vec<u8> = vec![0; size];
           allocations.push(vec);
       }
       
       let duration = start.elapsed();
       let avg_alloc_time = duration.as_micros() as f64 / count as f64;
       
       println!("{} allocation: {:.3}μs avg ({} random sizes)", 
               pattern, avg_alloc_time, count);
       
       Ok(())
   }
   
   fn test_mixed_allocation(pattern: &str, count: usize, base_size: usize) -> Result<(), Box<dyn std::error::Error>> {
       let start = Instant::now();
       let mut allocations = Vec::new();
       
       for i in 0..count {
           let size = match i % 4 {
               0 => base_size / 4,     // Small
               1 => base_size,         // Medium
               2 => base_size * 4,     // Large
               _ => base_size * 16,    // Very large
           };
           let vec: Vec<u8> = vec![0; size];
           allocations.push(vec);
       }
       
       let duration = start.elapsed();
       let avg_alloc_time = duration.as_micros() as f64 / count as f64;
       
       println!("{} allocation: {:.3}μs avg ({} mixed sizes)", 
               pattern, avg_alloc_time, count);
       
       Ok(())
   }
   
   fn measure_fragmentation_impact() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Fragmentation Impact Analysis ===");
       
       // Create fragmentation by allocating and deallocating
       let mut large_allocs = Vec::new();
       let mut small_allocs = Vec::new();
       
       println!("Creating fragmentation pattern...");
       
       // Allocate large blocks
       for _ in 0..100 {
           large_allocs.push(vec![0u8; 1024 * 1024]); // 1MB blocks
       }
       
       // Free every other large block to create gaps
       let mut i = 0;
       large_allocs.retain(|_| {
           i += 1;
           i % 2 == 0
       });
       
       println!("Measuring allocation performance in fragmented state...");
       
       // Now try to allocate small blocks (should fit in gaps)
       let start = Instant::now();
       for _ in 0..1000 {
           small_allocs.push(vec![0u8; 512 * 1024]); // 512KB blocks
       }
       let fragmented_time = start.elapsed();
       
       // Clear and measure clean allocation
       large_allocs.clear();
       small_allocs.clear();
       
       let start = Instant::now();
       for _ in 0..1000 {
           small_allocs.push(vec![0u8; 512 * 1024]); // 512KB blocks
       }
       let clean_time = start.elapsed();
       
       let fragmented_avg = fragmented_time.as_micros() as f64 / 1000.0;
       let clean_avg = clean_time.as_micros() as f64 / 1000.0;
       let overhead = ((fragmented_avg - clean_avg) / clean_avg) * 100.0;
       
       println!("Clean allocation: {:.3}μs avg", clean_avg);
       println!("Fragmented allocation: {:.3}μs avg", fragmented_avg);
       println!("Fragmentation overhead: {:.1}%", overhead);
       
       Ok(())
   }
   
   fn measure_cleanup_effectiveness() -> Result<(), Box<dyn std::error::Error>> {
       println!("\n=== Cleanup Effectiveness Analysis ===");
       
       let mut memory_tracker = HashMap::new();
       
       // Simulate allocation/deallocation cycles
       for cycle in 0..10 {
           let start = Instant::now();
           
           // Allocate phase
           let mut allocations = Vec::new();
           for i in 0..100 {
               let size = (i % 5 + 1) * 1024; // 1KB to 5KB
               allocations.push(vec![0u8; size]);
           }
           
           let alloc_time = start.elapsed();
           
           // Cleanup phase
           let cleanup_start = Instant::now();
           allocations.clear(); // Force deallocation
           let cleanup_time = cleanup_start.elapsed();
           
           memory_tracker.insert(cycle, (alloc_time, cleanup_time));
           
           println!("Cycle {}: Alloc={:.3}ms, Cleanup={:.3}ms", 
                   cycle, alloc_time.as_millis(), cleanup_time.as_micros() as f64 / 1000.0);
       }
       
       // Calculate averages
       let total_cycles = memory_tracker.len() as f64;
       let avg_alloc: f64 = memory_tracker.values().map(|(a, _)| a.as_micros() as f64).sum::<f64>() / total_cycles;
       let avg_cleanup: f64 = memory_tracker.values().map(|(_, c)| c.as_micros() as f64).sum::<f64>() / total_cycles;
       
       println!("\nAverage allocation time: {:.3}μs", avg_alloc);
       println!("Average cleanup time: {:.3}μs", avg_cleanup);
       
       Ok(())
   }
   ```
2. Run: `cargo run --release --bin measure_fragmentation`
3. Commit: `git add src/bin/measure_fragmentation.rs && git commit -m "Measure memory fragmentation impact on allocation performance"`

## Success Criteria
- [ ] Fragmentation measurement tool created
- [ ] Different allocation patterns tested
- [ ] Fragmentation impact quantified
- [ ] Cleanup effectiveness measured
- [ ] Results committed

## Next Task
task_167_benchmark_garbage_collection.md