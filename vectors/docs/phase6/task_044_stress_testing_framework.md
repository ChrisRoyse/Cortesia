# Task 044: Stress Testing Framework

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates a stress testing framework that simulates enterprise load conditions and validates system stability under extreme conditions.

## Project Structure
tests/
  stress_testing_framework.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive stress testing framework that validates system behavior under extreme load conditions including high concurrency, memory pressure, and prolonged operation.

## Requirements
1. Create comprehensive integration test
2. Test system under extreme load conditions
3. Validate stability during resource exhaustion scenarios
4. Handle graceful degradation under stress
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use tokio::sync::Semaphore;
use tokio::time::sleep;

#[tokio::test]
async fn test_extreme_concurrency_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let stress_tester = StressTester::new(temp_dir.path()).await?;
    
    // Test with extremely high concurrency (1000 concurrent users)
    let concurrent_users = 1000;
    let queries_per_user = 10;
    
    let success_counter = Arc::new(AtomicU64::new(0));
    let error_counter = Arc::new(AtomicU64::new(0));
    let active_queries = Arc::new(AtomicU64::new(0));
    let max_active = Arc::new(AtomicU64::new(0));
    
    let semaphore = Arc::new(Semaphore::new(concurrent_users));
    let mut handles = Vec::new();
    
    let start_time = Instant::now();
    
    for user_id in 0..concurrent_users {
        let permit = semaphore.clone().acquire_owned().await?;
        let stress_tester_clone = stress_tester.clone();
        let success_counter_clone = success_counter.clone();
        let error_counter_clone = error_counter.clone();
        let active_queries_clone = active_queries.clone();
        let max_active_clone = max_active.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = permit;
            
            for query_id in 0..queries_per_user {
                let current_active = active_queries_clone.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_clone.fetch_max(current_active, Ordering::SeqCst);
                
                let query = format!("stress_test_user_{}_query_{}", user_id, query_id);
                
                match stress_tester_clone.search(&query).await {
                    Ok(_) => success_counter_clone.fetch_add(1, Ordering::SeqCst),
                    Err(_) => error_counter_clone.fetch_add(1, Ordering::SeqCst),
                };
                
                active_queries_clone.fetch_sub(1, Ordering::SeqCst);
                
                // Small delay to prevent overwhelming the system
                sleep(Duration::from_millis(1)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await?;
    }
    
    let total_time = start_time.elapsed();
    let total_queries = concurrent_users * queries_per_user;
    let successful_queries = success_counter.load(Ordering::SeqCst);
    let failed_queries = error_counter.load(Ordering::SeqCst);
    let max_concurrent = max_active.load(Ordering::SeqCst);
    
    // Assert stress test results
    let success_rate = successful_queries as f64 / total_queries as f64;
    assert!(success_rate >= 0.95, "Success rate too low under stress: {:.2}%", success_rate * 100.0);
    
    let qps = successful_queries as f64 / total_time.as_secs_f64();
    assert!(qps >= 500.0, "Throughput too low under extreme stress: {:.2} QPS", qps);
    
    println!("Extreme concurrency stress test: {}/{} queries successful, {:.2} QPS, max concurrent: {}", 
             successful_queries, total_queries, qps, max_concurrent);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_pressure_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let stress_tester = StressTester::new(temp_dir.path()).await?;
    
    // Generate large amounts of data to stress memory
    let large_query_data = "large_test_data ".repeat(10000); // ~150KB query
    let baseline_memory = get_memory_usage()?;
    
    let mut handles = Vec::new();
    let should_continue = Arc::new(AtomicBool::new(true));
    
    // Spawn multiple memory-intensive tasks
    for i in 0..50 {
        let stress_tester_clone = stress_tester.clone();
        let large_data = large_query_data.clone();
        let continue_flag = should_continue.clone();
        
        let handle = tokio::spawn(async move {
            let mut iteration = 0;
            while continue_flag.load(Ordering::SeqCst) {
                let query = format!("{}_iteration_{}", large_data, iteration);
                let _ = stress_tester_clone.search(&query).await;
                iteration += 1;
                
                // Check memory periodically
                if iteration % 10 == 0 {
                    if let Ok(current_memory) = get_memory_usage() {
                        if current_memory > baseline_memory + 2_000_000_000 { // 2GB limit
                            println!("Memory limit reached in task {}, stopping", i);
                            break;
                        }
                    }
                }
                
                sleep(Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }
    
    // Run for a limited time to prevent infinite memory growth
    sleep(Duration::from_secs(30)).await;
    should_continue.store(false, Ordering::SeqCst);
    
    // Wait for all tasks to complete
    for handle in handles {
        let _ = handle.await;
    }
    
    // Check final memory usage
    let final_memory = get_memory_usage()?;
    let memory_increase = final_memory.saturating_sub(baseline_memory);
    
    println!("Memory pressure test: baseline {}MB, final {}MB, increase {}MB", 
             baseline_memory / 1_000_000, final_memory / 1_000_000, memory_increase / 1_000_000);
    
    // System should handle memory pressure gracefully
    assert!(memory_increase < 1_000_000_000, // 1GB increase limit
           "Excessive memory usage under pressure: {}MB increase", memory_increase / 1_000_000);
    
    Ok(())
}

#[tokio::test]
async fn test_sustained_load_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let stress_tester = StressTester::new(temp_dir.path()).await?;
    
    // Run sustained load for extended period
    let duration = Duration::from_secs(300); // 5 minutes
    let target_qps = 100.0;
    let query_interval = Duration::from_millis((1000.0 / target_qps) as u64);
    
    let start_time = Instant::now();
    let end_time = start_time + duration;
    
    let success_counter = Arc::new(AtomicU64::new(0));
    let error_counter = Arc::new(AtomicU64::new(0));
    let latency_sum = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    let query_counter = Arc::new(AtomicU64::new(0));
    
    // Spawn sustained load generator
    for worker_id in 0..10 {
        let stress_tester_clone = stress_tester.clone();
        let success_counter_clone = success_counter.clone();
        let error_counter_clone = error_counter.clone();
        let latency_sum_clone = latency_sum.clone();
        let query_counter_clone = query_counter.clone();
        
        let handle = tokio::spawn(async move {
            while Instant::now() < end_time {
                let query_id = query_counter_clone.fetch_add(1, Ordering::SeqCst);
                let query = format!("sustained_load_worker_{}_query_{}", worker_id, query_id);
                
                let query_start = Instant::now();
                match stress_tester_clone.search(&query).await {
                    Ok(_) => {
                        success_counter_clone.fetch_add(1, Ordering::SeqCst);
                        let latency = query_start.elapsed().as_millis() as u64;
                        latency_sum_clone.fetch_add(latency, Ordering::SeqCst);
                    },
                    Err(_) => error_counter_clone.fetch_add(1, Ordering::SeqCst),
                }
                
                sleep(query_interval).await;
            }
        });
        handles.push(handle);
    }
    
    // Monitor performance during sustained load
    let monitor_handle = tokio::spawn({
        let success_counter = success_counter.clone();
        let error_counter = error_counter.clone();
        
        async move {
            let mut last_success = 0;
            let mut last_check = Instant::now();
            
            while Instant::now() < end_time {
                sleep(Duration::from_secs(10)).await;
                
                let current_success = success_counter.load(Ordering::SeqCst);
                let current_errors = error_counter.load(Ordering::SeqCst);
                let elapsed = last_check.elapsed();
                
                let recent_qps = (current_success - last_success) as f64 / elapsed.as_secs_f64();
                let error_rate = current_errors as f64 / (current_success + current_errors) as f64;
                
                println!("Sustained load check: {:.1} QPS, {:.2}% error rate", recent_qps, error_rate * 100.0);
                
                last_success = current_success;
                last_check = Instant::now();
            }
        }
    });
    
    // Wait for all workers to complete
    for handle in handles {
        handle.await?;
    }
    monitor_handle.abort();
    
    let total_time = start_time.elapsed();
    let successful_queries = success_counter.load(Ordering::SeqCst);
    let failed_queries = error_counter.load(Ordering::SeqCst);
    let total_latency = latency_sum.load(Ordering::SeqCst);
    
    let actual_qps = successful_queries as f64 / total_time.as_secs_f64();
    let error_rate = failed_queries as f64 / (successful_queries + failed_queries) as f64;
    let avg_latency = if successful_queries > 0 { total_latency / successful_queries } else { 0 };
    
    // Assert sustained load requirements
    assert!(actual_qps >= target_qps * 0.8, "Sustained QPS too low: {:.1}", actual_qps);
    assert!(error_rate <= 0.05, "Error rate too high during sustained load: {:.2}%", error_rate * 100.0);
    assert!(avg_latency <= 500, "Average latency too high during sustained load: {}ms", avg_latency);
    
    println!("Sustained load results: {:.1} QPS, {:.2}% errors, {}ms avg latency", 
             actual_qps, error_rate * 100.0, avg_latency);
    
    Ok(())
}

#[tokio::test]
async fn test_resource_exhaustion_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let stress_tester = StressTester::new(temp_dir.path()).await?;
    
    // Test system behavior when resources are exhausted
    let concurrent_limit = 2000; // Intentionally high to stress system
    let semaphore = Arc::new(Semaphore::new(concurrent_limit));
    
    let success_counter = Arc::new(AtomicU64::new(0));
    let timeout_counter = Arc::new(AtomicU64::new(0));
    let error_counter = Arc::new(AtomicU64::new(0));
    
    let mut handles = Vec::new();
    
    // Create resource exhaustion scenario
    for i in 0..concurrent_limit {
        let permit = semaphore.clone().acquire_owned().await?;
        let stress_tester_clone = stress_tester.clone();
        let success_counter_clone = success_counter.clone();
        let timeout_counter_clone = timeout_counter.clone();
        let error_counter_clone = error_counter.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = permit;
            let query = format!("resource_exhaustion_test_{}", i);
            
            // Use timeout to detect resource exhaustion
            match tokio::time::timeout(Duration::from_secs(30), stress_tester_clone.search(&query)).await {
                Ok(Ok(_)) => success_counter_clone.fetch_add(1, Ordering::SeqCst),
                Ok(Err(_)) => error_counter_clone.fetch_add(1, Ordering::SeqCst),
                Err(_) => timeout_counter_clone.fetch_add(1, Ordering::SeqCst), // Timeout
            };
        });
        handles.push(handle);
        
        // Add small delay to prevent immediate resource exhaustion
        if i % 100 == 0 {
            sleep(Duration::from_millis(10)).await;
        }
    }
    
    // Wait for all tasks with a global timeout
    let start_time = Instant::now();
    for handle in handles {
        if start_time.elapsed() > Duration::from_secs(120) {
            handle.abort();
        } else {
            let _ = handle.await;
        }
    }
    
    let successful_queries = success_counter.load(Ordering::SeqCst);
    let timeout_queries = timeout_counter.load(Ordering::SeqCst);
    let failed_queries = error_counter.load(Ordering::SeqCst);
    let total_queries = successful_queries + timeout_queries + failed_queries;
    
    // System should handle resource exhaustion gracefully
    let completion_rate = (successful_queries + failed_queries) as f64 / total_queries as f64;
    assert!(completion_rate >= 0.7, "Too many timeouts under resource exhaustion: {:.1}%", 
           (1.0 - completion_rate) * 100.0);
    
    println!("Resource exhaustion test: {}/{} completed, {} timeouts, {} errors", 
             successful_queries + failed_queries, total_queries, timeout_queries, failed_queries);
    
    Ok(())
}

#[tokio::test]
async fn test_rapid_scale_up_down() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let stress_tester = StressTester::new(temp_dir.path()).await?;
    
    // Test rapid scaling from low to high concurrency and back
    let scale_phases = vec![
        (10, Duration::from_secs(30)),   // Low load
        (500, Duration::from_secs(60)),  // Scale up rapidly
        (1000, Duration::from_secs(30)), // Peak load
        (100, Duration::from_secs(30)),  // Scale down
        (10, Duration::from_secs(30)),   // Return to low load
    ];
    
    for (concurrency, duration) in scale_phases {
        let phase_start = Instant::now();
        let phase_end = phase_start + duration;
        
        let success_counter = Arc::new(AtomicU64::new(0));
        let error_counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        
        // Spawn workers for this phase
        for worker_id in 0..concurrency {
            let stress_tester_clone = stress_tester.clone();
            let success_counter_clone = success_counter.clone();
            let error_counter_clone = error_counter.clone();
            
            let handle = tokio::spawn(async move {
                let mut query_id = 0;
                while Instant::now() < phase_end {
                    let query = format!("scale_test_c{}_w{}_q{}", concurrency, worker_id, query_id);
                    
                    match stress_tester_clone.search(&query).await {
                        Ok(_) => success_counter_clone.fetch_add(1, Ordering::SeqCst),
                        Err(_) => error_counter_clone.fetch_add(1, Ordering::SeqCst),
                    };
                    
                    query_id += 1;
                    sleep(Duration::from_millis(100)).await;
                }
            });
            handles.push(handle);
        }
        
        // Wait for phase to complete
        for handle in handles {
            let _ = handle.await;
        }
        
        let phase_duration = phase_start.elapsed();
        let phase_success = success_counter.load(Ordering::SeqCst);
        let phase_errors = error_counter.load(Ordering::SeqCst);
        let phase_qps = phase_success as f64 / phase_duration.as_secs_f64();
        let error_rate = phase_errors as f64 / (phase_success + phase_errors) as f64;
        
        println!("Scale phase ({}): {:.1} QPS, {:.1}% errors", concurrency, phase_qps, error_rate * 100.0);
        
        // Assert each phase performs adequately
        assert!(error_rate <= 0.1, "High error rate during scale phase {}: {:.1}%", 
               concurrency, error_rate * 100.0);
    }
    
    Ok(())
}

fn get_memory_usage() -> Result<u64> {
    // Platform-specific memory usage measurement
    #[cfg(unix)]
    {
        use std::fs;
        let contents = fs::read_to_string("/proc/self/status")?;
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Ok(parts[1].parse::<u64>()? * 1024);
                }
            }
        }
        Ok(0)
    }
    
    #[cfg(windows)]
    {
        // Simplified for Windows - would use actual Windows API
        Ok(0)
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        Ok(0)
    }
}
```

## Success Criteria
- System handles 1000+ concurrent users with 95%+ success rate
- Memory usage stays within reasonable bounds under pressure
- Sustained load of 100+ QPS maintained for 5 minutes
- Graceful degradation under resource exhaustion (70%+ completion rate)
- System recovers properly after scaling events
- Error rates stay below 10% during stress conditions
- No system crashes or panics under extreme load

## Time Limit
10 minutes maximum