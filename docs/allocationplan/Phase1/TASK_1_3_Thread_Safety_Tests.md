# Task 1.3: Thread Safety Tests

**Duration**: 2 hours  
**Complexity**: Medium  
**Dependencies**: Task 1.2 (Atomic State Transitions)  
**AI Assistant Suitability**: High - Systematic concurrency testing  

## Objective

Create comprehensive thread safety tests to verify that the cortical column implementation is production-ready for high-concurrency neuromorphic processing. This task focuses on testing rather than implementing new functionality.

## Specification

Design and implement stress tests that verify:

**Concurrency Guarantees**:
- No data races under extreme load
- No deadlocks or livelocks
- Consistent state transitions under contention
- Memory safety with thousands of threads

**Performance Under Load**:
- Throughput scaling with thread count
- Latency distribution under contention
- Memory usage stability during stress
- No performance degradation after extended runs

**Edge Case Coverage**:
- Rapid state transitions
- Simultaneous exclusive access attempts
- Memory pressure scenarios
- Thread spawning/termination patterns

## Implementation Guide

### Step 1: Concurrency Test Framework

```rust
// tests/thread_safety_tests.rs
use llmkg::{EnhancedCorticalColumn, ColumnState};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use crossbeam::utils::Backoff;
use rand::Rng;

/// Test harness for concurrency experiments
struct ConcurrencyTestHarness {
    column: Arc<EnhancedCorticalColumn>,
    thread_count: usize,
    operations_per_thread: usize,
    test_duration: Duration,
}

impl ConcurrencyTestHarness {
    fn new(thread_count: usize, operations_per_thread: usize) -> Self {
        Self {
            column: Arc::new(EnhancedCorticalColumn::new(42)),
            thread_count,
            operations_per_thread,
            test_duration: Duration::from_secs(5),
        }
    }
    
    /// Run a test with specified operation distribution
    fn run_operation_test<F>(&self, operation_fn: F) -> ConcurrencyTestResult
    where
        F: Fn(Arc<EnhancedCorticalColumn>, usize) -> OperationResult + Send + Sync + 'static,
    {
        let barrier = Arc::new(Barrier::new(self.thread_count));
        let results = Arc::new(Mutex::new(Vec::new()));
        let operation_fn = Arc::new(operation_fn);
        
        let mut handles = vec![];
        
        for thread_id in 0..self.thread_count {
            let column = self.column.clone();
            let barrier = barrier.clone();
            let results = results.clone();
            let operation_fn = operation_fn.clone();
            let ops_count = self.operations_per_thread;
            
            handles.push(thread::spawn(move || {
                barrier.wait(); // Synchronize start
                
                let thread_result = operation_fn(column, ops_count);
                
                results.lock().unwrap().push(ThreadResult {
                    thread_id,
                    operation_result: thread_result,
                });
            }));
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let thread_results = results.lock().unwrap().clone();
        ConcurrencyTestResult::analyze(thread_results, &self.column)
    }
}

#[derive(Debug, Clone)]
struct ThreadResult {
    thread_id: usize,
    operation_result: OperationResult,
}

#[derive(Debug, Clone)]
struct OperationResult {
    successful_operations: u64,
    failed_operations: u64,
    total_time: Duration,
    operation_times: Vec<Duration>,
}

#[derive(Debug)]
struct ConcurrencyTestResult {
    total_operations: u64,
    successful_operations: u64,
    success_rate: f64,
    total_duration: Duration,
    throughput_ops_per_sec: f64,
    latency_percentiles: LatencyPercentiles,
    final_column_state: ColumnState,
    consistency_verified: bool,
}

#[derive(Debug)]
struct LatencyPercentiles {
    p50: Duration,
    p95: Duration,
    p99: Duration,
    p999: Duration,
    max: Duration,
}

impl ConcurrencyTestResult {
    fn analyze(thread_results: Vec<ThreadResult>, column: &EnhancedCorticalColumn) -> Self {
        let mut all_times = Vec::new();
        let mut total_successful = 0;
        let mut total_operations = 0;
        let mut total_duration = Duration::ZERO;
        
        for result in &thread_results {
            total_successful += result.operation_result.successful_operations;
            total_operations += result.operation_result.successful_operations + result.operation_result.failed_operations;
            total_duration = total_duration.max(result.operation_result.total_time);
            all_times.extend(result.operation_result.operation_times.iter().cloned());
        }
        
        // Sort times for percentile calculation
        all_times.sort();
        
        let latency_percentiles = if all_times.is_empty() {
            LatencyPercentiles {
                p50: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                p999: Duration::ZERO,
                max: Duration::ZERO,
            }
        } else {
            LatencyPercentiles {
                p50: all_times[all_times.len() * 50 / 100],
                p95: all_times[all_times.len() * 95 / 100],
                p99: all_times[all_times.len() * 99 / 100],
                p999: all_times[all_times.len() * 999 / 1000],
                max: all_times[all_times.len() - 1],
            }
        };
        
        Self {
            total_operations,
            successful_operations: total_successful,
            success_rate: total_successful as f64 / total_operations as f64,
            total_duration,
            throughput_ops_per_sec: total_successful as f64 / total_duration.as_secs_f64(),
            latency_percentiles,
            final_column_state: column.current_state(),
            consistency_verified: Self::verify_consistency(column),
        }
    }
    
    fn verify_consistency(column: &EnhancedCorticalColumn) -> bool {
        let metrics = column.performance_metrics();
        
        // Basic sanity checks
        metrics.total_transitions == metrics.successful_transitions + metrics.failed_transitions
            && metrics.success_rate() >= 0.0 
            && metrics.success_rate() <= 1.0
    }
}
```

### Step 2: Specific Concurrency Tests

```rust
#[test]
fn test_massive_concurrent_activations() {
    let harness = ConcurrencyTestHarness::new(100, 1000);
    
    let result = harness.run_operation_test(|column, ops_count| {
        let mut successful = 0;
        let mut failed = 0;
        let mut times = Vec::new();
        let start_time = Instant::now();
        
        for _ in 0..ops_count {
            let op_start = Instant::now();
            
            // Try to activate with random level
            let activation_level = rand::thread_rng().gen_range(0.0..=1.0);
            match column.try_activate_with_level(activation_level) {
                Ok(_) => {
                    successful += 1;
                    // Reset immediately for next attempt
                    let _ = column.try_reset();
                }
                Err(_) => failed += 1,
            }
            
            times.push(op_start.elapsed());
        }
        
        OperationResult {
            successful_operations: successful,
            failed_operations: failed,
            total_time: start_time.elapsed(),
            operation_times: times,
        }
    });
    
    // Verify results
    assert!(result.success_rate > 0.1); // At least 10% should succeed
    assert!(result.throughput_ops_per_sec > 1000.0); // Minimum throughput
    assert!(result.latency_percentiles.p99 < Duration::from_micros(100)); // p99 < 100μs
    assert!(result.consistency_verified);
    
    println!("Massive activation test: {:#?}", result);
}

#[test]
fn test_state_transition_chains() {
    let harness = ConcurrencyTestHarness::new(50, 500);
    
    let result = harness.run_operation_test(|column, ops_count| {
        let mut successful = 0;
        let mut failed = 0;
        let mut times = Vec::new();
        let start_time = Instant::now();
        
        for _ in 0..ops_count {
            let op_start = Instant::now();
            
            // Attempt full state transition chain
            let chain_success = 
                column.try_activate_with_level(0.8).is_ok() &&
                column.try_compete_with_strength(0.9).is_ok() &&
                column.try_allocate().is_ok() &&
                column.try_enter_refractory().is_ok() &&
                column.try_reset().is_ok();
            
            if chain_success {
                successful += 1;
            } else {
                failed += 1;
                // Reset to known state if chain failed
                let _ = column.try_reset();
            }
            
            times.push(op_start.elapsed());
        }
        
        OperationResult {
            successful_operations: successful,
            failed_operations: failed,
            total_time: start_time.elapsed(),
            operation_times: times,
        }
    });
    
    assert!(result.success_rate > 0.05); // At least 5% full chains should succeed
    assert!(result.consistency_verified);
    
    println!("State transition chains: {:#?}", result);
}

#[test]
fn test_exclusive_access_contention() {
    let harness = ConcurrencyTestHarness::new(200, 100);
    
    let result = harness.run_operation_test(|column, ops_count| {
        let mut successful = 0;
        let mut failed = 0;
        let mut times = Vec::new();
        let start_time = Instant::now();
        
        for _ in 0..ops_count {
            let op_start = Instant::now();
            
            if let Some(exclusive) = column.try_acquire_exclusive() {
                // Hold exclusive access briefly
                let _ = exclusive.set_activation(0.5);
                thread::sleep(Duration::from_nanos(100)); // Tiny hold time
                successful += 1;
                // exclusive is dropped here, releasing access
            } else {
                failed += 1;
            }
            
            times.push(op_start.elapsed());
            
            // Small backoff to reduce contention
            let backoff = Backoff::new();
            backoff.snooze();
        }
        
        OperationResult {
            successful_operations: successful,
            failed_operations: failed,
            total_time: start_time.elapsed(),
            operation_times: times,
        }
    });
    
    // With 200 threads and brief holds, should get reasonable success rate
    assert!(result.success_rate > 0.01); // At least 1% should get exclusive access
    assert_eq!(result.final_column_state, ColumnState::Available); // Should be released
    assert!(result.consistency_verified);
    
    println!("Exclusive access contention: {:#?}", result);
}

#[test]
fn test_memory_pressure_stability() {
    // Test with many columns under memory pressure
    let columns: Vec<Arc<EnhancedCorticalColumn>> = (0..1000)
        .map(|i| Arc::new(EnhancedCorticalColumn::new(i)))
        .collect();
    
    let barrier = Arc::new(Barrier::new(50));
    let mut handles = vec![];
    
    for thread_id in 0..50 {
        let columns = columns.clone();
        let barrier = barrier.clone();
        
        handles.push(thread::spawn(move || {
            barrier.wait();
            
            // Each thread works with subset of columns
            let start_idx = thread_id * 20;
            let end_idx = start_idx + 20;
            
            for _ in 0..100 {
                for col_idx in start_idx..end_idx {
                    let column = &columns[col_idx];
                    
                    // Random operations
                    match rand::thread_rng().gen_range(0..4) {
                        0 => { let _ = column.try_activate_with_level(0.5); }
                        1 => { let _ = column.update_activation(0.7); }
                        2 => { let _ = column.try_reset(); }
                        _ => { let _ = column.try_acquire_exclusive(); }
                    }
                }
            }
            
            // Verify all columns are in valid states
            for col_idx in start_idx..end_idx {
                let column = &columns[col_idx];
                let state = column.current_state();
                let activation = column.activation_level();
                
                assert!(matches!(
                    state,
                    ColumnState::Available | 
                    ColumnState::Activated | 
                    ColumnState::Competing | 
                    ColumnState::Allocated | 
                    ColumnState::Refractory
                ));
                assert!(activation >= 0.0 && activation <= 1.0);
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Final consistency check across all columns
    for column in &columns {
        assert!(column.performance_metrics().success_rate() >= 0.0);
    }
}

#[test]
fn test_long_running_stress() {
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut handles = vec![];
    
    // Start multiple worker threads
    for worker_id in 0..10 {
        let column = column.clone();
        let stop_flag = stop_flag.clone();
        
        handles.push(thread::spawn(move || {
            let mut local_ops = 0u64;
            
            while !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                match worker_id % 3 {
                    0 => {
                        // Activation worker
                        if column.try_activate_with_level(0.5).is_ok() {
                            thread::sleep(Duration::from_micros(10));
                            let _ = column.try_reset();
                        }
                    }
                    1 => {
                        // Update worker
                        let _ = column.update_activation(0.3);
                    }
                    _ => {
                        // Exclusive access worker
                        if let Some(_exclusive) = column.try_acquire_exclusive() {
                            thread::sleep(Duration::from_micros(5));
                        }
                    }
                }
                
                local_ops += 1;
                
                // Brief pause to prevent spinning
                if local_ops % 1000 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            
            local_ops
        }));
    }
    
    // Run for 2 seconds
    thread::sleep(Duration::from_secs(2));
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    
    // Collect results
    let operation_counts: Vec<u64> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    let total_ops: u64 = operation_counts.iter().sum();
    println!("Long-running stress completed {} operations", total_ops);
    
    // Should have performed many operations
    assert!(total_ops > 10_000);
    
    // Column should still be in valid state
    let final_state = column.current_state();
    assert!(matches!(
        final_state,
        ColumnState::Available | 
        ColumnState::Activated | 
        ColumnState::Competing | 
        ColumnState::Allocated | 
        ColumnState::Refractory
    ));
    
    // Metrics should be consistent
    let metrics = column.performance_metrics();
    assert_eq!(
        metrics.total_transitions,
        metrics.successful_transitions + metrics.failed_transitions
    );
}

#[test]
fn test_rapid_thread_creation_destruction() {
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    
    // Rapidly create and destroy threads that interact with the column
    for batch in 0..20 {
        let mut handles = vec![];
        
        // Create 50 threads
        for _ in 0..50 {
            let column = column.clone();
            handles.push(thread::spawn(move || {
                // Each thread does a few operations then exits
                for i in 0..10 {
                    match i % 3 {
                        0 => { let _ = column.try_activate_with_level(0.6); }
                        1 => { let _ = column.update_activation(0.4); }
                        _ => { let _ = column.try_reset(); }
                    }
                }
            }));
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify column is still in valid state after each batch
        let state = column.current_state();
        let activation = column.activation_level();
        
        assert!(matches!(
            state,
            ColumnState::Available | 
            ColumnState::Activated | 
            ColumnState::Competing | 
            ColumnState::Allocated | 
            ColumnState::Refractory
        ));
        assert!(activation >= 0.0 && activation <= 1.0);
        
        if batch % 5 == 0 {
            println!("Batch {} completed, state: {:?}", batch, state);
        }
    }
    
    // Final metrics should be reasonable
    let metrics = column.performance_metrics();
    assert!(metrics.total_transitions > 0);
    assert!(metrics.success_rate() >= 0.0);
}

#[test]
fn test_no_data_races_with_sanitizer() {
    // This test is designed to catch data races when run with ThreadSanitizer
    // Run with: cargo test --target x86_64-unknown-linux-gnu -- --test-threads=1
    
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    let mut handles = vec![];
    
    // Many threads doing overlapping reads and writes
    for _ in 0..100 {
        let column = column.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                // Mix of operations that could race
                let _ = column.activation_level(); // Read
                let _ = column.current_state(); // Read
                let _ = column.update_activation(0.5); // Write
                let _ = column.try_activate_with_level(0.3); // Write
                let _ = column.time_since_transition(); // Read
                let _ = column.performance_metrics(); // Read
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // If we get here without ThreadSanitizer errors, no data races detected
    assert!(true);
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 8/8 tests passing consistently
2. **Performance verified**: 
   - Throughput > 1000 ops/sec under contention
   - P99 latency < 100μs
   - Success rate > 5% in high contention scenarios
3. **Stability confirmed**: Long-running test completes without issues
4. **Memory safety**: No data races detected with ThreadSanitizer
5. **Resource management**: No memory leaks in extended runs

## Verification Commands

```bash
# Run all thread safety tests
cargo test thread_safety_tests --release

# Run with verbose output to see performance data
cargo test thread_safety_tests --release -- --nocapture

# Run with ThreadSanitizer (Linux only)
RUSTFLAGS="-Z sanitizer=thread" cargo test test_no_data_races_with_sanitizer --target x86_64-unknown-linux-gnu

# Stress test (run multiple times)
for i in {1..5}; do
  echo "Stress run $i"
  cargo test test_long_running_stress --release
done

# Memory usage check
valgrind --tool=massif cargo test test_memory_pressure_stability --release
```

## Expected Performance Results

```
Massive concurrent activations:
├── Success rate: 15-30%
├── Throughput: 5,000-15,000 ops/sec
├── P99 latency: 20-80μs
└── Memory consistent: ✓

State transition chains:
├── Success rate: 8-15%
├── Latency: 50-150μs per chain
└── Consistency verified: ✓

Exclusive access contention:
├── Success rate: 2-5%
├── Final state: Available
└── No deadlocks: ✓

Long-running stress:
├── Total operations: 50,000+
├── Runtime: 2 seconds
├── Final state: Valid
└── Metrics consistent: ✓
```

## Files to Create

1. `tests/thread_safety_tests.rs`
2. `benches/concurrency_benchmarks.rs` (optional)

## Dependencies to Add to Cargo.toml

```toml
[dev-dependencies]
crossbeam = "0.8"
rand = "0.8"
```

## Expected Completion Time

2 hours for an AI assistant:
- 30 minutes: Test harness implementation
- 45 minutes: Individual test cases
- 30 minutes: Stress tests and validation
- 15 minutes: Performance verification and documentation

## Next Task

Task 1.4: Biological Activation (neuromorphic activation dynamics)