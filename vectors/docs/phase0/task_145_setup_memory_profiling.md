# Micro-Task 145: Setup Memory Profiling

## Objective
Configure memory profiling tools for tracking allocation patterns and memory usage during vector operations.

## Context
Memory profiling is essential for validating allocation efficiency and ensuring the system stays within memory limits while meeting the <5ms allocation target.

## Prerequisites
- Task 144 completed (CPU profiling configured)
- Profiling environment established
- Benchmark tools installed

## Time Estimate
9 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Add memory profiling dependency to `Cargo.toml`:
   ```toml
   [dev-dependencies]
   dhat = "0.3"
   jemallocator = "0.5"
   ```
3. Create memory profiling configuration `memory_profiling.toml`:
   ```toml
   [memory]
   track_allocations = true
   track_deallocations = true
   stack_trace_depth = 16
   sample_rate = 1
   
   [heap_analysis]
   enabled = true
   detect_leaks = true
   track_peak_usage = true
   fragmentation_analysis = true
   
   [reporting]
   format = "json"
   output_directory = "memory_reports"
   include_call_stacks = true
   
   [thresholds]
   max_allocation_size_mb = 100
   max_peak_usage_mb = 1024
   leak_detection_threshold_bytes = 1024
   ```
4. Create memory profiling script `profile_memory.bat`:
   ```batch
   @echo off
   echo Starting memory profiling...
   if not exist memory_reports mkdir memory_reports
   set RUST_LOG=debug
   cargo bench --bench allocation_benchmark --features "dhat-heap"
   cargo bench --bench memory_benchmark --features "dhat-heap"
   echo Memory profiling complete. Reports in memory_reports/
   ```
5. Test memory profiling: `cargo check --features "dhat-heap"`
6. Commit setup: `git add Cargo.toml memory_profiling.toml profile_memory.bat && git commit -m "Setup memory profiling for allocation analysis"`

## Expected Output
- Memory profiling configuration
- DHAT heap profiling integration
- Memory profiling runner script
- Leak detection capability

## Success Criteria
- [ ] Memory profiling dependencies added
- [ ] Memory profiling configuration created
- [ ] Profiling script working
- [ ] DHAT integration validated
- [ ] Memory profiling setup committed

## Validation Commands
```batch
# Test memory profiling features
cargo check --features "dhat-heap"

# Verify dependencies
cargo tree | findstr dhat

# Test configuration
type memory_profiling.toml
```

## Next Task
task_146_create_benchmark_data_sets.md

## Notes
- DHAT provides detailed heap profiling
- jemalloc offers better allocation tracking
- Memory reports help identify allocation patterns
- Leak detection prevents memory growth issues