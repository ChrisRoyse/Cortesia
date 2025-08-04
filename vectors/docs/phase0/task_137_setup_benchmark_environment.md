# Micro-Task 137: Setup Benchmark Environment

## Objective
Create the basic environment and configuration for performance benchmarking of the vector search system.

## Context
Benchmarking requires controlled environment and consistent measurement tools. This task establishes the foundation for all performance measurements and validates the <5ms allocation target framework.

## Prerequisites
- Task 136 completed (Final test data generation complete)
- Criterion benchmarking framework configured
- Test data generated and validated

## Time Estimate
8 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Create benchmark configuration `benchmark_config.toml`:
   ```toml
   [environment]
   cpu_frequency_fixed = true
   power_management = "high_performance"
   background_processes_minimal = true
   
   [targets]
   allocation_time_ms = 5.0
   memory_limit_mb = 1024
   cpu_usage_percent = 80
   concurrent_threads = 8
   
   [measurement]
   warmup_iterations = 100
   benchmark_iterations = 1000
   statistical_confidence = 0.95
   measurement_time_seconds = 30
   
   [windows_specific]
   process_priority = "high"
   timer_resolution = "high"
   memory_working_set = "large"
   ```
3. Create benchmark runner script `run_benchmarks.bat`:
   ```batch
   @echo off
   echo Setting high performance mode...
   powercfg /s 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
   echo Increasing process priority...
   wmic process where name="cmd.exe" CALL setpriority "high priority"
   echo Running benchmarks...
   cargo bench --workspace --bench "*" > benchmark_results.txt 2>&1
   echo Benchmarks complete. Results in benchmark_results.txt
   ```
4. Test configuration: `cargo bench --bench allocation_benchmark --dry-run`
5. Commit configuration: `git add benchmark_config.toml run_benchmarks.bat && git commit -m "Setup benchmark environment with 5ms allocation target"`

## Expected Output
- Benchmark configuration file with performance targets
- Windows-optimized benchmark runner script  
- Validated environment setup
- Performance target framework established

## Success Criteria
- [ ] Configuration file created with <5ms allocation target
- [ ] Benchmark runner script working on Windows
- [ ] High performance mode configuration validated
- [ ] Environment setup committed to git
- [ ] Dry run benchmark executes successfully

## Validation Commands
```batch
# Test configuration loading
type benchmark_config.toml

# Test benchmark runner
run_benchmarks.bat --help

# Verify performance mode
powercfg /getactivescheme
```

## Next Task
task_138_create_baseline_measurement_framework.md

## Notes
- Windows performance mode is critical for consistent measurements
- High process priority reduces measurement variance
- Timer resolution affects measurement precision
- Working set configuration prevents memory paging during benchmarks