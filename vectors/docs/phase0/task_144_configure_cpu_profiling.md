# Micro-Task 144: Configure CPU Profiling

## Objective
Setup CPU profiling tools and configuration for detailed performance analysis during benchmarking.

## Context
CPU profiling helps identify bottlenecks and validate the <5ms allocation target by analyzing where time is spent during vector operations.

## Prerequisites
- Task 143 completed (Memory usage baseline measured)
- Windows performance monitoring configured
- Benchmark environment established

## Time Estimate
8 minutes

## Instructions
1. Navigate to project root: `cd C:\code\LLMKG\vectors`
2. Create CPU profiling configuration `cpu_profiling.toml`:
   ```toml
   [profiling]
   enabled = true
   sample_rate_hz = 1000
   stack_depth = 32
   include_system_calls = false
   
   [output]
   format = "flamegraph"
   directory = "profiling_results"
   filename_prefix = "cpu_profile"
   
   [filters]
   include_crates = ["vector-search", "tantivy-core", "lancedb-integration"]
   exclude_system = true
   exclude_std = false
   
   [windows_specific]
   use_etw = true
   high_resolution_timer = true
   kernel_mode_events = false
   ```
3. Create profiling runner script `profile_cpu.bat`:
   ```batch
   @echo off
   echo Starting CPU profiling session...
   if not exist profiling_results mkdir profiling_results
   cargo install flamegraph --locked
   cargo flamegraph --bench allocation_benchmark --output profiling_results/cpu_allocation.svg
   cargo flamegraph --bench search_benchmark --output profiling_results/cpu_search.svg
   echo CPU profiling complete. Results in profiling_results/
   ```
4. Test profiling setup: `cargo check --bench allocation_benchmark`
5. Commit configuration: `git add cpu_profiling.toml profile_cpu.bat && git commit -m "Configure CPU profiling for benchmark analysis"`

## Expected Output
- CPU profiling configuration file
- Profiling runner script for Windows
- Validated profiling setup
- Flamegraph generation capability

## Success Criteria
- [ ] CPU profiling configuration created
- [ ] Profiling script working on Windows
- [ ] Flamegraph tool installed and verified
- [ ] Profiling setup committed to git
- [ ] Test benchmark compilation successful

## Validation Commands
```batch
# Test configuration
type cpu_profiling.toml

# Test flamegraph installation
cargo install flamegraph --dry-run

# Verify benchmark exists
cargo bench --bench allocation_benchmark --list
```

## Next Task
task_145_setup_memory_profiling.md

## Notes
- ETW (Event Tracing for Windows) provides precise timing
- High resolution timers improve accuracy
- Flamegraphs visualize CPU usage patterns
- Profiling results help optimize allocation paths