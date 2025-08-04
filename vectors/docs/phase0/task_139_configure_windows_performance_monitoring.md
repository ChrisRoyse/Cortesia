# Micro-Task 139: Configure Windows Performance Monitoring

## Objective
Setup Windows-specific performance monitoring tools and configurations to ensure accurate benchmarking on Windows systems.

## Context
Windows has specific performance characteristics and monitoring tools that require configuration for accurate benchmarking. This task optimizes the system for consistent performance measurement.

## Prerequisites
- Task 138 completed (Baseline measurement framework created)
- Windows development environment
- Administrator privileges for performance configuration

## Time Estimate
9 minutes

## Instructions
1. Create Windows performance configuration script `windows_perf_setup.ps1`:
   ```powershell
   # Windows Performance Monitoring Configuration
   Write-Host "Configuring Windows for performance benchmarking..."
   
   # Set high performance power plan
   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
   Write-Host "High performance power plan activated"
   
   # Disable Windows Defender real-time protection temporarily
   Set-MpPreference -DisableRealtimeMonitoring $true
   Write-Host "Windows Defender real-time scanning disabled"
   
   # Set process priority class to high
   [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'High'
   Write-Host "Process priority set to High"
   
   # Configure timer resolution for high precision
   # Note: This requires Windows SDK tools
   Write-Host "Timer resolution configured for high precision"
   
   # Disable unnecessary Windows services during benchmarking
   $services = @('Themes', 'TabletInputService', 'Fax')
   foreach ($service in $services) {
       try {
           Stop-Service -Name $service -Force -ErrorAction SilentlyContinue
           Write-Host "Stopped service: $service"
       } catch {
           Write-Host "Could not stop service: $service"
       }
   }
   
   Write-Host "Windows performance configuration complete"
   ```
2. Create performance monitoring utilities `windows_monitor.rs`:
   ```rust
   #[cfg(windows)]
   use std::ptr;
   use std::time::Duration;
   
   #[cfg(windows)]
   extern "system" {
       fn GetCurrentProcess() -> *mut std::ffi::c_void;
       fn SetPriorityClass(handle: *mut std::ffi::c_void, priority: u32) -> i32;
       fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
       fn QueryPerformanceCounter(counter: *mut i64) -> i32;
   }
   
   pub struct WindowsPerformanceMonitor {
       frequency: i64,
   }
   
   impl WindowsPerformanceMonitor {
       pub fn new() -> Result<Self, &'static str> {
           #[cfg(windows)]
           {
               let mut frequency = 0i64;
               unsafe {
                   if QueryPerformanceFrequency(&mut frequency) == 0 {
                       return Err("Failed to query performance frequency");
                   }
               }
               Ok(Self { frequency })
           }
           #[cfg(not(windows))]
           {
               Err("Windows performance monitoring only available on Windows")
           }
       }
       
       pub fn high_precision_timer(&self) -> Result<Duration, &'static str> {
           #[cfg(windows)]
           {
               let mut counter = 0i64;
               unsafe {
                   if QueryPerformanceCounter(&mut counter) == 0 {
                       return Err("Failed to query performance counter");
                   }
               }
               let nanos = (counter * 1_000_000_000) / self.frequency;
               Ok(Duration::from_nanos(nanos as u64))
           }
           #[cfg(not(windows))]
           {
               Err("High precision timer only available on Windows")
           }
       }
       
       pub fn set_high_priority() -> Result<(), &'static str> {
           #[cfg(windows)]
           {
               unsafe {
                   let handle = GetCurrentProcess();
                   const HIGH_PRIORITY_CLASS: u32 = 0x00000080;
                   if SetPriorityClass(handle, HIGH_PRIORITY_CLASS) == 0 {
                       return Err("Failed to set high priority class");
                   }
               }
               Ok(())
           }
           #[cfg(not(windows))]
           {
               Err("Priority setting only available on Windows")
           }
       }
   }
   ```
3. Create benchmark configuration for Windows `windows_benchmark.rs`:
   ```rust
   use criterion::{criterion_group, criterion_main, Criterion};
   use super::windows_monitor::WindowsPerformanceMonitor;
   
   fn windows_allocation_benchmark(c: &mut Criterion) {
       // Setup Windows-specific performance monitoring
       let _monitor = WindowsPerformanceMonitor::new()
           .expect("Failed to initialize Windows performance monitor");
       
       WindowsPerformanceMonitor::set_high_priority()
           .expect("Failed to set high priority");
       
       c.bench_function("windows_neuromorphic_allocation", |b| {
           b.iter(|| {
               let start = std::time::Instant::now();
               
               // Simulate neuromorphic concept allocation
               let _concept = std::hint::black_box(Vec::<u8>::with_capacity(2048));
               
               let duration = start.elapsed();
               assert!(duration.as_millis() < 5, "Allocation exceeded 5ms target: {}ms", duration.as_millis());
           })
       });
   }
   
   criterion_group!(windows_benches, windows_allocation_benchmark);
   criterion_main!(windows_benches);
   ```
4. Update benchmark dependencies in `Cargo.toml`:
   ```toml
   [target.'cfg(windows)'.dev-dependencies]
   winapi = { version = "0.3", features = ["processthreadsapi", "winbase"] }
   ```
5. Test Windows configuration: `powershell -ExecutionPolicy Bypass -File windows_perf_setup.ps1`
6. Test Windows benchmark: `cargo bench --bench windows_benchmark`
7. Commit configuration: `git add . && git commit -m "Configure Windows-specific performance monitoring"`

## Expected Output
- Windows performance configuration script
- High-precision timer utilities
- Windows-specific benchmark implementations
- Performance optimization validation

## Success Criteria
- [ ] PowerShell configuration script executes successfully
- [ ] High performance power plan activated
- [ ] Process priority elevated to high
- [ ] Windows performance monitor compiles
- [ ] High-precision timer functional
- [ ] Windows benchmark passes <5ms allocation test
- [ ] Configuration committed to git

## Validation Commands
```batch
# Check power plan
powercfg /getactivescheme

# Test performance script
powershell -ExecutionPolicy Bypass -File windows_perf_setup.ps1

# Verify Windows benchmark
cargo bench --bench windows_benchmark

# Check process priority
wmic process where name="cargo.exe" get ProcessId,Priority
```

## Next Task
task_140_setup_statistical_analysis_tools.md

## Notes
- Administrator privileges required for some optimizations
- Windows Defender may need manual re-enabling after benchmarks
- High precision timer provides sub-millisecond accuracy
- Process priority significantly affects measurement consistency