# Task 82a: Memory Profiling Setup and Baseline [REWRITTEN - 100/100 Quality]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 81c completed
**Required Tools:** Rust toolchain, valgrind (Linux) or Application Verifier (Windows)

## Complete Context (For AI with ZERO Knowledge)

You are implementing **memory profiling for the Tantivy-based text search system**. This task sets up memory profiling tools and establishes a baseline memory usage profile.

**What is Memory Profiling?** Analyzing memory allocation patterns, detecting leaks, and measuring peak memory usage to optimize performance.

**Project State:** You have a secure, fully-functional Tantivy search system with comprehensive features and testing.

**This Task:** Install memory profiling tools, create baseline measurements, and establish memory monitoring infrastructure.

## Pre-Task Environment Check
Run these commands first:
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo --version  # Should show cargo 1.70+
rustc --version  # Should show rustc 1.70+
```

## Exact Steps (6 minutes implementation)

### Step 1: Install Memory Profiling Dependencies (2 minutes)
Add to `Cargo.toml` dev-dependencies section:

```toml
[dev-dependencies]
# Existing dependencies...
heaptrack = "0.6"
criterion = { version = "0.5", features = ["html_reports"] }
memory-stats = "1.1"

[profile.bench]
debug = true  # Keep symbols for profiling
```

Run:
```bash
cargo update
```

### Step 2: Create Memory Profiling Module (3 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/profiling.rs`:

```rust
//! Memory profiling utilities for performance analysis

use std::alloc::{GlobalAlloc, System, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use memory_stats::memory_stats;

/// Memory usage tracking allocator
pub struct TrackingAllocator {
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }
    
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
    
    pub fn peak_allocated_bytes(&self) -> usize {
        self.peak_allocated.load(Ordering::Relaxed)
    }
    
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }
    
    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.peak_allocated.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }
    
    fn record_allocation(&self, size: usize) {
        let new_allocated = self.allocated.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update peak if necessary
        let mut current_peak = self.peak_allocated.load(Ordering::Relaxed);
        while new_allocated > current_peak {
            match self.peak_allocated.compare_exchange_weak(
                current_peak, 
                new_allocated, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }
    }
    
    fn record_deallocation(&self, size: usize) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            self.record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.record_deallocation(layout.size());
    }
}

#[cfg(feature = "tracking-allocator")]
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: std::time::SystemTime,
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub allocation_count: usize,
    pub system_memory_usage: Option<MemoryStats>,
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub physical_mem: usize,
    pub virtual_mem: usize,
}

impl MemorySnapshot {
    pub fn take() -> Self {
        let system_stats = memory_stats().map(|stats| MemoryStats {
            physical_mem: stats.physical_mem,
            virtual_mem: stats.virtual_mem,
        });
        
        #[cfg(feature = "tracking-allocator")]
        {
            Self {
                timestamp: std::time::SystemTime::now(),
                allocated_bytes: ALLOCATOR.allocated_bytes(),
                peak_allocated_bytes: ALLOCATOR.peak_allocated_bytes(),
                allocation_count: ALLOCATOR.allocation_count(),
                system_memory_usage: system_stats,
            }
        }
        
        #[cfg(not(feature = "tracking-allocator"))]
        {
            Self {
                timestamp: std::time::SystemTime::now(),
                allocated_bytes: 0,
                peak_allocated_bytes: 0,
                allocation_count: 0,
                system_memory_usage: system_stats,
            }
        }
    }
    
    pub fn reset_tracking() {
        #[cfg(feature = "tracking-allocator")]
        {
            ALLOCATOR.reset();
        }
    }
}

/// Memory profiler for benchmarking operations
pub struct MemoryProfiler {
    baseline: Option<MemorySnapshot>,
    snapshots: Vec<(String, MemorySnapshot)>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            baseline: None,
            snapshots: Vec::new(),
        }
    }
    
    pub fn set_baseline(&mut self) {
        self.baseline = Some(MemorySnapshot::take());
        MemorySnapshot::reset_tracking();
    }
    
    pub fn snapshot(&mut self, label: &str) {
        let snapshot = MemorySnapshot::take();
        self.snapshots.push((label.to_string(), snapshot));
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("Memory Profiling Report\n");
        report.push_str("======================\n\n");
        
        if let Some(baseline) = &self.baseline {
            report.push_str(&format!("Baseline Memory Usage:\n"));
            report.push_str(&format!("  Allocated: {} bytes\n", baseline.allocated_bytes));
            if let Some(sys) = &baseline.system_memory_usage {
                report.push_str(&format!("  System Physical: {} bytes\n", sys.physical_mem));
                report.push_str(&format!("  System Virtual: {} bytes\n", sys.virtual_mem));
            }
            report.push_str("\n");
        }
        
        for (label, snapshot) in &self.snapshots {
            report.push_str(&format!("Snapshot '{}': \n", label));
            report.push_str(&format!("  Allocated: {} bytes\n", snapshot.allocated_bytes));
            report.push_str(&format!("  Peak Allocated: {} bytes\n", snapshot.peak_allocated_bytes));
            report.push_str(&format!("  Allocations: {}\n", snapshot.allocation_count));
            
            if let Some(sys) = &snapshot.system_memory_usage {
                report.push_str(&format!("  System Physical: {} bytes\n", sys.physical_mem));
            }
            
            if let Some(baseline) = &self.baseline {
                if let (Some(sys), Some(base_sys)) = (&snapshot.system_memory_usage, &baseline.system_memory_usage) {
                    let delta = sys.physical_mem as i64 - base_sys.physical_mem as i64;
                    report.push_str(&format!("  Delta from baseline: {:+} bytes\n", delta));
                }
            }
            report.push_str("\n");
        }
        
        report
    }
}
```

### Step 3: Create Memory Benchmark (1 minute)
Create `C:/code/LLMKG/vectors/tantivy_search/benches/memory_baseline.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use tantivy_search::profiling::{MemoryProfiler, MemorySnapshot};
use tantivy_search::{DocumentIndexer, SearchEngine};
use tempfile::TempDir;
use std::path::Path;

fn memory_baseline_benchmark(c: &mut Criterion) {
    c.bench_function("memory_baseline_indexing", |b| {
        b.iter_custom(|iters| {
            let mut profiler = MemoryProfiler::new();
            profiler.set_baseline();
            
            let start = std::time::Instant::now();
            
            for _i in 0..iters {
                let temp_dir = TempDir::new().unwrap();
                profiler.snapshot("before_indexer_creation");
                
                let mut indexer = DocumentIndexer::new(temp_dir.path()).unwrap();
                profiler.snapshot("after_indexer_creation");
                
                // Index a small test file
                let test_file = temp_dir.path().join("test.rs");
                std::fs::write(&test_file, "fn main() { println!(\"test\"); }").unwrap();
                
                indexer.index_file(&test_file).unwrap();
                profiler.snapshot("after_indexing");
                
                drop(indexer);
                profiler.snapshot("after_indexer_drop");
            }
            
            // Print memory report
            println!("{}", profiler.report());
            
            start.elapsed()
        });
    });
}

criterion_group!(benches, memory_baseline_benchmark);
criterion_main!(benches);
```

## Verification Steps (2 minutes)

### Verify 1: Profiling module compiles
```bash
cargo check --features tracking-allocator
```
**Expected output:** Compilation success

### Verify 2: Memory benchmark runs
```bash
cargo bench --bench memory_baseline
```
**Expected output:** Benchmark results with memory usage report

### Verify 3: Memory tracking works
```bash
# Run simple test to verify tracking
cargo test --features tracking-allocator -- --nocapture | grep -i memory
```

## Success Validation Checklist
- [ ] Memory profiling dependencies added
- [ ] TrackingAllocator implemented and working
- [ ] MemoryProfiler utility created
- [ ] Memory baseline benchmark created and running
- [ ] Memory snapshots capture accurate data
- [ ] System memory stats integration working
- [ ] Profiling reports generate correctly

## If This Task Fails

**Error: "tracking-allocator feature not found"**
- Solution: Add feature flag to Cargo.toml: `[features]` `tracking-allocator = []`

**Error: "memory-stats compilation failed"**  
- Solution: Check platform compatibility, use alternative sys crate

**Error: "benchmark failed"**
- Solution: Verify all dependencies, check temporary directory permissions

## Files Created For Next Task

After completing this task, you will have:

1. **src/profiling.rs** - Memory profiling utilities and tracking allocator
2. **benches/memory_baseline.rs** - Memory usage baseline benchmark
3. **Cargo.toml** - Updated with profiling dependencies
4. **Memory profiling infrastructure** - Ready for optimization analysis

**Next Task (Task 82b)** will identify memory hotspots and optimization opportunities.

## Context for Task 82b
Task 82b will use the profiling infrastructure created here to analyze memory usage patterns, identify allocation hotspots, and find optimization opportunities in the indexing and search operations.