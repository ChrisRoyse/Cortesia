# Micro-Task 026: Configure Memory Profiling

## Objective
Setup memory profiling tools and configuration for identifying memory leaks and optimization opportunities.

## Context
Vector search systems can be memory-intensive. Proper memory profiling helps identify leaks, excessive allocations, and optimization opportunities. This task configures tools for memory analysis.

## Prerequisites
- Task 025 completed (Performance monitoring setup)
- Windows development environment
- Understanding of memory profiling needs

## Time Estimate
8 minutes

## Instructions
1. Add memory profiling dependencies to workspace `Cargo.toml`:
   ```toml
   [workspace.dependencies]
   # Add to existing dependencies section
   jemallocator = "0.5"
   peak_alloc = "0.2"
   ```
2. Create memory profiling configuration `memory_profile.toml`:
   ```toml
   [memory_profiling]
   enabled = true
   track_allocations = true
   dump_on_exit = true
   max_tracked_allocations = 10000
   
   [jemalloc]
   background_thread = true
   abort_on_oom = false
   stats_print = true
   
   [reporting]
   output_dir = "data/logs"
   allocation_report = "memory_allocations.json"
   leak_report = "memory_leaks.txt"
   ```
3. Create memory utilities `src/memory_utils.rs`:
   ```rust
   //! Memory profiling and monitoring utilities
   
   use std::alloc::{GlobalAlloc, Layout, System};
   use std::sync::atomic::{AtomicUsize, Ordering};
   
   /// Simple allocation tracker
   pub struct AllocationTracker;
   
   static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
   static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
   
   unsafe impl GlobalAlloc for AllocationTracker {
       unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
           let ptr = System.alloc(layout);
           if !ptr.is_null() {
               ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
           }
           ptr
       }
       
       unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
           System.dealloc(ptr, layout);
           DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
       }
   }
   
   /// Get current memory usage statistics
   pub fn memory_stats() -> MemoryStats {
       MemoryStats {
           allocated: ALLOCATED.load(Ordering::Relaxed),
           deallocated: DEALLOCATED.load(Ordering::Relaxed),
           net_allocated: ALLOCATED.load(Ordering::Relaxed) 
                         - DEALLOCATED.load(Ordering::Relaxed),
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct MemoryStats {
       pub allocated: usize,
       pub deallocated: usize,
       pub net_allocated: usize,
   }
   
   impl MemoryStats {
       pub fn allocated_mb(&self) -> f64 {
           self.allocated as f64 / (1024.0 * 1024.0)
       }
       
       pub fn net_allocated_mb(&self) -> f64 {
           self.net_allocated as f64 / (1024.0 * 1024.0)
       }
   }
   
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_memory_stats() {
           let stats = memory_stats();
           // Basic sanity check - stats should be non-negative
           assert!(stats.allocated >= stats.deallocated);
       }
   }
   ```
4. Create memory test: `rustc --test src/memory_utils.rs && memory_utils.exe`
5. Clean up: `del memory_utils.exe`
6. Commit memory profiling: `git add Cargo.toml memory_profile.toml src/memory_utils.rs && git commit -m "Configure memory profiling"`

## Expected Output
- Memory profiling dependencies added to workspace
- Memory profiling configuration created
- Memory tracking utilities implemented
- Memory profiling setup committed

## Success Criteria
- [ ] Memory profiling dependencies added to Cargo.toml
- [ ] `memory_profile.toml` configuration created
- [ ] Memory utilities compile and test successfully
- [ ] Memory profiling setup committed to Git

## Next Task
task_027_setup_development_scripts.md