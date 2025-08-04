# Micro-Task 152: Profile Memory Allocations

## Objective
Profile memory allocation patterns during search operations to identify optimization opportunities.

## Prerequisites
- Task 151 completed (Hybrid search performance measured)
- Memory profiling tools configured

## Time Estimate
8 minutes

## Instructions
1. Create memory allocation profiler `profile_allocations.rs`:
   ```rust
   use std::alloc::{GlobalAlloc, Layout, System};
   use std::sync::atomic::{AtomicUsize, Ordering};
   
   static ALLOCATOR: TrackingAllocator = TrackingAllocator;
   static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
   static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
   
   struct TrackingAllocator;
   
   unsafe impl GlobalAlloc for TrackingAllocator {
       unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
           let ptr = System.alloc(layout);
           if !ptr.is_null() {
               ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
           }
           ptr
       }
       
       unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
           DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
           System.dealloc(ptr, layout);
       }
   }
   
   #[global_allocator]
   static GLOBAL: TrackingAllocator = ALLOCATOR;
   
   fn main() {
       println!("Profiling memory allocations...");
       
       reset_counters();
       simulate_search_operation();
       report_allocation_stats("Search Operation");
       
       reset_counters();
       simulate_indexing_operation();
       report_allocation_stats("Indexing Operation");
       
       reset_counters();
       simulate_hybrid_search();
       report_allocation_stats("Hybrid Search");
   }
   
   fn reset_counters() {
       ALLOCATED.store(0, Ordering::Relaxed);
       DEALLOCATED.store(0, Ordering::Relaxed);
   }
   
   fn simulate_search_operation() {
       let _query_vector: Vec<f32> = vec![0.0; 384];
       let _results: Vec<String> = (0..100).map(|i| format!("result_{}", i)).collect();
   }
   
   fn simulate_indexing_operation() {
       let _documents: Vec<String> = (0..1000).map(|i| format!("document_{} content", i)).collect();
       let _index: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
   }
   
   fn simulate_hybrid_search() {
       simulate_search_operation();
       let _text_results: Vec<u32> = vec![1, 2, 3, 4, 5];
       let _vector_results: Vec<u32> = vec![6, 7, 8, 9, 10];
       let mut _combined = _text_results;
       _combined.extend(_vector_results);
   }
   
   fn report_allocation_stats(operation: &str) {
       let allocated = ALLOCATED.load(Ordering::Relaxed);
       let deallocated = DEALLOCATED.load(Ordering::Relaxed);
       let net = allocated as i64 - deallocated as i64;
       
       println!("{}: Allocated: {} bytes, Deallocated: {} bytes, Net: {} bytes", 
               operation, allocated, deallocated, net);
   }
   ```
2. Run profiler: `cargo run --release --bin profile_allocations`
3. Commit: `git add src/bin/profile_allocations.rs && git commit -m "Profile memory allocations during operations"`

## Success Criteria
- [ ] Memory allocation profiler created
- [ ] Allocation patterns measured
- [ ] Memory usage reported
- [ ] Profiler committed

## Next Task
task_153_benchmark_concurrent_operations.md