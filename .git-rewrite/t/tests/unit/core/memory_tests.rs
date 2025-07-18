//! Memory Management Unit Tests
//!
//! Tests for memory allocation, deallocation, usage tracking,
//! and memory leak detection in core components.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::core::{Entity, EntityKey, KnowledgeGraph, memory::MemoryManager};

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_manager_basic_operations() {
        let mut memory_manager = MemoryManager::new();
        
        // Test initial state
        assert_eq!(memory_manager.total_allocated(), 0);
        assert_eq!(memory_manager.active_allocations(), 0);
        
        // Test memory allocation
        let size1 = 1024;
        let ptr1 = memory_manager.allocate(size1).unwrap();
        assert_eq!(memory_manager.total_allocated(), size1);
        assert_eq!(memory_manager.active_allocations(), 1);
        
        // Test second allocation
        let size2 = 2048;
        let ptr2 = memory_manager.allocate(size2).unwrap();
        assert_eq!(memory_manager.total_allocated(), size1 + size2);
        assert_eq!(memory_manager.active_allocations(), 2);
        
        // Test deallocation
        memory_manager.deallocate(ptr1, size1).unwrap();
        assert_eq!(memory_manager.total_allocated(), size2);
        assert_eq!(memory_manager.active_allocations(), 1);
        
        // Test final cleanup
        memory_manager.deallocate(ptr2, size2).unwrap();
        assert_eq!(memory_manager.total_allocated(), 0);
        assert_eq!(memory_manager.active_allocations(), 0);
    }

    #[test]
    fn test_memory_leak_detection() {
        let mut memory_manager = MemoryManager::new();
        
        // Allocate memory without deallocating
        let _ptr1 = memory_manager.allocate(1024).unwrap();
        let _ptr2 = memory_manager.allocate(2048).unwrap();
        
        // Check for leaks
        let leaks = memory_manager.detect_leaks();
        assert_eq!(leaks.len(), 2);
        assert_eq!(leaks[0].size, 1024);
        assert_eq!(leaks[1].size, 2048);
        
        // Verify leak reporting
        let leak_report = memory_manager.generate_leak_report();
        assert!(leak_report.contains("2 memory leaks detected"));
        assert!(leak_report.contains("Total leaked: 3072 bytes"));
    }

    #[test]
    fn test_entity_memory_tracking() {
        let mut entity = Entity::new(EntityKey::from_hash("memory_test"), "Memory Test".to_string());
        let initial_memory = entity.memory_usage();
        
        // Add attributes and verify memory tracking
        entity.add_attribute("key1", "value1");
        let memory_after_1 = entity.memory_usage();
        assert!(memory_after_1 > initial_memory);
        
        let expected_increase = "key1".len() + "value1".len() + ATTRIBUTE_OVERHEAD;
        assert_eq!(memory_after_1 - initial_memory, expected_increase as u64);
        
        // Add more attributes
        entity.add_attribute("key2", "value2");
        entity.add_attribute("key3", "value3");
        let memory_after_3 = entity.memory_usage();
        
        let total_expected = initial_memory + 
            ("key1".len() + "value1".len() + ATTRIBUTE_OVERHEAD) as u64 +
            ("key2".len() + "value2".len() + ATTRIBUTE_OVERHEAD) as u64 +
            ("key3".len() + "value3".len() + ATTRIBUTE_OVERHEAD) as u64;
        
        assert_eq!(memory_after_3, total_expected);
        
        // Remove attribute and verify memory decrease
        entity.remove_attribute("key2");
        let memory_after_removal = entity.memory_usage();
        let expected_decrease = ("key2".len() + "value2".len() + ATTRIBUTE_OVERHEAD) as u64;
        assert_eq!(memory_after_3 - memory_after_removal, expected_decrease);
    }

    #[test]
    fn test_graph_memory_scaling() {
        let sizes = vec![10, 100, 1000];
        let mut previous_memory_per_entity = 0u64;
        
        for &size in &sizes {
            let graph = create_test_graph(size, size * 2);
            let total_memory = graph.memory_usage();
            let memory_per_entity = total_memory / size as u64;
            
            println!("Size: {}, Memory per entity: {} bytes", size, memory_per_entity);
            
            // Memory per entity should be reasonable and relatively stable
            assert!(memory_per_entity < 1000, "Memory per entity too high: {}", memory_per_entity);
            
            if previous_memory_per_entity > 0 {
                // Memory per entity shouldn't grow significantly with graph size
                let ratio = memory_per_entity as f64 / previous_memory_per_entity as f64;
                assert!(ratio < 2.0, "Memory per entity growing too fast: {:.2}x", ratio);
            }
            
            previous_memory_per_entity = memory_per_entity;
        }
    }

    #[test]
    fn test_memory_fragmentation() {
        let mut memory_manager = MemoryManager::new();
        let allocation_count = 1000;
        let allocation_size = 128;
        
        // Allocate many small chunks
        let mut ptrs = Vec::new();
        for _ in 0..allocation_count {
            let ptr = memory_manager.allocate(allocation_size).unwrap();
            ptrs.push(ptr);
        }
        
        let total_allocated = memory_manager.total_allocated();
        let expected_total = allocation_count * allocation_size;
        assert_eq!(total_allocated, expected_total);
        
        // Deallocate every other allocation to create fragmentation
        for (i, &ptr) in ptrs.iter().enumerate() {
            if i % 2 == 0 {
                memory_manager.deallocate(ptr, allocation_size).unwrap();
            }
        }
        
        // Check fragmentation metrics
        let fragmentation = memory_manager.calculate_fragmentation();
        assert!(fragmentation >= 0.0 && fragmentation <= 1.0);
        println!("Memory fragmentation: {:.2}%", fragmentation * 100.0);
        
        // For this pattern, fragmentation should be significant
        assert!(fragmentation > 0.3, "Expected significant fragmentation");
        
        // Test compaction
        memory_manager.compact().unwrap();
        let fragmentation_after_compact = memory_manager.calculate_fragmentation();
        assert!(fragmentation_after_compact < fragmentation,
               "Compaction should reduce fragmentation");
    }

    #[test]
    fn test_memory_pool_allocation() {
        let pool_size = 64 * 1024; // 64KB pool
        let mut memory_manager = MemoryManager::with_pool(pool_size);
        
        // Test pool allocation
        let small_allocs = 32;
        let alloc_size = 1024;
        let mut ptrs = Vec::new();
        
        for _ in 0..small_allocs {
            let ptr = memory_manager.allocate(alloc_size).unwrap();
            ptrs.push(ptr);
        }
        
        // Verify all allocations came from the pool
        assert!(memory_manager.pool_utilization() > 0.0);
        assert!(memory_manager.pool_utilization() <= 1.0);
        
        let expected_utilization = (small_allocs * alloc_size) as f64 / pool_size as f64;
        let actual_utilization = memory_manager.pool_utilization();
        assert!((actual_utilization - expected_utilization).abs() < 0.1,
               "Pool utilization incorrect: {} vs {}", actual_utilization, expected_utilization);
        
        // Test pool overflow - allocate more than pool size
        let large_alloc_size = pool_size + 1024;
        let overflow_ptr = memory_manager.allocate(large_alloc_size);
        assert!(overflow_ptr.is_ok(), "Should handle pool overflow gracefully");
        
        // Clean up
        for &ptr in &ptrs {
            memory_manager.deallocate(ptr, alloc_size).unwrap();
        }
        memory_manager.deallocate(overflow_ptr.unwrap(), large_alloc_size).unwrap();
    }

    #[test]
    fn test_memory_alignment() {
        let mut memory_manager = MemoryManager::new();
        
        // Test various alignment requirements
        let alignments = vec![1, 2, 4, 8, 16, 32, 64];
        
        for &alignment in &alignments {
            let ptr = memory_manager.allocate_aligned(1024, alignment).unwrap();
            
            // Verify alignment
            let address = ptr as usize;
            assert_eq!(address % alignment, 0, "Allocation not aligned to {} bytes", alignment);
            
            memory_manager.deallocate(ptr, 1024).unwrap();
        }
        
        // Test default alignment
        let default_ptr = memory_manager.allocate(1024).unwrap();
        let default_address = default_ptr as usize;
        let default_alignment = memory_manager.default_alignment();
        assert_eq!(default_address % default_alignment, 0,
                  "Default allocation not properly aligned");
        
        memory_manager.deallocate(default_ptr, 1024).unwrap();
    }

    #[test]
    fn test_memory_pressure_handling() {
        let memory_limit = 1024 * 1024; // 1MB limit
        let mut memory_manager = MemoryManager::with_limit(memory_limit);
        
        // Allocate up to the limit
        let alloc_size = 128 * 1024; // 128KB per allocation
        let max_allocs = memory_limit / alloc_size;
        let mut ptrs = Vec::new();
        
        for i in 0..max_allocs {
            let ptr = memory_manager.allocate(alloc_size);
            assert!(ptr.is_ok(), "Allocation {} should succeed", i);
            ptrs.push(ptr.unwrap());
        }
        
        // Next allocation should fail or trigger cleanup
        let over_limit_result = memory_manager.allocate(alloc_size);
        assert!(over_limit_result.is_err() || memory_manager.is_under_pressure(),
               "Should detect memory pressure");
        
        // Test pressure relief
        if memory_manager.is_under_pressure() {
            // Deallocate some memory
            for _ in 0..max_allocs/2 {
                if let Some(ptr) = ptrs.pop() {
                    memory_manager.deallocate(ptr, alloc_size).unwrap();
                }
            }
            
            assert!(!memory_manager.is_under_pressure(),
                   "Memory pressure should be relieved");
        }
        
        // Clean up remaining allocations
        for ptr in ptrs {
            memory_manager.deallocate(ptr, alloc_size).unwrap();
        }
    }

    #[test]
    fn test_memory_statistics() {
        let mut memory_manager = MemoryManager::new();
        
        // Perform various allocations
        let ptrs_and_sizes = vec![
            (memory_manager.allocate(1024).unwrap(), 1024),
            (memory_manager.allocate(2048).unwrap(), 2048),
            (memory_manager.allocate(4096).unwrap(), 4096),
        ];
        
        let stats = memory_manager.get_statistics();
        
        // Verify basic statistics
        assert_eq!(stats.total_allocated, 1024 + 2048 + 4096);
        assert_eq!(stats.active_allocations, 3);
        assert_eq!(stats.peak_memory_usage, 1024 + 2048 + 4096);
        
        // Deallocate one allocation
        memory_manager.deallocate(ptrs_and_sizes[1].0, ptrs_and_sizes[1].1).unwrap();
        
        let stats_after_dealloc = memory_manager.get_statistics();
        assert_eq!(stats_after_dealloc.total_allocated, 1024 + 4096);
        assert_eq!(stats_after_dealloc.active_allocations, 2);
        assert_eq!(stats_after_dealloc.peak_memory_usage, 1024 + 2048 + 4096); // Peak unchanged
        
        // Test allocation size distribution
        assert!(stats_after_dealloc.allocation_size_histogram.contains_key(&1024));
        assert!(stats_after_dealloc.allocation_size_histogram.contains_key(&4096));
        assert!(!stats_after_dealloc.allocation_size_histogram.contains_key(&2048));
        
        // Clean up
        memory_manager.deallocate(ptrs_and_sizes[0].0, ptrs_and_sizes[0].1).unwrap();
        memory_manager.deallocate(ptrs_and_sizes[2].0, ptrs_and_sizes[2].1).unwrap();
    }

    #[test]
    fn test_memory_performance() {
        let allocation_count = 10000;
        let allocation_size = 64;
        
        // Test allocation performance
        let mut memory_manager = MemoryManager::new();
        let (ptrs, allocation_time) = measure_execution_time(|| {
            let mut ptrs = Vec::new();
            for _ in 0..allocation_count {
                let ptr = memory_manager.allocate(allocation_size).unwrap();
                ptrs.push(ptr);
            }
            ptrs
        });
        
        println!("Allocation time for {} allocations: {:?}", allocation_count, allocation_time);
        let time_per_allocation = allocation_time.as_nanos() / allocation_count as u128;
        assert!(time_per_allocation < 10000, "Allocation too slow: {} ns", time_per_allocation);
        
        // Test deallocation performance
        let (_, deallocation_time) = measure_execution_time(|| {
            for &ptr in &ptrs {
                memory_manager.deallocate(ptr, allocation_size).unwrap();
            }
        });
        
        println!("Deallocation time for {} deallocations: {:?}", allocation_count, deallocation_time);
        let time_per_deallocation = deallocation_time.as_nanos() / allocation_count as u128;
        assert!(time_per_deallocation < 5000, "Deallocation too slow: {} ns", time_per_deallocation);
        
        // Test total memory management overhead
        let total_time = allocation_time + deallocation_time;
        let total_bytes = allocation_count * allocation_size;
        let throughput = total_bytes as f64 / total_time.as_secs_f64() / (1024.0 * 1024.0); // MB/s
        
        println!("Memory management throughput: {:.2} MB/s", throughput);
        assert!(throughput > 100.0, "Memory management throughput too low: {:.2} MB/s", throughput);
    }

    #[test]
    fn test_concurrent_memory_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let memory_manager = Arc::new(Mutex::new(MemoryManager::new()));
        let thread_count = 4;
        let allocations_per_thread = 1000;
        let allocation_size = 128;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let manager_clone = Arc::clone(&memory_manager);
            
            let handle = thread::spawn(move || {
                let mut local_ptrs = Vec::new();
                
                // Allocate memory
                for _ in 0..allocations_per_thread {
                    let ptr = {
                        let mut manager = manager_clone.lock().unwrap();
                        manager.allocate(allocation_size).unwrap()
                    };
                    local_ptrs.push(ptr);
                }
                
                // Deallocate memory
                for &ptr in &local_ptrs {
                    let mut manager = manager_clone.lock().unwrap();
                    manager.deallocate(ptr, allocation_size).unwrap();
                }
                
                thread_id
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            let thread_id = handle.join().unwrap();
            println!("Thread {} completed successfully", thread_id);
        }
        
        // Verify final state
        let manager = memory_manager.lock().unwrap();
        assert_eq!(manager.total_allocated(), 0);
        assert_eq!(manager.active_allocations(), 0);
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_operations, thread_count * allocations_per_thread * 2); // alloc + dealloc
    }
}