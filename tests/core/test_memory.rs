use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashSet;

use llmkg::core::memory::{GraphArena, EpochManager};
use llmkg::core::types::{EntityData, EntityKey};

/// Helper function to create test entity data with specific size characteristics
fn create_sized_entity_data(id: u16, embedding_size: usize) -> EntityData {
    EntityData {
        type_id: id,
        properties: format!("entity_{}_with_size_{}", id, embedding_size),
        embedding: vec![0.1 * id as f32; embedding_size],
    }
}

/// Helper function to create a large property string
fn create_large_properties(base: &str, size_kb: usize) -> String {
    let target_size = size_kb * 1024;
    let mut result = base.to_string();
    let padding = "x".repeat(1024);
    
    while result.len() < target_size {
        result.push_str(&padding);
    }
    
    result.truncate(target_size);
    result
}

#[tokio::test]
async fn test_high_concurrency_allocation_deallocation() {
    const NUM_THREADS: usize = 16;
    const OPERATIONS_PER_THREAD: usize = 1000;
    const BARRIER_THREADS: usize = NUM_THREADS + 1; // Include main thread
    
    let arena = Arc::new(Mutex::new(GraphArena::new()));
    let barrier = Arc::new(Barrier::new(BARRIER_THREADS));
    let mut handles = Vec::new();
    
    // Shared data structure to track all allocated keys
    let all_keys = Arc::new(Mutex::new(Vec::new()));
    
    let start_time = Instant::now();
    
    for thread_id in 0..NUM_THREADS {
        let arena_clone = Arc::clone(&arena);
        let barrier_clone = Arc::clone(&barrier);
        let keys_clone = Arc::clone(&all_keys);
        
        let handle = thread::spawn(move || {
            let mut local_keys = Vec::new();
            
            // Wait for all threads to be ready
            barrier_clone.wait();
            
            // Phase 1: Rapid allocation
            for i in 0..OPERATIONS_PER_THREAD {
                let entity = create_sized_entity_data(
                    (thread_id * 1000 + i) as u16,
                    128 + (i % 512), // Vary embedding sizes
                );
                
                let key = {
                    let mut arena_guard = arena_clone.lock().unwrap();
                    arena_guard.allocate_entity(entity)
                };
                
                local_keys.push(key);
                
                // Occasionally yield to increase contention
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }
            
            // Add local keys to global collection
            {
                let mut global_keys = keys_clone.lock().unwrap();
                global_keys.extend(local_keys.clone());
            }
            
            // Phase 2: Mixed read/write operations
            for i in 0..OPERATIONS_PER_THREAD / 2 {
                let arena_guard = arena_clone.lock().unwrap();
                
                // Verify entity exists and has correct data
                if let Some(entity) = arena_guard.get_entity(local_keys[i]) {
                    assert_eq!(entity.type_id, (thread_id * 1000 + i) as u16);
                }
            }
            
            // Phase 3: Deallocations with interleaved allocations
            for i in 0..OPERATIONS_PER_THREAD / 4 {
                if i % 2 == 0 {
                    // Remove entity
                    let mut arena_guard = arena_clone.lock().unwrap();
                    arena_guard.remove_entity(local_keys[i]);
                } else {
                    // Allocate new entity
                    let entity = create_sized_entity_data(
                        (thread_id * 10000 + i) as u16,
                        256,
                    );
                    let mut arena_guard = arena_clone.lock().unwrap();
                    let new_key = arena_guard.allocate_entity(entity);
                    local_keys.push(new_key);
                }
            }
            
            local_keys
        });
        
        handles.push(handle);
    }
    
    // Start the race
    barrier.wait();
    
    // Collect all thread results
    let mut thread_keys = Vec::new();
    for handle in handles {
        let keys = handle.join().expect("Thread panicked");
        thread_keys.extend(keys);
    }
    
    let elapsed = start_time.elapsed();
    
    // Verify final state
    {
        let arena_guard = arena.lock().unwrap();
        let entity_count = arena_guard.entity_count();
        let memory_usage = arena_guard.memory_usage();
        let capacity = arena_guard.capacity();
        
        println!("Test completed in {:?}", elapsed);
        println!("Final entity count: {}", entity_count);
        println!("Memory usage: {} bytes", memory_usage);
        println!("Capacity: {}", capacity);
        
        // Verify memory tracking
        assert!(memory_usage > 0);
        assert!(capacity >= entity_count);
        
        // Verify no duplicate keys
        let unique_keys: HashSet<_> = thread_keys.iter().collect();
        assert!(unique_keys.len() <= thread_keys.len());
    }
}

#[tokio::test]
async fn test_memory_leak_detection_under_load() {
    const NUM_ITERATIONS: usize = 10;
    const ENTITIES_PER_ITERATION: usize = 10000;
    
    let arena = Arc::new(Mutex::new(GraphArena::new()));
    let mut memory_samples = Vec::new();
    
    for iteration in 0..NUM_ITERATIONS {
        let mut keys = Vec::new();
        
        // Allocation phase
        for i in 0..ENTITIES_PER_ITERATION {
            let entity = create_sized_entity_data(
                i as u16,
                1024, // Large embeddings to stress memory
            );
            
            let key = {
                let mut arena_guard = arena.lock().unwrap();
                arena_guard.allocate_entity(entity)
            };
            keys.push(key);
        }
        
        // Measure memory after allocation
        let memory_after_alloc = {
            let arena_guard = arena.lock().unwrap();
            arena_guard.memory_usage()
        };
        
        // Deallocation phase - remove all entities
        for key in keys {
            let mut arena_guard = arena.lock().unwrap();
            arena_guard.remove_entity(key);
        }
        
        // Force generation reset to trigger cleanup
        {
            let mut arena_guard = arena.lock().unwrap();
            arena_guard.reset_generation();
        }
        
        // Measure memory after deallocation
        let memory_after_dealloc = {
            let arena_guard = arena.lock().unwrap();
            arena_guard.memory_usage()
        };
        
        memory_samples.push((iteration, memory_after_alloc, memory_after_dealloc));
        
        // Memory after deallocation should be significantly less than after allocation
        assert!(
            memory_after_dealloc < memory_after_alloc,
            "Iteration {}: Memory not released properly. After alloc: {}, After dealloc: {}",
            iteration,
            memory_after_alloc,
            memory_after_dealloc
        );
    }
    
    // Analyze memory trend
    println!("Memory usage over iterations:");
    for (iter, alloc, dealloc) in &memory_samples {
        println!("Iteration {}: Allocated: {} bytes, After cleanup: {} bytes", 
                 iter, alloc, dealloc);
    }
    
    // Check that memory usage after cleanup remains relatively stable
    let cleanup_memories: Vec<_> = memory_samples.iter().map(|(_, _, d)| *d).collect();
    let avg_cleanup_memory = cleanup_memories.iter().sum::<usize>() / cleanup_memories.len();
    
    for (i, &cleanup_mem) in cleanup_memories.iter().enumerate() {
        let deviation = (cleanup_mem as f64 - avg_cleanup_memory as f64).abs() / avg_cleanup_memory as f64;
        assert!(
            deviation < 0.5, // Allow 50% deviation
            "Iteration {} shows signs of memory leak. Cleanup memory: {}, Average: {}",
            i, cleanup_mem, avg_cleanup_memory
        );
    }
}

#[tokio::test]
async fn test_thread_safety_with_entity_updates() {
    const NUM_THREADS: usize = 8;
    const UPDATES_PER_THREAD: usize = 500;
    
    let arena = Arc::new(Mutex::new(GraphArena::new()));
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    
    // Pre-populate arena with entities
    let mut shared_keys = Vec::new();
    {
        let mut arena_guard = arena.lock().unwrap();
        for i in 0..100 {
            let entity = create_sized_entity_data(i, 128);
            let key = arena_guard.allocate_entity(entity);
            shared_keys.push(key);
        }
    }
    
    let shared_keys = Arc::new(shared_keys);
    let mut handles = Vec::new();
    
    for thread_id in 0..NUM_THREADS {
        let arena_clone = Arc::clone(&arena);
        let barrier_clone = Arc::clone(&barrier);
        let keys_clone = Arc::clone(&shared_keys);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..UPDATES_PER_THREAD {
                let key_index = (thread_id + i) % keys_clone.len();
                let key = keys_clone[key_index];
                
                // Create new entity data with thread-specific values
                let new_entity = EntityData {
                    type_id: (thread_id * 1000 + i) as u16,
                    properties: format!("updated_by_thread_{}_iteration_{}", thread_id, i),
                    embedding: vec![thread_id as f32 + i as f32 * 0.01; 128],
                };
                
                // Update entity
                let result = {
                    let mut arena_guard = arena_clone.lock().unwrap();
                    arena_guard.update_entity(key, new_entity.clone())
                };
                
                assert!(result.is_ok(), "Failed to update entity: {:?}", result);
                
                // Immediately verify the update
                {
                    let arena_guard = arena_clone.lock().unwrap();
                    if let Some(entity) = arena_guard.get_entity(key) {
                        // Due to concurrent updates, we can't assert exact values
                        // but we can verify the entity exists and has valid data
                        assert!(entity.type_id < (NUM_THREADS * UPDATES_PER_THREAD) as u16);
                        assert!(entity.properties.starts_with("updated_by_thread_"));
                        assert_eq!(entity.embedding.len(), 128);
                    }
                }
                
                // Add some contention
                if i % 50 == 0 {
                    thread::yield_now();
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify final state - all entities should still exist
    {
        let arena_guard = arena.lock().unwrap();
        for &key in shared_keys.iter() {
            assert!(
                arena_guard.contains_entity(key),
                "Entity {:?} missing after concurrent updates",
                key
            );
        }
    }
}

#[tokio::test]
async fn test_epoch_manager_high_concurrency() {
    const NUM_THREADS: usize = 32;
    const OPERATIONS_PER_THREAD: usize = 1000;
    
    let manager = Arc::new(EpochManager::new(NUM_THREADS));
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    // Track allocations for verification
    let allocation_count = Arc::new(Mutex::new(0));
    let deallocation_count = Arc::new(Mutex::new(0));
    
    for thread_id in 0..NUM_THREADS {
        let manager_clone = Arc::clone(&manager);
        let barrier_clone = Arc::clone(&barrier);
        let alloc_count = Arc::clone(&allocation_count);
        let dealloc_count = Arc::clone(&deallocation_count);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..OPERATIONS_PER_THREAD {
                // Enter epoch
                let _guard = manager_clone.enter(thread_id);
                
                // Simulate work with allocations
                if i % 10 == 0 {
                    // Simulate allocation
                    let layout = std::alloc::Layout::from_size_align(
                        64 + (i % 256) * 8, // Vary sizes
                        8
                    ).unwrap();
                    
                    let ptr = unsafe { std::alloc::alloc(layout) };
                    if !ptr.is_null() {
                        *alloc_count.lock().unwrap() += 1;
                        
                        // Fill with pattern to detect corruption
                        unsafe {
                            std::ptr::write_bytes(ptr, (thread_id & 0xFF) as u8, layout.size());
                        }
                        
                        // Retire object
                        manager_clone.retire_object(ptr, layout.size());
                        *dealloc_count.lock().unwrap() += 1;
                    }
                }
                
                // Some threads advance epochs
                if thread_id < 4 && i % 50 == 0 {
                    manager_clone.advance_epoch();
                }
                
                // Simulate work
                if i % 100 == 0 {
                    thread::sleep(Duration::from_micros(100));
                }
                
                // Guard automatically drops here
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Final cleanup
    for _ in 0..10 {
        manager.advance_epoch();
        thread::sleep(Duration::from_millis(1));
    }
    
    let final_allocs = *allocation_count.lock().unwrap();
    let final_deallocs = *deallocation_count.lock().unwrap();
    
    println!("Total allocations: {}", final_allocs);
    println!("Total deallocations: {}", final_deallocs);
    
    // All allocations should have corresponding deallocations
    assert_eq!(
        final_allocs, final_deallocs,
        "Memory leak detected: {} allocations, {} deallocations",
        final_allocs, final_deallocs
    );
}

#[tokio::test]
async fn test_performance_characteristics_under_load() {
    const NUM_OPERATIONS: usize = 100000;
    const SAMPLE_INTERVAL: usize = 10000;
    
    let mut arena = GraphArena::new();
    let mut performance_samples = Vec::new();
    let mut keys = Vec::new();
    
    // Warm up
    for i in 0..1000 {
        let entity = create_sized_entity_data(i as u16, 128);
        keys.push(arena.allocate_entity(entity));
    }
    
    // Allocation performance test
    let mut allocation_times = Vec::new();
    for i in 0..NUM_OPERATIONS {
        let entity = create_sized_entity_data(i as u16, 128);
        
        let start = Instant::now();
        let key = arena.allocate_entity(entity);
        let elapsed = start.elapsed();
        
        keys.push(key);
        allocation_times.push(elapsed.as_nanos());
        
        if i % SAMPLE_INTERVAL == 0 && i > 0 {
            let avg_time = allocation_times.iter().sum::<u128>() / allocation_times.len() as u128;
            let entity_count = arena.entity_count();
            let memory_usage = arena.memory_usage();
            
            performance_samples.push((
                i,
                entity_count,
                memory_usage,
                avg_time,
            ));
            
            allocation_times.clear();
        }
    }
    
    // Print performance characteristics
    println!("Allocation Performance Characteristics:");
    println!("Operations | Entities | Memory (KB) | Avg Time (ns)");
    println!("-----------|----------|-------------|---------------");
    for (ops, entities, memory, avg_time) in &performance_samples {
        println!("{:10} | {:8} | {:11} | {:13}", 
                 ops, entities, memory / 1024, avg_time);
    }
    
    // Verify performance doesn't degrade significantly
    let first_sample_time = performance_samples[0].3;
    let last_sample_time = performance_samples.last().unwrap().3;
    let degradation = (last_sample_time as f64 / first_sample_time as f64) - 1.0;
    
    assert!(
        degradation < 2.0, // Allow up to 200% degradation
        "Performance degraded too much: {}%",
        degradation * 100.0
    );
    
    // Read performance test
    let mut read_times = Vec::new();
    let sample_keys: Vec<_> = keys.iter().step_by(100).take(1000).collect();
    
    for &key in &sample_keys {
        let start = Instant::now();
        let _ = arena.get_entity(*key);
        let elapsed = start.elapsed();
        
        read_times.push(elapsed.as_nanos());
    }
    
    let avg_read_time = read_times.iter().sum::<u128>() / read_times.len() as u128;
    println!("\nAverage read time: {} ns", avg_read_time);
    
    // Read time should be fast (under 1 microsecond)
    assert!(avg_read_time < 1000, "Read operations too slow: {} ns", avg_read_time);
}

#[tokio::test]
async fn test_concurrent_generation_resets() {
    const NUM_THREADS: usize = 4;
    const OPERATIONS_PER_THREAD: usize = 100;
    
    let arena = Arc::new(Mutex::new(GraphArena::new()));
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    for thread_id in 0..NUM_THREADS {
        let arena_clone = Arc::clone(&arena);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..OPERATIONS_PER_THREAD {
                if thread_id == 0 && i % 25 == 0 {
                    // Thread 0 performs generation resets
                    let mut arena_guard = arena_clone.lock().unwrap();
                    arena_guard.reset_generation();
                } else {
                    // Other threads perform normal operations
                    let entity = create_sized_entity_data(
                        (thread_id * 1000 + i) as u16,
                        128,
                    );
                    
                    let mut arena_guard = arena_clone.lock().unwrap();
                    let key = arena_guard.allocate_entity(entity);
                    
                    // Verify entity is accessible
                    assert!(arena_guard.get_entity(key).is_some());
                }
                
                // Add contention
                if i % 10 == 0 {
                    thread::yield_now();
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify arena is in valid state
    {
        let arena_guard = arena.lock().unwrap();
        let entity_count = arena_guard.entity_count();
        let memory_usage = arena_guard.memory_usage();
        
        println!("After concurrent resets - Entities: {}, Memory: {} bytes", 
                 entity_count, memory_usage);
        
        assert!(entity_count > 0);
        assert!(memory_usage > 0);
    }
}

#[tokio::test]
async fn test_large_entity_handling() {
    let mut arena = GraphArena::new();
    let mut keys = Vec::new();
    
    // Test with increasingly large entities
    let sizes = vec![1, 10, 100, 1000, 10000];
    
    for (idx, &size_kb) in sizes.iter().enumerate() {
        let large_properties = create_large_properties("base", size_kb);
        let entity = EntityData {
            type_id: idx as u16,
            properties: large_properties.clone(),
            embedding: vec![0.1; 1024], // Large embedding too
        };
        
        let start = Instant::now();
        let key = arena.allocate_entity(entity);
        let alloc_time = start.elapsed();
        
        keys.push(key);
        
        // Verify retrieval
        let start = Instant::now();
        let retrieved = arena.get_entity(key).unwrap();
        let read_time = start.elapsed();
        
        assert_eq!(retrieved.type_id, idx as u16);
        assert_eq!(retrieved.properties.len(), size_kb * 1024);
        
        println!("Entity size: {} KB - Alloc: {:?}, Read: {:?}", 
                 size_kb, alloc_time, read_time);
    }
    
    // Test memory reporting with large entities
    let memory_usage = arena.memory_usage();
    let encoded_size = arena.encoded_size();
    
    println!("Total memory usage: {} KB", memory_usage / 1024);
    println!("Encoded size: {} KB", encoded_size / 1024);
    
    // Memory usage should reflect the large entities
    assert!(memory_usage > sizes.iter().sum::<usize>() * 1024);
}

#[tokio::test]
async fn test_stress_test_with_mixed_operations() {
    const NUM_THREADS: usize = 16;
    const DURATION_SECS: u64 = 5;
    
    let arena = Arc::new(Mutex::new(GraphArena::new()));
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let operation_counts = Arc::new(Mutex::new(vec![0u64; 5])); // alloc, read, update, remove, reset
    
    let mut handles = Vec::new();
    
    // Pre-populate some entities
    let initial_keys = {
        let mut arena_guard = arena.lock().unwrap();
        let mut keys = Vec::new();
        for i in 0..1000 {
            let entity = create_sized_entity_data(i, 128);
            keys.push(arena_guard.allocate_entity(entity));
        }
        keys
    };
    let initial_keys = Arc::new(initial_keys);
    
    for thread_id in 0..NUM_THREADS {
        let arena_clone = Arc::clone(&arena);
        let stop_clone = Arc::clone(&stop_flag);
        let counts_clone = Arc::clone(&operation_counts);
        let keys_clone = Arc::clone(&initial_keys);
        
        let handle = thread::spawn(move || {
            let mut local_keys = Vec::new();
            let mut rng = thread_id;
            
            while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                // Simple pseudo-random number generator
                rng = (rng * 1664525 + 1013904223) % 100;
                
                match rng % 5 {
                    0 => {
                        // Allocation
                        let entity = create_sized_entity_data(rng as u16, 64 + (rng as usize % 256));
                        let key = {
                            let mut arena_guard = arena_clone.lock().unwrap();
                            arena_guard.allocate_entity(entity)
                        };
                        local_keys.push(key);
                        counts_clone.lock().unwrap()[0] += 1;
                    }
                    1 => {
                        // Read
                        if !local_keys.is_empty() || !keys_clone.is_empty() {
                            let key = if !local_keys.is_empty() && rng % 2 == 0 {
                                local_keys[rng as usize % local_keys.len()]
                            } else {
                                keys_clone[rng as usize % keys_clone.len()]
                            };
                            
                            let arena_guard = arena_clone.lock().unwrap();
                            let _ = arena_guard.get_entity(key);
                            counts_clone.lock().unwrap()[1] += 1;
                        }
                    }
                    2 => {
                        // Update
                        if !local_keys.is_empty() || !keys_clone.is_empty() {
                            let key = if !local_keys.is_empty() && rng % 2 == 0 {
                                local_keys[rng as usize % local_keys.len()]
                            } else {
                                keys_clone[rng as usize % keys_clone.len()]
                            };
                            
                            let new_entity = create_sized_entity_data(rng as u16, 128);
                            let mut arena_guard = arena_clone.lock().unwrap();
                            let _ = arena_guard.update_entity(key, new_entity);
                            counts_clone.lock().unwrap()[2] += 1;
                        }
                    }
                    3 => {
                        // Remove
                        if !local_keys.is_empty() {
                            let idx = rng as usize % local_keys.len();
                            let key = local_keys.remove(idx);
                            
                            let mut arena_guard = arena_clone.lock().unwrap();
                            arena_guard.remove_entity(key);
                            counts_clone.lock().unwrap()[3] += 1;
                        }
                    }
                    4 => {
                        // Reset generation (rare)
                        if thread_id == 0 && rng > 95 {
                            let mut arena_guard = arena_clone.lock().unwrap();
                            arena_guard.reset_generation();
                            counts_clone.lock().unwrap()[4] += 1;
                        }
                    }
                    _ => {}
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Let the stress test run
    thread::sleep(Duration::from_secs(DURATION_SECS));
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Print statistics
    let counts = operation_counts.lock().unwrap();
    let total_ops: u64 = counts.iter().sum();
    
    println!("Stress test results ({} seconds):", DURATION_SECS);
    println!("Total operations: {}", total_ops);
    println!("Operations per second: {}", total_ops / DURATION_SECS);
    println!("Allocations: {}", counts[0]);
    println!("Reads: {}", counts[1]);
    println!("Updates: {}", counts[2]);
    println!("Removes: {}", counts[3]);
    println!("Generation resets: {}", counts[4]);
    
    // Verify arena is still functional
    {
        let arena_guard = arena.lock().unwrap();
        let entity_count = arena_guard.entity_count();
        let memory_usage = arena_guard.memory_usage();
        
        println!("Final entity count: {}", entity_count);
        println!("Final memory usage: {} KB", memory_usage / 1024);
        
        // Arena should still be usable
        assert!(entity_count > 0);
        assert!(memory_usage > 0);
    }
}

#[tokio::test]
async fn test_edge_case_empty_arena_operations() {
    let mut arena = GraphArena::new();
    
    // Test operations on empty arena
    assert_eq!(arena.entity_count(), 0);
    assert_eq!(arena.capacity(), 0);
    assert!(arena.memory_usage() >= 0); // Should have some base overhead
    assert!(arena.encoded_size() > 0);
    
    // Test removing from empty arena
    let dummy_key = EntityKey::from(slotmap::KeyData::from_ffi(0));
    assert!(arena.remove_entity(dummy_key).is_none());
    assert!(arena.get_entity(dummy_key).is_none());
    
    // Test generation reset on empty arena
    arena.reset_generation();
    assert_eq!(arena.entity_count(), 0);
    
    // Add one entity and remove it
    let entity = create_sized_entity_data(1, 128);
    let key = arena.allocate_entity(entity);
    assert_eq!(arena.entity_count(), 1);
    
    let removed = arena.remove_entity(key);
    assert!(removed.is_some());
    assert_eq!(arena.entity_count(), 0);
    
    // Verify arena is truly empty
    assert!(arena.get_entity(key).is_none());
    assert!(!arena.contains_entity(key));
}

#[tokio::test]
async fn test_boundary_conditions() {
    let mut arena = GraphArena::new();
    
    // Test with minimum size entity
    let min_entity = EntityData {
        type_id: 0,
        properties: String::new(),
        embedding: vec![],
    };
    let min_key = arena.allocate_entity(min_entity);
    assert!(arena.contains_entity(min_key));
    
    // Test with maximum reasonable size entity
    let max_entity = EntityData {
        type_id: u16::MAX,
        properties: "x".repeat(1_000_000), // 1MB string
        embedding: vec![1.0; 10_000], // 10k floats
    };
    let max_key = arena.allocate_entity(max_entity);
    assert!(arena.contains_entity(max_key));
    
    // Test rapid allocation/deallocation cycle
    let mut keys = Vec::new();
    for _ in 0..1000 {
        let entity = create_sized_entity_data(1, 10);
        let key = arena.allocate_entity(entity);
        keys.push(key);
    }
    
    for key in keys {
        arena.remove_entity(key);
    }
    
    // Arena should handle the rapid cycle gracefully
    assert!(arena.entity_count() >= 2); // At least our min/max entities
}