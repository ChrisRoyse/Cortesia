use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType,
    BrainInspiredRelationship, RelationType, ActivationPattern
};
use std::sync::Arc;
use tokio::sync::Barrier;
use tokio::time::{timeout, Duration, Instant};
use std::collections::HashMap;

/// Test concurrent reads from multiple threads
#[tokio::test]
async fn test_concurrent_reads() {
    let config = ActivationConfig::default();
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Set up a network with 10 entities
    let mut entity_keys = Vec::new();
    for i in 0..10 {
        let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Hidden);
        let key = engine.add_entity(entity).await.unwrap();
        entity_keys.push(key);
    }
    
    // Spawn 100 concurrent readers
    let num_readers = 100;
    let barrier = Arc::new(Barrier::new(num_readers));
    let mut handles = Vec::new();
    
    for i in 0..num_readers {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            // Wait for all threads to be ready
            barrier_clone.wait().await;
            
            // Perform multiple reads
            for _ in 0..10 {
                let state = engine_clone.get_current_state().await.unwrap();
                assert!(state.len() >= 10);
                
                let stats = engine_clone.get_activation_statistics().await.unwrap();
                assert_eq!(stats.total_entities, 10);
            }
            
            i
        });
        
        handles.push(handle);
    }
    
    // Wait for all readers to complete
    let start = Instant::now();
    for handle in handles {
        let reader_id = handle.await.unwrap();
        assert!(reader_id < num_readers);
    }
    let duration = start.elapsed();
    
    // Concurrent reads should complete quickly
    assert!(duration < Duration::from_secs(5), "Concurrent reads took too long: {:?}", duration);
}

/// Test concurrent propagation from multiple threads
#[tokio::test]
async fn test_concurrent_propagation() {
    let config = ActivationConfig {
        max_iterations: 10,
        convergence_threshold: 0.01,
        decay_rate: 0.1,
        inhibition_strength: 2.0,
        default_threshold: 0.1,
    };
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Create a more complex network
    let num_entities = 20;
    let mut entity_keys = Vec::new();
    for i in 0..num_entities {
        let direction = match i % 3 {
            0 => EntityDirection::Input,
            1 => EntityDirection::Hidden,
            _ => EntityDirection::Output,
        };
        let entity = BrainInspiredEntity::new(format!("entity_{}", i), direction);
        let key = entity.id;
        entity_keys.push(key);
        engine.add_entity(entity).await.unwrap();
    }
    
    // Add some relationships
    for i in 0..num_entities-1 {
        let rel = BrainInspiredRelationship::new(entity_keys[i], entity_keys[i+1], RelationType::RelatedTo);
        engine.add_relationship(rel).await.unwrap();
    }
    
    // Add logic gates
    for i in 0..5 {
        let gate_type = match i % 3 {
            0 => LogicGateType::And,
            1 => LogicGateType::Or,
            _ => LogicGateType::Not,
        };
        let gate = LogicGate::new(gate_type, 0.5);
        engine.add_logic_gate(gate).await.unwrap();
    }
    
    // Run multiple propagations concurrently
    let num_propagations = 50;
    let barrier = Arc::new(Barrier::new(num_propagations));
    let mut handles = Vec::new();
    let entity_keys_arc = Arc::new(entity_keys);
    
    for i in 0..num_propagations {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        let entity_keys_clone = Arc::clone(&entity_keys_arc);
        
        let handle = tokio::spawn(async move {
            // Create unique activation pattern for this thread
            let mut pattern = ActivationPattern::new(format!("pattern_{}", i));
            pattern.activations.insert(entity_keys_clone[i % num_entities], 0.8);
            pattern.activations.insert(entity_keys_clone[(i + 1) % num_entities], 0.6);
            
            // Wait for all threads to be ready
            barrier_clone.wait().await;
            
            // Perform propagation with timeout to catch deadlocks
            let result = timeout(
                Duration::from_secs(10),
                engine_clone.propagate_activation(&pattern)
            ).await.expect("Propagation timed out - possible deadlock")
             .unwrap();
            
            // Verify result
            assert!(!result.final_activations.is_empty());
            assert!(result.iterations_completed > 0);
            assert!(result.total_energy >= 0.0);
            
            result
        });
        
        handles.push(handle);
    }
    
    // Wait for all propagations to complete
    let start = Instant::now();
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    let duration = start.elapsed();
    
    println!("Concurrent propagations completed in {:?}", duration);
    assert!(duration < Duration::from_secs(30), "Concurrent propagations took too long");
    
    // Verify all propagations completed successfully
    assert_eq!(results.len(), num_propagations);
    for result in results {
        assert!(result.iterations_completed <= 10);
    }
}

/// Test read-write conflicts
#[tokio::test]
async fn test_read_write_conflicts() {
    let config = ActivationConfig::default();
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Set up initial entities
    let mut entity_keys = Vec::new();
    for i in 0..20 {
        let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Hidden);
        let key = engine.add_entity(entity).await.unwrap();
        entity_keys.push(key);
    }
    
    let num_readers = 50;
    let num_writers = 10;
    let total_threads = num_readers + num_writers;
    let barrier = Arc::new(Barrier::new(total_threads));
    let mut handles = Vec::new();
    
    // Spawn readers
    for i in 0..num_readers {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            let mut read_count = 0;
            let start = Instant::now();
            
            // Perform continuous reads for 2 seconds
            while start.elapsed() < Duration::from_secs(2) {
                let state = engine_clone.get_current_state().await.unwrap();
                assert!(state.len() >= 20);
                read_count += 1;
                
                // Small yield to allow other tasks
                tokio::task::yield_now().await;
            }
            
            (i, read_count)
        });
        
        handles.push(handle);
    }
    
    // Spawn writers
    let entity_keys_arc = Arc::new(entity_keys);
    for i in 0..num_writers {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        let entity_keys_clone = Arc::clone(&entity_keys_arc);
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            let mut write_count = 0;
            let start = Instant::now();
            
            // Perform writes for 2 seconds
            while start.elapsed() < Duration::from_secs(2) {
                // Add new entity
                let entity_id = 100 + i * 100 + write_count;
                let entity = BrainInspiredEntity::new(
                    format!("writer_{}_{}", i, entity_id), 
                    EntityDirection::Hidden
                );
                let new_key = engine_clone.add_entity(entity).await.unwrap();
                
                // Add relationship to a random existing entity
                if write_count > 0 && !entity_keys_clone.is_empty() {
                    let target_key = entity_keys_clone[write_count % entity_keys_clone.len()];
                    let rel = BrainInspiredRelationship::new(
                        new_key,
                        target_key,
                        RelationType::RelatedTo
                    );
                    engine_clone.add_relationship(rel).await.unwrap();
                }
                
                write_count += 1;
                
                // Small delay between writes
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            
            (num_readers + i, write_count)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut total_reads = 0;
    let mut total_writes = 0;
    
    for handle in handles {
        let (thread_id, count) = handle.await.unwrap();
        if thread_id < num_readers {
            total_reads += count;
        } else {
            total_writes += count;
        }
    }
    
    println!("Total reads: {}, Total writes: {}", total_reads, total_writes);
    
    // Verify reads weren't blocked by writes
    assert!(total_reads > 1000, "Too few reads - readers may have been blocked");
    assert!(total_writes > 10, "Too few writes completed");
}

/// Test deadlock prevention with circular dependencies
#[tokio::test]
async fn test_deadlock_prevention() {
    let config = ActivationConfig::default();
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Create circular network: A -> B -> C -> A
    let entity_names = vec!["A", "B", "C", "D", "E"];
    let mut entity_keys = Vec::new();
    
    // First create all entities
    for name in &entity_names {
        let entity = BrainInspiredEntity::new(name.to_string(), EntityDirection::Hidden);
        let key = engine.add_entity(entity).await.unwrap();
        entity_keys.push(key);
    }
    
    // Then create circular relationships
    for i in 0..entity_keys.len() {
        let next = (i + 1) % entity_keys.len();
        let rel = BrainInspiredRelationship::new(entity_keys[i], entity_keys[next], RelationType::RelatedTo);
        engine.add_relationship(rel).await.unwrap();
        
        // Add reverse relationship for more complexity
        let mut rev_rel = BrainInspiredRelationship::new(entity_keys[next], entity_keys[i], RelationType::RelatedTo);
        rev_rel.is_inhibitory = true;
        engine.add_relationship(rev_rel).await.unwrap();
    }
    
    // Add logic gates that might create dependencies
    for i in 0..3 {
        let gate = LogicGate::new(LogicGateType::And, 0.4);
        engine.add_logic_gate(gate).await.unwrap();
    }
    
    // Run concurrent operations that could deadlock
    let num_operations = 100;
    let barrier = Arc::new(Barrier::new(num_operations));
    let mut handles = Vec::new();
    let entity_keys_arc = Arc::new(entity_keys);
    
    for i in 0..num_operations {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        let entity_keys_clone = Arc::clone(&entity_keys_arc);
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            // Mix of operations that could cause deadlocks
            match i % 4 {
                0 => {
                    // Propagation
                    let mut pattern = ActivationPattern::new(format!("deadlock_test_{}", i));
                    pattern.activations.insert(entity_keys_clone[i % 5], 0.9);
                    
                    let result = timeout(
                        Duration::from_secs(5),
                        engine_clone.propagate_activation(&pattern)
                    ).await;
                    
                    assert!(result.is_ok(), "Propagation deadlocked!");
                    assert!(result.unwrap().is_ok());
                }
                1 => {
                    // Reset activations
                    let result = timeout(
                        Duration::from_secs(5),
                        engine_clone.reset_activations()
                    ).await;
                    
                    assert!(result.is_ok(), "Reset deadlocked!");
                    assert!(result.unwrap().is_ok());
                }
                2 => {
                    // Get statistics
                    let result = timeout(
                        Duration::from_secs(5),
                        engine_clone.get_activation_statistics()
                    ).await;
                    
                    assert!(result.is_ok(), "Statistics deadlocked!");
                    assert!(result.unwrap().is_ok());
                }
                _ => {
                    // Add more relationships
                    let rel = BrainInspiredRelationship::new(
                        entity_keys_clone[i % 5],
                        entity_keys_clone[(i + 2) % 5],
                        RelationType::RelatedTo
                    );
                    
                    let result = timeout(
                        Duration::from_secs(5),
                        engine_clone.add_relationship(rel)
                    ).await;
                    
                    assert!(result.is_ok(), "Add relationship deadlocked!");
                    assert!(result.unwrap().is_ok());
                }
            }
        });
        
        handles.push(handle);
    }
    
    // All operations should complete without deadlock
    let start = Instant::now();
    for handle in handles {
        handle.await.unwrap();
    }
    let duration = start.elapsed();
    
    println!("Deadlock prevention test completed in {:?}", duration);
    assert!(duration < Duration::from_secs(20), "Operations took too long - possible deadlock");
}

/// Test performance scaling with increasing thread count
#[tokio::test]
async fn test_performance_scaling() {
    let config = ActivationConfig {
        max_iterations: 5,
        convergence_threshold: 0.01,
        decay_rate: 0.1,
        inhibition_strength: 2.0,
        default_threshold: 0.1,
    };
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Create a substantial network
    let num_entities = 100;
    for i in 0..num_entities {
        let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Hidden);
        engine.add_entity(entity).await.unwrap();
    }
    
    // Add relationships to create connected network
    for i in 0..num_entities-1 {
        for j in 0..3 {
            let target = (i + j + 1) % num_entities;
            // Get entity keys from the stored entities
            let state = engine.get_current_state().await.unwrap();
            let entity_keys: Vec<_> = state.keys().cloned().collect();
            let rel = BrainInspiredRelationship::new(entity_keys[i], entity_keys[target], RelationType::RelatedTo);
            engine.add_relationship(rel).await.unwrap();
        }
    }
    
    // Test with different thread counts
    let thread_counts = vec![1, 2, 4, 8, 16, 32, 64];
    let mut results = Vec::new();
    
    for &num_threads in &thread_counts {
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = Vec::new();
        
        let operations_per_thread = 1000 / num_threads;
        
        let start = Instant::now();
        
        for i in 0..num_threads {
            let engine_clone = Arc::clone(&engine);
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = tokio::spawn(async move {
                barrier_clone.wait().await;
                
                for j in 0..operations_per_thread {
                    // Mix of operations
                    match (i * operations_per_thread + j) % 3 {
                        0 => {
                            let _ = engine_clone.get_current_state().await.unwrap();
                        }
                        1 => {
                            let _ = engine_clone.get_activation_statistics().await.unwrap();
                        }
                        _ => {
                            let mut pattern = ActivationPattern::new(format!("perf_{}_{}", i, j));
                            // Get entity keys from the stored entities
                            let state = engine_clone.get_current_state().await.unwrap();
                            let entity_keys: Vec<_> = state.keys().cloned().collect();
                            pattern.activations.insert(entity_keys[j % num_entities], 0.7);
                            let _ = engine_clone.propagate_activation(&pattern).await.unwrap();
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.await.unwrap();
        }
        
        let duration = start.elapsed();
        let ops_per_second = (num_threads * operations_per_thread) as f64 / duration.as_secs_f64();
        
        results.push((num_threads, duration, ops_per_second));
        println!("Threads: {}, Duration: {:?}, Ops/sec: {:.2}", 
                 num_threads, duration, ops_per_second);
    }
    
    // Verify performance improves with more threads (up to a point)
    let single_thread_ops = results[0].2;
    let multi_thread_ops = results[3].2; // 8 threads
    
    assert!(multi_thread_ops > single_thread_ops * 2.0, 
            "Multi-threaded performance should be significantly better than single-threaded");
    
    // Verify no significant performance degradation with many threads
    let max_thread_ops = results.last().unwrap().2;
    assert!(max_thread_ops > single_thread_ops, 
            "Performance shouldn't degrade with many threads");
}

/// Test high contention scenario with 1000+ operations
#[tokio::test]
async fn test_high_contention() {
    let config = ActivationConfig {
        max_iterations: 3,
        convergence_threshold: 0.01,
        decay_rate: 0.1,
        inhibition_strength: 2.0,
        default_threshold: 0.1,
    };
    let engine = Arc::new(ActivationPropagationEngine::new(config));
    
    // Create initial network and store entity keys
    let mut entity_keys = Vec::new();
    for i in 0..50 {
        let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Hidden);
        let key = entity.id;
        entity_keys.push(key);
        engine.add_entity(entity).await.unwrap();
    }
    let entity_keys = Arc::new(entity_keys);
    
    // Run 1000 concurrent operations
    let num_operations = 1000;
    let barrier = Arc::new(Barrier::new(num_operations));
    let mut handles = Vec::new();
    
    let start = Instant::now();
    
    for i in 0..num_operations {
        let engine_clone = Arc::clone(&engine);
        let barrier_clone = Arc::clone(&barrier);
        let entity_keys_clone = Arc::clone(&entity_keys);
        
        let handle = tokio::spawn(async move {
            barrier_clone.wait().await;
            
            // Perform operation based on index
            match i % 10 {
                0..=3 => {
                    // 40% reads
                    let _ = engine_clone.get_current_state().await.unwrap();
                }
                4..=6 => {
                    // 30% propagations
                    let mut pattern = ActivationPattern::new(format!("contention_{}", i));
                    pattern.activations.insert(entity_keys_clone[i % 50], 0.6);
                    let _ = timeout(
                        Duration::from_secs(5),
                        engine_clone.propagate_activation(&pattern)
                    ).await.expect("Propagation timed out").unwrap();
                }
                7..=8 => {
                    // 20% writes (add entities)
                    let entity = BrainInspiredEntity::new(
                        format!("contention_entity_{}", i), 
                        EntityDirection::Hidden
                    );
                    let _ = engine_clone.add_entity(entity).await.unwrap();
                }
                _ => {
                    // 10% statistics
                    let _ = engine_clone.get_activation_statistics().await.unwrap();
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations
    for handle in handles {
        handle.await.unwrap();
    }
    
    let duration = start.elapsed();
    let ops_per_second = num_operations as f64 / duration.as_secs_f64();
    
    println!("High contention test: {} operations in {:?} ({:.2} ops/sec)", 
             num_operations, duration, ops_per_second);
    
    // Should complete in reasonable time despite high contention
    assert!(duration < Duration::from_secs(30), 
            "High contention test took too long: {:?}", duration);
    
    // Verify engine state is consistent
    let final_stats = engine.get_activation_statistics().await.unwrap();
    assert!(final_stats.total_entities >= 50);
    println!("Final entity count: {}", final_stats.total_entities);
}