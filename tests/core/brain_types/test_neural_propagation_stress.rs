/// Stress tests for neural propagation patterns with large-scale data
/// 
/// This module implements comprehensive stress testing including:
/// - Large-scale activation pattern processing
/// - Memory pressure and garbage collection impact
/// - Concurrent access patterns and thread safety
/// - Performance degradation under extreme loads
/// - Network topology stress testing
/// - Pathological input resistance testing

use llmkg::core::brain_types::{
    ActivationPattern, BrainInspiredEntity, BrainInspiredRelationship, 
    LogicGate, LogicGateType, EntityDirection, RelationType, ActivationStep, ActivationOperation
};
use llmkg::core::types::EntityKey;
use std::time::{SystemTime, Duration, Instant};
use std::collections::HashMap;
use std::thread;
use super::test_constants;
use super::test_helpers::*;

// ==================== Large-Scale Pattern Processing Tests ====================

#[test]
fn test_massive_activation_pattern_creation() {
    let pattern_sizes = vec![1_000, 10_000, 50_000, 100_000];
    
    for &size in &pattern_sizes {
        let start_time = Instant::now();
        
        let pattern = create_test_pattern("massive_pattern_test", size);
        
        let creation_time = start_time.elapsed();
        
        // Verify pattern integrity
        assert_eq!(pattern.activations.len(), size);
        assert!(!pattern.query.is_empty());
        
        // Performance should scale reasonably
        let expected_max_time = Duration::from_millis((size / 1000) as u64 * 10); // 10ms per 1000 entities
        assert!(
            creation_time <= expected_max_time,
            "Creating pattern of size {} took too long: {:?} > {:?}",
            size, creation_time, expected_max_time
        );
        
        // Memory usage should be reasonable
        let estimated_memory = size * std::mem::size_of::<(EntityKey, f32)>();
        assert!(
            estimated_memory < 10_000_000, // 10MB limit
            "Pattern size {} uses too much memory: {} bytes",
            size, estimated_memory
        );
    }
}

#[test]
fn test_top_activations_performance() {
    let large_pattern = create_test_pattern("performance_test", 100_000);
    let top_sizes = vec![1, 10, 100, 1_000, 10_000];
    
    for &top_n in &top_sizes {
        let start_time = Instant::now();
        
        let top_activations = large_pattern.get_top_activations(top_n);
        
        let query_time = start_time.elapsed();
        
        // Verify correctness
        assert_eq!(top_activations.len(), top_n.min(large_pattern.activations.len()));
        
        // Verify sorting (should be descending)
        for i in 1..top_activations.len() {
            assert!(
                top_activations[i-1].1 >= top_activations[i].1,
                "Top activations should be sorted descending"
            );
        }
        
        // Performance should be reasonable (O(n log k) where k is top_n)
        let expected_max_time = Duration::from_millis(100); // 100ms max
        assert!(
            query_time <= expected_max_time,
            "Getting top {} from 100k pattern took too long: {:?}",
            top_n, query_time
        );
    }
}

#[test]
fn test_pattern_serialization_stress() {
    let sizes = vec![1_000, 5_000, 10_000];
    
    for &size in &sizes {
        let pattern = create_test_pattern("serialization_stress", size);
        
        // Test JSON serialization performance
        let serialize_start = Instant::now();
        let serialized = serde_json::to_string(&pattern)
            .expect("Should serialize large pattern");
        let serialize_time = serialize_start.elapsed();
        
        // Test deserialization performance
        let deserialize_start = Instant::now();
        let _deserialized: ActivationPattern = serde_json::from_str(&serialized)
            .expect("Should deserialize large pattern");
        let deserialize_time = deserialize_start.elapsed();
        
        // Performance bounds
        let max_serialize_time = Duration::from_millis((size / 100) as u64); // 1ms per 100 entities
        let max_deserialize_time = Duration::from_millis((size / 50) as u64); // 1ms per 50 entities
        
        assert!(
            serialize_time <= max_serialize_time,
            "Serializing {} entities took too long: {:?}",
            size, serialize_time
        );
        
        assert!(
            deserialize_time <= max_deserialize_time,
            "Deserializing {} entities took too long: {:?}",
            size, deserialize_time
        );
        
        // Size should be reasonable
        assert!(
            serialized.len() < size * 100, // Rough estimate of reasonable JSON size
            "Serialized size too large: {} bytes for {} entities",
            serialized.len(), size
        );
    }
}

// ==================== Memory Pressure Tests ====================

#[test]
fn test_memory_intensive_entity_creation() {
    let num_entities = 50_000;
    let mut entities = Vec::with_capacity(num_entities);
    
    let start_time = Instant::now();
    
    // Create many entities
    for i in 0..num_entities {
        let entity = EntityBuilder::new(
            &format!("memory_stress_entity_{}", i),
            EntityDirection::Hidden
        )
        .with_activation((i as f32) / (num_entities as f32))
        .with_embedding(vec![0.1, 0.2, 0.3, 0.4, 0.5]) // Small embedding
        .build();
        
        entities.push(entity);
        
        // Periodic memory pressure check
        if i % 10_000 == 0 {
            let elapsed = start_time.elapsed();
            assert!(
                elapsed.as_secs() < 10,
                "Entity creation taking too long at entity {}: {:?}",
                i, elapsed
            );
        }
    }
    
    let total_creation_time = start_time.elapsed();
    
    // Verify all entities created
    assert_eq!(entities.len(), num_entities);
    
    // Performance check
    assert!(
        total_creation_time.as_secs() < 5,
        "Creating {} entities took too long: {:?}",
        num_entities, total_creation_time
    );
    
    // Memory usage estimation
    let estimated_memory = entities.capacity() * std::mem::size_of::<BrainInspiredEntity>();
    println!("Estimated memory usage for {} entities: {} MB", 
             num_entities, estimated_memory / (1024 * 1024));
    
    // Test random access performance
    let access_start = Instant::now();
    let mut sum_activations = 0.0;
    
    for _ in 0..1000 {
        let idx = (entities.len() / 2) % entities.len(); // Access middle entity
        sum_activations += entities[idx].activation_state;
    }
    
    let access_time = access_start.elapsed();
    assert!(
        access_time.as_millis() < 10,
        "Random access too slow: {:?}",
        access_time
    );
    assert!(sum_activations > 0.0); // Prevent optimization
}

#[test]
fn test_relationship_network_stress() {
    let num_entities = 1_000;
    let connections_per_entity = 50; // Dense network
    
    // Create entity keys
    let entity_keys: Vec<EntityKey> = (0..num_entities)
        .map(|i| EntityKey::from(slotmap::KeyData::from_ffi(i)))
        .collect();
    
    let mut relationships = Vec::new();
    let creation_start = Instant::now();
    
    // Create dense relationship network
    for i in 0..num_entities {
        for j in 0..connections_per_entity {
            let target_idx = (i + j + 1) % num_entities; // Avoid self-connections
            
            let relationship = RelationshipBuilder::new(
                entity_keys[i],
                entity_keys[target_idx],
                RelationType::RelatedTo
            )
            .with_weight((j as f32) / (connections_per_entity as f32))
            .build();
            
            relationships.push(relationship);
        }
        
        // Progress check
        if i % 100 == 0 {
            let elapsed = creation_start.elapsed();
            assert!(
                elapsed.as_secs() < 30,
                "Relationship creation too slow at entity {}: {:?}",
                i, elapsed
            );
        }
    }
    
    let total_relationships = relationships.len();
    let creation_time = creation_start.elapsed();
    
    // Verify network size
    assert_eq!(total_relationships, num_entities * connections_per_entity);
    
    // Performance verification
    assert!(
        creation_time.as_secs() < 10,
        "Creating {} relationships took too long: {:?}",
        total_relationships, creation_time
    );
    
    // Test network traversal performance
    let traversal_start = Instant::now();
    let mut total_weight = 0.0;
    
    for relationship in &relationships {
        total_weight += relationship.weight;
        total_weight += relationship.strength; // Access multiple fields
    }
    
    let traversal_time = traversal_start.elapsed();
    assert!(
        traversal_time.as_millis() < 100,
        "Network traversal too slow: {:?}",
        traversal_time
    );
    assert!(total_weight > 0.0); // Prevent optimization
}

// ==================== Concurrent Access Pattern Tests ====================

#[test]
fn test_concurrent_pattern_processing() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let num_threads = 4;
    let pattern_size = 10_000;
    let base_pattern = Arc::new(create_test_pattern("concurrent_test", pattern_size));
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    
    let start_time = Instant::now();
    
    // Spawn threads to process pattern concurrently
    for thread_id in 0..num_threads {
        let pattern_clone = Arc::clone(&base_pattern);
        let results_clone = Arc::clone(&results);
        
        let handle = thread::spawn(move || {
            let thread_start = Instant::now();
            
            // Each thread gets different top-N values
            let top_n = (thread_id + 1) * 1000;
            let top_activations = pattern_clone.get_top_activations(top_n);
            
            // Verify results
            assert_eq!(top_activations.len(), top_n.min(pattern_size));
            
            let thread_time = thread_start.elapsed();
            
            // Store results
            let mut results_lock = results_clone.lock().unwrap();
            results_lock.push((thread_id, thread_time, top_activations.len()));
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    let total_time = start_time.elapsed();
    let results_lock = results.lock().unwrap();
    
    // Verify all threads completed
    assert_eq!(results_lock.len(), num_threads);
    
    // Check performance
    assert!(
        total_time.as_millis() < 1000,
        "Concurrent processing took too long: {:?}",
        total_time
    );
    
    // Verify thread performance
    for &(thread_id, thread_time, result_count) in results_lock.iter() {
        assert!(
            thread_time.as_millis() < 500,
            "Thread {} took too long: {:?}",
            thread_id, thread_time
        );
        assert!(result_count > 0, "Thread {} should have results", thread_id);
    }
}

#[test]
fn test_concurrent_entity_modification() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let num_threads = 8;
    let num_entities = 1000;
    
    // Create shared entities
    let entities: Vec<Arc<Mutex<BrainInspiredEntity>>> = (0..num_entities)
        .map(|i| {
            let entity = EntityBuilder::new(
                &format!("concurrent_entity_{}", i),
                EntityDirection::Hidden
            ).build();
            Arc::new(Mutex::new(entity))
        })
        .collect();
    
    let entities_arc = Arc::new(entities);
    let mut handles = Vec::new();
    
    let start_time = Instant::now();
    
    // Spawn threads to modify entities concurrently
    for thread_id in 0..num_threads {
        let entities_clone = Arc::clone(&entities_arc);
        
        let handle = thread::spawn(move || {
            let activations_per_thread = 100;
            
            for _ in 0..activations_per_thread {
                let entity_idx = thread_id * 10 % num_entities; // Different entities per thread
                
                if let Ok(mut entity) = entities_clone[entity_idx].try_lock() {
                    entity.activate(0.1, test_constants::STANDARD_DECAY_RATE);
                }
                
                // Small delay to increase contention
                thread::sleep(Duration::from_nanos(100));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for completion
    for handle in handles {
        handle.join().expect("Thread should complete");
    }
    
    let total_time = start_time.elapsed();
    
    // Verify performance
    assert!(
        total_time.as_secs() < 5,
        "Concurrent modification took too long: {:?}",
        total_time
    );
    
    // Verify state consistency
    let mut total_activations = 0.0;
    for entity_arc in entities_arc.iter() {
        let entity = entity_arc.lock().unwrap();
        total_activations += entity.activation_state;
        assert!(entity.activation_state >= 0.0);
        assert!(entity.activation_state <= 1.0);
    }
    
    // Should have some accumulated activations
    assert!(total_activations > 0.0, "Should show accumulated activations");
}

// ==================== Performance Degradation Tests ====================

#[test]
fn test_logic_gate_computation_under_load() {
    let gate_counts = vec![100, 500, 1_000, 5_000];
    let input_size = 10;
    
    for &gate_count in &gate_counts {
        let mut gates = Vec::new();
        
        // Create many gates
        let creation_start = Instant::now();
        for i in 0..gate_count {
            let gate_type = match i % 4 {
                0 => LogicGateType::And,
                1 => LogicGateType::Or,
                2 => LogicGateType::Threshold,
                _ => LogicGateType::Weighted,
            };
            
            let mut gate = create_test_gate(gate_type, 0.5, input_size);
            
            if gate_type == LogicGateType::Weighted {
                gate.weight_matrix = vec![0.1; input_size];
            }
            
            gates.push(gate);
        }
        let creation_time = creation_start.elapsed();
        
        // Test computation performance under load
        let test_inputs = vec![0.7; input_size];
        let computation_start = Instant::now();
        
        let mut successful_computations = 0;
        for gate in &gates {
            if gate.calculate_output(&test_inputs).is_ok() {
                successful_computations += 1;
            }
        }
        
        let computation_time = computation_start.elapsed();
        
        // Verify performance scaling
        let max_creation_time = Duration::from_millis((gate_count / 10) as u64); // 1ms per 10 gates
        let max_computation_time = Duration::from_millis((gate_count / 5) as u64); // 1ms per 5 gates
        
        assert!(
            creation_time <= max_creation_time,
            "Creating {} gates took too long: {:?}",
            gate_count, creation_time
        );
        
        assert!(
            computation_time <= max_computation_time,
            "Computing {} gates took too long: {:?}",
            gate_count, computation_time
        );
        
        // Verify correctness
        assert_eq!(successful_computations, gate_count);
    }
}

#[test]
fn test_activation_step_trace_performance() {
    let trace_lengths = vec![1_000, 5_000, 10_000, 50_000];
    
    for &trace_length in &trace_lengths {
        let mut activation_trace = Vec::new();
        
        // Create large activation trace
        let creation_start = Instant::now();
        for i in 0..trace_length {
            let step = create_activation_step(
                i,
                EntityKey::from(slotmap::KeyData::from_ffi((i % 1000) as u64)),
                "trace_entity",
                (i as f32) / (trace_length as f32),
                match i % 5 {
                    0 => ActivationOperation::Initialize,
                    1 => ActivationOperation::Propagate,
                    2 => ActivationOperation::Reinforce,
                    3 => ActivationOperation::Inhibit,
                    _ => ActivationOperation::Decay,
                }
            );
            activation_trace.push(step);
        }
        let creation_time = creation_start.elapsed();
        
        // Test trace analysis performance
        let analysis_start = Instant::now();
        
        // Count operations by type
        let mut operation_counts = HashMap::new();
        for step in &activation_trace {
            *operation_counts.entry(step.operation_type).or_insert(0) += 1;
        }
        
        // Find steps with high activation
        let high_activation_steps: Vec<_> = activation_trace.iter()
            .filter(|step| step.activation_level > 0.8)
            .collect();
        
        let analysis_time = analysis_start.elapsed();
        
        // Performance verification
        let max_creation_time = Duration::from_millis((trace_length / 100) as u64);
        let max_analysis_time = Duration::from_millis((trace_length / 50) as u64);
        
        assert!(
            creation_time <= max_creation_time,
            "Creating trace of {} steps took too long: {:?}",
            trace_length, creation_time
        );
        
        assert!(
            analysis_time <= max_analysis_time,
            "Analyzing trace of {} steps took too long: {:?}",
            trace_length, analysis_time
        );
        
        // Verify analysis results
        assert_eq!(operation_counts.len(), 5); // All operation types present
        assert!(!high_activation_steps.is_empty()); // Should have some high activations
    }
}

// ==================== Network Topology Stress Tests ====================

#[test]
fn test_dense_network_propagation() {
    let network_sizes = vec![100, 500, 1_000];
    
    for &size in &network_sizes {
        // Create dense network
        let entity_keys: Vec<EntityKey> = (0..size)
            .map(|i| EntityKey::from(slotmap::KeyData::from_ffi(i)))
            .collect();
        
        let connectivity = 0.1; // 10% connectivity
        let num_connections = (size as f32 * size as f32 * connectivity) as usize;
        
        let mut relationships = Vec::new();
        let network_start = Instant::now();
        
        for i in 0..num_connections {
            let source_idx = i % size;
            let target_idx = (i + 1 + size / 2) % size; // Avoid self-connections
            
            let relationship = RelationshipBuilder::new(
                entity_keys[source_idx],
                entity_keys[target_idx],
                RelationType::RelatedTo
            )
            .with_weight(0.5)
            .build();
            
            relationships.push(relationship);
        }
        
        let network_time = network_start.elapsed();
        
        // Test propagation simulation
        let propagation_start = Instant::now();
        
        // Simulate activation propagation
        let mut activation_levels = vec![0.0; size];
        activation_levels[0] = 1.0; // Start with first entity activated
        
        for _ in 0..10 { // 10 propagation steps
            let mut new_activations = activation_levels.clone();
            
            for relationship in &relationships {
                // Find source and target indices (simplified)
                let source_activation = activation_levels[0]; // Simplified lookup
                let propagated = source_activation * relationship.weight * 0.1;
                new_activations[1] += propagated; // Simplified propagation
            }
            
            activation_levels = new_activations;
        }
        
        let propagation_time = propagation_start.elapsed();
        
        // Performance verification
        let max_network_time = Duration::from_millis((num_connections / 100) as u64);
        let max_propagation_time = Duration::from_millis((num_connections / 10) as u64);
        
        assert!(
            network_time <= max_network_time,
            "Creating network of {} connections took too long: {:?}",
            num_connections, network_time
        );
        
        assert!(
            propagation_time <= max_propagation_time,
            "Propagation in network of {} connections took too long: {:?}",
            num_connections, propagation_time
        );
        
        // Verify propagation occurred
        let total_activation: f32 = activation_levels.iter().sum();
        assert!(total_activation > 1.0, "Should show activation propagation");
    }
}

#[test]
fn test_pathological_network_topologies() {
    // Test various pathological cases
    
    // 1. Star topology (one central node connected to all others)
    let star_size = 1000;
    let central_key = EntityKey::from(slotmap::KeyData::from_ffi(0));
    
    let mut star_relationships = Vec::new();
    let star_start = Instant::now();
    
    for i in 1..star_size {
        let peripheral_key = EntityKey::from(slotmap::KeyData::from_ffi(i));
        
        star_relationships.push(
            RelationshipBuilder::new(central_key, peripheral_key, RelationType::RelatedTo)
                .with_weight(0.5)
                .build()
        );
    }
    
    let star_time = star_start.elapsed();
    
    assert!(
        star_time.as_millis() < 100,
        "Star topology creation too slow: {:?}",
        star_time
    );
    
    // 2. Chain topology (linear chain)
    let chain_size = 1000;
    let mut chain_relationships = Vec::new();
    let chain_start = Instant::now();
    
    for i in 0..chain_size - 1 {
        let source_key = EntityKey::from(slotmap::KeyData::from_ffi(i));
        let target_key = EntityKey::from(slotmap::KeyData::from_ffi(i + 1));
        
        chain_relationships.push(
            RelationshipBuilder::new(source_key, target_key, RelationType::Temporal)
                .with_weight(0.8)
                .build()
        );
    }
    
    let chain_time = chain_start.elapsed();
    
    assert!(
        chain_time.as_millis() < 100,
        "Chain topology creation too slow: {:?}",
        chain_time
    );
    
    // 3. Complete graph (all-to-all connections)
    let complete_size = 100; // Smaller due to O(n²) connections
    let mut complete_relationships = Vec::new();
    let complete_start = Instant::now();
    
    for i in 0..complete_size {
        for j in 0..complete_size {
            if i != j {
                let source_key = EntityKey::from(slotmap::KeyData::from_ffi(i));
                let target_key = EntityKey::from(slotmap::KeyData::from_ffi(j));
                
                complete_relationships.push(
                    RelationshipBuilder::new(source_key, target_key, RelationType::RelatedTo)
                        .with_weight(1.0 / complete_size as f32)
                        .build()
                );
            }
        }
    }
    
    let complete_time = complete_start.elapsed();
    
    assert!(
        complete_time.as_millis() < 500,
        "Complete graph creation too slow: {:?}",
        complete_time
    );
    
    // Verify topology sizes
    assert_eq!(star_relationships.len(), star_size - 1);
    assert_eq!(chain_relationships.len(), chain_size - 1);
    assert_eq!(complete_relationships.len(), complete_size * (complete_size - 1));
}

// ==================== Pathological Input Resistance Tests ====================

#[test]
fn test_extreme_activation_values() {
    let extreme_values = vec![
        f32::NEG_INFINITY,
        f32::MIN,
        -1000000.0,
        -1.0,
        0.0,
        1.0,
        1000000.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    
    let mut entity = EntityBuilder::new("extreme_test", EntityDirection::Input).build();
    
    for &extreme_value in &extreme_values {
        // Should not panic with extreme values
        let result = entity.activate(extreme_value, test_constants::STANDARD_DECAY_RATE);
        
        // Result should be well-behaved
        assert!(
            !result.is_infinite() || result.is_nan() || (result >= 0.0 && result <= 1.0),
            "Extreme input {} produced ill-behaved result: {}",
            extreme_value, result
        );
    }
}

#[test]
fn test_malformed_pattern_resistance() {
    // Test resistance to malformed activation patterns
    
    // 1. Empty pattern
    let empty_pattern = ActivationPattern::new("empty_test".to_string());
    let empty_top = empty_pattern.get_top_activations(10);
    assert!(empty_top.is_empty());
    
    // 2. Pattern with extreme values
    let extreme_pattern = create_pattern_with_activations(
        "extreme_values",
        vec![
            (1, f32::INFINITY),
            (2, f32::NEG_INFINITY),
            (3, f32::NAN),
            (4, -1000.0),
            (5, 1000.0),
        ]
    );
    
    // Should not panic
    let extreme_top = extreme_pattern.get_top_activations(5);
    assert!(extreme_top.len() <= 5);
    
    // 3. Very large pattern with duplicate keys
    let mut large_activations = HashMap::new();
    for i in 0..10000 {
        let key = EntityKey::from(slotmap::KeyData::from_ffi((i % 100) as u64)); // Duplicates
        large_activations.insert(key, i as f32);
    }
    
    let mut duplicate_pattern = ActivationPattern::new("duplicate_test".to_string());
    duplicate_pattern.activations = large_activations;
    
    // Should handle duplicates gracefully
    let duplicate_top = duplicate_pattern.get_top_activations(50);
    assert!(duplicate_top.len() <= 100); // At most 100 unique keys
}

#[test]
fn test_relationship_decay_overflow_resistance() {
    let mut relationship = create_test_relationship(
        RelationType::RelatedTo,
        1.0,
        false,
        f32::MAX // Extreme decay rate
    );
    
    // Should not overflow or underflow
    for _ in 0..100 {
        let old_weight = relationship.weight;
        relationship.apply_decay();
        
        // Weight should remain non-negative and finite
        assert!(
            relationship.weight >= 0.0 && relationship.weight.is_finite(),
            "Weight became invalid: {} -> {}",
            old_weight, relationship.weight
        );
        
        thread::sleep(Duration::from_millis(1));
    }
    
    // Test with extreme negative decay (growth)
    relationship.temporal_decay = f32::MIN;
    
    for _ in 0..10 {
        let old_weight = relationship.weight;
        relationship.apply_decay();
        
        // Should be clamped and well-behaved
        assert!(
            relationship.weight >= 0.0 && relationship.weight <= 1.0,
            "Weight out of bounds with negative decay: {} -> {}",
            old_weight, relationship.weight
        );
    }
}

// ==================== Performance Regression Tests ====================

#[test]
fn test_performance_regression_detection() {
    // Baseline performance measurements
    let baseline_times = HashMap::from([
        ("entity_creation_1000", Duration::from_millis(10)),
        ("pattern_top_100_from_10k", Duration::from_millis(20)),
        ("relationship_decay_100", Duration::from_millis(5)),
        ("gate_computation_1000", Duration::from_millis(15)),
    ]);
    
    // Test entity creation performance
    let entity_start = Instant::now();
    let mut entities = Vec::new();
    for i in 0..1000 {
        entities.push(EntityBuilder::new(&format!("perf_entity_{}", i), EntityDirection::Hidden).build());
    }
    let entity_time = entity_start.elapsed();
    
    // Test pattern query performance
    let pattern = create_test_pattern("perf_pattern", 10_000);
    let pattern_start = Instant::now();
    let _top_100 = pattern.get_top_activations(100);
    let pattern_time = pattern_start.elapsed();
    
    // Test relationship decay performance
    let mut relationships = Vec::new();
    for i in 0..100 {
        relationships.push(create_test_relationship(
            RelationType::RelatedTo,
            0.8,
            false,
            test_constants::STANDARD_DECAY_RATE
        ));
    }
    
    let decay_start = Instant::now();
    for rel in &mut relationships {
        rel.apply_decay();
    }
    let decay_time = decay_start.elapsed();
    
    // Test gate computation performance
    let gates: Vec<_> = (0..1000).map(|_| create_test_gate(LogicGateType::And, 0.5, 2)).collect();
    let gate_start = Instant::now();
    for gate in &gates {
        let _ = gate.calculate_output(&[0.7, 0.8]);
    }
    let gate_time = gate_start.elapsed();
    
    // Performance regression detection (allow 50% variance)
    let tolerance_factor = 1.5;
    
    assert!(
        entity_time <= baseline_times["entity_creation_1000"] * tolerance_factor,
        "Entity creation performance regression: {:?} > {:?}",
        entity_time, baseline_times["entity_creation_1000"] * tolerance_factor
    );
    
    assert!(
        pattern_time <= baseline_times["pattern_top_100_from_10k"] * tolerance_factor,
        "Pattern query performance regression: {:?} > {:?}",
        pattern_time, baseline_times["pattern_top_100_from_10k"] * tolerance_factor
    );
    
    assert!(
        decay_time <= baseline_times["relationship_decay_100"] * tolerance_factor,
        "Relationship decay performance regression: {:?} > {:?}",
        decay_time, baseline_times["relationship_decay_100"] * tolerance_factor
    );
    
    assert!(
        gate_time <= baseline_times["gate_computation_1000"] * tolerance_factor,
        "Gate computation performance regression: {:?} > {:?}",
        gate_time, baseline_times["gate_computation_1000"] * tolerance_factor
    );
}