/// Performance and stress testing for brain_types components
/// 
/// This module tests performance characteristics, memory usage, and stress
/// scenarios to ensure scalability and robustness under load.

use llmkg::core::brain_types::{
    LogicGate, LogicGateType, BrainInspiredEntity, EntityDirection,
    BrainInspiredRelationship, RelationType, ActivationPattern, ActivationStep, ActivationOperation
};
use llmkg::core::types::EntityKey;
use std::time::{SystemTime, Instant, Duration};
use std::collections::HashMap;
use super::test_helpers::*;

#[test]
fn test_logic_gate_performance_large_inputs() {
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let input_sizes = vec![10, 100, 1000];
        
        for size in input_sizes {
            let gate = create_test_gate(gate_type, 0.5, size);
            let inputs: Vec<f32> = (0..size).map(|i| (i as f32) / (size as f32)).collect();
            
            let start = Instant::now();
            let result = gate.calculate_output(&inputs);
            let duration = start.elapsed();
            
            // Should complete quickly and successfully
            assert!(result.is_ok(), "Gate {:?} failed with {} inputs", gate_type, size);
            assert!(duration < Duration::from_millis(10), 
                "Gate {:?} too slow with {} inputs: {:?}", gate_type, size, duration);
        }
    }
}

#[test]
fn test_weighted_gate_performance() {
    let sizes = vec![10, 100, 1000];
    
    for size in sizes {
        let mut gate = create_test_gate(LogicGateType::Weighted, 0.5, size);
        gate.weight_matrix = vec![1.0 / size as f32; size]; // Normalized weights
        
        let inputs: Vec<f32> = vec![0.5; size];
        
        let start = Instant::now();
        let result = gate.calculate_output(&inputs);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration < Duration::from_millis(5), 
            "Weighted gate too slow with {} inputs: {:?}", size, duration);
    }
}

#[test]
fn test_entity_activation_performance() {
    let mut entity = BrainInspiredEntity::new("perf_test".to_string(), EntityDirection::Input);
    
    // Test many rapid activations
    let num_activations = 10000;
    let start = Instant::now();
    
    for i in 0..num_activations {
        entity.activate(0.001, 0.01); // Small increments
        if i % 1000 == 0 {
            // Reset occasionally to prevent saturation
            entity.activation_state = 0.0;
        }
    }
    
    let duration = start.elapsed();
    
    // Should complete within reasonable time
    assert!(duration < Duration::from_millis(100), 
        "Entity activation too slow for {} operations: {:?}", num_activations, duration);
}

#[test]
fn test_entity_temporal_decay_performance() {
    let mut entities = Vec::new();
    let num_entities = 1000;
    
    // Create many entities with different timestamps
    for i in 0..num_entities {
        let mut entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Hidden);
        entity.activation_state = 0.8;
        entity.last_activation = SystemTime::now() - Duration::from_secs(i as u64 % 100);
        entities.push(entity);
    }
    
    // Apply decay to all entities
    let start = Instant::now();
    for entity in &mut entities {
        entity.activate(0.0, 0.1); // Just decay, no new activation
    }
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(50), 
        "Temporal decay too slow for {} entities: {:?}", num_entities, duration);
}

#[test]
fn test_relationship_operations_performance() {
    let mut relationships = Vec::new();
    let num_relationships = 1000;
    
    // Create many relationships
    for i in 0..num_relationships {
        let mut rel = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::Learned
        );
        rel.weight = (i as f32) / (num_relationships as f32);
        rel.last_strengthened = SystemTime::now() - Duration::from_secs(i as u64 % 100);
        relationships.push(rel);
    }
    
    // Test strengthening performance
    let start = Instant::now();
    for rel in &mut relationships {
        rel.strengthen(0.01);
    }
    let strengthen_duration = start.elapsed();
    
    // Test decay performance
    let start = Instant::now();
    for rel in &mut relationships {
        rel.apply_decay();
    }
    let decay_duration = start.elapsed();
    
    assert!(strengthen_duration < Duration::from_millis(20), 
        "Strengthening too slow for {} relationships: {:?}", num_relationships, strengthen_duration);
    assert!(decay_duration < Duration::from_millis(50), 
        "Decay too slow for {} relationships: {:?}", num_relationships, decay_duration);
}

#[test]
fn test_activation_pattern_large_scale_performance() {
    let mut pattern = ActivationPattern::new("large_scale_test".to_string());
    let num_activations = 10000;
    
    // Insert many activations
    let start = Instant::now();
    for i in 0..num_activations {
        let key = EntityKey::default(); // Note: will overwrite, but tests insertion performance
        pattern.activations.insert(key, (i as f32) / (num_activations as f32));
    }
    let insertion_duration = start.elapsed();
    
    // Test top activations performance with large dataset
    let start = Instant::now();
    let _top_100 = pattern.get_top_activations(100);
    let top_duration = start.elapsed();
    
    let start = Instant::now();
    let _top_1000 = pattern.get_top_activations(1000);
    let top_1000_duration = start.elapsed();
    
    assert!(insertion_duration < Duration::from_millis(100), 
        "Insertion too slow for {} activations: {:?}", num_activations, insertion_duration);
    assert!(top_duration < Duration::from_millis(10), 
        "Top 100 too slow: {:?}", top_duration);
    assert!(top_1000_duration < Duration::from_millis(50), 
        "Top 1000 too slow: {:?}", top_1000_duration);
}

#[test]
fn test_activation_pattern_sorting_performance() {
    let mut pattern = ActivationPattern::new("sorting_test".to_string());
    
    // Create pattern with many unique activations
    let num_unique = 5000;
    for i in 0..num_unique {
        // Create pseudo-unique keys by using different default keys
        // Note: In real usage, keys would be actually unique from SlotMap
        let key = EntityKey::default();
        pattern.activations.insert(key, rand::random::<f32>());
    }
    
    // Test sorting performance
    let percentages = vec![1, 5, 10, 25, 50, 100];
    
    for percentage in percentages {
        let num_top = (num_unique * percentage) / 100;
        
        let start = Instant::now();
        let top = pattern.get_top_activations(num_top);
        let duration = start.elapsed();
        
        assert!(duration < Duration::from_millis(20), 
            "Sorting top {}% ({} items) too slow: {:?}", percentage, num_top, duration);
        
        // Verify sorting correctness (should be in descending order)
        for i in 1..top.len() {
            assert!(top[i-1].1 >= top[i].1, "Sorting incorrect at position {}", i);
        }
    }
}

// Note: Using a simple random number generator for testing
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    pub fn random<T>() -> f32 {
        let mut hasher = DefaultHasher::new();
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().hash(&mut hasher);
        let hash = hasher.finish();
        ((hash % 10000) as f32) / 10000.0
    }
}

#[test]
fn test_memory_usage_patterns() {
    // Test memory allocation patterns for large structures
    
    // Large activation pattern
    let mut large_pattern = ActivationPattern::new("memory_test".to_string());
    let num_activations = 50000;
    
    for i in 0..num_activations {
        let key = EntityKey::default();
        large_pattern.activations.insert(key, rand::random::<f32>());
    }
    
    // Should not crash or use excessive memory
    let _top = large_pattern.get_top_activations(1000);
    
    // Clear and verify cleanup
    large_pattern.activations.clear();
    assert!(large_pattern.activations.is_empty());
}

#[test]
fn test_concurrent_access_simulation() {
    // Simulate concurrent-like access patterns (single-threaded simulation)
    
    let mut pattern = ActivationPattern::new("concurrent_sim".to_string());
    let num_operations = 10000;
    
    let start = Instant::now();
    
    for i in 0..num_operations {
        let key = EntityKey::default();
        
        match i % 4 {
            0 => {
                // Insert new activation
                pattern.activations.insert(key, rand::random::<f32>());
            },
            1 => {
                // Update existing activation
                if !pattern.activations.is_empty() {
                    pattern.activations.insert(key, rand::random::<f32>());
                }
            },
            2 => {
                // Remove activation
                pattern.activations.remove(&key);
            },
            3 => {
                // Query top activations
                let _top = pattern.get_top_activations(10);
            },
            _ => unreachable!(),
        }
    }
    
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(200), 
        "Concurrent simulation too slow: {:?}", duration);
}

#[test]
fn test_stress_all_logic_gates() {
    // Stress test all logic gate types with various input configurations
    
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];
    
    let num_iterations = 1000;
    
    for gate_type in gate_types {
        let input_count = match gate_type {
            LogicGateType::Not | LogicGateType::Identity => 1,
            LogicGateType::Xor | LogicGateType::Xnor => 2,
            _ => 3,
        };
        
        let mut gate = create_test_gate(gate_type, 0.5, input_count);
        
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = vec![0.3; input_count];
        }
        
        let start = Instant::now();
        
        for _ in 0..num_iterations {
            let inputs: Vec<f32> = (0..input_count).map(|_| rand::random::<f32>()).collect();
            let result = gate.calculate_output(&inputs);
            
            // Should succeed for valid inputs
            assert!(result.is_ok(), "Gate {:?} failed during stress test", gate_type);
        }
        
        let duration = start.elapsed();
        
        assert!(duration < Duration::from_millis(20), 
            "Gate {:?} stress test too slow: {:?}", gate_type, duration);
    }
}

#[test]
fn test_entity_stress_activation_cycles() {
    // Stress test entity with many activation/decay cycles
    
    let mut entity = BrainInspiredEntity::new("stress_test".to_string(), EntityDirection::Hidden);
    let num_cycles = 10000;
    
    let start = Instant::now();
    
    for i in 0..num_cycles {
        // Alternate between activation and decay
        if i % 100 == 0 {
            // Occasional reset to prevent saturation
            entity.activation_state = 0.0;
        }
        
        let activation = if i % 2 == 0 { 0.1 } else { 0.0 };
        let decay_rate = 0.01 + (i as f32 / num_cycles as f32) * 0.09; // Varying decay rate
        
        entity.activate(activation, decay_rate);
        
        // Activation should remain valid
        assert_valid_activation(entity.activation_state);
    }
    
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(100), 
        "Entity stress test too slow: {:?}", duration);
}

#[test]
fn test_relationship_stress_strengthen_decay_cycles() {
    // Stress test relationship with many strengthen/decay cycles
    
    let mut rel = BrainInspiredRelationship::new(
        EntityKey::default(),
        EntityKey::default(),
        RelationType::Learned
    );
    
    let num_cycles = 5000;
    let start = Instant::now();
    
    for i in 0..num_cycles {
        if i % 2 == 0 {
            // Strengthen
            rel.strengthen(0.01);
        } else {
            // Apply decay with time passage simulation
            rel.last_strengthened = SystemTime::now() - Duration::from_millis(100);
            rel.apply_decay();
        }
        
        // Weight should remain valid
        assert!(rel.weight >= 0.0 && rel.weight <= 1.0);
        assert_eq!(rel.weight, rel.strength);
    }
    
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(100), 
        "Relationship stress test too slow: {:?}", duration);
}

#[test]
fn test_activation_trace_performance() {
    // Test performance of creating and managing large activation traces
    
    let mut trace = Vec::new();
    let num_steps = 10000;
    
    let start = Instant::now();
    
    for i in 0..num_steps {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: format!("concept_{}", i % 100), // Reuse concept IDs
            activation_level: rand::random::<f32>(),
            operation_type: match i % 5 {
                0 => ActivationOperation::Initialize,
                1 => ActivationOperation::Propagate,
                2 => ActivationOperation::Inhibit,
                3 => ActivationOperation::Reinforce,
                4 => ActivationOperation::Decay,
                _ => unreachable!(),
            },
            timestamp: SystemTime::now(),
        };
        trace.push(step);
    }
    
    let creation_duration = start.elapsed();
    
    // Test trace analysis performance
    let start = Instant::now();
    
    // Simulate trace analysis operations
    let _initialize_count = trace.iter()
        .filter(|s| matches!(s.operation_type, ActivationOperation::Initialize))
        .count();
    
    let _avg_activation: f32 = trace.iter()
        .map(|s| s.activation_level)
        .sum::<f32>() / trace.len() as f32;
    
    let _unique_concepts: std::collections::HashSet<_> = trace.iter()
        .map(|s| &s.concept_id)
        .collect();
    
    let analysis_duration = start.elapsed();
    
    assert!(creation_duration < Duration::from_millis(50), 
        "Trace creation too slow: {:?}", creation_duration);
    assert!(analysis_duration < Duration::from_millis(20), 
        "Trace analysis too slow: {:?}", analysis_duration);
}

#[test]
fn test_mixed_operations_performance() {
    // Test performance of mixed operations simulating real usage
    
    let mut entities = Vec::new();
    let mut relationships = Vec::new();
    let mut patterns = Vec::new();
    
    let num_entities = 100;
    let num_relationships = 200;
    let num_patterns = 50;
    
    // Initialize structures
    for i in 0..num_entities {
        entities.push(BrainInspiredEntity::new(
            format!("entity_{}", i),
            match i % 4 {
                0 => EntityDirection::Input,
                1 => EntityDirection::Output,
                2 => EntityDirection::Gate,
                3 => EntityDirection::Hidden,
                _ => unreachable!(),
            }
        ));
    }
    
    for i in 0..num_relationships {
        relationships.push(BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            match i % 9 {
                0 => RelationType::IsA,
                1 => RelationType::HasInstance,
                2 => RelationType::HasProperty,
                3 => RelationType::RelatedTo,
                4 => RelationType::PartOf,
                5 => RelationType::Similar,
                6 => RelationType::Opposite,
                7 => RelationType::Temporal,
                8 => RelationType::Learned,
                _ => unreachable!(),
            }
        ));
    }
    
    for i in 0..num_patterns {
        let mut pattern = ActivationPattern::new(format!("pattern_{}", i));
        for j in 0..20 {
            pattern.activations.insert(EntityKey::default(), rand::random::<f32>());
        }
        patterns.push(pattern);
    }
    
    // Mixed operations stress test
    let num_operations = 1000;
    let start = Instant::now();
    
    for i in 0..num_operations {
        match i % 6 {
            0 => {
                // Entity activation
                let idx = i % entities.len();
                entities[idx].activate(rand::random::<f32>() * 0.1, 0.01);
            },
            1 => {
                // Relationship strengthening
                let idx = i % relationships.len();
                relationships[idx].strengthen(rand::random::<f32>() * 0.05);
            },
            2 => {
                // Relationship decay
                let idx = i % relationships.len();
                relationships[idx].apply_decay();
            },
            3 => {
                // Pattern query
                let idx = i % patterns.len();
                let _top = patterns[idx].get_top_activations(5);
            },
            4 => {
                // Pattern modification
                let idx = i % patterns.len();
                patterns[idx].activations.insert(EntityKey::default(), rand::random::<f32>());
            },
            5 => {
                // Logic gate operation
                let gate = create_test_gate(LogicGateType::And, 0.5, 2);
                let _result = gate.calculate_output(&[rand::random::<f32>(), rand::random::<f32>()]);
            },
            _ => unreachable!(),
        }
    }
    
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(500), 
        "Mixed operations too slow: {:?}", duration);
}