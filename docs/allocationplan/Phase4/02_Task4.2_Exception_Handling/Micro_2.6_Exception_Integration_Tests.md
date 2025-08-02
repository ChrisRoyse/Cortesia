# Micro Phase 2.6: Exception Integration Tests

**Estimated Time**: 25 minutes
**Dependencies**: Micro 2.5 (Exception Storage Optimization)
**Objective**: Create comprehensive integration tests for the complete exception handling workflow

## Task Description

Develop comprehensive integration tests that validate the entire exception handling system working together. These tests verify end-to-end workflows, cross-component interactions, and real-world usage scenarios.

The tests ensure that all exception components (storage, detection, application, learning, optimization) work seamlessly together and handle edge cases properly.

## Deliverables

Create `tests/integration/task_4_2_exceptions.rs` with:

1. **Full workflow tests**: Complete exception lifecycle from detection to application
2. **Cross-component integration**: Verify components work together correctly
3. **Performance integration**: Test system performance under realistic loads
4. **Error handling**: Validate proper error propagation and recovery
5. **Concurrency tests**: Ensure thread safety across all components

## Success Criteria

- [ ] All integration tests pass consistently (100% success rate)
- [ ] Tests cover >95% of exception system code paths
- [ ] Performance tests validate sub-millisecond response times
- [ ] Concurrency tests handle 100+ concurrent operations
- [ ] Error scenarios are properly handled and recoverable
- [ ] Tests can run in isolation and in any order

## Implementation Requirements

```rust
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::test;

use crate::exceptions::{
    ExceptionStore, ExceptionDetector, ExceptionHandler, PatternLearner, StorageOptimizer
};
use crate::core::{InheritanceNode, NodeId, PropertyValue};

pub struct IntegratedExceptionSystem {
    store: Arc<ExceptionStore>,
    detector: ExceptionDetector,
    handler: ExceptionHandler,
    pattern_learner: PatternLearner,
    storage_optimizer: StorageOptimizer,
}

impl IntegratedExceptionSystem {
    pub fn new() -> Self;
    
    pub fn process_node_with_exceptions(
        &mut self,
        node: &InheritanceNode,
        inherited_properties: &HashMap<String, PropertyValue>
    ) -> HashMap<String, PropertyValue>;
    
    pub fn learn_from_user_feedback(
        &mut self,
        node: NodeId,
        property: &str,
        feedback: UserFeedback
    );
    
    pub fn optimize_system(&mut self) -> SystemOptimizationResults;
}

#[derive(Debug)]
pub struct SystemOptimizationResults {
    pub detection_accuracy_improvement: f32,
    pub storage_efficiency_gain: f32,
    pub average_response_time_improvement: Duration,
    pub memory_usage_reduction: usize,
}

// Test utilities
pub fn create_test_inheritance_hierarchy() -> Vec<InheritanceNode>;
pub fn create_realistic_property_conflicts() -> Vec<(NodeId, String, PropertyValue, PropertyValue)>;
pub fn simulate_user_feedback_session() -> Vec<(NodeId, String, UserFeedback)>;
```

## Test Requirements

Must pass comprehensive integration tests:
```rust
#[tokio::test]
async fn test_complete_exception_workflow() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Create test hierarchy: Animal -> Bird -> Penguin
    let animal = InheritanceNode::new(NodeId(1), "Animal");
    animal.local_properties.insert("can_move".to_string(), PropertyValue::Boolean(true));
    
    let mut bird = InheritanceNode::new(NodeId(2), "Bird");
    bird.parent_ids.push(NodeId(1));
    bird.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(true));
    bird.local_properties.insert("has_wings".to_string(), PropertyValue::Boolean(true));
    
    let mut penguin = InheritanceNode::new(NodeId(3), "Penguin");
    penguin.parent_ids.push(NodeId(2));
    penguin.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(false));
    penguin.local_properties.insert("habitat".to_string(), PropertyValue::String("antarctica".to_string()));
    
    // Simulate inherited properties (what penguin should inherit from bird)
    let mut inherited = HashMap::new();
    inherited.insert("can_move".to_string(), PropertyValue::Boolean(true));
    inherited.insert("can_fly".to_string(), PropertyValue::Boolean(true));
    inherited.insert("has_wings".to_string(), PropertyValue::Boolean(true));
    
    // Process penguin node - should detect can_fly exception
    let resolved_properties = system.process_node_with_exceptions(&penguin, &inherited);
    
    // Verify exception was detected and applied
    assert_eq!(resolved_properties.get("can_fly"), Some(&PropertyValue::Boolean(false)));
    assert_eq!(resolved_properties.get("has_wings"), Some(&PropertyValue::Boolean(true))); // No exception
    assert_eq!(resolved_properties.get("can_move"), Some(&PropertyValue::Boolean(true))); // Inherited
    
    // Verify exception was stored
    let stored_exception = system.store.get_exception(NodeId(3), "can_fly");
    assert!(stored_exception.is_some());
    
    let exception = stored_exception.unwrap();
    assert_eq!(exception.inherited_value, PropertyValue::Boolean(true));
    assert_eq!(exception.actual_value, PropertyValue::Boolean(false));
    assert_eq!(exception.source, ExceptionSource::Detected);
    assert!(exception.confidence > 0.8);
}

#[test]
fn test_pattern_learning_integration() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Train system with multiple flightless birds
    let flightless_birds = vec![
        ("Penguin", NodeId(10)),
        ("Ostrich", NodeId(11)),
        ("Emu", NodeId(12)),
        ("Cassowary", NodeId(13)),
        ("Kiwi", NodeId(14)),
    ];
    
    for (name, node_id) in &flightless_birds {
        let mut bird = InheritanceNode::new(*node_id, name);
        bird.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(false));
        
        let mut inherited = HashMap::new();
        inherited.insert("can_fly".to_string(), PropertyValue::Boolean(true));
        
        // Process each bird - should learn pattern
        system.process_node_with_exceptions(&bird, &inherited);
    }
    
    // Test on new flightless bird - should have high confidence
    let mut dodo = InheritanceNode::new(NodeId(15), "Dodo");
    dodo.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(false));
    
    let mut inherited = HashMap::new();
    inherited.insert("can_fly".to_string(), PropertyValue::Boolean(true));
    
    let resolved = system.process_node_with_exceptions(&dodo, &inherited);
    
    assert_eq!(resolved.get("can_fly"), Some(&PropertyValue::Boolean(false)));
    
    // Check that pattern learner has high confidence
    let exception = system.store.get_exception(NodeId(15), "can_fly").unwrap();
    assert!(exception.confidence > 0.9); // Should be very confident due to learned pattern
}

#[test]
fn test_storage_optimization_integration() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Create many similar exceptions to test compression
    for i in 0..1000 {
        let mut node = InheritanceNode::new(NodeId(i), &format!("TestNode_{}", i));
        node.local_properties.insert("test_property".to_string(), 
                                   PropertyValue::String(format!("exception_value_{}", i)));
        
        let mut inherited = HashMap::new();
        inherited.insert("test_property".to_string(), 
                        PropertyValue::String("default_value".to_string()));
        
        system.process_node_with_exceptions(&node, &inherited);
    }
    
    // Run optimization
    let optimization_results = system.optimize_system();
    
    // Should achieve significant compression due to similar exception patterns
    assert!(optimization_results.storage_efficiency_gain > 0.5); // >50% efficiency gain
    assert!(optimization_results.memory_usage_reduction > 0);
    
    // Verify all exceptions still retrievable after optimization
    for i in 0..1000 {
        let exception = system.store.get_exception(NodeId(i), "test_property");
        assert!(exception.is_some());
        assert_eq!(exception.unwrap().actual_value, 
                  PropertyValue::String(format!("exception_value_{}", i)));
    }
}

#[tokio::test]
async fn test_concurrent_exception_processing() {
    let system = Arc::new(IntegratedExceptionSystem::new());
    let num_threads = 10;
    let operations_per_thread = 100;
    
    // Spawn concurrent tasks
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let system_clone = Arc::clone(&system);
        
        let handle = tokio::spawn(async move {
            for i in 0..operations_per_thread {
                let node_id = NodeId(thread_id * operations_per_thread + i);
                let mut node = InheritanceNode::new(node_id, &format!("Node_{}_{}", thread_id, i));
                
                // Add some variety in exceptions
                match i % 3 {
                    0 => {
                        node.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(false));
                        let mut inherited = HashMap::new();
                        inherited.insert("can_fly".to_string(), PropertyValue::Boolean(true));
                        
                        // Note: This would need to be made async or use different pattern
                        // since process_node_with_exceptions is not async
                        // For now, just test the thread-safe components directly
                        
                        system_clone.store.add_exception(
                            node_id, 
                            "can_fly".to_string(),
                            Exception {
                                inherited_value: PropertyValue::Boolean(true),
                                actual_value: PropertyValue::Boolean(false),
                                reason: "Concurrent test".to_string(),
                                source: ExceptionSource::Detected,
                                created_at: Instant::now(),
                                confidence: 0.8,
                            }
                        );
                    },
                    1 => {
                        // Test retrieval
                        let _ = system_clone.store.get_exception(node_id, "can_fly");
                    },
                    2 => {
                        // Test statistics
                        let _ = system_clone.store.get_stats();
                    },
                    _ => {}
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.expect("Task failed");
    }
    
    // Verify system integrity after concurrent operations
    let stats = system.store.get_stats();
    let expected_exceptions = num_threads * operations_per_thread / 3; // Only 1/3 are store operations
    assert!(stats.total_exceptions.load(std::sync::atomic::Ordering::Relaxed) > 0);
    
    // System should still be responsive
    let test_exception = system.store.get_exception(NodeId(0), "can_fly");
    assert!(test_exception.is_some());
}

#[test]
fn test_error_handling_and_recovery() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Test invalid node handling
    let invalid_node = InheritanceNode::new(NodeId(u64::MAX), "InvalidNode");
    let empty_inherited = HashMap::new();
    
    // Should handle gracefully without panicking
    let result = system.process_node_with_exceptions(&invalid_node, &empty_inherited);
    assert!(result.is_empty() || !result.is_empty()); // Just shouldn't panic
    
    // Test malformed property values
    let mut node = InheritanceNode::new(NodeId(1), "TestNode");
    node.local_properties.insert("weird_property".to_string(), 
                                PropertyValue::String("".to_string())); // Empty string
    
    let mut inherited = HashMap::new();
    inherited.insert("weird_property".to_string(), 
                    PropertyValue::Boolean(true)); // Type mismatch
    
    let result = system.process_node_with_exceptions(&node, &inherited);
    
    // Should handle type mismatches gracefully
    assert!(result.contains_key("weird_property"));
    
    // Test system recovery after errors
    let mut normal_node = InheritanceNode::new(NodeId(2), "NormalNode");
    normal_node.local_properties.insert("normal_prop".to_string(), PropertyValue::Boolean(false));
    
    let mut normal_inherited = HashMap::new();
    normal_inherited.insert("normal_prop".to_string(), PropertyValue::Boolean(true));
    
    let normal_result = system.process_node_with_exceptions(&normal_node, &normal_inherited);
    assert_eq!(normal_result.get("normal_prop"), Some(&PropertyValue::Boolean(false)));
}

#[test]
fn test_performance_under_load() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Create realistic load scenario
    let num_nodes = 1000;
    let properties_per_node = 5;
    
    let start_time = Instant::now();
    
    for node_idx in 0..num_nodes {
        let mut node = InheritanceNode::new(NodeId(node_idx), &format!("LoadTestNode_{}", node_idx));
        let mut inherited = HashMap::new();
        
        for prop_idx in 0..properties_per_node {
            let prop_name = format!("prop_{}", prop_idx);
            
            // Create some exceptions (20% chance)
            if node_idx % 5 == 0 {
                node.local_properties.insert(prop_name.clone(), 
                                           PropertyValue::String(format!("exception_{}", node_idx)));
                inherited.insert(prop_name.clone(), 
                               PropertyValue::String("inherited_value".to_string()));
            } else {
                inherited.insert(prop_name.clone(), 
                               PropertyValue::String("normal_value".to_string()));
            }
        }
        
        system.process_node_with_exceptions(&node, &inherited);
    }
    
    let total_time = start_time.elapsed();
    let avg_time_per_node = total_time / num_nodes as u32;
    
    // Should process each node in reasonable time
    assert!(avg_time_per_node < Duration::from_millis(1)); // <1ms per node
    
    // Verify all exceptions were processed
    let stats = system.store.get_stats();
    let expected_exceptions = num_nodes / 5 * properties_per_node; // 20% exception rate
    assert!(stats.total_exceptions.load(std::sync::atomic::Ordering::Relaxed) >= expected_exceptions / 2);
    
    println!("Processed {} nodes with {} exceptions in {:?}", 
             num_nodes, 
             stats.total_exceptions.load(std::sync::atomic::Ordering::Relaxed),
             total_time);
}

#[test]
fn test_system_state_consistency() {
    let mut system = IntegratedExceptionSystem::new();
    
    // Perform various operations
    let operations = vec![
        ("Node1", "prop1", PropertyValue::Boolean(false), PropertyValue::Boolean(true)),
        ("Node2", "prop1", PropertyValue::String("special".to_string()), PropertyValue::String("normal".to_string())),
        ("Node3", "prop2", PropertyValue::Boolean(false), PropertyValue::Boolean(true)),
        ("Node1", "prop2", PropertyValue::String("override".to_string()), PropertyValue::String("base".to_string())),
    ];
    
    for (node_name, prop_name, actual_val, inherited_val) in operations {
        let node_id = NodeId(node_name.len() as u64); // Simple ID generation
        let mut node = InheritanceNode::new(node_id, node_name);
        node.local_properties.insert(prop_name.to_string(), actual_val.clone());
        
        let mut inherited = HashMap::new();
        inherited.insert(prop_name.to_string(), inherited_val);
        
        system.process_node_with_exceptions(&node, &inherited);
    }
    
    // Verify system consistency
    let store_stats = system.store.get_stats();
    
    // All stored exceptions should be retrievable
    for (node_name, prop_name, actual_val, _) in [
        ("Node1", "prop1", PropertyValue::Boolean(false)),
        ("Node2", "prop1", PropertyValue::String("special".to_string())),
        ("Node3", "prop2", PropertyValue::Boolean(false)),
        ("Node1", "prop2", PropertyValue::String("override".to_string())),
    ] {
        let node_id = NodeId(node_name.len() as u64);
        let exception = system.store.get_exception(node_id, prop_name);
        assert!(exception.is_some());
        assert_eq!(exception.unwrap().actual_value, actual_val);
    }
    
    // Cross-component consistency checks
    let node1_exceptions = system.store.get_node_exceptions(NodeId(5)); // "Node1".len() = 5
    assert_eq!(node1_exceptions.len(), 2); // Should have 2 exceptions for Node1
    
    println!("System consistency verified with {} total exceptions", 
             store_stats.total_exceptions.load(std::sync::atomic::Ordering::Relaxed));
}
```

## File Location
`tests/integration/task_4_2_exceptions.rs`

## Next Micro Phase
After completion, proceed to Micro 2.7: Exception Performance Validation