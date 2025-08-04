# Micro Phase 4.6: Optimization Integration Tests

**Estimated Time**: 25 minutes
**Dependencies**: Micro 4.5 Complete (Optimization Metrics)
**Objective**: Implement comprehensive integration tests that validate the complete optimization workflow and cross-component interactions

## Task Description

Create thorough integration tests that verify the entire optimization system works correctly as a unified whole, testing component interactions, workflow orchestration, and end-to-end optimization scenarios.

## Deliverables

Create `tests/integration/task_4_4_optimization.rs` with:

1. **Complete workflow tests**: End-to-end optimization pipeline validation
2. **Component interaction tests**: Verify proper communication between optimization components
3. **Performance integration tests**: Validate optimization performance under realistic loads
4. **Error handling tests**: Test system behavior under failure conditions
5. **Cross-system compatibility tests**: Ensure optimization works with other system components

## Success Criteria

- [ ] 100% test coverage of optimization component interactions
- [ ] All workflow scenarios pass validation
- [ ] Performance tests meet established benchmarks
- [ ] Error conditions handled gracefully
- [ ] Integration with core hierarchy operations validated
- [ ] Resource cleanup verified after optimization operations

## Implementation Requirements

```rust
use crate::core::InheritanceHierarchy;
use crate::optimization::{
    reorganizer::HierarchyReorganizer,
    balancer::HierarchyBalancer,
    pruner::HierarchyPruner,
    incremental::IncrementalOptimizer,
    metrics::OptimizationMetrics,
};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Complete optimization workflow integration tests
#[cfg(test)]
mod optimization_integration_tests {
    use super::*;
    
    /// Test data builders for consistent test hierarchies
    struct TestHierarchyBuilder;
    
    impl TestHierarchyBuilder {
        fn create_complete_test_scenario() -> OptimizationTestScenario {
            // Implementation for creating comprehensive test data
        }
        
        fn create_performance_test_hierarchy(size: usize) -> InheritanceHierarchy {
            // Implementation for creating performance test hierarchies
        }
        
        fn create_problematic_hierarchy() -> InheritanceHierarchy {
            // Implementation for creating hierarchies with known optimization opportunities
        }
    }
    
    /// Test scenario container for comprehensive optimization testing
    struct OptimizationTestScenario {
        hierarchy: InheritanceHierarchy,
        expected_improvements: ExpectedImprovements,
        performance_targets: PerformanceTargets,
        validation_criteria: ValidationCriteria,
    }
    
    struct ExpectedImprovements {
        min_depth_reduction: f32,
        min_node_reduction: f32,
        min_quality_improvement: f32,
        min_memory_savings: f32,
    }
    
    struct PerformanceTargets {
        max_reorganization_time: Duration,
        max_balancing_time: Duration,
        max_pruning_time: Duration,
        max_incremental_response: Duration,
        max_metrics_overhead: Duration,
    }
    
    struct ValidationCriteria {
        semantic_preservation: bool,
        performance_improvement: bool,
        memory_efficiency: bool,
        error_handling: bool,
    }
}
```

## Test Requirements

Must pass complete optimization integration tests:
```rust
#[test]
fn test_complete_optimization_workflow() {
    let scenario = TestHierarchyBuilder::create_complete_test_scenario();
    let mut hierarchy = scenario.hierarchy;
    
    // Record initial state
    let initial_metrics = capture_hierarchy_metrics(&hierarchy);
    let initial_quality = calculate_quality_score(&hierarchy);
    
    // Create optimization components
    let reorganizer = HierarchyReorganizer::new(0.8);
    let balancer = HierarchyBalancer::new(1);
    let pruner = HierarchyPruner::new();
    let mut incremental_optimizer = IncrementalOptimizer::new();
    let mut metrics = OptimizationMetrics::new();
    
    // Step 1: Reorganization
    let start = Instant::now();
    let reorg_result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    let reorg_time = start.elapsed();
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Reorganization, reorg_time, reorg_result.nodes_moved
    ));
    
    // Step 2: Balancing
    let start = Instant::now();
    let balance_result = balancer.balance_hierarchy(&mut hierarchy);
    let balance_time = start.elapsed();
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Balancing, balance_time, balance_result.operations_performed.len()
    ));
    
    // Step 3: Pruning
    let start = Instant::now();
    let prune_result = pruner.prune_hierarchy(&mut hierarchy);
    let prune_time = start.elapsed();
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Pruning, prune_time, prune_result.nodes_removed
    ));
    
    // Validate final state
    let final_metrics = capture_hierarchy_metrics(&hierarchy);
    let final_quality = calculate_quality_score(&hierarchy);
    
    // Verify improvements meet expectations
    assert!(final_metrics.depth < initial_metrics.depth * (1.0 - scenario.expected_improvements.min_depth_reduction));
    assert!(final_metrics.node_count <= initial_metrics.node_count); // Should not increase
    assert!(final_quality > initial_quality * (1.0 + scenario.expected_improvements.min_quality_improvement));
    
    // Verify performance targets
    assert!(reorg_time < scenario.performance_targets.max_reorganization_time);
    assert!(balance_time < scenario.performance_targets.max_balancing_time);
    assert!(prune_time < scenario.performance_targets.max_pruning_time);
    
    // Verify semantic preservation
    validate_semantic_correctness(&hierarchy, &scenario.validation_criteria);
}

#[test]
fn test_incremental_optimization_integration() {
    let mut hierarchy = TestHierarchyBuilder::create_performance_test_hierarchy(1000);
    let mut incremental_optimizer = IncrementalOptimizer::new();
    let mut metrics = OptimizationMetrics::new();
    
    // Apply series of changes with incremental optimization
    let changes = generate_realistic_change_sequence(100);
    let mut response_times = Vec::new();
    
    for change in changes {
        let start = Instant::now();
        
        // Apply change to hierarchy
        apply_hierarchy_change(&mut hierarchy, &change);
        
        // Process with incremental optimizer
        let result = incremental_optimizer.process_change(&mut hierarchy, change.clone());
        
        let elapsed = start.elapsed();
        response_times.push(elapsed);
        
        // Record metrics
        metrics.record_optimization_event(OptimizationEvent {
            timestamp: Instant::now(),
            operation_type: OptimizationType::Incremental,
            execution_time: elapsed,
            nodes_affected: result.changes_processed,
            quality_improvement: result.optimization_efficiency,
            memory_delta: 0, // Simplified for test
            success: true,
            error_details: None,
        });
        
        // Validate response time
        assert!(elapsed < Duration::from_millis(10), 
            "Incremental optimization took too long: {:?}", elapsed);
    }
    
    // Verify consistent performance
    let avg_response_time = response_times.iter().sum::<Duration>() / response_times.len() as u32;
    assert!(avg_response_time < Duration::from_millis(5));
    
    // Verify optimization quality maintained
    let final_quality = metrics.calculate_hierarchy_quality(&hierarchy);
    assert!(final_quality.overall_score > 0.7);
}

#[test]
fn test_component_interaction_correctness() {
    let mut hierarchy = TestHierarchyBuilder::create_problematic_hierarchy();
    
    // Test interaction: Reorganizer -> Balancer
    let reorganizer = HierarchyReorganizer::new(0.75);
    let balancer = HierarchyBalancer::new(1);
    
    // Record state after reorganization
    let reorg_result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    let post_reorg_quality = calculate_quality_score(&hierarchy);
    
    // Apply balancer to reorganized hierarchy
    let balance_result = balancer.balance_hierarchy(&mut hierarchy);
    let post_balance_quality = calculate_quality_score(&hierarchy);
    
    // Balancing should improve upon reorganization
    assert!(post_balance_quality >= post_reorg_quality);
    assert!(balance_result.final_metrics.balance_factor <= 1);
    
    // Test interaction: Balancer -> Pruner
    let pruner = HierarchyPruner::new();
    let prune_result = pruner.prune_hierarchy(&mut hierarchy);
    let post_prune_quality = calculate_quality_score(&hierarchy);
    
    // Pruning should maintain or improve quality
    assert!(post_prune_quality >= post_balance_quality * 0.95); // Allow small degradation for size reduction
    
    // Verify no invalid states created
    validate_hierarchy_integrity(&hierarchy);
}

#[test]
fn test_optimization_under_concurrent_changes() {
    let mut hierarchy = TestHierarchyBuilder::create_performance_test_hierarchy(500);
    let mut incremental_optimizer = IncrementalOptimizer::new();
    
    // Simulate concurrent changes during optimization
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let hierarchy_arc = Arc::new(Mutex::new(hierarchy));
    let optimizer_arc = Arc::new(Mutex::new(incremental_optimizer));
    
    let mut handles = Vec::new();
    
    // Spawn threads for concurrent changes
    for thread_id in 0..4 {
        let hierarchy_clone = Arc::clone(&hierarchy_arc);
        let optimizer_clone = Arc::clone(&optimizer_arc);
        
        let handle = thread::spawn(move || {
            for i in 0..25 {
                let change = generate_thread_specific_change(thread_id, i);
                
                // Apply change with proper locking
                {
                    let mut hierarchy = hierarchy_clone.lock().unwrap();
                    let mut optimizer = optimizer_clone.lock().unwrap();
                    
                    apply_hierarchy_change(&mut *hierarchy, &change);
                    let result = optimizer.process_change(&mut *hierarchy, change);
                    
                    // Verify optimization succeeded
                    assert!(result.changes_processed > 0);
                }
                
                // Small delay to allow interleaving
                thread::sleep(Duration::from_millis(1));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final hierarchy state
    let final_hierarchy = hierarchy_arc.lock().unwrap();
    validate_hierarchy_integrity(&*final_hierarchy);
    
    let final_quality = calculate_quality_score(&*final_hierarchy);
    assert!(final_quality.overall_score > 0.6); // Should maintain reasonable quality
}

#[test]
fn test_error_handling_and_recovery() {
    let mut hierarchy = TestHierarchyBuilder::create_complete_test_scenario().hierarchy;
    let mut metrics = OptimizationMetrics::new();
    
    // Test reorganizer error handling
    let faulty_reorganizer = create_faulty_reorganizer(); // Simulates failures
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        faulty_reorganizer.reorganize_hierarchy(&mut hierarchy)
    }));
    
    // Should handle errors gracefully
    match result {
        Ok(_) => {
            // If successful, verify hierarchy wasn't corrupted
            validate_hierarchy_integrity(&hierarchy);
        }
        Err(_) => {
            // If failed, hierarchy should be unchanged
            let quality = calculate_quality_score(&hierarchy);
            assert!(quality.overall_score > 0.0);
        }
    }
    
    // Test recovery from partial optimization failure
    let initial_state = hierarchy.clone();
    
    // Attempt optimization that might fail partway through
    let reorganizer = HierarchyReorganizer::new(0.9); // High threshold might cause issues
    
    match reorganizer.reorganize_hierarchy(&mut hierarchy) {
        Ok(result) => {
            // Verify successful optimization
            assert!(result.operations_performed.len() > 0);
            validate_hierarchy_integrity(&hierarchy);
        }
        Err(_) => {
            // Verify hierarchy can recover
            hierarchy = initial_state;
            validate_hierarchy_integrity(&hierarchy);
        }
    }
}

#[test]
fn test_memory_management_during_optimization() {
    let mut hierarchy = TestHierarchyBuilder::create_performance_test_hierarchy(2000);
    
    // Measure initial memory usage
    let initial_memory = measure_memory_usage();
    
    // Perform optimization sequence
    let reorganizer = HierarchyReorganizer::new(0.8);
    let balancer = HierarchyBalancer::new(1);
    let pruner = HierarchyPruner::new();
    let mut metrics = OptimizationMetrics::new();
    
    // Track memory during each phase
    let mem_before_reorg = measure_memory_usage();
    reorganizer.reorganize_hierarchy(&mut hierarchy);
    let mem_after_reorg = measure_memory_usage();
    
    let mem_before_balance = measure_memory_usage();
    balancer.balance_hierarchy(&mut hierarchy);
    let mem_after_balance = measure_memory_usage();
    
    let mem_before_prune = measure_memory_usage();
    pruner.prune_hierarchy(&mut hierarchy);
    let mem_after_prune = measure_memory_usage();
    
    // Verify memory is properly managed
    assert!(mem_after_reorg <= mem_before_reorg * 1.1); // Allow 10% increase during operation
    assert!(mem_after_balance <= mem_before_balance * 1.1);
    assert!(mem_after_prune <= mem_before_prune); // Pruning should reduce memory
    
    // Final memory should be reasonable
    let final_memory = measure_memory_usage();
    assert!(final_memory <= initial_memory); // Overall memory should not increase
}

#[test]
fn test_optimization_metrics_integration() {
    let mut hierarchy = TestHierarchyBuilder::create_complete_test_scenario().hierarchy;
    let mut metrics = OptimizationMetrics::new();
    
    // Perform full optimization workflow with metrics tracking
    let reorganizer = HierarchyReorganizer::new(0.8);
    let balancer = HierarchyBalancer::new(1);
    let pruner = HierarchyPruner::new();
    
    // Each optimization should be recorded in metrics
    let reorg_result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Reorganization,
        Duration::from_millis(50),
        reorg_result.operations_performed.len()
    ));
    
    let balance_result = balancer.balance_hierarchy(&mut hierarchy);
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Balancing,
        Duration::from_millis(30),
        balance_result.operations_performed.len()
    ));
    
    let prune_result = pruner.prune_hierarchy(&mut hierarchy);
    metrics.record_optimization_event(create_optimization_event(
        OptimizationType::Pruning,
        Duration::from_millis(20),
        prune_result.nodes_removed
    ));
    
    // Generate comprehensive report
    let report = metrics.generate_optimization_report(Duration::from_secs(300));
    
    // Verify all operations recorded
    assert_eq!(report.total_optimizations, 3);
    assert_eq!(report.successful_optimizations, 3);
    assert!(report.total_quality_improvement > 0.0);
    
    // Verify recommendations generated
    let recommendations = metrics.generate_recommendations(&hierarchy);
    assert!(!recommendations.is_empty());
    
    // Verify dashboard data availability
    let dashboard = metrics.get_real_time_dashboard_data();
    assert!(dashboard.current_quality_score.overall_score > 0.0);
    assert_eq!(dashboard.recent_improvements.len(), 3);
}

#[test]
fn test_cross_system_compatibility() {
    let mut hierarchy = TestHierarchyBuilder::create_performance_test_hierarchy(1000);
    
    // Test compatibility with core hierarchy operations during optimization
    let mut incremental_optimizer = IncrementalOptimizer::new();
    
    // Perform optimization while doing normal hierarchy operations
    let mut changes = Vec::new();
    
    for i in 0..50 {
        // Mix optimization changes with regular operations
        let change = if i % 3 == 0 {
            // Regular hierarchy modification
            HierarchyChange::NodeAdded {
                node: NodeId::new(),
                parent: hierarchy.find_random_node(),
                properties: HashMap::new(),
            }
        } else {
            // Property update
            HierarchyChange::PropertyChanged {
                node: hierarchy.find_random_node(),
                property: "test_prop".to_string(),
                old_value: None,
                new_value: Some(PropertyValue::String(format!("value_{}", i))),
            }
        };
        
        changes.push(change);
    }
    
    // Process all changes with incremental optimization
    let result = incremental_optimizer.batch_process_changes(&mut hierarchy, changes);
    
    // Verify system remained stable
    assert!(result.optimization_efficiency > 0.5);
    validate_hierarchy_integrity(&hierarchy);
    
    // Test querying during optimization
    let query_results = perform_test_queries(&hierarchy);
    validate_query_results(&query_results);
}

// Helper functions for integration tests
fn capture_hierarchy_metrics(hierarchy: &InheritanceHierarchy) -> HierarchyMetrics {
    HierarchyMetrics {
        node_count: hierarchy.node_count(),
        depth: hierarchy.max_depth(),
        average_children: hierarchy.average_children_per_node(),
        memory_usage: hierarchy.estimated_memory_usage(),
    }
}

fn calculate_quality_score(hierarchy: &InheritanceHierarchy) -> QualityScore {
    let metrics = OptimizationMetrics::new();
    let quality = metrics.calculate_hierarchy_quality(hierarchy);
    QualityScore {
        overall: quality.overall_score,
        depth: quality.depth_score,
        balance: quality.balance_score,
        efficiency: quality.inheritance_efficiency,
    }
}

fn validate_semantic_correctness(hierarchy: &InheritanceHierarchy, criteria: &ValidationCriteria) {
    if criteria.semantic_preservation {
        // Verify all inheritance relationships preserved
        for node in hierarchy.all_nodes() {
            for property in node.all_property_names() {
                let value = hierarchy.get_property(node.id, &property);
                assert!(value.is_some(), "Property '{}' lost on node {:?}", property, node.id);
            }
        }
    }
}

fn validate_hierarchy_integrity(hierarchy: &InheritanceHierarchy) {
    // Verify no orphaned nodes
    for node in hierarchy.all_nodes() {
        if node.id != hierarchy.get_root().unwrap() {
            assert!(hierarchy.get_parent(node.id).is_some(), 
                "Node {:?} is orphaned", node.id);
        }
    }
    
    // Verify no cycles
    assert!(!hierarchy.has_cycles(), "Hierarchy contains cycles");
    
    // Verify property inheritance works
    for node in hierarchy.all_nodes() {
        for property in node.all_property_names() {
            hierarchy.get_property(node.id, &property); // Should not panic
        }
    }
}

struct HierarchyMetrics {
    node_count: usize,
    depth: u32,
    average_children: f32,
    memory_usage: usize,
}

struct QualityScore {
    overall: f32,
    depth: f32,
    balance: f32,
    efficiency: f32,
}
```

## File Location
`tests/integration/task_4_4_optimization.rs`

## Next Micro Phase
After completion, proceed to Micro 4.7: Optimization Performance Tests