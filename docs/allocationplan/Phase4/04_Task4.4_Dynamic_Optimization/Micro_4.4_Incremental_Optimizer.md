# Micro Phase 4.4: Incremental Optimizer

**Estimated Time**: 40 minutes
**Dependencies**: Micro 4.3 Complete (Dead Branch Pruner)
**Objective**: Implement real-time incremental optimization that responds to hierarchy changes with minimal latency and computational overhead

## Task Description

Create an intelligent incremental optimization system that can detect hierarchy changes in real-time and apply targeted optimizations without requiring full hierarchy rebuilds, maintaining optimal structure with sub-10ms response times.

## Deliverables

Create `src/optimization/incremental.rs` with:

1. **IncrementalOptimizer struct**: Real-time optimization engine with change detection
2. **Change tracking**: Monitor hierarchy modifications and trigger targeted optimizations
3. **Micro-optimizations**: Small, fast optimization operations for local changes
4. **Optimization scheduling**: Prioritize and batch optimization operations
5. **Performance monitoring**: Track optimization effectiveness and overhead

## Success Criteria

- [ ] Responds to hierarchy changes in < 10ms
- [ ] Maintains global optimization quality within 5% of full rebuild
- [ ] Processes 100 simultaneous changes efficiently
- [ ] Memory overhead < 10% for change tracking
- [ ] Zero performance degradation during normal operations
- [ ] Optimization quality improves over time with usage patterns

## Implementation Requirements

```rust
pub struct IncrementalOptimizer {
    change_threshold: u32,
    optimization_interval: Duration,
    max_batch_size: usize,
    performance_tracking: bool,
    optimization_history: VecDeque<OptimizationEvent>,
}

#[derive(Debug, Clone)]
pub enum HierarchyChange {
    NodeAdded { 
        node: NodeId, 
        parent: NodeId, 
        properties: HashMap<String, PropertyValue> 
    },
    NodeRemoved { 
        node: NodeId, 
        former_parent: Option<NodeId> 
    },
    NodeMoved { 
        node: NodeId, 
        old_parent: NodeId, 
        new_parent: NodeId 
    },
    PropertyChanged { 
        node: NodeId, 
        property: String, 
        old_value: Option<PropertyValue>, 
        new_value: Option<PropertyValue> 
    },
    SubtreeAdded { 
        root: NodeId, 
        nodes: Vec<NodeId> 
    },
    SubtreeRemoved { 
        former_root: NodeId, 
        nodes: Vec<NodeId> 
    },
}

#[derive(Debug, Clone)]
pub struct OptimizationTarget {
    pub affected_nodes: Vec<NodeId>,
    pub optimization_type: OptimizationType,
    pub priority: OptimizationPriority,
    pub estimated_benefit: f32,
    pub estimated_cost: Duration,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    LocalRebalancing { subtree_root: NodeId },
    PropertyConsolidation { nodes: Vec<NodeId> },
    RedundancyRemoval { node: NodeId },
    StructuralSimplification { area: Vec<NodeId> },
    CacheOptimization { nodes: Vec<NodeId> },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Immediate,  // < 1ms response required
    High,       // < 10ms response
    Medium,     // < 100ms response
    Low,        // Can wait for batch processing
    Background, // Idle time only
}

#[derive(Debug)]
pub struct IncrementalResult {
    pub changes_processed: usize,
    pub optimizations_applied: usize,
    pub total_response_time: Duration,
    pub average_response_time: Duration,
    pub optimization_efficiency: f32,
    pub memory_overhead: usize,
}

impl IncrementalOptimizer {
    pub fn new() -> Self;
    
    pub fn process_change(&mut self, hierarchy: &mut InheritanceHierarchy, change: HierarchyChange) -> IncrementalResult;
    
    pub fn identify_optimization_targets(&self, hierarchy: &InheritanceHierarchy, change: &HierarchyChange) -> Vec<OptimizationTarget>;
    
    pub fn apply_incremental_optimization(&mut self, hierarchy: &mut InheritanceHierarchy, target: &OptimizationTarget) -> bool;
    
    pub fn batch_process_changes(&mut self, hierarchy: &mut InheritanceHierarchy, changes: Vec<HierarchyChange>) -> IncrementalResult;
    
    pub fn schedule_optimization(&mut self, target: OptimizationTarget);
    
    pub fn process_optimization_queue(&mut self, hierarchy: &mut InheritanceHierarchy, time_budget: Duration) -> IncrementalResult;
    
    pub fn estimate_optimization_impact(&self, hierarchy: &InheritanceHierarchy, target: &OptimizationTarget) -> f32;
    
    pub fn validate_optimization_quality(&self, hierarchy: &InheritanceHierarchy) -> f32;
}

#[derive(Debug, Clone)]
struct ChangeTracker {
    recent_changes: VecDeque<TimestampedChange>,
    affected_regions: HashMap<NodeId, ChangeRegion>,
    optimization_candidates: BinaryHeap<OptimizationTarget>,
    performance_metrics: PerformanceTracker,
}

#[derive(Debug, Clone)]
struct TimestampedChange {
    change: HierarchyChange,
    timestamp: Instant,
    processed: bool,
}

#[derive(Debug, Clone)]
struct ChangeRegion {
    center: NodeId,
    affected_nodes: HashSet<NodeId>,
    change_count: u32,
    last_optimized: Instant,
    optimization_benefit: f32,
}

#[derive(Debug)]
struct PerformanceTracker {
    response_times: VecDeque<Duration>,
    optimization_effectiveness: VecDeque<f32>,
    memory_usage: VecDeque<usize>,
    queue_lengths: VecDeque<usize>,
}
```

## Test Requirements

Must pass incremental optimization tests:
```rust
#[test]
fn test_single_node_addition_response() {
    let mut hierarchy = create_balanced_hierarchy(100);
    let mut optimizer = IncrementalOptimizer::new();
    
    let parent = hierarchy.find_random_node();
    let properties = vec![("color", "blue")].into_iter().collect();
    
    let start = Instant::now();
    let change = HierarchyChange::NodeAdded { 
        node: NodeId::new(), 
        parent, 
        properties 
    };
    let result = optimizer.process_change(&mut hierarchy, change);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(10)); // < 10ms response
    assert_eq!(result.changes_processed, 1);
    assert!(result.average_response_time < Duration::from_millis(10));
}

#[test]
fn test_batch_change_processing() {
    let mut hierarchy = create_medium_hierarchy(500);
    let mut optimizer = IncrementalOptimizer::new();
    
    // Create 100 simultaneous changes
    let mut changes = Vec::new();
    for i in 0..100 {
        let parent = hierarchy.find_random_node();
        changes.push(HierarchyChange::NodeAdded {
            node: NodeId::new(),
            parent,
            properties: vec![("id", i.to_string())].into_iter().collect(),
        });
    }
    
    let start = Instant::now();
    let result = optimizer.batch_process_changes(&mut hierarchy, changes);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(500)); // Reasonable batch time
    assert_eq!(result.changes_processed, 100);
    assert!(result.optimization_efficiency > 0.8); // Good efficiency
}

#[test]
fn test_local_rebalancing_trigger() {
    let mut hierarchy = create_balanced_hierarchy(200);
    let mut optimizer = IncrementalOptimizer::new();
    
    // Add many nodes to one branch to create imbalance
    let branch_root = hierarchy.find_random_node();
    let mut current_parent = branch_root;
    
    for i in 0..10 {
        let change = HierarchyChange::NodeAdded {
            node: NodeId::new(),
            parent: current_parent,
            properties: HashMap::new(),
        };
        current_parent = match change {
            HierarchyChange::NodeAdded { node, .. } => node,
            _ => unreachable!(),
        };
        
        let result = optimizer.process_change(&mut hierarchy, change);
        
        // Should trigger local rebalancing after several additions
        if i > 5 {
            assert!(result.optimizations_applied > 0);
        }
    }
}

#[test]
fn test_property_consolidation_detection() {
    let mut hierarchy = create_hierarchy_with_scattered_properties();
    let mut optimizer = IncrementalOptimizer::new();
    
    // Add properties that should trigger consolidation
    let nodes = hierarchy.get_nodes_in_subtree(hierarchy.get_root().unwrap());
    
    for (i, node) in nodes.iter().take(5).enumerate() {
        let change = HierarchyChange::PropertyChanged {
            node: *node,
            property: "common_property".to_string(),
            old_value: None,
            new_value: Some(PropertyValue::String("common_value".to_string())),
        };
        
        let targets = optimizer.identify_optimization_targets(&hierarchy, &change);
        
        // Should identify property consolidation opportunity
        let has_consolidation = targets.iter().any(|t| {
            matches!(t.optimization_type, OptimizationType::PropertyConsolidation { .. })
        });
        
        if i >= 3 { // After multiple nodes have the same property
            assert!(has_consolidation);
        }
    }
}

#[test]
fn test_optimization_queue_prioritization() {
    let mut hierarchy = create_complex_hierarchy();
    let mut optimizer = IncrementalOptimizer::new();
    
    // Schedule optimizations with different priorities
    optimizer.schedule_optimization(OptimizationTarget {
        affected_nodes: vec![NodeId::new()],
        optimization_type: OptimizationType::LocalRebalancing { subtree_root: NodeId::new() },
        priority: OptimizationPriority::Low,
        estimated_benefit: 0.1,
        estimated_cost: Duration::from_millis(50),
    });
    
    optimizer.schedule_optimization(OptimizationTarget {
        affected_nodes: vec![NodeId::new()],
        optimization_type: OptimizationType::RedundancyRemoval { node: NodeId::new() },
        priority: OptimizationPriority::Immediate,
        estimated_benefit: 0.8,
        estimated_cost: Duration::from_millis(5),
    });
    
    optimizer.schedule_optimization(OptimizationTarget {
        affected_nodes: vec![NodeId::new()],
        optimization_type: OptimizationType::StructuralSimplification { area: vec![NodeId::new()] },
        priority: OptimizationPriority::High,
        estimated_benefit: 0.5,
        estimated_cost: Duration::from_millis(20),
    });
    
    // Process with limited time budget
    let result = optimizer.process_optimization_queue(&mut hierarchy, Duration::from_millis(30));
    
    // Should process Immediate and High priority first
    assert!(result.optimizations_applied >= 2);
}

#[test]
fn test_memory_overhead_tracking() {
    let mut hierarchy = create_large_hierarchy(1000);
    let mut optimizer = IncrementalOptimizer::new();
    
    let initial_memory = std::mem::size_of_val(&hierarchy) + std::mem::size_of_val(&optimizer);
    
    // Process many changes to build up tracking data
    for i in 0..500 {
        let change = HierarchyChange::NodeAdded {
            node: NodeId::new(),
            parent: hierarchy.find_random_node(),
            properties: HashMap::new(),
        };
        optimizer.process_change(&mut hierarchy, change);
    }
    
    let final_memory = std::mem::size_of_val(&hierarchy) + std::mem::size_of_val(&optimizer);
    let overhead = final_memory - initial_memory;
    let overhead_percentage = (overhead as f32 / initial_memory as f32) * 100.0;
    
    assert!(overhead_percentage < 10.0); // < 10% memory overhead
}

#[test]
fn test_optimization_quality_preservation() {
    let mut reference_hierarchy = create_complex_hierarchy();
    let mut incremental_hierarchy = reference_hierarchy.clone();
    let mut optimizer = IncrementalOptimizer::new();
    
    // Apply full optimization to reference
    let full_optimizer = create_full_optimizer();
    full_optimizer.optimize_hierarchy(&mut reference_hierarchy);
    let reference_quality = calculate_hierarchy_quality(&reference_hierarchy);
    
    // Apply incremental optimizations
    let changes = generate_random_changes(100);
    for change in changes {
        apply_change_to_hierarchy(&mut incremental_hierarchy, &change);
        optimizer.process_change(&mut incremental_hierarchy, change);
    }
    
    let incremental_quality = calculate_hierarchy_quality(&incremental_hierarchy);
    let quality_difference = (reference_quality - incremental_quality) / reference_quality;
    
    assert!(quality_difference < 0.05); // Within 5% of full optimization
}

#[test]
fn test_real_time_performance_monitoring() {
    let mut hierarchy = create_dynamic_hierarchy();
    let mut optimizer = IncrementalOptimizer::new();
    
    let mut response_times = Vec::new();
    
    // Process changes and monitor performance
    for _ in 0..200 {
        let change = generate_random_change(&hierarchy);
        
        let start = Instant::now();
        let result = optimizer.process_change(&mut hierarchy, change);
        let elapsed = start.elapsed();
        
        response_times.push(elapsed);
        
        // All responses should be fast
        assert!(elapsed < Duration::from_millis(10));
        assert!(result.optimization_efficiency > 0.7);
    }
    
    // Performance should be consistent
    let avg_time: Duration = response_times.iter().sum::<Duration>() / response_times.len() as u32;
    let max_time = response_times.iter().max().unwrap();
    
    assert!(avg_time < Duration::from_millis(5)); // Very fast average
    assert!(*max_time < Duration::from_millis(15)); // No outliers
}

#[test]
fn test_optimization_effectiveness_tracking() {
    let mut hierarchy = create_suboptimal_hierarchy();
    let mut optimizer = IncrementalOptimizer::new();
    
    let initial_quality = optimizer.validate_optimization_quality(&hierarchy);
    
    // Apply many changes that should trigger optimizations
    let changes = generate_optimization_triggering_changes(50);
    for change in changes {
        optimizer.process_change(&mut hierarchy, change);
    }
    
    let final_quality = optimizer.validate_optimization_quality(&hierarchy);
    
    // Quality should improve over time
    assert!(final_quality > initial_quality);
    
    // Optimizer should track effectiveness
    assert!(optimizer.optimization_history.len() > 0);
}
```

## File Location
`src/optimization/incremental.rs`

## Next Micro Phase
After completion, proceed to Micro 4.5: Optimization Metrics