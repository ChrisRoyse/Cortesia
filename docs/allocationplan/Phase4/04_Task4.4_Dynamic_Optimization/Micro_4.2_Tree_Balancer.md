# Micro Phase 4.2: Tree Balancer

**Estimated Time**: 40 minutes
**Dependencies**: Micro 4.1 Complete (Hierarchy Reorganizer)
**Objective**: Implement AVL-style balancing algorithms for inheritance hierarchies to optimize depth and access patterns

## Task Description

Create a sophisticated tree balancing system that applies classic tree balancing algorithms to inheritance hierarchies, reducing depth imbalances and improving traversal performance while maintaining semantic correctness.

## Deliverables

Create `src/optimization/balancer.rs` with:

1. **HierarchyBalancer struct**: Core balancing engine with rotation operations
2. **Balance metrics**: Calculate and track hierarchy balance factors
3. **Rotation operations**: Left/right rotations adapted for inheritance trees
4. **Rebalancing triggers**: Detect when rebalancing is needed
5. **Incremental balancing**: Maintain balance after individual node operations

## Success Criteria

- [ ] Reduces maximum hierarchy depth by > 30%
- [ ] Maintains balance factor < 2 for all subtrees
- [ ] Processes rebalancing in < 100ms for 1,000 node trees
- [ ] Preserves all inheritance relationships during balancing
- [ ] Improves average access time by > 25%
- [ ] Memory overhead for balancing operations < 5%

## Implementation Requirements

```rust
pub struct HierarchyBalancer {
    max_imbalance: i32,
    rebalance_threshold: f32,
    preserve_semantics: bool,
    performance_tracking: bool,
}

#[derive(Debug, Clone)]
pub enum BalanceOperation {
    LeftRotation { 
        pivot: NodeId, 
        new_root: NodeId 
    },
    RightRotation { 
        pivot: NodeId, 
        new_root: NodeId 
    },
    LeftRightRotation { 
        pivot: NodeId, 
        intermediate: NodeId, 
        new_root: NodeId 
    },
    RightLeftRotation { 
        pivot: NodeId, 
        intermediate: NodeId, 
        new_root: NodeId 
    },
    SubtreeRebalance { 
        root: NodeId, 
        operations: Vec<BalanceOperation> 
    },
}

#[derive(Debug)]
pub struct BalanceMetrics {
    pub max_depth: u32,
    pub average_depth: f32,
    pub balance_factor: i32,
    pub imbalanced_nodes: usize,
    pub depth_variance: f32,
}

#[derive(Debug)]
pub struct BalancingResult {
    pub operations_performed: Vec<BalanceOperation>,
    pub initial_metrics: BalanceMetrics,
    pub final_metrics: BalanceMetrics,
    pub depth_reduction: u32,
    pub access_time_improvement: f32,
    pub execution_time: Duration,
}

impl HierarchyBalancer {
    pub fn new(max_imbalance: i32) -> Self;
    
    pub fn balance_hierarchy(&self, hierarchy: &mut InheritanceHierarchy) -> BalancingResult;
    
    pub fn calculate_balance_metrics(&self, hierarchy: &InheritanceHierarchy) -> BalanceMetrics;
    
    pub fn needs_rebalancing(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> bool;
    
    pub fn perform_rotation(&self, hierarchy: &mut InheritanceHierarchy, operation: &BalanceOperation) -> bool;
    
    pub fn calculate_balance_factor(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> i32;
    
    pub fn find_imbalanced_nodes(&self, hierarchy: &InheritanceHierarchy) -> Vec<NodeId>;
    
    pub fn estimate_balancing_benefit(&self, hierarchy: &InheritanceHierarchy, operation: &BalanceOperation) -> f32;
}

#[derive(Debug, Clone)]
struct BalanceNode {
    id: NodeId,
    height: u32,
    balance_factor: i32,
    left_child: Option<NodeId>,
    right_child: Option<NodeId>,
    parent: Option<NodeId>,
}
```

## Test Requirements

Must pass hierarchy balancing tests:
```rust
#[test]
fn test_deep_chain_balancing() {
    let mut hierarchy = create_linear_chain_hierarchy(15); // Deep linear chain
    let balancer = HierarchyBalancer::new(1);
    
    let initial_metrics = balancer.calculate_balance_metrics(&hierarchy);
    assert!(initial_metrics.max_depth >= 14);
    assert!(initial_metrics.balance_factor > 10);
    
    let result = balancer.balance_hierarchy(&mut hierarchy);
    
    let final_metrics = balancer.calculate_balance_metrics(&hierarchy);
    assert!(final_metrics.max_depth < initial_metrics.max_depth * 0.7); // >30% reduction
    assert!(final_metrics.balance_factor <= 1); // Well balanced
    assert!(result.depth_reduction >= 5);
    
    // Verify all nodes still present and accessible
    assert_eq!(hierarchy.node_count(), 15);
}

#[test]
fn test_left_right_rotation() {
    let mut hierarchy = create_left_heavy_hierarchy();
    let balancer = HierarchyBalancer::new(1);
    
    let root = hierarchy.get_root().unwrap();
    let initial_balance = balancer.calculate_balance_factor(&hierarchy, root);
    assert!(initial_balance < -1); // Left heavy
    
    let result = balancer.balance_hierarchy(&mut hierarchy);
    
    let final_balance = balancer.calculate_balance_factor(&hierarchy, root);
    assert!(final_balance.abs() <= 1); // Balanced
    
    // Verify rotation operations were performed
    let has_rotation = result.operations_performed.iter().any(|op| {
        matches!(op, BalanceOperation::LeftRotation { .. } | BalanceOperation::LeftRightRotation { .. })
    });
    assert!(has_rotation);
}

#[test]
fn test_semantic_preservation_during_balancing() {
    let original_hierarchy = create_animal_hierarchy();
    let mut test_hierarchy = original_hierarchy.clone();
    let balancer = HierarchyBalancer::new(1);
    
    // Record all inheritance relationships
    let mut original_relationships = HashMap::new();
    for node in original_hierarchy.all_nodes() {
        let ancestors = original_hierarchy.get_all_ancestors(node.id);
        original_relationships.insert(node.id, ancestors);
    }
    
    // Balance hierarchy
    balancer.balance_hierarchy(&mut test_hierarchy);
    
    // Verify all inheritance relationships preserved
    for (node_id, expected_ancestors) in original_relationships {
        let actual_ancestors = test_hierarchy.get_all_ancestors(node_id);
        assert_eq!(actual_ancestors.len(), expected_ancestors.len(),
            "Node {:?} lost/gained ancestors during balancing", node_id);
        
        // All original ancestors should still be ancestors (path may have changed)
        for ancestor in expected_ancestors {
            assert!(test_hierarchy.is_ancestor(ancestor, node_id),
                "Lost inheritance relationship: {:?} -> {:?}", ancestor, node_id);
        }
    }
}

#[test]
fn test_incremental_balancing_performance() {
    let mut hierarchy = create_balanced_hierarchy(500);
    let balancer = HierarchyBalancer::new(1);
    
    // Add nodes that will create imbalance
    for i in 0..100 {
        let leaf = hierarchy.find_random_leaf();
        hierarchy.add_child(leaf, format!("NewNode_{}", i), HashMap::new());
    }
    
    let start = Instant::now();
    let result = balancer.balance_hierarchy(&mut hierarchy);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(100)); // < 100ms for incremental balance
    
    let final_metrics = balancer.calculate_balance_metrics(&hierarchy);
    assert!(final_metrics.balance_factor <= 1);
    assert_eq!(hierarchy.node_count(), 600); // All nodes preserved
}

#[test]
fn test_balance_factor_calculation() {
    let mut hierarchy = InheritanceHierarchy::new();
    let root = hierarchy.add_root("Root", HashMap::new());
    let left = hierarchy.add_child(root, "Left", HashMap::new());
    let right = hierarchy.add_child(root, "Right", HashMap::new());
    let left_left = hierarchy.add_child(left, "LeftLeft", HashMap::new());
    let left_right = hierarchy.add_child(left, "LeftRight", HashMap::new());
    
    let balancer = HierarchyBalancer::new(1);
    
    // Root should have balance factor of 0 (left subtree height 2, right subtree height 1)
    let root_balance = balancer.calculate_balance_factor(&hierarchy, root);
    assert_eq!(root_balance, 1); // Left heavier by 1
    
    // Left node should be balanced
    let left_balance = balancer.calculate_balance_factor(&hierarchy, left);
    assert_eq!(left_balance, 0); // Both children at same level
    
    // Leaf nodes should have balance factor 0
    let leaf_balance = balancer.calculate_balance_factor(&hierarchy, left_left);
    assert_eq!(leaf_balance, 0);
}

#[test]
fn test_access_time_improvement() {
    let mut hierarchy = create_worst_case_hierarchy(); // Heavily unbalanced
    let balancer = HierarchyBalancer::new(1);
    
    // Measure initial access times
    let initial_access_time = measure_average_access_time(&hierarchy);
    
    let result = balancer.balance_hierarchy(&mut hierarchy);
    
    // Measure final access times
    let final_access_time = measure_average_access_time(&hierarchy);
    
    let improvement = (initial_access_time - final_access_time) / initial_access_time;
    assert!(improvement > 0.25); // >25% improvement
    assert!(result.access_time_improvement >= 0.25);
}

#[test]
fn test_large_hierarchy_balancing() {
    let mut hierarchy = create_random_hierarchy(2000);
    let balancer = HierarchyBalancer::new(2);
    
    let initial_metrics = balancer.calculate_balance_metrics(&hierarchy);
    
    let start = Instant::now();
    let result = balancer.balance_hierarchy(&mut hierarchy);
    let elapsed = start.elapsed();
    
    let final_metrics = balancer.calculate_balance_metrics(&hierarchy);
    
    assert!(elapsed < Duration::from_millis(1000)); // Reasonable performance
    assert!(final_metrics.max_depth < initial_metrics.max_depth);
    assert!(final_metrics.balance_factor <= 2);
    assert_eq!(hierarchy.node_count(), 2000);
}
```

## File Location
`src/optimization/balancer.rs`

## Next Micro Phase
After completion, proceed to Micro 4.3: Dead Branch Pruner