# Micro Phase 4.3: Dead Branch Pruner

**Estimated Time**: 35 minutes
**Dependencies**: Micro 4.2 Complete (Tree Balancer)
**Objective**: Implement intelligent pruning system to identify and safely remove unused, empty, or redundant nodes from inheritance hierarchies

## Task Description

Create a sophisticated pruning system that can identify nodes that provide no semantic value (empty property sets, no unique inheritance contributions, unused intermediate nodes) and safely remove them while preserving all meaningful inheritance relationships.

## Deliverables

Create `src/optimization/pruner.rs` with:

1. **HierarchyPruner struct**: Core pruning engine with safety validation
2. **Dead branch detection**: Identify nodes that can be safely removed
3. **Redundancy analysis**: Find nodes that duplicate parent functionality
4. **Safe removal operations**: Remove nodes while preserving inheritance chains
5. **Orphan prevention**: Ensure child nodes are properly reparented

## Success Criteria

- [ ] Identifies 100% of safely removable nodes without false positives
- [ ] Reduces total node count by > 15% on redundant hierarchies
- [ ] Maintains all meaningful inheritance relationships
- [ ] Processes 5,000 nodes in < 200ms
- [ ] Zero orphaned nodes after pruning operations
- [ ] Memory usage reduction proportional to nodes removed

## Implementation Requirements

```rust
pub struct HierarchyPruner {
    preserve_intermediate_nodes: bool,
    min_property_contribution: u32,
    safety_validation: bool,
    dry_run_mode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PruningReason {
    EmptyNode { 
        node: NodeId, 
        no_unique_properties: bool 
    },
    RedundantNode { 
        node: NodeId, 
        equivalent_parent: NodeId 
    },
    UnusedIntermediate { 
        node: NodeId, 
        can_bypass: bool 
    },
    DuplicateSubtree { 
        nodes: Vec<NodeId>, 
        original: NodeId 
    },
    DeadLeaf { 
        node: NodeId, 
        no_references: bool 
    },
}

#[derive(Debug, Clone)]
pub struct PruningOperation {
    pub reason: PruningReason,
    pub nodes_to_remove: Vec<NodeId>,
    pub reparenting_operations: Vec<(NodeId, NodeId)>, // (child, new_parent)
    pub property_merging: Vec<(NodeId, HashMap<String, PropertyValue>)>,
    pub estimated_benefit: f32,
}

#[derive(Debug)]
pub struct PruningResult {
    pub operations_performed: Vec<PruningOperation>,
    pub nodes_removed: usize,
    pub properties_merged: usize,
    pub reparenting_operations: usize,
    pub memory_savings: usize,
    pub hierarchy_size_reduction: f32,
    pub execution_time: Duration,
}

impl HierarchyPruner {
    pub fn new() -> Self;
    
    pub fn prune_hierarchy(&self, hierarchy: &mut InheritanceHierarchy) -> PruningResult;
    
    pub fn find_prunable_nodes(&self, hierarchy: &InheritanceHierarchy) -> Vec<PruningOperation>;
    
    pub fn is_node_empty(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> bool;
    
    pub fn is_node_redundant(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Option<NodeId>;
    
    pub fn can_bypass_node(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> bool;
    
    pub fn find_duplicate_subtrees(&self, hierarchy: &InheritanceHierarchy) -> Vec<(Vec<NodeId>, NodeId)>;
    
    pub fn validate_pruning_safety(&self, hierarchy: &InheritanceHierarchy, operation: &PruningOperation) -> bool;
    
    pub fn execute_pruning_operation(&self, hierarchy: &mut InheritanceHierarchy, operation: &PruningOperation) -> bool;
}

#[derive(Debug, Clone)]
struct NodeAnalysis {
    id: NodeId,
    unique_properties: usize,
    inheritance_contribution: f32,
    reference_count: usize,
    children_count: usize,
    can_be_bypassed: bool,
    equivalent_nodes: Vec<NodeId>,
}
```

## Test Requirements

Must pass hierarchy pruning tests:
```rust
#[test]
fn test_empty_node_removal() {
    let mut hierarchy = create_hierarchy_with_empty_nodes();
    let pruner = HierarchyPruner::new();
    
    // Add some empty intermediate nodes
    let root = hierarchy.get_root().unwrap();
    let empty1 = hierarchy.add_child(root, "Empty1", HashMap::new()); // No properties
    let empty2 = hierarchy.add_child(empty1, "Empty2", HashMap::new()); // No properties
    let useful = hierarchy.add_child(empty2, "Useful", vec![("color", "red")].into_iter().collect());
    
    let initial_count = hierarchy.node_count();
    
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    // Empty nodes should be removed, useful node should be reparented to root
    assert_eq!(result.nodes_removed, 2);
    assert!(result.reparenting_operations > 0);
    
    // Useful node should now be child of root
    let useful_parent = hierarchy.get_parent(useful).unwrap();
    assert_eq!(useful_parent, root);
    
    // Properties should still be accessible
    assert_eq!(hierarchy.get_property(useful, "color").unwrap(), PropertyValue::String("red".to_string()));
}

#[test]
fn test_redundant_node_identification() {
    let mut hierarchy = InheritanceHierarchy::new();
    let root = hierarchy.add_root("Root", vec![("base", "value")].into_iter().collect());
    let parent = hierarchy.add_child(root, "Parent", vec![("color", "blue")].into_iter().collect());
    let redundant = hierarchy.add_child(parent, "Redundant", vec![("color", "blue")].into_iter().collect()); // Same as parent
    let child = hierarchy.add_child(redundant, "Child", vec![("size", "large")].into_iter().collect());
    
    let pruner = HierarchyPruner::new();
    
    let prunable = pruner.find_prunable_nodes(&hierarchy);
    
    // Should identify redundant node
    let redundant_operation = prunable.iter().find(|op| {
        matches!(op.reason, PruningReason::RedundantNode { node, .. } if node == redundant)
    });
    assert!(redundant_operation.is_some());
    
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    // Redundant node removed, child reparented to parent
    assert_eq!(result.nodes_removed, 1);
    let child_parent = hierarchy.get_parent(child).unwrap();
    assert_eq!(child_parent, parent);
    
    // Child should still inherit color from parent
    assert_eq!(hierarchy.get_property(child, "color").unwrap(), PropertyValue::String("blue".to_string()));
}

#[test]
fn test_intermediate_node_bypassing() {
    let mut hierarchy = InheritanceHierarchy::new();
    let root = hierarchy.add_root("Root", HashMap::new());
    let intermediate = hierarchy.add_child(root, "Intermediate", HashMap::new()); // No unique properties
    let child1 = hierarchy.add_child(intermediate, "Child1", HashMap::new());
    let child2 = hierarchy.add_child(intermediate, "Child2", HashMap::new());
    
    let pruner = HierarchyPruner::new();
    
    let can_bypass = pruner.can_bypass_node(&hierarchy, intermediate);
    assert!(can_bypass);
    
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    // Intermediate node removed, children reparented to root
    assert_eq!(result.nodes_removed, 1);
    assert_eq!(result.reparenting_operations, 2);
    
    assert_eq!(hierarchy.get_parent(child1).unwrap(), root);
    assert_eq!(hierarchy.get_parent(child2).unwrap(), root);
}

#[test]
fn test_duplicate_subtree_removal() {
    let mut hierarchy = create_hierarchy_with_duplicates();
    let pruner = HierarchyPruner::new();
    
    // Create duplicate subtrees
    let root = hierarchy.get_root().unwrap();
    
    // Original subtree
    let original = hierarchy.add_child(root, "Original", HashMap::new());
    let orig_child1 = hierarchy.add_child(original, "Child1", vec![("type", "A")].into_iter().collect());
    let orig_child2 = hierarchy.add_child(original, "Child2", vec![("type", "B")].into_iter().collect());
    
    // Duplicate subtree (identical structure and properties)
    let duplicate = hierarchy.add_child(root, "Duplicate", HashMap::new());
    let dup_child1 = hierarchy.add_child(duplicate, "Child1", vec![("type", "A")].into_iter().collect());
    let dup_child2 = hierarchy.add_child(duplicate, "Child2", vec![("type", "B")].into_iter().collect());
    
    let duplicates = pruner.find_duplicate_subtrees(&hierarchy);
    assert!(!duplicates.is_empty());
    
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    // Duplicate subtree should be removed
    assert!(result.nodes_removed >= 3); // duplicate + its children
    
    // Original subtree should remain
    assert!(hierarchy.node_exists(original));
    assert!(hierarchy.node_exists(orig_child1));
    assert!(hierarchy.node_exists(orig_child2));
}

#[test]
fn test_safety_validation() {
    let mut hierarchy = create_complex_hierarchy();
    let pruner = HierarchyPruner::new();
    
    // Find all prunable operations
    let operations = pruner.find_prunable_nodes(&hierarchy);
    
    // All operations should pass safety validation
    for operation in &operations {
        assert!(pruner.validate_pruning_safety(&hierarchy, operation),
            "Unsafe pruning operation proposed: {:?}", operation.reason);
    }
    
    // Apply operations and verify no orphans
    let original_nodes: HashSet<_> = hierarchy.all_nodes().iter().map(|n| n.id).collect();
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    let remaining_nodes: HashSet<_> = hierarchy.all_nodes().iter().map(|n| n.id).collect();
    let removed_nodes: HashSet<_> = original_nodes.difference(&remaining_nodes).cloned().collect();
    
    // Verify no orphaned nodes
    for node in hierarchy.all_nodes() {
        if node.id != hierarchy.get_root().unwrap() {
            let parent = hierarchy.get_parent(node.id);
            assert!(parent.is_some(), "Node {:?} was orphaned after pruning", node.id);
            assert!(!removed_nodes.contains(&parent.unwrap()), 
                "Node {:?} parent was removed but node wasn't reparented", node.id);
        }
    }
}

#[test]
fn test_property_preservation() {
    let original_hierarchy = create_hierarchy_with_complex_properties();
    let mut test_hierarchy = original_hierarchy.clone();
    let pruner = HierarchyPruner::new();
    
    // Record all property resolutions before pruning
    let mut original_properties = HashMap::new();
    for node in original_hierarchy.all_nodes() {
        for property in node.all_property_names() {
            let value = original_hierarchy.get_property(node.id, &property);
            original_properties.insert((node.id, property), value);
        }
    }
    
    let result = pruner.prune_hierarchy(&mut test_hierarchy);
    
    // Verify all remaining nodes still resolve properties correctly
    for ((node_id, property), expected_value) in original_properties {
        if test_hierarchy.node_exists(node_id) {
            let actual_value = test_hierarchy.get_property(node_id, &property);
            assert_eq!(actual_value, expected_value,
                "Property '{}' on node {:?} changed after pruning", property, node_id);
        }
    }
}

#[test]
fn test_pruning_performance() {
    let mut hierarchy = create_large_sparse_hierarchy(5000); // Many empty/redundant nodes
    let pruner = HierarchyPruner::new();
    
    let start = Instant::now();
    let result = pruner.prune_hierarchy(&mut hierarchy);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(200)); // < 200ms for 5k nodes
    assert!(result.nodes_removed > 0); // Should find some nodes to remove
    assert!(result.hierarchy_size_reduction > 0.05); // At least 5% reduction
}

#[test]
fn test_memory_usage_reduction() {
    let mut hierarchy = create_bloated_hierarchy(); // Many unnecessary nodes
    let pruner = HierarchyPruner::new();
    
    let initial_memory = hierarchy.estimated_memory_usage();
    
    let result = pruner.prune_hierarchy(&mut hierarchy);
    
    let final_memory = hierarchy.estimated_memory_usage();
    let memory_reduction = initial_memory - final_memory;
    
    assert!(memory_reduction > 0);
    assert_eq!(result.memory_savings, memory_reduction);
    assert!(result.hierarchy_size_reduction >= 0.15); // >15% reduction
}
```

## File Location
`src/optimization/pruner.rs`

## Next Micro Phase
After completion, proceed to Micro 4.4: Incremental Optimizer