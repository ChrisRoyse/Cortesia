# Micro Phase 4.1: Hierarchy Reorganizer

**Estimated Time**: 45 minutes
**Dependencies**: Task 4.3 Complete (Property Compression)
**Objective**: Implement intelligent hierarchy restructuring to optimize property inheritance patterns

## Task Description

Create a sophisticated reorganization system that can identify suboptimal hierarchy structures and automatically reorganize them for better inheritance efficiency, reduced depth, and improved cache locality.

## Deliverables

Create `src/optimization/reorganizer.rs` with:

1. **HierarchyReorganizer struct**: Core reorganization engine
2. **Reorganization strategies**: Multiple algorithms for different scenarios
3. **Similarity-based clustering**: Group similar nodes for better structure
4. **Structural analysis**: Identify reorganization opportunities
5. **Safe restructuring**: Maintain semantic correctness during changes

## Success Criteria

- [ ] Reduces average hierarchy depth by > 30%
- [ ] Improves property inheritance efficiency by > 20%
- [ ] Maintains 100% semantic correctness during reorganization
- [ ] Processes 10,000 nodes in < 500ms
- [ ] Identifies all beneficial reorganization opportunities
- [ ] Memory usage remains bounded during restructuring

## Implementation Requirements

```rust
pub struct HierarchyReorganizer {
    similarity_threshold: f32,
    max_depth_reduction: u32,
    clustering_algorithm: ClusteringAlgorithm,
    validation_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum ReorganizationOperation {
    MergeNodes { 
        nodes: Vec<NodeId>, 
        new_parent: NodeId 
    },
    SplitNode { 
        node: NodeId, 
        groups: Vec<Vec<NodeId>> 
    },
    MoveSubtree { 
        subtree_root: NodeId, 
        new_parent: NodeId 
    },
    CreateIntermediateNode { 
        children: Vec<NodeId>, 
        parent: NodeId, 
        properties: HashMap<String, PropertyValue> 
    },
}

#[derive(Debug)]
pub struct ReorganizationResult {
    pub operations_performed: Vec<ReorganizationOperation>,
    pub depth_reduction: u32,
    pub nodes_moved: usize,
    pub nodes_merged: usize,
    pub intermediate_nodes_created: usize,
    pub inheritance_efficiency_improvement: f32,
    pub execution_time: Duration,
}

impl HierarchyReorganizer {
    pub fn new(similarity_threshold: f32) -> Self;
    
    pub fn reorganize_hierarchy(&self, hierarchy: &mut InheritanceHierarchy) -> ReorganizationResult;
    
    pub fn find_reorganization_opportunities(&self, hierarchy: &InheritanceHierarchy) -> Vec<ReorganizationOperation>;
    
    pub fn calculate_node_similarity(&self, hierarchy: &InheritanceHierarchy, node1: NodeId, node2: NodeId) -> f32;
    
    pub fn estimate_reorganization_benefit(&self, hierarchy: &InheritanceHierarchy, operation: &ReorganizationOperation) -> f32;
    
    pub fn validate_reorganization(&self, hierarchy: &InheritanceHierarchy, operation: &ReorganizationOperation) -> bool;
}

#[derive(Debug, Clone)]
enum ClusteringAlgorithm {
    PropertySimilarity,
    StructuralSimilarity,
    HybridSimilarity,
    InheritancePattern,
}
```

## Test Requirements

Must pass hierarchy reorganization tests:
```rust
#[test]
fn test_depth_reduction() {
    let mut hierarchy = create_deep_linear_hierarchy(20); // Very deep chain
    let reorganizer = HierarchyReorganizer::new(0.8);
    
    let initial_depth = hierarchy.max_depth();
    assert_eq!(initial_depth, 19); // 20 nodes in chain = depth 19
    
    let result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    
    let final_depth = hierarchy.max_depth();
    assert!(final_depth < initial_depth * 0.7); // >30% reduction
    assert!(result.depth_reduction >= 6);
    
    // Verify all nodes still accessible
    assert_eq!(hierarchy.node_count(), 20);
}

#[test]
fn test_similarity_based_clustering() {
    let mut hierarchy = create_scattered_animal_hierarchy();
    let reorganizer = HierarchyReorganizer::new(0.75);
    
    // Initially scattered: Dog breeds under different parents
    // Should cluster similar animals together
    
    let result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    
    // All dog breeds should now be close together
    let golden_retriever = hierarchy.get_node_by_name("Golden Retriever").unwrap();
    let labrador = hierarchy.get_node_by_name("Labrador").unwrap();
    let beagle = hierarchy.get_node_by_name("Beagle").unwrap();
    
    // They should share a common ancestor at most 2 levels up
    let gr_ancestors = hierarchy.get_ancestors(golden_retriever, 3);
    let lab_ancestors = hierarchy.get_ancestors(labrador, 3);
    let beagle_ancestors = hierarchy.get_ancestors(beagle, 3);
    
    let common_ancestors: Vec<_> = gr_ancestors.iter()
        .filter(|&a| lab_ancestors.contains(a) && beagle_ancestors.contains(a))
        .collect();
    
    assert!(!common_ancestors.is_empty());
    assert!(result.nodes_moved > 0);
}

#[test]
fn test_intermediate_node_creation() {
    let mut hierarchy = create_flat_hierarchy(); // Many children under one parent
    let reorganizer = HierarchyReorganizer::new(0.8);
    
    let root = hierarchy.get_node_by_name("Root").unwrap();
    let initial_children = hierarchy.get_children(root).len();
    assert!(initial_children > 10); // Flat structure
    
    let result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    
    let final_children = hierarchy.get_children(root).len();
    assert!(final_children < initial_children); // Should create intermediate nodes
    assert!(result.intermediate_nodes_created > 0);
    
    // Verify all original nodes still reachable
    for i in 0..initial_children {
        let node_name = format!("Child_{}", i);
        assert!(hierarchy.get_node_by_name(&node_name).is_some());
    }
}

#[test]
fn test_semantic_preservation() {
    let original_hierarchy = create_complex_hierarchy();
    let mut test_hierarchy = original_hierarchy.clone();
    let reorganizer = HierarchyReorganizer::new(0.7);
    
    // Record all property values before reorganization
    let mut original_properties = HashMap::new();
    for node in original_hierarchy.all_nodes() {
        for property in node.all_property_names() {
            let value = original_hierarchy.get_property(node.id, &property);
            original_properties.insert((node.id, property), value);
        }
    }
    
    // Reorganize
    reorganizer.reorganize_hierarchy(&mut test_hierarchy);
    
    // Verify all properties still resolve to same values
    for ((node_id, property), expected_value) in original_properties {
        let actual_value = test_hierarchy.get_property(node_id, &property);
        assert_eq!(actual_value, expected_value, 
            "Property '{}' on node {:?} changed from {:?} to {:?}", 
            property, node_id, expected_value, actual_value);
    }
}

#[test]
fn test_reorganization_performance() {
    let mut hierarchy = create_large_hierarchy(10000);
    let reorganizer = HierarchyReorganizer::new(0.75);
    
    let start = Instant::now();
    let result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(500)); // < 500ms for 10k nodes
    assert!(result.operations_performed.len() > 0); // Should find some optimizations
    assert_eq!(hierarchy.node_count(), 10000); // No nodes lost
}

#[test]
fn test_inheritance_efficiency_improvement() {
    let mut hierarchy = create_inefficient_hierarchy();
    let reorganizer = HierarchyReorganizer::new(0.8);
    
    // Measure initial inheritance efficiency
    let initial_efficiency = calculate_inheritance_efficiency(&hierarchy);
    
    let result = reorganizer.reorganize_hierarchy(&mut hierarchy);
    
    // Measure final inheritance efficiency
    let final_efficiency = calculate_inheritance_efficiency(&hierarchy);
    
    assert!(final_efficiency > initial_efficiency * 1.2); // >20% improvement
    assert!(result.inheritance_efficiency_improvement >= 0.2);
}
```

## File Location
`src/optimization/reorganizer.rs`

## Next Micro Phase
After completion, proceed to Micro 4.2: Tree Balancer