# Micro Phase 1.6: Multiple Inheritance DAG Support

**Estimated Time**: 40 minutes
**Dependencies**: Micro 1.5 (Property Cache System)
**Objective**: Extend hierarchy to support multiple inheritance using Directed Acyclic Graphs

## Task Description

Enhance the hierarchy system to properly support multiple inheritance scenarios using DAG structures while maintaining performance and preventing cycles.

## Deliverables

Create `src/hierarchy/dag.rs` with:

1. **DAG validation**: Ensure no cycles when adding multiple parents
2. **C3 linearization**: Method resolution order for multiple inheritance
3. **Diamond problem resolution**: Handle common ancestor scenarios
4. **Topological sorting**: Order nodes for property resolution
5. **Conflict detection**: Identify and resolve property conflicts

## Success Criteria

- [ ] Prevents cycle creation when adding multiple parents
- [ ] C3 linearization produces consistent method resolution order
- [ ] Diamond inheritance resolves deterministically
- [ ] Topological sort is stable and reproducible
- [ ] Conflict resolution follows Python/C++ rules
- [ ] Performance remains O(log n) for property lookup

## Implementation Requirements

```rust
pub struct DAGManager {
    topology_cache: DashMap<NodeId, Vec<NodeId>>,
    conflict_resolver: ConflictResolver,
    cycle_detector: CycleDetector,
}

impl DAGManager {
    pub fn new() -> Self;
    
    pub fn add_parent(&self, hierarchy: &mut InheritanceHierarchy, child: NodeId, parent: NodeId) -> Result<(), DAGError>;
    
    pub fn compute_mro(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Vec<NodeId>;
    
    pub fn detect_conflicts(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Vec<PropertyConflict>;
    
    pub fn validate_dag(&self, hierarchy: &InheritanceHierarchy) -> DAGValidationResult;
}

#[derive(Debug)]
pub struct PropertyConflict {
    pub property_name: String,
    pub conflicting_sources: Vec<(NodeId, PropertyValue)>,
    pub resolution_strategy: ConflictResolution,
}

#[derive(Debug, Clone)]
pub enum ConflictResolution {
    FirstParent,      // Use first parent's value
    LastParent,       // Use last parent's value  
    MostSpecific,     // Use most derived class value
    Explicit(NodeId), // Explicitly specified source
}
```

## Test Requirements

Must pass multiple inheritance tests:
```rust
#[test]
fn test_diamond_inheritance() {
    let mut hierarchy = InheritanceHierarchy::new();
    let dag = DAGManager::new();
    
    // Create diamond: A -> B, A -> C, B -> D, C -> D
    let a = hierarchy.create_node("A");
    let b = hierarchy.create_child("B", a);
    let c = hierarchy.create_child("C", a);
    let d = hierarchy.create_child("D", b);
    
    // Add second parent to D
    dag.add_parent(&mut hierarchy, d, c).unwrap();
    
    // Compute MRO
    let mro = dag.compute_mro(&hierarchy, d);
    
    // Should follow C3 linearization
    assert_eq!(mro, vec![d, b, c, a]);
}

#[test]
fn test_cycle_prevention() {
    let mut hierarchy = InheritanceHierarchy::new();
    let dag = DAGManager::new();
    
    let a = hierarchy.create_node("A");
    let b = hierarchy.create_child("B", a);
    let c = hierarchy.create_child("C", b);
    
    // Attempt to create cycle: A -> B -> C -> A
    let result = dag.add_parent(&mut hierarchy, a, c);
    
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), DAGError::CycleDetected(_)));
}

#[test]
fn test_conflict_detection() {
    let mut hierarchy = create_conflicting_hierarchy();
    let dag = DAGManager::new();
    
    let node = hierarchy.get_node_by_name("ConflictNode").unwrap();
    let conflicts = dag.detect_conflicts(&hierarchy, node);
    
    assert!(!conflicts.is_empty());
    
    for conflict in conflicts {
        assert!(conflict.conflicting_sources.len() >= 2);
        assert!(!conflict.property_name.is_empty());
    }
}
```

## File Location
`src/hierarchy/dag.rs`

## Next Micro Phase
After completion, proceed to Micro 1.7: Integration Tests