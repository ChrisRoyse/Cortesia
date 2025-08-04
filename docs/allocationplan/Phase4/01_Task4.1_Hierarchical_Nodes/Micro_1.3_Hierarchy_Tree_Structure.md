# Micro Phase 1.3: Hierarchy Tree Structure

**Estimated Time**: 45 minutes
**Dependencies**: Micro 1.1 (Basic Node Structure), Micro 1.2 (Property Value System)
**Objective**: Implement the main hierarchy container that manages all nodes

## Task Description

Create the `InheritanceHierarchy` struct that will serve as the main container for managing inheritance nodes, relationships, and lookups.

## Deliverables

Create `src/hierarchy/tree.rs` with:

1. **InheritanceHierarchy struct**: Main container with concurrent access
2. **Node management**: Create, retrieve, update, delete operations
3. **Relationship management**: Parent-child relationship tracking
4. **Name indexing**: Fast lookup by node name
5. **Basic statistics**: Node count, depth calculation

## Success Criteria

- [ ] Thread-safe concurrent access using DashMap
- [ ] O(1) node lookup by ID
- [ ] O(1) node lookup by name
- [ ] Parent-child relationships properly maintained
- [ ] Depth calculation works for arbitrary hierarchies
- [ ] Memory usage scales linearly with node count

## Implementation Requirements

```rust
pub struct InheritanceHierarchy {
    nodes: DashMap<NodeId, InheritanceNode>,
    name_index: DashMap<String, NodeId>,
    next_id: AtomicU64,
    creation_time: Instant,
}

impl InheritanceHierarchy {
    pub fn new() -> Self;
    pub fn create_node(&self, name: &str) -> NodeId;
    pub fn create_child(&self, name: &str, parent: NodeId) -> NodeId;
    pub fn get_node(&self, id: NodeId) -> Option<DashMapRef<NodeId, InheritanceNode>>;
    pub fn get_node_by_name(&self, name: &str) -> Option<NodeId>;
    pub fn add_parent(&self, child: NodeId, parent: NodeId) -> Result<(), HierarchyError>;
    pub fn remove_parent(&self, child: NodeId, parent: NodeId) -> Result<(), HierarchyError>;
    pub fn node_count(&self) -> usize;
    pub fn max_depth(&self) -> u32;
    pub fn get_depth(&self, node: NodeId) -> Option<u32>;
}
```

## Test Requirements

Must pass hierarchy management tests:
```rust
#[test]
fn test_node_creation_and_lookup() {
    let hierarchy = InheritanceHierarchy::new();
    
    let id = hierarchy.create_node("TestNode");
    assert!(hierarchy.get_node(id).is_some());
    assert_eq!(hierarchy.get_node_by_name("TestNode"), Some(id));
    assert_eq!(hierarchy.node_count(), 1);
}

#[test]
fn test_parent_child_relationships() {
    let hierarchy = InheritanceHierarchy::new();
    
    let parent = hierarchy.create_node("Parent");
    let child = hierarchy.create_child("Child", parent);
    
    let parent_node = hierarchy.get_node(parent).unwrap();
    let child_node = hierarchy.get_node(child).unwrap();
    
    assert!(parent_node.children.contains(&child));
    assert!(child_node.parents.contains(&parent));
}

#[test]
fn test_depth_calculation() {
    let hierarchy = InheritanceHierarchy::new();
    
    let root = hierarchy.create_node("Root");
    let child = hierarchy.create_child("Child", root);
    let grandchild = hierarchy.create_child("Grandchild", child);
    
    assert_eq!(hierarchy.get_depth(root), Some(0));
    assert_eq!(hierarchy.get_depth(child), Some(1));
    assert_eq!(hierarchy.get_depth(grandchild), Some(2));
    assert_eq!(hierarchy.max_depth(), 2);
}
```

## File Location
`src/hierarchy/tree.rs`

## Next Micro Phase
After completion, proceed to Micro 1.4: Property Resolution Engine