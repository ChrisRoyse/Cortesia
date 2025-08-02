# AI Prompt: Micro Phase 1.3 - Hierarchy Tree Structure

You are tasked with implementing the main hierarchy container that manages all nodes, relationships, and lookups. Your goal is to create `src/hierarchy/tree.rs` with a thread-safe InheritanceHierarchy struct that serves as the central coordinator for the inheritance system.

## Your Task
Implement the `InheritanceHierarchy` struct that will manage nodes, relationships, name indexing, and hierarchy statistics with high-performance concurrent access patterns.

## Specific Requirements
1. Create `src/hierarchy/tree.rs` with InheritanceHierarchy struct using concurrent data structures
2. Implement thread-safe node management (create, retrieve, update, delete)
3. Add efficient relationship management for parent-child connections
4. Implement fast name-based indexing for O(1) lookups
5. Add depth calculation and hierarchy statistics
6. Ensure memory usage scales linearly with node count
7. Handle error cases and prevent cycles in relationships

## Expected Code Structure
You must implement these exact signatures:

```rust
use dashmap::{DashMap, mapref::one::Ref as DashMapRef};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crate::hierarchy::node::{NodeId, InheritanceNode};

#[derive(Debug)]
pub enum HierarchyError {
    NodeNotFound(NodeId),
    NameAlreadyExists(String),
    CycleDetected,
    InvalidRelationship,
}

impl std::fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Implement error display
    }
}

impl std::error::Error for HierarchyError {}

pub struct InheritanceHierarchy {
    nodes: DashMap<NodeId, InheritanceNode>,
    name_index: DashMap<String, NodeId>,
    next_id: AtomicU64,
    creation_time: Instant,
}

impl InheritanceHierarchy {
    pub fn new() -> Self {
        // Initialize with empty collections and starting ID
    }
    
    pub fn create_node(&self, name: &str) -> Result<NodeId, HierarchyError> {
        // Create new node with unique ID
        // Check for name conflicts
        // Add to both nodes map and name index
    }
    
    pub fn create_child(&self, name: &str, parent: NodeId) -> Result<NodeId, HierarchyError> {
        // Create node and establish parent-child relationship
        // Update depth based on parent depth
    }
    
    pub fn get_node(&self, id: NodeId) -> Option<DashMapRef<NodeId, InheritanceNode>> {
        // Return reference to node for read operations
    }
    
    pub fn get_node_by_name(&self, name: &str) -> Option<NodeId> {
        // Lookup node ID by name
    }
    
    pub fn add_parent(&self, child: NodeId, parent: NodeId) -> Result<(), HierarchyError> {
        // Add parent-child relationship
        // Check for cycles before adding
        // Update both nodes' relationship lists
        // Recalculate depths if needed
    }
    
    pub fn remove_parent(&self, child: NodeId, parent: NodeId) -> Result<(), HierarchyError> {
        // Remove parent-child relationship
        // Update both nodes' relationship lists
        // Recalculate depths if needed
    }
    
    pub fn node_count(&self) -> usize {
        // Return total number of nodes
    }
    
    pub fn max_depth(&self) -> u32 {
        // Find the maximum depth in the hierarchy
    }
    
    pub fn get_depth(&self, node: NodeId) -> Option<u32> {
        // Return depth of specific node
    }
    
    pub fn detect_cycle(&self, start: NodeId, target: NodeId) -> bool {
        // Detect if adding edge from start to target would create cycle
        // Use DFS to check if target is reachable from start
    }
    
    pub fn recalculate_depths(&self) {
        // Recalculate depths for all nodes
        // Use topological ordering
    }
    
    pub fn get_roots(&self) -> Vec<NodeId> {
        // Return all nodes with no parents (depth 0)
    }
    
    pub fn get_children(&self, node: NodeId) -> Vec<NodeId> {
        // Return direct children of node
    }
    
    pub fn get_parents(&self, node: NodeId) -> Vec<NodeId> {
        // Return direct parents of node
    }
    
    pub fn get_ancestors(&self, node: NodeId) -> Vec<NodeId> {
        // Return all ancestors (parents, grandparents, etc.)
    }
    
    pub fn get_descendants(&self, node: NodeId) -> Vec<NodeId> {
        // Return all descendants (children, grandchildren, etc.)
    }
}

impl Default for InheritanceHierarchy {
    fn default() -> Self {
        Self::new()
    }
}
```

## Success Criteria (You must verify these)
- [ ] Thread-safe concurrent access using DashMap works correctly
- [ ] O(1) node lookup by ID (direct hash map access)
- [ ] O(1) node lookup by name (through name index)
- [ ] Parent-child relationships properly maintained bidirectionally
- [ ] Depth calculation works for arbitrary hierarchies including multiple inheritance
- [ ] Memory usage scales linearly with node count
- [ ] Cycle detection prevents invalid relationships
- [ ] Error handling provides meaningful error messages
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation_and_lookup() {
        let hierarchy = InheritanceHierarchy::new();
        
        let id = hierarchy.create_node("TestNode").unwrap();
        assert!(hierarchy.get_node(id).is_some());
        assert_eq!(hierarchy.get_node_by_name("TestNode"), Some(id));
        assert_eq!(hierarchy.node_count(), 1);
    }

    #[test]
    fn test_name_uniqueness() {
        let hierarchy = InheritanceHierarchy::new();
        
        let id1 = hierarchy.create_node("TestNode").unwrap();
        let result = hierarchy.create_node("TestNode");
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HierarchyError::NameAlreadyExists(_)));
    }

    #[test]
    fn test_parent_child_relationships() {
        let hierarchy = InheritanceHierarchy::new();
        
        let parent = hierarchy.create_node("Parent").unwrap();
        let child = hierarchy.create_child("Child", parent).unwrap();
        
        let parent_node = hierarchy.get_node(parent).unwrap();
        let child_node = hierarchy.get_node(child).unwrap();
        
        assert!(parent_node.children.contains(&child));
        assert!(child_node.parents.contains(&parent));
    }

    #[test]
    fn test_depth_calculation() {
        let hierarchy = InheritanceHierarchy::new();
        
        let root = hierarchy.create_node("Root").unwrap();
        let child = hierarchy.create_child("Child", root).unwrap();
        let grandchild = hierarchy.create_child("Grandchild", child).unwrap();
        
        assert_eq!(hierarchy.get_depth(root), Some(0));
        assert_eq!(hierarchy.get_depth(child), Some(1));
        assert_eq!(hierarchy.get_depth(grandchild), Some(2));
        assert_eq!(hierarchy.max_depth(), 2);
    }

    #[test]
    fn test_multiple_inheritance() {
        let hierarchy = InheritanceHierarchy::new();
        
        let parent1 = hierarchy.create_node("Parent1").unwrap();
        let parent2 = hierarchy.create_node("Parent2").unwrap();
        let child = hierarchy.create_node("Child").unwrap();
        
        hierarchy.add_parent(child, parent1).unwrap();
        hierarchy.add_parent(child, parent2).unwrap();
        
        let child_node = hierarchy.get_node(child).unwrap();
        assert_eq!(child_node.parents.len(), 2);
        assert!(child_node.parents.contains(&parent1));
        assert!(child_node.parents.contains(&parent2));
    }

    #[test]
    fn test_cycle_detection() {
        let hierarchy = InheritanceHierarchy::new();
        
        let node1 = hierarchy.create_node("Node1").unwrap();
        let node2 = hierarchy.create_child("Node2", node1).unwrap();
        let node3 = hierarchy.create_child("Node3", node2).unwrap();
        
        // Try to create cycle: node1 -> node2 -> node3 -> node1
        let result = hierarchy.add_parent(node1, node3);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HierarchyError::CycleDetected));
    }

    #[test]
    fn test_ancestor_descendant_queries() {
        let hierarchy = InheritanceHierarchy::new();
        
        let root = hierarchy.create_node("Root").unwrap();
        let child = hierarchy.create_child("Child", root).unwrap();
        let grandchild = hierarchy.create_child("Grandchild", child).unwrap();
        
        let ancestors = hierarchy.get_ancestors(grandchild);
        assert!(ancestors.contains(&child));
        assert!(ancestors.contains(&root));
        
        let descendants = hierarchy.get_descendants(root);
        assert!(descendants.contains(&child));
        assert!(descendants.contains(&grandchild));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let hierarchy = Arc::new(InheritanceHierarchy::new());
        let mut handles = vec![];
        
        // Create nodes concurrently
        for i in 0..10 {
            let h = Arc::clone(&hierarchy);
            let handle = thread::spawn(move || {
                h.create_node(&format!("Node{}", i))
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }
        
        // Verify all nodes were created
        assert_eq!(hierarchy.node_count(), 10);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let hierarchy = InheritanceHierarchy::new();
        
        // Create many nodes to test linear scaling
        for i in 0..1000 {
            hierarchy.create_node(&format!("Node{}", i)).unwrap();
        }
        
        assert_eq!(hierarchy.node_count(), 1000);
        // Memory usage should scale linearly - no quadratic growth
    }
}
```

## File to Create
Create exactly this file: `src/hierarchy/tree.rs`

## Dependencies Required
You may need to add dependencies to Cargo.toml:
```toml
[dependencies]
dashmap = "5.0"
```

## Implementation Notes
1. **Cycle Detection**: Use DFS to detect cycles before adding relationships
2. **Depth Calculation**: For multiple inheritance, use the minimum depth from any parent + 1
3. **Thread Safety**: DashMap provides interior mutability for concurrent access
4. **Memory Efficiency**: Avoid storing redundant data; use references where possible

## When Complete
Respond with "MICRO PHASE 1.3 COMPLETE" and a brief summary of what you implemented, including:
- Cycle detection strategy used
- Depth calculation approach for multiple inheritance
- Performance characteristics achieved
- Any optimizations implemented
- Confirmation that all tests pass