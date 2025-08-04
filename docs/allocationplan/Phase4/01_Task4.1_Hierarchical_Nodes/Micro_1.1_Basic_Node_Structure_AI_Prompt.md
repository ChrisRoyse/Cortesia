# AI Prompt: Micro Phase 1.1 - Basic Node Structure

You are tasked with implementing the foundational data structures for the inheritance node system. Your goal is to create `src/hierarchy/node.rs` with complete implementation that will form the foundation of the hierarchical system.

## Your Task
Implement the basic `InheritanceNode` struct and `NodeId` type that will store inheritance relationships, properties, and metadata for hierarchical knowledge structures.

## Specific Requirements
1. Create `src/hierarchy/node.rs` with all required structures
2. Implement NodeId type with unique identifier generation
3. Implement InheritanceNode struct with all required fields
4. Implement NodeMetadata struct for tracking depth, timing, and access patterns
5. Add proper trait implementations (Debug, Clone, PartialEq, Eq, Hash where appropriate)
6. Ensure thread-safe access patterns for shared metadata

## Expected Code Structure
You must implement these exact signatures:

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn generate() -> Self {
        // Implement unique ID generation (use atomic counter or UUID)
    }
}

#[derive(Debug, Clone)]
pub struct InheritanceNode {
    pub id: NodeId,
    pub name: String,
    pub parents: Vec<NodeId>,
    pub children: Vec<NodeId>,
    pub local_properties: HashMap<String, PropertyValue>,
    pub exceptions: HashMap<String, Exception>,
    pub metadata: NodeMetadata,
}

impl InheritanceNode {
    pub fn new(id: NodeId, name: &str) -> Self {
        // Implement constructor with default values
    }
    
    pub fn add_parent(&mut self, parent_id: NodeId) {
        // Add parent relationship
    }
    
    pub fn add_child(&mut self, child_id: NodeId) {
        // Add child relationship
    }
    
    pub fn add_property(&mut self, key: String, value: PropertyValue) {
        // Add local property
    }
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub depth: u32,
    pub creation_time: Instant,
    pub last_modified: AtomicU64,
    pub access_count: AtomicU32,
}

impl NodeMetadata {
    pub fn new(depth: u32) -> Self {
        // Implement constructor with current time
    }
    
    pub fn record_access(&self) {
        // Increment access count atomically
    }
    
    pub fn update_modified_time(&self) {
        // Update last_modified to current timestamp
    }
}

// You'll need these types - create placeholder implementations for now
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    // Add other variants as needed
}

#[derive(Debug, Clone)]
pub struct Exception {
    // Placeholder - will be implemented in later phases
}
```

## Success Criteria (You must verify these)
- [ ] NodeId generates unique identifiers for each call
- [ ] InheritanceNode stores parents, children, and properties correctly
- [ ] NodeMetadata tracks depth, creation time, access count atomically
- [ ] All fields properly typed with appropriate collections
- [ ] Basic trait implementations work correctly (Debug, Clone)
- [ ] Thread-safe access to atomic fields in metadata
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let id = NodeId::generate();
        let node = InheritanceNode::new(id, "TestNode");
        
        assert_eq!(node.id, id);
        assert_eq!(node.name, "TestNode");
        assert!(node.parents.is_empty());
        assert!(node.children.is_empty());
        assert!(node.local_properties.is_empty());
        assert_eq!(node.metadata.depth, 0);
    }

    #[test]
    fn test_unique_node_ids() {
        let id1 = NodeId::generate();
        let id2 = NodeId::generate();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parent_child_relationships() {
        let parent_id = NodeId::generate();
        let child_id = NodeId::generate();
        
        let mut parent = InheritanceNode::new(parent_id, "Parent");
        let mut child = InheritanceNode::new(child_id, "Child");
        
        parent.add_child(child_id);
        child.add_parent(parent_id);
        
        assert!(parent.children.contains(&child_id));
        assert!(child.parents.contains(&parent_id));
    }

    #[test]
    fn test_metadata_access_tracking() {
        let id = NodeId::generate();
        let node = InheritanceNode::new(id, "TestNode");
        
        let initial_count = node.metadata.access_count.load(Ordering::Relaxed);
        node.metadata.record_access();
        let after_count = node.metadata.access_count.load(Ordering::Relaxed);
        
        assert_eq!(after_count, initial_count + 1);
    }

    #[test]
    fn test_property_addition() {
        let id = NodeId::generate();
        let mut node = InheritanceNode::new(id, "TestNode");
        
        node.add_property("test_prop".to_string(), PropertyValue::String("test_value".to_string()));
        
        assert!(node.local_properties.contains_key("test_prop"));
        assert_eq!(
            node.local_properties.get("test_prop"),
            Some(&PropertyValue::String("test_value".to_string()))
        );
    }
}
```

## File to Create
Create exactly this file: `src/hierarchy/node.rs`

## Dependencies Required
You may need to add dependencies to Cargo.toml:
```toml
[dependencies]
uuid = "1.0"  # If using UUID for node IDs
```

## When Complete
Respond with "MICRO PHASE 1.1 COMPLETE" and a brief summary of what you implemented, including:
- NodeId generation strategy used
- Any design decisions made
- Confirmation that all tests pass