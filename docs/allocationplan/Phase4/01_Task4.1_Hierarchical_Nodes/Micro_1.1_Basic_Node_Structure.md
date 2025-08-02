# Micro Phase 1.1: Basic Node Structure

**Estimated Time**: 30 minutes
**Dependencies**: None
**Objective**: Create the foundational data structures for inheritance nodes

## Task Description

Implement the basic `InheritanceNode` struct and `NodeId` type that will form the foundation of the hierarchical system.

## Deliverables

Create `src/hierarchy/node.rs` with:

1. **NodeId type**: Unique identifier for nodes
2. **InheritanceNode struct**: Core node structure
3. **NodeMetadata struct**: Tracking metadata
4. **Basic node operations**: Creation, property access

## Success Criteria

- [ ] NodeId generates unique identifiers
- [ ] InheritanceNode stores parents, children, and properties
- [ ] NodeMetadata tracks depth, creation time, access count
- [ ] All fields properly typed with appropriate collections
- [ ] Basic trait implementations (Debug, Clone)

## Implementation Requirements

```rust
// Expected struct signatures:

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

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

pub struct NodeMetadata {
    pub depth: u32,
    pub creation_time: Instant,
    pub last_modified: AtomicU64,
    pub access_count: AtomicU32,
}
```

## Test Requirements

Must pass basic creation test:
```rust
#[test]
fn test_node_creation() {
    let id = NodeId::generate();
    let node = InheritanceNode::new(id, "TestNode");
    
    assert_eq!(node.id, id);
    assert_eq!(node.name, "TestNode");
    assert!(node.parents.is_empty());
    assert!(node.children.is_empty());
    assert!(node.local_properties.is_empty());
}
```

## File Location
`src/hierarchy/node.rs`

## Next Micro Phase
After completion, proceed to Micro 1.2: Property Value System