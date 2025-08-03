# Task 11b: Create Inheritance Chain Types

**Time**: 3 minutes
**Dependencies**: 11a_inheritance_types.md
**Stage**: Inheritance System

## Objective
Add data structures for inheritance chains and validation.

## Implementation
Add to `src/inheritance/hierarchy_types.rs`:

```rust
#[derive(Debug, Clone)]
pub struct InheritanceChain {
    pub child_concept_id: String,
    pub chain: Vec<InheritanceLink>,
    pub total_depth: u32,
    pub is_valid: bool,
    pub has_cycles: bool,
}

#[derive(Debug, Clone)]
pub struct InheritanceLink {
    pub parent_concept_id: String,
    pub relationship_id: String,
    pub inheritance_type: InheritanceType,
    pub depth_from_child: u32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct HierarchyValidationResult {
    pub is_valid: bool,
    pub validation_errors: Vec<String>,
    pub cycle_detected: bool,
    pub max_depth_exceeded: bool,
    pub orphaned_concepts: Vec<String>,
}
```

## Success Criteria
- File compiles without errors
- All structs are properly defined

## Next Task
11c_hierarchy_manager_struct.md