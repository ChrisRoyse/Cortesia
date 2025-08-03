# Task 11a: Create Inheritance Data Types

**Time**: 4 minutes
**Dependencies**: 10_spike_pattern_processing.md
**Stage**: Inheritance System

## Objective
Create the basic data structures for inheritance relationships.

## Implementation
Create `src/inheritance/hierarchy_types.rs`:

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceRelationship {
    pub id: String,
    pub parent_concept_id: String,
    pub child_concept_id: String,
    pub inheritance_type: InheritanceType,
    pub inheritance_weight: f32,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub is_active: bool,
    pub inheritance_depth: u32,
    pub precedence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InheritanceType {
    ClassInheritance,
    PrototypeInheritance,
    MixinInheritance,
    InterfaceInheritance,
    CompositionInheritance,
    Custom(String),
}
```

## Success Criteria
- File compiles without errors
- Enums and structs are properly defined

## Next Task
11b_inheritance_chain_types.md