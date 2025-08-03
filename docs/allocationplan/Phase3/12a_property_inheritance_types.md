# Task 12a: Create Property Inheritance Types

**Time**: 4 minutes
**Dependencies**: 11j_cycle_detection.md
**Stage**: Inheritance System

## Objective
Create data structures for property inheritance resolution.

## Implementation
Create `src/inheritance/property_types.rs`:

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ResolvedProperties {
    pub concept_id: String,
    pub direct_properties: Vec<PropertyNode>,
    pub inherited_properties: Vec<InheritedProperty>,
    pub resolution_time: DateTime<Utc>,
    pub total_property_count: usize,
}

#[derive(Debug, Clone)]
pub struct InheritedProperty {
    pub property: PropertyNode,
    pub source_concept_id: String,
    pub inheritance_depth: i32,
    pub inheritance_strength: f32,
    pub has_exception: bool,
    pub exception_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PropertyNode {
    pub name: String,
    pub value: PropertyValue,
    pub is_inheritable: bool,
    pub inheritance_priority: u32,
}

#[derive(Debug, Clone)]
pub enum PropertyValue {
    Text(String),
    Number(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
}
```

## Success Criteria
- File compiles without errors
- All property structures are defined

## Next Task
12b_inheritance_engine_struct.md