# Task 24a: Create Conflict Types Enum
**Time**: 3 minutes (45 sec read, 1.5 min implement, 45 sec verify)
**Dependencies**: None
**Stage**: Conflict Detection Foundation

## Objective
Define the basic types of conflicts that can occur in the knowledge graph.

## Implementation
Create file `src/inheritance/validation/conflict_types.rs`:
```rust
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictType {
    Property,
    Temporal,
    Semantic,
    Inheritance,
    Relationship,
    Constraint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictStatus {
    Detected,
    Analyzing,
    Resolved,
    Ignored,
}

impl fmt::Display for ConflictType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictType::Property => write!(f, "Property Conflict"),
            ConflictType::Temporal => write!(f, "Temporal Conflict"),
            ConflictType::Semantic => write!(f, "Semantic Conflict"),
            ConflictType::Inheritance => write!(f, "Inheritance Conflict"),
            ConflictType::Relationship => write!(f, "Relationship Conflict"),
            ConflictType::Constraint => write!(f, "Constraint Conflict"),
        }
    }
}
```

## Success Criteria
- All conflict types defined
- Severity and status enums created
- Display trait implemented
- Serialization support added

**Next**: 24b