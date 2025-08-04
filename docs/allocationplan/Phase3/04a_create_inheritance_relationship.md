# Task 04a: Create Inheritance Relationship Structure

**Estimated Time**: 8 minutes  
**Dependencies**: 03h_test_all_node_types.md  
**Next Task**: 04b_create_property_relationship.md  

## Objective
Create the InheritsFromRelationship data structure for hierarchical inheritance.

## Single Action
Create relationship types file with InheritsFromRelationship struct.

## File to Create
File: `src/storage/relationship_types.rs`
```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InheritsFromRelationship {
    pub id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub inheritance_type: InheritanceType,
    pub inheritance_depth: i32,
    pub property_mask: Vec<String>,
    pub exception_count: i32,
    pub strength: f32,
    pub established_at: DateTime<Utc>,
    pub last_validated: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InheritanceType {
    Direct,      // Direct parent-child relationship
    Multiple,    // Multiple inheritance (diamond problem)
    Interface,   // Interface-like inheritance
    Mixin,       // Mixin-style inheritance
    Prototype,   // Prototype-based inheritance
}

impl InheritsFromRelationship {
    pub fn new(
        source_node_id: String,
        target_node_id: String,
        inheritance_type: InheritanceType,
        depth: i32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            target_node_id,
            inheritance_type,
            inheritance_depth: depth,
            property_mask: Vec::new(),
            exception_count: 0,
            strength: 1.0,
            established_at: now,
            last_validated: now,
            is_active: true,
        }
    }
    
    pub fn with_property_mask(mut self, mask: Vec<String>) -> Self {
        self.property_mask = mask;
        self
    }
    
    pub fn add_exception(&mut self) {
        self.exception_count += 1;
    }
    
    pub fn validate(&mut self) -> bool {
        self.last_validated = Utc::now();
        // Basic validation logic
        self.inheritance_depth >= 0 && 
        self.strength >= 0.0 && 
        self.strength <= 1.0 &&
        !self.source_node_id.is_empty() &&
        !self.target_node_id.is_empty()
    }
    
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }
}

#[cfg(test)]
mod inheritance_tests {
    use super::*;
    
    #[test]
    fn test_inheritance_relationship_creation() {
        let rel = InheritsFromRelationship::new(
            "child_node".to_string(),
            "parent_node".to_string(),
            InheritanceType::Direct,
            1,
        );
        
        assert_eq!(rel.source_node_id, "child_node");
        assert_eq!(rel.target_node_id, "parent_node");
        assert_eq!(rel.inheritance_type, InheritanceType::Direct);
        assert_eq!(rel.inheritance_depth, 1);
        assert!(rel.is_active);
        assert_eq!(rel.exception_count, 0);
    }
    
    #[test]
    fn test_inheritance_validation() {
        let mut rel = InheritsFromRelationship::new(
            "a".to_string(),
            "b".to_string(),
            InheritanceType::Multiple,
            2,
        );
        
        assert!(rel.validate());
        
        // Test invalid cases
        rel.inheritance_depth = -1;
        assert!(!rel.validate());
        
        rel.inheritance_depth = 2;
        rel.strength = 2.0;
        assert!(!rel.validate());
    }
    
    #[test]
    fn test_property_mask() {
        let rel = InheritsFromRelationship::new(
            "child".to_string(),
            "parent".to_string(),
            InheritanceType::Interface,
            1,
        ).with_property_mask(vec!["prop1".to_string(), "prop2".to_string()]);
        
        assert_eq!(rel.property_mask.len(), 2);
        assert!(rel.property_mask.contains(&"prop1".to_string()));
    }
}
```

## Module Update
Add to `src/storage/mod.rs`:
```rust
pub mod relationship_types;

pub use relationship_types::*;
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run inheritance tests
cargo test inheritance_tests
```

## Acceptance Criteria
- [ ] InheritsFromRelationship struct compiles
- [ ] All inheritance types defined
- [ ] Validation logic works
- [ ] Property masking system functions
- [ ] Tests pass

## Duration
6-8 minutes for inheritance relationship implementation.