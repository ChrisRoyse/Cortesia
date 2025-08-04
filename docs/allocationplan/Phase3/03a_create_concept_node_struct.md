# Task 03a: Create Concept Node Struct

**Estimated Time**: 8 minutes  
**Dependencies**: 02e_create_relationship_indices.md  
**Next Task**: 03b_create_memory_node_struct.md  

## Objective
Create the ConceptNode data structure with core properties and serialization.

## Single Action
Create Rust struct for Concept nodes with essential fields.

## File to Create
File: `src/storage/node_types.rs`
```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConceptNode {
    pub id: String,
    pub name: String,
    pub concept_type: String,
    pub ttfs_encoding: Option<f32>,
    pub inheritance_depth: i32,
    pub property_count: i32,
    pub inherited_property_count: i32,
    pub semantic_embedding: Vec<f32>,
    pub creation_timestamp: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_frequency: i32,
    pub confidence_score: f32,
    pub source_attribution: Option<String>,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Valid,
    Pending,
    Invalid,
    RequiresReview,
}

impl ConceptNode {
    pub fn new(name: String, concept_type: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            concept_type,
            ttfs_encoding: None,
            inheritance_depth: 0,
            property_count: 0,
            inherited_property_count: 0,
            semantic_embedding: Vec::new(),
            creation_timestamp: now,
            last_accessed: now,
            access_frequency: 0,
            confidence_score: 1.0,
            source_attribution: None,
            validation_status: ValidationStatus::Valid,
        }
    }
    
    pub fn with_ttfs_encoding(mut self, encoding: f32) -> Self {
        self.ttfs_encoding = Some(encoding);
        self
    }
    
    pub fn with_semantic_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.semantic_embedding = embedding;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_concept_node_creation() {
        let concept = ConceptNode::new("TestConcept".to_string(), "Entity".to_string());
        
        assert_eq!(concept.name, "TestConcept");
        assert_eq!(concept.concept_type, "Entity");
        assert_eq!(concept.inheritance_depth, 0);
        assert_eq!(concept.confidence_score, 1.0);
        assert_eq!(concept.validation_status, ValidationStatus::Valid);
        assert!(!concept.id.is_empty());
    }
    
    #[test]
    fn test_concept_builder_pattern() {
        let concept = ConceptNode::new("Test".to_string(), "Type".to_string())
            .with_ttfs_encoding(0.85)
            .with_semantic_embedding(vec![0.1, 0.2, 0.3]);
            
        assert_eq!(concept.ttfs_encoding, Some(0.85));
        assert_eq!(concept.semantic_embedding, vec![0.1, 0.2, 0.3]);
    }
}
```

## Module Update
Add to `src/storage/mod.rs`:
```rust
pub mod node_types;

pub use node_types::*;
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test concept_node
```

## Acceptance Criteria
- [ ] ConceptNode struct compiles without errors
- [ ] All required fields included
- [ ] Implements Serialize/Deserialize
- [ ] Constructor and builder methods work
- [ ] Tests pass

## Duration
6-8 minutes for struct creation and testing.