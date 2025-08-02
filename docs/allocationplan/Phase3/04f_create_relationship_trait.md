# Task 04f: Create Relationship Trait Interface

**Estimated Time**: 9 minutes  
**Dependencies**: 04e_create_neural_relationship.md  
**Next Task**: 05a_create_basic_node_operations.md  

## Objective
Create a common trait interface for all relationship types to enable polymorphic operations.

## Single Action
Add GraphRelationship trait and implement it for all relationship types.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
use std::any::Any;

/// Common trait for all graph relationship types
pub trait GraphRelationship: Send + Sync {
    fn id(&self) -> &str;
    fn relationship_type(&self) -> &str;
    fn source_node_id(&self) -> &str;
    fn target_node_id(&self) -> &str;
    fn established_at(&self) -> DateTime<Utc>;
    fn last_modified(&self) -> DateTime<Utc>;
    fn strength(&self) -> f32;
    fn is_active(&self) -> bool;
    fn as_any(&self) -> &dyn Any;
    fn validate(&self) -> bool;
    fn to_json(&self) -> Result<String, serde_json::Error>;
}

// Implement GraphRelationship for InheritsFromRelationship
impl GraphRelationship for InheritsFromRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "INHERITS_FROM"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_node_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_validated
    }
    
    fn strength(&self) -> f32 {
        self.strength
    }
    
    fn is_active(&self) -> bool {
        self.is_active
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        let mut mutable_self = self.clone();
        mutable_self.validate()
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphRelationship for HasPropertyRelationship
impl GraphRelationship for HasPropertyRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "HAS_PROPERTY"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_property_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_modified
    }
    
    fn strength(&self) -> f32 {
        self.confidence
    }
    
    fn is_active(&self) -> bool {
        self.is_active
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        Self::validate(self)
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphRelationship for SemanticallyRelatedRelationship
impl GraphRelationship for SemanticallyRelatedRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "SEMANTICALLY_RELATED"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_node_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_computed
    }
    
    fn strength(&self) -> f32 {
        self.similarity_score
    }
    
    fn is_active(&self) -> bool {
        !self.is_weak_relationship()
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        Self::validate(self)
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphRelationship for TemporalSequenceRelationship
impl GraphRelationship for TemporalSequenceRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "TEMPORAL_SEQUENCE"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_node_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn strength(&self) -> f32 {
        self.confidence * self.causal_strength
    }
    
    fn is_active(&self) -> bool {
        self.confidence > 0.3
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        Self::validate(self)
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphRelationship for NeuralConnectionRelationship
impl GraphRelationship for NeuralConnectionRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "NEURAL_CONNECTION"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_node_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_activated
    }
    
    fn strength(&self) -> f32 {
        self.get_effective_strength()
    }
    
    fn is_active(&self) -> bool {
        Self::is_active(self)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        Self::validate(self)
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod relationship_trait_tests {
    use super::*;
    use crate::storage::node_types::PropertySource;
    
    #[test]
    fn test_polymorphic_relationship_operations() {
        let relationships: Vec<Box<dyn GraphRelationship>> = vec![
            Box::new(InheritsFromRelationship::new(
                "child".to_string(),
                "parent".to_string(),
                InheritanceType::Direct,
                1,
            )),
            Box::new(HasPropertyRelationship::new(
                "node".to_string(),
                "property".to_string(),
                PropertySource::Direct,
                "system".to_string(),
            )),
            Box::new(SemanticallyRelatedRelationship::new(
                "concept_a".to_string(),
                "concept_b".to_string(),
                0.8,
                SimilarityType::Conceptual,
            )),
            Box::new(TemporalSequenceRelationship::new(
                "event_1".to_string(),
                "event_2".to_string(),
                SequenceType::Causal,
                Utc::now(),
                Utc::now() + chrono::Duration::hours(1),
            )),
            Box::new(NeuralConnectionRelationship::new(
                "neuron_a".to_string(),
                "neuron_b".to_string(),
                ConnectionType::Excitatory,
            )),
        ];
        
        for (i, rel) in relationships.iter().enumerate() {
            println!("Testing relationship {}: {}", i, rel.relationship_type());
            
            assert!(!rel.id().is_empty());
            assert!(!rel.relationship_type().is_empty());
            assert!(!rel.source_node_id().is_empty());
            assert!(!rel.target_node_id().is_empty());
            assert!(rel.validate());
            assert!(rel.strength() >= 0.0);
            
            let json_result = rel.to_json();
            assert!(json_result.is_ok(), "JSON serialization failed for {}", rel.relationship_type());
        }
    }
    
    #[test]
    fn test_relationship_type_identification() {
        let inheritance_rel = InheritsFromRelationship::new(
            "a".to_string(),
            "b".to_string(),
            InheritanceType::Direct,
            1,
        );
        
        let semantic_rel = SemanticallyRelatedRelationship::new(
            "x".to_string(),
            "y".to_string(),
            0.7,
            SimilarityType::Contextual,
        );
        
        assert_eq!(inheritance_rel.relationship_type(), "INHERITS_FROM");
        assert_eq!(semantic_rel.relationship_type(), "SEMANTICALLY_RELATED");
    }
    
    #[test]
    fn test_relationship_strength_consistency() {
        let neural_conn = NeuralConnectionRelationship::new(
            "n1".to_string(),
            "n2".to_string(),
            ConnectionType::Excitatory,
        ).with_strength(0.6);
        
        // Strength should be effective strength (connection_strength * synaptic_weight)
        let expected_strength = neural_conn.connection_strength * neural_conn.synaptic_weight;
        assert!((neural_conn.strength() - expected_strength).abs() < 0.01);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run relationship trait tests
cargo test relationship_trait_tests
```

## Acceptance Criteria
- [ ] GraphRelationship trait compiles without errors
- [ ] All relationship types implement the trait
- [ ] Polymorphic operations work correctly
- [ ] Strength calculation is consistent
- [ ] Tests pass

## Duration
7-9 minutes for trait implementation and testing.