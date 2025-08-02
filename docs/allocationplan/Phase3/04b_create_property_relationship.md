# Task 04b: Create Property Relationship Structure

**Estimated Time**: 7 minutes  
**Dependencies**: 04a_create_inheritance_relationship.md  
**Next Task**: 04c_create_semantic_relationship.md  

## Objective
Create the HasPropertyRelationship data structure for property ownership.

## Single Action
Add HasPropertyRelationship struct to relationship_types.rs.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
use crate::storage::node_types::PropertySource;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HasPropertyRelationship {
    pub id: String,
    pub source_node_id: String,      // Node that has the property
    pub target_property_id: String,  // Property node ID
    pub property_source: PropertySource,
    pub inheritance_path: Vec<String>,
    pub override_level: i32,
    pub is_default: bool,
    pub confidence: f32,
    pub established_by: String,
    pub established_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub is_active: bool,
}

impl HasPropertyRelationship {
    pub fn new(
        source_node_id: String,
        target_property_id: String,
        property_source: PropertySource,
        established_by: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            target_property_id,
            property_source,
            inheritance_path: Vec::new(),
            override_level: 0,
            is_default: false,
            confidence: 1.0,
            established_by,
            established_at: now,
            last_modified: now,
            is_active: true,
        }
    }
    
    pub fn with_inheritance_path(mut self, path: Vec<String>) -> Self {
        self.inheritance_path = path;
        self.property_source = PropertySource::Inherited;
        self
    }
    
    pub fn set_override_level(mut self, level: i32) -> Self {
        self.override_level = level;
        self
    }
    
    pub fn mark_as_default(mut self) -> Self {
        self.is_default = true;
        self
    }
    
    pub fn update_confidence(&mut self, confidence: f32) {
        self.confidence = confidence.clamp(0.0, 1.0);
        self.last_modified = Utc::now();
    }
    
    pub fn validate(&self) -> bool {
        !self.source_node_id.is_empty() &&
        !self.target_property_id.is_empty() &&
        !self.established_by.is_empty() &&
        self.confidence >= 0.0 &&
        self.confidence <= 1.0 &&
        self.override_level >= 0
    }
    
    pub fn is_inherited(&self) -> bool {
        matches!(self.property_source, PropertySource::Inherited) && 
        !self.inheritance_path.is_empty()
    }
}

#[cfg(test)]
mod property_relationship_tests {
    use super::*;
    
    #[test]
    fn test_property_relationship_creation() {
        let rel = HasPropertyRelationship::new(
            "concept_1".to_string(),
            "property_1".to_string(),
            PropertySource::Direct,
            "system".to_string(),
        );
        
        assert_eq!(rel.source_node_id, "concept_1");
        assert_eq!(rel.target_property_id, "property_1");
        assert_eq!(rel.property_source, PropertySource::Direct);
        assert!(rel.is_active);
        assert!(!rel.is_default);
        assert_eq!(rel.override_level, 0);
    }
    
    #[test]
    fn test_inherited_property_relationship() {
        let rel = HasPropertyRelationship::new(
            "child".to_string(),
            "inherited_prop".to_string(),
            PropertySource::Direct,
            "inheritance_system".to_string(),
        ).with_inheritance_path(vec!["parent".to_string(), "grandparent".to_string()]);
        
        assert!(rel.is_inherited());
        assert_eq!(rel.property_source, PropertySource::Inherited);
        assert_eq!(rel.inheritance_path.len(), 2);
    }
    
    #[test]
    fn test_property_relationship_validation() {
        let mut rel = HasPropertyRelationship::new(
            "node".to_string(),
            "prop".to_string(),
            PropertySource::Computed,
            "user".to_string(),
        );
        
        assert!(rel.validate());
        
        // Test invalid confidence
        rel.update_confidence(1.5);
        assert_eq!(rel.confidence, 1.0); // Should be clamped
        
        // Test validation with empty fields
        rel.source_node_id = String::new();
        assert!(!rel.validate());
    }
    
    #[test]
    fn test_override_levels() {
        let rel = HasPropertyRelationship::new(
            "concept".to_string(),
            "overridden_prop".to_string(),
            PropertySource::Direct,
            "override_system".to_string(),
        ).set_override_level(3)
         .mark_as_default();
        
        assert_eq!(rel.override_level, 3);
        assert!(rel.is_default);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run property relationship tests
cargo test property_relationship_tests
```

## Acceptance Criteria
- [ ] HasPropertyRelationship struct compiles
- [ ] Inheritance path tracking works
- [ ] Override level system functions
- [ ] Confidence updating works
- [ ] Tests pass

## Duration
5-7 minutes for property relationship implementation.