# Task 04c: Create Semantic Relationship Structure

**Estimated Time**: 8 minutes  
**Dependencies**: 04b_create_property_relationship.md  
**Next Task**: 04d_create_temporal_relationship.md  

## Objective
Create the SemanticallyRelatedRelationship for semantic similarity connections.

## Single Action
Add SemanticallyRelatedRelationship struct to relationship_types.rs.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticallyRelatedRelationship {
    pub id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub similarity_score: f32,
    pub similarity_type: SimilarityType,
    pub semantic_distance: f32,
    pub context_vector: Vec<f32>,
    pub computation_method: String,
    pub confidence: f32,
    pub established_at: DateTime<Utc>,
    pub last_computed: DateTime<Utc>,
    pub usage_count: i32,
    pub is_bidirectional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimilarityType {
    Conceptual,      // Concepts are similar in meaning
    Contextual,      // Similar in usage context
    Structural,      // Similar in structure/properties
    Functional,      // Similar in function/behavior
    Associative,     // Mentally associated
    Temporal,        // Related in time
}

impl SemanticallyRelatedRelationship {
    pub fn new(
        source_node_id: String,
        target_node_id: String,
        similarity_score: f32,
        similarity_type: SimilarityType,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            target_node_id,
            similarity_score: similarity_score.clamp(0.0, 1.0),
            similarity_type,
            semantic_distance: 1.0 - similarity_score.clamp(0.0, 1.0),
            context_vector: Vec::new(),
            computation_method: "default".to_string(),
            confidence: 1.0,
            established_at: now,
            last_computed: now,
            usage_count: 0,
            is_bidirectional: true,
        }
    }
    
    pub fn with_context_vector(mut self, vector: Vec<f32>) -> Self {
        self.context_vector = vector;
        self
    }
    
    pub fn with_computation_method(mut self, method: String) -> Self {
        self.computation_method = method;
        self
    }
    
    pub fn set_unidirectional(mut self) -> Self {
        self.is_bidirectional = false;
        self
    }
    
    pub fn update_similarity_score(&mut self, new_score: f32) {
        self.similarity_score = new_score.clamp(0.0, 1.0);
        self.semantic_distance = 1.0 - self.similarity_score;
        self.last_computed = Utc::now();
    }
    
    pub fn record_usage(&mut self) {
        self.usage_count += 1;
    }
    
    pub fn is_strong_relationship(&self) -> bool {
        self.similarity_score > 0.7 && self.confidence > 0.8
    }
    
    pub fn is_weak_relationship(&self) -> bool {
        self.similarity_score < 0.3 || self.confidence < 0.5
    }
    
    pub fn validate(&self) -> bool {
        !self.source_node_id.is_empty() &&
        !self.target_node_id.is_empty() &&
        self.source_node_id != self.target_node_id && // No self-loops
        self.similarity_score >= 0.0 &&
        self.similarity_score <= 1.0 &&
        self.confidence >= 0.0 &&
        self.confidence <= 1.0
    }
}

#[cfg(test)]
mod semantic_relationship_tests {
    use super::*;
    
    #[test]
    fn test_semantic_relationship_creation() {
        let rel = SemanticallyRelatedRelationship::new(
            "concept_a".to_string(),
            "concept_b".to_string(),
            0.85,
            SimilarityType::Conceptual,
        );
        
        assert_eq!(rel.source_node_id, "concept_a");
        assert_eq!(rel.target_node_id, "concept_b");
        assert_eq!(rel.similarity_score, 0.85);
        assert_eq!(rel.semantic_distance, 0.15);
        assert!(rel.is_bidirectional);
        assert!(rel.validate());
    }
    
    #[test]
    fn test_similarity_score_update() {
        let mut rel = SemanticallyRelatedRelationship::new(
            "node1".to_string(),
            "node2".to_string(),
            0.5,
            SimilarityType::Contextual,
        );
        
        let original_time = rel.last_computed;
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        rel.update_similarity_score(0.9);
        
        assert_eq!(rel.similarity_score, 0.9);
        assert_eq!(rel.semantic_distance, 0.1);
        assert!(rel.last_computed > original_time);
    }
    
    #[test]
    fn test_relationship_strength_classification() {
        let strong_rel = SemanticallyRelatedRelationship::new(
            "a".to_string(),
            "b".to_string(),
            0.8,
            SimilarityType::Functional,
        );
        
        let weak_rel = SemanticallyRelatedRelationship::new(
            "c".to_string(),
            "d".to_string(),
            0.2,
            SimilarityType::Associative,
        );
        
        assert!(strong_rel.is_strong_relationship());
        assert!(!strong_rel.is_weak_relationship());
        
        assert!(!weak_rel.is_strong_relationship());
        assert!(weak_rel.is_weak_relationship());
    }
    
    #[test]
    fn test_context_vector() {
        let rel = SemanticallyRelatedRelationship::new(
            "concept1".to_string(),
            "concept2".to_string(),
            0.7,
            SimilarityType::Structural,
        ).with_context_vector(vec![0.1, 0.2, 0.3, 0.4])
         .with_computation_method("cosine_similarity".to_string());
        
        assert_eq!(rel.context_vector, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(rel.computation_method, "cosine_similarity");
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run semantic relationship tests
cargo test semantic_relationship_tests
```

## Acceptance Criteria
- [ ] SemanticallyRelatedRelationship struct compiles
- [ ] Similarity scoring system works
- [ ] Relationship strength classification functions
- [ ] Context vector support works
- [ ] Tests pass

## Duration
6-8 minutes for semantic relationship implementation.