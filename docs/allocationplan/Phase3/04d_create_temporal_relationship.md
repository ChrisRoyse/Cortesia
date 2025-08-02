# Task 04d: Create Temporal Relationship Structure

**Estimated Time**: 7 minutes  
**Dependencies**: 04c_create_semantic_relationship.md  
**Next Task**: 04e_create_neural_relationship.md  

## Objective
Create the TemporalSequenceRelationship for time-based ordering connections.

## Single Action
Add TemporalSequenceRelationship struct to relationship_types.rs.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalSequenceRelationship {
    pub id: String,
    pub source_node_id: String,      // Earlier node
    pub target_node_id: String,      // Later node
    pub sequence_type: SequenceType,
    pub temporal_distance: i64,      // Milliseconds between events
    pub confidence: f32,
    pub established_at: DateTime<Utc>,
    pub source_timestamp: DateTime<Utc>,
    pub target_timestamp: DateTime<Utc>,
    pub context: Option<String>,
    pub causal_strength: f32,
    pub is_direct: bool,             // Direct vs indirect sequence
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SequenceType {
    Causal,          // A causes B
    Temporal,        // A happens before B
    Procedural,      // A is step before B in process
    Narrative,       // A precedes B in story/narrative
    Logical,         // A logically precedes B
    Developmental,   // A develops into B
}

impl TemporalSequenceRelationship {
    pub fn new(
        source_node_id: String,
        target_node_id: String,
        sequence_type: SequenceType,
        source_timestamp: DateTime<Utc>,
        target_timestamp: DateTime<Utc>,
    ) -> Self {
        let temporal_distance = (target_timestamp - source_timestamp)
            .num_milliseconds();
        
        Self {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            target_node_id,
            sequence_type,
            temporal_distance,
            confidence: 1.0,
            established_at: Utc::now(),
            source_timestamp,
            target_timestamp,
            context: None,
            causal_strength: 0.5, // Default medium causal strength
            is_direct: true,
        }
    }
    
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
    
    pub fn with_causal_strength(mut self, strength: f32) -> Self {
        self.causal_strength = strength.clamp(0.0, 1.0);
        self
    }
    
    pub fn set_indirect(mut self) -> Self {
        self.is_direct = false;
        self
    }
    
    pub fn get_duration_seconds(&self) -> f64 {
        self.temporal_distance as f64 / 1000.0
    }
    
    pub fn get_duration_minutes(&self) -> f64 {
        self.get_duration_seconds() / 60.0
    }
    
    pub fn is_near_simultaneous(&self) -> bool {
        self.temporal_distance.abs() < 1000 // Within 1 second
    }
    
    pub fn is_strong_causal(&self) -> bool {
        matches!(self.sequence_type, SequenceType::Causal) && 
        self.causal_strength > 0.7 &&
        self.confidence > 0.8
    }
    
    pub fn validate(&self) -> bool {
        !self.source_node_id.is_empty() &&
        !self.target_node_id.is_empty() &&
        self.source_node_id != self.target_node_id &&
        self.confidence >= 0.0 &&
        self.confidence <= 1.0 &&
        self.causal_strength >= 0.0 &&
        self.causal_strength <= 1.0
    }
}

#[cfg(test)]
mod temporal_relationship_tests {
    use super::*;
    use chrono::Duration;
    
    #[test]
    fn test_temporal_relationship_creation() {
        let now = Utc::now();
        let later = now + Duration::hours(2);
        
        let rel = TemporalSequenceRelationship::new(
            "event_a".to_string(),
            "event_b".to_string(),
            SequenceType::Causal,
            now,
            later,
        );
        
        assert_eq!(rel.source_node_id, "event_a");
        assert_eq!(rel.target_node_id, "event_b");
        assert_eq!(rel.sequence_type, SequenceType::Causal);
        assert!(rel.temporal_distance > 0);
        assert!(rel.is_direct);
        assert!(rel.validate());
    }
    
    #[test]
    fn test_temporal_distance_calculations() {
        let start = Utc::now();
        let end = start + Duration::minutes(30);
        
        let rel = TemporalSequenceRelationship::new(
            "start_event".to_string(),
            "end_event".to_string(),
            SequenceType::Procedural,
            start,
            end,
        );
        
        assert_eq!(rel.get_duration_minutes(), 30.0);
        assert_eq!(rel.get_duration_seconds(), 1800.0);
        assert!(!rel.is_near_simultaneous());
    }
    
    #[test]
    fn test_near_simultaneous_events() {
        let time1 = Utc::now();
        let time2 = time1 + Duration::milliseconds(500);
        
        let rel = TemporalSequenceRelationship::new(
            "event1".to_string(),
            "event2".to_string(),
            SequenceType::Temporal,
            time1,
            time2,
        );
        
        assert!(rel.is_near_simultaneous());
    }
    
    #[test]
    fn test_causal_strength() {
        let now = Utc::now();
        let later = now + Duration::hours(1);
        
        let rel = TemporalSequenceRelationship::new(
            "cause".to_string(),
            "effect".to_string(),
            SequenceType::Causal,
            now,
            later,
        ).with_causal_strength(0.9)
         .with_context("Strong causal relationship".to_string());
        
        assert!(rel.is_strong_causal());
        assert_eq!(rel.causal_strength, 0.9);
        assert_eq!(rel.context, Some("Strong causal relationship".to_string()));
    }
    
    #[test]
    fn test_sequence_types() {
        let now = Utc::now();
        let later = now + Duration::seconds(10);
        
        let narrative_rel = TemporalSequenceRelationship::new(
            "chapter1".to_string(),
            "chapter2".to_string(),
            SequenceType::Narrative,
            now,
            later,
        );
        
        let developmental_rel = TemporalSequenceRelationship::new(
            "larva".to_string(),
            "butterfly".to_string(),
            SequenceType::Developmental,
            now,
            later,
        ).set_indirect();
        
        assert_eq!(narrative_rel.sequence_type, SequenceType::Narrative);
        assert_eq!(developmental_rel.sequence_type, SequenceType::Developmental);
        assert!(!developmental_rel.is_direct);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run temporal relationship tests
cargo test temporal_relationship_tests
```

## Acceptance Criteria
- [ ] TemporalSequenceRelationship struct compiles
- [ ] Temporal distance calculations work
- [ ] Causal strength assessment functions
- [ ] Duration calculation methods work
- [ ] Tests pass

## Duration
5-7 minutes for temporal relationship implementation.