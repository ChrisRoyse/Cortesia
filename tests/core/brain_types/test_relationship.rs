//! Comprehensive tests for BrainInspiredRelationship functionality
//! 
//! This module provides exhaustive testing for brain-inspired relationships including:
//! - Relationship creation and initialization
//! - Hebbian learning and weight strengthening
//! - Temporal decay and weight management
//! - Inhibitory connections and properties
//! - Metadata and usage tracking
//! - Edge cases and boundary conditions
//! - Performance characteristics

use crate::core::brain_types::{BrainInspiredRelationship, RelationType};
use crate::core::types::EntityKey;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;

/// Test fixture for creating various brain-inspired relationships
pub struct RelationshipTestFixture {
    pub isa_relationship: BrainInspiredRelationship,
    pub has_instance_relationship: BrainInspiredRelationship,
    pub has_property_relationship: BrainInspiredRelationship,
    pub related_to_relationship: BrainInspiredRelationship,
    pub part_of_relationship: BrainInspiredRelationship,
    pub similar_relationship: BrainInspiredRelationship,
    pub opposite_relationship: BrainInspiredRelationship,
    pub temporal_relationship: BrainInspiredRelationship,
    pub learned_relationship: BrainInspiredRelationship,
}

impl RelationshipTestFixture {
    pub fn new() -> Self {
        let source = EntityKey::default();
        let target = EntityKey::default();
        
        Self {
            isa_relationship: BrainInspiredRelationship::new(source, target, RelationType::IsA),
            has_instance_relationship: BrainInspiredRelationship::new(source, target, RelationType::HasInstance),
            has_property_relationship: BrainInspiredRelationship::new(source, target, RelationType::HasProperty),
            related_to_relationship: BrainInspiredRelationship::new(source, target, RelationType::RelatedTo),
            part_of_relationship: BrainInspiredRelationship::new(source, target, RelationType::PartOf),
            similar_relationship: BrainInspiredRelationship::new(source, target, RelationType::Similar),
            opposite_relationship: BrainInspiredRelationship::new(source, target, RelationType::Opposite),
            temporal_relationship: BrainInspiredRelationship::new(source, target, RelationType::Temporal),
            learned_relationship: BrainInspiredRelationship::new(source, target, RelationType::Learned),
        }
    }
    
    pub fn create_relationship_with_metadata() -> BrainInspiredRelationship {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.metadata.insert("source_concept".to_string(), "animal".to_string());
        relationship.metadata.insert("target_concept".to_string(), "mammal".to_string());
        relationship.metadata.insert("confidence".to_string(), "0.95".to_string());
        relationship.metadata.insert("learned_from".to_string(), "training_data".to_string());
        
        relationship
    }
    
    pub fn create_inhibitory_relationship() -> BrainInspiredRelationship {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::Opposite
        );
        
        relationship.is_inhibitory = true;
        relationship.weight = 0.8;
        relationship.strength = 0.8;
        relationship.temporal_decay = 0.05; // Slower decay for important inhibitory connections
        
        relationship
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Relationship Creation Tests
    #[test]
    fn test_relationship_creation() {
        let source = EntityKey::default();
        let target = EntityKey::default();
        let relationship = BrainInspiredRelationship::new(source, target, RelationType::IsA);
        
        assert_eq!(relationship.source, source);
        assert_eq!(relationship.target, target);
        assert_eq!(relationship.source_key, source); // Compatibility field
        assert_eq!(relationship.target_key, target); // Compatibility field
        assert_eq!(relationship.relation_type, RelationType::IsA);
        assert_eq!(relationship.weight, 1.0);
        assert_eq!(relationship.strength, 1.0);
        assert!(!relationship.is_inhibitory);
        assert_eq!(relationship.temporal_decay, 0.1);
        assert_eq!(relationship.activation_count, 0);
        assert_eq!(relationship.usage_count, 0);
        assert!(relationship.metadata.is_empty());
        
        // Temporal fields should be initialized to now
        let now = SystemTime::now();
        let time_diff = now.duration_since(relationship.last_strengthened).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "last_strengthened should be recent");
        
        let time_diff = now.duration_since(relationship.last_update).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "last_update should be recent");
        
        let time_diff = now.duration_since(relationship.creation_time).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "creation_time should be recent");
        
        let time_diff = now.duration_since(relationship.ingestion_time).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "ingestion_time should be recent");
    }

    #[test]
    fn test_relationship_creation_different_types() {
        let source = EntityKey::default();
        let target = EntityKey::default();
        
        let types = vec![
            RelationType::IsA,
            RelationType::HasInstance,
            RelationType::HasProperty,
            RelationType::RelatedTo,
            RelationType::PartOf,
            RelationType::Similar,
            RelationType::Opposite,
            RelationType::Temporal,
            RelationType::Learned,
        ];
        
        for relation_type in types {
            let relationship = BrainInspiredRelationship::new(source, target, relation_type);
            assert_eq!(relationship.relation_type, relation_type);
            assert_eq!(relationship.weight, 1.0);
            assert_eq!(relationship.strength, 1.0);
        }
    }

    // Hebbian Learning Tests
    #[test]
    fn test_basic_strengthening() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_weight = relationship.weight;
        let initial_count = relationship.activation_count;
        let initial_usage_count = relationship.usage_count;
        
        relationship.strengthen(0.1);
        
        assert!(relationship.weight > initial_weight, "Weight should increase after strengthening");
        assert_eq!(relationship.strength, relationship.weight, "Strength should match weight");
        assert_eq!(relationship.activation_count, initial_count + 1, "Activation count should increment");
        assert_eq!(relationship.usage_count, initial_usage_count + 1, "Usage count should increment");
        
        // Times should be updated
        assert!(relationship.last_strengthened <= SystemTime::now());
        assert!(relationship.last_update <= SystemTime::now());
    }

    #[test]
    fn test_strengthening_accumulation() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Multiple strengthening operations
        relationship.strengthen(0.1);
        let weight_after_first = relationship.weight;
        
        relationship.strengthen(0.1);
        let weight_after_second = relationship.weight;
        
        relationship.strengthen(0.1);
        let weight_after_third = relationship.weight;
        
        assert!(weight_after_second > weight_after_first, "Weight should continue increasing");
        assert!(weight_after_third > weight_after_second, "Weight should continue increasing");
        assert_eq!(relationship.activation_count, 3, "Should track multiple activations");
    }

    #[test]
    fn test_strengthening_clamping() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Strengthen to maximum
        relationship.strengthen(0.5); // Should clamp to 1.0
        assert_eq!(relationship.weight, 1.0, "Weight should be clamped to 1.0");
        assert_eq!(relationship.strength, 1.0, "Strength should match weight");
    }

    #[test]
    fn test_strengthening_with_different_rates() {
        let mut rel1 = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let mut rel2 = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Different learning rates
        rel1.strengthen(0.1);
        rel2.strengthen(0.2);
        
        assert!(rel2.weight > rel1.weight, "Higher learning rate should result in higher weight");
    }

    #[test]
    fn test_strengthening_from_low_weight() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Set low initial weight
        relationship.weight = 0.1;
        relationship.strength = 0.1;
        
        relationship.strengthen(0.2);
        assert_eq!(relationship.weight, 0.3, "Should add to existing weight");
        assert_eq!(relationship.strength, 0.3, "Strength should match");
    }

    // Temporal Decay Tests
    #[test]
    fn test_temporal_decay_immediate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.strength = 1.0;
        
        // Immediate decay (no time passed)
        let result = relationship.apply_decay();
        assert_eq!(result, 1.0, "No decay should occur immediately");
        assert_eq!(relationship.weight, 1.0);
        assert_eq!(relationship.strength, 1.0);
    }

    #[test]
    fn test_temporal_decay_calculation() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.strength = 1.0;
        relationship.temporal_decay = 0.1;
        
        // Manually set last_strengthened to simulate time passing
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        
        let result = relationship.apply_decay();
        // Expected: 1.0 * exp(-0.1 * 1) ≈ 1.0 * 0.9048 ≈ 0.9048
        assert!((result - 0.9048).abs() < 0.01, "Decay should follow exponential formula");
        assert!((relationship.weight - 0.9048).abs() < 0.01, "Weight should be updated");
        assert_eq!(relationship.strength, relationship.weight, "Strength should match weight");
    }

    #[test]
    fn test_high_decay_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.temporal_decay = 1.0; // High decay rate
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(2);
        
        let result = relationship.apply_decay();
        // Expected: 1.0 * exp(-1.0 * 2) ≈ 1.0 * 0.1353 ≈ 0.1353
        assert!((result - 0.1353).abs() < 0.01, "High decay rate should significantly reduce weight");
    }

    #[test]
    fn test_low_decay_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.temporal_decay = 0.01; // Low decay rate
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(10);
        
        let result = relationship.apply_decay();
        // Expected: 1.0 * exp(-0.01 * 10) ≈ 1.0 * 0.9048 ≈ 0.9048
        assert!((result - 0.9048).abs() < 0.01, "Low decay rate should preserve weight better");
    }

    #[test]
    fn test_very_long_decay() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.temporal_decay = 0.1;
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(3600); // 1 hour
        
        let result = relationship.apply_decay();
        // After 1 hour with decay rate 0.1, weight should be very small
        assert!(result < 0.01, "Very long decay should reduce weight to near zero");
    }

    #[test]
    fn test_decay_updates_timestamp() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_update_time = relationship.last_update;
        
        // Small delay to ensure timestamp difference
        thread::sleep(Duration::from_millis(10));
        relationship.apply_decay();
        
        assert!(relationship.last_update > initial_update_time, "Decay should update timestamp");
    }

    // Inhibitory Connection Tests
    #[test]
    fn test_inhibitory_relationship_creation() {
        let inhibitory_rel = RelationshipTestFixture::create_inhibitory_relationship();
        
        assert!(inhibitory_rel.is_inhibitory, "Should be marked as inhibitory");
        assert_eq!(inhibitory_rel.relation_type, RelationType::Opposite);
        assert_eq!(inhibitory_rel.weight, 0.8);
        assert_eq!(inhibitory_rel.temporal_decay, 0.05);
    }

    #[test]
    fn test_inhibitory_vs_excitatory() {
        let mut excitatory = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::Similar
        );
        
        let mut inhibitory = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::Opposite
        );
        inhibitory.is_inhibitory = true;
        
        assert!(!excitatory.is_inhibitory, "Excitatory should not be inhibitory");
        assert!(inhibitory.is_inhibitory, "Inhibitory should be marked as such");
        
        // Both should strengthen the same way
        excitatory.strengthen(0.1);
        inhibitory.strengthen(0.1);
        
        assert!(excitatory.weight > 1.0);
        assert!(inhibitory.weight > 1.0);
    }

    // Metadata Management Tests
    #[test]
    fn test_metadata_management() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Add metadata
        relationship.metadata.insert("source_concept".to_string(), "dog".to_string());
        relationship.metadata.insert("target_concept".to_string(), "animal".to_string());
        relationship.metadata.insert("confidence".to_string(), "0.9".to_string());
        relationship.metadata.insert("discovered_at".to_string(), "2024-01-01".to_string());
        
        assert_eq!(relationship.metadata.len(), 4);
        assert_eq!(relationship.metadata.get("source_concept").unwrap(), "dog");
        assert_eq!(relationship.metadata.get("target_concept").unwrap(), "animal");
        assert_eq!(relationship.metadata.get("confidence").unwrap(), "0.9");
        assert_eq!(relationship.metadata.get("discovered_at").unwrap(), "2024-01-01");
    }

    #[test]
    fn test_complex_metadata() {
        let relationship = RelationshipTestFixture::create_relationship_with_metadata();
        
        assert_eq!(relationship.metadata.len(), 4);
        assert!(relationship.metadata.contains_key("source_concept"));
        assert!(relationship.metadata.contains_key("target_concept"));
        assert!(relationship.metadata.contains_key("confidence"));
        assert!(relationship.metadata.contains_key("learned_from"));
    }

    #[test]
    fn test_metadata_modification() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Add metadata
        relationship.metadata.insert("version".to_string(), "1.0".to_string());
        assert_eq!(relationship.metadata.get("version").unwrap(), "1.0");
        
        // Update metadata
        relationship.metadata.insert("version".to_string(), "2.0".to_string());
        assert_eq!(relationship.metadata.get("version").unwrap(), "2.0");
        
        // Remove metadata
        relationship.metadata.remove("version");
        assert!(relationship.metadata.get("version").is_none());
    }

    // Usage Tracking Tests
    #[test]
    fn test_usage_tracking() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        assert_eq!(relationship.activation_count, 0);
        assert_eq!(relationship.usage_count, 0);
        
        // Strengthening should update both counters
        for i in 1..=5 {
            relationship.strengthen(0.01);
            assert_eq!(relationship.activation_count, i);
            assert_eq!(relationship.usage_count, i);
        }
    }

    #[test]
    fn test_usage_consistency() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Both counters should always be equal after strengthening
        for _ in 0..100 {
            relationship.strengthen(0.001);
            assert_eq!(relationship.activation_count, relationship.usage_count);
        }
    }

    // Serialization Tests
    #[test]
    fn test_relationship_serialization() {
        let relationship = RelationshipTestFixture::create_relationship_with_metadata();
        
        // Test serialization to JSON
        let serialized = serde_json::to_string(&relationship);
        assert!(serialized.is_ok(), "Relationship should be serializable");
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("RelatedTo"));
        assert!(json_str.contains("animal"));
        assert!(json_str.contains("mammal"));
    }

    #[test]
    fn test_relationship_deserialization() {
        let relationship = RelationshipTestFixture::create_relationship_with_metadata();
        
        // Serialize and deserialize
        let serialized = serde_json::to_string(&relationship).unwrap();
        let deserialized: Result<BrainInspiredRelationship, _> = serde_json::from_str(&serialized);
        
        assert!(deserialized.is_ok(), "Relationship should be deserializable");
        
        let restored_relationship = deserialized.unwrap();
        assert_eq!(restored_relationship.relation_type, relationship.relation_type);
        assert_eq!(restored_relationship.weight, relationship.weight);
        assert_eq!(restored_relationship.strength, relationship.strength);
        assert_eq!(restored_relationship.is_inhibitory, relationship.is_inhibitory);
        assert_eq!(restored_relationship.metadata.len(), relationship.metadata.len());
    }

    // Edge Cases and Error Handling
    #[test]
    fn test_zero_learning_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_weight = relationship.weight;
        relationship.strengthen(0.0);
        
        assert_eq!(relationship.weight, initial_weight, "Zero learning rate should not change weight");
        assert_eq!(relationship.activation_count, 1, "Should still increment counters");
    }

    #[test]
    fn test_negative_learning_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_weight = relationship.weight;
        relationship.strengthen(-0.1);
        
        assert!(relationship.weight < initial_weight, "Negative learning rate should decrease weight");
        assert!(relationship.weight >= 0.0, "Weight should not go below 0");
    }

    #[test]
    fn test_large_learning_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.strengthen(10.0); // Very large learning rate
        assert_eq!(relationship.weight, 1.0, "Should clamp to maximum weight");
    }

    #[test]
    fn test_zero_decay_rate() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 0.8;
        relationship.temporal_decay = 0.0;
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1000);
        
        let result = relationship.apply_decay();
        assert_eq!(result, 0.8, "Zero decay rate should preserve weight");
    }

    #[test]
    fn test_extreme_decay_rates() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.weight = 1.0;
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        
        // Very small decay rate
        relationship.temporal_decay = f32::MIN_POSITIVE;
        let result = relationship.apply_decay();
        assert!(result > 0.99, "Very small decay should barely affect weight");
        
        // Very large decay rate
        relationship.weight = 1.0;
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        relationship.temporal_decay = f32::MAX;
        let result = relationship.apply_decay();
        assert!(result < 0.01, "Very large decay should nearly eliminate weight");
    }

    // Cloning and Copying Tests
    #[test]
    fn test_relationship_cloning() {
        let relationship = RelationshipTestFixture::create_relationship_with_metadata();
        let cloned_relationship = relationship.clone();
        
        assert_eq!(relationship.source, cloned_relationship.source);
        assert_eq!(relationship.target, cloned_relationship.target);
        assert_eq!(relationship.relation_type, cloned_relationship.relation_type);
        assert_eq!(relationship.weight, cloned_relationship.weight);
        assert_eq!(relationship.strength, cloned_relationship.strength);
        assert_eq!(relationship.is_inhibitory, cloned_relationship.is_inhibitory);
        assert_eq!(relationship.temporal_decay, cloned_relationship.temporal_decay);
        assert_eq!(relationship.activation_count, cloned_relationship.activation_count);
        assert_eq!(relationship.usage_count, cloned_relationship.usage_count);
        assert_eq!(relationship.metadata.len(), cloned_relationship.metadata.len());
    }

    // Performance Tests
    #[test]
    fn test_strengthening_performance() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            relationship.strengthen(0.0001);
            if relationship.weight >= 1.0 {
                relationship.weight = 0.5; // Reset to prevent saturation
                relationship.strength = 0.5;
            }
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "10000 strengthen operations should complete quickly");
    }

    #[test]
    fn test_decay_performance() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            relationship.apply_decay();
            // Reset weight occasionally to prevent it from going to zero
            if relationship.weight < 0.1 {
                relationship.weight = 1.0;
                relationship.strength = 1.0;
            }
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "10000 decay operations should complete quickly");
    }

    #[test]
    fn test_metadata_access_performance() {
        let relationship = RelationshipTestFixture::create_relationship_with_metadata();
        
        let start = std::time::Instant::now();
        for _ in 0..100000 {
            let _ = relationship.metadata.get("source_concept");
            let _ = relationship.metadata.get("target_concept");
            let _ = relationship.metadata.get("confidence");
            let _ = relationship.metadata.get("nonexistent");
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "Metadata access should be fast");
    }

    // Boundary Value Tests
    #[test]
    fn test_weight_boundaries() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Test minimum weight
        relationship.weight = 0.0;
        relationship.strength = 0.0;
        relationship.strengthen(0.1);
        assert_eq!(relationship.weight, 0.1);
        
        // Test maximum weight
        relationship.weight = 1.0;
        relationship.strength = 1.0;
        relationship.strengthen(0.1);
        assert_eq!(relationship.weight, 1.0, "Should clamp at maximum");
        
        // Test beyond maximum
        relationship.weight = 0.95;
        relationship.strength = 0.95;
        relationship.strengthen(0.1);
        assert_eq!(relationship.weight, 1.0, "Should clamp at maximum");
    }

    #[test]
    fn test_counter_boundaries() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Test counter overflow protection (in practice, unlikely to overflow u64)
        relationship.activation_count = u64::MAX - 1;
        relationship.usage_count = u64::MAX - 1;
        
        relationship.strengthen(0.01);
        
        assert_eq!(relationship.activation_count, u64::MAX);
        assert_eq!(relationship.usage_count, u64::MAX);
    }

    // State Consistency Tests
    #[test]
    fn test_weight_strength_consistency() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // After any operation, weight and strength should be equal
        for _ in 0..10 {
            relationship.strengthen(0.05);
            assert_eq!(relationship.weight, relationship.strength, 
                      "Weight and strength should always be equal");
        }
        
        for _ in 0..5 {
            relationship.apply_decay();
            assert_eq!(relationship.weight, relationship.strength, 
                      "Weight and strength should always be equal after decay");
        }
    }

    #[test]
    fn test_temporal_consistency() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        let initial_strengthened = relationship.last_strengthened;
        let initial_update = relationship.last_update;
        
        // Small delay
        thread::sleep(Duration::from_millis(10));
        relationship.strengthen(0.1);
        
        // Times should be updated and consistent
        assert!(relationship.last_strengthened > initial_strengthened);
        assert!(relationship.last_update >= relationship.last_strengthened);
        
        thread::sleep(Duration::from_millis(10));
        relationship.apply_decay();
        
        // Update time should be more recent than strengthened time
        assert!(relationship.last_update > relationship.last_strengthened);
    }

    // Integration Tests
    #[test]
    fn test_strengthen_decay_cycle() {
        let mut relationship = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            RelationType::RelatedTo
        );
        
        // Repeated strengthen-decay cycles
        for _ in 0..5 {
            relationship.strengthen(0.2);
            thread::sleep(Duration::from_millis(10));
            relationship.apply_decay();
        }
        
        assert!(relationship.weight > 0.0, "Should maintain some weight after cycles");
        assert!(relationship.activation_count > 0, "Should track activations");
        assert_eq!(relationship.weight, relationship.strength, "Should maintain consistency");
    }

    // Memory and Resource Tests
    #[test]
    fn test_memory_usage() {
        // Create many relationships to test memory usage patterns
        let mut relationships = Vec::new();
        
        for i in 0..1000 {
            let mut relationship = BrainInspiredRelationship::new(
                EntityKey::default(),
                EntityKey::default(),
                match i % 9 {
                    0 => RelationType::IsA,
                    1 => RelationType::HasInstance,
                    2 => RelationType::HasProperty,
                    3 => RelationType::RelatedTo,
                    4 => RelationType::PartOf,
                    5 => RelationType::Similar,
                    6 => RelationType::Opposite,
                    7 => RelationType::Temporal,
                    _ => RelationType::Learned,
                }
            );
            
            // Add some metadata
            relationship.metadata.insert("id".to_string(), i.to_string());
            relationship.strengthen(0.01);
            
            relationships.push(relationship);
        }
        
        assert_eq!(relationships.len(), 1000);
        
        // Test accessing random relationships
        for _ in 0..100 {
            let idx = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % 1000) as usize;
            let relationship = &relationships[idx];
            assert!(relationship.weight >= 1.0);
            assert!(relationship.activation_count > 0);
        }
    }
}