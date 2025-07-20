//! Comprehensive tests for ActivationPattern and ActivationStep functionality
//! 
//! This module provides exhaustive testing for activation patterns including:
//! - ActivationPattern creation and management
//! - Top activation retrieval and sorting
//! - ActivationStep creation and tracking
//! - ActivationOperation enum functionality
//! - Temporal aspects and timestamps
//! - Edge cases and boundary conditions
//! - Performance characteristics

use crate::core::brain_types::{
    ActivationPattern, ActivationStep, ActivationOperation
};
use crate::core::types::EntityKey;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};

/// Test fixture for creating various activation patterns and steps
pub struct ActivationPatternTestFixture;

impl ActivationPatternTestFixture {
    pub fn create_simple_pattern() -> ActivationPattern {
        let mut pattern = ActivationPattern::new("test query".to_string());
        
        // Add some basic activations
        pattern.activations.insert(EntityKey::default(), 0.8);
        pattern.activations.insert(EntityKey::default(), 0.6);
        pattern.activations.insert(EntityKey::default(), 0.9);
        
        pattern
    }
    
    pub fn create_large_pattern() -> ActivationPattern {
        let mut pattern = ActivationPattern::new("large query test".to_string());
        
        // Add many activations with varying strengths
        for i in 0..100 {
            let activation = (i as f32) / 100.0;
            pattern.activations.insert(EntityKey::default(), activation);
        }
        
        pattern
    }
    
    pub fn create_pattern_with_duplicates() -> ActivationPattern {
        let mut pattern = ActivationPattern::new("duplicate test".to_string());
        
        // Add activations with same values
        pattern.activations.insert(EntityKey::default(), 0.5);
        pattern.activations.insert(EntityKey::default(), 0.8);
        pattern.activations.insert(EntityKey::default(), 0.5);
        pattern.activations.insert(EntityKey::default(), 0.9);
        pattern.activations.insert(EntityKey::default(), 0.5);
        
        pattern
    }
    
    pub fn create_activation_step(
        step_id: usize,
        concept_id: String,
        activation_level: f32,
        operation_type: ActivationOperation
    ) -> ActivationStep {
        ActivationStep {
            step_id,
            entity_key: EntityKey::default(),
            concept_id,
            activation_level,
            operation_type,
            timestamp: SystemTime::now(),
        }
    }
    
    pub fn create_activation_sequence() -> Vec<ActivationStep> {
        vec![
            Self::create_activation_step(1, "animal".to_string(), 0.8, ActivationOperation::Initialize),
            Self::create_activation_step(2, "mammal".to_string(), 0.7, ActivationOperation::Propagate),
            Self::create_activation_step(3, "dog".to_string(), 0.9, ActivationOperation::Propagate),
            Self::create_activation_step(4, "cat".to_string(), 0.1, ActivationOperation::Inhibit),
            Self::create_activation_step(5, "dog".to_string(), 0.95, ActivationOperation::Reinforce),
            Self::create_activation_step(6, "old_concept".to_string(), 0.2, ActivationOperation::Decay),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ActivationPattern Creation Tests
    #[test]
    fn test_activation_pattern_creation() {
        let pattern = ActivationPattern::new("test query".to_string());
        
        assert_eq!(pattern.query, "test query");
        assert!(pattern.activations.is_empty());
        
        // Timestamp should be recent
        let now = SystemTime::now();
        let time_diff = now.duration_since(pattern.timestamp).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "Timestamp should be recent");
    }

    #[test]
    fn test_activation_pattern_with_empty_query() {
        let pattern = ActivationPattern::new("".to_string());
        assert_eq!(pattern.query, "");
        assert!(pattern.activations.is_empty());
    }

    #[test]
    fn test_activation_pattern_with_long_query() {
        let long_query = "a".repeat(10000);
        let pattern = ActivationPattern::new(long_query.clone());
        assert_eq!(pattern.query, long_query);
        assert_eq!(pattern.query.len(), 10000);
    }

    #[test]
    fn test_activation_pattern_with_unicode_query() {
        let unicode_query = "What is 动物 🐕 αβγ?".to_string();
        let pattern = ActivationPattern::new(unicode_query.clone());
        assert_eq!(pattern.query, unicode_query);
    }

    // Activation Management Tests
    #[test]
    fn test_adding_activations() {
        let mut pattern = ActivationPattern::new("test".to_string());
        
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let key3 = EntityKey::default();
        
        pattern.activations.insert(key1, 0.8);
        pattern.activations.insert(key2, 0.6);
        pattern.activations.insert(key3, 0.9);
        
        assert_eq!(pattern.activations.len(), 3);
        assert_eq!(pattern.activations.get(&key1), Some(&0.8));
        assert_eq!(pattern.activations.get(&key2), Some(&0.6));
        assert_eq!(pattern.activations.get(&key3), Some(&0.9));
    }

    #[test]
    fn test_updating_activations() {
        let mut pattern = ActivationPattern::new("test".to_string());
        
        let key = EntityKey::default();
        pattern.activations.insert(key, 0.5);
        assert_eq!(pattern.activations.get(&key), Some(&0.5));
        
        // Update the same key
        pattern.activations.insert(key, 0.8);
        assert_eq!(pattern.activations.get(&key), Some(&0.8));
        assert_eq!(pattern.activations.len(), 1);
    }

    #[test]
    fn test_removing_activations() {
        let mut pattern = ActivationPattern::new("test".to_string());
        
        let key = EntityKey::default();
        pattern.activations.insert(key, 0.7);
        assert_eq!(pattern.activations.len(), 1);
        
        pattern.activations.remove(&key);
        assert_eq!(pattern.activations.len(), 0);
        assert_eq!(pattern.activations.get(&key), None);
    }

    // Top Activations Tests
    #[test]
    fn test_get_top_activations_basic() {
        let mut pattern = ActivationPattern::new("test".to_string());
        
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let key3 = EntityKey::default();
        
        pattern.activations.insert(key1, 0.8);
        pattern.activations.insert(key2, 0.6);
        pattern.activations.insert(key3, 0.9);
        
        let top_2 = pattern.get_top_activations(2);
        assert_eq!(top_2.len(), 2);
        
        // Should be sorted in descending order
        assert_eq!(top_2[0].1, 0.9); // Highest activation
        assert_eq!(top_2[1].1, 0.8); // Second highest
    }

    #[test]
    fn test_get_top_activations_more_than_available() {
        let pattern = ActivationPatternTestFixture::create_simple_pattern();
        
        let top_10 = pattern.get_top_activations(10);
        assert_eq!(top_10.len(), 3, "Should return only available activations");
        
        // Should be sorted in descending order
        assert!(top_10[0].1 >= top_10[1].1, "Should be sorted descending");
        assert!(top_10[1].1 >= top_10[2].1, "Should be sorted descending");
    }

    #[test]
    fn test_get_top_activations_empty_pattern() {
        let pattern = ActivationPattern::new("empty".to_string());
        
        let top_5 = pattern.get_top_activations(5);
        assert!(top_5.is_empty(), "Empty pattern should return empty results");
    }

    #[test]
    fn test_get_top_activations_zero_request() {
        let pattern = ActivationPatternTestFixture::create_simple_pattern();
        
        let top_0 = pattern.get_top_activations(0);
        assert!(top_0.is_empty(), "Zero request should return empty results");
    }

    #[test]
    fn test_get_top_activations_sorting() {
        let pattern = ActivationPatternTestFixture::create_large_pattern();
        
        let top_10 = pattern.get_top_activations(10);
        assert_eq!(top_10.len(), 10);
        
        // Verify strict descending order
        for i in 1..top_10.len() {
            assert!(top_10[i-1].1 >= top_10[i].1, 
                   "Activations should be in descending order: {} >= {}", 
                   top_10[i-1].1, top_10[i].1);
        }
    }

    #[test]
    fn test_get_top_activations_with_duplicates() {
        let pattern = ActivationPatternTestFixture::create_pattern_with_duplicates();
        
        let top_all = pattern.get_top_activations(10);
        assert_eq!(top_all.len(), 5); // Should have 5 activations total
        
        // Should be sorted, with duplicates handled consistently
        assert_eq!(top_all[0].1, 0.9); // Highest
        assert_eq!(top_all[1].1, 0.8); // Second highest
        
        // The remaining should all be 0.5 (duplicates)
        for i in 2..top_all.len() {
            assert_eq!(top_all[i].1, 0.5);
        }
    }

    #[test]
    fn test_get_top_activations_boundary_values() {
        let mut pattern = ActivationPattern::new("boundary".to_string());
        
        // Add boundary activation values
        pattern.activations.insert(EntityKey::default(), 0.0);
        pattern.activations.insert(EntityKey::default(), 1.0);
        pattern.activations.insert(EntityKey::default(), 0.5);
        pattern.activations.insert(EntityKey::default(), f32::MIN_POSITIVE);
        pattern.activations.insert(EntityKey::default(), 1.0 - f32::EPSILON);
        
        let top_all = pattern.get_top_activations(10);
        assert_eq!(top_all.len(), 5);
        
        // Should handle boundary values correctly
        assert_eq!(top_all[0].1, 1.0);
        assert!((top_all[1].1 - (1.0 - f32::EPSILON)).abs() < f32::EPSILON);
        assert_eq!(top_all[2].1, 0.5);
        assert_eq!(top_all[3].1, f32::MIN_POSITIVE);
        assert_eq!(top_all[4].1, 0.0);
    }

    #[test]
    fn test_get_top_activations_special_float_values() {
        let mut pattern = ActivationPattern::new("special".to_string());
        
        // Add special float values
        pattern.activations.insert(EntityKey::default(), f32::INFINITY);
        pattern.activations.insert(EntityKey::default(), f32::NEG_INFINITY);
        pattern.activations.insert(EntityKey::default(), f32::NAN);
        pattern.activations.insert(EntityKey::default(), 0.5);
        
        let top_all = pattern.get_top_activations(10);
        // Should handle special values gracefully (implementation-dependent behavior)
        assert_eq!(top_all.len(), 4);
    }

    // ActivationStep Tests
    #[test]
    fn test_activation_step_creation() {
        let step = ActivationPatternTestFixture::create_activation_step(
            1,
            "test_concept".to_string(),
            0.8,
            ActivationOperation::Initialize
        );
        
        assert_eq!(step.step_id, 1);
        assert_eq!(step.concept_id, "test_concept");
        assert_eq!(step.activation_level, 0.8);
        assert_eq!(step.operation_type, ActivationOperation::Initialize);
        
        // Timestamp should be recent
        let now = SystemTime::now();
        let time_diff = now.duration_since(step.timestamp).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "Timestamp should be recent");
    }

    #[test]
    fn test_activation_step_all_operations() {
        let operations = vec![
            ActivationOperation::Initialize,
            ActivationOperation::Propagate,
            ActivationOperation::Inhibit,
            ActivationOperation::Reinforce,
            ActivationOperation::Decay,
        ];
        
        for (i, operation) in operations.iter().enumerate() {
            let step = ActivationStep {
                step_id: i,
                entity_key: EntityKey::default(),
                concept_id: format!("concept_{}", i),
                activation_level: (i as f32) / 10.0,
                operation_type: *operation,
                timestamp: SystemTime::now(),
            };
            
            assert_eq!(step.step_id, i);
            assert_eq!(step.concept_id, format!("concept_{}", i));
            assert_eq!(step.operation_type, *operation);
        }
    }

    #[test]
    fn test_activation_step_boundary_values() {
        // Test with extreme activation levels
        let step_zero = ActivationStep {
            step_id: 0,
            entity_key: EntityKey::default(),
            concept_id: "zero".to_string(),
            activation_level: 0.0,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        };
        assert_eq!(step_zero.activation_level, 0.0);
        
        let step_max = ActivationStep {
            step_id: 1,
            entity_key: EntityKey::default(),
            concept_id: "max".to_string(),
            activation_level: 1.0,
            operation_type: ActivationOperation::Reinforce,
            timestamp: SystemTime::now(),
        };
        assert_eq!(step_max.activation_level, 1.0);
        
        let step_negative = ActivationStep {
            step_id: 2,
            entity_key: EntityKey::default(),
            concept_id: "negative".to_string(),
            activation_level: -0.5,
            operation_type: ActivationOperation::Inhibit,
            timestamp: SystemTime::now(),
        };
        assert_eq!(step_negative.activation_level, -0.5);
    }

    #[test]
    fn test_activation_step_empty_concept() {
        let step = ActivationStep {
            step_id: 0,
            entity_key: EntityKey::default(),
            concept_id: "".to_string(),
            activation_level: 0.5,
            operation_type: ActivationOperation::Propagate,
            timestamp: SystemTime::now(),
        };
        
        assert_eq!(step.concept_id, "");
        assert_eq!(step.activation_level, 0.5);
    }

    // ActivationOperation Tests
    #[test]
    fn test_activation_operation_equality() {
        assert_eq!(ActivationOperation::Initialize, ActivationOperation::Initialize);
        assert_eq!(ActivationOperation::Propagate, ActivationOperation::Propagate);
        assert_eq!(ActivationOperation::Inhibit, ActivationOperation::Inhibit);
        assert_eq!(ActivationOperation::Reinforce, ActivationOperation::Reinforce);
        assert_eq!(ActivationOperation::Decay, ActivationOperation::Decay);
        
        assert_ne!(ActivationOperation::Initialize, ActivationOperation::Propagate);
        assert_ne!(ActivationOperation::Inhibit, ActivationOperation::Decay);
    }

    #[test]
    fn test_activation_operation_debug() {
        // Test that debug formatting works
        let ops = vec![
            ActivationOperation::Initialize,
            ActivationOperation::Propagate,
            ActivationOperation::Inhibit,
            ActivationOperation::Reinforce,
            ActivationOperation::Decay,
        ];
        
        for op in ops {
            let debug_str = format!("{:?}", op);
            assert!(!debug_str.is_empty(), "Debug format should not be empty");
        }
    }

    // Sequence and Timeline Tests
    #[test]
    fn test_activation_sequence() {
        let sequence = ActivationPatternTestFixture::create_activation_sequence();
        
        assert_eq!(sequence.len(), 6);
        
        // Verify sequence order
        for i in 0..sequence.len() {
            assert_eq!(sequence[i].step_id, i + 1);
        }
        
        // Verify operation types
        assert_eq!(sequence[0].operation_type, ActivationOperation::Initialize);
        assert_eq!(sequence[1].operation_type, ActivationOperation::Propagate);
        assert_eq!(sequence[2].operation_type, ActivationOperation::Propagate);
        assert_eq!(sequence[3].operation_type, ActivationOperation::Inhibit);
        assert_eq!(sequence[4].operation_type, ActivationOperation::Reinforce);
        assert_eq!(sequence[5].operation_type, ActivationOperation::Decay);
        
        // Verify concept progression
        assert_eq!(sequence[0].concept_id, "animal");
        assert_eq!(sequence[1].concept_id, "mammal");
        assert_eq!(sequence[2].concept_id, "dog");
        assert_eq!(sequence[3].concept_id, "cat");
        assert_eq!(sequence[4].concept_id, "dog"); // Reinforcement
        assert_eq!(sequence[5].concept_id, "old_concept");
    }

    #[test]
    fn test_temporal_ordering() {
        let mut steps = Vec::new();
        
        // Create steps with small delays
        for i in 0..5 {
            let step = ActivationStep {
                step_id: i,
                entity_key: EntityKey::default(),
                concept_id: format!("concept_{}", i),
                activation_level: 0.5,
                operation_type: ActivationOperation::Propagate,
                timestamp: SystemTime::now(),
            };
            steps.push(step);
            
            // Small delay to ensure different timestamps
            std::thread::sleep(Duration::from_millis(1));
        }
        
        // Verify temporal ordering
        for i in 1..steps.len() {
            assert!(steps[i].timestamp >= steps[i-1].timestamp, 
                   "Timestamps should be in chronological order");
        }
    }

    // Serialization Tests
    #[test]
    fn test_activation_pattern_serialization() {
        let pattern = ActivationPatternTestFixture::create_simple_pattern();
        
        let serialized = serde_json::to_string(&pattern);
        assert!(serialized.is_ok(), "ActivationPattern should be serializable");
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("test query"));
        assert!(json_str.contains("activations"));
        assert!(json_str.contains("timestamp"));
    }

    #[test]
    fn test_activation_pattern_deserialization() {
        let pattern = ActivationPatternTestFixture::create_simple_pattern();
        
        let serialized = serde_json::to_string(&pattern).unwrap();
        let deserialized: Result<ActivationPattern, _> = serde_json::from_str(&serialized);
        
        assert!(deserialized.is_ok(), "ActivationPattern should be deserializable");
        
        let restored_pattern = deserialized.unwrap();
        assert_eq!(restored_pattern.query, pattern.query);
        assert_eq!(restored_pattern.activations.len(), pattern.activations.len());
    }

    #[test]
    fn test_activation_step_serialization() {
        let step = ActivationPatternTestFixture::create_activation_step(
            1,
            "test".to_string(),
            0.8,
            ActivationOperation::Initialize
        );
        
        let serialized = serde_json::to_string(&step);
        assert!(serialized.is_ok(), "ActivationStep should be serializable");
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("test"));
        assert!(json_str.contains("Initialize"));
    }

    #[test]
    fn test_activation_operation_serialization() {
        let operations = vec![
            ActivationOperation::Initialize,
            ActivationOperation::Propagate,
            ActivationOperation::Inhibit,
            ActivationOperation::Reinforce,
            ActivationOperation::Decay,
        ];
        
        for operation in operations {
            let serialized = serde_json::to_string(&operation);
            assert!(serialized.is_ok(), "ActivationOperation should be serializable");
            
            let deserialized: Result<ActivationOperation, _> = serde_json::from_str(&serialized.unwrap());
            assert!(deserialized.is_ok(), "ActivationOperation should be deserializable");
            assert_eq!(deserialized.unwrap(), operation);
        }
    }

    // Performance Tests
    #[test]
    fn test_top_activations_performance() {
        let pattern = ActivationPatternTestFixture::create_large_pattern();
        
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = pattern.get_top_activations(10);
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "Top activations should be computed quickly");
    }

    #[test]
    fn test_large_pattern_creation() {
        let start = std::time::Instant::now();
        
        let mut pattern = ActivationPattern::new("performance test".to_string());
        for i in 0..10000 {
            pattern.activations.insert(EntityKey::default(), (i as f32) / 10000.0);
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000, "Large pattern creation should be fast");
        assert_eq!(pattern.activations.len(), 10000);
    }

    #[test]
    fn test_activation_step_creation_performance() {
        let start = std::time::Instant::now();
        
        let mut steps = Vec::new();
        for i in 0..10000 {
            let step = ActivationStep {
                step_id: i,
                entity_key: EntityKey::default(),
                concept_id: format!("concept_{}", i),
                activation_level: (i as f32) / 10000.0,
                operation_type: match i % 5 {
                    0 => ActivationOperation::Initialize,
                    1 => ActivationOperation::Propagate,
                    2 => ActivationOperation::Inhibit,
                    3 => ActivationOperation::Reinforce,
                    _ => ActivationOperation::Decay,
                },
                timestamp: SystemTime::now(),
            };
            steps.push(step);
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000, "Creating many activation steps should be fast");
        assert_eq!(steps.len(), 10000);
    }

    // Edge Cases and Robustness Tests
    #[test]
    fn test_pattern_with_many_zero_activations() {
        let mut pattern = ActivationPattern::new("zeros".to_string());
        
        // Add many zero activations
        for _ in 0..1000 {
            pattern.activations.insert(EntityKey::default(), 0.0);
        }
        
        // Add a few non-zero activations
        pattern.activations.insert(EntityKey::default(), 0.1);
        pattern.activations.insert(EntityKey::default(), 0.05);
        
        let top_5 = pattern.get_top_activations(5);
        assert_eq!(top_5.len(), 5);
        assert_eq!(top_5[0].1, 0.1);
        assert_eq!(top_5[1].1, 0.05);
        
        // Rest should be zeros
        for i in 2..5 {
            assert_eq!(top_5[i].1, 0.0);
        }
    }

    #[test]
    fn test_pattern_with_identical_activations() {
        let mut pattern = ActivationPattern::new("identical".to_string());
        
        // Add many identical activations
        for _ in 0..100 {
            pattern.activations.insert(EntityKey::default(), 0.5);
        }
        
        let top_10 = pattern.get_top_activations(10);
        assert_eq!(top_10.len(), 10);
        
        // All should have the same activation value
        for activation in top_10 {
            assert_eq!(activation.1, 0.5);
        }
    }

    #[test]
    fn test_empty_pattern_operations() {
        let pattern = ActivationPattern::new("empty".to_string());
        
        assert!(pattern.activations.is_empty());
        assert_eq!(pattern.get_top_activations(1).len(), 0);
        assert_eq!(pattern.get_top_activations(100).len(), 0);
    }

    #[test]
    fn test_single_activation_pattern() {
        let mut pattern = ActivationPattern::new("single".to_string());
        pattern.activations.insert(EntityKey::default(), 0.7);
        
        let top_1 = pattern.get_top_activations(1);
        assert_eq!(top_1.len(), 1);
        assert_eq!(top_1[0].1, 0.7);
        
        let top_5 = pattern.get_top_activations(5);
        assert_eq!(top_5.len(), 1);
        assert_eq!(top_5[0].1, 0.7);
    }

    // Memory and Resource Tests
    #[test]
    fn test_memory_usage_patterns() {
        // Test memory usage with various pattern sizes
        let mut patterns = Vec::new();
        
        for size in [10, 100, 1000] {
            let mut pattern = ActivationPattern::new(format!("pattern_{}", size));
            
            for i in 0..size {
                pattern.activations.insert(EntityKey::default(), (i as f32) / (size as f32));
            }
            
            patterns.push(pattern);
        }
        
        // Verify patterns were created correctly
        assert_eq!(patterns[0].activations.len(), 10);
        assert_eq!(patterns[1].activations.len(), 100);
        assert_eq!(patterns[2].activations.len(), 1000);
        
        // Test accessing patterns
        for pattern in &patterns {
            let top_5 = pattern.get_top_activations(5);
            assert!(top_5.len() <= 5);
            assert!(top_5.len() <= pattern.activations.len());
        }
    }

    #[test]
    fn test_activation_step_memory_usage() {
        let mut steps = Vec::new();
        
        // Create a large sequence of steps
        for i in 0..10000 {
            let step = ActivationStep {
                step_id: i,
                entity_key: EntityKey::default(),
                concept_id: if i % 100 == 0 { 
                    format!("important_concept_{}", i / 100) 
                } else { 
                    format!("concept_{}", i) 
                },
                activation_level: (i as f32) / 10000.0,
                operation_type: match i % 5 {
                    0 => ActivationOperation::Initialize,
                    1 => ActivationOperation::Propagate,
                    2 => ActivationOperation::Inhibit,
                    3 => ActivationOperation::Reinforce,
                    _ => ActivationOperation::Decay,
                },
                timestamp: SystemTime::now(),
            };
            steps.push(step);
        }
        
        assert_eq!(steps.len(), 10000);
        
        // Test accessing random steps
        for _ in 0..100 {
            let idx = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % 10000) as usize;
            let step = &steps[idx];
            assert_eq!(step.step_id, idx);
            assert!(step.activation_level >= 0.0);
            assert!(step.activation_level <= 1.0);
        }
    }

    // Integration Tests
    #[test]
    fn test_pattern_step_integration() {
        // Create a pattern and corresponding steps
        let mut pattern = ActivationPattern::new("integration test".to_string());
        let mut steps = Vec::new();
        
        let concepts = ["animal", "mammal", "dog", "cat", "fish"];
        
        for (i, concept) in concepts.iter().enumerate() {
            let activation = (i + 1) as f32 / concepts.len() as f32;
            
            // Add to pattern
            pattern.activations.insert(EntityKey::default(), activation);
            
            // Create corresponding step
            let step = ActivationStep {
                step_id: i,
                entity_key: EntityKey::default(),
                concept_id: concept.to_string(),
                activation_level: activation,
                operation_type: if i == 0 {
                    ActivationOperation::Initialize
                } else {
                    ActivationOperation::Propagate
                },
                timestamp: SystemTime::now(),
            };
            steps.push(step);
        }
        
        // Verify pattern and steps are consistent
        assert_eq!(pattern.activations.len(), concepts.len());
        assert_eq!(steps.len(), concepts.len());
        
        let top_activations = pattern.get_top_activations(concepts.len());
        assert_eq!(top_activations.len(), concepts.len());
        
        // Top activation should correspond to last concept (highest activation)
        assert_eq!(top_activations[0].1, 1.0);
        assert_eq!(steps.last().unwrap().activation_level, 1.0);
    }
}