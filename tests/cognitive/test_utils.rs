/// Shared test utilities for cognitive module tests
use llmkg::core::types::{EntityKey, EntityData};
use llmkg::cognitive::working_memory::{MemoryContent, MemoryItem};
use slotmap::SlotMap;
use std::time::Instant;

/// Creates a set of unique EntityKeys for testing
pub fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    let mut keys = Vec::new();
    
    for i in 0..count {
        let key = sm.insert(EntityData {
            type_id: 1,
            properties: format!("test_entity_{}", i),
            embedding: vec![0.0; 64],
        });
        keys.push(key);
    }
    
    keys
}

/// Creates test memory items with specified activation levels
pub fn create_test_memory_items(activation_levels: Vec<f32>) -> Vec<MemoryItem> {
    activation_levels.into_iter()
        .enumerate()
        .map(|(i, activation)| MemoryItem {
            content: MemoryContent::Concept(format!("test_concept_{}", i)),
            activation_level: activation,
            timestamp: Instant::now(),
            importance_score: activation * 0.9,
            access_count: 1,
            decay_factor: 0.1,
        })
        .collect()
}

/// Creates a memory item with specific properties
pub fn create_memory_item(
    content: &str,
    activation_level: f32,
    importance_score: f32,
    access_count: usize,
) -> MemoryItem {
    MemoryItem {
        content: MemoryContent::Concept(content.to_string()),
        activation_level,
        timestamp: Instant::now(),
        importance_score,
        access_count,
        decay_factor: 0.1,
    }
}

/// Test fixture for measuring performance
pub struct PerformanceTimer {
    start: Instant,
    operation: String,
}

impl PerformanceTimer {
    pub fn new(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
        }
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    pub fn assert_within_ms(&self, max_ms: f64) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed <= max_ms,
            "{} took {:.2}ms, expected less than {:.2}ms",
            self.operation,
            elapsed,
            max_ms
        );
    }
}

/// Generates a set of test scenarios for attention testing
pub mod scenarios {
    use super::*;
    
    pub struct AttentionScenario {
        pub name: String,
        pub targets: Vec<EntityKey>,
        pub expected_focus_count: usize,
        pub cognitive_load: f32,
    }
    
    pub fn generate_scenarios() -> Vec<AttentionScenario> {
        vec![
            AttentionScenario {
                name: "Single target focus".to_string(),
                targets: create_test_entity_keys(1),
                expected_focus_count: 1,
                cognitive_load: 0.2,
            },
            AttentionScenario {
                name: "Moderate multi-target".to_string(),
                targets: create_test_entity_keys(3),
                expected_focus_count: 3,
                cognitive_load: 0.5,
            },
            AttentionScenario {
                name: "High cognitive load".to_string(),
                targets: create_test_entity_keys(5),
                expected_focus_count: 3, // System should limit focus under high load
                cognitive_load: 0.8,
            },
            AttentionScenario {
                name: "Overload scenario".to_string(),
                targets: create_test_entity_keys(10),
                expected_focus_count: 2, // Severe limitation under overload
                cognitive_load: 0.95,
            },
        ]
    }
}