//! Test data fixtures for consistent testing across all test types

use crate::core::types::EntityKey;
use crate::cognitive::types::CognitivePatternType;
use std::collections::HashMap;

/// Standard test queries for different cognitive patterns
pub struct TestQueries {
    pub factual: Vec<&'static str>,
    pub creative: Vec<&'static str>,
    pub analytical: Vec<&'static str>,
    pub relational: Vec<&'static str>,
}

impl TestQueries {
    pub fn new() -> Self {
        Self {
            factual: vec![
                "What is artificial intelligence?",
                "Define quantum computing",
                "What are the properties of water?",
                "How does photosynthesis work?",
            ],
            creative: vec![
                "Give me creative uses for a paperclip",
                "Brainstorm innovative energy solutions",
                "What are unusual applications of AI?",
                "Generate creative writing prompts",
            ],
            analytical: vec![
                "Analyze the pros and cons of renewable energy",
                "Compare different machine learning approaches",
                "Evaluate the impact of social media",
                "Assess the challenges of space exploration",
            ],
            relational: vec![
                "How does music relate to mathematics?",
                "What connections exist between art and science?",
                "How are emotions and memory connected?",
                "What is the relationship between diet and health?",
            ],
        }
    }
}

/// Standard test entities for attention and memory tests
pub fn create_standard_test_entities() -> Vec<EntityKey> {
    vec![
        EntityKey::from_hash("concept_ai"),
        EntityKey::from_hash("concept_ml"),
        EntityKey::from_hash("concept_quantum"),
        EntityKey::from_hash("concept_text_processing"),
        EntityKey::from_hash("concept_deep_learning"),
    ]
}

/// Performance test data sets
pub struct PerformanceTestData {
    pub small_dataset: Vec<EntityKey>,
    pub medium_dataset: Vec<EntityKey>,
    pub large_dataset: Vec<EntityKey>,
    pub stress_dataset: Vec<EntityKey>,
}

impl PerformanceTestData {
    pub fn new() -> Self {
        Self {
            small_dataset: (0..10).map(|i| EntityKey::from_hash(&format!("entity_{}", i))).collect(),
            medium_dataset: (0..100).map(|i| EntityKey::from_hash(&format!("entity_{}", i))).collect(),
            large_dataset: (0..1000).map(|i| EntityKey::from_hash(&format!("entity_{}", i))).collect(),
            stress_dataset: (0..10000).map(|i| EntityKey::from_hash(&format!("entity_{}", i))).collect(),
        }
    }
}

/// Test data for attention scores and weights
pub struct AttentionTestData {
    pub entity_attention_scores: HashMap<EntityKey, f32>,
    pub query_weights: HashMap<String, f32>,
    pub decay_factors: Vec<f32>,
}

impl AttentionTestData {
    pub fn new() -> Self {
        let mut entity_attention_scores = HashMap::new();
        entity_attention_scores.insert(EntityKey::from_hash("high_attention"), 0.95);
        entity_attention_scores.insert(EntityKey::from_hash("medium_attention"), 0.65);
        entity_attention_scores.insert(EntityKey::from_hash("low_attention"), 0.35);
        entity_attention_scores.insert(EntityKey::from_hash("minimal_attention"), 0.15);

        let mut query_weights = HashMap::new();
        query_weights.insert("factual_weight".to_string(), 0.8);
        query_weights.insert("creative_weight".to_string(), 0.6);
        query_weights.insert("analytical_weight".to_string(), 0.7);
        query_weights.insert("relational_weight".to_string(), 0.5);

        Self {
            entity_attention_scores,
            query_weights,
            decay_factors: vec![0.9, 0.8, 0.7, 0.6, 0.5],
        }
    }
}

/// Test data for pattern recognition
pub struct PatternTestData {
    pub pattern_sequences: Vec<Vec<CognitivePatternType>>,
    pub expected_transitions: HashMap<(CognitivePatternType, CognitivePatternType), f32>,
}

impl PatternTestData {
    pub fn new() -> Self {
        let pattern_sequences = vec![
            vec![CognitivePatternType::Convergent, CognitivePatternType::Divergent],
            vec![CognitivePatternType::Divergent, CognitivePatternType::Lateral],
            vec![CognitivePatternType::Lateral, CognitivePatternType::Convergent],
            vec![CognitivePatternType::Convergent, CognitivePatternType::Lateral, CognitivePatternType::Divergent],
        ];

        let mut expected_transitions = HashMap::new();
        expected_transitions.insert((CognitivePatternType::Convergent, CognitivePatternType::Divergent), 0.7);
        expected_transitions.insert((CognitivePatternType::Divergent, CognitivePatternType::Lateral), 0.6);
        expected_transitions.insert((CognitivePatternType::Lateral, CognitivePatternType::Convergent), 0.8);

        Self {
            pattern_sequences,
            expected_transitions,
        }
    }
}

/// Test data for edge cases and error conditions
pub struct EdgeCaseTestData {
    pub empty_queries: Vec<&'static str>,
    pub malformed_queries: Vec<&'static str>,
    pub extreme_values: Vec<f32>,
    pub unicode_test_strings: Vec<&'static str>,
}

impl EdgeCaseTestData {
    pub fn new() -> Self {
        Self {
            empty_queries: vec!["", "   ", "\t\n"],
            malformed_queries: vec![
                "???!!!",
                "<<>><<>>",
                "null undefined NaN",
            ],
            extreme_values: vec![f32::MIN, f32::MAX, f32::NEG_INFINITY, f32::INFINITY, f32::NAN],
            unicode_test_strings: vec![
                "Hello 世界",
                "Привет мир",
                "مرحبا بالعالم",
                "🚀🌟💡",
            ],
        }
    }
}

/// Test data for temporal patterns
pub struct TemporalTestData {
    pub timestamps: Vec<u64>,
    pub time_intervals: Vec<std::time::Duration>,
    pub decay_curves: Vec<Vec<f32>>,
}

impl TemporalTestData {
    pub fn new() -> Self {
        use std::time::Duration;
        
        Self {
            timestamps: vec![
                1000000000,
                1000001000,
                1000002000,
                1000003000,
                1000004000,
            ],
            time_intervals: vec![
                Duration::from_millis(100),
                Duration::from_secs(1),
                Duration::from_secs(10),
                Duration::from_secs(60),
                Duration::from_secs(3600),
            ],
            decay_curves: vec![
                vec![1.0, 0.9, 0.81, 0.73, 0.66],
                vec![1.0, 0.8, 0.64, 0.51, 0.41],
                vec![1.0, 0.7, 0.49, 0.34, 0.24],
            ],
        }
    }
}

/// Test data for memory patterns
pub struct MemoryTestData {
    pub memory_traces: Vec<(EntityKey, f32, u64)>,
    pub reinforcement_patterns: Vec<Vec<f32>>,
    pub forgetting_curves: HashMap<String, Vec<f32>>,
}

impl MemoryTestData {
    pub fn new() -> Self {
        let memory_traces = vec![
            (EntityKey::from_hash("strong_memory"), 0.95, 1000000000),
            (EntityKey::from_hash("medium_memory"), 0.70, 1000001000),
            (EntityKey::from_hash("weak_memory"), 0.40, 1000002000),
            (EntityKey::from_hash("fading_memory"), 0.20, 1000003000),
        ];

        let reinforcement_patterns = vec![
            vec![0.5, 0.7, 0.85, 0.92, 0.96],
            vec![0.3, 0.5, 0.65, 0.75, 0.82],
            vec![0.1, 0.3, 0.45, 0.55, 0.62],
        ];

        let mut forgetting_curves = HashMap::new();
        forgetting_curves.insert("rapid_decay".to_string(), vec![1.0, 0.5, 0.25, 0.125, 0.0625]);
        forgetting_curves.insert("normal_decay".to_string(), vec![1.0, 0.7, 0.49, 0.34, 0.24]);
        forgetting_curves.insert("slow_decay".to_string(), vec![1.0, 0.9, 0.81, 0.73, 0.66]);

        Self {
            memory_traces,
            reinforcement_patterns,
            forgetting_curves,
        }
    }
}

/// Master test data provider
pub struct TestDataProvider {
    pub queries: TestQueries,
    pub entities: Vec<EntityKey>,
    pub performance: PerformanceTestData,
    pub attention: AttentionTestData,
    pub patterns: PatternTestData,
    pub edge_cases: EdgeCaseTestData,
    pub temporal: TemporalTestData,
    pub memory: MemoryTestData,
}

impl TestDataProvider {
    pub fn new() -> Self {
        Self {
            queries: TestQueries::new(),
            entities: create_standard_test_entities(),
            performance: PerformanceTestData::new(),
            attention: AttentionTestData::new(),
            patterns: PatternTestData::new(),
            edge_cases: EdgeCaseTestData::new(),
            temporal: TemporalTestData::new(),
            memory: MemoryTestData::new(),
        }
    }

    /// Get test data for a specific cognitive pattern type
    pub fn get_queries_for_pattern(&self, pattern: CognitivePatternType) -> Vec<&str> {
        match pattern {
            CognitivePatternType::Convergent => self.queries.analytical.iter().copied().collect(),
            CognitivePatternType::Divergent => self.queries.creative.iter().copied().collect(),
            CognitivePatternType::Lateral => self.queries.relational.iter().copied().collect(),
            CognitivePatternType::Systems => self.queries.analytical.iter().copied().collect(),
            CognitivePatternType::Critical => self.queries.analytical.iter().copied().collect(),
            CognitivePatternType::Abstract => self.queries.creative.iter().copied().collect(),
            CognitivePatternType::Adaptive => self.queries.relational.iter().copied().collect(),
            CognitivePatternType::ChainOfThought => self.queries.analytical.iter().copied().collect(),
            CognitivePatternType::TreeOfThoughts => self.queries.creative.iter().copied().collect(),
        }
    }

    /// Get entities with pre-assigned attention scores
    pub fn get_entities_with_scores(&self) -> Vec<(EntityKey, f32)> {
        self.attention.entity_attention_scores
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Get test data for specific performance scenarios
    pub fn get_performance_dataset(&self, size: &str) -> &Vec<EntityKey> {
        match size {
            "small" => &self.performance.small_dataset,
            "medium" => &self.performance.medium_dataset,
            "large" => &self.performance.large_dataset,
            "stress" => &self.performance.stress_dataset,
            _ => &self.performance.medium_dataset,
        }
    }
}

/// Utility functions for test data generation
pub mod generators {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    /// Generate random entity keys with consistent naming
    pub fn generate_entities(count: usize, prefix: &str) -> Vec<EntityKey> {
        (0..count)
            .map(|i| EntityKey::from_hash(&format!("{}_{}", prefix, i)))
            .collect()
    }

    /// Generate random attention scores within a range
    pub fn generate_attention_scores(count: usize, min: f32, max: f32, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| rng.gen_range(min..=max))
            .collect()
    }

    /// Generate temporal sequences for testing
    pub fn generate_temporal_sequence(
        start_time: u64,
        count: usize,
        interval_ms: u64,
    ) -> Vec<u64> {
        (0..count)
            .map(|i| start_time + (i as u64 * interval_ms))
            .collect()
    }

    /// Generate decay curves for testing
    pub fn generate_decay_curve(
        initial_value: f32,
        decay_factor: f32,
        steps: usize,
    ) -> Vec<f32> {
        (0..steps)
            .map(|i| initial_value * decay_factor.powi(i as i32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_provider_initialization() {
        let provider = TestDataProvider::new();
        
        assert!(!provider.queries.factual.is_empty());
        assert!(!provider.queries.creative.is_empty());
        assert!(!provider.entities.is_empty());
        assert_eq!(provider.performance.small_dataset.len(), 10);
        assert_eq!(provider.performance.medium_dataset.len(), 100);
    }

    #[test]
    fn test_generators() {
        let entities = generators::generate_entities(5, "test");
        assert_eq!(entities.len(), 5);
        // Note: Cannot directly compare EntityKey content as it's opaque

        let scores = generators::generate_attention_scores(10, 0.0, 1.0, 42);
        assert_eq!(scores.len(), 10);
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));

        let decay = generators::generate_decay_curve(1.0, 0.9, 5);
        assert_eq!(decay.len(), 5);
        assert!((decay[0] - 1.0).abs() < f32::EPSILON);
        assert!((decay[1] - 0.9).abs() < f32::EPSILON);
    }
}