use crate::core::brain_types::EntityDirection;
use crate::cognitive::{ReasoningStrategy, CognitivePatternType};

/// Configuration for Phase 1 integration with Phase 2 cognitive capabilities
#[derive(Debug, Clone)]
pub struct Phase1Config {
    pub embedding_dim: usize,
    pub neural_server_endpoint: String,
    pub enable_temporal_tracking: bool,
    pub enable_sdr_storage: bool,
    pub enable_real_time_updates: bool,
    pub enable_cognitive_patterns: bool,
    pub activation_config: crate::core::activation_config::ActivationConfig,
}

impl Default for Phase1Config {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            neural_server_endpoint: "localhost:9000".to_string(),
            enable_temporal_tracking: true,
            enable_sdr_storage: true,
            enable_real_time_updates: true,
            enable_cognitive_patterns: true,
            activation_config: crate::core::activation_config::ActivationConfig::default(),
        }
    }
}

/// Result of a neural query
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query: String,
    pub cognitive_pattern: String,
    pub final_activations: std::collections::HashMap<crate::core::types::EntityKey, f32>,
    pub iterations_completed: usize,
    pub converged: bool,
    pub entities_info: Vec<EntityInfo>,
    pub total_energy: f32,
}

/// Information about an entity in query results
#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub entity_key: crate::core::types::EntityKey,
    pub concept_id: String,
    pub direction: EntityDirection,
    pub activation_level: f32,
}

/// Comprehensive Phase 1 statistics
#[derive(Debug, Clone)]
pub struct Phase1Statistics {
    pub brain_statistics: crate::core::brain_enhanced_graph::BrainStatistics,
    pub activation_statistics: crate::core::activation_config::ActivationStatistics,
    pub update_statistics: crate::streaming::temporal_updates::UpdateStatistics,
    pub current_queue_size: usize,
    pub neural_server_connected: bool,
}

/// Result of cognitive query processing
#[derive(Debug, Clone)]
pub struct CognitiveQueryResult {
    pub query: String,
    pub final_answer: String,
    pub strategy_used: ReasoningStrategy,
    pub confidence: f32,
    pub execution_time_ms: u64,
    pub patterns_executed: Vec<CognitivePatternType>,
    pub quality_metrics: crate::cognitive::QualityMetrics,
}