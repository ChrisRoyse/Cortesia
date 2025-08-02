//! Common test fixtures for LLMKG tests

use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::types::{EntityKey, EntityData};
use crate::error::Result;
use std::sync::Arc;
use slotmap::SlotMap;

/// Creates a test graph with default configuration
pub fn create_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph"))
}

/// Creates a test graph with custom embedding dimension
pub fn create_test_graph_with_dim(embedding_dim: usize) -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new(embedding_dim).expect("Failed to create test graph"))
}

/// Creates a test knowledge graph (alias for consistency with README examples)
pub fn create_test_knowledge_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    create_test_graph()
}

/// Creates standard test entities with realistic data
pub fn create_standard_test_entities() -> Vec<EntityData> {
    vec![
        EntityData::new(1, "artificial_intelligence".to_string(), vec![0.1, 0.2, 0.3, 0.4]),
        EntityData::new(2, "machine_learning".to_string(), vec![0.2, 0.3, 0.4, 0.5]),
        EntityData::new(3, "text_processing".to_string(), vec![0.3, 0.4, 0.5, 0.6]),
        EntityData::new(4, "deep_learning".to_string(), vec![0.4, 0.5, 0.6, 0.7]),
        EntityData::new(5, "natural_language_processing".to_string(), vec![0.5, 0.6, 0.7, 0.8]),
        EntityData::new(6, "computer_vision".to_string(), vec![0.6, 0.7, 0.8, 0.9]),
        EntityData::new(7, "reinforcement_learning".to_string(), vec![0.7, 0.8, 0.9, 1.0]),
    ]
}

/// Creates test entities with specific characteristics for cognitive testing
pub fn create_cognitive_test_entities() -> Vec<EntityData> {
    vec![
        EntityData::new(10, "problem_solving".to_string(), vec![0.9, 0.1, 0.5, 0.3]),
        EntityData::new(11, "creative_thinking".to_string(), vec![0.2, 0.9, 0.7, 0.4]),
        EntityData::new(12, "logical_reasoning".to_string(), vec![0.8, 0.2, 0.9, 0.1]),
        EntityData::new(13, "pattern_recognition".to_string(), vec![0.3, 0.7, 0.6, 0.9]),
        EntityData::new(14, "abstract_thinking".to_string(), vec![0.6, 0.4, 0.8, 0.7]),
    ]
}

/// Creates a diverse set of test entities for various test scenarios
pub fn create_diverse_test_entities() -> Vec<EntityData> {
    let mut entities = create_standard_test_entities();
    entities.extend(create_cognitive_test_entities());
    
    // Add some domain-specific entities
    entities.extend(vec![
        EntityData::new(20, "photosynthesis".to_string(), vec![0.1, 0.9, 0.2, 0.8]),
        EntityData::new(21, "democracy".to_string(), vec![0.7, 0.3, 0.9, 0.2]),
        EntityData::new(22, "quantum_physics".to_string(), vec![0.9, 0.8, 0.1, 0.6]),
        EntityData::new(23, "climate_change".to_string(), vec![0.4, 0.6, 0.8, 0.3]),
        EntityData::new(24, "urban_planning".to_string(), vec![0.5, 0.4, 0.7, 0.9]),
    ]);
    
    entities
}

/// Creates entity keys for testing without requiring a full graph
pub fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    let mut keys = Vec::new();
    
    for i in 0..count {
        let key = sm.insert(EntityData::new(
            i as u16 + 1,
            format!("test_entity_{i}"),
            vec![i as f32 / 100.0; 4],
        ));
        keys.push(key);
    }
    
    keys
}

/// Creates a test graph populated with standard test entities
pub async fn create_populated_test_graph() -> Result<Arc<BrainEnhancedKnowledgeGraph>> {
    let graph = create_test_graph();
    let _entities = create_standard_test_entities();
    
    // TODO: Add entities to graph when API is available
    // For now, just return the empty graph
    Ok(graph)
}

/// Creates a test graph with specific configuration for cognitive testing
pub async fn create_cognitive_test_graph() -> Result<Arc<BrainEnhancedKnowledgeGraph>> {
    let graph = create_test_graph();
    let _entities = create_cognitive_test_entities();
    
    // TODO: Add cognitive entities to graph when API is available
    Ok(graph)
}

/// Creates a minimal test graph for lightweight tests
pub fn create_minimal_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create minimal test graph"))
}

/// Helper function to create test graph configurations
pub struct TestGraphConfig {
    pub embedding_dim: usize,
    pub entity_count: usize,
    pub enable_caching: bool,
    pub enable_indexing: bool,
}

impl Default for TestGraphConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 96,
            entity_count: 100,
            enable_caching: true,
            enable_indexing: true,
        }
    }
}

/// Creates a test graph with custom configuration
pub async fn create_configured_test_graph(config: TestGraphConfig) -> Result<Arc<BrainEnhancedKnowledgeGraph>> {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(config.embedding_dim)?);
    
    // TODO: Apply configuration settings when API is available
    // - Set entity count
    // - Enable/disable caching
    // - Enable/disable indexing
    
    Ok(graph)
}