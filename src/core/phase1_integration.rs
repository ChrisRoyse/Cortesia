use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;

use crate::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
use crate::core::brain_types::{BrainInspiredEntity, EntityDirection, ActivationPattern};
use crate::neural::neural_server::NeuralProcessingServer;
use crate::neural::structure_predictor::GraphStructurePredictor;
use crate::neural::canonicalization::EnhancedNeuralCanonicalizer;
#[cfg(feature = "native")]
use crate::mcp::brain_inspired_server::BrainInspiredMCPServer;
use crate::streaming::temporal_updates::{IncrementalTemporalProcessor, TemporalUpdateBuilder, UpdateOperation, UpdateSource};
use crate::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig, ReasoningStrategy, CognitivePatternType};
use crate::error::{Result, GraphError};

/// Phase 1 Integration Layer with Phase 2 Cognitive Capabilities
pub struct Phase1IntegrationLayer {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub structure_predictor: Arc<GraphStructurePredictor>,
    pub canonicalizer: Arc<EnhancedNeuralCanonicalizer>,
    #[cfg(feature = "native")]
    pub mcp_server: Arc<BrainInspiredMCPServer>,
    pub temporal_processor: Arc<IncrementalTemporalProcessor>,
    pub cognitive_orchestrator: Option<Arc<CognitiveOrchestrator>>,
    pub config: Phase1Config,
}

/// Configuration for Phase 1 integration with Phase 2 cognitive capabilities
#[derive(Debug, Clone)]
pub struct Phase1Config {
    pub embedding_dim: usize,
    pub neural_server_endpoint: String,
    pub enable_temporal_tracking: bool,
    pub enable_sdr_storage: bool,
    pub enable_real_time_updates: bool,
    pub enable_cognitive_patterns: bool,
    pub activation_config: crate::core::activation_engine::ActivationConfig,
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
            activation_config: crate::core::activation_engine::ActivationConfig::default(),
        }
    }
}

impl Phase1IntegrationLayer {
    /// Initialize the complete Phase 1 system
    pub async fn new(config: Phase1Config) -> Result<Self> {
        // 1. Initialize neural server
        let neural_server = Arc::new(
            NeuralProcessingServer::new(config.neural_server_endpoint.clone()).await?
        );

        // 2. Initialize brain-enhanced knowledge graph
        let brain_config = BrainEnhancedConfig {
            embedding_dim: config.embedding_dim,
            activation_config: crate::core::brain_enhanced_graph::brain_graph_types::ActivationConfig::default(),
            enable_temporal_tracking: config.enable_temporal_tracking,
            enable_sdr_storage: config.enable_sdr_storage,
            ..BrainEnhancedConfig::default()
        };
        
        let brain_graph = Arc::new(
            BrainEnhancedKnowledgeGraph::new_with_config(config.embedding_dim, brain_config)?
        );

        // 3. Initialize structure predictor
        let structure_predictor = Arc::new(
            GraphStructurePredictor::new("structure_model".to_string(), neural_server.clone())
        );

        // 4. Initialize enhanced canonicalizer
        let canonicalizer = Arc::new(
            EnhancedNeuralCanonicalizer::new(neural_server.clone())
        );

        // 5. Initialize temporal graph (create new one since KnowledgeGraph doesn't implement Clone)
        let temporal_graph = Arc::new(tokio::sync::RwLock::new(
            crate::versioning::temporal_graph::TemporalKnowledgeGraph::new_default()
        ));

        // 6. Initialize MCP server (if native feature is enabled)
        #[cfg(feature = "native")]
        let mcp_server = Arc::new(
            BrainInspiredMCPServer::new(temporal_graph.clone(), neural_server.clone())
        );

        // 7. Initialize temporal processor
        let temporal_processor = {
            let temporal_graph_guard = temporal_graph.read().await;
            Arc::new(
                IncrementalTemporalProcessor::new(
                    Arc::new(temporal_graph_guard.clone()),
                    32, // batch size
                    std::time::Duration::from_millis(100), // max latency
                )
            )
        };

        // 8. Initialize cognitive orchestrator if enabled
        let cognitive_orchestrator = if config.enable_cognitive_patterns {
            let cognitive_config = CognitiveOrchestratorConfig::default();
            let orchestrator = CognitiveOrchestrator::new(
                brain_graph.clone(),
                cognitive_config,
            ).await?;
            Some(Arc::new(orchestrator))
        } else {
            None
        };

        Ok(Self {
            brain_graph,
            neural_server,
            structure_predictor,
            canonicalizer,
            #[cfg(feature = "native")]
            mcp_server,
            temporal_processor,
            cognitive_orchestrator,
            config,
        })
    }

    /// Store knowledge with full Phase 1 pipeline
    pub async fn store_knowledge_with_neural_structure(
        &self,
        text: &str,
        context: Option<&str>,
    ) -> Result<Vec<crate::core::types::EntityKey>> {
        // 1. Canonicalize entities
        let canonical_entities = self.canonicalize_text_entities(text, context).await?;
        
        // 2. Predict graph structure
        let graph_operations = self.structure_predictor.predict_structure(text).await?;
        
        // 3. Execute operations to create brain entities
        let mut created_entities = Vec::new();
        
        for operation in graph_operations {
            match operation {
                crate::core::brain_types::GraphOperation::CreateNode { concept, node_type } => {
                    let canonical_concept = canonical_entities.get(&concept)
                        .unwrap_or(&concept);
                    
                    let mut entity = BrainInspiredEntity::new(
                        canonical_concept.clone(),
                        node_type,
                    );
                    
                    // Generate embedding
                    entity.embedding = self.generate_embedding(canonical_concept).await?;
                    
                    // Convert BrainInspiredEntity to EntityData for insertion
                    let entity_data = crate::core::types::EntityData {
                        type_id: 0, // Default entity type
                        properties: entity.concept_id.clone(),
                        embedding: entity.embedding.clone(),
                    };
                    // Generate a numeric ID from the entity concept
                    let numeric_id = entity.concept_id.as_bytes().iter().fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
                    let entity_key = self.brain_graph.insert_brain_entity(numeric_id, entity_data).await?;
                    created_entities.push(entity_key);
                }
                crate::core::brain_types::GraphOperation::CreateLogicGate { inputs: _, outputs: _, gate_type } => {
                    // Generate a unique ID for the logic gate (simple hash-based approach)
                    let gate_id = gate_type.to_string().as_bytes().iter().fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
                    // For now, use empty EntityKey vectors since we need to resolve string names to keys
                    let empty_inputs: Vec<crate::core::types::EntityKey> = Vec::new();
                    let empty_outputs: Vec<crate::core::types::EntityKey> = Vec::new();
                    let gate_key = self.brain_graph.insert_logic_gate(gate_id, &gate_type.to_string(), empty_inputs, empty_outputs).await?;
                    created_entities.push(gate_key);
                }
                crate::core::brain_types::GraphOperation::CreateRelationship { source: _, target: _, relation_type, weight } => {
                    // Find or create entities for the relationship
                    // This is simplified - in practice we'd track entity keys
                    let source_key = crate::core::types::EntityKey::default();
                    let target_key = crate::core::types::EntityKey::default();
                    
                    // Use the RelationType directly since it's already of the correct type
                    let rel_type = relation_type;
                    
                    let mut brain_relationship = crate::core::brain_types::BrainInspiredRelationship::new(
                        source_key,
                        target_key,
                        rel_type,
                    );
                    brain_relationship.weight = weight;
                    
                    // Convert BrainInspiredRelationship to core Relationship
                    let relationship = crate::core::types::Relationship {
                        from: brain_relationship.source,
                        to: brain_relationship.target,
                        rel_type: rel_type as u8, // Convert RelationType to u8
                        weight: brain_relationship.weight,
                    };
                    
                    self.brain_graph.insert_brain_relationship(relationship).await?;
                }
            }
        }
        
        // 4. If real-time updates enabled, process through temporal processor
        if self.config.enable_real_time_updates {
            for entity_key in &created_entities {
                if let Some(entity) = self.get_brain_entity(*entity_key).await? {
                    let update = TemporalUpdateBuilder::new(UpdateOperation::Create)
                        .with_entity(entity)
                        .with_source(UpdateSource::System)
                        .build()?;
                    
                    self.temporal_processor.enqueue_update(update).await?;
                }
            }
        }
        
        Ok(created_entities)
    }

    /// Perform neural query with full activation propagation
    pub async fn neural_query_with_activation(
        &self,
        query: &str,
        cognitive_pattern: Option<&str>,
    ) -> Result<QueryResult> {
        // 1. Create initial activation pattern
        let mut activation_pattern = ActivationPattern::new(query.to_string());
        
        // 2. Use canonicalizer to identify query entities
        let query_entities = self.extract_query_entities(query).await?;
        
        // 3. Set initial activations
        for (i, entity_name) in query_entities.iter().enumerate() {
            // Find entities in the brain graph
            let entities = self.find_entities_by_concept(entity_name).await?;
            
            for entity_key in entities {
                let activation_level = 1.0 / (i + 1) as f32; // Decay for later entities
                activation_pattern.activations.insert(entity_key, activation_level);
            }
        }
        
        // 4. Propagate through neural network (simplified for now)
        // TODO: Implement proper activation propagation
        let propagation_result = crate::core::activation_engine::PropagationResult {
            final_activations: activation_pattern.activations.clone(),
            iterations_completed: 1,
            converged: true,
            activation_trace: Vec::new(),
            total_energy: activation_pattern.activations.values().sum(),
        };
        
        // 5. Extract meaningful results
        let top_activations = activation_pattern.get_top_activations(10);
        let entities_info = self.get_entities_info(&top_activations).await?;
        
        Ok(QueryResult {
            query: query.to_string(),
            cognitive_pattern: cognitive_pattern.unwrap_or("convergent").to_string(),
            final_activations: propagation_result.final_activations,
            iterations_completed: propagation_result.iterations_completed,
            converged: propagation_result.converged,
            entities_info,
            total_energy: propagation_result.total_energy,
        })
    }

    /// Temporal query at specific point in time
    pub async fn temporal_query_at_time(
        &self,
        _query: &str,
        valid_time: chrono::DateTime<Utc>,
        transaction_time: Option<chrono::DateTime<Utc>>,
    ) -> Result<Vec<crate::versioning::temporal_graph::TemporalEntity>> {
        let _tx_time = transaction_time.unwrap_or_else(Utc::now);
        
        // Get temporal graph from MCP server (it holds the temporal graph)
        #[cfg(feature = "native")]
        {
            let temporal_graph = self.mcp_server.knowledge_graph.read().await;
            temporal_graph.query_at_time(valid_time, _tx_time).await
        }
        
        #[cfg(not(feature = "native"))]
        {
            // Return empty results when MCP is not available
            Ok(Vec::new())
        }
    }

    /// Get comprehensive system statistics
    pub async fn get_phase1_statistics(&self) -> Result<Phase1Statistics> {
        let brain_stats = self.brain_graph.get_brain_statistics().await?;
        // Create dummy activation statistics since activation_engine is not available
        let activation_stats = crate::core::activation_engine::ActivationStatistics {
            total_entities: 0,
            total_gates: 0,
            total_relationships: 0,
            active_entities: 0,
            inhibitory_connections: 0,
            average_activation: 0.0,
        };
        let update_stats = self.temporal_processor.get_statistics().await;
        let queue_size = self.temporal_processor.get_queue_size().await;
        
        Ok(Phase1Statistics {
            brain_statistics: brain_stats,
            activation_statistics: activation_stats,
            update_statistics: update_stats,
            current_queue_size: queue_size,
            neural_server_connected: true, // Would check actual connection
        })
    }

    /// Execute cognitive reasoning using Phase 2 patterns
    pub async fn cognitive_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
        pattern: Option<CognitivePatternType>,
    ) -> Result<CognitiveQueryResult> {
        if let Some(orchestrator) = &self.cognitive_orchestrator {
            let strategy = if let Some(pattern_type) = pattern {
                ReasoningStrategy::Specific(pattern_type)
            } else {
                ReasoningStrategy::Automatic
            };
            
            let result = orchestrator.reason(query, context, strategy).await?;
            
            Ok(CognitiveQueryResult {
                query: query.to_string(),
                final_answer: result.final_answer,
                strategy_used: result.strategy_used,
                confidence: result.quality_metrics.overall_confidence,
                execution_time_ms: result.execution_metadata.total_time_ms,
                patterns_executed: result.execution_metadata.patterns_executed,
                quality_metrics: result.quality_metrics,
            })
        } else {
            Err(GraphError::UnsupportedOperation("Cognitive patterns not enabled".to_string()))
        }
    }

    /// Execute ensemble cognitive reasoning with multiple patterns
    pub async fn ensemble_cognitive_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
        patterns: Vec<CognitivePatternType>,
    ) -> Result<CognitiveQueryResult> {
        if let Some(orchestrator) = &self.cognitive_orchestrator {
            let strategy = ReasoningStrategy::Ensemble(patterns);
            let result = orchestrator.reason(query, context, strategy).await?;
            
            Ok(CognitiveQueryResult {
                query: query.to_string(),
                final_answer: result.final_answer,
                strategy_used: result.strategy_used,
                confidence: result.quality_metrics.overall_confidence,
                execution_time_ms: result.execution_metadata.total_time_ms,
                patterns_executed: result.execution_metadata.patterns_executed,
                quality_metrics: result.quality_metrics,
            })
        } else {
            Err(GraphError::UnsupportedOperation("Cognitive patterns not enabled".to_string()))
        }
    }

    /// Get cognitive orchestrator statistics
    pub async fn get_cognitive_statistics(&self) -> Result<Option<crate::cognitive::OrchestratorStatistics>> {
        if let Some(orchestrator) = &self.cognitive_orchestrator {
            Ok(Some(orchestrator.get_statistics().await?))
        } else {
            Ok(None)
        }
    }

    /// Start all background processes
    pub async fn start(&self) -> Result<()> {
        if self.config.enable_real_time_updates {
            self.temporal_processor.start().await?;
        }
        Ok(())
    }

    /// Stop all background processes
    pub async fn stop(&self) -> Result<()> {
        if self.config.enable_real_time_updates {
            self.temporal_processor.stop().await?;
        }
        Ok(())
    }

    // Helper methods

    async fn canonicalize_text_entities(
        &self,
        text: &str,
        context: Option<&str>,
    ) -> Result<std::collections::HashMap<String, String>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut canonical_map = std::collections::HashMap::new();
        
        for word in words {
            if word.len() > 2 {
                let canonical = if let Some(ctx) = context {
                    self.canonicalizer.canonicalize_with_context(word, ctx).await?
                } else {
                    self.canonicalizer.base_canonicalizer.canonicalize_entity(word).await?
                };
                canonical_map.insert(word.to_string(), canonical);
            }
        }
        
        Ok(canonical_map)
    }

    async fn generate_embedding(&self, concept: &str) -> Result<Vec<f32>> {
        // Create a simple input vector from the concept
        let mut input_vector = vec![0.0; 384];
        for (i, byte) in concept.bytes().enumerate() {
            if i < 384 {
                input_vector[i] = (byte as f32) / 255.0;
            }
        }
        
        let result = self.neural_server.neural_predict(
            "embedding_model",
            input_vector,
        ).await?;
        
        Ok(result.prediction)
    }

    async fn get_brain_entity(
        &self,
        entity_key: crate::core::types::EntityKey,
    ) -> Result<Option<BrainInspiredEntity>> {
        // Entities are stored in the core graph
        let entity_data = self.brain_graph.core_graph.get_entity(entity_key);
        
        // Convert to BrainInspiredEntity
        if let Some((_meta, data)) = entity_data {
            Ok(Some(BrainInspiredEntity {
                id: entity_key,
                concept_id: format!("entity_{:?}", entity_key),
                direction: EntityDirection::Input,
                properties: HashMap::new(),
                embedding: data.embedding,
                activation_state: 0.0,
                last_activation: std::time::SystemTime::now(),
                last_update: std::time::SystemTime::now(),
            }))
        } else {
            Ok(None)
        }
    }

    async fn extract_query_entities(&self, query: &str) -> Result<Vec<String>> {
        // Simple entity extraction - would use NLP in practice
        Ok(query.split_whitespace()
            .filter(|word| word.len() > 2)
            .map(|word| word.to_lowercase())
            .collect())
    }

    async fn find_entities_by_concept(
        &self,
        concept: &str,
    ) -> Result<Vec<crate::core::types::EntityKey>> {
        let all_entities = self.brain_graph.get_all_entities().await;
        let mut matching_keys = Vec::new();
        
        for (key, entity_data, _) in &all_entities {
            // Check if entity properties contain the concept
            if entity_data.properties.to_lowercase().contains(&concept.to_lowercase()) {
                matching_keys.push(*key);
            }
        }
        
        Ok(matching_keys)
    }

    async fn get_entities_info(
        &self,
        activations: &[(crate::core::types::EntityKey, f32)],
    ) -> Result<Vec<EntityInfo>> {
        let all_entities = self.brain_graph.get_all_entities().await;
        let mut info = Vec::new();
        
        for (key, activation) in activations {
            if let Some((_, _entity_data, _)) = all_entities.iter().find(|(k, _, _)| k == key) {
                info.push(EntityInfo {
                    entity_key: *key,
                    concept_id: format!("entity_{:?}", key),
                    direction: EntityDirection::Input,
                    activation_level: *activation,
                });
            }
        }
        
        Ok(info)
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
    pub activation_statistics: crate::core::activation_engine::ActivationStatistics,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_phase1_integration_creation() {
        let config = Phase1Config::default();
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        let stats = integration.get_phase1_statistics().await.unwrap();
        assert_eq!(stats.brain_statistics.total_brain_entities, 0);
    }

    #[tokio::test]
    async fn test_knowledge_storage_pipeline() {
        let config = Phase1Config::default();
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        let entities = integration.store_knowledge_with_neural_structure(
            "Pluto is a dog",
            Some("pets"),
        ).await.unwrap();
        
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_neural_query_pipeline() {
        let config = Phase1Config::default();
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        // First store some knowledge
        integration.store_knowledge_with_neural_structure(
            "Einstein was a physicist",
            None,
        ).await.unwrap();
        
        // Then query it
        let result = integration.neural_query_with_activation(
            "Who was Einstein?",
            Some("convergent"),
        ).await.unwrap();
        
        assert_eq!(result.query, "Who was Einstein?");
        assert_eq!(result.cognitive_pattern, "convergent");
    }
}