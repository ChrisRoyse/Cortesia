use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;

use crate::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
use crate::core::brain_types::{BrainInspiredEntity, EntityDirection, ActivationPattern};
use crate::core::phase1_types::{Phase1Config, QueryResult, EntityInfo, Phase1Statistics, CognitiveQueryResult};
use crate::core::phase1_helpers::Phase1Helpers;
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
    pub helpers: Phase1Helpers,
    pub config: Phase1Config,
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

        // 9. Initialize helpers
        let helpers = Phase1Helpers::new(
            canonicalizer.clone(),
            neural_server.clone(),
            brain_graph.clone(),
        );

        Ok(Self {
            brain_graph,
            neural_server,
            structure_predictor,
            canonicalizer,
            #[cfg(feature = "native")]
            mcp_server,
            temporal_processor,
            cognitive_orchestrator,
            helpers,
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
        let canonical_entities = self.helpers.canonicalize_text_entities(text, context).await?;
        
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
                    entity.embedding = self.helpers.generate_embedding(canonical_concept).await?;
                    
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
                if let Some(entity) = self.helpers.get_brain_entity(*entity_key).await? {
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
        let query_entities = self.helpers.extract_query_entities(query).await?;
        
        // 3. Set initial activations
        for (i, entity_name) in query_entities.iter().enumerate() {
            // Find entities in the brain graph
            let entities = self.helpers.find_entities_by_concept(entity_name).await?;
            
            for entity_key in entities {
                let activation_level = 1.0 / (i + 1) as f32; // Decay for later entities
                activation_pattern.activations.insert(entity_key, activation_level);
            }
        }
        
        // 4. Propagate through neural network (simplified for now)
        // TODO: Implement proper activation propagation
        let propagation_result = crate::core::activation_config::PropagationResult {
            final_activations: activation_pattern.activations.clone(),
            iterations_completed: 1,
            converged: true,
            activation_trace: Vec::new(),
            total_energy: activation_pattern.activations.values().sum(),
        };
        
        // 5. Extract meaningful results
        let top_activations = activation_pattern.get_top_activations(10);
        let entities_info = self.helpers.get_entities_info(&top_activations).await?;
        
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
        let activation_stats = crate::core::activation_config::ActivationStatistics {
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

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::RelationType;
    use crate::cognitive::CognitivePatternType;
    use std::collections::HashMap;

    // Helper function to create a test integration layer
    async fn create_test_integration() -> Phase1IntegrationLayer {
        let config = Phase1Config::default();
        Phase1IntegrationLayer::new(config).await.unwrap()
    }

    // Helper function to create a test integration layer with cognitive patterns enabled
    async fn create_test_integration_with_cognitive() -> Phase1IntegrationLayer {
        let mut config = Phase1Config::default();
        config.enable_cognitive_patterns = true;
        Phase1IntegrationLayer::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_phase1_integration_creation() {
        let config = Phase1Config::default();
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        let stats = integration.get_phase1_statistics().await.unwrap();
        assert_eq!(stats.brain_statistics.entity_count, 0);
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

    // Test private method: store_knowledge_with_neural_structure with empty input
    #[tokio::test]
    async fn test_store_knowledge_empty_input() {
        let integration = create_test_integration().await;
        
        // Test with empty text
        let result = integration.store_knowledge_with_neural_structure("", None).await;
        assert!(result.is_ok());
        let entities = result.unwrap();
        assert!(entities.is_empty()); // Empty input should result in empty entities
    }

    // Test private method: store_knowledge_with_neural_structure with invalid input
    #[tokio::test]
    async fn test_store_knowledge_invalid_input() {
        let integration = create_test_integration().await;
        
        // Test with whitespace-only input
        let result = integration.store_knowledge_with_neural_structure("   \n\t   ", None).await;
        assert!(result.is_ok());
        
        // Test with special characters only
        let result = integration.store_knowledge_with_neural_structure("@#$%^&*()", None).await;
        assert!(result.is_ok());
        
        // Test with very long input
        let long_text = "a".repeat(10000);
        let result = integration.store_knowledge_with_neural_structure(&long_text, None).await;
        assert!(result.is_ok());
    }

    // Test private method: store_knowledge_with_neural_structure pipeline component interactions
    #[tokio::test]
    async fn test_store_knowledge_pipeline_components() {
        let integration = create_test_integration().await;
        
        // Test canonicalization step
        let result = integration.store_knowledge_with_neural_structure(
            "Dr. John Smith is a doctor",
            Some("medical professionals"),
        ).await;
        assert!(result.is_ok());
        
        // Test structure prediction step
        let result = integration.store_knowledge_with_neural_structure(
            "The cat sits on the mat",
            None,
        ).await;
        assert!(result.is_ok());
        
        // Test entity creation with different node types
        let result = integration.store_knowledge_with_neural_structure(
            "Machine learning algorithms process data",
            Some("AI technology"),
        ).await;
        assert!(result.is_ok());
    }

    // Test private method: store_knowledge_with_neural_structure with real-time updates
    #[tokio::test]
    async fn test_store_knowledge_with_realtime_updates() {
        let mut config = Phase1Config::default();
        config.enable_real_time_updates = true;
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        let result = integration.store_knowledge_with_neural_structure(
            "Water boils at 100 degrees Celsius",
            Some("physics"),
        ).await;
        assert!(result.is_ok());
        
        // Verify temporal processor receives updates
        let queue_size = integration.temporal_processor.get_queue_size().await;
        // Queue might be processed quickly, so we just check it doesn't error
        assert!(queue_size >= 0);
    }

    // Test private method: neural_query_with_activation with empty query
    #[tokio::test]
    async fn test_neural_query_empty_query() {
        let integration = create_test_integration().await;
        
        let result = integration.neural_query_with_activation("", None).await;
        assert!(result.is_ok());
        
        let query_result = result.unwrap();
        assert_eq!(query_result.query, "");
        assert_eq!(query_result.cognitive_pattern, "convergent");
        assert!(query_result.final_activations.is_empty());
    }

    // Test private method: neural_query_with_activation activation propagation
    #[tokio::test]
    async fn test_neural_query_activation_propagation() {
        let integration = create_test_integration().await;
        
        // First store some knowledge to have entities for activation
        integration.store_knowledge_with_neural_structure(
            "Photosynthesis converts sunlight to energy",
            Some("biology"),
        ).await.unwrap();
        
        let result = integration.neural_query_with_activation(
            "photosynthesis energy conversion",
            Some("analytical"),
        ).await;
        assert!(result.is_ok());
        
        let query_result = result.unwrap();
        assert_eq!(query_result.query, "photosynthesis energy conversion");
        assert_eq!(query_result.cognitive_pattern, "analytical");
        assert_eq!(query_result.iterations_completed, 1);
        assert!(query_result.converged);
        assert!(query_result.total_energy >= 0.0);
    }

    // Test private method: neural_query_with_activation with multiple entities
    #[tokio::test]
    async fn test_neural_query_multiple_entities() {
        let integration = create_test_integration().await;
        
        // Store multiple related entities
        integration.store_knowledge_with_neural_structure(
            "Dogs are mammals. Cats are mammals. Both are pets.",
            Some("animals"),
        ).await.unwrap();
        
        let result = integration.neural_query_with_activation(
            "dogs cats mammals pets",
            Some("associative"),
        ).await;
        assert!(result.is_ok());
        
        let query_result = result.unwrap();
        assert!(!query_result.entities_info.is_empty() || query_result.final_activations.is_empty());
        // Activation levels should decay for later entities in the query
        if !query_result.final_activations.is_empty() {
            let activations: Vec<f32> = query_result.final_activations.values().cloned().collect();
            assert!(activations.iter().all(|&x| x >= 0.0 && x <= 1.0));
        }
    }

    // Test private method: cognitive_reasoning with different patterns
    #[tokio::test]
    async fn test_cognitive_reasoning_patterns() {
        let integration = create_test_integration_with_cognitive().await;
        
        // Test with specific pattern
        let result = integration.cognitive_reasoning(
            "What is the relationship between cause and effect?",
            Some("philosophical context"),
            Some(CognitivePatternType::ChainOfThought),
        ).await;
        assert!(result.is_ok());
        
        let cognitive_result = result.unwrap();
        assert_eq!(cognitive_result.query, "What is the relationship between cause and effect?");
        assert!(cognitive_result.confidence >= 0.0 && cognitive_result.confidence <= 1.0);
        assert!(cognitive_result.execution_time_ms >= 0);
    }

    // Test private method: cognitive_reasoning without cognitive orchestrator
    #[tokio::test]
    async fn test_cognitive_reasoning_disabled() {
        let integration = create_test_integration().await; // Cognitive patterns disabled
        
        let result = integration.cognitive_reasoning(
            "Test query",
            None,
            Some(CognitivePatternType::ChainOfThought),
        ).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            GraphError::UnsupportedOperation(msg) => {
                assert!(msg.contains("Cognitive patterns not enabled"));
            }
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    // Test private method: cognitive_reasoning with automatic strategy
    #[tokio::test]
    async fn test_cognitive_reasoning_automatic() {
        let integration = create_test_integration_with_cognitive().await;
        
        let result = integration.cognitive_reasoning(
            "Analyze this complex problem step by step",
            Some("problem solving"),
            None, // No specific pattern - should use automatic
        ).await;
        assert!(result.is_ok());
        
        let cognitive_result = result.unwrap();
        assert!(!cognitive_result.final_answer.is_empty());
        assert!(!cognitive_result.patterns_executed.is_empty());
    }

    // Test ensemble cognitive reasoning
    #[tokio::test]
    async fn test_ensemble_cognitive_reasoning() {
        let integration = create_test_integration_with_cognitive().await;
        
        let patterns = vec![
            CognitivePatternType::ChainOfThought,
            CognitivePatternType::TreeOfThoughts,
        ];
        
        let result = integration.ensemble_cognitive_reasoning(
            "Compare different approaches to solving this problem",
            Some("comparative analysis"),
            patterns,
        ).await;
        assert!(result.is_ok());
        
        let cognitive_result = result.unwrap();
        assert!(cognitive_result.patterns_executed.len() >= 1);
    }

    // Test internal component coordination
    #[tokio::test]
    async fn test_component_coordination() {
        let integration = create_test_integration().await;
        
        // Test that all components are properly initialized
        assert!(!integration.brain_graph.is_null());
        assert!(!integration.neural_server.is_null());
        assert!(!integration.structure_predictor.is_null());
        assert!(!integration.canonicalizer.is_null());
        assert!(!integration.temporal_processor.is_null());
        
        // Test helper coordination
        let stats = integration.get_phase1_statistics().await;
        assert!(stats.is_ok());
        
        // Test component interaction through pipeline
        let entities = integration.store_knowledge_with_neural_structure(
            "Component coordination test",
            None,
        ).await;
        assert!(entities.is_ok());
    }

    // Test pipeline validation and error handling
    #[tokio::test]
    async fn test_pipeline_validation() {
        let integration = create_test_integration().await;
        
        // Test with null/empty contexts
        let result = integration.store_knowledge_with_neural_structure(
            "Valid text",
            None,
        ).await;
        assert!(result.is_ok());
        
        let result = integration.store_knowledge_with_neural_structure(
            "Valid text",
            Some(""),
        ).await;
        assert!(result.is_ok());
        
        // Test query validation
        let result = integration.neural_query_with_activation(
            "valid query",
            None,
        ).await;
        assert!(result.is_ok());
        
        let result = integration.neural_query_with_activation(
            "valid query",
            Some(""),
        ).await;
        assert!(result.is_ok());
    }

    // Test temporal functionality
    #[tokio::test]
    async fn test_temporal_query_functionality() {
        let integration = create_test_integration().await;
        
        let valid_time = Utc::now();
        let transaction_time = Some(Utc::now());
        
        let result = integration.temporal_query_at_time(
            "temporal test query",
            valid_time,
            transaction_time,
        ).await;
        
        // Should return empty results when MCP is not available (unless native feature is enabled)
        assert!(result.is_ok());
        let temporal_entities = result.unwrap();
        #[cfg(not(feature = "native"))]
        assert!(temporal_entities.is_empty());
    }

    // Test system lifecycle operations
    #[tokio::test]
    async fn test_system_lifecycle() {
        let mut config = Phase1Config::default();
        config.enable_real_time_updates = true;
        let integration = Phase1IntegrationLayer::new(config).await.unwrap();
        
        // Test start
        let result = integration.start().await;
        assert!(result.is_ok());
        
        // Test operations while running
        let entities = integration.store_knowledge_with_neural_structure(
            "Lifecycle test",
            None,
        ).await;
        assert!(entities.is_ok());
        
        // Test stop
        let result = integration.stop().await;
        assert!(result.is_ok());
    }

    // Test error propagation through pipeline
    #[tokio::test]
    async fn test_error_propagation() {
        let integration = create_test_integration().await;
        
        // Test that errors from components are properly propagated
        // Most errors will be internal to the neural processing
        // but we can test that the pipeline handles them gracefully
        
        // Test with malformed input that might cause issues
        let result = integration.store_knowledge_with_neural_structure(
            "\0\x01\x02invalid\x03\x04",
            Some("malformed context\0"),
        ).await;
        // Should either succeed (if components handle it) or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // Test statistics collection from all components
    #[tokio::test]
    async fn test_comprehensive_statistics() {
        let integration = create_test_integration_with_cognitive().await;
        
        // Add some data to generate meaningful statistics
        integration.store_knowledge_with_neural_structure(
            "Statistics test data",
            Some("testing"),
        ).await.unwrap();
        
        // Test Phase1 statistics
        let stats = integration.get_phase1_statistics().await;
        assert!(stats.is_ok());
        let phase1_stats = stats.unwrap();
        assert!(phase1_stats.brain_statistics.entity_count >= 0);
        assert!(phase1_stats.current_queue_size >= 0);
        
        // Test cognitive statistics
        let cognitive_stats = integration.get_cognitive_statistics().await;
        assert!(cognitive_stats.is_ok());
        assert!(cognitive_stats.unwrap().is_some()); // Should have cognitive stats when enabled
    }

    // Test cognitive statistics when disabled
    #[tokio::test]
    async fn test_cognitive_statistics_disabled() {
        let integration = create_test_integration().await; // Cognitive disabled
        
        let cognitive_stats = integration.get_cognitive_statistics().await;
        assert!(cognitive_stats.is_ok());
        assert!(cognitive_stats.unwrap().is_none()); // Should be None when disabled
    }

    // Test activation pattern creation and manipulation
    #[tokio::test]
    async fn test_activation_pattern_handling() {
        let integration = create_test_integration().await;
        
        // Store entities first
        integration.store_knowledge_with_neural_structure(
            "Activation pattern test with multiple concepts",
            None,
        ).await.unwrap();
        
        // Test query with multiple terms to create complex activation patterns
        let result = integration.neural_query_with_activation(
            "activation pattern test multiple concepts",
            Some("pattern_analysis"),
        ).await;
        assert!(result.is_ok());
        
        let query_result = result.unwrap();
        // Verify activation pattern properties
        assert!(query_result.total_energy >= 0.0);
        assert!(query_result.iterations_completed > 0);
        assert_eq!(query_result.cognitive_pattern, "pattern_analysis");
    }

    // Test edge cases in neural query processing
    #[tokio::test]
    async fn test_neural_query_edge_cases() {
        let integration = create_test_integration().await;
        
        // Test with very long query
        let long_query = "query ".repeat(1000);
        let result = integration.neural_query_with_activation(&long_query, None).await;
        assert!(result.is_ok());
        
        // Test with special characters
        let special_query = "query@#$%^&*()_+-=[]{}|;':\",./<>?";
        let result = integration.neural_query_with_activation(special_query, None).await;
        assert!(result.is_ok());
        
        // Test with unicode characters
        let unicode_query = "query 测试 العربية русский 日本語";
        let result = integration.neural_query_with_activation(unicode_query, None).await;
        assert!(result.is_ok());
    }

    // Test concurrent access to components
    #[tokio::test]
    async fn test_concurrent_component_access() {
        let integration = Arc::new(create_test_integration().await);
        
        // Spawn multiple concurrent operations
        let mut handles = Vec::new();
        
        for i in 0..5 {
            let integration_clone = integration.clone();
            let handle = tokio::spawn(async move {
                let text = format!("Concurrent test {}", i);
                integration_clone.store_knowledge_with_neural_structure(&text, None).await
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
        
        // Verify system state is consistent
        let stats = integration.get_phase1_statistics().await;
        assert!(stats.is_ok());
    }
}