//! Builder patterns for creating test instances

use crate::cognitive::attention_manager::AttentionManager;
use crate::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::activation_engine::ActivationPropagationEngine;
use crate::core::activation_config::ActivationConfig;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::sync::Arc;

/// Builder for creating configured AttentionManager instances for testing
pub struct AttentionManagerBuilder {
    graph: Option<Arc<BrainEnhancedKnowledgeGraph>>,
    embedding_dim: usize,
}

impl AttentionManagerBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            graph: None,
            embedding_dim: 96, // default embedding dimension for tests
        }
    }
    
    /// Sets a custom graph
    pub fn with_graph(mut self, graph: Arc<BrainEnhancedKnowledgeGraph>) -> Self {
        self.graph = Some(graph);
        self
    }
    
    /// Sets a custom embedding dimension
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }
    
    /// Builds just the AttentionManager asynchronously
    pub async fn build(self) -> Result<AttentionManager> {
        let graph = match self.graph {
            Some(g) => g,
            None => Arc::new(BrainEnhancedKnowledgeGraph::new_for_test()?),
        };
        
        let orchestrator = Arc::new(
            CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default()).await?
        );
        let activation_engine = Arc::new(ActivationPropagationEngine::new(ActivationConfig::default()));
        let working_memory = Arc::new(
            WorkingMemorySystem::new(activation_engine.clone(), graph.sdr_storage.clone()).await?
        );
        
        AttentionManager::new(orchestrator, activation_engine, working_memory).await
    }
    
    /// Builds AttentionManager with all dependencies exposed asynchronously
    pub async fn build_with_deps(self) -> Result<(
        AttentionManager,
        Arc<CognitiveOrchestrator>,
        Arc<ActivationPropagationEngine>,
        Arc<WorkingMemorySystem>,
    )> {
        let graph = match self.graph {
            Some(g) => g,
            None => Arc::new(BrainEnhancedKnowledgeGraph::new_for_test()?),
        };
        
        let orchestrator = Arc::new(
            CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default()).await?
        );
        let activation_engine = Arc::new(ActivationPropagationEngine::new(ActivationConfig::default()));
        let working_memory = Arc::new(
            WorkingMemorySystem::new(activation_engine.clone(), graph.sdr_storage.clone()).await?
        );
        
        let attention_manager = AttentionManager::new(
            orchestrator.clone(),
            activation_engine.clone(),
            working_memory.clone(),
        ).await?;
        
        Ok((attention_manager, orchestrator, activation_engine, working_memory))
    }
}

impl Default for AttentionManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test instances of cognitive patterns
pub struct CognitivePatternBuilder {
    graph: Option<Arc<BrainEnhancedKnowledgeGraph>>,
}

impl CognitivePatternBuilder {
    pub fn new() -> Self {
        Self { graph: None }
    }
    
    pub fn with_graph(mut self, graph: Arc<BrainEnhancedKnowledgeGraph>) -> Self {
        self.graph = Some(graph);
        self
    }
    
    pub fn build_convergent(self) -> crate::cognitive::convergent::ConvergentThinking {
        use crate::cognitive::convergent::ConvergentThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        ConvergentThinking::new(graph)
    }
    
    pub fn build_divergent(self) -> crate::cognitive::divergent::DivergentThinking {
        use crate::cognitive::divergent::DivergentThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        DivergentThinking::new(graph)
    }
    
    pub fn build_lateral(self) -> crate::cognitive::lateral::LateralThinking {
        use crate::cognitive::lateral::LateralThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        LateralThinking::new(graph)
    }
}