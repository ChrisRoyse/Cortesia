//! Builder patterns for creating test instances

use crate::cognitive::attention_manager::AttentionManager;
use crate::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::activation_engine::ActivationPropagationEngine;
use crate::core::activation_config::ActivationConfig;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::neural::neural_server::NeuralProcessingServer;
use crate::monitoring::brain_metrics_collector::BrainMetricsCollector;
use crate::monitoring::performance::PerformanceMonitor;
use crate::federation::coordinator::FederationCoordinator;
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
    
    pub fn build_systems(self) -> crate::cognitive::systems::SystemsThinking {
        use crate::cognitive::systems::SystemsThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        SystemsThinking::new(graph)
    }
    
    pub fn build_critical(self) -> crate::cognitive::critical::CriticalThinking {
        use crate::cognitive::critical::CriticalThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        CriticalThinking::new(graph)
    }
    
    pub fn build_abstract(self) -> crate::cognitive::abstract_pattern::AbstractThinking {
        use crate::cognitive::abstract_pattern::AbstractThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        AbstractThinking::new(graph)
    }
    
    pub fn build_adaptive(self) -> crate::cognitive::adaptive::AdaptiveThinking {
        use crate::cognitive::adaptive::AdaptiveThinking;
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        AdaptiveThinking::new(graph)
    }
}

impl Default for CognitivePatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test instances of CognitiveOrchestrator
pub struct CognitiveOrchestratorBuilder {
    graph: Option<Arc<BrainEnhancedKnowledgeGraph>>,
    config: CognitiveOrchestratorConfig,
}

impl CognitiveOrchestratorBuilder {
    pub fn new() -> Self {
        Self {
            graph: None,
            config: CognitiveOrchestratorConfig::default(),
        }
    }
    
    pub fn with_graph(mut self, graph: Arc<BrainEnhancedKnowledgeGraph>) -> Self {
        self.graph = Some(graph);
        self
    }
    
    pub fn with_config(mut self, config: CognitiveOrchestratorConfig) -> Self {
        self.config = config;
        self
    }
    
    pub async fn build(self) -> Result<CognitiveOrchestrator> {
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        CognitiveOrchestrator::new(graph, self.config).await
    }
}

impl Default for CognitiveOrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test instances of WorkingMemorySystem
pub struct WorkingMemoryBuilder {
    graph: Option<Arc<BrainEnhancedKnowledgeGraph>>,
    activation_engine: Option<Arc<ActivationPropagationEngine>>,
}

impl WorkingMemoryBuilder {
    pub fn new() -> Self {
        Self {
            graph: None,
            activation_engine: None,
        }
    }
    
    pub fn with_graph(mut self, graph: Arc<BrainEnhancedKnowledgeGraph>) -> Self {
        self.graph = Some(graph);
        self
    }
    
    pub fn with_activation_engine(mut self, engine: Arc<ActivationPropagationEngine>) -> Self {
        self.activation_engine = Some(engine);
        self
    }
    
    pub async fn build(self) -> Result<WorkingMemorySystem> {
        let graph = self.graph.unwrap_or_else(|| super::fixtures::create_test_graph());
        let activation_engine = self.activation_engine.unwrap_or_else(|| {
            Arc::new(ActivationPropagationEngine::new(ActivationConfig::default()))
        });
        
        WorkingMemorySystem::new(activation_engine, graph.sdr_storage.clone()).await
    }
}

impl Default for WorkingMemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test query contexts
pub struct QueryContextBuilder {
    domain: Option<String>,
    confidence_threshold: f32,
    max_depth: Option<usize>,
    required_evidence: Option<usize>,
    reasoning_trace: bool,
}

impl QueryContextBuilder {
    pub fn new() -> Self {
        Self {
            domain: None,
            confidence_threshold: 0.7,
            max_depth: Some(5),
            required_evidence: Some(1),
            reasoning_trace: false,
        }
    }
    
    pub fn with_domain(mut self, domain: String) -> Self {
        self.domain = Some(domain);
        self
    }
    
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }
    
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
    
    pub fn with_required_evidence(mut self, evidence: usize) -> Self {
        self.required_evidence = Some(evidence);
        self
    }
    
    pub fn with_reasoning_trace(mut self, trace: bool) -> Self {
        self.reasoning_trace = trace;
        self
    }
    
    pub fn build(self) -> crate::cognitive::QueryContext {
        use crate::cognitive::QueryContext;
        
        QueryContext {
            domain: self.domain,
            confidence_threshold: self.confidence_threshold,
            max_depth: self.max_depth,
            required_evidence: self.required_evidence,
            reasoning_trace: self.reasoning_trace,
            // Phase 4 extensions with defaults
            user_id: None,
            session_id: None,
            conversation_history: Vec::new(),
            domain_context: None,
            urgency_level: 0.5,
            expected_response_time: None,
            query_intent: None,
        }
    }
}

impl Default for QueryContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating test pattern parameters
pub struct PatternParametersBuilder {
    max_depth: Option<usize>,
    activation_threshold: Option<f32>,
    exploration_breadth: Option<usize>,
    creativity_threshold: Option<f32>,
    validation_level: Option<crate::cognitive::ValidationLevel>,
    pattern_type: Option<crate::cognitive::PatternType>,
    reasoning_strategy: Option<crate::cognitive::ReasoningStrategy>,
}

impl PatternParametersBuilder {
    pub fn new() -> Self {
        Self {
            max_depth: Some(5),
            activation_threshold: Some(0.5),
            exploration_breadth: Some(10),
            creativity_threshold: Some(0.3),
            validation_level: Some(crate::cognitive::ValidationLevel::Basic),
            pattern_type: Some(crate::cognitive::PatternType::Structural),
            reasoning_strategy: Some(crate::cognitive::ReasoningStrategy::Automatic),
        }
    }
    
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
    
    pub fn with_activation_threshold(mut self, threshold: f32) -> Self {
        self.activation_threshold = Some(threshold);
        self
    }
    
    pub fn with_exploration_breadth(mut self, breadth: usize) -> Self {
        self.exploration_breadth = Some(breadth);
        self
    }
    
    pub fn with_creativity_threshold(mut self, threshold: f32) -> Self {
        self.creativity_threshold = Some(threshold);
        self
    }
    
    pub fn with_validation_level(mut self, level: crate::cognitive::ValidationLevel) -> Self {
        self.validation_level = Some(level);
        self
    }
    
    pub fn with_pattern_type(mut self, pattern_type: crate::cognitive::PatternType) -> Self {
        self.pattern_type = Some(pattern_type);
        self
    }
    
    pub fn with_reasoning_strategy(mut self, strategy: crate::cognitive::ReasoningStrategy) -> Self {
        self.reasoning_strategy = Some(strategy);
        self
    }
    
    pub fn build(self) -> crate::cognitive::PatternParameters {
        use crate::cognitive::PatternParameters;
        
        PatternParameters {
            max_depth: self.max_depth,
            activation_threshold: self.activation_threshold,
            exploration_breadth: self.exploration_breadth,
            creativity_threshold: self.creativity_threshold,
            validation_level: self.validation_level,
            pattern_type: self.pattern_type,
            reasoning_strategy: self.reasoning_strategy,
        }
    }
}

impl Default for PatternParametersBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Test builder functions for benchmarking and cognitive system testing
/// 
/// These functions create simplified test instances for performance benchmarking
/// while maintaining compatibility with the cognitive architecture.

/// Create a test cognitive orchestrator for benchmarking
pub async fn build_test_cognitive_orchestrator() -> CognitiveOrchestrator {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph"));
    CognitiveOrchestrator::new(graph, CognitiveOrchestratorConfig::default())
        .await
        .expect("Failed to create test cognitive orchestrator")
}

/// Create a test attention manager for benchmarking
pub async fn build_test_attention_manager() -> AttentionManager {
    AttentionManagerBuilder::new()
        .build()
        .await
        .expect("Failed to create test attention manager")
}

/// Create a test working memory system for benchmarking
pub async fn build_test_working_memory() -> WorkingMemorySystem {
    WorkingMemoryBuilder::new()
        .build()
        .await
        .expect("Failed to create test working memory")
}

/// Create a test brain metrics collector for benchmarking
pub async fn build_test_brain_metrics_collector() -> BrainMetricsCollector {
    use tokio::sync::RwLock;
    let brain_graph = Arc::new(RwLock::new(BrainEnhancedKnowledgeGraph::new_for_test()
        .expect("Failed to create test graph")));
    BrainMetricsCollector::new(brain_graph)
}

/// Create a test performance monitor for benchmarking
pub async fn build_test_performance_monitor() -> PerformanceMonitor {
    PerformanceMonitor::new_with_defaults()
        .await
        .expect("Failed to create test performance monitor")
}

/// Create a test neural server for benchmarking
pub async fn build_test_neural_server() -> NeuralProcessingServer {
    NeuralProcessingServer::new_mock()
}

/// Create a test federation coordinator for benchmarking
pub async fn build_test_federation_coordinator() -> FederationCoordinator {
    use crate::federation::registry::DatabaseRegistry;
    let registry = Arc::new(DatabaseRegistry::new().expect("Failed to create registry"));
    
    // Use tokio runtime directly since this is test support
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create runtime");
    runtime.block_on(FederationCoordinator::new(registry)).expect("Failed to create federation coordinator")
}

/// Create a lightweight brain enhanced knowledge graph for benchmarking
pub fn build_test_brain_graph() -> BrainEnhancedKnowledgeGraph {
    BrainEnhancedKnowledgeGraph::new_for_test()
        .expect("Failed to create test brain graph")
}