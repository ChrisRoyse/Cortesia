use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

use crate::core::types::EntityKey;
use crate::core::brain_types::ActivationStep;
use crate::core::brain_types::RelationType;
use crate::error::Result;
use std::time::{Duration, Instant};

/// Context for query processing
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    pub domain: Option<String>,
    pub confidence_threshold: f32,
    pub max_depth: Option<usize>,
    pub required_evidence: Option<usize>,
    pub reasoning_trace: bool,
    // Phase 4 extensions
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub conversation_history: Vec<String>,
    pub domain_context: Option<String>,
    pub urgency_level: f32,
    pub expected_response_time: Option<Duration>,
    pub query_intent: Option<String>,
}

impl QueryContext {
    pub fn new() -> Self {
        Self {
            domain: None,
            confidence_threshold: 0.7,
            max_depth: Some(5),
            required_evidence: Some(1),
            reasoning_trace: false,
            // Phase 4 extensions
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

/// Core trait for all cognitive patterns
#[async_trait]
pub trait CognitivePattern: Send + Sync {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult>;
    
    fn get_pattern_type(&self) -> CognitivePatternType;
    fn get_optimal_use_cases(&self) -> Vec<String>;
    fn estimate_complexity(&self, query: &str) -> ComplexityEstimate;
}

/// Types of cognitive patterns available
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitivePatternType {
    Convergent,
    Divergent,
    Lateral,
    Systems,
    Critical,
    Abstract,
    Adaptive,
    ChainOfThought,
    TreeOfThoughts,
    Analytical,
    PatternRecognition,
    Linguistic,
    Creative,
    Ensemble,
    Unknown,
}

/// Parameters for cognitive pattern execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternParameters {
    pub max_depth: Option<usize>,
    pub activation_threshold: Option<f32>,
    pub exploration_breadth: Option<usize>,
    pub creativity_threshold: Option<f32>,
    pub validation_level: Option<ValidationLevel>,
    pub pattern_type: Option<PatternType>,
    pub reasoning_strategy: Option<ReasoningStrategy>,
}

impl Default for PatternParameters {
    fn default() -> Self {
        Self {
            max_depth: Some(5),
            activation_threshold: Some(0.5),
            exploration_breadth: Some(10),
            creativity_threshold: Some(0.3),
            validation_level: Some(ValidationLevel::Basic),
            pattern_type: Some(PatternType::Structural),
            reasoning_strategy: Some(ReasoningStrategy::Automatic),
        }
    }
}

/// Result from cognitive pattern execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    pub pattern_type: CognitivePatternType,
    pub answer: String,
    pub confidence: f32,
    pub reasoning_trace: Vec<ActivationStep>,
    pub metadata: ResultMetadata,
}

/// Metadata about pattern execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    pub execution_time_ms: u64,
    pub nodes_activated: usize,
    pub iterations_completed: usize,
    pub converged: bool,
    pub total_energy: f32,
    pub additional_info: HashMap<String, String>,
}

/// Single step in activation propagation

/// Types of activation operations

/// Complexity estimate for query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityEstimate {
    pub computational_complexity: u32,
    pub estimated_time_ms: u64,
    pub memory_requirements_mb: u32,
    pub confidence: f32,
    pub parallelizable: bool,
}

/// Strategy for reasoning execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    Automatic,
    Specific(CognitivePatternType),
    Ensemble(Vec<CognitivePatternType>),
}

/// Validation levels for critical thinking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationLevel {
    Basic,
    Comprehensive,
    Rigorous,
}

/// Pattern types for abstract thinking
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    Structural,
    Temporal,
    Semantic,
    Usage,
}

/// Types of exploration for divergent thinking
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExplorationType {
    Instances,
    Categories,
    Properties,
    Associations,
    Creative,
}

/// Types of systems reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemsReasoningType {
    AttributeInheritance,
    Classification,
    SystemAnalysis,
    EmergentProperties,
}

/// Analysis scope for abstract thinking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisScope {
    Local(EntityKey),
    Regional(Vec<EntityKey>),
    Global,
}

/// Results from convergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergentResult {
    pub answer: String,
    pub confidence: f32,
    pub reasoning_trace: Vec<ActivationStep>,
    pub supporting_facts: Vec<EntityKey>,
    // Advanced fields for enhanced cognitive processing
    pub execution_time_ms: u64,
    pub semantic_similarity_score: f32,
    pub attention_weights: Vec<f32>,
    pub uncertainty_estimate: f32,
}

/// Results from divergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergentResult {
    pub explorations: Vec<ExplorationPath>,
    pub creativity_scores: Vec<f32>,
    pub total_paths_explored: usize,
}

/// Single exploration path in divergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPath {
    pub path: Vec<EntityKey>,
    pub concepts: Vec<String>,
    pub concept: String,  // Primary concept for this path
    pub relevance_score: f32,
    pub novelty_score: f32,
}

/// Results from lateral thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralResult {
    pub bridges: Vec<BridgePath>,
    pub novelty_analysis: NoveltyAnalysis,
    pub confidence_distribution: Vec<f32>,
}

/// Bridge path connecting disparate concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePath {
    pub path: Vec<EntityKey>,
    pub intermediate_concepts: Vec<String>,
    pub novelty_score: f32,
    pub plausibility_score: f32,
    pub explanation: String,
    pub bridge_id: String,
    pub start_concept: String,
    pub end_concept: String,
    pub bridge_concepts: Vec<String>,
    pub creativity_score: f32,
    pub connection_strength: f32,
}

/// Analysis of novelty in lateral thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyAnalysis {
    pub overall_novelty: f32,
    pub concept_uniqueness: Vec<f32>,
    pub path_creativity: f32,
    pub average_novelty: f32,
    pub highest_novelty_path: Option<String>,
    pub top_creative_insights: Vec<String>,
}

/// Results from systems thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemsResult {
    pub hierarchy_path: Vec<EntityKey>,
    pub inherited_attributes: Vec<InheritedAttribute>,
    pub exception_handling: Vec<Exception>,
    pub system_complexity: f32,
}

/// Attribute inherited through hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritedAttribute {
    pub attribute_name: String,
    pub value: String,
    pub source_entity: EntityKey,
    pub inheritance_depth: usize,
    pub confidence: f32,
}

/// Exception in systems reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exception {
    pub exception_type: ExceptionType,
    pub description: String,
    pub affected_entities: Vec<EntityKey>,
    pub resolution_strategy: String,
}

/// Types of exceptions in reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionType {
    Contradiction,
    MissingData,
    InconsistentInheritance,
    CircularReference,
}

/// Results from critical thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalResult {
    pub resolved_facts: Vec<ResolvedFact>,
    pub contradictions_found: Vec<Contradiction>,
    pub resolution_strategy: ResolutionStrategy,
    pub confidence_intervals: Vec<ConfidenceInterval>,
    pub uncertainty_analysis: UncertaintyAnalysis,
}

/// Fact resolved through critical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedFact {
    pub fact_statement: String,
    pub confidence: f32,
    pub supporting_evidence: Vec<EntityKey>,
    pub conflicting_evidence: Vec<EntityKey>,
}

/// Contradiction found in knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub statement_a: String,
    pub statement_b: String,
    pub conflict_type: ConflictType,
    pub severity: f32,
    pub contradiction_type: String,
    pub conflicting_facts: Vec<EntityKey>,
}

/// Types of conflicts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    DirectContradiction,
    InheritanceConflict,
    TemporalInconsistency,
    SourceDisagreement,
}

/// Strategy for resolving contradictions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    PreferLocal,
    PreferTrusted,
    WeightedAverage,
    ExpertSystem,
    PreferHigherConfidence,
    LogicalPriority,
}

/// Confidence interval for facts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub entity_key: EntityKey,
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub mean_confidence: f32,
}

/// Analysis of uncertainty in reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyAnalysis {
    pub overall_uncertainty: f32,
    pub source_reliability: HashMap<String, f32>,
    pub knowledge_gaps: Vec<String>,
}

/// Results from abstract thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractResult {
    pub patterns_found: Vec<DetectedPattern>,
    pub abstractions: Vec<AbstractionCandidate>,
    pub refactoring_opportunities: Vec<RefactoringOpportunity>,
    pub efficiency_gains: EfficiencyAnalysis,
}

/// Pattern detected in abstract analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_id: String,
    pub id: String,
    pub pattern_type: PatternType,
    pub confidence: f32,
    pub entities_involved: Vec<EntityKey>,
    pub affected_entities: Vec<EntityKey>,
    pub description: String,
    pub frequency: f32,
}

/// Candidate for abstraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionCandidate {
    pub abstraction_name: String,
    pub entities_to_abstract: Vec<EntityKey>,
    pub potential_savings: f32,
    pub implementation_complexity: u32,
    pub complexity_reduction: f32,
    pub implementation_effort: f32,
    pub abstraction_type: String,
    pub source_patterns: Vec<String>,
}

/// Opportunity for graph refactoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringOpportunity {
    pub opportunity_type: RefactoringType,
    pub description: String,
    pub entities_affected: Vec<EntityKey>,
    pub estimated_benefit: f32,
}

/// Types of refactoring opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefactoringType {
    ConceptMerging,
    HierarchyReorganization,
    RedundancyElimination,
    PerformanceOptimization,
}

/// Analysis of efficiency improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyAnalysis {
    pub query_time_improvement: f32,
    pub memory_reduction: f32,
    pub accuracy_improvement: f32,
    pub maintainability_score: f32,
}

/// Results from adaptive thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveResult {
    pub final_answer: String,
    pub strategy_used: StrategySelection,
    pub pattern_contributions: Vec<PatternContribution>,
    pub confidence_distribution: ConfidenceDistribution,
    pub learning_update: LearningUpdate,
}

/// Strategy selection details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelection {
    pub selected_patterns: Vec<CognitivePatternType>,
    pub selection_confidence: f32,
    pub reasoning: String,
    pub execution_order: Vec<usize>,
}

/// Contribution from individual pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternContribution {
    pub pattern_type: CognitivePatternType,
    pub contribution_weight: f32,
    pub partial_result: String,
    pub confidence: f32,
}

/// Distribution of confidence across patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceDistribution {
    pub mean_confidence: f32,
    pub variance: f32,
    pub individual_confidences: Vec<f32>,
    pub ensemble_confidence: f32,
}

/// Learning update from pattern execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningUpdate {
    pub performance_feedback: f32,
    pub strategy_effectiveness: f32,
    pub recommended_adjustments: Vec<String>,
    pub model_updates: Vec<ModelUpdate>,
}

/// Update to cognitive models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    pub model_id: String,
    pub update_type: UpdateType,
    pub update_data: Vec<f32>,
    pub confidence: f32,
}

/// Types of model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    WeightAdjustment,
    ParameterTuning,
    ArchitectureModification,
    TrainingDataAddition,
}

/// Query characteristics for adaptive pattern selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCharacteristics {
    pub complexity_score: f32,
    pub ambiguity_level: f32,
    pub domain_specificity: f32,
    pub temporal_aspect: bool,
    pub creative_requirement: f32,
    pub factual_focus: f32,
    pub abstraction_level: f32,
}

/// Result of reasoning execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    pub query: String,
    pub final_answer: String,
    pub strategy_used: ReasoningStrategy,
    pub execution_metadata: ExecutionMetadata,
    pub quality_metrics: QualityMetrics,
}

/// Metadata about reasoning execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub total_time_ms: u64,
    pub patterns_executed: Vec<CognitivePatternType>,
    pub nodes_activated: usize,
    pub energy_consumed: f32,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Quality metrics for reasoning results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_confidence: f32,
    pub consistency_score: f32,
    pub completeness_score: f32,
    pub novelty_score: f32,
    pub efficiency_score: f32,
}

/// Error types specific to cognitive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveError {
    PatternNotFound(CognitivePatternType),
    InvalidParameters(String),
    ActivationFailure(String),
    InsufficientData(String),
    ContradictionUnresolved(String),
    TimeoutError(u64),
    ModelLoadError(String),
}

impl std::fmt::Display for CognitiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitiveError::PatternNotFound(pattern) => {
                write!(f, "Cognitive pattern not found: {:?}", pattern)
            }
            CognitiveError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            CognitiveError::ActivationFailure(msg) => {
                write!(f, "Activation failure: {}", msg)
            }
            CognitiveError::InsufficientData(msg) => {
                write!(f, "Insufficient data: {}", msg)
            }
            CognitiveError::ContradictionUnresolved(msg) => {
                write!(f, "Contradiction unresolved: {}", msg)
            }
            CognitiveError::TimeoutError(timeout) => {
                write!(f, "Operation timed out after {} ms", timeout)
            }
            CognitiveError::ModelLoadError(msg) => {
                write!(f, "Model load error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CognitiveError {}

// Display implementations for enums that need formatting
impl std::fmt::Display for ExceptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExceptionType::Contradiction => write!(f, "Contradiction"),
            ExceptionType::MissingData => write!(f, "Missing Data"),
            ExceptionType::InconsistentInheritance => write!(f, "Inconsistent Inheritance"),
            ExceptionType::CircularReference => write!(f, "Circular Reference"),
        }
    }
}

impl std::fmt::Display for RefactoringType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefactoringType::ConceptMerging => write!(f, "Concept Merging"),
            RefactoringType::HierarchyReorganization => write!(f, "Hierarchy Reorganization"),
            RefactoringType::RedundancyElimination => write!(f, "Redundancy Elimination"),
            RefactoringType::PerformanceOptimization => write!(f, "Performance Optimization"),
        }
    }
}

impl std::fmt::Display for CognitivePatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitivePatternType::Convergent => write!(f, "Convergent"),
            CognitivePatternType::Divergent => write!(f, "Divergent"),
            CognitivePatternType::Lateral => write!(f, "Lateral"),
            CognitivePatternType::Systems => write!(f, "Systems"),
            CognitivePatternType::Critical => write!(f, "Critical"),
            CognitivePatternType::Abstract => write!(f, "Abstract"),
            CognitivePatternType::Adaptive => write!(f, "Adaptive"),
            CognitivePatternType::ChainOfThought => write!(f, "ChainOfThought"),
            CognitivePatternType::TreeOfThoughts => write!(f, "TreeOfThoughts"),
        }
    }
}

/// Exploration map for divergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationMap {
    pub starting_entities: Vec<EntityKey>,
    pub exploration_waves: Vec<ExplorationWave>,
    pub total_entities_explored: usize,
    pub exploration_depth: usize,
    pub edges: Vec<ExplorationEdge>,
    pub neighbors_cache: HashMap<EntityKey, Vec<EntityKey>>,
    pub activated_nodes: HashMap<String, f32>,
    pub total_concepts_explored: usize,
    pub creative_bridges_found: usize,
}

/// Edge in exploration map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationEdge {
    pub from: EntityKey,
    pub to: EntityKey,
    pub relation_type: RelationType,
    pub weight: f32,
}

impl ExplorationMap {
    pub fn new() -> Self {
        Self {
            starting_entities: Vec::new(),
            exploration_waves: Vec::new(),
            total_entities_explored: 0,
            exploration_depth: 0,
            edges: Vec::new(),
            neighbors_cache: HashMap::new(),
            activated_nodes: HashMap::new(),
            total_concepts_explored: 0,
            creative_bridges_found: 0,
        }
    }
    
    pub fn add_node(&mut self, entity_key: EntityKey, activation: f32, depth: usize) {
        if depth >= self.exploration_waves.len() {
            self.exploration_waves.resize(depth + 1, ExplorationWave {
                wave_id: depth,
                entities: Vec::new(),
                activation_levels: Vec::new(),
                exploration_type: ExplorationType::Instances,
            });
        }
        
        self.exploration_waves[depth].entities.push(entity_key);
        self.exploration_waves[depth].activation_levels.push(activation);
        self.total_entities_explored += 1;
        self.exploration_depth = self.exploration_depth.max(depth);
        
        if depth == 0 {
            self.starting_entities.push(entity_key);
        }
    }
    
    pub fn add_edge(&mut self, from: EntityKey, to: EntityKey, relation_type: RelationType, weight: f32) {
        self.edges.push(ExplorationEdge {
            from,
            to,
            relation_type,
            weight,
        });
        
        // Update neighbors cache
        self.neighbors_cache.entry(from).or_insert_with(Vec::new).push(to);
        self.neighbors_cache.entry(to).or_insert_with(Vec::new).push(from);
    }
    
    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<EntityKey> {
        if depth < self.exploration_waves.len() {
            self.exploration_waves[depth].entities.clone()
        } else {
            Vec::new()
        }
    }
    
    pub fn get_high_activation_endpoints(&self, count: usize) -> Vec<EntityKey> {
        let mut all_entities = Vec::new();
        
        for wave in &self.exploration_waves {
            for (i, &entity_key) in wave.entities.iter().enumerate() {
                if i < wave.activation_levels.len() {
                    all_entities.push((entity_key, wave.activation_levels[i]));
                }
            }
        }
        
        // Sort by activation level
        all_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top count entities
        all_entities.into_iter().take(count).map(|(key, _)| key).collect()
    }
    
    pub fn get_neighbors(&self, entity_key: EntityKey) -> Vec<EntityKey> {
        self.neighbors_cache.get(&entity_key).cloned().unwrap_or_else(Vec::new)
    }
}

/// Single wave of exploration in divergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationWave {
    pub wave_id: usize,
    pub entities: Vec<EntityKey>,
    pub activation_levels: Vec<f32>,
    pub exploration_type: ExplorationType,
}

/// Result of propagation in convergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationResult {
    pub best_output: String,
    pub confidence: f32,
    pub activation_path: Vec<ActivationStep>,
    pub supporting_entities: Vec<EntityKey>,
}

/// Best answer extracted from convergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestAnswer {
    pub answer: String,
    pub confidence: f32,
    pub supporting_entities: Vec<EntityKey>,
}

/// Hierarchy traversal result for systems thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyTraversal {
    pub path: Vec<EntityKey>,
    pub exceptions: Vec<Exception>,
    pub traversal_depth: usize,
}

/// Cache for hierarchy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyCache {
    pub cached_hierarchies: HashMap<EntityKey, Vec<EntityKey>>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl HierarchyCache {
    pub fn new() -> Self {
        Self {
            cached_hierarchies: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

/// Resolution for inhibitory logic in critical thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InhibitoryResolution {
    pub resolved_facts: Vec<ResolvedFact>,
    pub strategy: ResolutionStrategy,
}

/// Structural patterns for abstract thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralPatterns {
    pub node_patterns: Vec<NodePattern>,
    pub edge_patterns: Vec<EdgePattern>,
    pub subgraph_patterns: Vec<SubgraphPattern>,
}

/// Individual node pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub entities: Vec<EntityKey>,
}

/// Individual edge pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub relationships: Vec<(EntityKey, EntityKey)>,
}

/// Subgraph pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphPattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub entities: Vec<EntityKey>,
    pub edges: Vec<(EntityKey, EntityKey)>,
}

/// Performance metrics for the cognitive orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_queries_processed: u64,
    pub average_response_time_ms: f64,
    pub pattern_usage_stats: HashMap<CognitivePatternType, u64>,
    pub success_rate: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub active_entities: u64,
}

// Graph query types for GraphQueryEngine


#[derive(Debug, Clone)]
pub struct Pattern {
    pub pattern_id: String,
    pub entities: Vec<EntityKey>,
    pub confidence: f32,
    pub pattern_type: String,
}

#[derive(Debug, Clone)]
pub struct Path {
    pub path_id: String,
    pub entities: Vec<EntityKey>,
    pub total_weight: f32,
    pub path_length: usize,
}

#[derive(Debug, Clone)]
pub struct Subgraph {
    pub entities: Vec<EntityKey>,
    pub connections: Vec<(EntityKey, EntityKey, f32)>,
    pub center: Option<EntityKey>,
}

#[derive(Debug, Clone)]
pub struct StructureAnalysis {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f32,
    pub clusters: Vec<Vec<EntityKey>>,
}

#[derive(Debug, Clone)]
pub struct Synthesis {
    pub central_concept: EntityKey,
    pub supporting_patterns: Vec<EntityKey>,
    pub confidence: f32,
    pub explanation: String,
}

// Memory-related types for cognitive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIndex {
    pub buffer_type: String,
    pub item_id: uuid::Uuid,
    pub importance_score: f32,
    #[serde(skip)]
    pub last_accessed: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionGuidance {
    pub focus_entities: Vec<EntityKey>,
    pub attention_weights: Vec<f32>,
    pub focus_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsolidationResult {
    pub items_consolidated: usize,
    pub items_evicted: usize,
    pub average_importance: f32,
    pub consolidation_time_ms: u64,
}

