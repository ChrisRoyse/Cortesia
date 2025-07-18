use crate::core::types::EntityKey;
use crate::cognitive::types::CognitivePatternType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

// LearningResult struct (moved from meta_learning to avoid circular dependency)
#[derive(Debug, Clone)]
pub struct LearningResult {
    pub success: bool,
    pub performance_achieved: f32,
    pub learning_efficiency: f32,
    pub generalization_score: f32,
    pub resource_efficiency: f32,
    pub insights_gained: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationEvent {
    pub entity_key: EntityKey,
    pub activation_strength: f32,
    pub timestamp: Instant,
    pub context: ActivationContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationContext {
    pub query_id: String,
    pub cognitive_pattern: CognitivePatternType,
    pub user_session: Option<String>,
    pub outcome_quality: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningContext {
    pub performance_pressure: f32,
    pub user_satisfaction_level: f32,
    pub learning_urgency: f32,
    pub session_id: String,
    pub learning_goals: Vec<LearningGoal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningGoal {
    pub goal_type: LearningGoalType,
    pub target_improvement: f32,
    pub deadline: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningGoalType {
    PerformanceImprovement,
    MemoryEfficiency,
    ResponseAccuracy,
    UserSatisfaction,
}

#[derive(Debug, Clone)]
pub struct LearningUpdate {
    pub strengthened_connections: Vec<WeightChange>,
    pub weakened_connections: Vec<WeightChange>,
    pub new_connections: Vec<WeightChange>,
    pub pruned_connections: Vec<WeightChange>,
    pub learning_efficiency: f32,
    pub inhibition_updates: Vec<InhibitionChange>,
}

#[derive(Debug, Clone)]
pub struct WeightChange {
    pub source: EntityKey,
    pub target: EntityKey,
    pub old_weight: f32,
    pub new_weight: f32,
    pub change_magnitude: f32,
}

#[derive(Debug, Clone)]
pub struct InhibitionChange {
    pub competition_group: String,
    pub entities_affected: Vec<EntityKey>,
    pub strength_change: f32,
    pub change_reason: InhibitionChangeReason,
}

#[derive(Debug, Clone)]
pub enum InhibitionChangeReason {
    HebbianLearning,
    CompetitionOptimization,
    PerformanceImprovement,
    UserFeedback,
}

#[derive(Debug, Clone)]
pub struct CorrelationUpdate {
    pub source_entity: EntityKey,
    pub target_entity: EntityKey,
    pub source_activation: f32,
    pub target_activation: f32,
    pub correlation_strength: f32,
    pub creates_competition: bool,
    pub is_inhibitory: bool,
}

#[derive(Debug, Clone)]
pub struct LearningStatistics {
    pub total_weight_changes: u64,
    pub successful_predictions: u64,
    pub failed_predictions: u64,
    pub average_learning_rate: f32,
    pub learning_stability: f32,
    pub convergence_metrics: ConvergenceMetrics,
}

#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub parameter_stability: f32,
    pub performance_stability: f32,
    pub last_significant_change: SystemTime,
    pub convergence_confidence: f32,
}

impl LearningStatistics {
    pub fn new() -> Self {
        Self {
            total_weight_changes: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_learning_rate: 0.01,
            learning_stability: 1.0,
            convergence_metrics: ConvergenceMetrics {
                parameter_stability: 0.0,
                performance_stability: 0.0,
                last_significant_change: SystemTime::now(),
                convergence_confidence: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoactivationTracker {
    pub activation_history: HashMap<EntityKey, Vec<ActivationEvent>>,
    pub correlation_matrix: HashMap<(EntityKey, EntityKey), f32>,
    pub temporal_window: Duration,
    pub correlation_threshold: f32,
}

impl CoactivationTracker {
    pub fn new() -> Self {
        Self {
            activation_history: HashMap::new(),
            correlation_matrix: HashMap::new(),
            temporal_window: Duration::from_secs(300), // 5 minutes
            correlation_threshold: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub enum STDPResult {
    NoChange,
    WeightChanged {
        weight_change: f32,
        timing_difference: f32,
        plasticity_type: PlasticityType,
    },
}

#[derive(Debug, Clone)]
pub enum PlasticityType {
    Potentiation,
    Depression,
}

#[derive(Debug, Clone)]
pub struct OptimizationOpportunities {
    pub patterns_detected: Vec<DetectedPattern>,
    pub optimization_candidates: Vec<OptimizationCandidate>,
    pub efficiency_predictions: EfficiencyPredictions,
    pub priority_ranking: Vec<OptimizationPriority>,
    pub hebbian_insights: HebbianInsights,
}

#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub confidence: f32,
    pub affected_entities: Vec<EntityKey>,
    pub frequency: u32,
    pub impact_score: f32,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Structural,
    Temporal,
    Semantic,
    Usage,
    CommonAttributePattern,
    HierarchicalDuplication,
    FrequentSubgraph,
    SparseConnection,
}

#[derive(Debug, Clone)]
pub enum OptimizationCandidate {
    AttributeBubbling(AttributeBubblingCandidate),
    HierarchyConsolidation(HierarchyConsolidationCandidate),
    SubgraphFactorization(SubgraphFactorizationCandidate),
    ConnectionPruning(ConnectionPruningCandidate),
}

#[derive(Debug, Clone)]
pub struct AttributeBubblingCandidate {
    pub pattern_id: String,
    pub bubbling_opportunities: Vec<BubblingOpportunity>,
    pub total_entities_affected: usize,
    pub estimated_storage_reduction: f32,
    pub efficiency_gain: f32,
}

#[derive(Debug, Clone)]
pub struct BubblingOpportunity {
    pub attribute: String,
    pub source_entities: Vec<EntityKey>,
    pub target_parent: EntityKey,
    pub coverage_percentage: f32,
    pub efficiency_gain: f32,
}

#[derive(Debug, Clone)]
pub struct HierarchyConsolidationCandidate {
    pub consolidation_benefit: f32,
    pub redundant_hierarchies: Vec<EntityKey>,
    pub target_hierarchy: EntityKey,
}

#[derive(Debug, Clone)]
pub struct SubgraphFactorizationCandidate {
    pub frequency: u32,
    pub size: usize,
    pub subgraph_entities: Vec<EntityKey>,
    pub factorization_benefit: f32,
}

#[derive(Debug, Clone)]
pub struct ConnectionPruningCandidate {
    pub pruning_safety: f32,
    pub weak_connections: Vec<(EntityKey, EntityKey)>,
    pub expected_performance_impact: f32,
}

#[derive(Debug, Clone)]
pub struct EfficiencyPredictions {
    pub storage_improvement: f32,
    pub query_speed_improvement: f32,
    pub memory_usage_reduction: f32,
    pub overall_efficiency_gain: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizationPriority {
    pub candidate_id: String,
    pub priority_score: f32,
    pub implementation_difficulty: f32,
    pub expected_benefit: f32,
}

#[derive(Debug, Clone)]
pub struct HebbianInsights {
    pub learning_patterns: Vec<String>,
    pub connection_strengths: HashMap<(EntityKey, EntityKey), f32>,
    pub competitive_dynamics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CoordinatedLearningResults {
    pub session_id: Uuid,
    pub hebbian_results: Option<LearningUpdate>,
    pub homeostasis_results: Option<crate::learning::homeostasis::HomeostasisUpdate>,
    pub optimization_results: Option<OptimizationResult>,
    pub adaptive_results: Option<AdaptiveLearningResult>,
    pub coordination_quality: f32,
    pub inter_system_conflicts: Vec<String>,
    pub overall_learning_effectiveness: f32,
}

#[derive(Debug, Clone)]
pub struct AdaptiveLearningResult {
    pub cycle_id: Uuid,
    pub duration: Duration,
    pub hebbian_updates: LearningUpdate,
    pub optimization_updates: OptimizationResult,
    pub optimization_result: OptimizationResult,
    pub cognitive_updates: CognitiveParameterUpdates,
    pub orchestration_updates: OrchestrationUpdates,
    pub performance_improvement: f32,
    pub next_cycle_schedule: SystemTime,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimizations_applied: Vec<AppliedOptimization>,
    pub performance_impact: f32,
    pub stability_impact: f32,
    pub rollback_available: bool,
}

#[derive(Debug, Clone)]
pub struct AppliedOptimization {
    pub optimization_type: OptimizationType,
    pub entities_affected: Vec<EntityKey>,
    pub performance_improvement: f32,
    pub implementation_timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    AttributeBubbling,
    HierarchyConsolidation,
    SubgraphFactorization,
    ConnectionPruning,
    ParameterTuning,
}

#[derive(Debug, Clone)]
pub struct CognitiveParameterUpdates {
    pub attention_parameters: AttentionParameterUpdates,
    pub memory_parameters: MemoryParameterUpdates,
    pub inhibition_parameters: InhibitionParameterUpdates,
}

#[derive(Debug, Clone)]
pub struct AttentionParameterUpdates {
    pub focus_strength_adjustment: f32,
    pub shift_speed_adjustment: f32,
    pub capacity_adjustment: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryParameterUpdates {
    pub capacity_adjustments: HashMap<String, f32>,
    pub decay_rate_adjustments: HashMap<String, f32>,
    pub consolidation_threshold_adjustments: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct InhibitionParameterUpdates {
    pub competition_strength_adjustments: HashMap<String, f32>,
    pub threshold_adjustments: HashMap<String, f32>,
    pub new_competition_groups: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OrchestrationUpdates {
    pub pattern_selection_adjustments: HashMap<CognitivePatternType, f32>,
    pub ensemble_weight_adjustments: HashMap<CognitivePatternType, f32>,
    pub strategy_preference_updates: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct UserFeedback {
    pub feedback_id: Uuid,
    pub session_id: String,
    pub query_id: String,
    pub satisfaction_score: f32,
    pub response_quality: f32,
    pub response_speed: f32,
    pub accuracy_rating: f32,
    pub feedback_text: Option<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub query_latencies: Vec<Duration>,
    pub memory_usage: Vec<f32>,
    pub accuracy_scores: Vec<f32>,
    pub user_satisfaction: Vec<f32>,
    pub system_stability: f32,
    pub error_rates: HashMap<String, f32>,
    pub throughput_metrics: ThroughputMetrics,
    pub timestamp: SystemTime,
    pub system_health: f32,
    pub overall_performance_score: f32,
    pub component_scores: HashMap<String, f32>,
    pub bottlenecks: Vec<String>,
}

impl Default for PerformanceData {
    fn default() -> Self {
        Self {
            query_latencies: Vec::new(),
            memory_usage: Vec::new(),
            accuracy_scores: Vec::new(),
            user_satisfaction: Vec::new(),
            system_stability: 0.8,
            error_rates: HashMap::new(),
            throughput_metrics: ThroughputMetrics {
                queries_per_second: 10.0,
                successful_queries: 100,
                failed_queries: 5,
                average_response_time: Duration::from_millis(200),
            },
            timestamp: SystemTime::now(),
            system_health: 0.8,
            overall_performance_score: 0.7,
            component_scores: HashMap::new(),
            bottlenecks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub queries_per_second: f32,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_response_time: Duration,
}

#[derive(Debug, Clone)]
pub struct LearningAnalysis {
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub satisfaction_patterns: SatisfactionAnalysis,
    pub performance_correlations: CorrelationAnalysis,
    pub learning_targets: Vec<LearningTarget>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub affected_components: Vec<String>,
    pub suggested_improvements: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    Memory,
    Computation,
    Attention,
    Inhibition,
    Orchestration,
}

#[derive(Debug, Clone)]
pub struct SatisfactionAnalysis {
    pub satisfaction_trends: Vec<f32>,
    pub problem_areas: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub performance_satisfaction_correlation: f32,
    pub speed_satisfaction_correlation: f32,
    pub accuracy_satisfaction_correlation: f32,
    pub significant_correlations: Vec<(String, String, f32)>,
}

#[derive(Debug, Clone)]
pub struct LearningTarget {
    pub target_type: LearningTargetType,
    pub importance: f32,
    pub feasibility: f32,
    pub expected_impact: f32,
    pub implementation_plan: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum LearningTargetType {
    WeightAdjustment,
    ParameterTuning,
    StructureOptimization,
    BehaviorModification,
}