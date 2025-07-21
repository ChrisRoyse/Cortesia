//! Type definitions for the competitive inhibition system

use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::cognitive::types::CognitivePatternType;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct InhibitionMatrix {
    pub lateral_inhibition: HashMap<(EntityKey, EntityKey), f32>,
    pub hierarchical_inhibition: HashMap<(EntityKey, EntityKey), f32>,
    pub contextual_inhibition: HashMap<(EntityKey, EntityKey), f32>,
    pub temporal_inhibition: HashMap<(EntityKey, EntityKey), f32>,
}

impl InhibitionMatrix {
    pub fn new() -> Self {
        Self {
            lateral_inhibition: HashMap::new(),
            hierarchical_inhibition: HashMap::new(),
            contextual_inhibition: HashMap::new(),
            temporal_inhibition: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompetitionGroup {
    pub group_id: String,
    pub competing_entities: Vec<EntityKey>,
    pub competition_type: CompetitionType,
    pub winner_takes_all: bool,
    pub inhibition_strength: f32,
    pub priority: f32,
    pub temporal_dynamics: TemporalDynamics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompetitionType {
    Semantic,      // Competing semantic concepts
    Temporal,      // Competing temporal states
    Hierarchical,  // Different levels of abstraction
    Contextual,    // Context-dependent competition
    Spatial,       // Spatial competition
    Causal,        // Causal relationship competition
}

#[derive(Debug, Clone)]
pub struct TemporalDynamics {
    pub onset_delay: Duration,
    pub peak_time: Duration,
    pub decay_time: Duration,
    pub oscillation_frequency: Option<f32>,
}

impl Default for TemporalDynamics {
    fn default() -> Self {
        Self {
            onset_delay: Duration::from_millis(10),
            peak_time: Duration::from_millis(50),
            decay_time: Duration::from_millis(200),
            oscillation_frequency: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InhibitionConfig {
    pub global_inhibition_strength: f32,
    pub lateral_inhibition_strength: f32,
    pub hierarchical_inhibition_strength: f32,
    pub contextual_inhibition_strength: f32,
    pub winner_takes_all_threshold: f32,
    pub soft_competition_factor: f32,
    pub temporal_integration_window: Duration,
    pub enable_learning: bool,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self {
            global_inhibition_strength: 0.5,
            lateral_inhibition_strength: 0.7,
            hierarchical_inhibition_strength: 0.6,
            contextual_inhibition_strength: 0.4,
            winner_takes_all_threshold: 0.8,
            soft_competition_factor: 0.3,
            temporal_integration_window: Duration::from_millis(100),
            enable_learning: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InhibitionResult {
    pub competition_results: Vec<GroupCompetitionResult>,
    pub hierarchical_result: HierarchicalInhibitionResult,
    pub exception_result: ExceptionHandlingResult,
    pub final_pattern: ActivationPattern,
    pub inhibition_strength_applied: f32,
}

#[derive(Debug, Clone)]
pub struct GroupCompetitionResult {
    pub group_id: String,
    pub pre_competition: Vec<(EntityKey, f32)>,
    pub post_competition: Vec<(EntityKey, f32)>,
    pub winner: Option<EntityKey>,
    pub competition_intensity: f32,
    pub suppressed_entities: Vec<EntityKey>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalInhibitionResult {
    pub hierarchical_layers: Vec<HierarchicalLayer>,
    pub specificity_winners: Vec<EntityKey>,
    pub generality_suppressed: Vec<EntityKey>,
    pub abstraction_levels: HashMap<EntityKey, u32>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalLayer {
    pub layer_level: u32,
    pub entities: Vec<EntityKey>,
    pub inhibition_strength: f32,
    pub dominant_entity: Option<EntityKey>,
}

#[derive(Debug, Clone)]
pub struct ExceptionHandlingResult {
    pub exceptions_detected: Vec<InhibitionException>,
    pub resolutions_applied: Vec<ExceptionResolution>,
    pub unresolved_conflicts: Vec<String>,
    pub pattern_modified: bool,
}

#[derive(Debug, Clone)]
pub enum InhibitionException {
    MutualExclusion(EntityKey, EntityKey),
    TemporalConflict(EntityKey, EntityKey),
    HierarchicalInconsistency(EntityKey, EntityKey),
    ResourceContention(Vec<EntityKey>),
    CausalViolation(EntityKey, EntityKey),
    ContextualInappropriateness(EntityKey, String),
}

#[derive(Debug, Clone)]
pub struct ExceptionResolution {
    pub exception_type: String,
    pub affected_entities: Vec<EntityKey>,
    pub resolution_strategy: ResolutionStrategy,
    pub effectiveness: f32,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    Suppression(EntityKey),
    TemporalSequencing(Vec<EntityKey>),
    HierarchicalReordering,
    ResourceAllocation(HashMap<EntityKey, f32>),
    ContextualAdjustment,
    CausalReordering,
}

#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub cognitive_patterns_involved: Vec<CognitivePatternType>,
    pub pattern_specific_inhibitions: Vec<PatternSpecificInhibition>,
    pub cross_pattern_conflicts: Vec<String>,
    pub integration_success: bool,
}

#[derive(Debug, Clone)]
pub struct PatternSpecificInhibition {
    pub pattern_type: CognitivePatternType,
    pub inhibition_profile: InhibitionProfile,
    pub entities_affected: Vec<EntityKey>,
}

#[derive(Debug, Clone)]
pub struct InhibitionProfile {
    pub convergent_factor: f32,
    pub divergent_factor: f32,
    pub lateral_spread: f32,
    pub critical_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct InhibitionPerformanceMetrics {
    pub timestamp: SystemTime,
    pub processing_time: Duration,
    pub processing_time_ms: f64,
    pub entities_processed: usize,
    pub competition_groups_resolved: usize,
    pub competitions_resolved: usize,
    pub exceptions_handled: usize,
    pub efficiency_score: f32,
    pub effectiveness_score: f32,
}

#[derive(Debug, Clone)]
pub struct AdaptationSuggestion {
    pub suggestion_type: AdaptationType,
    pub target_parameter: String,
    pub current_value: f32,
    pub recommended_value: f32,
    pub expected_improvement: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum AdaptationType {
    StrengthAdjustment,
    ThresholdModification,
    TemporalAdjustment,
    GroupReorganization,
}

#[derive(Debug, Clone)]
pub struct InhibitionLearningResult {
    pub learning_applied: bool,
    pub parameter_adjustments: Vec<ParameterAdjustment>,
    pub performance_improvement_estimate: f32,
    pub learning_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ParameterAdjustment {
    pub parameter_name: String,
    pub old_value: f32,
    pub new_value: f32,
    pub adjustment_type: ParameterAdjustmentType,
    pub expected_improvement: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterAdjustmentType {
    StrengthOptimization,
    CompetitionOptimization,
    TemporalOptimization,
    ThresholdAdjustment,
}

#[derive(Debug, Clone)]
pub struct LearningStatus {
    pub learning_enabled: bool,
    pub parameters_learned: Vec<String>,
    pub learning_confidence: f32,
    pub adaptation_count: usize,
    pub last_learning_timestamp: SystemTime,
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
    LearningAdjustment,
}