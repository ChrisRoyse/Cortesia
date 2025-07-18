//! Type definitions for Phase 4 Cognitive Integration

use crate::cognitive::types::CognitivePatternType;
use crate::core::types::EntityKey;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Learning insights from Hebbian and other learning systems
#[derive(Debug, Clone)]
pub struct LearningInsights {
    pub hebbian_connection_strengths: HashMap<(EntityKey, EntityKey), f32>,
    pub effective_cognitive_patterns: HashMap<CognitivePatternType, f32>,
    pub learning_derived_optimizations: Vec<CognitiveOptimization>,
    pub user_preference_patterns: HashMap<String, f32>,
    pub performance_correlations: HashMap<String, f32>,
}

/// Adaptive strategies for cognitive pattern selection
#[derive(Debug, Clone)]
pub struct AdaptiveStrategies {
    pub pattern_selection_weights: HashMap<CognitivePatternType, f32>,
    pub ensemble_composition_rules: Vec<EnsembleRule>,
    pub context_sensitive_adaptations: HashMap<String, ContextAdaptation>,
    pub learned_shortcuts: Vec<CognitiveShortcut>,
    pub failure_recovery_strategies: HashMap<String, RecoveryStrategy>,
}

/// Historical performance data for patterns
#[derive(Debug, Clone)]
pub struct PatternPerformanceHistory {
    pub pattern_usage_statistics: HashMap<CognitivePatternType, UsageStatistics>,
    pub success_rates_by_context: HashMap<String, HashMap<CognitivePatternType, f32>>,
    pub performance_trends: HashMap<CognitivePatternType, Vec<PerformanceDataPoint>>,
    pub user_satisfaction_by_pattern: HashMap<CognitivePatternType, f32>,
}

/// Usage statistics for cognitive patterns
#[derive(Debug, Clone)]
pub struct UsageStatistics {
    pub total_invocations: u64,
    pub successful_completions: u64,
    pub average_duration: Duration,
    pub average_quality_score: f32,
    pub last_used: SystemTime,
}

/// Performance data point for tracking
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    pub timestamp: SystemTime,
    pub quality_score: f32,
    pub duration: Duration,
    pub context: String,
    pub user_satisfaction: f32,
}

/// Cognitive optimization recommendations
#[derive(Debug, Clone)]
pub struct CognitiveOptimization {
    pub optimization_type: CognitiveOptimizationType,
    pub target_pattern: CognitivePatternType,
    pub description: String,
    pub expected_improvement: f32,
    pub implementation_details: HashMap<String, String>,
}

/// Types of cognitive optimizations
#[derive(Debug, Clone)]
pub enum CognitiveOptimizationType {
    ParameterTuning,
    PatternCombination,
    ContextSpecialization,
    ShortcutLearning,
    ErrorCorrection,
}

/// Ensemble rule for pattern combination
#[derive(Debug, Clone)]
pub struct EnsembleRule {
    pub rule_name: String,
    pub applicable_contexts: Vec<String>,
    pub pattern_combinations: HashMap<CognitivePatternType, f32>,
    pub effectiveness_score: f32,
}

/// Context-specific adaptation settings
#[derive(Debug, Clone)]
pub struct ContextAdaptation {
    pub context_name: String,
    pub adaptation_parameters: HashMap<String, f32>,
    pub success_rate: f32,
    pub last_updated: SystemTime,
}

/// Learned cognitive shortcuts
#[derive(Debug, Clone)]
pub struct CognitiveShortcut {
    pub shortcut_name: String,
    pub trigger_conditions: Vec<String>,
    pub shortcut_action: String,
    pub time_saved: Duration,
    pub accuracy_maintained: f32,
}

/// Recovery strategy for failed patterns
#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub strategy_name: String,
    pub fallback_patterns: Vec<CognitivePatternType>,
    pub recovery_actions: Vec<String>,
    pub success_probability: f32,
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub pattern_response_times: HashMap<CognitivePatternType, Duration>,
    pub pattern_quality_scores: HashMap<CognitivePatternType, f32>,
    pub overall_satisfaction: f32,
    pub error_rates: HashMap<String, f32>,
    pub establishment_date: SystemTime,
}

/// Current performance metrics
#[derive(Debug, Clone)]
pub struct CurrentPerformance {
    pub pattern_response_times: HashMap<CognitivePatternType, Duration>,
    pub pattern_quality_scores: HashMap<CognitivePatternType, f32>,
    pub overall_satisfaction: f32,
    pub error_rates: HashMap<String, f32>,
    pub last_updated: SystemTime,
}

/// Learning impact analysis
#[derive(Debug, Clone)]
pub struct LearningImpactAnalysis {
    pub learning_contributions: HashMap<String, f32>,
    pub adaptation_effectiveness: f32,
    pub optimization_benefits: HashMap<String, f32>,
    pub user_satisfaction_improvements: f32,
}

/// Throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub queries_per_second: f32,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_response_time: Duration,
}

/// Performance data collection
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

/// Configuration for Phase 4 cognitive system
#[derive(Debug, Clone)]
pub struct Phase4CognitiveConfig {
    pub learning_integration_level: LearningIntegrationLevel,
    pub adaptation_aggressiveness: f32,
    pub personalization_enabled: bool,
    pub continuous_optimization: bool,
    pub safety_mode: SafetyMode,
}

/// Levels of learning integration
#[derive(Debug, Clone)]
pub enum LearningIntegrationLevel {
    Minimal,
    Standard,
    Deep,
    Experimental,
}

/// Safety modes for adaptation
#[derive(Debug, Clone)]
pub enum SafetyMode {
    Conservative,
    Balanced,
    Aggressive,
}

/// Learning benefit assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningBenefitAssessment {
    pub overall_improvement: f32,
    pub satisfaction_improvement: f32,
    pub pattern_improvements: HashMap<CognitivePatternType, f32>,
    pub learning_effectiveness: f32,
    pub recommendation: String,
}

/// Safety constraints for adaptations
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    pub max_performance_degradation: f32,
    pub rollback_threshold: f32,
    pub validation_required: bool,
    pub user_approval_threshold: f32,
}

/// Rollback management
#[derive(Debug, Clone)]
pub struct RollbackManager {
    pub checkpoints: Vec<String>,
    pub max_checkpoints: usize,
    pub auto_rollback_enabled: bool,
}

/// User impact assessment
#[derive(Debug, Clone)]
pub struct UserImpactAssessment {
    pub satisfaction_change: f32,
    pub interaction_quality_change: f32,
    pub task_completion_rate_change: f32,
    pub user_retention_impact: f32,
}

impl Default for LearningInsights {
    fn default() -> Self {
        Self {
            hebbian_connection_strengths: HashMap::new(),
            effective_cognitive_patterns: HashMap::new(),
            learning_derived_optimizations: Vec::new(),
            user_preference_patterns: HashMap::new(),
            performance_correlations: HashMap::new(),
        }
    }
}

impl Default for AdaptiveStrategies {
    fn default() -> Self {
        Self {
            pattern_selection_weights: HashMap::new(),
            ensemble_composition_rules: Vec::new(),
            context_sensitive_adaptations: HashMap::new(),
            learned_shortcuts: Vec::new(),
            failure_recovery_strategies: HashMap::new(),
        }
    }
}

impl Default for PatternPerformanceHistory {
    fn default() -> Self {
        Self {
            pattern_usage_statistics: HashMap::new(),
            success_rates_by_context: HashMap::new(),
            performance_trends: HashMap::new(),
            user_satisfaction_by_pattern: HashMap::new(),
        }
    }
}

/// Result from Phase 4 cognitive query
#[derive(Debug, Clone)]
pub struct Phase4QueryResult {
    pub primary_result: String,
    pub confidence: f32,
    pub pattern_used: CognitivePatternType,
    pub learning_insights: Option<LearningInsights>,
    pub performance_metrics: PerformanceMetrics,
}

/// Result from Phase 4 learning operations
#[derive(Debug, Clone)]
pub struct Phase4LearningResult {
    pub success: bool,
    pub insights_gained: LearningInsights,
    pub adaptations_made: Vec<String>,
    pub performance_impact: f32,
}

/// Performance metrics for Phase 4 operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time: Duration,
    pub quality_score: f32,
    pub resource_usage: f32,
}

/// Types of learning algorithms available in Phase 4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LearningAlgorithmType {
    Hebbian,
    Reinforcement,
    Bayesian,
    Evolutionary,
    Neural,
}