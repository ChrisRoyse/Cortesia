//! Type definitions for optimization agent

use crate::core::types::EntityKey;
use crate::learning::types::*;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Graph optimization agent for performance improvements
#[derive(Debug, Clone)]
pub struct GraphOptimizationAgent {
    pub pattern_detector: PatternDetector,
    pub efficiency_analyzer: EfficiencyAnalyzer,
    pub optimization_scheduler: OptimizationScheduler,
    pub safety_validator: SafetyValidator,
    pub rollback_manager: RollbackManager,
    pub impact_predictor: ImpactPredictor,
    pub bottleneck_detector: BottleneckDetector,
    pub pattern_cache: PatternCache,
}

/// Pattern detection system
#[derive(Debug, Clone)]
pub struct PatternDetector {
    pub detection_threshold: f32,
    pub analysis_config: PatternAnalysisConfig,
    pub last_analysis: Option<Instant>,
}

/// Pattern analysis configuration
#[derive(Debug, Clone)]
pub struct PatternAnalysisConfig {
    pub analysis_scope: AnalysisScope,
    pub similarity_threshold: f32,
    pub min_pattern_size: usize,
    pub max_pattern_size: usize,
    pub enable_caching: bool,
    pub cache_ttl: Duration,
}

/// Analysis scope for pattern detection
#[derive(Debug, Clone)]
pub enum AnalysisScope {
    LocalEntities(HashSet<EntityKey>),
    GlobalGraph,
    SubgraphRegion(Vec<EntityKey>),
    RecentActivity(Duration),
}

/// Performance metrics collection
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub query_latency: Duration,
    pub memory_usage: usize,
    pub cache_hit_rate: f32,
    pub throughput: f32,
    pub error_rate: f32,
    pub resource_utilization: f32,
    pub optimization_impact: f32,
}

/// Efficiency analysis system
#[derive(Debug, Clone)]
pub struct EfficiencyAnalyzer {
    pub metrics_history: Vec<PerformanceMetrics>,
    pub baseline_metrics: PerformanceMetrics,
    pub efficiency_threshold: f32,
    pub analysis_window: Duration,
}

/// Bottleneck detection system
#[derive(Debug, Clone)]
pub struct BottleneckDetector {
    pub detection_sensitivity: f32,
    pub monitoring_window: Duration,
    pub bottleneck_threshold: f32,
    pub identified_bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub affected_entities: Vec<EntityKey>,
    pub suggested_optimization: OptimizationType,
    pub estimated_improvement: f32,
    pub detection_time: Instant,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    QueryLatency,
    MemoryUsage,
    CacheEfficiency,
    NetworkTraversal,
    IndexFragmentation,
    ResourceContention,
}

/// Optimization scheduling system
#[derive(Debug, Clone)]
pub struct OptimizationScheduler {
    pub schedule_config: ScheduleConfig,
    pub pending_optimizations: Vec<ScheduledOptimization>,
    pub execution_history: Vec<ExecutionRecord>,
    pub priority_queue: Vec<PriorityItem>,
}

/// Scheduling configuration
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    pub max_concurrent_optimizations: usize,
    pub priority_weights: PriorityWeights,
    pub execution_window: Duration,
    pub cooldown_period: Duration,
}

/// Priority weights for scheduling
#[derive(Debug, Clone)]
pub struct PriorityWeights {
    pub efficiency_gain: f32,
    pub safety_score: f32,
    pub execution_cost: f32,
    pub user_impact: f32,
}

/// Scheduled optimization task
#[derive(Debug, Clone)]
pub struct ScheduledOptimization {
    pub optimization_id: String,
    pub optimization_type: OptimizationType,
    pub priority: f32,
    pub scheduled_time: Instant,
    pub estimated_duration: Duration,
    pub safety_score: f32,
    pub expected_improvement: f32,
}

/// Execution record for tracking
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub optimization_id: String,
    pub execution_time: Instant,
    pub duration: Duration,
    pub success: bool,
    pub actual_improvement: f32,
    pub rollback_required: bool,
}

/// Priority queue item
#[derive(Debug, Clone)]
pub struct PriorityItem {
    pub optimization_id: String,
    pub priority_score: f32,
    pub insertion_time: Instant,
}

/// Safety validation system
#[derive(Debug, Clone)]
pub struct SafetyValidator {
    pub validation_rules: Vec<SafetyRule>,
    pub safety_threshold: f32,
    pub validation_history: Vec<ValidationResult>,
}

/// Safety validation rule
#[derive(Debug, Clone)]
pub struct SafetyRule {
    pub rule_id: String,
    pub rule_type: SafetyRuleType,
    pub severity: SafetySeverity,
    pub validation_fn: String, // Function name for validation
}

/// Types of safety rules
#[derive(Debug, Clone)]
pub enum SafetyRuleType {
    DataIntegrity,
    PerformanceRegression,
    ResourceLimits,
    ConcurrentAccess,
    TransactionSafety,
}

/// Safety severity levels
#[derive(Debug, Clone)]
pub enum SafetySeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub rule_id: String,
    pub passed: bool,
    pub score: f32,
    pub details: String,
    pub validation_time: Instant,
}

/// Rollback management system
#[derive(Debug, Clone)]
pub struct RollbackManager {
    pub checkpoints: Vec<OptimizationCheckpoint>,
    pub rollback_history: Vec<RollbackRecord>,
    pub auto_rollback_enabled: bool,
    pub rollback_threshold: f32,
}

/// Optimization checkpoint
#[derive(Debug, Clone)]
pub struct OptimizationCheckpoint {
    pub checkpoint_id: String,
    pub creation_time: Instant,
    pub graph_state: GraphState,
    pub performance_baseline: PerformanceMetrics,
    pub optimization_context: OptimizationContext,
}

/// Graph state snapshot
#[derive(Debug, Clone)]
pub struct GraphState {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub structure_hash: u64,
    pub performance_metrics: PerformanceMetrics,
}

/// Optimization context
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    pub optimization_type: OptimizationType,
    pub affected_entities: Vec<EntityKey>,
    pub parameters: HashMap<String, f32>,
    pub safety_requirements: Vec<String>,
}

/// Rollback record
#[derive(Debug, Clone)]
pub struct RollbackRecord {
    pub rollback_id: String,
    pub rollback_time: Instant,
    pub reason: RollbackReason,
    pub checkpoint_id: String,
    pub success: bool,
}

/// Reasons for rollback
#[derive(Debug, Clone)]
pub enum RollbackReason {
    PerformanceRegression,
    SafetyViolation,
    UserRequest,
    SystemError,
    TimeoutExpired,
}

/// Impact prediction system
#[derive(Debug, Clone)]
pub struct ImpactPredictor {
    pub prediction_models: Vec<PredictionModel>,
    pub historical_data: Vec<OptimizationImpact>,
    pub prediction_accuracy: f32,
}

/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub model_id: String,
    pub model_type: ModelType,
    pub accuracy: f32,
    pub training_data_size: usize,
    pub last_updated: Instant,
}

/// Types of prediction models
#[derive(Debug, Clone)]
pub enum ModelType {
    LinearRegression,
    NeuralNetwork,
    DecisionTree,
    EnsembleModel,
}

/// Optimization impact record
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    pub optimization_type: OptimizationType,
    pub predicted_improvement: f32,
    pub actual_improvement: f32,
    pub execution_time: Duration,
    pub side_effects: Vec<SideEffect>,
}

/// Side effect of optimization
#[derive(Debug, Clone)]
pub struct SideEffect {
    pub effect_type: SideEffectType,
    pub severity: f32,
    pub description: String,
    pub mitigation: Option<String>,
}

/// Types of side effects
#[derive(Debug, Clone)]
pub enum SideEffectType {
    MemoryIncrease,
    LatencyIncrease,
    AccuracyDecrease,
    ResourceContention,
    CacheInvalidation,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    AttributeBubbling,
    HierarchyConsolidation,
    SubgraphFactorization,
    ConnectionPruning,
    IndexOptimization,
    CacheOptimization,
    QueryOptimization,
    MemoryOptimization,
}

/// Pattern cache system
#[derive(Debug, Clone)]
pub struct PatternCache {
    pub cached_patterns: HashMap<String, CachedPattern>,
    pub cache_capacity: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub last_cleanup: Instant,
}

/// Cached pattern entry
#[derive(Debug, Clone)]
pub struct CachedPattern {
    pub pattern_id: String,
    pub pattern_type: OptimizationType,
    pub detection_time: Instant,
    pub expiry_time: Instant,
    pub hit_count: usize,
    pub efficiency_score: f32,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_id: String,
    pub optimization_type: OptimizationType,
    pub affected_entities: Vec<EntityKey>,
    pub estimated_improvement: f32,
    pub implementation_cost: f32,
    pub risk_level: RiskLevel,
    pub prerequisites: Vec<String>,
}

/// Risk levels for optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization plan
#[derive(Debug, Clone)]
pub struct OptimizationPlan {
    pub plan_id: String,
    pub optimizations: Vec<OptimizationOpportunity>,
    pub execution_order: Vec<String>,
    pub total_estimated_improvement: f32,
    pub total_estimated_cost: f32,
    pub estimated_duration: Duration,
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
    pub rollback_plan: String,
}

/// Risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    pub factor_type: RiskFactorType,
    pub probability: f32,
    pub impact: f32,
    pub description: String,
}

/// Types of risk factors
#[derive(Debug, Clone)]
pub enum RiskFactorType {
    DataCorruption,
    PerformanceDegradation,
    SystemInstability,
    UserDisruption,
    ResourceExhaustion,
}

/// Mitigation strategy
#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    pub strategy_id: String,
    pub strategy_type: MitigationType,
    pub effectiveness: f32,
    pub implementation_cost: f32,
    pub description: String,
}

/// Types of mitigation strategies
#[derive(Debug, Clone)]
pub enum MitigationType {
    PreventiveCheckpoint,
    GradualRollout,
    MonitoringAlert,
    AutomaticRollback,
    LoadBalancing,
}

/// Default implementations
impl Default for PatternAnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_scope: AnalysisScope::GlobalGraph,
            similarity_threshold: 0.8,
            min_pattern_size: 2,
            max_pattern_size: 10,
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600),
        }
    }
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            max_concurrent_optimizations: 3,
            priority_weights: PriorityWeights {
                efficiency_gain: 0.4,
                safety_score: 0.3,
                execution_cost: 0.2,
                user_impact: 0.1,
            },
            execution_window: Duration::from_secs(300),
            cooldown_period: Duration::from_secs(60),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            query_latency: Duration::from_millis(100),
            memory_usage: 0,
            cache_hit_rate: 0.0,
            throughput: 0.0,
            error_rate: 0.0,
            resource_utilization: 0.0,
            optimization_impact: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Calculate overall performance score
    pub fn performance_score(&self) -> f32 {
        let latency_score = 1.0 - (self.query_latency.as_millis() as f32 / 1000.0).min(1.0);
        let cache_score = self.cache_hit_rate;
        let throughput_score = self.throughput.min(1.0);
        let error_score = 1.0 - self.error_rate;
        let resource_score = 1.0 - self.resource_utilization;
        
        (latency_score + cache_score + throughput_score + error_score + resource_score) / 5.0
    }
    
    /// Check if metrics indicate good performance
    pub fn is_performing_well(&self) -> bool {
        self.performance_score() > 0.7
    }
}

impl OptimizationOpportunity {
    /// Calculate priority score
    pub fn priority_score(&self) -> f32 {
        let improvement_factor = self.estimated_improvement * 2.0;
        let cost_factor = 1.0 / (self.implementation_cost + 1.0);
        let risk_factor = match self.risk_level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.8,
            RiskLevel::High => 0.5,
            RiskLevel::Critical => 0.2,
        };
        
        improvement_factor * cost_factor * risk_factor
    }
    
    /// Check if opportunity is worth pursuing
    pub fn is_worthwhile(&self) -> bool {
        self.priority_score() > 0.3 && self.risk_level != RiskLevel::Critical
    }
}