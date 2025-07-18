//! Type definitions for adaptive learning system

use crate::cognitive::types::CognitivePatternType;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Configuration for adaptive learning system
#[derive(Debug, Clone)]
pub struct AdaptiveLearningConfig {
    pub learning_cycle_frequency: Duration,
    pub emergency_adaptation_threshold: f32,
    pub max_concurrent_adaptations: usize,
    pub performance_tracking_window: Duration,
    pub feedback_weight_decay: f32,
    pub adaptation_aggressiveness: f32,
}

/// Query performance metrics
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_latency: Duration,
    pub latency_distribution: Vec<Duration>,
    pub query_complexity_scores: Vec<f32>,
}

/// Cognitive pattern performance metrics
#[derive(Debug, Clone)]
pub struct CognitiveMetrics {
    pub pattern_usage_frequency: HashMap<CognitivePatternType, u64>,
    pub pattern_success_rates: HashMap<CognitivePatternType, f32>,
    pub attention_efficiency: f32,
    pub memory_utilization: f32,
    pub inhibition_effectiveness: f32,
    pub orchestration_quality: f32,
}

/// System resource metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub memory_usage: f32,
    pub cpu_utilization: f32,
    pub storage_efficiency: f32,
    pub cache_performance: HashMap<String, f32>,
    pub error_rates: HashMap<String, f32>,
}

/// User interaction metrics
#[derive(Debug, Clone)]
pub struct UserInteractionMetrics {
    pub session_durations: Vec<Duration>,
    pub user_satisfaction_scores: Vec<f32>,
    pub task_completion_rates: Vec<f32>,
    pub feedback_sentiment: Vec<f32>,
    pub repeat_usage_patterns: HashMap<String, u32>,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub overall_performance_score: f32,
    pub component_scores: HashMap<String, f32>,
    pub bottlenecks: Vec<String>,
    pub system_health: f32,
}

/// Performance data aggregation
#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub query_latencies: Vec<Duration>,
    pub memory_usage: Vec<f32>,
    pub accuracy_scores: Vec<f32>,
    pub user_satisfaction: Vec<f32>,
    pub system_stability: f32,
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub latency_threshold: Duration,
    pub error_rate_threshold: f32,
    pub satisfaction_threshold: f32,
    pub memory_threshold: f32,
    pub cpu_threshold: f32,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub description: String,
    pub potential_solutions: Vec<String>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckType {
    Memory,
    Computation,
    Network,
    Storage,
    Cognitive,
    User,
}

/// User feedback structure
#[derive(Debug, Clone)]
pub struct UserFeedback {
    pub feedback_id: Uuid,
    pub timestamp: SystemTime,
    pub feedback_type: FeedbackType,
    pub satisfaction_score: f32,
    pub accuracy_rating: f32,
    pub response_quality: f32,
    pub response_speed: f32,
    pub context: String,
    pub suggestions: Vec<String>,
}

/// Types of user feedback
#[derive(Debug, Clone)]
pub enum FeedbackType {
    Explicit,
    Implicit,
    System,
}

/// System feedback structure
#[derive(Debug, Clone)]
pub struct SystemFeedback {
    pub feedback_id: Uuid,
    pub timestamp: SystemTime,
    pub component: String,
    pub metric_type: String,
    pub value: f32,
    pub severity: f32,
    pub context: HashMap<String, String>,
}

/// Feedback aggregation configuration
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    pub feedback_retention_period: Duration,
    pub implicit_feedback_weight: f32,
    pub explicit_feedback_weight: f32,
    pub system_feedback_weight: f32,
    pub temporal_decay_factor: f32,
}

/// User satisfaction analysis
#[derive(Debug, Clone)]
pub struct SatisfactionAnalysis {
    pub satisfaction_trends: Vec<f32>,
    pub problem_areas: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

/// Correlation analysis between metrics
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub performance_satisfaction_correlation: f32,
    pub speed_satisfaction_correlation: f32,
    pub accuracy_satisfaction_correlation: f32,
    pub significant_correlations: Vec<(String, String, f32)>,
}

/// Learning target identification
#[derive(Debug, Clone)]
pub struct LearningTarget {
    pub target_type: LearningTargetType,
    pub importance: f32,
    pub feasibility: f32,
    pub description: String,
    pub expected_impact: f32,
}

/// Types of learning targets
#[derive(Debug, Clone)]
pub enum LearningTargetType {
    StructureOptimization,
    ParameterTuning,
    BehaviorModification,
    PatternImprovement,
}

/// Learning task types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LearningTaskType {
    EmergencyAdaptation,
    HebbianLearning,
    GraphOptimization,
    ParameterTuning,
    UserFeedbackIntegration,
}

/// Scheduled learning task
#[derive(Debug, Clone)]
pub struct ScheduledLearningTask {
    pub task_id: Uuid,
    pub task_type: LearningTaskType,
    pub priority: f32,
    pub scheduled_time: SystemTime,
    pub estimated_resources: ResourceRequirement,
    pub dependencies: Vec<Uuid>,
}

/// Resource requirements for learning tasks
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    pub memory_mb: f32,
    pub cpu_cores: f32,
    pub storage_mb: f32,
    pub network_bandwidth_mbps: f32,
}

/// Learning schedule configuration
#[derive(Debug, Clone)]
pub struct LearningScheduleConfig {
    pub base_learning_frequency: Duration,
    pub adaptive_scheduling: bool,
    pub priority_weights: HashMap<LearningTaskType, f32>,
    pub resource_constraints: ResourceConstraints,
}

/// Resource constraints for learning tasks
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory_usage: f32,
    pub max_cpu_usage: f32,
    pub max_learning_duration: Duration,
    pub concurrent_task_limit: usize,
}

/// Adaptation record for tracking
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    pub record_id: Uuid,
    pub timestamp: SystemTime,
    pub adaptation_type: AdaptationType,
    pub performance_before: f32,
    pub performance_after: f32,
    pub success: bool,
    pub impact_assessment: String,
}

/// Types of adaptations
#[derive(Debug, Clone)]
pub enum AdaptationType {
    ParameterAdjustment,
    StructureModification,
    BehaviorChange,
    EmergencyResponse,
}

/// Emergency triggers
#[derive(Debug, Clone)]
pub enum EmergencyTrigger {
    SystemFailure,
    PerformanceCollapse,
    UserExodus,
    ResourceExhaustion,
}

/// Emergency context
#[derive(Debug, Clone)]
pub struct EmergencyContext {
    pub trigger_type: EmergencyTrigger,
    pub severity: f32,
    pub affected_components: Vec<String>,
    pub performance_before: f32,
    pub emergency_actions: Vec<String>,
}

/// Collected metrics structure
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    pub query_metrics: QueryMetrics,
    pub cognitive_metrics: CognitiveMetrics,
    pub system_metrics: SystemMetrics,
    pub user_interaction_metrics: UserInteractionMetrics,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            learning_cycle_frequency: Duration::from_secs(3600), // 1 hour
            emergency_adaptation_threshold: 0.8,
            max_concurrent_adaptations: 2,
            performance_tracking_window: Duration::from_secs(3600),
            feedback_weight_decay: 0.95,
            adaptation_aggressiveness: 0.5,
        }
    }
}

impl Default for ResourceRequirement {
    fn default() -> Self {
        Self {
            memory_mb: 100.0,
            cpu_cores: 0.5,
            storage_mb: 50.0,
            network_bandwidth_mbps: 1.0,
        }
    }
}

impl Default for LearningScheduleConfig {
    fn default() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(LearningTaskType::EmergencyAdaptation, 1.0);
        priority_weights.insert(LearningTaskType::HebbianLearning, 0.8);
        priority_weights.insert(LearningTaskType::GraphOptimization, 0.6);
        priority_weights.insert(LearningTaskType::ParameterTuning, 0.4);
        
        Self {
            base_learning_frequency: Duration::from_secs(3600), // 1 hour
            adaptive_scheduling: true,
            priority_weights,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_usage: 0.8,
            max_cpu_usage: 0.7,
            max_learning_duration: Duration::from_secs(1800), // 30 minutes
            concurrent_task_limit: 3,
        }
    }
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            feedback_retention_period: Duration::from_secs(86400 * 7), // 1 week
            implicit_feedback_weight: 0.3,
            explicit_feedback_weight: 0.7,
            system_feedback_weight: 0.5,
            temporal_decay_factor: 0.95,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            latency_threshold: Duration::from_millis(500),
            error_rate_threshold: 0.1,
            satisfaction_threshold: 0.7,
            memory_threshold: 0.8,
            cpu_threshold: 0.8,
        }
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            average_latency: Duration::from_millis(200),
            latency_distribution: Vec::new(),
            query_complexity_scores: Vec::new(),
        }
    }
}

impl Default for CognitiveMetrics {
    fn default() -> Self {
        Self {
            pattern_usage_frequency: HashMap::new(),
            pattern_success_rates: HashMap::new(),
            attention_efficiency: 0.8,
            memory_utilization: 0.7,
            inhibition_effectiveness: 0.75,
            orchestration_quality: 0.8,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            memory_usage: 0.6,
            cpu_utilization: 0.5,
            storage_efficiency: 0.8,
            cache_performance: HashMap::new(),
            error_rates: HashMap::new(),
        }
    }
}

impl Default for UserInteractionMetrics {
    fn default() -> Self {
        Self {
            session_durations: Vec::new(),
            user_satisfaction_scores: Vec::new(),
            task_completion_rates: Vec::new(),
            feedback_sentiment: Vec::new(),
            repeat_usage_patterns: HashMap::new(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            query_metrics: QueryMetrics::default(),
            cognitive_metrics: CognitiveMetrics::default(),
            system_metrics: SystemMetrics::default(),
            user_interaction_metrics: UserInteractionMetrics::default(),
        }
    }
}