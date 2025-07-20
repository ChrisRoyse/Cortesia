//! Types and data structures for Phase 4 Learning System

use crate::learning::types::*;
use crate::learning::adaptive_learning::ResourceRequirement;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use uuid::Uuid;

/// Active learning session tracking
#[derive(Debug, Clone)]
pub struct ActiveLearningSession {
    pub session_id: Uuid,
    pub session_type: LearningSessionType,
    pub start_time: SystemTime,
    pub expected_duration: Duration,
    pub participants: Vec<LearningParticipant>,
    pub progress: LearningProgress,
    pub resources_allocated: ResourceAllocation,
}

/// Types of learning sessions
#[derive(Debug, Clone)]
pub enum LearningSessionType {
    RoutineLearning,
    PerformanceOptimization,
    EmergencyAdaptation,
    ProactiveLearning,
    UserFeedbackIntegration,
}

/// Learning system participants
#[derive(Debug, Clone)]
pub enum LearningParticipant {
    HebbianEngine,
    HomeostasisSystem,
    AdaptiveLearning,
    CognitiveOrchestrator,
}

/// Progress tracking for learning sessions
#[derive(Debug, Clone)]
pub struct LearningProgress {
    pub completion_percentage: f32,
    pub milestones_achieved: Vec<String>,
    pub current_phase: String,
    pub estimated_remaining_time: Duration,
    pub performance_impact_so_far: f32,
}

/// Resource allocation for learning sessions
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub memory_allocated_mb: f32,
    pub cpu_cores_allocated: f32,
    pub priority_level: f32,
    pub time_budget: Duration,
}

/// Learning schedule management
#[derive(Debug, Clone)]
pub struct LearningSchedule {
    pub scheduled_sessions: Vec<ScheduledSession>,
    pub recurring_schedules: Vec<RecurringSchedule>,
    pub conditional_triggers: Vec<ConditionalTrigger>,
}

/// Scheduled learning session
#[derive(Debug, Clone)]
pub struct ScheduledSession {
    pub session_id: Uuid,
    pub session_type: LearningSessionType,
    pub scheduled_time: SystemTime,
    pub priority: f32,
    pub dependencies: Vec<Uuid>,
    pub resource_requirements: ResourceRequirement,
}

/// Recurring schedule definition
#[derive(Debug, Clone)]
pub struct RecurringSchedule {
    pub schedule_id: Uuid,
    pub session_type: LearningSessionType,
    pub frequency: Duration,
    pub last_execution: Option<SystemTime>,
    pub next_execution: SystemTime,
    pub conditions: Vec<ScheduleCondition>,
}

/// Conditions for schedule execution
#[derive(Debug, Clone)]
pub enum ScheduleCondition {
    PerformanceThreshold(f32),
    UserActivityLevel(f32),
    SystemLoad(f32),
    TimeOfDay(u8), // Hour of day
    ErrorRate(f32),
}

/// Conditional trigger for automated learning
#[derive(Debug, Clone)]
pub struct ConditionalTrigger {
    pub trigger_id: Uuid,
    pub condition: TriggerCondition,
    pub action: TriggerAction,
    pub cooldown_period: Duration,
    pub last_triggered: Option<SystemTime>,
}

/// Trigger conditions
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    PerformanceDrop { threshold: f32, duration: Duration },
    ErrorSpike { rate: f32, window: Duration },
    UserSatisfactionDrop { threshold: f32 },
    ResourceExhaustion { resource_type: String, threshold: f32 },
    PatternDetected { pattern_name: String },
}

/// Actions triggered by conditions
#[derive(Debug, Clone)]
pub enum TriggerAction {
    StartEmergencyLearning,
    OptimizePerformance,
    AdjustParameters,
    NotifyAdministrator,
    ActivateConservationMode,
}

/// System coordination state
#[derive(Debug, Clone)]
pub struct CoordinationState {
    pub current_coordination_mode: CoordinationMode,
    pub coordination_effectiveness: f32,
    pub inter_system_communication_quality: f32,
    pub learning_coherence_score: f32,
    pub last_coordination_update: SystemTime,
}

/// Coordination modes
#[derive(Debug, Clone)]
pub enum CoordinationMode {
    Balanced,        // All systems working together
    HebbianFocused,  // Prioritize connection learning
    OptimizationFocused, // Prioritize structure optimization
    EmergencyMode,   // Crisis response
    ConservationMode, // Minimal resource usage
}

/// Strategy types for learning coordination
#[derive(Debug, Clone)]
pub enum StrategyType {
    Conservative,
    Emergency,
    Aggressive,
    Focused,
    Balanced,
}

/// Coordination approaches
#[derive(Debug, Clone)]
pub enum CoordinationApproach {
    Synchronized,
    Sequential,
    Parallel,
    Emergency,
}

/// System assessment results
#[derive(Debug, Clone)]
pub struct SystemAssessment {
    pub overall_health: f32,
    pub performance_trends: Vec<String>,
    pub bottlenecks: Vec<String>,
    pub learning_opportunities: Vec<String>,
    pub risk_factors: Vec<String>,
    pub readiness_for_learning: f32,
}

/// Learning strategy definition
#[derive(Debug, Clone)]
pub struct LearningStrategy {
    pub strategy_type: StrategyType,
    pub priority_areas: Vec<String>,
    pub resource_allocation: ResourceRequirement,
    pub coordination_approach: CoordinationApproach,
    pub safety_level: f32,
    pub expected_duration: Duration,
}

/// Coordination execution result
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    pub session_id: Uuid,
    pub coordination_mode: CoordinationMode,
    pub participants_activated: Vec<LearningParticipant>,
    pub resource_allocation: ResourceRequirement,
    pub synchronization_points: Vec<String>,
}

/// Homeostasis balancing result
#[derive(Debug, Clone)]
pub struct HomeostasisBalancingResult {
    pub balancing_applied: bool,
    pub stability_improvement: f32,
    pub adjustments_made: usize,
    pub emergency_intervention: bool,
}

/// Structure optimization result
#[derive(Debug, Clone)]
pub struct StructureOptimizationResult {
    pub optimizations_applied: usize,
    pub performance_improvement: f32,
    pub structural_changes: Vec<String>,
    pub efficiency_gains: f32,
}

/// System parameter adaptation result
#[derive(Debug, Clone)]
pub struct SystemParameterAdaptation {
    pub parameters_changed: HashMap<String, f32>,
    pub adaptation_rationale: String,
    pub expected_impact: f32,
}

/// Validation result for learning changes
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub success: bool,
    pub performance_improvement: f32,
    pub validation_details: String,
    pub changes_committed: bool,
}

/// Comprehensive learning result
#[derive(Debug, Clone)]
pub struct ComprehensiveLearningResult {
    pub session_id: Uuid,
    pub duration: Duration,
    pub system_assessment: SystemAssessment,
    pub learning_strategy: LearningStrategy,
    pub coordination_result: CoordinationResult,
    pub learning_results: CoordinatedLearningResults,
    pub homeostasis_result: HomeostasisBalancingResult,
    pub optimization_result: StructureOptimizationResult,
    pub adaptation_result: SystemParameterAdaptation,
    pub validation_result: ValidationResult,
    pub overall_success: bool,
    pub performance_improvement: f32,
}

/// Emergency response tracking
#[derive(Debug, Clone)]
pub struct EmergencyResponse {
    pub protocol_name: String,
    pub actions_taken: Vec<String>,
    pub success: bool,
    pub recovery_time: Duration,
    pub performance_impact: f32,
}

/// Result from Hebbian learning operations
#[derive(Debug, Clone)]
pub struct HebbianLearningResult {
    pub connections_updated: usize,
    pub learning_efficiency: f32,
    pub structural_changes: Vec<String>,
    pub performance_impact: f32,
}

/// Result from homeostasis operations
#[derive(Debug, Clone)]
pub struct HomeostasisResult {
    pub synapses_normalized: usize,
    pub homeostasis_factor: f32,
    pub stability_improvements: Vec<String>,
    pub impact_score: f32,
}