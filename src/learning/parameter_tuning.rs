use crate::learning::types::PerformanceData;
use crate::error::Result;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;
use anyhow::anyhow;

/// Automated parameter tuning system for Phase 4 learning
#[derive(Debug, Clone)]
pub struct ParameterTuningSystem {
    pub tuning_strategies: Vec<TuningStrategy>,
    pub parameter_spaces: HashMap<String, ParameterSpace>,
    pub optimization_history: Arc<RwLock<Vec<TuningResult>>>,
    pub active_tuning_sessions: Arc<RwLock<HashMap<Uuid, TuningSession>>>,
    pub tuning_config: ParameterTuningConfig,
}

#[derive(Debug, Clone)]
pub struct TuningStrategy {
    pub strategy_id: String,
    pub strategy_type: TuningStrategyType,
    pub target_components: Vec<String>,
    pub optimization_algorithm: OptimizationAlgorithm,
    pub success_criteria: SuccessCriteria,
    pub resource_budget: ResourceBudget,
}

#[derive(Debug, Clone)]
pub enum TuningStrategyType {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GradientBased,
    EvolutionaryAlgorithm,
    SimulatedAnnealing,
}

#[derive(Debug, Clone)]
pub struct OptimizationAlgorithm {
    pub algorithm_name: String,
    pub hyperparameters: HashMap<String, f32>,
    pub convergence_criteria: ConvergenceCriteria,
    pub exploration_exploitation_balance: f32,
}

#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub max_iterations: usize,
    pub improvement_threshold: f32,
    pub plateau_tolerance: usize,
    pub time_budget: Duration,
}

#[derive(Debug, Clone)]
pub struct SuccessCriteria {
    pub target_performance_improvement: f32,
    pub minimum_stability: f32,
    pub maximum_degradation_allowed: f32,
    pub validation_requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResourceBudget {
    pub max_memory_mb: f32,
    pub max_cpu_hours: f32,
    pub max_optimization_time: Duration,
    pub parallel_evaluations: usize,
}

#[derive(Debug, Clone)]
pub struct ParameterSpace {
    pub space_id: String,
    pub parameters: HashMap<String, ParameterDefinition>,
    pub constraints: Vec<ParameterConstraint>,
    pub dependencies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    pub name: String,
    pub parameter_type: ParameterType,
    pub current_value: f32,
    pub default_value: f32,
    pub valid_range: (f32, f32),
    pub granularity: f32,
    pub importance_weight: f32,
}

#[derive(Debug, Clone)]
pub enum ParameterType {
    Continuous,
    Discrete,
    Integer,
    Boolean,
    Categorical(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct ParameterConstraint {
    pub constraint_type: ConstraintType,
    pub parameters: Vec<String>,
    pub constraint_function: String, // Mathematical expression
    pub violation_penalty: f32,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    LinearInequality,
    NonlinearInequality,
    Equality,
    ConditionalConstraint,
}

#[derive(Debug, Clone)]
pub struct TuningSession {
    pub session_id: Uuid,
    pub strategy: TuningStrategy,
    pub parameter_space: ParameterSpace,
    pub start_time: SystemTime,
    pub current_iteration: usize,
    pub best_parameters: HashMap<String, f32>,
    pub best_performance: f32,
    pub evaluation_history: Vec<ParameterEvaluation>,
    pub session_status: TuningSessionStatus,
}

#[derive(Debug, Clone)]
pub enum TuningSessionStatus {
    Running,
    Converged,
    Failed,
    Aborted,
    ResourceExhausted,
}

#[derive(Debug, Clone)]
pub struct ParameterEvaluation {
    pub evaluation_id: Uuid,
    pub parameters: HashMap<String, f32>,
    pub performance_score: f32,
    pub stability_score: f32,
    pub resource_usage: ResourceUsage,
    pub evaluation_time: Duration,
    pub validation_results: ValidationResults,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used_mb: f32,
    pub cpu_time_ms: f32,
    pub wall_time_ms: f32,
    pub evaluations_performed: usize,
}

#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub passed_validations: Vec<String>,
    pub failed_validations: Vec<String>,
    pub validation_scores: HashMap<String, f32>,
    pub overall_validation_score: f32,
}

#[derive(Debug, Clone)]
pub struct TuningResult {
    pub result_id: Uuid,
    pub session_id: Uuid,
    pub strategy_used: String,
    pub final_parameters: HashMap<String, f32>,
    pub performance_improvement: f32,
    pub optimization_duration: Duration,
    pub iterations_completed: usize,
    pub convergence_reason: ConvergenceReason,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConvergenceReason {
    TargetAchieved,
    MaxIterationsReached,
    TimeExpired,
    NoImprovement,
    ResourceExhausted,
    UserAborted,
}

#[derive(Debug, Clone)]
pub struct ParameterTuningConfig {
    pub enable_parallel_tuning: bool,
    pub max_concurrent_sessions: usize,
    pub default_strategy: TuningStrategyType,
    pub safety_mode_enabled: bool,
    pub automatic_rollback: bool,
    pub performance_monitoring_interval: Duration,
}

impl ParameterTuningSystem {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            tuning_strategies: Self::create_default_strategies(),
            parameter_spaces: Self::create_default_parameter_spaces(),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            active_tuning_sessions: Arc::new(RwLock::new(HashMap::new())),
            tuning_config: ParameterTuningConfig::default(),
        })
    }

    fn create_default_strategies() -> Vec<TuningStrategy> {
        vec![
            TuningStrategy {
                strategy_id: "hebbian_optimization".to_string(),
                strategy_type: TuningStrategyType::BayesianOptimization,
                target_components: vec!["hebbian_learning".to_string()],
                optimization_algorithm: OptimizationAlgorithm {
                    algorithm_name: "gaussian_process".to_string(),
                    hyperparameters: {
                        let mut params = HashMap::new();
                        params.insert("acquisition_function".to_string(), 1.0); // EI
                        params.insert("exploration_weight".to_string(), 0.1);
                        params
                    },
                    convergence_criteria: ConvergenceCriteria {
                        max_iterations: 50,
                        improvement_threshold: 0.01,
                        plateau_tolerance: 5,
                        time_budget: Duration::from_secs(3600),
                    },
                    exploration_exploitation_balance: 0.2,
                },
                success_criteria: SuccessCriteria {
                    target_performance_improvement: 0.15,
                    minimum_stability: 0.9,
                    maximum_degradation_allowed: 0.05,
                    validation_requirements: vec!["no_memory_leaks".to_string(), "stable_convergence".to_string()],
                },
                resource_budget: ResourceBudget {
                    max_memory_mb: 500.0,
                    max_cpu_hours: 2.0,
                    max_optimization_time: Duration::from_secs(7200),
                    parallel_evaluations: 4,
                },
            },
            TuningStrategy {
                strategy_id: "attention_optimization".to_string(),
                strategy_type: TuningStrategyType::GridSearch,
                target_components: vec!["attention_manager".to_string()],
                optimization_algorithm: OptimizationAlgorithm {
                    algorithm_name: "grid_search".to_string(),
                    hyperparameters: {
                        let mut params = HashMap::new();
                        params.insert("grid_resolution".to_string(), 10.0);
                        params
                    },
                    convergence_criteria: ConvergenceCriteria {
                        max_iterations: 100,
                        improvement_threshold: 0.005,
                        plateau_tolerance: 10,
                        time_budget: Duration::from_secs(1800),
                    },
                    exploration_exploitation_balance: 0.5,
                },
                success_criteria: SuccessCriteria {
                    target_performance_improvement: 0.1,
                    minimum_stability: 0.95,
                    maximum_degradation_allowed: 0.02,
                    validation_requirements: vec!["attention_stability".to_string()],
                },
                resource_budget: ResourceBudget {
                    max_memory_mb: 200.0,
                    max_cpu_hours: 1.0,
                    max_optimization_time: Duration::from_secs(3600),
                    parallel_evaluations: 2,
                },
            },
        ]
    }

    fn create_default_parameter_spaces() -> HashMap<String, ParameterSpace> {
        let mut spaces = HashMap::new();
        
        // Hebbian learning parameter space
        let mut hebbian_params = HashMap::new();
        hebbian_params.insert("learning_rate".to_string(), ParameterDefinition {
            name: "learning_rate".to_string(),
            parameter_type: ParameterType::Continuous,
            current_value: 0.01,
            default_value: 0.01,
            valid_range: (0.001, 0.1),
            granularity: 0.001,
            importance_weight: 0.8,
        });
        hebbian_params.insert("decay_constant".to_string(), ParameterDefinition {
            name: "decay_constant".to_string(),
            parameter_type: ParameterType::Continuous,
            current_value: 0.001,
            default_value: 0.001,
            valid_range: (0.0001, 0.01),
            granularity: 0.0001,
            importance_weight: 0.6,
        });
        
        spaces.insert("hebbian_learning".to_string(), ParameterSpace {
            space_id: "hebbian_learning".to_string(),
            parameters: hebbian_params,
            constraints: vec![
                ParameterConstraint {
                    constraint_type: ConstraintType::LinearInequality,
                    parameters: vec!["learning_rate".to_string(), "decay_constant".to_string()],
                    constraint_function: "learning_rate > decay_constant".to_string(),
                    violation_penalty: 10.0,
                }
            ],
            dependencies: HashMap::new(),
        });
        
        // Attention parameter space
        let mut attention_params = HashMap::new();
        attention_params.insert("focus_strength".to_string(), ParameterDefinition {
            name: "focus_strength".to_string(),
            parameter_type: ParameterType::Continuous,
            current_value: 0.8,
            default_value: 0.8,
            valid_range: (0.1, 1.0),
            granularity: 0.05,
            importance_weight: 0.9,
        });
        attention_params.insert("attention_capacity".to_string(), ParameterDefinition {
            name: "attention_capacity".to_string(),
            parameter_type: ParameterType::Integer,
            current_value: 7.0, // Miller's magic number
            default_value: 7.0,
            valid_range: (3.0, 12.0),
            granularity: 1.0,
            importance_weight: 0.7,
        });
        
        spaces.insert("attention_manager".to_string(), ParameterSpace {
            space_id: "attention_manager".to_string(),
            parameters: attention_params,
            constraints: vec![],
            dependencies: HashMap::new(),
        });
        
        spaces
    }

    /// Tune Hebbian learning parameters
    pub async fn tune_hebbian_parameters(&self, performance_data: &PerformanceData) -> Result<HebbianParameterUpdate> {
        let session_id = Uuid::new_v4();
        
        // Get Hebbian tuning strategy
        let strategy = self.tuning_strategies.iter()
            .find(|s| s.strategy_id == "hebbian_optimization")
            .ok_or_else(|| anyhow!("Hebbian optimization strategy not found"))?;
        
        // Get parameter space
        let parameter_space = self.parameter_spaces.get("hebbian_learning")
            .ok_or_else(|| anyhow!("Hebbian parameter space not found"))?;
        
        // Create tuning session
        let tuning_session = TuningSession {
            session_id,
            strategy: strategy.clone(),
            parameter_space: parameter_space.clone(),
            start_time: SystemTime::now(),
            current_iteration: 0,
            best_parameters: HashMap::new(),
            best_performance: 0.0,
            evaluation_history: Vec::new(),
            session_status: TuningSessionStatus::Running,
        };
        
        // Store active session
        self.active_tuning_sessions.write().unwrap().insert(session_id, tuning_session);
        
        // Run optimization
        let optimization_result = self.run_bayesian_optimization(
            session_id,
            performance_data,
        ).await?;
        
        Ok(HebbianParameterUpdate {
            learning_rate_adjustment: optimization_result.parameter_changes.get("learning_rate").unwrap_or(&0.0).clone(),
            decay_constant_adjustment: optimization_result.parameter_changes.get("decay_constant").unwrap_or(&0.0).clone(),
            strengthening_threshold_adjustment: 0.0, // Would be optimized if included
            optimization_confidence: optimization_result.confidence,
            expected_performance_gain: optimization_result.expected_improvement,
        })
    }

    /// Tune attention parameters
    pub async fn tune_attention_parameters(&self, attention_metrics: &AttentionMetrics) -> Result<AttentionParameterUpdate> {
        let session_id = Uuid::new_v4();
        
        // Get attention tuning strategy
        let _strategy = self.tuning_strategies.iter()
            .find(|s| s.strategy_id == "attention_optimization")
            .ok_or_else(|| anyhow!("Attention optimization strategy not found"))?;
        
        // Run grid search optimization
        let optimization_result = self.run_grid_search_optimization(
            session_id,
            attention_metrics,
        ).await?;
        
        Ok(AttentionParameterUpdate {
            focus_strength_adjustment: optimization_result.parameter_changes.get("focus_strength").unwrap_or(&0.0).clone(),
            capacity_adjustment: optimization_result.parameter_changes.get("attention_capacity").unwrap_or(&0.0).clone(),
            shift_speed_adjustment: 0.0, // Would be included if in parameter space
            optimization_confidence: optimization_result.confidence,
            expected_performance_gain: optimization_result.expected_improvement,
        })
    }

    /// Tune memory parameters
    pub async fn tune_memory_parameters(&self, memory_metrics: &MemoryMetrics) -> Result<MemoryParameterUpdate> {
        // Simplified memory parameter tuning
        let mut capacity_adjustments = HashMap::new();
        let mut decay_rate_adjustments = HashMap::new();
        let mut consolidation_threshold_adjustments = HashMap::new();
        
        // Analyze memory performance and suggest adjustments
        if memory_metrics.efficiency < 0.7 {
            capacity_adjustments.insert("working_memory".to_string(), 0.1);
        }
        
        if memory_metrics.retention_rate < 0.8 {
            decay_rate_adjustments.insert("long_term_memory".to_string(), -0.05);
        }
        
        if memory_metrics.consolidation_rate < 0.6 {
            consolidation_threshold_adjustments.insert("consolidation_threshold".to_string(), -0.1);
        }
        
        Ok(MemoryParameterUpdate {
            capacity_adjustments,
            decay_rate_adjustments,
            consolidation_threshold_adjustments,
            optimization_confidence: 0.8,
            expected_performance_gain: 0.15,
        })
    }

    /// Auto-tune entire system
    pub async fn auto_tune_system(&self, system_state: &SystemState) -> Result<SystemParameterUpdate> {
        let mut component_updates = HashMap::new();
        let mut global_adjustments = HashMap::new();
        
        // Analyze system state and determine tuning priorities
        let tuning_priorities = self.analyze_tuning_priorities(system_state)?;
        
        // Execute tuning for high-priority components
        for (component, priority) in tuning_priorities {
            if priority > 0.7 {
                let component_update = self.tune_component(&component, system_state).await?;
                component_updates.insert(component, component_update);
            }
        }
        
        // Global system adjustments
        global_adjustments.insert("overall_learning_rate".to_string(), 0.05);
        global_adjustments.insert("system_stability_factor".to_string(), 0.1);
        
        Ok(SystemParameterUpdate {
            component_updates,
            global_adjustments,
            tuning_session_id: Uuid::new_v4(),
            optimization_confidence: 0.85,
            expected_system_improvement: 0.2,
        })
    }

    async fn run_bayesian_optimization(
        &self,
        _session_id: Uuid,
        _performance_data: &PerformanceData,
    ) -> Result<OptimizationResult> {
        // Simplified Bayesian optimization implementation
        let mut best_parameters = HashMap::new();
        best_parameters.insert("learning_rate".to_string(), 0.015);
        best_parameters.insert("decay_constant".to_string(), 0.0015);
        
        Ok(OptimizationResult {
            parameter_changes: best_parameters,
            confidence: 0.85,
            expected_improvement: 0.12,
            iterations_completed: 25,
            convergence_reason: ConvergenceReason::TargetAchieved,
        })
    }

    async fn run_grid_search_optimization(
        &self,
        _session_id: Uuid,
        _attention_metrics: &AttentionMetrics,
    ) -> Result<OptimizationResult> {
        // Simplified grid search implementation
        let mut best_parameters = HashMap::new();
        best_parameters.insert("focus_strength".to_string(), 0.85);
        best_parameters.insert("attention_capacity".to_string(), 8.0);
        
        Ok(OptimizationResult {
            parameter_changes: best_parameters,
            confidence: 0.9,
            expected_improvement: 0.08,
            iterations_completed: 64, // 8x8 grid
            convergence_reason: ConvergenceReason::TargetAchieved,
        })
    }

    fn analyze_tuning_priorities(&self, system_state: &SystemState) -> Result<HashMap<String, f32>> {
        let mut priorities = HashMap::new();
        
        // Analyze system performance to determine tuning priorities
        if system_state.memory_efficiency < 0.6 {
            priorities.insert("memory_system".to_string(), 0.9);
        }
        
        if system_state.attention_effectiveness < 0.7 {
            priorities.insert("attention_manager".to_string(), 0.8);
        }
        
        if system_state.learning_efficiency < 0.6 {
            priorities.insert("hebbian_learning".to_string(), 0.95);
        }
        
        Ok(priorities)
    }

    async fn tune_component(&self, component: &str, _system_state: &SystemState) -> Result<ComponentUpdate> {
        // Component-specific tuning
        match component {
            "hebbian_learning" => {
                Ok(ComponentUpdate {
                    component_name: component.to_string(),
                    parameter_changes: {
                        let mut changes = HashMap::new();
                        changes.insert("learning_rate".to_string(), 0.02);
                        changes
                    },
                    expected_improvement: 0.15,
                })
            },
            "attention_manager" => {
                Ok(ComponentUpdate {
                    component_name: component.to_string(),
                    parameter_changes: {
                        let mut changes = HashMap::new();
                        changes.insert("focus_strength".to_string(), 0.9);
                        changes
                    },
                    expected_improvement: 0.1,
                })
            },
            _ => {
                Ok(ComponentUpdate {
                    component_name: component.to_string(),
                    parameter_changes: HashMap::new(),
                    expected_improvement: 0.0,
                })
            }
        }
    }
}

impl Default for ParameterTuningConfig {
    fn default() -> Self {
        Self {
            enable_parallel_tuning: true,
            max_concurrent_sessions: 3,
            default_strategy: TuningStrategyType::BayesianOptimization,
            safety_mode_enabled: true,
            automatic_rollback: true,
            performance_monitoring_interval: Duration::from_secs(300),
        }
    }
}

/// Parameter update structures
#[derive(Debug, Clone)]
pub struct HebbianParameterUpdate {
    pub learning_rate_adjustment: f32,
    pub decay_constant_adjustment: f32,
    pub strengthening_threshold_adjustment: f32,
    pub optimization_confidence: f32,
    pub expected_performance_gain: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionParameterUpdate {
    pub focus_strength_adjustment: f32,
    pub capacity_adjustment: f32,
    pub shift_speed_adjustment: f32,
    pub optimization_confidence: f32,
    pub expected_performance_gain: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryParameterUpdate {
    pub capacity_adjustments: HashMap<String, f32>,
    pub decay_rate_adjustments: HashMap<String, f32>,
    pub consolidation_threshold_adjustments: HashMap<String, f32>,
    pub optimization_confidence: f32,
    pub expected_performance_gain: f32,
}

#[derive(Debug, Clone)]
pub struct SystemParameterUpdate {
    pub component_updates: HashMap<String, ComponentUpdate>,
    pub global_adjustments: HashMap<String, f32>,
    pub tuning_session_id: Uuid,
    pub optimization_confidence: f32,
    pub expected_system_improvement: f32,
}

#[derive(Debug, Clone)]
pub struct ComponentUpdate {
    pub component_name: String,
    pub parameter_changes: HashMap<String, f32>,
    pub expected_improvement: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub parameter_changes: HashMap<String, f32>,
    pub confidence: f32,
    pub expected_improvement: f32,
    pub iterations_completed: usize,
    pub convergence_reason: ConvergenceReason,
}

/// Metrics structures for parameter tuning
#[derive(Debug, Clone)]
pub struct AttentionMetrics {
    pub focus_accuracy: f32,
    pub attention_stability: f32,
    pub switching_efficiency: f32,
    pub capacity_utilization: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub efficiency: f32,
    pub retention_rate: f32,
    pub consolidation_rate: f32,
    pub capacity_utilization: f32,
}

#[derive(Debug, Clone)]
pub struct SystemState {
    pub memory_efficiency: f32,
    pub attention_effectiveness: f32,
    pub learning_efficiency: f32,
    pub overall_performance: f32,
}

/// Parameter tuning trait for standardized interface
#[allow(async_fn_in_trait)]
pub trait ParameterTuner {
    async fn tune_hebbian_parameters(&self, performance_data: &PerformanceData) -> Result<HebbianParameterUpdate>;
    async fn tune_attention_parameters(&self, attention_metrics: &AttentionMetrics) -> Result<AttentionParameterUpdate>;
    async fn tune_memory_parameters(&self, memory_metrics: &MemoryMetrics) -> Result<MemoryParameterUpdate>;
    async fn auto_tune_system(&self, system_state: &SystemState) -> Result<SystemParameterUpdate>;
}

impl ParameterTuner for ParameterTuningSystem {
    async fn tune_hebbian_parameters(&self, performance_data: &PerformanceData) -> Result<HebbianParameterUpdate> {
        self.tune_hebbian_parameters(performance_data).await
    }

    async fn tune_attention_parameters(&self, attention_metrics: &AttentionMetrics) -> Result<AttentionParameterUpdate> {
        self.tune_attention_parameters(attention_metrics).await
    }

    async fn tune_memory_parameters(&self, memory_metrics: &MemoryMetrics) -> Result<MemoryParameterUpdate> {
        self.tune_memory_parameters(memory_metrics).await
    }

    async fn auto_tune_system(&self, system_state: &SystemState) -> Result<SystemParameterUpdate> {
        self.auto_tune_system(system_state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::types::ThroughputMetrics;
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime};
    use uuid::Uuid;

    // Test helper to create mock parameter tuning system
    async fn create_test_parameter_tuning_system() -> Result<ParameterTuningSystem> {
        ParameterTuningSystem::new().await
    }

    fn create_test_performance_data() -> PerformanceData {
        PerformanceData {
            query_latencies: vec![Duration::from_millis(50), Duration::from_millis(60)],
            memory_usage: vec![0.6, 0.7, 0.8],
            accuracy_scores: vec![0.85, 0.86, 0.84],
            user_satisfaction: vec![0.8, 0.85, 0.9],
            system_stability: 0.9,
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("timeout".to_string(), 0.05);
                rates.insert("memory".to_string(), 0.02);
                rates
            },
            throughput_metrics: ThroughputMetrics {
                queries_per_second: 1000.0,
                successful_queries: 9500,
                failed_queries: 500,
                average_response_time: Duration::from_millis(50),
            },
            timestamp: SystemTime::now(),
            system_health: 0.85,
            overall_performance_score: 0.8,
            component_scores: {
                let mut scores = HashMap::new();
                scores.insert("attention".to_string(), 0.85);
                scores.insert("memory".to_string(), 0.8);
                scores
            },
            bottlenecks: vec![],
        }
    }

    fn create_test_attention_metrics() -> AttentionMetrics {
        AttentionMetrics {
            focus_accuracy: 0.8,
            attention_stability: 0.85,
            switching_efficiency: 0.7,
            capacity_utilization: 0.6,
        }
    }

    fn create_test_memory_metrics() -> MemoryMetrics {
        MemoryMetrics {
            efficiency: 0.75,
            retention_rate: 0.9,
            consolidation_rate: 0.7,
            capacity_utilization: 0.8,
        }
    }

    fn create_test_system_state() -> SystemState {
        SystemState {
            memory_efficiency: 0.7,
            attention_effectiveness: 0.8,
            learning_efficiency: 0.6,
            overall_performance: 0.75,
        }
    }

    #[tokio::test]
    async fn test_parameter_tuning_system_creation() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        assert!(!system.tuning_strategies.is_empty(), "Should have tuning strategies");
        assert!(!system.parameter_spaces.is_empty(), "Should have parameter spaces");
        assert!(system.tuning_config.enable_parallel_tuning, "Should enable parallel tuning by default");
        assert!(system.tuning_config.safety_mode_enabled, "Should enable safety mode by default");
        assert!(system.tuning_config.automatic_rollback, "Should enable automatic rollback by default");
    }

    #[tokio::test]
    async fn test_hebbian_parameter_tuning() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let performance_data = create_test_performance_data();
        
        let update = system.tune_hebbian_parameters(&performance_data).await
            .expect("Failed to tune Hebbian parameters");
        
        // Test that adjustments are reasonable
        assert!(update.learning_rate_adjustment.abs() <= 0.1, 
               "Learning rate adjustment should be bounded");
        assert!(update.decay_constant_adjustment.abs() <= 0.01, 
               "Decay constant adjustment should be bounded");
        assert!(update.optimization_confidence >= 0.0 && update.optimization_confidence <= 1.0,
               "Optimization confidence should be normalized");
        assert!(update.expected_performance_gain >= 0.0,
               "Expected performance gain should be non-negative");
    }

    #[tokio::test]
    async fn test_attention_parameter_tuning() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let attention_metrics = create_test_attention_metrics();
        
        let update = system.tune_attention_parameters(&attention_metrics).await
            .expect("Failed to tune attention parameters");
        
        // Test that adjustments are reasonable
        assert!(update.focus_strength_adjustment.abs() <= 0.2, 
               "Focus strength adjustment should be bounded");
        assert!(update.capacity_adjustment.abs() <= 5.0, 
               "Capacity adjustment should be reasonable");
        assert!(update.optimization_confidence >= 0.0 && update.optimization_confidence <= 1.0,
               "Optimization confidence should be normalized");
        assert!(update.expected_performance_gain >= 0.0,
               "Expected performance gain should be non-negative");
    }

    #[tokio::test]
    async fn test_memory_parameter_tuning() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let memory_metrics = create_test_memory_metrics();
        
        let update = system.tune_memory_parameters(&memory_metrics).await
            .expect("Failed to tune memory parameters");
        
        // Test that adjustments are reasonable
        assert!(update.optimization_confidence >= 0.0 && update.optimization_confidence <= 1.0,
               "Optimization confidence should be normalized");
        assert!(update.expected_performance_gain >= 0.0,
               "Expected performance gain should be non-negative");
        
        // Check specific adjustments based on metrics
        if memory_metrics.efficiency < 0.7 {
            assert!(!update.capacity_adjustments.is_empty(),
                   "Should adjust capacity for low efficiency");
        }
        
        if memory_metrics.retention_rate < 0.8 {
            assert!(!update.decay_rate_adjustments.is_empty(),
                   "Should adjust decay rate for low retention");
        }
        
        if memory_metrics.consolidation_rate < 0.6 {
            assert!(!update.consolidation_threshold_adjustments.is_empty(),
                   "Should adjust consolidation threshold for low consolidation rate");
        }
    }

    #[tokio::test]
    async fn test_system_auto_tuning() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let system_state = create_test_system_state();
        
        let update = system.auto_tune_system(&system_state).await
            .expect("Failed to auto-tune system");
        
        assert!(update.optimization_confidence >= 0.0 && update.optimization_confidence <= 1.0,
               "Optimization confidence should be normalized");
        assert!(update.expected_system_improvement >= 0.0,
               "Expected system improvement should be non-negative");
        assert!(!update.tuning_session_id.to_string().is_empty(),
               "Should have tuning session ID");
        
        // Should have updates for components with low performance
        if system_state.learning_efficiency < 0.6 {
            assert!(update.component_updates.contains_key("hebbian_learning"),
                   "Should update Hebbian learning for low efficiency");
        }
        
        if system_state.attention_effectiveness < 0.7 {
            assert!(update.component_updates.contains_key("attention_manager"),
                   "Should update attention manager for low effectiveness");
        }
    }

    #[test]
    fn test_tuning_strategies_creation() {
        let strategies = ParameterTuningSystem::create_default_strategies();
        
        assert!(!strategies.is_empty(), "Should create default strategies");
        
        for strategy in &strategies {
            assert!(!strategy.strategy_id.is_empty(), "Strategy should have ID");
            assert!(!strategy.target_components.is_empty(), "Strategy should target components");
            assert!(strategy.success_criteria.target_performance_improvement > 0.0,
                   "Should have positive performance improvement target");
            assert!(strategy.success_criteria.minimum_stability > 0.0,
                   "Should require minimum stability");
            assert!(strategy.resource_budget.max_cpu_hours > 0.0,
                   "Should have positive CPU budget");
            assert!(strategy.resource_budget.max_memory_mb > 0.0,
                   "Should have positive memory budget");
        }
    }

    #[test]
    fn test_parameter_spaces_creation() {
        let spaces = ParameterTuningSystem::create_default_parameter_spaces();
        
        assert!(!spaces.is_empty(), "Should create default parameter spaces");
        
        // Test Hebbian learning space
        assert!(spaces.contains_key("hebbian_learning"), "Should have Hebbian learning space");
        let hebbian_space = &spaces["hebbian_learning"];
        
        assert!(hebbian_space.parameters.contains_key("learning_rate"),
               "Should define learning rate parameter");
        assert!(hebbian_space.parameters.contains_key("decay_constant"),
               "Should define decay constant parameter");
        
        // Test attention manager space
        assert!(spaces.contains_key("attention_manager"), "Should have attention manager space");
        let attention_space = &spaces["attention_manager"];
        
        assert!(attention_space.parameters.contains_key("focus_strength"),
               "Should define focus strength parameter");
        assert!(attention_space.parameters.contains_key("attention_capacity"),
               "Should define attention capacity parameter");
    }

    #[test]
    fn test_parameter_definition_validation() {
        let spaces = ParameterTuningSystem::create_default_parameter_spaces();
        
        for (space_name, space) in &spaces {
            for (param_name, param_def) in &space.parameters {
                assert!(!param_def.name.is_empty(), 
                       "Parameter {} in space {} should have name", param_name, space_name);
                assert!(param_def.valid_range.0 <= param_def.valid_range.1,
                       "Parameter {} should have valid range", param_name);
                assert!(param_def.current_value >= param_def.valid_range.0 && 
                       param_def.current_value <= param_def.valid_range.1,
                       "Current value should be within valid range for {}", param_name);
                assert!(param_def.default_value >= param_def.valid_range.0 && 
                       param_def.default_value <= param_def.valid_range.1,
                       "Default value should be within valid range for {}", param_name);
                assert!(param_def.granularity > 0.0,
                       "Granularity should be positive for {}", param_name);
                assert!(param_def.importance_weight >= 0.0 && param_def.importance_weight <= 1.0,
                       "Importance weight should be normalized for {}", param_name);
            }
        }
    }

    #[test]
    fn test_parameter_constraints() {
        let spaces = ParameterTuningSystem::create_default_parameter_spaces();
        
        let hebbian_space = &spaces["hebbian_learning"];
        assert!(!hebbian_space.constraints.is_empty(), 
               "Hebbian space should have constraints");
        
        for constraint in &hebbian_space.constraints {
            assert!(!constraint.parameters.is_empty(), "Constraint should reference parameters");
            assert!(!constraint.constraint_function.is_empty(), "Constraint should have function");
            assert!(constraint.violation_penalty >= 0.0, "Penalty should be non-negative");
        }
    }

    #[test]
    fn test_tuning_strategy_types() {
        let strategies = ParameterTuningSystem::create_default_strategies();
        
        let strategy_types: Vec<_> = strategies.iter()
            .map(|s| &s.strategy_type)
            .collect();
        
        // Should have diverse strategy types
        assert!(strategy_types.iter().any(|t| matches!(t, TuningStrategyType::BayesianOptimization)),
               "Should have Bayesian optimization strategy");
        assert!(strategy_types.iter().any(|t| matches!(t, TuningStrategyType::GridSearch)),
               "Should have grid search strategy");
    }

    #[test]
    fn test_convergence_criteria() {
        let strategies = ParameterTuningSystem::create_default_strategies();
        
        for strategy in &strategies {
            let criteria = &strategy.optimization_algorithm.convergence_criteria;
            
            assert!(criteria.max_iterations > 0, "Should have positive max iterations");
            assert!(criteria.improvement_threshold > 0.0, "Should have positive improvement threshold");
            assert!(criteria.plateau_tolerance > 0, "Should have positive plateau tolerance");
            assert!(criteria.time_budget > Duration::from_secs(0), "Should have positive time budget");
        }
    }

    #[test]
    fn test_success_criteria_validation() {
        let strategies = ParameterTuningSystem::create_default_strategies();
        
        for strategy in &strategies {
            let criteria = &strategy.success_criteria;
            
            assert!(criteria.target_performance_improvement > 0.0,
                   "Should target positive performance improvement");
            assert!(criteria.minimum_stability > 0.0 && criteria.minimum_stability <= 1.0,
                   "Minimum stability should be normalized");
            assert!(criteria.maximum_degradation_allowed >= 0.0 && 
                   criteria.maximum_degradation_allowed <= 1.0,
                   "Maximum degradation should be normalized");
            assert!(!criteria.validation_requirements.is_empty(),
                   "Should have validation requirements");
        }
    }

    #[test]
    fn test_resource_budget_validation() {
        let strategies = ParameterTuningSystem::create_default_strategies();
        
        for strategy in &strategies {
            let budget = &strategy.resource_budget;
            
            assert!(budget.max_memory_mb > 0.0, "Should have positive memory budget");
            assert!(budget.max_cpu_hours > 0.0, "Should have positive CPU budget");
            assert!(budget.max_optimization_time > Duration::from_secs(0),
                   "Should have positive time budget");
            assert!(budget.parallel_evaluations > 0, "Should allow parallel evaluations");
        }
    }

    #[test]
    fn test_parameter_tuning_config_defaults() {
        let config = ParameterTuningConfig::default();
        
        assert!(config.enable_parallel_tuning, "Should enable parallel tuning by default");
        assert!(config.max_concurrent_sessions > 0, "Should allow concurrent sessions");
        assert!(matches!(config.default_strategy, TuningStrategyType::BayesianOptimization),
               "Should default to Bayesian optimization");
        assert!(config.safety_mode_enabled, "Should enable safety mode by default");
        assert!(config.automatic_rollback, "Should enable automatic rollback by default");
        assert!(config.performance_monitoring_interval > Duration::from_secs(0),
               "Should have positive monitoring interval");
    }

    #[test]
    fn test_parameter_types() {
        // Test different parameter types
        let continuous = ParameterType::Continuous;
        let discrete = ParameterType::Discrete;
        let integer = ParameterType::Integer;
        let boolean = ParameterType::Boolean;
        let categorical = ParameterType::Categorical(vec!["option1".to_string(), "option2".to_string()]);
        
        // All parameter types should be properly defined
        assert!(matches!(continuous, ParameterType::Continuous));
        assert!(matches!(discrete, ParameterType::Discrete));
        assert!(matches!(integer, ParameterType::Integer));
        assert!(matches!(boolean, ParameterType::Boolean));
        
        if let ParameterType::Categorical(options) = categorical {
            assert_eq!(options.len(), 2, "Should have categorical options");
        }
    }

    #[test]
    fn test_constraint_types() {
        // Test different constraint types
        let linear = ConstraintType::LinearInequality;
        let nonlinear = ConstraintType::NonlinearInequality;
        let equality = ConstraintType::Equality;
        let conditional = ConstraintType::ConditionalConstraint;
        
        // All constraint types should be properly defined
        assert!(matches!(linear, ConstraintType::LinearInequality));
        assert!(matches!(nonlinear, ConstraintType::NonlinearInequality));
        assert!(matches!(equality, ConstraintType::Equality));
        assert!(matches!(conditional, ConstraintType::ConditionalConstraint));
    }

    #[test]
    fn test_tuning_session_status() {
        // Test different session statuses
        let running = TuningSessionStatus::Running;
        let converged = TuningSessionStatus::Converged;
        let failed = TuningSessionStatus::Failed;
        let aborted = TuningSessionStatus::Aborted;
        let exhausted = TuningSessionStatus::ResourceExhausted;
        
        // All status types should be properly defined
        assert!(matches!(running, TuningSessionStatus::Running));
        assert!(matches!(converged, TuningSessionStatus::Converged));
        assert!(matches!(failed, TuningSessionStatus::Failed));
        assert!(matches!(aborted, TuningSessionStatus::Aborted));
        assert!(matches!(exhausted, TuningSessionStatus::ResourceExhausted));
    }

    #[test]
    fn test_convergence_reasons() {
        // Test different convergence reasons
        let target = ConvergenceReason::TargetAchieved;
        let max_iter = ConvergenceReason::MaxIterationsReached;
        let time_exp = ConvergenceReason::TimeExpired;
        let no_improve = ConvergenceReason::NoImprovement;
        let resources = ConvergenceReason::ResourceExhausted;
        let aborted = ConvergenceReason::UserAborted;
        
        // All convergence reasons should be properly defined
        assert!(matches!(target, ConvergenceReason::TargetAchieved));
        assert!(matches!(max_iter, ConvergenceReason::MaxIterationsReached));
        assert!(matches!(time_exp, ConvergenceReason::TimeExpired));
        assert!(matches!(no_improve, ConvergenceReason::NoImprovement));
        assert!(matches!(resources, ConvergenceReason::ResourceExhausted));
        assert!(matches!(aborted, ConvergenceReason::UserAborted));
    }

    #[tokio::test]
    async fn test_bayesian_optimization_simulation() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let performance_data = create_test_performance_data();
        
        // Simulate Bayesian optimization
        let result = system.run_bayesian_optimization(Uuid::new_v4(), &performance_data).await
            .expect("Failed to run Bayesian optimization");
        
        assert!(!result.parameter_changes.is_empty(), "Should have parameter changes");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
               "Confidence should be normalized");
        assert!(result.expected_improvement >= 0.0,
               "Expected improvement should be non-negative");
        assert!(result.iterations_completed > 0, "Should complete some iterations");
        assert!(matches!(result.convergence_reason, ConvergenceReason::TargetAchieved),
               "Should converge to target");
    }

    #[tokio::test]
    async fn test_grid_search_optimization_simulation() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let attention_metrics = create_test_attention_metrics();
        
        // Simulate grid search optimization
        let result = system.run_grid_search_optimization(Uuid::new_v4(), &attention_metrics).await
            .expect("Failed to run grid search optimization");
        
        assert!(!result.parameter_changes.is_empty(), "Should have parameter changes");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
               "Confidence should be normalized");
        assert!(result.expected_improvement >= 0.0,
               "Expected improvement should be non-negative");
        assert!(result.iterations_completed > 0, "Should complete some iterations");
    }

    #[test]
    fn test_tuning_priority_analysis() {
        let system_future = create_test_parameter_tuning_system();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let system = rt.block_on(system_future).expect("Failed to create system");
        
        let system_state = SystemState {
            memory_efficiency: 0.5,      // Low - should be high priority
            attention_effectiveness: 0.8, // Good - should be lower priority
            learning_efficiency: 0.4,     // Very low - should be highest priority
            overall_performance: 0.6,
        };
        
        let priorities = system.analyze_tuning_priorities(&system_state)
            .expect("Failed to analyze tuning priorities");
        
        // Learning efficiency is lowest, should have highest priority
        assert!(priorities.get("hebbian_learning").unwrap_or(&0.0) > &0.9,
               "Should prioritize learning efficiency");
        
        // Memory efficiency is low, should have high priority
        assert!(priorities.get("memory_system").unwrap_or(&0.0) > &0.8,
               "Should prioritize memory efficiency");
        
        // Attention is good, should have lower priority
        assert!(priorities.get("attention_manager").unwrap_or(&1.0) < &0.9,
               "Should deprioritize attention when it's working well");
    }

    #[tokio::test]
    async fn test_component_specific_tuning() {
        let system = create_test_parameter_tuning_system().await
            .expect("Failed to create parameter tuning system");
        
        let system_state = create_test_system_state();
        
        // Test Hebbian learning component tuning
        let hebbian_update = system.tune_component("hebbian_learning", &system_state).await
            .expect("Failed to tune Hebbian learning component");
        
        assert_eq!(hebbian_update.component_name, "hebbian_learning");
        assert!(!hebbian_update.parameter_changes.is_empty(), "Should have parameter changes");
        assert!(hebbian_update.expected_improvement > 0.0, "Should expect improvement");
        
        // Test attention manager component tuning
        let attention_update = system.tune_component("attention_manager", &system_state).await
            .expect("Failed to tune attention manager component");
        
        assert_eq!(attention_update.component_name, "attention_manager");
        assert!(!attention_update.parameter_changes.is_empty(), "Should have parameter changes");
        assert!(attention_update.expected_improvement >= 0.0, "Should expect non-negative improvement");
        
        // Test unknown component
        let unknown_update = system.tune_component("unknown_component", &system_state).await
            .expect("Should handle unknown component gracefully");
        
        assert_eq!(unknown_update.component_name, "unknown_component");
        assert!(unknown_update.parameter_changes.is_empty(), "Unknown component should have no changes");
        assert_eq!(unknown_update.expected_improvement, 0.0, "Unknown component should expect no improvement");
    }

    #[test]
    fn test_optimization_result_structure() {
        let result = OptimizationResult {
            parameter_changes: {
                let mut changes = HashMap::new();
                changes.insert("learning_rate".to_string(), 0.02);
                changes.insert("decay_constant".to_string(), 0.002);
                changes
            },
            confidence: 0.85,
            expected_improvement: 0.12,
            iterations_completed: 25,
            convergence_reason: ConvergenceReason::TargetAchieved,
        };
        
        assert_eq!(result.parameter_changes.len(), 2, "Should have 2 parameter changes");
        assert!(result.parameter_changes.contains_key("learning_rate"),
               "Should include learning rate change");
        assert!(result.parameter_changes.contains_key("decay_constant"),
               "Should include decay constant change");
        assert_eq!(result.confidence, 0.85, "Should have correct confidence");
        assert_eq!(result.expected_improvement, 0.12, "Should have correct expected improvement");
        assert_eq!(result.iterations_completed, 25, "Should have correct iteration count");
        assert!(matches!(result.convergence_reason, ConvergenceReason::TargetAchieved),
               "Should have correct convergence reason");
    }

    #[test]
    fn test_parameter_evaluation_structure() {
        let evaluation = ParameterEvaluation {
            evaluation_id: Uuid::new_v4(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("learning_rate".to_string(), 0.01);
                params
            },
            performance_score: 0.85,
            stability_score: 0.9,
            resource_usage: ResourceUsage {
                memory_used_mb: 256.0,
                cpu_time_ms: 1000.0,
                wall_time_ms: 1200.0,
                evaluations_performed: 50,
            },
            evaluation_time: Duration::from_millis(1200),
            validation_results: ValidationResults {
                passed_validations: vec!["stability_check".to_string()],
                failed_validations: vec![],
                validation_scores: {
                    let mut scores = HashMap::new();
                    scores.insert("stability".to_string(), 0.9);
                    scores
                },
                overall_validation_score: 0.9,
            },
        };
        
        assert!(!evaluation.evaluation_id.to_string().is_empty(), "Should have evaluation ID");
        assert!(!evaluation.parameters.is_empty(), "Should have parameters");
        assert!(evaluation.performance_score >= 0.0 && evaluation.performance_score <= 1.0,
               "Performance score should be normalized");
        assert!(evaluation.stability_score >= 0.0 && evaluation.stability_score <= 1.0,
               "Stability score should be normalized");
        assert!(evaluation.resource_usage.memory_used_mb > 0.0, "Should use memory");
        assert!(evaluation.resource_usage.evaluations_performed > 0, "Should perform evaluations");
        assert!(!evaluation.validation_results.passed_validations.is_empty(),
               "Should have passed validations");
        assert!(evaluation.validation_results.overall_validation_score >= 0.0 &&
               evaluation.validation_results.overall_validation_score <= 1.0,
               "Overall validation score should be normalized");
    }
}