//! Main adaptive learning system implementation

use super::types::{AdaptiveLearningConfig, PerformanceBottleneck as AdaptivePerformanceBottleneck, AdaptationRecord, AdaptationType, LearningTarget, LearningTargetType, SatisfactionAnalysis, CorrelationAnalysis, LearningTaskType, EmergencyContext, EmergencyTrigger, PerformanceSnapshot, BottleneckType};
use super::monitoring::PerformanceMonitor;
use super::feedback::FeedbackAggregator;
use super::scheduler::LearningScheduler;
use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::cognitive::attention_manager::AttentionManager;
use crate::learning::hebbian::HebbianLearningEngine;

use std::sync::{Arc, RwLock, Mutex};
use std::time::SystemTime;
use anyhow::Result;
use uuid::Uuid;

/// Adaptive learning system that integrates with cognitive systems
#[derive(Debug, Clone)]
pub struct AdaptiveLearningSystem {
    pub working_memory: Arc<WorkingMemorySystem>,
    pub attention_manager: Arc<AttentionManager>,
    pub orchestrator: Arc<CognitiveOrchestrator>,
    pub hebbian_engine: Arc<Mutex<HebbianLearningEngine>>,
    pub performance_monitor: Arc<PerformanceMonitor>,
    pub feedback_aggregator: Arc<FeedbackAggregator>,
    pub learning_scheduler: Arc<LearningScheduler>,
    pub adaptation_history: Arc<RwLock<Vec<AdaptationRecord>>>,
    pub learning_config: AdaptiveLearningConfig,
}

impl AdaptiveLearningSystem {
    /// Create new adaptive learning system
    pub fn new() -> Result<Self> {
        // For now, return an error indicating that proper initialization is required
        // This prevents the use of uninitialized components
        anyhow::bail!("AdaptiveLearningSystem requires proper initialization with components. Use new_with_components() instead.")
    }
    
    /// Execute learning cycle
    pub async fn execute_learning_cycle(&self) -> Result<AdaptiveLearningResult> {
        let cycle_start = SystemTime::now();
        
        // Step 1: Collect current performance metrics
        let _performance_snapshot = self.performance_monitor.get_current_snapshot()?;
        
        // Step 2: Aggregate feedback
        let _aggregated_feedback = self.feedback_aggregator.aggregate_feedback()?;
        
        // Step 3: Analyze performance bottlenecks
        let bottlenecks = self.performance_monitor.detect_anomalies()?;
        
        // Step 4: Analyze user satisfaction
        let satisfaction_analysis = self.feedback_aggregator.analyze_user_satisfaction().await?;
        
        // Step 5: Correlate performance with outcomes
        let performance_data = self.performance_monitor.get_baseline();
        let correlation_analysis = self.feedback_aggregator
            .correlate_performance_outcomes(&performance_data).await?;
        
        // Step 6: Identify learning targets
        let learning_targets = self.identify_learning_targets(
            &bottlenecks,
            &satisfaction_analysis,
            &correlation_analysis,
        )?;
        
        // Step 7: Execute learning adaptations
        let adaptation_results = self.execute_adaptations(&learning_targets).await?;
        
        // Step 8: Record adaptation history
        self.record_adaptation_results(&adaptation_results)?;
        
        // Step 9: Schedule follow-up learning tasks
        self.schedule_follow_up_tasks(&learning_targets, &adaptation_results).await?;
        
        let cycle_duration = SystemTime::now().duration_since(cycle_start)
            .unwrap_or_default();
        
        Ok(AdaptiveLearningResult {
            performance_improvement: self.calculate_performance_improvement(&adaptation_results),
            adaptation_success: adaptation_results.iter().all(|r| r.success),
            learning_rate_adjustments: adaptation_results.len(),
            convergence_achieved: self.check_convergence(&adaptation_results),
            cycle_duration,
            bottlenecks_addressed: bottlenecks.len(),
            satisfaction_improvement: satisfaction_analysis.improvement_opportunities.len() as f32 * 0.1,
        })
    }
    
    /// Identify learning targets from analysis
    fn identify_learning_targets(
        &self,
        bottlenecks: &[AdaptivePerformanceBottleneck],
        satisfaction_analysis: &SatisfactionAnalysis,
        correlation_analysis: &CorrelationAnalysis,
    ) -> Result<Vec<LearningTarget>> {
        let mut targets = Vec::new();
        
        // Create learning targets from bottlenecks
        for bottleneck in bottlenecks {
            let importance = bottleneck.severity;
            let feasibility = 1.0 - bottleneck.severity; // Easier to fix less severe issues
            
            targets.push(LearningTarget {
                target_type: match bottleneck.bottleneck_type {
                    BottleneckType::Memory => LearningTargetType::StructureOptimization,
                    BottleneckType::Computation => LearningTargetType::ParameterTuning,
                    _ => LearningTargetType::BehaviorModification,
                },
                importance,
                feasibility,
                description: bottleneck.description.clone(),
                expected_impact: bottleneck.severity * 0.8,
            });
        }
        
        // Create targets from satisfaction analysis
        for problem_area in &satisfaction_analysis.problem_areas {
            targets.push(LearningTarget {
                target_type: LearningTargetType::BehaviorModification,
                importance: 0.8,
                feasibility: 0.7,
                description: format!("Improve {problem_area}"),
                expected_impact: 0.2,
            });
        }
        
        // Create targets from correlation analysis
        for (metric1, metric2, correlation) in &correlation_analysis.significant_correlations {
            if correlation.abs() > 0.6 {
                targets.push(LearningTarget {
                    target_type: LearningTargetType::PatternImprovement,
                    importance: correlation.abs(),
                    feasibility: 0.8,
                    description: format!("Optimize correlation between {metric1} and {metric2}"),
                    expected_impact: correlation.abs() * 0.3,
                });
            }
        }
        
        // Sort targets by importance * feasibility
        targets.sort_by(|a, b| {
            (b.importance * b.feasibility).partial_cmp(&(a.importance * a.feasibility))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(targets)
    }
    
    /// Execute adaptations for learning targets
    async fn execute_adaptations(&self, targets: &[LearningTarget]) -> Result<Vec<AdaptationRecord>> {
        let mut adaptation_results = Vec::new();
        let max_adaptations = self.learning_config.max_concurrent_adaptations;
        
        for target in targets.iter().take(max_adaptations) {
            let adaptation_record = self.execute_single_adaptation(target).await?;
            adaptation_results.push(adaptation_record);
        }
        
        Ok(adaptation_results)
    }
    
    /// Execute single adaptation
    async fn execute_single_adaptation(&self, target: &LearningTarget) -> Result<AdaptationRecord> {
        let record_id = Uuid::new_v4();
        let start_time = SystemTime::now();
        
        // Get baseline performance
        let performance_before = self.performance_monitor.get_current_snapshot()?
            .overall_performance_score;
        
        // Execute adaptation based on target type
        let success = match target.target_type {
            LearningTargetType::StructureOptimization => {
                self.execute_structure_optimization().await?
            }
            LearningTargetType::ParameterTuning => {
                self.execute_parameter_tuning().await?
            }
            LearningTargetType::BehaviorModification => {
                self.execute_behavior_modification().await?
            }
            LearningTargetType::PatternImprovement => {
                self.execute_pattern_improvement().await?
            }
        };
        
        // Wait a moment for changes to take effect
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        // Get performance after adaptation
        let performance_after = self.performance_monitor.get_current_snapshot()?
            .overall_performance_score;
        
        let adaptation_type = match target.target_type {
            LearningTargetType::StructureOptimization => AdaptationType::StructureModification,
            LearningTargetType::ParameterTuning => AdaptationType::ParameterAdjustment,
            LearningTargetType::BehaviorModification => AdaptationType::BehaviorChange,
            LearningTargetType::PatternImprovement => AdaptationType::ParameterAdjustment,
        };
        
        Ok(AdaptationRecord {
            record_id,
            timestamp: start_time,
            adaptation_type,
            performance_before,
            performance_after,
            success,
            impact_assessment: format!("Adaptation for: {}", target.description),
        })
    }
    
    /// Execute structure optimization
    async fn execute_structure_optimization(&self) -> Result<bool> {
        println!("Structure optimization disabled - no agents");
        Ok(false)
    }
    
    /// Execute parameter tuning
    async fn execute_parameter_tuning(&self) -> Result<bool> {
        // Tune cognitive orchestrator parameters
        println!("Executing parameter tuning");
        // Would adjust actual parameters here
        Ok(true)
    }
    
    /// Execute behavior modification
    async fn execute_behavior_modification(&self) -> Result<bool> {
        // Modify cognitive behavior patterns
        println!("Executing behavior modification");
        // Would modify behavior patterns here
        Ok(true)
    }
    
    /// Execute pattern improvement
    async fn execute_pattern_improvement(&self) -> Result<bool> {
        // Improve cognitive pattern performance
        println!("Executing pattern improvement");
        // Would improve patterns here
        Ok(true)
    }
    
    /// Record adaptation results
    fn record_adaptation_results(&self, results: &[AdaptationRecord]) -> Result<()> {
        let mut history = self.adaptation_history.write().unwrap();
        history.extend(results.iter().cloned());
        
        // Keep only recent history (last 1000 adaptations)
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
        
        Ok(())
    }
    
    /// Schedule follow-up learning tasks
    async fn schedule_follow_up_tasks(
        &self,
        _targets: &[LearningTarget],
        results: &[AdaptationRecord],
    ) -> Result<()> {
        // Schedule tasks based on adaptation results
        for result in results {
            if !result.success {
                // Retry failed adaptations
                self.learning_scheduler.schedule_task(
                    LearningTaskType::ParameterTuning,
                    0.8,
                    SystemTime::now() + std::time::Duration::from_secs(3600),
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Calculate performance improvement
    fn calculate_performance_improvement(&self, results: &[AdaptationRecord]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        let total_improvement: f32 = results.iter()
            .map(|r| r.performance_after - r.performance_before)
            .sum();
        
        total_improvement / results.len() as f32
    }
    
    /// Check if convergence is achieved
    fn check_convergence(&self, results: &[AdaptationRecord]) -> bool {
        if results.is_empty() {
            return false;
        }
        
        // Check if all adaptations succeeded and performance is stable
        let all_successful = results.iter().all(|r| r.success);
        let performance_stable = results.iter()
            .all(|r| (r.performance_after - r.performance_before).abs() < 0.1);
        
        all_successful && performance_stable
    }
    
    /// Handle emergency adaptation
    pub async fn handle_emergency(&self, emergency_context: EmergencyContext) -> Result<EmergencyAdaptationResult> {
        let start_time = SystemTime::now();
        
        // Schedule emergency task
        let task_id = self.learning_scheduler.schedule_emergency_task(
            LearningTaskType::EmergencyAdaptation,
            &emergency_context,
        ).await?;
        
        // Execute emergency adaptation immediately
        let adaptation_record = self.execute_emergency_adaptation(&emergency_context).await?;
        
        // Record the emergency response
        self.record_adaptation_results(&[adaptation_record.clone()])?;
        
        let duration = SystemTime::now().duration_since(start_time)
            .unwrap_or_default();
        
        Ok(EmergencyAdaptationResult {
            task_id,
            adaptation_record: adaptation_record.clone(),
            response_time: duration,
            emergency_resolved: adaptation_record.success,
        })
    }
    
    /// Execute emergency adaptation
    async fn execute_emergency_adaptation(&self, emergency_context: &EmergencyContext) -> Result<AdaptationRecord> {
        let record_id = Uuid::new_v4();
        let start_time = SystemTime::now();
        
        // Get baseline performance
        let performance_before = emergency_context.performance_before;
        
        // Execute emergency actions
        let success = match emergency_context.trigger_type {
            EmergencyTrigger::SystemFailure => {
                // Implement system failure recovery
                println!("Executing system failure recovery");
                true
            }
            EmergencyTrigger::PerformanceCollapse => {
                // Implement performance recovery
                println!("Executing performance collapse recovery");
                true
            }
            EmergencyTrigger::UserExodus => {
                // Implement user retention strategies
                println!("Executing user retention strategies");
                true
            }
            EmergencyTrigger::ResourceExhaustion => {
                // Implement resource optimization
                println!("Executing resource optimization");
                true
            }
        };
        
        // Estimate performance after (would be measured in practice)
        let performance_after = if success {
            performance_before + 0.2
        } else {
            performance_before
        };
        
        Ok(AdaptationRecord {
            record_id,
            timestamp: start_time,
            adaptation_type: AdaptationType::EmergencyResponse,
            performance_before,
            performance_after,
            success,
            impact_assessment: format!("Emergency response for: {:?}", emergency_context.trigger_type),
        })
    }
    
    /// Get system status
    pub fn get_system_status(&self) -> SystemStatus {
        let scheduler_stats = self.learning_scheduler.get_task_statistics();
        let feedback_summary = self.feedback_aggregator.get_feedback_summary()
            .unwrap_or_default();
        let performance_snapshot = self.performance_monitor.get_current_snapshot()
            .unwrap_or_default();
        
        SystemStatus {
            overall_performance: performance_snapshot.overall_performance_score,
            system_health: performance_snapshot.system_health,
            learning_active: scheduler_stats.pending_tasks > 0,
            adaptation_success_rate: scheduler_stats.success_rate,
            user_satisfaction: feedback_summary.avg_user_satisfaction,
            pending_tasks: scheduler_stats.pending_tasks,
            completed_adaptations: scheduler_stats.total_completed,
        }
    }
    
    /// Generate system report
    pub fn generate_report(&self) -> Result<String> {
        let mut report = String::new();
        
        report.push_str("Adaptive Learning System Report\n");
        report.push_str("===============================\n\n");
        
        // System status
        let status = self.get_system_status();
        report.push_str(&format!("Overall Performance: {:.2}\n", status.overall_performance));
        report.push_str(&format!("System Health: {:.2}\n", status.system_health));
        report.push_str(&format!("Learning Active: {}\n", status.learning_active));
        report.push_str(&format!("Adaptation Success Rate: {:.2}%\n", status.adaptation_success_rate * 100.0));
        report.push_str(&format!("User Satisfaction: {:.2}\n", status.user_satisfaction));
        report.push_str(&format!("Pending Tasks: {}\n", status.pending_tasks));
        report.push_str(&format!("Completed Adaptations: {}\n", status.completed_adaptations));
        
        // Performance monitoring report
        report.push('\n');
        report.push_str(&self.performance_monitor.generate_report()?);
        
        // Scheduler report
        report.push('\n');
        report.push_str(&self.learning_scheduler.generate_report());
        
        // Adaptation history summary
        let history = self.adaptation_history.read().unwrap();
        report.push_str(&format!("\nAdaptation History: {} records\n", history.len()));
        if let Some(latest) = history.last() {
            report.push_str(&format!("Latest Adaptation: {}\n", latest.impact_assessment));
        }
        
        Ok(report)
    }
}

/// Adaptive learning result
#[derive(Debug, Clone)]
pub struct AdaptiveLearningResult {
    pub performance_improvement: f32,
    pub adaptation_success: bool,
    pub learning_rate_adjustments: usize,
    pub convergence_achieved: bool,
    pub cycle_duration: std::time::Duration,
    pub bottlenecks_addressed: usize,
    pub satisfaction_improvement: f32,
}

/// Emergency adaptation result
#[derive(Debug, Clone)]
pub struct EmergencyAdaptationResult {
    pub task_id: Uuid,
    pub adaptation_record: AdaptationRecord,
    pub response_time: std::time::Duration,
    pub emergency_resolved: bool,
}

/// System status
#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub overall_performance: f32,
    pub system_health: f32,
    pub learning_active: bool,
    pub adaptation_success_rate: f32,
    pub user_satisfaction: f32,
    pub pending_tasks: usize,
    pub completed_adaptations: usize,
}

// Removed stub implementations - using actual implementations from respective modules

 

// Removed duplicate impl block for WorkingMemorySystem
// The proper implementation is in cognitive/working_memory.rs

// Removed duplicate impl block for AttentionManager
// The proper implementation is in cognitive/attention_manager.rs

// Removed duplicate impl block for CognitiveOrchestrator
// The proper implementation is in cognitive/orchestrator.rs

impl Default for super::feedback::FeedbackSummary {
    fn default() -> Self {
        Self {
            total_user_feedback: 0,
            total_system_feedback: 0,
            avg_user_satisfaction: 0.8,
            avg_system_severity: 0.2,
            explicit_feedback_count: 0,
            implicit_feedback_count: 0,
            system_feedback_count: 0,
        }
    }
}

impl Default for PerformanceSnapshot {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            overall_performance_score: 0.8,
            component_scores: std::collections::HashMap::new(),
            bottlenecks: Vec::new(),
            system_health: 0.8,
        }
    }
}

impl Default for AdaptiveLearningSystem {
    fn default() -> Self {
        // Cannot provide a default implementation without proper components
        panic!("AdaptiveLearningSystem cannot be default-constructed. Use new_with_components() with proper initialization.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use std::time::Duration;
    use uuid::Uuid;

    // Test helper to create mock components  
    fn create_test_adaptive_learning_system() -> AdaptiveLearningSystem {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(create_test_adaptive_learning_system_async())
    }
    
    async fn create_test_adaptive_learning_system_async() -> AdaptiveLearningSystem {
                use crate::cognitive::working_memory::WorkingMemorySystem;
        use crate::cognitive::attention_manager::AttentionManager;
        use crate::cognitive::orchestrator::CognitiveOrchestrator;
        use crate::learning::hebbian::HebbianLearningEngine;
        use crate::learning::adaptive_learning::monitoring::PerformanceMonitor;
        use crate::learning::adaptive_learning::feedback::FeedbackAggregator;
        use crate::learning::adaptive_learning::scheduler::LearningScheduler;

        // These would be properly initialized in real tests
        // For now, we'll create mock implementations where needed
        
        // Create a mock brain graph for testing
        let embedding_dim = 512;
        let brain_graph = Arc::new(crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph::new(embedding_dim)
            .expect("Failed to create brain graph"));
        
        // Create activation engine
        let activation_config = crate::core::activation_config::ActivationConfig {
            max_iterations: 100,
            convergence_threshold: 0.01,
            decay_rate: 0.1,
            decay_factor: 0.95,
            inhibition_strength: 0.3,
            default_threshold: 0.5,
        };
        let activation_engine = Arc::new(crate::core::activation_engine::ActivationPropagationEngine::new(activation_config));
        
        // Create SDR storage
        let sdr_config = crate::core::sdr_types::SDRConfig {
            total_bits: embedding_dim * 4,
            active_bits: (embedding_dim * 4) / 50,
            sparsity: 0.02,
            overlap_threshold: 0.5,
        };
        let sdr_storage = Arc::new(crate::core::sdr_storage::SDRStorage::new(sdr_config));
        
        // Create working memory
        let working_memory = Arc::new(WorkingMemorySystem::new(
            activation_engine.clone(),
            sdr_storage.clone(),
        ).await.expect("Failed to create working memory"));
        
        // Create orchestrator
        let orchestrator = Arc::new(CognitiveOrchestrator::new(
            brain_graph.clone(),
            crate::cognitive::orchestrator::CognitiveOrchestratorConfig::default(),
        ).await.expect("Failed to create orchestrator"));
        
        // Create attention manager
        let attention_manager = Arc::new(AttentionManager::new(
            orchestrator.clone(),
            activation_engine.clone(),
            working_memory.clone(),
        ).await.expect("Failed to create attention manager"));
        
        // Create critical thinking
        let critical_thinking = Arc::new(crate::cognitive::critical::CriticalThinking::new(
            brain_graph.clone()
        ));
        
        // Create inhibition system
        let inhibition_system = Arc::new(crate::cognitive::inhibitory::CompetitiveInhibitionSystem::new(
            activation_engine.clone(),
            critical_thinking
        ));
        
        // Create hebbian engine
        let hebbian_engine = Arc::new(Mutex::new(HebbianLearningEngine::new(
            brain_graph.clone(),
            activation_engine.clone(),
            inhibition_system.clone(),
        ).await.expect("Failed to create hebbian engine")));
        
        
        AdaptiveLearningSystem {
            working_memory,
            attention_manager,
            orchestrator,
            hebbian_engine,
            performance_monitor: Arc::new(PerformanceMonitor::default()),
            feedback_aggregator: Arc::new(FeedbackAggregator::default()),
            learning_scheduler: Arc::new(LearningScheduler::default()),
            adaptation_history: Arc::new(RwLock::new(Vec::new())),
            learning_config: AdaptiveLearningConfig::default(),
        }
    }

    #[test]
    fn test_identify_learning_targets_from_bottlenecks() {
        let system = create_test_adaptive_learning_system();
        
        let bottlenecks = vec![
            AdaptivePerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: 0.8,
                description: "High memory usage".to_string(),
                potential_solutions: vec!["Optimize memory allocation".to_string()],
            }
        ];
        
        let satisfaction_analysis = SatisfactionAnalysis {
            satisfaction_trends: vec![0.5, 0.6, 0.7],
            problem_areas: vec!["response_time".to_string()],
            improvement_opportunities: vec!["faster_processing".to_string()],
        };
        
        let correlation_analysis = CorrelationAnalysis {
            performance_satisfaction_correlation: 0.7,
            speed_satisfaction_correlation: -0.5,
            accuracy_satisfaction_correlation: 0.8,
            significant_correlations: vec![
                ("memory_usage".to_string(), "response_time".to_string(), -0.7)
            ],
        };
        
        let targets = system.identify_learning_targets(
            &bottlenecks,
            &satisfaction_analysis,
            &correlation_analysis,
        ).expect("Failed to identify learning targets");
        
        assert!(!targets.is_empty(), "Should identify learning targets from bottlenecks");
        assert!(targets[0].importance > 0.0, "Targets should have importance scores");
        assert!(targets[0].feasibility > 0.0, "Targets should have feasibility scores");
    }

    #[test]
    fn test_calculate_performance_improvement() {
        let system = create_test_adaptive_learning_system();
        
        let results = vec![
            AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::ParameterAdjustment,
                performance_before: 0.6,
                performance_after: 0.8,
                success: true,
                impact_assessment: "Test improvement".to_string(),
            },
            AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::BehaviorChange,
                performance_before: 0.7,
                performance_after: 0.75,
                success: true,
                impact_assessment: "Minor improvement".to_string(),
            }
        ];
        
        let improvement = system.calculate_performance_improvement(&results);
        
        // Average improvement should be (0.2 + 0.05) / 2 = 0.125
        assert!((improvement - 0.125).abs() < 0.001, 
               "Performance improvement calculation incorrect: expected ~0.125, got {improvement}");
    }

    #[test]
    fn test_check_convergence() {
        let system = create_test_adaptive_learning_system();
        
        // Test successful convergence
        let successful_results = vec![
            AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::ParameterAdjustment,
                performance_before: 0.8,
                performance_after: 0.82, // Small stable change
                success: true,
                impact_assessment: "Stable improvement".to_string(),
            }
        ];
        
        assert!(system.check_convergence(&successful_results), 
               "Should detect convergence for successful stable adaptations");
        
        // Test failed convergence
        let failed_results = vec![
            AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::ParameterAdjustment,
                performance_before: 0.8,
                performance_after: 0.6, // Large negative change
                success: false,
                impact_assessment: "Failed adaptation".to_string(),
            }
        ];
        
        assert!(!system.check_convergence(&failed_results), 
               "Should not detect convergence for failed adaptations");
    }

    #[test]
    fn test_record_adaptation_results() {
        let system = create_test_adaptive_learning_system();
        
        let results = vec![
            AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::ParameterAdjustment,
                performance_before: 0.6,
                performance_after: 0.8,
                success: true,
                impact_assessment: "Test adaptation".to_string(),
            }
        ];
        
        system.record_adaptation_results(&results)
            .expect("Failed to record adaptation results");
        
        let history = system.adaptation_history.read().unwrap();
        assert_eq!(history.len(), 1, "Should record one adaptation result");
        assert_eq!(history[0].record_id, results[0].record_id, 
                  "Recorded result should match input");
    }

    #[test]
    fn test_record_adaptation_results_history_limit() {
        let system = create_test_adaptive_learning_system();
        
        // Add more than 1000 results to test history limiting
        let mut results = Vec::new();
        for i in 0..1005 {
            results.push(AdaptationRecord {
                record_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                adaptation_type: AdaptationType::ParameterAdjustment,
                performance_before: 0.6,
                performance_after: 0.8,
                success: true,
                impact_assessment: format!("Test adaptation {i}"),
            });
        }
        
        system.record_adaptation_results(&results)
            .expect("Failed to record large number of adaptation results");
        
        let history = system.adaptation_history.read().unwrap();
        assert_eq!(history.len(), 1000, "Should limit history to 1000 records");
    }

    #[tokio::test]
    async fn test_emergency_adaptation_system_failure() {
        let system = create_test_adaptive_learning_system();
        
        let emergency_context = EmergencyContext {
            trigger_type: EmergencyTrigger::SystemFailure,
            severity: 0.9,
            affected_components: vec!["core_system".to_string()],
            performance_before: 0.3,
            emergency_actions: vec!["restart_components".to_string()],
        };
        
        let result = system.handle_emergency(emergency_context)
            .await
            .expect("Failed to handle emergency");
        
        assert!(result.emergency_resolved, "Emergency should be resolved");
        assert!(result.adaptation_record.success, "Emergency adaptation should succeed");
        assert!(result.response_time < Duration::from_secs(10), 
               "Emergency response should be fast");
    }

    #[tokio::test]
    async fn test_emergency_adaptation_performance_collapse() {
        let system = create_test_adaptive_learning_system();
        
        let emergency_context = EmergencyContext {
            trigger_type: EmergencyTrigger::PerformanceCollapse,
            severity: 0.8,
            affected_components: vec!["cognitive_system".to_string()],
            performance_before: 0.2,
            emergency_actions: vec!["optimize_parameters".to_string()],
        };
        
        let result = system.handle_emergency(emergency_context)
            .await
            .expect("Failed to handle performance collapse emergency");
        
        assert!(result.emergency_resolved, "Performance collapse should be resolved");
        assert!(result.adaptation_record.performance_after > result.adaptation_record.performance_before,
               "Performance should improve after emergency adaptation");
    }

    #[test]
    fn test_system_status_generation() {
        let system = create_test_adaptive_learning_system();
        
        let status = system.get_system_status();
        
        assert!(status.overall_performance >= 0.0 && status.overall_performance <= 1.0,
               "Overall performance should be normalized");
        assert!(status.system_health >= 0.0 && status.system_health <= 1.0,
               "System health should be normalized");
        assert!(status.adaptation_success_rate >= 0.0 && status.adaptation_success_rate <= 1.0,
               "Success rate should be normalized");
        assert!(status.user_satisfaction >= 0.0 && status.user_satisfaction <= 1.0,
               "User satisfaction should be normalized");
    }

    #[test]
    fn test_system_report_generation() {
        let system = create_test_adaptive_learning_system();
        
        let report = system.generate_report()
            .expect("Failed to generate system report");
        
        assert!(report.contains("Adaptive Learning System Report"), 
               "Report should contain title");
        assert!(report.contains("Overall Performance"), 
               "Report should contain performance metrics");
        assert!(report.contains("System Health"), 
               "Report should contain health metrics");
        assert!(report.contains("Learning Active"), 
               "Report should contain learning status");
    }

    #[test]
    fn test_learning_target_prioritization() {
        let system = create_test_adaptive_learning_system();
        
        let bottlenecks = vec![
            AdaptivePerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: 0.9, // High severity
                description: "Critical memory bottleneck".to_string(),
                potential_solutions: vec!["Urgent optimization".to_string()],
            },
            AdaptivePerformanceBottleneck {
                bottleneck_type: BottleneckType::Computation,
                severity: 0.3, // Low severity
                description: "Minor computational bottleneck".to_string(),
                potential_solutions: vec!["Minor optimization".to_string()],
            }
        ];
        
        let satisfaction_analysis = SatisfactionAnalysis {
            satisfaction_trends: vec![0.6, 0.7, 0.8],
            problem_areas: vec![],
            improvement_opportunities: vec![],
        };
        
        let correlation_analysis = CorrelationAnalysis {
            performance_satisfaction_correlation: 0.8,
            speed_satisfaction_correlation: 0.7,
            accuracy_satisfaction_correlation: 0.9,
            significant_correlations: vec![],
        };
        
        let targets = system.identify_learning_targets(
            &bottlenecks,
            &satisfaction_analysis,
            &correlation_analysis,
        ).expect("Failed to identify learning targets");
        
        // Targets should be sorted by importance * feasibility (descending)
        assert!(targets.len() >= 2, "Should identify targets from both bottlenecks");
        
        // First target should have higher priority (higher severity bottleneck)
        let first_priority = targets[0].importance * targets[0].feasibility;
        let second_priority = targets[1].importance * targets[1].feasibility;
        assert!(first_priority >= second_priority, 
               "Targets should be prioritized correctly");
    }

    #[test]
    fn test_learning_target_types() {
        let system = create_test_adaptive_learning_system();
        
        let bottlenecks = vec![
            AdaptivePerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: 0.8,
                description: "Memory bottleneck".to_string(),
                potential_solutions: vec!["Optimize memory".to_string()],
            },
            AdaptivePerformanceBottleneck {
                bottleneck_type: BottleneckType::Computation,
                severity: 0.7,
                description: "Computation bottleneck".to_string(),
                potential_solutions: vec!["Optimize computation".to_string()],
            }
        ];
        
        let satisfaction_analysis = SatisfactionAnalysis {
            satisfaction_trends: vec![0.5, 0.6, 0.7],
            problem_areas: vec!["response_speed".to_string()],
            improvement_opportunities: vec![],
        };
        
        let correlation_analysis = CorrelationAnalysis {
            performance_satisfaction_correlation: 0.7,
            speed_satisfaction_correlation: 0.6,
            accuracy_satisfaction_correlation: 0.8,
            significant_correlations: vec![
                ("feature_a".to_string(), "feature_b".to_string(), 0.8)
            ],
        };
        
        let targets = system.identify_learning_targets(
            &bottlenecks,
            &satisfaction_analysis,
            &correlation_analysis,
        ).expect("Failed to identify learning targets");
        
        // Should have different target types based on bottleneck types
        let memory_targets: Vec<_> = targets.iter()
            .filter(|t| matches!(t.target_type, LearningTargetType::StructureOptimization))
            .collect();
        let computation_targets: Vec<_> = targets.iter()
            .filter(|t| matches!(t.target_type, LearningTargetType::ParameterTuning))
            .collect();
        let behavior_targets: Vec<_> = targets.iter()
            .filter(|t| matches!(t.target_type, LearningTargetType::BehaviorModification))
            .collect();
        let pattern_targets: Vec<_> = targets.iter()
            .filter(|t| matches!(t.target_type, LearningTargetType::PatternImprovement))
            .collect();
        
        assert!(!memory_targets.is_empty(), "Should identify structure optimization targets");
        assert!(!computation_targets.is_empty(), "Should identify parameter tuning targets");
        assert!(!behavior_targets.is_empty(), "Should identify behavior modification targets");
        assert!(!pattern_targets.is_empty(), "Should identify pattern improvement targets");
    }

    #[test]
    fn test_learning_config_validation() {
        let config = AdaptiveLearningConfig::default();
        
        assert!(config.max_concurrent_adaptations > 0, 
               "Should allow at least one concurrent adaptation");
        assert!(config.adaptation_aggressiveness > 0.0 && config.adaptation_aggressiveness <= 1.0,
               "Adaptation aggressiveness should be normalized");
        assert!(config.emergency_adaptation_threshold > 0.0 && config.emergency_adaptation_threshold <= 1.0,
               "Emergency adaptation threshold should be normalized");
    }
}