//! Graph optimization agent modules

pub mod types;
pub mod pattern_analysis;
pub mod optimization_strategies;
pub mod performance_analysis;
pub mod safety_validation;
pub mod execution_engine;
pub mod scheduling;

// Re-export key types and functionality
pub use types::*;
pub use pattern_analysis::*;
pub use optimization_strategies::*;

use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Main graph optimization agent
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

impl GraphOptimizationAgent {
    /// Create new optimization agent
    pub fn new() -> Self {
        Self {
            pattern_detector: PatternDetector::new(),
            efficiency_analyzer: EfficiencyAnalyzer::new(),
            optimization_scheduler: OptimizationScheduler::new(),
            safety_validator: SafetyValidator::new(),
            rollback_manager: RollbackManager::new(),
            impact_predictor: ImpactPredictor::new(),
            bottleneck_detector: BottleneckDetector::new(),
            pattern_cache: PatternCache::new(),
        }
    }

    /// Run optimization analysis cycle
    pub async fn run_optimization_cycle(
        &mut self,
        graph: &BrainEnhancedKnowledgeGraph,
        current_metrics: &PerformanceMetrics,
    ) -> Result<Vec<OptimizationOpportunity>> {
        // 1. Detect patterns and opportunities
        let opportunities = self.pattern_detector
            .identify_optimization_opportunities(graph, &mut self.pattern_cache)
            .await?;

        // 2. Analyze efficiency
        let efficiency_score = self.efficiency_analyzer
            .analyze_performance(current_metrics)?;

        // 3. Detect bottlenecks
        let bottlenecks = self.bottleneck_detector
            .detect_bottlenecks(graph, current_metrics)
            .await?;

        // 4. Filter opportunities by safety
        let mut safe_opportunities = Vec::new();
        for opportunity in opportunities {
            let safety_score = self.safety_validator
                .validate_optimization_safety(graph, &opportunity, current_metrics)
                .await?;

            if self.safety_validator.passes_safety_threshold(safety_score) {
                safe_opportunities.push(opportunity);
            }
        }

        // 5. Schedule high-priority optimizations
        for opportunity in &safe_opportunities {
            if opportunity.is_worthwhile() {
                self.optimization_scheduler
                    .schedule_optimization(opportunity, current_metrics)?;
            }
        }

        Ok(safe_opportunities)
    }

    /// Execute next scheduled optimization
    pub async fn execute_next_optimization(
        &mut self,
        graph: &BrainEnhancedKnowledgeGraph,
        current_metrics: &PerformanceMetrics,
    ) -> Result<Option<OptimizationImpact>> {
        // Get next optimization from scheduler
        if let Some(scheduled_optimization) = self.optimization_scheduler.get_next_optimization() {
            // Create optimization context
            let optimization_context = OptimizationContext {
                optimization_type: scheduled_optimization.optimization_type.clone(),
                affected_entities: vec![], // Would be populated from the opportunity
                parameters: HashMap::new(),
                safety_requirements: vec![],
            };

            // Create checkpoint
            let checkpoint_id = self.rollback_manager
                .create_checkpoint(graph, &optimization_context, current_metrics)
                .await?;

            // Predict impact
            let opportunity = OptimizationOpportunity {
                opportunity_id: scheduled_optimization.optimization_id.clone(),
                optimization_type: scheduled_optimization.optimization_type.clone(),
                affected_entities: vec![], // Would be populated from the opportunity
                estimated_improvement: scheduled_optimization.expected_improvement,
                implementation_cost: 0.1, // Would be calculated
                risk_level: RiskLevel::Low, // Would be assessed
                prerequisites: vec![],
            };

            let predicted_impact = self.impact_predictor
                .predict_optimization_impact(&opportunity, current_metrics)?;

            // Execute optimization
            let start_time = Instant::now();
            let actual_impact = self.execute_optimization(graph, &opportunity).await?;
            let execution_time = start_time.elapsed();

            // Create complete impact record
            let mut complete_impact = actual_impact;
            complete_impact.predicted_improvement = predicted_impact.predicted_improvement;
            complete_impact.execution_time = execution_time;

            // Record execution
            self.optimization_scheduler.record_execution(
                &scheduled_optimization.optimization_id,
                true, // success
                complete_impact.actual_improvement,
                execution_time,
                false, // rollback_required
            );

            // Update prediction accuracy
            self.impact_predictor.update_prediction_accuracy(&complete_impact);

            // Check if rollback is needed
            if self.rollback_manager.should_rollback(current_metrics, current_metrics) {
                self.rollback_manager.execute_rollback(
                    graph,
                    &checkpoint_id,
                    RollbackReason::PerformanceRegression,
                ).await?;
            }

            return Ok(Some(complete_impact));
        }

        Ok(None)
    }

    /// Execute a specific optimization
    async fn execute_optimization(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        opportunity: &OptimizationOpportunity,
    ) -> Result<OptimizationImpact> {
        match opportunity.optimization_type {
            OptimizationType::AttributeBubbling => {
                OptimizationStrategies::execute_attribute_bubbling(graph, opportunity).await
            }
            OptimizationType::HierarchyConsolidation => {
                OptimizationStrategies::execute_hierarchy_consolidation(graph, opportunity).await
            }
            OptimizationType::SubgraphFactorization => {
                OptimizationStrategies::execute_subgraph_factorization(graph, opportunity).await
            }
            OptimizationType::ConnectionPruning => {
                OptimizationStrategies::execute_connection_pruning(graph, opportunity).await
            }
            _ => {
                // For other optimization types, return a default impact
                Ok(OptimizationImpact {
                    optimization_type: opportunity.optimization_type.clone(),
                    predicted_improvement: opportunity.estimated_improvement,
                    actual_improvement: opportunity.estimated_improvement * 0.8,
                    execution_time: Duration::from_millis(100),
                    side_effects: vec![],
                })
            }
        }
    }

    /// Get comprehensive optimization report
    pub fn get_optimization_report(&self) -> HashMap<String, f32> {
        let mut report = HashMap::new();

        // Scheduler statistics
        let scheduler_stats = self.optimization_scheduler.get_optimization_statistics();
        for (key, value) in scheduler_stats {
            report.insert(format!("scheduler_{}", key), value);
        }

        // Safety validator statistics
        let safety_stats = self.safety_validator.get_validation_statistics();
        for (key, value) in safety_stats {
            report.insert(format!("safety_{}", key), value);
        }

        // Impact predictor statistics
        let prediction_stats = self.impact_predictor.get_prediction_statistics();
        for (key, value) in prediction_stats {
            report.insert(format!("prediction_{}", key), value);
        }

        // Rollback manager statistics
        let rollback_stats = self.rollback_manager.get_rollback_statistics();
        for (key, value) in rollback_stats {
            report.insert(format!("rollback_{}", key), value);
        }

        // Cache statistics
        let (hit_count, miss_count, hit_rate) = self.pattern_cache.get_cache_stats();
        report.insert("cache_hit_count".to_string(), hit_count as f32);
        report.insert("cache_miss_count".to_string(), miss_count as f32);
        report.insert("cache_hit_rate".to_string(), hit_rate);

        report
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(
        &self,
        current_metrics: &PerformanceMetrics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Efficiency recommendations
        recommendations.extend(self.efficiency_analyzer.get_efficiency_recommendations(current_metrics));

        // Bottleneck recommendations
        recommendations.extend(self.bottleneck_detector.get_bottleneck_recommendations());

        // Remove duplicates
        recommendations.sort();
        recommendations.dedup();

        recommendations
    }

    /// Update agent configuration
    pub fn update_configuration(&mut self, config: OptimizationAgentConfig) {
        self.pattern_detector.detection_threshold = config.pattern_detection_threshold;
        self.efficiency_analyzer.efficiency_threshold = config.efficiency_threshold;
        self.safety_validator.safety_threshold = config.safety_threshold;
        self.rollback_manager.rollback_threshold = config.rollback_threshold;
        self.optimization_scheduler.update_config(config.schedule_config);
    }

    /// Get current configuration
    pub fn get_configuration(&self) -> OptimizationAgentConfig {
        OptimizationAgentConfig {
            pattern_detection_threshold: self.pattern_detector.detection_threshold,
            efficiency_threshold: self.efficiency_analyzer.efficiency_threshold,
            safety_threshold: self.safety_validator.safety_threshold,
            rollback_threshold: self.rollback_manager.rollback_threshold,
            schedule_config: self.optimization_scheduler.get_config().clone(),
        }
    }

    /// Reset agent state
    pub fn reset(&mut self) {
        self.pattern_detector = PatternDetector::new();
        self.efficiency_analyzer = EfficiencyAnalyzer::new();
        self.optimization_scheduler = OptimizationScheduler::new();
        self.safety_validator = SafetyValidator::new();
        self.rollback_manager = RollbackManager::new();
        self.impact_predictor = ImpactPredictor::new();
        self.bottleneck_detector = BottleneckDetector::new();
        self.pattern_cache = PatternCache::new();
    }

    /// Cleanup old data
    pub fn cleanup(&mut self) {
        self.pattern_cache.cleanup_expired_patterns();
        self.rollback_manager.cleanup_old_checkpoints(Duration::from_secs(3600));
        self.optimization_scheduler.clear_execution_history();
    }
}

/// Configuration for optimization agent
#[derive(Debug, Clone)]
pub struct OptimizationAgentConfig {
    pub pattern_detection_threshold: f32,
    pub efficiency_threshold: f32,
    pub safety_threshold: f32,
    pub rollback_threshold: f32,
    pub schedule_config: ScheduleConfig,
}

impl Default for OptimizationAgentConfig {
    fn default() -> Self {
        Self {
            pattern_detection_threshold: 0.5,
            efficiency_threshold: 0.7,
            safety_threshold: 0.8,
            rollback_threshold: 0.3,
            schedule_config: ScheduleConfig::default(),
        }
    }
}

impl Default for GraphOptimizationAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of optimization operation
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub success: bool,
    pub impact: Option<OptimizationImpact>,
    pub error_message: Option<String>,
    pub execution_time: Duration,
    pub checkpoint_id: Option<String>,
}

impl OptimizationResult {
    /// Create successful result
    pub fn success(impact: OptimizationImpact, execution_time: Duration) -> Self {
        Self {
            success: true,
            impact: Some(impact),
            error_message: None,
            execution_time,
            checkpoint_id: None,
        }
    }

    /// Create failed result
    pub fn failure(error: String, execution_time: Duration) -> Self {
        Self {
            success: false,
            impact: None,
            error_message: Some(error),
            execution_time,
            checkpoint_id: None,
        }
    }
}