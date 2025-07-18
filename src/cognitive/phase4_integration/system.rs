//! Main Phase 4 Cognitive System implementation

use super::types::*;
use super::orchestrator::LearningEnhancedOrchestrator;
use super::interface::CognitiveLearningInterface;
use super::performance::CognitivePerformanceTracker;
use super::adaptation::AdaptationEngine;
use crate::cognitive::phase3_integration::Phase3IntegratedCognitiveSystem;
use crate::cognitive::types::CognitivePatternType;
use crate::learning::phase4_integration::Phase4LearningSystem;
use crate::learning::types::{
    ThroughputMetrics as LearningThroughputMetrics
};

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::SystemTime;
use anyhow::Result;

/// Phase 4 Cognitive System Integration
/// 
/// This module provides the interface between the cognitive reasoning capabilities
/// and the Phase 4 learning systems. It enables cognitive patterns to benefit from
/// continuous learning while maintaining backward compatibility with Phase 3.
#[derive(Debug, Clone)]
pub struct Phase4CognitiveSystem {
    pub phase3_system: Arc<Phase3IntegratedCognitiveSystem>,
    pub phase4_learning: Arc<Phase4LearningSystem>,
    pub learning_enhanced_orchestrator: Arc<LearningEnhancedOrchestrator>,
    pub cognitive_learning_interface: Arc<CognitiveLearningInterface>,
    pub performance_tracker: Arc<RwLock<CognitivePerformanceTracker>>,
    pub adaptation_engine: Arc<AdaptationEngine>,
    pub configuration: Phase4CognitiveConfig,
}

impl Phase4CognitiveSystem {
    /// Create new Phase 4 cognitive system
    pub fn new(
        phase3_system: Arc<Phase3IntegratedCognitiveSystem>,
        phase4_learning: Arc<Phase4LearningSystem>,
        configuration: Option<Phase4CognitiveConfig>,
    ) -> Result<Self> {
        let config = configuration.unwrap_or_default();
        
        // Create learning-enhanced orchestrator
        let base_orchestrator = phase3_system.get_base_orchestrator()?;
        let learning_enhanced_orchestrator = Arc::new(
            LearningEnhancedOrchestrator::new(base_orchestrator)
        );
        
        // Create cognitive-learning interface
        let cognitive_learning_interface = Arc::new(
            CognitiveLearningInterface::new(phase4_learning.clone())
        );
        
        // Create performance tracker
        let performance_tracker = Arc::new(RwLock::new(CognitivePerformanceTracker::new()));
        
        // Create adaptation engine
        let adaptation_engine = Arc::new(
            futures::executor::block_on(AdaptationEngine::new())?
        );
        
        Ok(Self {
            phase3_system,
            phase4_learning,
            learning_enhanced_orchestrator,
            cognitive_learning_interface,
            performance_tracker,
            adaptation_engine,
            configuration: config,
        })
    }
    
    /// Execute enhanced cognitive query with learning integration
    pub async fn execute_enhanced_query(&self, query: &str, context: &str) -> Result<EnhancedCognitiveResult> {
        // Step 1: Get learning-informed pattern weights
        let pattern_weights = self.learning_enhanced_orchestrator
            .get_learning_informed_weights(context)?;
        
        // Step 2: Check for applicable learned shortcuts
        let shortcuts = self.learning_enhanced_orchestrator
            .get_applicable_shortcuts(context)?;
        
        // Step 3: Execute Phase 3 query with enhanced orchestration
        let phase3_result = self.phase3_system.execute_query(query, context).await?;
        
        // Step 4: Apply learning insights to improve result
        let learning_insights = self.cognitive_learning_interface
            .get_learning_insights_for_patterns().await?;
        
        // Step 5: Get recommended ensemble composition
        let ensemble_composition = self.learning_enhanced_orchestrator
            .get_recommended_ensemble(context)?;
        
        // Step 6: Apply optimizations from learning system
        let optimizations = self.cognitive_learning_interface
            .apply_learning_optimizations().await?;
        
        // Step 7: Record performance and learn from execution
        let performance_data = self.create_performance_data(&phase3_result, &pattern_weights)?;
        self.record_and_learn_from_execution(&performance_data, context).await?;
        
        Ok(EnhancedCognitiveResult {
            base_result: phase3_result,
            pattern_weights,
            shortcuts_applied: shortcuts,
            learning_insights,
            ensemble_composition,
            optimizations_applied: optimizations,
            performance_data,
        })
    }
    
    /// Integrate learning feedback into cognitive system
    pub async fn integrate_learning_feedback(&self, feedback: LearningFeedback) -> Result<()> {
        // Process cognitive feedback
        // Convert cognitive PerformanceData to LearningPerformanceData
        let learning_perf_data = crate::learning::types::PerformanceData {
            query_latencies: feedback.performance_data.query_latencies.clone(),
            memory_usage: feedback.performance_data.memory_usage.clone(),
            accuracy_scores: feedback.performance_data.accuracy_scores.clone(),
            user_satisfaction: feedback.performance_data.user_satisfaction.clone(),
            system_stability: feedback.performance_data.system_stability,
            error_rates: feedback.performance_data.error_rates.clone(),
            throughput_metrics: LearningThroughputMetrics {
                queries_per_second: feedback.performance_data.throughput_metrics.queries_per_second,
                successful_queries: feedback.performance_data.throughput_metrics.successful_queries,
                failed_queries: feedback.performance_data.throughput_metrics.failed_queries,
                average_response_time: feedback.performance_data.throughput_metrics.average_response_time,
            },
            timestamp: feedback.performance_data.timestamp,
            system_health: feedback.performance_data.system_health,
            overall_performance_score: feedback.performance_data.overall_performance_score,
            component_scores: feedback.performance_data.component_scores.clone(),
            bottlenecks: feedback.performance_data.bottlenecks.clone(),
        };
        
        self.cognitive_learning_interface
            .process_cognitive_feedback(
                feedback.pattern_type,
                &learning_perf_data,
                &feedback.context,
            ).await?;
        
        // Update learning insights in orchestrator
        self.learning_enhanced_orchestrator
            .update_learning_insights(feedback.learning_insights)?;
        
        // Update adaptive strategies
        self.learning_enhanced_orchestrator
            .update_adaptive_strategies(&feedback.performance_data)?;
        
        // Record performance data
        self.performance_tracker.write().unwrap()
            .record_performance(feedback.performance_data);
        
        Ok(())
    }
    
    /// Trigger adaptation based on performance
    pub async fn trigger_adaptation(&self, performance_data: &super::types::PerformanceData) -> Result<Vec<super::adaptation::AdaptationEvent>> {
        // Check for adaptation triggers
        let conditions = self.adaptation_engine.check_triggers(performance_data)?;
        
        if !conditions.is_empty() {
            // Execute adaptations
            let events = self.adaptation_engine.execute_adaptation(conditions).await?;
            
            // Monitor adaptation effectiveness
            for event in &events {
                if !event.success {
                    // Rollback if adaptation failed
                    self.adaptation_engine.rollback_if_needed(event).await?;
                }
            }
            
            Ok(events)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get current learning benefits assessment
    pub async fn assess_learning_benefits(&self) -> Result<LearningBenefitAssessment> {
        let mut tracker = self.performance_tracker.write().unwrap();
        let analysis = tracker.analyze_learning_impact()?;
        
        let baseline_satisfaction = tracker.baseline_performance.overall_satisfaction;
        let current_satisfaction = tracker.current_performance.overall_satisfaction;
        let satisfaction_improvement = current_satisfaction - baseline_satisfaction;
        
        let performance_improvements = tracker.get_pattern_improvements();
        let overall_improvement = performance_improvements.values().sum::<f32>() / 
                                 performance_improvements.len() as f32;
        
        Ok(LearningBenefitAssessment {
            overall_improvement,
            satisfaction_improvement,
            pattern_improvements: performance_improvements,
            learning_effectiveness: analysis.adaptation_effectiveness,
            recommendation: if overall_improvement > 0.1 {
                "Continue current learning approach".to_string()
            } else if overall_improvement > 0.0 {
                "Consider adjusting learning parameters".to_string()
            } else {
                "Review learning strategy - potential issues detected".to_string()
            },
        })
    }
    
    /// Update configuration
    pub fn update_configuration(&mut self, new_config: Phase4CognitiveConfig) -> Result<()> {
        self.configuration = new_config;
        Ok(())
    }
    
    /// Get performance report
    pub fn get_performance_report(&self) -> String {
        let tracker = self.performance_tracker.read().unwrap();
        tracker.generate_performance_report()
    }
    
    /// Create performance data from execution
    fn create_performance_data(
        &self,
        phase3_result: &crate::cognitive::phase3_integration::Phase3QueryResult,
        pattern_weights: &HashMap<CognitivePatternType, f32>,
    ) -> Result<PerformanceData> {
        let mut component_scores = HashMap::new();
        
        // Add pattern scores to component scores
        for (pattern, weight) in pattern_weights {
            component_scores.insert(format!("{:?}", pattern), *weight);
        }
        
        Ok(PerformanceData {
            query_latencies: vec![phase3_result.performance_metrics.total_time],
            memory_usage: vec![0.5], // Estimated memory usage
            accuracy_scores: vec![phase3_result.confidence],
            user_satisfaction: vec![phase3_result.confidence * 0.9], // Estimated from confidence
            system_stability: 0.9, // Would be measured from system
            error_rates: HashMap::new(),
            throughput_metrics: ThroughputMetrics {
                queries_per_second: 10.0,
                successful_queries: 1,
                failed_queries: 0,
                average_response_time: phase3_result.performance_metrics.total_time,
            },
            timestamp: SystemTime::now(),
            system_health: 0.9,
            overall_performance_score: phase3_result.confidence,
            component_scores,
            bottlenecks: Vec::new(),
        })
    }
    
    /// Record performance and learn from execution
    async fn record_and_learn_from_execution(
        &self,
        performance_data: &PerformanceData,
        context: &str,
    ) -> Result<()> {
        // Record performance
        self.performance_tracker.write().unwrap()
            .record_performance(performance_data.clone());
        
        // Learn from execution for each pattern
        for (pattern_str, &score) in &performance_data.component_scores {
            if let Ok(pattern_type) = pattern_str.parse::<CognitivePatternType>() {
                self.learning_enhanced_orchestrator.learn_from_execution(
                    pattern_type,
                    context,
                    score > 0.7, // Success threshold
                    performance_data.throughput_metrics.average_response_time,
                    score,
                )?;
            }
        }
        
        // Check for adaptation triggers
        let _adaptation_events = self.trigger_adaptation(performance_data).await?;
        
        Ok(())
    }
}

/// Enhanced cognitive result with learning integration
#[derive(Debug, Clone)]
pub struct EnhancedCognitiveResult {
    pub base_result: crate::cognitive::phase3_integration::Phase3QueryResult,
    pub pattern_weights: HashMap<CognitivePatternType, f32>,
    pub shortcuts_applied: Vec<CognitiveShortcut>,
    pub learning_insights: HashMap<CognitivePatternType, f32>,
    pub ensemble_composition: HashMap<CognitivePatternType, f32>,
    pub optimizations_applied: Vec<CognitiveOptimization>,
    pub performance_data: super::types::PerformanceData,
}

/// Learning feedback structure
#[derive(Debug, Clone)]
pub struct LearningFeedback {
    pub pattern_type: CognitivePatternType,
    pub performance_data: super::types::PerformanceData,
    pub context: String,
    pub learning_insights: LearningInsights,
}

impl Default for Phase4CognitiveConfig {
    fn default() -> Self {
        Self {
            learning_integration_level: LearningIntegrationLevel::Standard,
            adaptation_aggressiveness: 0.6,
            personalization_enabled: true,
            continuous_optimization: true,
            safety_mode: SafetyMode::Balanced,
        }
    }
}

// Extension methods for Phase3IntegratedCognitiveSystem
impl crate::cognitive::phase3_integration::Phase3IntegratedCognitiveSystem {
    /// Get base orchestrator for Phase 4 integration (removed duplicate)
    // Duplicate method removed - use the one in phase3_integration.rs
    
    /// Execute query with context
    pub async fn execute_query(&self, query: &str, _context: &str) -> Result<crate::cognitive::phase3_integration::Phase3QueryResult> {
        // This would execute the actual Phase 3 query
        // For now, we'll return a mock result
        Ok(crate::cognitive::phase3_integration::Phase3QueryResult {
            query: query.to_string(),
            response: "Mock Phase 3 response".to_string(),
            confidence: 0.85,
            reasoning_trace: crate::cognitive::phase3_integration::ReasoningTrace::new(),
            performance_metrics: crate::cognitive::phase3_integration::QueryPerformanceMetrics {
                total_time: std::time::Duration::from_millis(150),
                pattern_execution_times: std::collections::HashMap::new(),
                memory_operation_times: std::collections::HashMap::new(),
                attention_shift_time: std::time::Duration::from_millis(5),
                inhibition_processing_time: std::time::Duration::from_millis(3),
                consolidation_time: std::time::Duration::from_millis(10),
            },
            system_state_changes: crate::cognitive::phase3_integration::SystemStateChanges::new(),
            overall_confidence: 0.85,
            pattern_results: std::collections::HashMap::new(),
            response_time: std::time::Duration::from_millis(150),
            query_complexity: crate::cognitive::types::ComplexityEstimate {
                computational_complexity: 1,
                estimated_time_ms: 150,
                memory_requirements_mb: 10,
                confidence: 0.8,
                parallelizable: true,
            },
            context: crate::cognitive::types::QueryContext::new(),
            primary_pattern: crate::cognitive::CognitivePatternType::Convergent,
        })
    }
}

// Extension methods for CognitivePatternType
impl std::str::FromStr for CognitivePatternType {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Convergent" => Ok(CognitivePatternType::Convergent),
            "Divergent" => Ok(CognitivePatternType::Divergent),
            "Lateral" => Ok(CognitivePatternType::Lateral),
            "Systems" => Ok(CognitivePatternType::Systems),
            "Critical" => Ok(CognitivePatternType::Critical),
            "Abstract" => Ok(CognitivePatternType::Abstract),
            "Adaptive" => Ok(CognitivePatternType::Adaptive),
            _ => Err(anyhow::anyhow!("Unknown cognitive pattern type: {}", s)),
        }
    }
}

// Extension methods for Phase3QueryResult
impl crate::cognitive::phase3_integration::Phase3QueryResult {
    /// Create new Phase 3 query result
    pub fn new(
        query: String,
        response: String,
        confidence: f32,
    ) -> Self {
        Self {
            query,
            response,
            confidence,
            reasoning_trace: crate::cognitive::phase3_integration::ReasoningTrace::new(),
            performance_metrics: crate::cognitive::phase3_integration::QueryPerformanceMetrics {
                total_time: std::time::Duration::from_millis(100),
                pattern_execution_times: std::collections::HashMap::new(),
                memory_operation_times: std::collections::HashMap::new(),
                attention_shift_time: std::time::Duration::from_millis(5),
                inhibition_processing_time: std::time::Duration::from_millis(3),
                consolidation_time: std::time::Duration::from_millis(10),
            },
            system_state_changes: crate::cognitive::phase3_integration::SystemStateChanges::new(),
            overall_confidence: confidence,
            pattern_results: std::collections::HashMap::new(),
            response_time: std::time::Duration::from_millis(100),
            query_complexity: crate::cognitive::types::ComplexityEstimate {
                computational_complexity: 1,
                estimated_time_ms: 100,
                memory_requirements_mb: 10,
                confidence: 0.8,
                parallelizable: true,
            },
            context: crate::cognitive::types::QueryContext::new(),
            primary_pattern: crate::cognitive::CognitivePatternType::Convergent,
        }
    }
}