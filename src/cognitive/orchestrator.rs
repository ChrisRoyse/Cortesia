use std::sync::Arc;
use std::collections::HashMap as AHashMap;

use crate::cognitive::types::*;
use crate::cognitive::{
    ConvergentThinking, DivergentThinking, LateralThinking, SystemsThinking,
    CriticalThinking, AbstractThinking, AdaptiveThinking,
};
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::monitoring::collectors::runtime_profiler::RuntimeProfiler;
use crate::trace_function;
// Neural server dependency removed - using pure graph operations
use crate::monitoring::performance::{PerformanceMonitor, Operation};
use crate::error::{Result, GraphError};

/// Central orchestrator for all cognitive patterns
pub struct CognitiveOrchestrator {
    patterns: AHashMap<CognitivePatternType, Arc<dyn CognitivePattern>>,
    adaptive_selector: Arc<AdaptiveThinking>,
    performance_monitor: Arc<PerformanceMonitor>,
    brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    config: CognitiveOrchestratorConfig,
    runtime_profiler: Option<Arc<RuntimeProfiler>>,
}

impl std::fmt::Debug for CognitiveOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CognitiveOrchestrator")
            .field("patterns", &self.patterns.keys().collect::<Vec<_>>())
            .field("adaptive_selector", &"AdaptiveThinking")
            .field("performance_monitor", &self.performance_monitor)
            .field("brain_graph", &"BrainEnhancedKnowledgeGraph")
            .field("config", &self.config)
            .finish()
    }
}

/// Configuration for the cognitive orchestrator
#[derive(Debug, Clone)]
pub struct CognitiveOrchestratorConfig {
    pub enable_adaptive_selection: bool,
    pub enable_ensemble_methods: bool,
    pub default_timeout_ms: u64,
    pub max_parallel_patterns: usize,
    pub performance_tracking: bool,
}

impl Default for CognitiveOrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_selection: true,
            enable_ensemble_methods: true,
            default_timeout_ms: 5000,
            max_parallel_patterns: 3,
            performance_tracking: true,
        }
    }
}

impl CognitiveOrchestrator {
    /// Create a new cognitive orchestrator with all patterns
    pub async fn new(
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        config: CognitiveOrchestratorConfig,
    ) -> Result<Self> {
        let performance_monitor = Arc::new(PerformanceMonitor::new_with_defaults().await?);
        
        // Initialize all cognitive patterns
        let convergent = Arc::new(ConvergentThinking::new(
            brain_graph.clone(),
        ));
        
        let divergent = Arc::new(DivergentThinking::new(
            brain_graph.clone(),
        ));
        
        let lateral = Arc::new(LateralThinking::new(
            brain_graph.clone(),
        ));
        
        let systems = Arc::new(SystemsThinking::new(
            brain_graph.clone(),
        ));
        
        let critical = Arc::new(CriticalThinking::new(
            brain_graph.clone(),
        ));
        
        let abstract_pattern = Arc::new(AbstractThinking::new(
            brain_graph.clone(),
        ));
        
        let adaptive = Arc::new(AdaptiveThinking::new(
            brain_graph.clone(),
        ));
        
        // Create pattern registry
        let mut patterns: AHashMap<CognitivePatternType, Arc<dyn CognitivePattern>> = AHashMap::new();
        patterns.insert(CognitivePatternType::Convergent, convergent);
        patterns.insert(CognitivePatternType::Divergent, divergent);
        patterns.insert(CognitivePatternType::Lateral, lateral);
        patterns.insert(CognitivePatternType::Systems, systems);
        patterns.insert(CognitivePatternType::Critical, critical);
        patterns.insert(CognitivePatternType::Abstract, abstract_pattern);
        patterns.insert(CognitivePatternType::Adaptive, adaptive.clone());
        
        Ok(Self {
            patterns,
            adaptive_selector: adaptive,
            performance_monitor,
            brain_graph,
            config,
            runtime_profiler: None,
        })
    }
    
    /// Set runtime profiler for function tracing
    pub fn set_runtime_profiler(&mut self, profiler: Arc<RuntimeProfiler>) {
        self.runtime_profiler = Some(profiler);
    }
    
    /// Main reasoning entry point
    pub async fn reason(
        &self,
        query: &str,
        context: Option<&str>,
        strategy: ReasoningStrategy,
    ) -> Result<ReasoningResult> {
        let _trace = if let Some(profiler) = &self.runtime_profiler {
            Some(trace_function!(profiler, "cognitive_reason", query.len(), context.map(|c| c.len()).unwrap_or(0)))
        } else {
            None
        };
        
        let start_time = std::time::Instant::now();
        
        let result = match strategy.clone() {
            ReasoningStrategy::Automatic => {
                if self.config.enable_adaptive_selection {
                    self.execute_adaptive_reasoning(query, context).await?
                } else {
                    // Fallback to convergent thinking
                    self.execute_specific_pattern(query, context, CognitivePatternType::Convergent).await?
                }
            }
            ReasoningStrategy::Specific(pattern_type) => {
                self.execute_specific_pattern(query, context, pattern_type).await?
            }
            ReasoningStrategy::Ensemble(pattern_types) => {
                if self.config.enable_ensemble_methods {
                    self.execute_ensemble_reasoning(query, context, pattern_types).await?
                } else {
                    return Err(GraphError::UnsupportedOperation("Ensemble methods disabled".to_string()));
                }
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Record performance metrics
        if self.config.performance_tracking {
            let operation = Operation {
                name: format!("cognitive_reasoning_{:?}", strategy),
                operation_type: "reasoning".to_string(),
                custom_metrics: {
                    let mut metrics = std::collections::HashMap::new();
                    metrics.insert("confidence".to_string(), result.quality_metrics.overall_confidence as f64);
                    metrics.insert("patterns_used".to_string(), result.execution_metadata.patterns_executed.len() as f64);
                    metrics
                },
            };
            
            let _ = self.performance_monitor.end_operation_tracking(
                &format!("reasoning_{}", query.chars().take(20).collect::<String>()),
                &operation,
                execution_time,
                true,
            ).await;
        }
        
        Ok(result)
    }
    
    /// Execute adaptive reasoning (automatic pattern selection)
    async fn execute_adaptive_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
    ) -> Result<ReasoningResult> {
        let adaptive_result = self.adaptive_selector.execute_adaptive_reasoning(
            query,
            context,
            self.get_available_patterns(),
        ).await?;
        
        Ok(ReasoningResult {
            query: query.to_string(),
            final_answer: adaptive_result.final_answer,
            strategy_used: ReasoningStrategy::Automatic,
            execution_metadata: ExecutionMetadata {
                total_time_ms: 0, // Will be filled by caller
                patterns_executed: adaptive_result.strategy_used.selected_patterns,
                nodes_activated: 0, // Would be aggregated from pattern results
                energy_consumed: 0.0,
                cache_hits: 0,
                cache_misses: 0,
            },
            quality_metrics: QualityMetrics {
                overall_confidence: adaptive_result.confidence_distribution.ensemble_confidence,
                consistency_score: self.calculate_consistency_score(&adaptive_result.pattern_contributions),
                completeness_score: self.calculate_completeness_score(&adaptive_result.pattern_contributions),
                novelty_score: self.calculate_novelty_score(&adaptive_result.pattern_contributions),
                efficiency_score: adaptive_result.learning_update.strategy_effectiveness,
            },
        })
    }
    
    /// Execute a specific cognitive pattern
    async fn execute_specific_pattern(
        &self,
        query: &str,
        context: Option<&str>,
        pattern_type: CognitivePatternType,
    ) -> Result<ReasoningResult> {
        let pattern = self.patterns.get(&pattern_type)
            .ok_or(GraphError::PatternNotFound(format!("{:?}", pattern_type)))?;
        
        let pattern_result = pattern.execute(
            query,
            context,
            PatternParameters::default(),
        ).await?;
        
        // Extract values before moving
        let answer = pattern_result.answer.clone();
        let confidence = pattern_result.confidence;
        let execution_time = pattern_result.metadata.execution_time_ms;
        let nodes_activated = pattern_result.metadata.nodes_activated;
        let energy_consumed = pattern_result.metadata.total_energy;
        let completeness_score = self.estimate_completeness(&pattern_result);
        let novelty_score = self.estimate_novelty(&pattern_result);
        let efficiency_score = self.calculate_efficiency_score(&pattern_result.metadata);
        
        Ok(ReasoningResult {
            query: query.to_string(),
            final_answer: answer,
            strategy_used: ReasoningStrategy::Specific(pattern_type),
            execution_metadata: ExecutionMetadata {
                total_time_ms: execution_time,
                patterns_executed: vec![pattern_type],
                nodes_activated,
                energy_consumed,
                cache_hits: 0,
                cache_misses: 0,
            },
            quality_metrics: QualityMetrics {
                overall_confidence: confidence,
                consistency_score: 1.0, // Single pattern is always consistent with itself
                completeness_score,
                novelty_score,
                efficiency_score,
            },
        })
    }
    
    /// Execute ensemble reasoning with multiple patterns
    async fn execute_ensemble_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
        pattern_types: Vec<CognitivePatternType>,
    ) -> Result<ReasoningResult> {
        let mut pattern_results = Vec::new();
        let mut _tasks: Vec<tokio::task::JoinHandle<Result<PatternResult>>> = Vec::new();
        
        // Execute patterns in parallel (up to max_parallel_patterns)
        for chunk in pattern_types.chunks(self.config.max_parallel_patterns) {
            let mut chunk_tasks = Vec::new();
            
            for &pattern_type in chunk {
                if let Some(pattern) = self.patterns.get(&pattern_type) {
                    let pattern_clone = pattern.clone();
                    let query_clone = query.to_string();
                    let context_clone = context.map(|s| s.to_string());
                    
                    let task = tokio::spawn(async move {
                        pattern_clone.execute(
                            &query_clone,
                            context_clone.as_deref(),
                            PatternParameters::default(),
                        ).await
                    });
                    
                    chunk_tasks.push((pattern_type, task));
                }
            }
            
            // Wait for chunk to complete
            for (pattern_type, task) in chunk_tasks {
                match task.await {
                    Ok(Ok(result)) => pattern_results.push((pattern_type, result)),
                    Ok(Err(e)) => {
                        log::warn!("Pattern {:?} failed: {}", pattern_type, e);
                        // Continue with other patterns
                    }
                    Err(e) => {
                        log::warn!("Pattern {:?} task failed: {}", pattern_type, e);
                    }
                }
            }
        }
        
        if pattern_results.is_empty() {
            return Err(GraphError::ProcessingError("All patterns failed".to_string()));
        }
        
        // Merge results using ensemble methods
        let ensemble_result = self.merge_pattern_results(pattern_results).await?;
        
        Ok(ensemble_result)
    }
    
    /// Merge results from multiple patterns using ensemble methods
    async fn merge_pattern_results(
        &self,
        pattern_results: Vec<(CognitivePatternType, PatternResult)>,
    ) -> Result<ReasoningResult> {
        if pattern_results.is_empty() {
            return Err(GraphError::ProcessingError("No results to merge".to_string()));
        }
        
        // Simple confidence-weighted averaging for now
        // In a full implementation, this would use sophisticated ensemble methods
        let mut total_weighted_confidence = 0.0;
        let mut total_weight = 0.0;
        let mut best_answer = String::new();
        let mut best_confidence = 0.0;
        
        let executed_patterns: Vec<CognitivePatternType> = pattern_results.iter()
            .map(|(pattern_type, _)| *pattern_type)
            .collect();
        
        let mut total_execution_time = 0;
        let mut total_nodes_activated = 0;
        let mut total_energy = 0.0;
        
        // Collect all non-empty answers for potential combination
        let mut all_answers: Vec<(CognitivePatternType, String, f32)> = Vec::new();
        
        for (pattern_type, result) in &pattern_results {
            let weight = self.get_pattern_weight(*pattern_type);
            total_weighted_confidence += result.confidence * weight;
            total_weight += weight;
            
            if !result.answer.trim().is_empty() {
                all_answers.push((*pattern_type, result.answer.clone(), result.confidence));
            }
            
            if result.confidence > best_confidence && !result.answer.trim().is_empty() {
                best_confidence = result.confidence;
                best_answer = result.answer.clone();
            }
            
            total_execution_time += result.metadata.execution_time_ms;
            total_nodes_activated += result.metadata.nodes_activated;
            total_energy += result.metadata.total_energy;
        }
        
        // If no single best answer, combine results from multiple patterns
        if best_answer.is_empty() && !all_answers.is_empty() {
            best_answer = self.combine_pattern_answers(&all_answers);
        } else if best_answer.is_empty() {
            // Fallback: provide a summary of what was attempted
            best_answer = format!(
                "Ensemble reasoning attempted with {} patterns but found limited results. Patterns used: {:?}",
                pattern_results.len(),
                executed_patterns
            );
        }
        
        let ensemble_confidence = if total_weight > 0.0 {
            total_weighted_confidence / total_weight
        } else {
            best_confidence
        };
        
        Ok(ReasoningResult {
            query: pattern_results[0].1.metadata.additional_info.get("query")
                .unwrap_or(&"Unknown".to_string()).clone(),
            final_answer: best_answer,
            strategy_used: ReasoningStrategy::Ensemble(executed_patterns.clone()),
            execution_metadata: ExecutionMetadata {
                total_time_ms: total_execution_time,
                patterns_executed: executed_patterns,
                nodes_activated: total_nodes_activated,
                energy_consumed: total_energy,
                cache_hits: 0,
                cache_misses: 0,
            },
            quality_metrics: QualityMetrics {
                overall_confidence: ensemble_confidence,
                consistency_score: self.calculate_ensemble_consistency(&pattern_results),
                completeness_score: self.calculate_ensemble_completeness(&pattern_results),
                novelty_score: self.calculate_ensemble_novelty(&pattern_results),
                efficiency_score: self.calculate_ensemble_efficiency(&pattern_results),
            },
        })
    }
    
    /// Get available cognitive patterns
    fn get_available_patterns(&self) -> Vec<CognitivePatternType> {
        self.patterns.keys().cloned().collect()
    }
    
    /// Combine answers from multiple patterns into a coherent response
    fn combine_pattern_answers(&self, answers: &[(CognitivePatternType, String, f32)]) -> String {
        if answers.is_empty() {
            return String::new();
        }
        
        if answers.len() == 1 {
            return answers[0].1.clone();
        }
        
        // Group answers by pattern type for better organization
        let mut combined = String::new();
        combined.push_str("Combined insights from multiple cognitive patterns:\n\n");
        
        for (pattern_type, answer, confidence) in answers {
            if confidence > &0.1 {  // Only include answers with minimal confidence
                let pattern_name = match pattern_type {
                    CognitivePatternType::Convergent => "Factual Analysis",
                    CognitivePatternType::Divergent => "Exploratory Findings",
                    CognitivePatternType::Lateral => "Creative Connections",
                    CognitivePatternType::Systems => "Systems Perspective",
                    CognitivePatternType::Critical => "Critical Analysis",
                    CognitivePatternType::Abstract => "Pattern Recognition",
                    CognitivePatternType::Adaptive => "Adaptive Insights",
                    CognitivePatternType::ChainOfThought => "Chain of Thought",
                    CognitivePatternType::TreeOfThoughts => "Tree of Thoughts",
                };
                combined.push_str(&format!("{}: {}\n", pattern_name, answer.trim()));
            }
        }
        
        combined.trim().to_string()
    }
    
    /// Get weight for a specific pattern in ensemble methods
    fn get_pattern_weight(&self, pattern_type: CognitivePatternType) -> f32 {
        // Default weights - could be learned over time
        match pattern_type {
            CognitivePatternType::Convergent => 1.0,
            CognitivePatternType::Divergent => 0.8,
            CognitivePatternType::Lateral => 0.6,
            CognitivePatternType::Systems => 0.9,
            CognitivePatternType::Critical => 1.1,
            CognitivePatternType::Abstract => 0.7,
            CognitivePatternType::Adaptive => 1.2,
            CognitivePatternType::ChainOfThought => 1.0,
            CognitivePatternType::TreeOfThoughts => 0.9,
        }
    }
    
    /// Calculate consistency score for pattern contributions
    fn calculate_consistency_score(&self, contributions: &[PatternContribution]) -> f32 {
        if contributions.len() < 2 {
            return 1.0;
        }
        
        // Simple consistency measure based on confidence variance
        let confidences: Vec<f32> = contributions.iter().map(|c| c.confidence).collect();
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>() / confidences.len() as f32;
        
        // Convert variance to consistency score (lower variance = higher consistency)
        1.0 / (1.0 + variance)
    }
    
    /// Calculate completeness score for pattern contributions
    fn calculate_completeness_score(&self, contributions: &[PatternContribution]) -> f32 {
        // Completeness based on number of patterns contributing and their weights
        let total_weight: f32 = contributions.iter().map(|c| c.contribution_weight).sum();
        let normalized_completeness = total_weight / contributions.len() as f32;
        normalized_completeness.min(1.0)
    }
    
    /// Calculate novelty score for pattern contributions
    fn calculate_novelty_score(&self, contributions: &[PatternContribution]) -> f32 {
        // Higher novelty if lateral or divergent thinking contributed significantly
        let mut novelty = 0.0;
        for contribution in contributions {
            let pattern_novelty = match contribution.pattern_type {
                CognitivePatternType::Lateral => 0.9,
                CognitivePatternType::Divergent => 0.7,
                CognitivePatternType::Abstract => 0.6,
                CognitivePatternType::Adaptive => 0.5,
                _ => 0.3,
            };
            novelty += pattern_novelty * contribution.contribution_weight;
        }
        novelty / contributions.len() as f32
    }
    
    /// Estimate completeness for single pattern result
    fn estimate_completeness(&self, result: &PatternResult) -> f32 {
        // Based on reasoning trace length and confidence
        let trace_completeness = (result.reasoning_trace.len() as f32 / 10.0).min(1.0);
        (trace_completeness + result.confidence) / 2.0
    }
    
    /// Estimate novelty for single pattern result
    fn estimate_novelty(&self, result: &PatternResult) -> f32 {
        match result.pattern_type {
            CognitivePatternType::Lateral => 0.9,
            CognitivePatternType::Divergent => 0.7,
            CognitivePatternType::Abstract => 0.6,
            _ => 0.3,
        }
    }
    
    /// Calculate efficiency score from execution metadata
    fn calculate_efficiency_score(&self, metadata: &ResultMetadata) -> f32 {
        // Efficiency based on time, energy, and convergence
        let time_efficiency = 1.0 / (1.0 + metadata.execution_time_ms as f32 / 1000.0);
        let energy_efficiency = 1.0 / (1.0 + metadata.total_energy);
        let convergence_bonus = if metadata.converged { 0.2 } else { 0.0 };
        
        (time_efficiency + energy_efficiency) / 2.0 + convergence_bonus
    }
    
    /// Calculate consistency for ensemble results
    fn calculate_ensemble_consistency(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        if results.len() < 2 {
            return 1.0;
        }
        
        let confidences: Vec<f32> = results.iter().map(|(_, r)| r.confidence).collect();
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>() / confidences.len() as f32;
        
        1.0 / (1.0 + variance)
    }
    
    /// Calculate completeness for ensemble results
    fn calculate_ensemble_completeness(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        let avg_completeness: f32 = results.iter()
            .map(|(_, r)| self.estimate_completeness(r))
            .sum::<f32>() / results.len() as f32;
        
        // Bonus for having multiple perspectives
        let diversity_bonus = 0.1 * (results.len() - 1) as f32;
        (avg_completeness + diversity_bonus).min(1.0)
    }
    
    /// Calculate novelty for ensemble results
    fn calculate_ensemble_novelty(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        results.iter()
            .map(|(_, r)| self.estimate_novelty(r))
            .fold(0.0, f32::max) // Take the maximum novelty
    }
    
    /// Calculate efficiency for ensemble results
    fn calculate_ensemble_efficiency(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        let avg_efficiency: f32 = results.iter()
            .map(|(_, r)| self.calculate_efficiency_score(&r.metadata))
            .sum::<f32>() / results.len() as f32;
        
        // Penalty for using multiple patterns (more resources)
        let resource_penalty = 0.1 * (results.len() - 1) as f32;
        (avg_efficiency - resource_penalty).max(0.0)
    }
    
    /// Get orchestrator statistics
    pub async fn get_statistics(&self) -> Result<OrchestratorStatistics> {
        Ok(OrchestratorStatistics {
            total_patterns: self.patterns.len(),
            available_patterns: self.get_available_patterns(),
            performance_metrics: AHashMap::new(), // Placeholder until monitoring is available
            config: self.config.clone(),
        })
    }
    
    /// Get performance metrics for the orchestrator
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self.performance_monitor.get_metrics().await?;
        
        Ok(PerformanceMetrics {
            total_queries_processed: metrics.get("total_queries").unwrap_or(&0.0).clone() as u64,
            average_response_time_ms: metrics.get("avg_response_time").unwrap_or(&0.0).clone() as f64,
            pattern_usage_stats: {
                let mut usage_stats = AHashMap::new();
                for pattern_type in &self.get_available_patterns() {
                    let usage_key = format!("pattern_{:?}_usage", pattern_type);
                    let usage_count = metrics.get(&usage_key).unwrap_or(&0.0).clone() as u64;
                    usage_stats.insert(*pattern_type, usage_count);
                }
                usage_stats
            },
            success_rate: metrics.get("success_rate").unwrap_or(&0.95).clone() as f64,
            cache_hit_rate: metrics.get("cache_hit_rate").unwrap_or(&0.0).clone() as f64,
            memory_usage_mb: metrics.get("memory_usage_mb").unwrap_or(&0.0).clone() as f64,
            active_entities: metrics.get("active_entities").unwrap_or(&0.0).clone() as u64,
        })
    }
}

/// Calculate pattern weight based on confidence, complexity, and priority
/// This is a private function used for pattern weighting in orchestration
pub(crate) fn calculate_pattern_weight(confidence: f32, complexity: u32, is_priority: bool) -> f32 {
    let mut weight = confidence;
    
    // Adjust for complexity (higher complexity reduces weight)
    weight *= 1.0 / (1.0 + complexity as f32 * 0.1);
    
    // Boost priority patterns
    if is_priority {
        weight *= 1.2;
    }
    
    weight
}

/// Statistics for the cognitive orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorStatistics {
    pub total_patterns: usize,
    pub available_patterns: Vec<CognitivePatternType>,
    pub performance_metrics: AHashMap<String, f32>,
    pub config: CognitiveOrchestratorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_pattern_weight() {
        // Direct test of private function
        assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.9, 5, false));
        assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.7, 5, true));
        assert!(calculate_pattern_weight(0.9, 3, true) > calculate_pattern_weight(0.9, 10, true));
        
        // Additional test cases for edge conditions
        assert!(calculate_pattern_weight(1.0, 0, true) > calculate_pattern_weight(1.0, 0, false));
        assert!(calculate_pattern_weight(0.5, 1, false) > calculate_pattern_weight(0.3, 1, false));
        
        // Test that weight decreases with complexity
        assert!(calculate_pattern_weight(0.8, 1, false) > calculate_pattern_weight(0.8, 5, false));
        assert!(calculate_pattern_weight(0.8, 5, false) > calculate_pattern_weight(0.8, 10, false));
    }
}