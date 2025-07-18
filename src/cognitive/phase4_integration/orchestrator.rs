//! Learning-enhanced cognitive orchestration

use super::types::*;
use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::cognitive::types::CognitivePatternType;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use anyhow::Result;

/// Learning-enhanced cognitive orchestrator
#[derive(Debug, Clone)]
pub struct LearningEnhancedOrchestrator {
    pub base_orchestrator: Arc<CognitiveOrchestrator>,
    pub learning_insights: Arc<RwLock<LearningInsights>>,
    pub adaptive_strategies: Arc<RwLock<AdaptiveStrategies>>,
    pub pattern_performance_history: Arc<RwLock<PatternPerformanceHistory>>,
}

impl LearningEnhancedOrchestrator {
    /// Create new learning-enhanced orchestrator
    pub fn new(base_orchestrator: Arc<CognitiveOrchestrator>) -> Self {
        Self {
            base_orchestrator,
            learning_insights: Arc::new(RwLock::new(LearningInsights::default())),
            adaptive_strategies: Arc::new(RwLock::new(AdaptiveStrategies::default())),
            pattern_performance_history: Arc::new(RwLock::new(PatternPerformanceHistory::default())),
        }
    }
    
    /// Get pattern selection weights based on learning insights
    pub fn get_learning_informed_weights(&self, context: &str) -> Result<HashMap<CognitivePatternType, f32>> {
        let insights = self.learning_insights.read().unwrap();
        let strategies = self.adaptive_strategies.read().unwrap();
        
        let mut weights = strategies.pattern_selection_weights.clone();
        
        // Apply context-sensitive adaptations
        if let Some(adaptation) = strategies.context_sensitive_adaptations.get(context) {
            let weights_clone = weights.clone();
            for (pattern, base_weight) in &weights_clone {
                if let Some(&adjustment) = adaptation.adaptation_parameters.get(&format!("{:?}", pattern)) {
                    weights.insert(pattern.clone(), base_weight * adjustment);
                }
            }
        }
        
        // Apply effective pattern insights
        for (pattern, effectiveness) in &insights.effective_cognitive_patterns {
            if let Some(current_weight) = weights.get_mut(pattern) {
                *current_weight *= effectiveness;
            }
        }
        
        Ok(weights)
    }
    
    /// Update learning insights from new data
    pub fn update_learning_insights(&self, new_insights: LearningInsights) -> Result<()> {
        let mut insights = self.learning_insights.write().unwrap();
        
        // Merge Hebbian connection strengths
        for ((entity1, entity2), strength) in new_insights.hebbian_connection_strengths {
            insights.hebbian_connection_strengths.insert((entity1, entity2), strength);
        }
        
        // Update effective cognitive patterns
        for (pattern, effectiveness) in new_insights.effective_cognitive_patterns {
            insights.effective_cognitive_patterns.insert(pattern, effectiveness);
        }
        
        // Add new optimizations
        insights.learning_derived_optimizations.extend(new_insights.learning_derived_optimizations);
        
        // Update user preference patterns
        for (preference, value) in new_insights.user_preference_patterns {
            insights.user_preference_patterns.insert(preference, value);
        }
        
        // Update performance correlations
        for (correlation, value) in new_insights.performance_correlations {
            insights.performance_correlations.insert(correlation, value);
        }
        
        Ok(())
    }
    
    /// Get recommended pattern combination for context
    pub fn get_recommended_ensemble(&self, context: &str) -> Result<HashMap<CognitivePatternType, f32>> {
        let strategies = self.adaptive_strategies.read().unwrap();
        
        // Find applicable ensemble rules
        let applicable_rules: Vec<&EnsembleRule> = strategies.ensemble_composition_rules
            .iter()
            .filter(|rule| rule.applicable_contexts.contains(&context.to_string()))
            .collect();
        
        if applicable_rules.is_empty() {
            // Return default balanced ensemble
            return Ok(self.get_default_ensemble());
        }
        
        // Select the most effective rule
        let best_rule = applicable_rules
            .iter()
            .max_by(|a, b| a.effectiveness_score.partial_cmp(&b.effectiveness_score).unwrap())
            .unwrap();
        
        Ok(best_rule.pattern_combinations.clone())
    }
    
    /// Get default ensemble composition
    fn get_default_ensemble(&self) -> HashMap<CognitivePatternType, f32> {
        let mut ensemble = HashMap::new();
        ensemble.insert(CognitivePatternType::Convergent, 0.3);
        ensemble.insert(CognitivePatternType::Divergent, 0.2);
        ensemble.insert(CognitivePatternType::Critical, 0.25);
        ensemble.insert(CognitivePatternType::Systems, 0.15);
        ensemble.insert(CognitivePatternType::Adaptive, 0.1);
        ensemble
    }
    
    /// Check for learned shortcuts applicable to context
    pub fn get_applicable_shortcuts(&self, context: &str) -> Result<Vec<CognitiveShortcut>> {
        let strategies = self.adaptive_strategies.read().unwrap();
        
        let shortcuts = strategies.learned_shortcuts
            .iter()
            .filter(|shortcut| {
                shortcut.trigger_conditions.iter().any(|condition| context.contains(condition))
            })
            .cloned()
            .collect();
        
        Ok(shortcuts)
    }
    
    /// Update adaptive strategies based on performance
    pub fn update_adaptive_strategies(&self, performance_data: &PerformanceData) -> Result<()> {
        let mut strategies = self.adaptive_strategies.write().unwrap();
        
        // Update pattern selection weights based on performance
        let weights_clone = strategies.pattern_selection_weights.clone();
        for (pattern, &current_weight) in &weights_clone {
            if let Some(&score) = performance_data.component_scores.get(&format!("{:?}", pattern)) {
                let adjustment = if score > 0.8 {
                    1.1 // Increase weight for high-performing patterns
                } else if score < 0.5 {
                    0.9 // Decrease weight for low-performing patterns
                } else {
                    1.0 // No change
                };
                
                strategies.pattern_selection_weights.insert(pattern.clone(), current_weight * adjustment);
            }
        }
        
        // Update ensemble effectiveness scores
        for rule in &mut strategies.ensemble_composition_rules {
            if performance_data.overall_performance_score > 0.8 {
                rule.effectiveness_score = (rule.effectiveness_score + 0.1).min(1.0);
            } else if performance_data.overall_performance_score < 0.5 {
                rule.effectiveness_score = (rule.effectiveness_score - 0.1).max(0.0);
            }
        }
        
        Ok(())
    }
    
    /// Get failure recovery strategy for pattern
    pub fn get_recovery_strategy(&self, failed_pattern: &CognitivePatternType) -> Result<Option<RecoveryStrategy>> {
        let strategies = self.adaptive_strategies.read().unwrap();
        
        let strategy_key = format!("{:?}", failed_pattern);
        Ok(strategies.failure_recovery_strategies.get(&strategy_key).cloned())
    }
    
    /// Learn from pattern execution results
    pub fn learn_from_execution(&self, 
        pattern: CognitivePatternType, 
        context: &str, 
        success: bool, 
        duration: std::time::Duration, 
        quality_score: f32
    ) -> Result<()> {
        let mut history = self.pattern_performance_history.write().unwrap();
        
        // Update usage statistics
        let stats = history.pattern_usage_statistics.entry(pattern.clone()).or_insert(UsageStatistics {
            total_invocations: 0,
            successful_completions: 0,
            average_duration: duration,
            average_quality_score: quality_score,
            last_used: std::time::SystemTime::now(),
        });
        
        stats.total_invocations += 1;
        if success {
            stats.successful_completions += 1;
        }
        
        // Update average duration
        stats.average_duration = std::time::Duration::from_nanos(
            ((stats.average_duration.as_nanos() as f64 * (stats.total_invocations - 1) as f64) +
             duration.as_nanos() as f64) as u64 / stats.total_invocations as u64
        );
        
        // Update average quality score
        stats.average_quality_score = 
            (stats.average_quality_score * (stats.total_invocations - 1) as f32 + quality_score) / 
            stats.total_invocations as f32;
        
        stats.last_used = std::time::SystemTime::now();
        
        // Update success rates by context
        let context_success_rates = history.success_rates_by_context.entry(context.to_string()).or_insert(HashMap::new());
        let pattern_success_rate = context_success_rates.entry(pattern.clone()).or_insert(0.0);
        
        let success_value = if success { 1.0 } else { 0.0 };
        *pattern_success_rate = (*pattern_success_rate + success_value) / 2.0;
        
        // Add performance data point
        let performance_trends = history.performance_trends.entry(pattern.clone()).or_insert(Vec::new());
        performance_trends.push(PerformanceDataPoint {
            timestamp: std::time::SystemTime::now(),
            quality_score,
            duration,
            context: context.to_string(),
            user_satisfaction: quality_score, // Simplified correlation
        });
        
        // Keep only recent data points (last 100)
        if performance_trends.len() > 100 {
            performance_trends.remove(0);
        }
        
        Ok(())
    }
    
    /// Get performance summary for pattern
    pub fn get_pattern_performance_summary(&self, pattern: &CognitivePatternType) -> Result<Option<UsageStatistics>> {
        let history = self.pattern_performance_history.read().unwrap();
        Ok(history.pattern_usage_statistics.get(pattern).cloned())
    }
}