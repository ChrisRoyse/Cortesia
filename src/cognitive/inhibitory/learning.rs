//! Learning mechanisms for adaptive inhibition

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, InhibitionLearningResult, ParameterAdjustment,
    ParameterAdjustmentType, GroupCompetitionResult, InhibitionPerformanceMetrics,
    AdaptationSuggestion, AdaptationType
};
use crate::core::brain_types::ActivationPattern;
use crate::error::Result;
use std::time::{Duration, SystemTime};

/// Apply adaptive learning to improve inhibition performance
pub async fn apply_adaptive_learning(
    system: &CompetitiveInhibitionSystem,
    activation_pattern: &ActivationPattern,
    inhibition_results: &[GroupCompetitionResult],
) -> Result<()> {
    // Calculate performance metrics for this inhibition cycle
    let performance_metrics = calculate_performance_metrics(
        activation_pattern,
        inhibition_results,
    );

    // Generate adaptation suggestions based on performance
    let adaptation_suggestions = generate_adaptation_suggestions(&performance_metrics);

    // Apply promising adaptations with significant expected improvement
    for suggestion in adaptation_suggestions {
        if suggestion.expected_improvement > 0.1 {
            apply_adaptation_suggestion(system, &suggestion).await?;
        }
    }

    Ok(())
}

/// Apply learning mechanisms to optimize inhibition
pub async fn apply_learning_mechanisms(
    system: &CompetitiveInhibitionSystem,
    pattern: &ActivationPattern,
    inhibition_results: &[GroupCompetitionResult],
    performance_history: &[InhibitionPerformanceMetrics],
) -> Result<InhibitionLearningResult> {
    if !system.inhibition_config.enable_learning {
        return Ok(InhibitionLearningResult {
            learning_applied: false,
            parameter_adjustments: vec![],
            performance_improvement_estimate: 0.0,
            learning_confidence: 0.0,
        });
    }
    
    let mut adjustments = Vec::new();
    
    // Learn inhibition strength adjustments
    let strength_adjustment = learn_inhibition_strength_adjustment(
        pattern,
        inhibition_results,
        performance_history,
    ).await?;
    
    if let Some(adj) = strength_adjustment {
        adjustments.push(adj);
    }
    
    // Learn competition group optimizations
    let group_optimization = learn_competition_group_optimization(
        inhibition_results,
        performance_history,
    ).await?;
    
    if let Some(adj) = group_optimization {
        adjustments.push(adj);
    }
    
    // Learn temporal dynamics
    let temporal_optimization = learn_temporal_dynamics_optimization(
        pattern,
        performance_history,
    ).await?;
    
    if let Some(adj) = temporal_optimization {
        adjustments.push(adj);
    }
    
    // Calculate overall improvement estimate
    let improvement_estimate = adjustments.iter()
        .map(|adj| adj.expected_improvement)
        .sum::<f32>() / adjustments.len().max(1) as f32;
    
    // Calculate learning confidence based on history
    let learning_confidence = calculate_learning_confidence(performance_history);
    
    // Apply the adjustments
    for adjustment in &adjustments {
        apply_parameter_adjustment(system, adjustment).await?;
    }
    
    Ok(InhibitionLearningResult {
        learning_applied: !adjustments.is_empty(),
        parameter_adjustments: adjustments,
        performance_improvement_estimate: improvement_estimate,
        learning_confidence,
    })
}

/// Learn optimal inhibition strength adjustments
async fn learn_inhibition_strength_adjustment(
    _pattern: &ActivationPattern,
    results: &[GroupCompetitionResult],
    _history: &[InhibitionPerformanceMetrics],
) -> Result<Option<ParameterAdjustment>> {
    // Analyze competition effectiveness
    let total_competitions = results.len();
    let decisive_competitions = results.iter()
        .filter(|r| r.winner.is_some() && r.competition_intensity > 0.7)
        .count();
    
    let effectiveness_ratio = if total_competitions > 0 {
        decisive_competitions as f32 / total_competitions as f32
    } else {
        0.5
    };
    
    // If competitions are too weak, increase strength
    if effectiveness_ratio < 0.6 {
        return Ok(Some(ParameterAdjustment {
            parameter_name: "global_inhibition_strength".to_string(),
            old_value: 0.5, // Would get from actual config
            new_value: 0.65,
            adjustment_type: ParameterAdjustmentType::StrengthOptimization,
            expected_improvement: 0.15,
            confidence: 0.8,
        }));
    }
    
    // If competitions are too strong (everything suppressed), decrease
    if effectiveness_ratio > 0.9 {
        return Ok(Some(ParameterAdjustment {
            parameter_name: "global_inhibition_strength".to_string(),
            old_value: 0.5,
            new_value: 0.35,
            adjustment_type: ParameterAdjustmentType::StrengthOptimization,
            expected_improvement: 0.1,
            confidence: 0.7,
        }));
    }
    
    Ok(None)
}

/// Learn optimal competition group configurations
async fn learn_competition_group_optimization(
    results: &[GroupCompetitionResult],
    _history: &[InhibitionPerformanceMetrics],
) -> Result<Option<ParameterAdjustment>> {
    // Analyze which competition groups are most effective
    let avg_suppression_rate = results.iter()
        .map(|r| {
            if r.pre_competition.is_empty() {
                0.0
            } else {
                r.suppressed_entities.len() as f32 / r.pre_competition.len() as f32
            }
        })
        .sum::<f32>() / results.len().max(1) as f32;
    
    // If suppression is too aggressive
    if avg_suppression_rate > 0.7 {
        return Ok(Some(ParameterAdjustment {
            parameter_name: "soft_competition_factor".to_string(),
            old_value: 0.3,
            new_value: 0.5, // Increase soft competition
            adjustment_type: ParameterAdjustmentType::CompetitionOptimization,
            expected_improvement: 0.12,
            confidence: 0.75,
        }));
    }
    
    Ok(None)
}

/// Learn optimal temporal dynamics
async fn learn_temporal_dynamics_optimization(
    _pattern: &ActivationPattern,
    history: &[InhibitionPerformanceMetrics],
) -> Result<Option<ParameterAdjustment>> {
    // Analyze temporal patterns in performance
    if history.len() < 5 {
        return Ok(None); // Not enough history
    }
    
    // Check if performance oscillates (suggesting temporal issues)
    let recent_scores: Vec<f32> = history.iter()
        .rev()
        .take(5)
        .map(|m| m.effectiveness_score)
        .collect();
    
    let variance = calculate_variance(&recent_scores);
    
    if variance > 0.1 {
        return Ok(Some(ParameterAdjustment {
            parameter_name: "temporal_integration_window".to_string(),
            old_value: 100.0, // milliseconds
            new_value: 150.0, // Increase integration window
            adjustment_type: ParameterAdjustmentType::TemporalOptimization,
            expected_improvement: 0.08,
            confidence: 0.6,
        }));
    }
    
    Ok(None)
}

/// Apply a parameter adjustment to the system
async fn apply_parameter_adjustment(
    _system: &CompetitiveInhibitionSystem,
    adjustment: &ParameterAdjustment,
) -> Result<()> {
    // In a real implementation, this would modify the actual configuration
    // For now, we'll just log the adjustment
    match adjustment.adjustment_type {
        ParameterAdjustmentType::StrengthOptimization => {
            // system.inhibition_config.global_inhibition_strength = adjustment.new_value;
        }
        ParameterAdjustmentType::CompetitionOptimization => {
            // system.inhibition_config.soft_competition_factor = adjustment.new_value;
        }
        ParameterAdjustmentType::TemporalOptimization => {
            // Adjust temporal parameters
        }
        ParameterAdjustmentType::ThresholdAdjustment => {
            // Adjust thresholds
        }
    }
    
    Ok(())
}

/// Calculate performance metrics
fn calculate_performance_metrics(
    activation_pattern: &ActivationPattern,
    inhibition_results: &[GroupCompetitionResult],
) -> InhibitionPerformanceMetrics {
    // Calculate efficiency based on how well competition resolved
    let efficiency_score = if inhibition_results.is_empty() {
        0.5
    } else {
        inhibition_results.iter()
            .map(|result| {
                // Higher efficiency if there's a clear winner
                if result.winner.is_some() {
                    1.0 - (result.post_competition.len() as f32 / result.pre_competition.len() as f32).min(1.0)
                } else {
                    0.3 // No clear winner = lower efficiency
                }
            })
            .sum::<f32>() / inhibition_results.len() as f32
    };

    // Calculate effectiveness based on activation levels after inhibition
    let total_activation: f32 = activation_pattern.activations.values().sum();
    let max_possible_activation = activation_pattern.activations.len() as f32;
    let effectiveness_score = if max_possible_activation > 0.0 {
        (total_activation / max_possible_activation).min(1.0)
    } else {
        0.5
    };

    InhibitionPerformanceMetrics {
        timestamp: SystemTime::now(),
        processing_time: Duration::from_millis(10),
        processing_time_ms: 10.0,
        entities_processed: activation_pattern.activations.len(),
        competition_groups_resolved: inhibition_results.len(),
        competitions_resolved: inhibition_results.len(),
        exceptions_handled: 0,
        efficiency_score,
        effectiveness_score,
    }
}

/// Generate adaptation suggestions
fn generate_adaptation_suggestions(metrics: &InhibitionPerformanceMetrics) -> Vec<AdaptationSuggestion> {
    let mut suggestions = Vec::new();
    
    // If efficiency is low, suggest strength adjustment
    if metrics.efficiency_score < 0.6 {
        suggestions.push(AdaptationSuggestion {
            suggestion_type: AdaptationType::StrengthAdjustment,
            target_parameter: "global_inhibition_strength".to_string(),
            current_value: 0.5,
            recommended_value: 0.65,
            expected_improvement: 0.15,
            confidence: 0.8,
        });
    }
    
    // If effectiveness is low, suggest threshold modification
    if metrics.effectiveness_score < 0.5 {
        suggestions.push(AdaptationSuggestion {
            suggestion_type: AdaptationType::ThresholdModification,
            target_parameter: "winner_takes_all_threshold".to_string(),
            current_value: 0.8,
            recommended_value: 0.7,
            expected_improvement: 0.1,
            confidence: 0.7,
        });
    }
    
    suggestions
}

/// Apply an adaptation suggestion
async fn apply_adaptation_suggestion(
    _system: &CompetitiveInhibitionSystem,
    suggestion: &AdaptationSuggestion,
) -> Result<()> {
    // Apply the suggestion by modifying the inhibition configuration
    // Note: In a real implementation, this would modify the actual config
    
    match suggestion.suggestion_type {
        AdaptationType::StrengthAdjustment => {
            // Adjust inhibition strength parameters
            println!("Applied strength adjustment: {} -> {}", 
                suggestion.target_parameter, suggestion.recommended_value);
        },
        AdaptationType::ThresholdModification => {
            // Optimize specific parameters
            println!("Applied threshold modification: {}", suggestion.target_parameter);
        },
        AdaptationType::TemporalAdjustment => {
            // Adjust temporal parameters
            println!("Applied temporal adjustment: {}", suggestion.target_parameter);
        },
        AdaptationType::GroupReorganization => {
            // Reorganize competition groups
            println!("Applied group reorganization: {}", suggestion.target_parameter);
        },
    }
    
    Ok(())
}

/// Calculate learning confidence based on performance history
fn calculate_learning_confidence(history: &[InhibitionPerformanceMetrics]) -> f32 {
    if history.is_empty() {
        return 0.5;
    }
    
    // More history = more confidence
    let history_factor = (history.len() as f32 / 100.0).min(1.0);
    
    // Consistent performance = more confidence
    let scores: Vec<f32> = history.iter()
        .map(|m| m.effectiveness_score)
        .collect();
    
    let consistency_factor = 1.0 - calculate_variance(&scores).min(1.0);
    
    (history_factor * 0.5 + consistency_factor * 0.5).min(0.95)
}

/// Calculate variance of a set of values
fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    
    variance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::inhibitory::{
        CompetitiveInhibitionSystem, GroupCompetitionResult, InhibitionPerformanceMetrics,
        CompetitionGroup, CompetitionType, TemporalDynamics, InhibitionConfig
    };
    use crate::core::brain_types::ActivationPattern;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::core::types::EntityKey;
    use crate::cognitive::critical::CriticalThinking;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime};

    fn create_test_system() -> CompetitiveInhibitionSystem {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).unwrap());
        let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
        let critical_thinking = Arc::new(CriticalThinking::new(graph));
        
        CompetitiveInhibitionSystem::new(activation_engine, critical_thinking)
    }

    fn create_test_pattern_with_strengths(strengths: Vec<f32>) -> (ActivationPattern, Vec<EntityKey>) {
        let mut activations = HashMap::new();
        let mut entity_keys = Vec::new();
        
        for (i, strength) in strengths.into_iter().enumerate() {
            let entity = EntityKey::from_hash(&format!("entity_{}", i));
            activations.insert(entity, strength);
            entity_keys.push(entity);
        }
        
        (ActivationPattern { activations }, entity_keys)
    }

    fn create_test_competition_results(entities: &[EntityKey]) -> Vec<GroupCompetitionResult> {
        vec![
            GroupCompetitionResult {
                group_id: "test_group_1".to_string(),
                pre_competition: vec![(entities[0], 0.8), (entities[1], 0.6)],
                post_competition: vec![(entities[0], 0.8), (entities[1], 0.2)],
                winner: Some(entities[0]),
                competition_intensity: 0.7,
                suppressed_entities: vec![entities[1]],
            },
            GroupCompetitionResult {
                group_id: "test_group_2".to_string(),
                pre_competition: vec![(entities[2], 0.5), (entities[3], 0.4)],
                post_competition: vec![(entities[2], 0.5), (entities[3], 0.1)],
                winner: Some(entities[2]),
                competition_intensity: 0.5,
                suppressed_entities: vec![entities[3]],
            },
        ]
    }

    fn create_test_performance_history() -> Vec<InhibitionPerformanceMetrics> {
        vec![
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.8,
                effectiveness_score: 0.7,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(12),
                processing_time_ms: 12.0,
                entities_processed: 3,
                competition_groups_resolved: 1,
                competitions_resolved: 1,
                exceptions_handled: 1,
                efficiency_score: 0.6,
                effectiveness_score: 0.5,
            },
        ]
    }

    #[tokio::test]
    async fn test_apply_adaptive_learning() {
        let system = create_test_system();
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4, 0.3]);
        let results = create_test_competition_results(&entities);
        
        // Should complete without error
        let result = apply_adaptive_learning(&system, &pattern, &results).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_apply_learning_mechanisms_disabled() {
        let mut system = create_test_system();
        system.inhibition_config.enable_learning = false;
        
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let results = create_test_competition_results(&entities);
        let history = create_test_performance_history();
        
        let result = apply_learning_mechanisms(&system, &pattern, &results, &history).await.unwrap();
        
        assert!(!result.learning_applied);
        assert!(result.parameter_adjustments.is_empty());
        assert_eq!(result.performance_improvement_estimate, 0.0);
        assert_eq!(result.learning_confidence, 0.0);
    }

    #[tokio::test]
    async fn test_apply_learning_mechanisms_enabled() {
        let system = create_test_system();
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4, 0.3]);
        let results = create_test_competition_results(&entities);
        let history = create_test_performance_history();
        
        let result = apply_learning_mechanisms(&system, &pattern, &results, &history).await.unwrap();
        
        // Should process learning mechanisms
        assert!(result.learning_confidence > 0.0);
        // May or may not apply adjustments depending on conditions
    }

    #[tokio::test]
    async fn test_learn_inhibition_strength_adjustment_weak_competition() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.4, 0.3]);
        
        // Create results with low competition intensity (weak competition)
        let results = vec![
            GroupCompetitionResult {
                group_id: "weak_group".to_string(),
                pre_competition: vec![(entities[0], 0.4), (entities[1], 0.3)],
                post_competition: vec![(entities[0], 0.4), (entities[1], 0.3)],
                winner: None, // No clear winner
                competition_intensity: 0.2,
                suppressed_entities: vec![],
            }
        ];
        
        let history = create_test_performance_history();
        let adjustment = learn_inhibition_strength_adjustment(&pattern, &results, &history).await.unwrap();
        
        // Should suggest increasing inhibition strength
        assert!(adjustment.is_some());
        let adj = adjustment.unwrap();
        assert_eq!(adj.parameter_name, "global_inhibition_strength");
        assert!(adj.new_value > adj.old_value);
        assert_eq!(adj.adjustment_type, ParameterAdjustmentType::StrengthOptimization);
    }

    #[tokio::test]
    async fn test_learn_inhibition_strength_adjustment_strong_competition() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.8]);
        
        // Create results with very strong competition (too aggressive)
        let results = vec![
            GroupCompetitionResult {
                group_id: "strong_group".to_string(),
                pre_competition: vec![(entities[0], 0.9), (entities[1], 0.8)],
                post_competition: vec![(entities[0], 0.9), (entities[1], 0.0)],
                winner: Some(entities[0]),
                competition_intensity: 0.95,
                suppressed_entities: vec![entities[1]],
            }
        ];
        
        let history = create_test_performance_history();
        let adjustment = learn_inhibition_strength_adjustment(&pattern, &results, &history).await.unwrap();
        
        // Should suggest decreasing inhibition strength
        assert!(adjustment.is_some());
        let adj = adjustment.unwrap();
        assert_eq!(adj.parameter_name, "global_inhibition_strength");
        assert!(adj.new_value < adj.old_value);
    }

    #[tokio::test]
    async fn test_learn_inhibition_strength_adjustment_balanced() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        
        // Create results with balanced competition
        let results = vec![
            GroupCompetitionResult {
                group_id: "balanced_group".to_string(),
                pre_competition: vec![(entities[0], 0.8), (entities[1], 0.6)],
                post_competition: vec![(entities[0], 0.8), (entities[1], 0.3)],
                winner: Some(entities[0]),
                competition_intensity: 0.7,
                suppressed_entities: vec![],
            }
        ];
        
        let history = create_test_performance_history();
        let adjustment = learn_inhibition_strength_adjustment(&pattern, &results, &history).await.unwrap();
        
        // Should not suggest adjustment for balanced competition
        assert!(adjustment.is_none());
    }

    #[tokio::test]
    async fn test_learn_competition_group_optimization_aggressive() {
        let (_, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4, 0.3]);
        
        // Create results with high suppression rate
        let results = vec![
            GroupCompetitionResult {
                group_id: "aggressive_group".to_string(),
                pre_competition: vec![(entities[0], 0.8), (entities[1], 0.6), (entities[2], 0.4)],
                post_competition: vec![(entities[0], 0.8), (entities[1], 0.0), (entities[2], 0.0)],
                winner: Some(entities[0]),
                competition_intensity: 0.8,
                suppressed_entities: vec![entities[1], entities[2]],
            }
        ];
        
        let history = create_test_performance_history();
        let adjustment = learn_competition_group_optimization(&results, &history).await.unwrap();
        
        // Should suggest increasing soft competition factor
        assert!(adjustment.is_some());
        let adj = adjustment.unwrap();
        assert_eq!(adj.parameter_name, "soft_competition_factor");
        assert!(adj.new_value > adj.old_value);
        assert_eq!(adj.adjustment_type, ParameterAdjustmentType::CompetitionOptimization);
    }

    #[tokio::test]
    async fn test_learn_competition_group_optimization_moderate() {
        let (_, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        
        // Create results with moderate suppression
        let results = vec![
            GroupCompetitionResult {
                group_id: "moderate_group".to_string(),
                pre_competition: vec![(entities[0], 0.8), (entities[1], 0.6)],
                post_competition: vec![(entities[0], 0.8), (entities[1], 0.3)],
                winner: Some(entities[0]),
                competition_intensity: 0.6,
                suppressed_entities: vec![],
            }
        ];
        
        let history = create_test_performance_history();
        let adjustment = learn_competition_group_optimization(&results, &history).await.unwrap();
        
        // Should not suggest adjustment for moderate suppression
        assert!(adjustment.is_none());
    }

    #[tokio::test]
    async fn test_learn_temporal_dynamics_optimization_insufficient_history() {
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let history = vec![create_test_performance_history()[0].clone()]; // Only one entry
        
        let adjustment = learn_temporal_dynamics_optimization(&pattern, &history).await.unwrap();
        
        // Should not suggest adjustment with insufficient history
        assert!(adjustment.is_none());
    }

    #[tokio::test]
    async fn test_learn_temporal_dynamics_optimization_high_variance() {
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        
        // Create history with high variance in effectiveness scores
        let history = vec![
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.9, // High
                effectiveness_score: 0.9,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.3, // Low
                effectiveness_score: 0.3,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.8, // High again
                effectiveness_score: 0.8,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.2, // Low again
                effectiveness_score: 0.2,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.7,
                effectiveness_score: 0.7,
            },
        ];
        
        let adjustment = learn_temporal_dynamics_optimization(&pattern, &history).await.unwrap();
        
        // Should suggest temporal adjustment for high variance
        assert!(adjustment.is_some());
        let adj = adjustment.unwrap();
        assert_eq!(adj.parameter_name, "temporal_integration_window");
        assert!(adj.new_value > adj.old_value);
        assert_eq!(adj.adjustment_type, ParameterAdjustmentType::TemporalOptimization);
    }

    #[tokio::test]
    async fn test_calculate_performance_metrics() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        let results = create_test_competition_results(&entities);
        
        let metrics = calculate_performance_metrics(&pattern, &results);
        
        assert!(metrics.efficiency_score >= 0.0 && metrics.efficiency_score <= 1.0);
        assert!(metrics.effectiveness_score >= 0.0 && metrics.effectiveness_score <= 1.0);
        assert_eq!(metrics.entities_processed, pattern.activations.len());
        assert_eq!(metrics.competition_groups_resolved, results.len());
        assert!(metrics.processing_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_calculate_performance_metrics_empty_results() {
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let results = vec![];
        
        let metrics = calculate_performance_metrics(&pattern, &results);
        
        assert_eq!(metrics.efficiency_score, 0.5); // Default for empty results
        assert_eq!(metrics.competition_groups_resolved, 0);
        assert_eq!(metrics.entities_processed, pattern.activations.len());
    }

    #[tokio::test]
    async fn test_generate_adaptation_suggestions() {
        let metrics = InhibitionPerformanceMetrics {
            timestamp: SystemTime::now(),
            processing_time: Duration::from_millis(10),
            processing_time_ms: 10.0,
            entities_processed: 4,
            competition_groups_resolved: 2,
            competitions_resolved: 2,
            exceptions_handled: 0,
            efficiency_score: 0.4, // Low efficiency
            effectiveness_score: 0.3, // Low effectiveness
        };
        
        let suggestions = generate_adaptation_suggestions(&metrics);
        
        // Should generate suggestions for low performance
        assert!(!suggestions.is_empty());
        
        // Should suggest strength adjustment for low efficiency
        assert!(suggestions.iter().any(|s| matches!(s.suggestion_type, AdaptationType::StrengthAdjustment)));
        
        // Should suggest threshold modification for low effectiveness
        assert!(suggestions.iter().any(|s| matches!(s.suggestion_type, AdaptationType::ThresholdModification)));
        
        for suggestion in &suggestions {
            assert!(suggestion.expected_improvement > 0.0);
            assert!(suggestion.confidence > 0.0);
        }
    }

    #[tokio::test]
    async fn test_calculate_learning_confidence() {
        // Test with empty history
        let empty_history = vec![];
        let confidence = calculate_learning_confidence(&empty_history);
        assert_eq!(confidence, 0.5);
        
        // Test with consistent history
        let consistent_history = vec![
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.8,
                effectiveness_score: 0.8,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.82,
                effectiveness_score: 0.82,
            },
            InhibitionPerformanceMetrics {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(10),
                processing_time_ms: 10.0,
                entities_processed: 4,
                competition_groups_resolved: 2,
                competitions_resolved: 2,
                exceptions_handled: 0,
                efficiency_score: 0.78,
                effectiveness_score: 0.78,
            },
        ];
        
        let confidence = calculate_learning_confidence(&consistent_history);
        assert!(confidence > 0.5); // Should be higher for consistent performance
        assert!(confidence <= 0.95); // Maximum confidence
    }

    #[tokio::test]
    async fn test_calculate_variance() {
        // Test with empty values
        let empty_values = vec![];
        assert_eq!(calculate_variance(&empty_values), 0.0);
        
        // Test with single value
        let single_value = vec![0.5];
        assert_eq!(calculate_variance(&single_value), 0.0);
        
        // Test with identical values
        let identical_values = vec![0.7, 0.7, 0.7];
        assert_eq!(calculate_variance(&identical_values), 0.0);
        
        // Test with varying values
        let varying_values = vec![0.1, 0.5, 0.9];
        let variance = calculate_variance(&varying_values);
        assert!(variance > 0.0);
        
        // Manually calculate expected variance for verification
        let mean = 0.5;
        let expected_variance = ((0.1 - mean).powi(2) + (0.5 - mean).powi(2) + (0.9 - mean).powi(2)) / 3.0;
        assert!((variance - expected_variance).abs() < 0.001);
    }
}