//! Performance metrics and analysis for inhibition

use crate::cognitive::inhibitory::{InhibitionPerformanceMetrics, GroupCompetitionResult};
use crate::core::brain_types::ActivationPattern;
use std::time::{Duration, SystemTime};

/// Calculate comprehensive performance metrics
pub fn calculate_comprehensive_metrics(
    pattern: &ActivationPattern,
    competition_results: &[GroupCompetitionResult],
    processing_time: Duration,
) -> InhibitionPerformanceMetrics {
    let efficiency_score = calculate_efficiency_score(competition_results);
    let effectiveness_score = calculate_effectiveness_score(pattern, competition_results);
    
    InhibitionPerformanceMetrics {
        timestamp: SystemTime::now(),
        processing_time,
        processing_time_ms: processing_time.as_millis() as f64,
        entities_processed: pattern.activations.len(),
        competition_groups_resolved: competition_results.len(),
        competitions_resolved: competition_results.iter()
            .filter(|r| r.winner.is_some())
            .count(),
        exceptions_handled: 0, // Would be passed in from exception handler
        efficiency_score,
        effectiveness_score,
    }
}

/// Calculate efficiency score based on competition resolution
pub fn calculate_efficiency_score(competition_results: &[GroupCompetitionResult]) -> f32 {
    if competition_results.is_empty() {
        return 0.5;
    }
    
    let total_score: f32 = competition_results.iter()
        .map(|result| {
            if result.pre_competition.is_empty() {
                0.5
            } else {
                // Efficiency is higher when:
                // 1. There's a clear winner
                // 2. Competition intensity is appropriate
                // 3. Not too many entities are completely suppressed
                
                let winner_score = if result.winner.is_some() { 1.0 } else { 0.3 };
                
                let suppression_ratio = result.suppressed_entities.len() as f32 
                    / result.pre_competition.len().max(1) as f32;
                let suppression_score = 1.0 - (suppression_ratio - 0.5).abs() * 2.0; // Optimal around 50%
                
                let intensity_score = if result.competition_intensity > 0.9 {
                    0.7 // Too intense
                } else if result.competition_intensity < 0.3 {
                    0.5 // Too weak
                } else {
                    1.0 // Just right
                };
                
                winner_score * 0.4 + suppression_score * 0.3 + intensity_score * 0.3
            }
        })
        .sum();
    
    total_score / competition_results.len() as f32
}

/// Calculate effectiveness score based on final activation pattern
pub fn calculate_effectiveness_score(
    pattern: &ActivationPattern,
    competition_results: &[GroupCompetitionResult],
) -> f32 {
    // Effectiveness measures how well the inhibition achieved its goals:
    // 1. Appropriate sparsity (not too dense, not too sparse)
    // 2. Clear differentiation between active entities
    // 3. Preservation of important information
    
    let active_count = pattern.activations.values()
        .filter(|&&v| v > 0.1)
        .count();
    
    let total_entities = pattern.activations.len().max(1);
    let sparsity = active_count as f32 / total_entities as f32;
    
    // Optimal sparsity around 20-40%
    let sparsity_score = if sparsity < 0.2 {
        sparsity * 5.0 // Too sparse
    } else if sparsity > 0.4 {
        1.0 - (sparsity - 0.4) * 2.0 // Too dense
    } else {
        1.0 // Optimal
    };
    
    // Calculate activation variance (higher variance = better differentiation)
    let activations: Vec<f32> = pattern.activations.values().copied().collect();
    let variance = calculate_variance(&activations);
    let differentiation_score = (variance * 4.0).min(1.0); // Normalize to 0-1
    
    // Information preservation (at least some strong activations remain)
    let max_activation = activations.iter().fold(0.0f32, |a, &b| a.max(b));
    let preservation_score = max_activation;
    
    sparsity_score * 0.4 + differentiation_score * 0.3 + preservation_score * 0.3
}

/// Calculate variance of activation values
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

/// Analyze metric trends over time
pub fn analyze_metric_trends(
    history: &[InhibitionPerformanceMetrics],
    window_size: usize,
) -> MetricTrends {
    if history.len() < window_size {
        return MetricTrends::default();
    }
    
    let recent = &history[history.len() - window_size..];
    
    let avg_efficiency = recent.iter()
        .map(|m| m.efficiency_score)
        .sum::<f32>() / window_size as f32;
    
    let avg_effectiveness = recent.iter()
        .map(|m| m.effectiveness_score)
        .sum::<f32>() / window_size as f32;
    
    let avg_processing_time = recent.iter()
        .map(|m| m.processing_time_ms)
        .sum::<f64>() / window_size as f64;
    
    // Calculate trends (positive = improving)
    let efficiency_trend = if history.len() > window_size * 2 {
        let older_avg = history[history.len() - window_size * 2..history.len() - window_size]
            .iter()
            .map(|m| m.efficiency_score)
            .sum::<f32>() / window_size as f32;
        avg_efficiency - older_avg
    } else {
        0.0
    };
    
    MetricTrends {
        avg_efficiency,
        avg_effectiveness,
        avg_processing_time_ms: avg_processing_time,
        efficiency_trend,
        effectiveness_trend: 0.0, // Similar calculation as efficiency_trend
        processing_time_trend: 0.0,
    }
}

#[derive(Debug, Default)]
pub struct MetricTrends {
    pub avg_efficiency: f32,
    pub avg_effectiveness: f32,
    pub avg_processing_time_ms: f64,
    pub efficiency_trend: f32,
    pub effectiveness_trend: f32,
    pub processing_time_trend: f64,
}