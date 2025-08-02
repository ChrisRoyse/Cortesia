//! Temporal dynamics for inhibition

use crate::cognitive::inhibitory::{GroupCompetitionResult, InhibitionConfig, TemporalDynamics};
use crate::core::brain_types::ActivationPattern;
use crate::error::Result;
use std::time::Duration;

/// Apply temporal dynamics to the activation pattern
pub async fn apply_temporal_dynamics(
    pattern: &mut ActivationPattern,
    competition_results: &[GroupCompetitionResult],
    config: &InhibitionConfig,
) -> Result<()> {
    // Apply temporal modulation based on competition results
    for result in competition_results {
        if let Some(winner) = result.winner {
            // Apply temporal boost to winner
            if let Some(strength) = pattern.activations.get_mut(&winner) {
                let temporal_factor = calculate_temporal_factor(
                    &TemporalDynamics::default(),
                    Duration::from_millis(0), // Current time offset
                );
                *strength = (*strength * temporal_factor).min(1.0);
            }
        }
        
        // Apply temporal decay to suppressed entities
        for entity in &result.suppressed_entities {
            if let Some(strength) = pattern.activations.get_mut(entity) {
                let decay_factor = calculate_decay_factor(config.temporal_integration_window);
                *strength *= decay_factor;
            }
        }
    }
    
    Ok(())
}

/// Calculate temporal modulation factor
pub fn calculate_temporal_factor(dynamics: &TemporalDynamics, elapsed: Duration) -> f32 {
    let elapsed_ms = elapsed.as_millis() as f32;
    let onset_ms = dynamics.onset_delay.as_millis() as f32;
    let peak_ms = dynamics.peak_time.as_millis() as f32;
    let decay_ms = dynamics.decay_time.as_millis() as f32;
    
    if elapsed_ms < onset_ms {
        // Before onset
        0.0
    } else if elapsed_ms < peak_ms {
        // Rising phase
        
        (elapsed_ms - onset_ms) / (peak_ms - onset_ms)
    } else if elapsed_ms < decay_ms {
        // Decay phase
        let progress = (elapsed_ms - peak_ms) / (decay_ms - peak_ms);
        1.0 - (progress * 0.7) // Decay to 30% of peak
    } else {
        // After decay
        0.3
    }
}

/// Calculate decay factor for temporal integration
fn calculate_decay_factor(integration_window: Duration) -> f32 {
    // Simple exponential decay
    let window_ms = integration_window.as_millis() as f32;
    let decay_rate = 1.0 / window_ms;
    
    (-decay_rate * 10.0).exp() // Assuming 10ms time step
}

/// Apply oscillatory dynamics if configured
pub fn apply_oscillation(
    strength: f32,
    frequency: f32,
    elapsed: Duration,
) -> f32 {
    let elapsed_secs = elapsed.as_secs_f32();
    let phase = elapsed_secs * frequency * 2.0 * std::f32::consts::PI;
    
    // Modulate strength with oscillation
    strength * (0.8 + 0.2 * phase.sin())
}