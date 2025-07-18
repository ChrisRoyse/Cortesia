//! Integration with cognitive patterns

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, IntegrationResult, PatternSpecificInhibition,
    InhibitionProfile
};
use crate::cognitive::types::CognitivePatternType;
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::collections::HashMap;

/// Integrate inhibition with cognitive patterns for pattern-specific modulation
pub async fn integrate_with_cognitive_patterns(
    system: &CompetitiveInhibitionSystem,
    pattern: &mut ActivationPattern,
    active_cognitive_patterns: &[CognitivePatternType],
) -> Result<IntegrationResult> {
    let mut pattern_specific_inhibitions = Vec::new();
    let mut cross_pattern_conflicts = Vec::new();
    
    // Apply pattern-specific inhibition for each active cognitive pattern
    for cognitive_pattern in active_cognitive_patterns {
        let inhibition_result = apply_pattern_specific_inhibition(
            pattern,
            *cognitive_pattern,
        ).await?;
        
        pattern_specific_inhibitions.push(inhibition_result);
    }
    
    // Check for conflicts between patterns
    detect_cross_pattern_conflicts(
        &pattern_specific_inhibitions,
        &mut cross_pattern_conflicts,
    );
    
    // Resolve conflicts if any
    let integration_success = cross_pattern_conflicts.is_empty();
    
    Ok(IntegrationResult {
        cognitive_patterns_involved: active_cognitive_patterns.to_vec(),
        pattern_specific_inhibitions,
        cross_pattern_conflicts,
        integration_success,
    })
}

/// Apply inhibition specific to a cognitive pattern type
async fn apply_pattern_specific_inhibition(
    pattern: &mut ActivationPattern,
    cognitive_pattern: CognitivePatternType,
) -> Result<PatternSpecificInhibition> {
    let inhibition_profile = get_inhibition_profile(cognitive_pattern);
    let mut affected_entities = Vec::new();
    
    match cognitive_pattern {
        CognitivePatternType::Convergent => {
            // Convergent thinking: strong lateral inhibition to focus on best solution
            apply_convergent_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Divergent => {
            // Divergent thinking: weak inhibition to allow multiple ideas
            apply_divergent_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Lateral => {
            // Lateral thinking: asymmetric inhibition for creative connections
            apply_lateral_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Critical => {
            // Critical thinking: targeted inhibition of weak arguments
            apply_critical_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Systems => {
            // Systems thinking: hierarchical inhibition preserving relationships
            apply_systems_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Abstract => {
            // Abstract thinking: inhibit concrete details, enhance patterns
            apply_abstract_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
        
        CognitivePatternType::Adaptive => {
            // Adaptive thinking: dynamic inhibition based on context
            apply_adaptive_inhibition(pattern, &inhibition_profile, &mut affected_entities);
        }
    }
    
    Ok(PatternSpecificInhibition {
        pattern_type: cognitive_pattern,
        inhibition_profile,
        entities_affected: affected_entities,
    })
}

/// Get the inhibition profile for a cognitive pattern
fn get_inhibition_profile(pattern_type: CognitivePatternType) -> InhibitionProfile {
    match pattern_type {
        CognitivePatternType::Convergent => InhibitionProfile {
            convergent_factor: 0.9,
            divergent_factor: 0.2,
            lateral_spread: 0.3,
            critical_threshold: 0.7,
        },
        
        CognitivePatternType::Divergent => InhibitionProfile {
            convergent_factor: 0.2,
            divergent_factor: 0.8,
            lateral_spread: 0.9,
            critical_threshold: 0.3,
        },
        
        CognitivePatternType::Lateral => InhibitionProfile {
            convergent_factor: 0.4,
            divergent_factor: 0.6,
            lateral_spread: 0.7,
            critical_threshold: 0.5,
        },
        
        CognitivePatternType::Critical => InhibitionProfile {
            convergent_factor: 0.7,
            divergent_factor: 0.3,
            lateral_spread: 0.4,
            critical_threshold: 0.8,
        },
        
        CognitivePatternType::Systems => InhibitionProfile {
            convergent_factor: 0.5,
            divergent_factor: 0.5,
            lateral_spread: 0.6,
            critical_threshold: 0.6,
        },
        
        CognitivePatternType::Abstract => InhibitionProfile {
            convergent_factor: 0.6,
            divergent_factor: 0.4,
            lateral_spread: 0.5,
            critical_threshold: 0.65,
        },
        
        CognitivePatternType::Adaptive => InhibitionProfile {
            convergent_factor: 0.5,
            divergent_factor: 0.5,
            lateral_spread: 0.5,
            critical_threshold: 0.5,
        },
    }
}

/// Apply convergent inhibition
fn apply_convergent_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Find the strongest activation
    if let Some((strongest_entity, max_strength)) = pattern.activations.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
        
        let threshold = max_strength * profile.critical_threshold;
        
        // Inhibit weaker activations
        for (entity, strength) in pattern.activations.iter_mut() {
            if *entity != *strongest_entity && *strength < threshold {
                *strength *= profile.convergent_factor;
                affected.push(*entity);
            }
        }
    }
}

/// Apply divergent inhibition
fn apply_divergent_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Minimal inhibition to preserve diversity
    for (entity, strength) in pattern.activations.iter_mut() {
        if *strength < profile.critical_threshold {
            *strength *= profile.divergent_factor;
            affected.push(*entity);
        }
    }
}

/// Apply lateral inhibition
fn apply_lateral_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Create lateral connections with asymmetric inhibition
    let entities: Vec<_> = pattern.activations.keys().copied().collect();
    
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let strength_i = pattern.activations.get(&entities[i]).copied().unwrap_or(0.0);
            let strength_j = pattern.activations.get(&entities[j]).copied().unwrap_or(0.0);
            
            // Asymmetric lateral inhibition
            if strength_i > strength_j * 1.5 {
                if let Some(strength) = pattern.activations.get_mut(&entities[j]) {
                    *strength *= profile.lateral_spread;
                    affected.push(entities[j]);
                }
            }
        }
    }
}

/// Apply critical inhibition
fn apply_critical_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Inhibit entities below critical threshold
    for (entity, strength) in pattern.activations.iter_mut() {
        if *strength < profile.critical_threshold {
            *strength *= 0.5; // Strong suppression of weak evidence
            affected.push(*entity);
        }
    }
}

/// Apply systems inhibition
fn apply_systems_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Preserve system relationships while inhibiting isolated nodes
    // Simplified: reduce activation of entities with few connections
    // In practice, would use actual graph connectivity
}

/// Apply abstract inhibition
fn apply_abstract_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Enhance abstract patterns while inhibiting concrete details
    // Simplified implementation
}

/// Apply adaptive inhibition
fn apply_adaptive_inhibition(
    pattern: &mut ActivationPattern,
    profile: &InhibitionProfile,
    affected: &mut Vec<EntityKey>,
) {
    // Dynamic inhibition based on current context
    // Simplified implementation
}

/// Detect conflicts between pattern-specific inhibitions
fn detect_cross_pattern_conflicts(
    inhibitions: &[PatternSpecificInhibition],
    conflicts: &mut Vec<String>,
) {
    // Check if different patterns are trying to inhibit/enhance the same entities differently
    let mut entity_patterns: HashMap<EntityKey, Vec<CognitivePatternType>> = HashMap::new();
    
    for inhibition in inhibitions {
        for entity in &inhibition.entities_affected {
            entity_patterns.entry(*entity)
                .or_insert_with(Vec::new)
                .push(inhibition.pattern_type);
        }
    }
    
    // Detect conflicts
    for (entity, patterns) in entity_patterns {
        if patterns.len() > 1 {
            // Check if patterns have conflicting profiles
            let profiles: Vec<_> = patterns.iter()
                .map(|p| get_inhibition_profile(*p))
                .collect();
            
            // Simple conflict detection: check if convergent factors differ significantly
            let max_convergent = profiles.iter().map(|p| p.convergent_factor).fold(0.0, f32::max);
            let min_convergent = profiles.iter().map(|p| p.convergent_factor).fold(1.0, f32::min);
            
            if max_convergent - min_convergent > 0.5 {
                conflicts.push(format!(
                    "Conflicting inhibition for entity {:?} between patterns {:?}",
                    entity, patterns
                ));
            }
        }
    }
}