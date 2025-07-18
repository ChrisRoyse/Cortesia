//! Exception handling for inhibition conflicts

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, ExceptionHandlingResult, InhibitionException,
    ExceptionResolution, ResolutionStrategy, GroupCompetitionResult,
    HierarchicalInhibitionResult
};
use crate::core::brain_types::ActivationPattern;
use crate::error::Result;
use std::collections::HashMap;

/// Handle special cases and exceptions in inhibition
pub async fn handle_inhibition_exceptions(
    system: &CompetitiveInhibitionSystem,
    pattern: &mut ActivationPattern,
    competition_results: &[GroupCompetitionResult],
    hierarchical_result: &HierarchicalInhibitionResult,
) -> Result<ExceptionHandlingResult> {
    let mut exceptions = Vec::new();
    let mut resolutions = Vec::new();
    
    // Detect various types of exceptions
    detect_mutual_exclusions(pattern, &mut exceptions);
    detect_temporal_conflicts(pattern, competition_results, &mut exceptions);
    detect_hierarchical_inconsistencies(hierarchical_result, &mut exceptions);
    detect_resource_contentions(pattern, &mut exceptions);
    
    // Resolve detected exceptions
    let mut pattern_modified = false;
    for exception in &exceptions {
        if let Some(resolution) = resolve_exception(exception, pattern, system).await? {
            apply_resolution(pattern, &resolution);
            resolutions.push(resolution);
            pattern_modified = true;
        }
    }
    
    // Identify unresolved conflicts
    let unresolved_conflicts = identify_unresolved_conflicts(&exceptions, &resolutions);
    
    Ok(ExceptionHandlingResult {
        exceptions_detected: exceptions,
        resolutions_applied: resolutions,
        unresolved_conflicts,
        pattern_modified,
    })
}

/// Detect mutual exclusion violations
fn detect_mutual_exclusions(
    pattern: &ActivationPattern,
    _exceptions: &mut Vec<InhibitionException>,
) {
    // Simplified: check for known mutually exclusive pairs
    // In practice, this would use a knowledge base of exclusions
    let active_entities: Vec<_> = pattern.activations
        .iter()
        .filter(|(_, &strength)| strength > 0.3)
        .map(|(entity, _)| *entity)
        .collect();
    
    // Example: detect if contradictory concepts are both active
    for i in 0..active_entities.len() {
        for _j in (i + 1)..active_entities.len() {
            // This would check against actual mutual exclusion rules
            // For now, we'll skip actual detection
        }
    }
}

/// Detect temporal conflicts
fn detect_temporal_conflicts(
    _pattern: &ActivationPattern,
    _competition_results: &[GroupCompetitionResult],
    _exceptions: &mut Vec<InhibitionException>,
) {
    // Check for temporal ordering violations
    // Simplified implementation
}

/// Detect hierarchical inconsistencies
fn detect_hierarchical_inconsistencies(
    hierarchical_result: &HierarchicalInhibitionResult,
    exceptions: &mut Vec<InhibitionException>,
) {
    // Check if more specific concepts are suppressed while general ones are active
    for (&entity, &level) in &hierarchical_result.abstraction_levels {
        if hierarchical_result.generality_suppressed.contains(&entity) {
            // Check if a more specific entity is active
            for (&other_entity, &other_level) in &hierarchical_result.abstraction_levels {
                if other_level < level && hierarchical_result.specificity_winners.contains(&other_entity) {
                    exceptions.push(InhibitionException::HierarchicalInconsistency(entity, other_entity));
                }
            }
        }
    }
}

/// Detect resource contention
fn detect_resource_contentions(
    pattern: &ActivationPattern,
    exceptions: &mut Vec<InhibitionException>,
) {
    // Check if too many entities are highly active (resource limitation)
    let highly_active: Vec<_> = pattern.activations
        .iter()
        .filter(|(_, &strength)| strength > 0.7)
        .map(|(entity, _)| *entity)
        .collect();
    
    if highly_active.len() > 5 { // Arbitrary threshold
        exceptions.push(InhibitionException::ResourceContention(highly_active));
    }
}

/// Resolve a detected exception
async fn resolve_exception(
    exception: &InhibitionException,
    pattern: &ActivationPattern,
    _system: &CompetitiveInhibitionSystem,
) -> Result<Option<ExceptionResolution>> {
    match exception {
        InhibitionException::MutualExclusion(entity_a, entity_b) => {
            // Suppress the weaker of the two
            let strength_a = pattern.activations.get(entity_a).copied().unwrap_or(0.0);
            let strength_b = pattern.activations.get(entity_b).copied().unwrap_or(0.0);
            
            let to_suppress = if strength_a > strength_b { *entity_b } else { *entity_a };
            
            Ok(Some(ExceptionResolution {
                exception_type: "MutualExclusion".to_string(),
                affected_entities: vec![*entity_a, *entity_b],
                resolution_strategy: ResolutionStrategy::Suppression(to_suppress),
                effectiveness: 0.9,
            }))
        }
        
        InhibitionException::TemporalConflict(entity_a, entity_b) => {
            Ok(Some(ExceptionResolution {
                exception_type: "TemporalConflict".to_string(),
                affected_entities: vec![*entity_a, *entity_b],
                resolution_strategy: ResolutionStrategy::TemporalSequencing(vec![*entity_a, *entity_b]),
                effectiveness: 0.8,
            }))
        }
        
        InhibitionException::HierarchicalInconsistency(general, specific) => {
            Ok(Some(ExceptionResolution {
                exception_type: "HierarchicalInconsistency".to_string(),
                affected_entities: vec![*general, *specific],
                resolution_strategy: ResolutionStrategy::HierarchicalReordering,
                effectiveness: 0.85,
            }))
        }
        
        InhibitionException::ResourceContention(entities) => {
            // Allocate resources proportionally
            let total_strength: f32 = entities.iter()
                .map(|e| pattern.activations.get(e).copied().unwrap_or(0.0))
                .sum();
            
            let mut allocation = HashMap::new();
            for entity in entities {
                let strength = pattern.activations.get(entity).copied().unwrap_or(0.0);
                allocation.insert(*entity, strength / total_strength);
            }
            
            Ok(Some(ExceptionResolution {
                exception_type: "ResourceContention".to_string(),
                affected_entities: entities.clone(),
                resolution_strategy: ResolutionStrategy::ResourceAllocation(allocation),
                effectiveness: 0.75,
            }))
        }
        
        _ => Ok(None), // Other exception types not implemented yet
    }
}

/// Apply a resolution to the activation pattern
fn apply_resolution(pattern: &mut ActivationPattern, resolution: &ExceptionResolution) {
    match &resolution.resolution_strategy {
        ResolutionStrategy::Suppression(entity) => {
            pattern.activations.insert(*entity, 0.0);
        }
        
        ResolutionStrategy::TemporalSequencing(entities) => {
            // In a real implementation, this would set up temporal ordering
            // For now, we'll reduce all but the first entity
            for (i, entity) in entities.iter().enumerate() {
                if i > 0 {
                    if let Some(strength) = pattern.activations.get_mut(entity) {
                        *strength *= 0.3; // Delay activation
                    }
                }
            }
        }
        
        ResolutionStrategy::HierarchicalReordering => {
            // Would reorder based on hierarchy
        }
        
        ResolutionStrategy::ResourceAllocation(allocation) => {
            for (entity, allocated_strength) in allocation {
                pattern.activations.insert(*entity, *allocated_strength);
            }
        }
        
        _ => {} // Other strategies not implemented
    }
}

/// Identify conflicts that couldn't be resolved
fn identify_unresolved_conflicts(
    exceptions: &[InhibitionException],
    resolutions: &[ExceptionResolution],
) -> Vec<String> {
    let mut unresolved = Vec::new();
    
    let resolved_count = resolutions.len();
    let exception_count = exceptions.len();
    
    if resolved_count < exception_count {
        unresolved.push(format!(
            "{} exceptions could not be resolved",
            exception_count - resolved_count
        ));
    }
    
    unresolved
}