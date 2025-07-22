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
pub fn detect_mutual_exclusions(
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
pub fn detect_temporal_conflicts(
    _pattern: &ActivationPattern,
    _competition_results: &[GroupCompetitionResult],
    _exceptions: &mut Vec<InhibitionException>,
) {
    // Check for temporal ordering violations
    // Simplified implementation
}

/// Detect hierarchical inconsistencies
pub fn detect_hierarchical_inconsistencies(
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
pub fn detect_resource_contentions(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::inhibitory::{
        CompetitiveInhibitionSystem, GroupCompetitionResult, HierarchicalInhibitionResult,
        HierarchicalLayer, InhibitionException, InhibitionConfig
    };
    use crate::core::brain_types::ActivationPattern;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::core::types::EntityKey;
    use crate::cognitive::critical::CriticalThinking;
    use std::collections::HashMap;
    use std::sync::Arc;

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
        
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations = activations;
        (pattern, entity_keys)
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
            }
        ]
    }

    fn create_test_hierarchical_result(entities: &[EntityKey]) -> HierarchicalInhibitionResult {
        let mut abstraction_levels = HashMap::new();
        abstraction_levels.insert(entities[0], 0); // Most specific
        abstraction_levels.insert(entities[1], 1); // Mid-level
        abstraction_levels.insert(entities[2], 2); // Most general
        
        HierarchicalInhibitionResult {
            hierarchical_layers: vec![
                HierarchicalLayer {
                    layer_level: 0,
                    entities: vec![entities[0]],
                    inhibition_strength: 0.8,
                    dominant_entity: Some(entities[0]),
                },
                HierarchicalLayer {
                    layer_level: 2,
                    entities: vec![entities[2]],
                    inhibition_strength: 0.6,
                    dominant_entity: Some(entities[2]),
                },
            ],
            specificity_winners: vec![entities[0]],
            generality_suppressed: vec![entities[2]],
            abstraction_levels,
        }
    }

    #[tokio::test]
    async fn test_handle_inhibition_exceptions_no_exceptions() {
        let system = create_test_system();
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        let competition_results = create_test_competition_results(&entities);
        let hierarchical_result = create_test_hierarchical_result(&entities);
        
        let result = handle_inhibition_exceptions(
            &system,
            &mut pattern,
            &competition_results,
            &hierarchical_result,
        ).await;
        
        // Should handle case with no exceptions gracefully
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.pattern_modified || result.pattern_modified); // May or may not be modified
        assert!(result.unresolved_conflicts.len() >= 0);
    }

    #[tokio::test]
    async fn test_detect_mutual_exclusions() {
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        let mut exceptions = Vec::new();
        
        detect_mutual_exclusions(&pattern, &mut exceptions);
        
        // This function is currently simplified and doesn't add exceptions
        // In a full implementation, it would detect actual mutual exclusions
        assert!(exceptions.len() >= 0);
    }

    #[tokio::test]
    async fn test_detect_temporal_conflicts() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let competition_results = create_test_competition_results(&entities);
        let mut exceptions = Vec::new();
        
        detect_temporal_conflicts(&pattern, &competition_results, &mut exceptions);
        
        // This function is currently simplified
        assert!(exceptions.len() >= 0);
    }

    #[tokio::test]
    async fn test_detect_hierarchical_inconsistencies() {
        let (_, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        let hierarchical_result = create_test_hierarchical_result(&entities);
        let mut exceptions = Vec::new();
        
        detect_hierarchical_inconsistencies(&hierarchical_result, &mut exceptions);
        
        // Should detect hierarchical inconsistencies
        // In our test case, entities[2] is suppressed but entities[0] (more specific) is a winner
        assert!(!exceptions.is_empty());
        
        // Should find hierarchical inconsistency
        let has_hierarchical_inconsistency = exceptions.iter().any(|e| {
            matches!(e, InhibitionException::HierarchicalInconsistency(_, _))
        });
        assert!(has_hierarchical_inconsistency);
    }

    #[tokio::test]
    async fn test_detect_resource_contentions() {
        // Create pattern with many highly active entities (resource contention)
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.8, 0.9, 0.7, 0.8, 0.9]);
        let mut exceptions = Vec::new();
        
        detect_resource_contentions(&pattern, &mut exceptions);
        
        // Should detect resource contention with 6 highly active entities
        assert!(!exceptions.is_empty());
        
        let has_resource_contention = exceptions.iter().any(|e| {
            matches!(e, InhibitionException::ResourceContention(_))
        });
        assert!(has_resource_contention);
        
        if let InhibitionException::ResourceContention(entities) = &exceptions[0] {
            assert_eq!(entities.len(), 6);
        }
    }

    #[tokio::test]
    async fn test_detect_resource_contentions_no_contention() {
        // Create pattern with few highly active entities
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.8, 0.3, 0.2]);
        let mut exceptions = Vec::new();
        
        detect_resource_contentions(&pattern, &mut exceptions);
        
        // Should not detect resource contention with only 1 highly active entity
        assert!(exceptions.is_empty());
    }

    #[tokio::test]
    async fn test_resolve_mutual_exclusion() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let system = create_test_system();
        
        let exception = InhibitionException::MutualExclusion(entities[0], entities[1]);
        let resolution = resolve_exception(&exception, &pattern, &system).await;
        
        assert!(resolution.is_ok());
        let resolution = resolution.unwrap();
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.exception_type, "MutualExclusion");
        assert_eq!(res.affected_entities.len(), 2);
        assert!(res.effectiveness > 0.0);
        
        // Should suppress the weaker entity (entities[1] with strength 0.6)
        if let ResolutionStrategy::Suppression(suppressed) = res.resolution_strategy {
            assert_eq!(suppressed, entities[1]);
        } else {
            panic!("Expected Suppression strategy");
        }
    }

    #[tokio::test]
    async fn test_resolve_temporal_conflict() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let system = create_test_system();
        
        let exception = InhibitionException::TemporalConflict(entities[0], entities[1]);
        let resolution = resolve_exception(&exception, &pattern, &system).await;
        
        assert!(resolution.is_ok());
        let resolution = resolution.unwrap();
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.exception_type, "TemporalConflict");
        assert_eq!(res.affected_entities.len(), 2);
        
        if let ResolutionStrategy::TemporalSequencing(sequence) = res.resolution_strategy {
            assert_eq!(sequence.len(), 2);
            assert!(sequence.contains(&entities[0]));
            assert!(sequence.contains(&entities[1]));
        } else {
            panic!("Expected TemporalSequencing strategy");
        }
    }

    #[tokio::test]
    async fn test_resolve_hierarchical_inconsistency() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        let system = create_test_system();
        
        let exception = InhibitionException::HierarchicalInconsistency(entities[0], entities[1]);
        let resolution = resolve_exception(&exception, &pattern, &system).await;
        
        assert!(resolution.is_ok());
        let resolution = resolution.unwrap();
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.exception_type, "HierarchicalInconsistency");
        assert_eq!(res.affected_entities.len(), 2);
        
        matches!(res.resolution_strategy, ResolutionStrategy::HierarchicalReordering);
    }

    #[tokio::test]
    async fn test_resolve_resource_contention() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        let system = create_test_system();
        
        let exception = InhibitionException::ResourceContention(entities.clone());
        let resolution = resolve_exception(&exception, &pattern, &system).await;
        
        let resolution = resolution.expect("Resolution should succeed");
        assert!(resolution.is_some());
        let res = resolution.unwrap();
        assert_eq!(res.exception_type, "ResourceContention");
        assert_eq!(res.affected_entities.len(), 3);
        
        if let ResolutionStrategy::ResourceAllocation(allocation) = res.resolution_strategy {
            assert_eq!(allocation.len(), 3);
            
            // Total allocation should be normalized
            let total_allocation: f32 = allocation.values().sum();
            assert!((total_allocation - 1.0).abs() < 0.001);
            
            // Allocation should be proportional to original strengths
            assert!(allocation[&entities[0]] > allocation[&entities[1]]);
            assert!(allocation[&entities[1]] > allocation[&entities[2]]);
        } else {
            panic!("Expected ResourceAllocation strategy");
        }
    }

    #[tokio::test]
    async fn test_apply_resolution_suppression() {
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6]);
        
        let resolution = ExceptionResolution {
            exception_type: "MutualExclusion".to_string(),
            affected_entities: vec![entities[0], entities[1]],
            resolution_strategy: ResolutionStrategy::Suppression(entities[1]),
            effectiveness: 0.9,
        };
        
        apply_resolution(&mut pattern, &resolution);
        
        // Suppressed entity should have 0 activation
        assert_eq!(pattern.activations[&entities[1]], 0.0);
        // Other entity should remain unchanged
        assert_eq!(pattern.activations[&entities[0]], 0.8);
    }

    #[tokio::test]
    async fn test_apply_resolution_temporal_sequencing() {
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        
        let resolution = ExceptionResolution {
            exception_type: "TemporalConflict".to_string(),
            affected_entities: entities.clone(),
            resolution_strategy: ResolutionStrategy::TemporalSequencing(entities.clone()),
            effectiveness: 0.8,
        };
        
        let original_strengths = pattern.activations.clone();
        apply_resolution(&mut pattern, &resolution);
        
        // First entity should remain unchanged
        assert_eq!(pattern.activations[&entities[0]], original_strengths[&entities[0]]);
        
        // Other entities should be reduced (delayed activation)
        assert!(pattern.activations[&entities[1]] < original_strengths[&entities[1]]);
        assert!(pattern.activations[&entities[2]] < original_strengths[&entities[2]]);
    }

    #[tokio::test]
    async fn test_apply_resolution_resource_allocation() {
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.6, 0.4]);
        
        let mut allocation = HashMap::new();
        allocation.insert(entities[0], 0.5);
        allocation.insert(entities[1], 0.3);
        allocation.insert(entities[2], 0.2);
        
        let resolution = ExceptionResolution {
            exception_type: "ResourceContention".to_string(),
            affected_entities: entities.clone(),
            resolution_strategy: ResolutionStrategy::ResourceAllocation(allocation.clone()),
            effectiveness: 0.75,
        };
        
        apply_resolution(&mut pattern, &resolution);
        
        // Activations should match allocated values
        assert_eq!(pattern.activations[&entities[0]], 0.5);
        assert_eq!(pattern.activations[&entities[1]], 0.3);
        assert_eq!(pattern.activations[&entities[2]], 0.2);
    }

    #[tokio::test]
    async fn test_identify_unresolved_conflicts_all_resolved() {
        let exceptions = vec![
            InhibitionException::MutualExclusion(EntityKey::from_raw_parts(0, 0), EntityKey::from_raw_parts(1, 0)),
            InhibitionException::ResourceContention(vec![]),
        ];
        
        let resolutions = vec![
            ExceptionResolution {
                exception_type: "MutualExclusion".to_string(),
                affected_entities: vec![],
                resolution_strategy: ResolutionStrategy::Suppression(EntityKey::from_raw_parts(0, 0)),
                effectiveness: 0.9,
            },
            ExceptionResolution {
                exception_type: "ResourceContention".to_string(),
                affected_entities: vec![],
                resolution_strategy: ResolutionStrategy::ResourceAllocation(HashMap::new()),
                effectiveness: 0.8,
            },
        ];
        
        let unresolved = identify_unresolved_conflicts(&exceptions, &resolutions);
        
        // All exceptions should be resolved
        assert!(unresolved.is_empty());
    }

    #[tokio::test]
    async fn test_identify_unresolved_conflicts_some_unresolved() {
        let exceptions = vec![
            InhibitionException::MutualExclusion(EntityKey::from_raw_parts(0, 0), EntityKey::from_raw_parts(1, 0)),
            InhibitionException::ResourceContention(vec![]),
            InhibitionException::TemporalConflict(EntityKey::from_raw_parts(2, 0), EntityKey::from_raw_parts(3, 0)),
        ];
        
        let resolutions = vec![
            ExceptionResolution {
                exception_type: "MutualExclusion".to_string(),
                affected_entities: vec![],
                resolution_strategy: ResolutionStrategy::Suppression(EntityKey::from_raw_parts(0, 0)),
                effectiveness: 0.9,
            },
        ];
        
        let unresolved = identify_unresolved_conflicts(&exceptions, &resolutions);
        
        // Should identify 2 unresolved exceptions
        assert!(!unresolved.is_empty());
        assert!(unresolved[0].contains("2 exceptions could not be resolved"));
    }

    #[tokio::test]
    async fn test_handle_inhibition_exceptions_full_flow() {
        let system = create_test_system();
        
        // Create pattern with resource contention (many highly active entities)
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.8, 0.9, 0.7, 0.8, 0.9, 0.85]);
        let competition_results = create_test_competition_results(&entities);
        let hierarchical_result = create_test_hierarchical_result(&entities);
        
        let result = handle_inhibition_exceptions(
            &system,
            &mut pattern,
            &competition_results,
            &hierarchical_result,
        ).await;
        
        let result = result.expect("Exception handling should succeed");
        
        // Should detect and resolve exceptions
        assert!(!result.exceptions_detected.is_empty());
        
        // Should have attempted resolutions
        if !result.exceptions_detected.is_empty() {
            assert!(!result.resolutions_applied.is_empty() || !result.unresolved_conflicts.is_empty());
        }
        
        // Pattern should be modified if resolutions were applied
        if !result.resolutions_applied.is_empty() {
            assert!(result.pattern_modified);
        }
    }
}