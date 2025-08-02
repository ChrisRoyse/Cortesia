//! Competition strategies and group competition logic

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, CompetitionGroup, CompetitionType,
    GroupCompetitionResult, InhibitionConfig
};
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Apply group-based competition to the activation pattern
pub async fn apply_group_competition(
    _system: &CompetitiveInhibitionSystem,
    working_pattern: &mut ActivationPattern,
    competition_groups: &Arc<RwLock<Vec<CompetitionGroup>>>,
    config: &InhibitionConfig,
) -> Result<Vec<GroupCompetitionResult>> {
    let mut results = Vec::new();
    let groups = competition_groups.read().await;
    
    // Sort groups by priority
    let mut sorted_groups: Vec<_> = groups.iter().collect();
    sorted_groups.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
    
    for group in sorted_groups {
        let result = match group.competition_type {
            CompetitionType::Semantic => {
                apply_semantic_competition(working_pattern, group, config).await?
            }
            CompetitionType::Temporal => {
                apply_temporal_competition(working_pattern, group, config).await?
            }
            CompetitionType::Hierarchical => {
                apply_hierarchical_competition(working_pattern, group, config).await?
            }
            CompetitionType::Contextual => {
                apply_contextual_competition(working_pattern, group, config).await?
            }
            CompetitionType::Spatial => {
                apply_spatial_competition(working_pattern, group, config).await?
            }
            CompetitionType::Causal => {
                apply_causal_competition(working_pattern, group, config).await?
            }
        };
        
        results.push(result);
    }
    
    Ok(results)
}

/// Apply semantic competition within a group
async fn apply_semantic_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    let pre_competition: Vec<_> = group.competing_entities.iter()
        .filter_map(|&entity| {
            pattern.activations.get(&entity).map(|&strength| (entity, strength))
        })
        .collect();
    
    if pre_competition.is_empty() {
        return Ok(GroupCompetitionResult {
            group_id: group.group_id.clone(),
            pre_competition: vec![],
            post_competition: vec![],
            winner: None,
            competition_intensity: 0.0,
            suppressed_entities: vec![],
        });
    }
    
    // Find the strongest activation
    let (winner, max_strength) = pre_competition.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    
    let mut post_competition = Vec::new();
    let mut suppressed_entities = Vec::new();
    
    if group.winner_takes_all && *max_strength > config.winner_takes_all_threshold {
        // Winner takes all - suppress all others
        for (entity, _) in &pre_competition {
            if *entity == *winner {
                post_competition.push((*entity, *max_strength));
            } else {
                pattern.activations.insert(*entity, 0.0);
                suppressed_entities.push(*entity);
                post_competition.push((*entity, 0.0));
            }
        }
    } else {
        // Soft competition - reduce others based on winner's strength
        let inhibition_factor = max_strength * group.inhibition_strength * config.soft_competition_factor;
        
        for (entity, original_strength) in &pre_competition {
            if *entity == *winner {
                post_competition.push((*entity, *original_strength));
            } else {
                let new_strength = (original_strength - inhibition_factor).max(0.0);
                pattern.activations.insert(*entity, new_strength);
                post_competition.push((*entity, new_strength));
                
                if new_strength == 0.0 {
                    suppressed_entities.push(*entity);
                }
            }
        }
    }
    
    Ok(GroupCompetitionResult {
        group_id: group.group_id.clone(),
        pre_competition: pre_competition.clone(),
        post_competition,
        winner: Some(*winner),
        competition_intensity: max_strength * group.inhibition_strength,
        suppressed_entities,
    })
}

/// Apply temporal competition - entities compete based on temporal dynamics
async fn apply_temporal_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    _config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    let pre_competition: Vec<_> = group.competing_entities.iter()
        .filter_map(|&entity| {
            pattern.activations.get(&entity).map(|&strength| (entity, strength))
        })
        .collect();
    
    // For temporal competition, apply phase-based inhibition
    let mut post_competition = Vec::new();
    let mut suppressed_entities = Vec::new();
    
    // Simple temporal competition: alternate suppression
    for (i, (entity, original_strength)) in pre_competition.iter().enumerate() {
        let temporal_factor = if i % 2 == 0 { 1.0 } else { 0.3 };
        let new_strength = original_strength * temporal_factor;
        
        pattern.activations.insert(*entity, new_strength);
        post_competition.push((*entity, new_strength));
        
        if new_strength < 0.1 {
            suppressed_entities.push(*entity);
        }
    }
    
    let winner = post_competition.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(entity, _)| *entity);
    
    Ok(GroupCompetitionResult {
        group_id: group.group_id.clone(),
        pre_competition,
        post_competition,
        winner,
        competition_intensity: group.inhibition_strength,
        suppressed_entities,
    })
}

/// Apply hierarchical competition - higher level concepts inhibit lower level ones
async fn apply_hierarchical_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    let pre_competition: Vec<_> = group.competing_entities.iter()
        .filter_map(|&entity| {
            pattern.activations.get(&entity).map(|&strength| (entity, strength))
        })
        .collect();
    
    // In hierarchical competition, assume entities are ordered by abstraction level
    let mut post_competition = Vec::new();
    let mut suppressed_entities = Vec::new();
    
    // Higher-level entities (earlier in list) inhibit lower-level ones
    for (i, (entity, original_strength)) in pre_competition.iter().enumerate() {
        let mut new_strength = *original_strength;
        
        // Check inhibition from higher-level entities
        for j in 0..i {
            let (_, inhibitor_strength) = pre_competition[j];
            if inhibitor_strength > config.hierarchical_inhibition_strength {
                new_strength *= 1.0 - (inhibitor_strength * group.inhibition_strength);
            }
        }
        
        new_strength = new_strength.max(0.0);
        pattern.activations.insert(*entity, new_strength);
        post_competition.push((*entity, new_strength));
        
        if new_strength < 0.1 {
            suppressed_entities.push(*entity);
        }
    }
    
    let winner = post_competition.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(entity, _)| *entity);
    
    Ok(GroupCompetitionResult {
        group_id: group.group_id.clone(),
        pre_competition,
        post_competition,
        winner,
        competition_intensity: group.inhibition_strength * config.hierarchical_inhibition_strength,
        suppressed_entities,
    })
}

/// Apply contextual competition
async fn apply_contextual_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    // For now, use semantic competition logic
    // In a full implementation, this would consider context vectors
    apply_semantic_competition(pattern, group, config).await
}

/// Apply spatial competition
async fn apply_spatial_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    // For now, use semantic competition logic
    // In a full implementation, this would consider spatial relationships
    apply_semantic_competition(pattern, group, config).await
}

/// Apply causal competition
async fn apply_causal_competition(
    pattern: &mut ActivationPattern,
    group: &CompetitionGroup,
    config: &InhibitionConfig,
) -> Result<GroupCompetitionResult> {
    // For now, use semantic competition logic
    // In a full implementation, this would consider causal relationships
    apply_semantic_competition(pattern, group, config).await
}

/// Apply soft competition between entities
pub async fn apply_soft_competition(
    pattern: &mut ActivationPattern,
    entity_pairs: &[(EntityKey, EntityKey)],
    inhibition_strength: f32,
) -> Result<()> {
    for (entity_a, entity_b) in entity_pairs {
        let strength_a = pattern.activations.get(entity_a).copied().unwrap_or(0.0);
        let strength_b = pattern.activations.get(entity_b).copied().unwrap_or(0.0);
        
        // Mutual inhibition based on relative strengths
        if strength_a > strength_b {
            let inhibition = (strength_a - strength_b) * inhibition_strength;
            pattern.activations.insert(*entity_b, (strength_b - inhibition).max(0.0));
        } else if strength_b > strength_a {
            let inhibition = (strength_b - strength_a) * inhibition_strength;
            pattern.activations.insert(*entity_a, (strength_a - inhibition).max(0.0));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::ActivationPattern;
    use crate::core::types::EntityKey;
    use crate::cognitive::inhibitory::{CompetitionType, TemporalDynamics, InhibitionConfig};
    use std::collections::HashMap;

    fn create_test_pattern_with_entities(strengths: Vec<f32>) -> (ActivationPattern, Vec<EntityKey>) {
        let mut activations = HashMap::new();
        let mut entity_keys = Vec::new();
        
        for (i, strength) in strengths.into_iter().enumerate() {
            let entity = EntityKey::from_hash(&format!("entity_{i}"));
            activations.insert(entity, strength);
            entity_keys.push(entity);
        }
        
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations = activations;
        (pattern, entity_keys)
    }

    fn create_test_group(entities: Vec<EntityKey>, competition_type: CompetitionType, winner_takes_all: bool) -> CompetitionGroup {
        CompetitionGroup {
            group_id: "test_group".to_string(),
            competing_entities: entities,
            competition_type,
            winner_takes_all,
            inhibition_strength: 0.8,
            priority: 0.7,
            temporal_dynamics: TemporalDynamics::default(),
        }
    }

    #[tokio::test]
    async fn test_apply_semantic_competition_winner_takes_all() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.9, 0.6, 0.3]);
        let group = create_test_group(entities.clone(), CompetitionType::Semantic, true);
        let config = InhibitionConfig::default();
        
        let result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Winner should be entity with highest activation (0.9)
        assert_eq!(result.winner, Some(entities[0]));
        assert_eq!(result.suppressed_entities.len(), 2);
        
        // Winner should retain strength, others should be suppressed to 0
        assert_eq!(pattern.activations[&entities[0]], 0.9);
        assert_eq!(pattern.activations[&entities[1]], 0.0);
        assert_eq!(pattern.activations[&entities[2]], 0.0);
        
        assert!(result.competition_intensity > 0.0);
    }

    #[tokio::test]
    async fn test_apply_semantic_competition_soft_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.8, 0.6, 0.4]);
        let group = create_test_group(entities.clone(), CompetitionType::Semantic, false);
        let config = InhibitionConfig::default();
        
        let result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Winner should be entity with highest activation
        assert_eq!(result.winner, Some(entities[0]));
        
        // Winner retains strength, others are reduced but not completely suppressed
        assert_eq!(pattern.activations[&entities[0]], 0.8);
        assert!(pattern.activations[&entities[1]] < 0.6);
        assert!(pattern.activations[&entities[2]] < 0.4);
        assert!(pattern.activations[&entities[1]] > 0.0);
        assert!(pattern.activations[&entities[2]] > 0.0);
    }

    #[tokio::test]
    async fn test_apply_semantic_competition_empty_activations() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![]);
        let group = create_test_group(entities, CompetitionType::Semantic, true);
        let config = InhibitionConfig::default();
        
        let result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        assert_eq!(result.winner, None);
        assert!(result.suppressed_entities.is_empty());
        assert_eq!(result.competition_intensity, 0.0);
    }

    #[tokio::test]
    async fn test_apply_temporal_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.8, 0.6, 0.4, 0.7]);
        let group = create_test_group(entities.clone(), CompetitionType::Temporal, false);
        let config = InhibitionConfig::default();
        
        let result = apply_temporal_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Temporal competition applies alternating suppression
        // Even indices (0, 2) should maintain strength, odd indices (1, 3) should be reduced
        assert_eq!(pattern.activations[&entities[0]], 0.8); // Even index - full strength
        assert_eq!(pattern.activations[&entities[1]], 0.6 * 0.3); // Odd index - reduced
        assert_eq!(pattern.activations[&entities[2]], 0.4); // Even index - full strength
        assert_eq!(pattern.activations[&entities[3]], 0.7 * 0.3); // Odd index - reduced
        
        assert!(result.winner.is_some());
        assert_eq!(result.competition_intensity, group.inhibition_strength);
    }

    #[tokio::test]
    async fn test_apply_hierarchical_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.9, 0.6, 0.4]);
        let group = create_test_group(entities.clone(), CompetitionType::Hierarchical, false);
        let config = InhibitionConfig::default();
        
        let result = apply_hierarchical_competition(&mut pattern, &group, &config).await.unwrap();
        
        // First entity (highest level) should inhibit others
        assert_eq!(pattern.activations[&entities[0]], 0.9); // Highest level retains strength
        assert!(pattern.activations[&entities[1]] < 0.6); // Lower level inhibited
        assert!(pattern.activations[&entities[2]] < 0.4); // Lowest level most inhibited
        
        assert_eq!(result.winner, Some(entities[0]));
    }

    #[tokio::test]
    async fn test_apply_contextual_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.8, 0.5]);
        let group = create_test_group(entities.clone(), CompetitionType::Contextual, false);
        let config = InhibitionConfig::default();
        
        // Should delegate to semantic competition
        let result = apply_contextual_competition(&mut pattern, &group, &config).await.unwrap();
        
        assert_eq!(result.winner, Some(entities[0]));
        assert!(result.competition_intensity > 0.0);
    }

    #[tokio::test]
    async fn test_apply_spatial_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.7, 0.6]);
        let group = create_test_group(entities.clone(), CompetitionType::Spatial, false);
        let config = InhibitionConfig::default();
        
        // Should delegate to semantic competition
        let result = apply_spatial_competition(&mut pattern, &group, &config).await.unwrap();
        
        assert_eq!(result.winner, Some(entities[0]));
        assert!(result.competition_intensity > 0.0);
    }

    #[tokio::test]
    async fn test_apply_causal_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.6, 0.8]);
        let group = create_test_group(entities.clone(), CompetitionType::Causal, false);
        let config = InhibitionConfig::default();
        
        // Should delegate to semantic competition
        let result = apply_causal_competition(&mut pattern, &group, &config).await.unwrap();
        
        assert_eq!(result.winner, Some(entities[1])); // Higher strength wins
        assert!(result.competition_intensity > 0.0);
    }

    #[tokio::test]
    async fn test_apply_soft_competition() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.8, 0.6, 0.4, 0.7]);
        let pairs = vec![(entities[0], entities[1]), (entities[2], entities[3])];
        
        apply_soft_competition(&mut pattern, &pairs, 0.5).await.unwrap();
        
        // Stronger entity in each pair should inhibit weaker one
        assert_eq!(pattern.activations[&entities[0]], 0.8); // Stronger in pair 1
        assert!(pattern.activations[&entities[1]] < 0.6); // Weaker in pair 1, inhibited
        assert_eq!(pattern.activations[&entities[2]], 0.4); // Weaker in pair 2, inhibited
        assert!(pattern.activations[&entities[3]] < 0.7); // Stronger in pair 2
    }

    #[tokio::test]
    async fn test_apply_soft_competition_equal_strengths() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.5, 0.5]);
        let pairs = vec![(entities[0], entities[1])];
        
        apply_soft_competition(&mut pattern, &pairs, 0.3).await.unwrap();
        
        // Equal strengths should remain unchanged
        assert_eq!(pattern.activations[&entities[0]], 0.5);
        assert_eq!(pattern.activations[&entities[1]], 0.5);
    }

    #[tokio::test]
    async fn test_competition_intensity_calculation() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.9, 0.1]);
        let group = create_test_group(entities.clone(), CompetitionType::Semantic, false);
        let config = InhibitionConfig::default();
        
        let result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Competition intensity should be based on winner strength and inhibition strength
        let expected_intensity = 0.9 * group.inhibition_strength;
        assert_eq!(result.competition_intensity, expected_intensity);
    }

    #[tokio::test]
    async fn test_winner_takes_all_threshold() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.85, 0.6]); // Above threshold
        let mut group = create_test_group(entities.clone(), CompetitionType::Semantic, true);
        group.winner_takes_all = true;
        let config = InhibitionConfig {
            winner_takes_all_threshold: 0.8,
            ..Default::default()
        };
        
        let result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Should trigger winner-takes-all since winner > threshold
        assert_eq!(pattern.activations[&entities[1]], 0.0);
        assert!(result.suppressed_entities.contains(&entities[1]));
    }

    #[tokio::test]
    async fn test_winner_takes_all_below_threshold() {
        let (mut pattern, entities) = create_test_pattern_with_entities(vec![0.7, 0.6]); // Below threshold
        let group = create_test_group(entities.clone(), CompetitionType::Semantic, true);
        let config = InhibitionConfig {
            winner_takes_all_threshold: 0.8,
            ..Default::default()
        };
        
        let _result = apply_semantic_competition(&mut pattern, &group, &config).await.unwrap();
        
        // Should use soft competition since winner < threshold
        assert!(pattern.activations[&entities[1]] > 0.0);
        assert!(pattern.activations[&entities[1]] < 0.6);
    }
}