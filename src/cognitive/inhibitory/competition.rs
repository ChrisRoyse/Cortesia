//! Competition strategies and group competition logic

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, CompetitionGroup, CompetitionType,
    GroupCompetitionResult, InhibitionConfig, TemporalDynamics
};
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Apply group-based competition to the activation pattern
pub async fn apply_group_competition(
    system: &CompetitiveInhibitionSystem,
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
        pre_competition,
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
    config: &InhibitionConfig,
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
    
    Ok(GroupCompetitionResult {
        group_id: group.group_id.clone(),
        pre_competition,
        post_competition,
        winner: post_competition.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(entity, _)| *entity),
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
    
    Ok(GroupCompetitionResult {
        group_id: group.group_id.clone(),
        pre_competition,
        post_competition,
        winner: post_competition.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(entity, _)| *entity),
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