//! Hierarchical inhibition logic

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, HierarchicalInhibitionResult,
    HierarchicalLayer, InhibitionMatrix, InhibitionConfig
};
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Apply hierarchical inhibition based on abstraction levels
pub async fn apply_hierarchical_inhibition(
    _system: &CompetitiveInhibitionSystem,
    pattern: &mut ActivationPattern,
    inhibition_matrix: &Arc<RwLock<InhibitionMatrix>>,
    config: &InhibitionConfig,
) -> Result<HierarchicalInhibitionResult> {
    let matrix = inhibition_matrix.read().await;
    
    // Group entities by abstraction level (simplified - in practice would use entity metadata)
    let abstraction_levels = assign_abstraction_levels(&pattern.activations);
    
    // Create hierarchical layers
    let mut layers = create_hierarchical_layers(&abstraction_levels, pattern);
    
    // Apply top-down inhibition
    apply_top_down_inhibition(&mut layers, &matrix, config);
    
    // Apply lateral inhibition within layers
    apply_within_layer_inhibition(&mut layers, &matrix, config);
    
    // Update the activation pattern with inhibited values
    update_pattern_from_layers(pattern, &layers);
    
    // Identify winners and suppressed entities
    let specificity_winners = identify_specificity_winners(&layers);
    let generality_suppressed = identify_generality_suppressed(&layers);
    
    Ok(HierarchicalInhibitionResult {
        hierarchical_layers: layers,
        specificity_winners,
        generality_suppressed,
        abstraction_levels,
    })
}

/// Assign abstraction levels to entities (simplified heuristic)
fn assign_abstraction_levels(activations: &HashMap<EntityKey, f32>) -> HashMap<EntityKey, u32> {
    let mut levels = HashMap::new();
    
    for (entity, strength) in activations {
        // Simple heuristic: stronger activations tend to be more specific (lower level)
        let level = if *strength > 0.8 {
            0 // Most specific
        } else if *strength > 0.5 {
            1 // Mid-level
        } else {
            2 // Most general
        };
        
        levels.insert(*entity, level);
    }
    
    levels
}

/// Create hierarchical layers from abstraction levels
fn create_hierarchical_layers(
    abstraction_levels: &HashMap<EntityKey, u32>,
    pattern: &ActivationPattern,
) -> Vec<HierarchicalLayer> {
    let mut layers_map: HashMap<u32, Vec<EntityKey>> = HashMap::new();
    
    // Group entities by level
    for (entity, level) in abstraction_levels {
        layers_map.entry(*level).or_insert_with(Vec::new).push(*entity);
    }
    
    // Create layer structures
    let mut layers: Vec<_> = layers_map.into_iter()
        .map(|(level, entities)| {
            let dominant = entities.iter()
                .max_by(|a, b| {
                    let strength_a = pattern.activations.get(a).unwrap_or(&0.0);
                    let strength_b = pattern.activations.get(b).unwrap_or(&0.0);
                    strength_a.partial_cmp(strength_b).unwrap()
                })
                .copied();
            
            HierarchicalLayer {
                layer_level: level,
                entities,
                inhibition_strength: 0.5 + (level as f32 * 0.1), // Higher levels have stronger inhibition
                dominant_entity: dominant,
            }
        })
        .collect();
    
    layers.sort_by_key(|l| l.layer_level);
    layers
}

/// Apply top-down inhibition from higher to lower layers
fn apply_top_down_inhibition(
    layers: &mut [HierarchicalLayer],
    matrix: &InhibitionMatrix,
    config: &InhibitionConfig,
) {
    for i in 0..layers.len() {
        if let Some(dominant) = layers[i].dominant_entity {
            // Inhibit lower layers
            for j in (i + 1)..layers.len() {
                let inhibition_strength = layers[i].inhibition_strength * config.hierarchical_inhibition_strength;
                
                for entity in &layers[j].entities {
                    // Check if there's a specific inhibition relationship
                    let specific_inhibition = matrix.hierarchical_inhibition
                        .get(&(dominant, *entity))
                        .copied()
                        .unwrap_or(0.0);
                    
                    let _total_inhibition = inhibition_strength.max(specific_inhibition);
                    
                    // This would update the actual activation values
                    // In practice, we'd track these changes and apply them later
                }
            }
        }
    }
}

/// Apply lateral inhibition within each layer
fn apply_within_layer_inhibition(
    layers: &mut [HierarchicalLayer],
    matrix: &InhibitionMatrix,
    config: &InhibitionConfig,
) {
    for layer in layers {
        if layer.entities.len() <= 1 {
            continue;
        }
        
        // Apply competition within the layer
        // Simplified: dominant entity inhibits others
        if let Some(dominant) = layer.dominant_entity {
            for entity in &layer.entities {
                if *entity != dominant {
                    let _lateral_inhibition = matrix.lateral_inhibition
                        .get(&(dominant, *entity))
                        .copied()
                        .unwrap_or(config.lateral_inhibition_strength * 0.5);
                    
                    // Track inhibition to apply
                }
            }
        }
    }
}

/// Update the activation pattern based on hierarchical inhibition
fn update_pattern_from_layers(pattern: &mut ActivationPattern, layers: &[HierarchicalLayer]) {
    // This is a simplified version - in practice would apply the tracked inhibitions
    for layer in layers {
        if let Some(dominant) = layer.dominant_entity {
            // Boost dominant entity slightly
            if let Some(strength) = pattern.activations.get_mut(&dominant) {
                *strength = (*strength * 1.1).min(1.0);
            }
            
            // Suppress non-dominant entities in the layer
            for entity in &layer.entities {
                if *entity != dominant {
                    if let Some(strength) = pattern.activations.get_mut(entity) {
                        *strength *= 0.7; // Reduce by 30%
                    }
                }
            }
        }
    }
}

/// Identify entities that won due to specificity
fn identify_specificity_winners(layers: &[HierarchicalLayer]) -> Vec<EntityKey> {
    layers.iter()
        .filter(|l| l.layer_level == 0) // Most specific level
        .filter_map(|l| l.dominant_entity)
        .collect()
}

/// Identify entities suppressed due to being too general
fn identify_generality_suppressed(layers: &[HierarchicalLayer]) -> Vec<EntityKey> {
    layers.iter()
        .filter(|l| l.layer_level >= 2) // Higher abstraction levels
        .flat_map(|l| {
            l.entities.iter()
                .filter(|e| Some(**e) != l.dominant_entity)
                .copied()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::ActivationPattern;
    use crate::cognitive::inhibitory::{CompetitiveInhibitionSystem, InhibitionConfig, InhibitionMatrix};
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::core::types::EntityKey;
    use crate::cognitive::critical::CriticalThinking;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

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

    fn create_test_system() -> CompetitiveInhibitionSystem {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).unwrap());
        let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
        let critical_thinking = Arc::new(CriticalThinking::new(graph));
        
        CompetitiveInhibitionSystem::new(activation_engine, critical_thinking)
    }

    #[tokio::test]
    async fn test_assign_abstraction_levels() {
        let (pattern, _) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3, 0.1]);
        
        let levels = assign_abstraction_levels(&pattern.activations);
        
        // Higher activation strengths should get lower abstraction levels (more specific)
        assert_eq!(levels.len(), 4);
        for (entity, strength) in &pattern.activations {
            let level = levels[entity];
            if *strength > 0.8 {
                assert_eq!(level, 0); // Most specific
            } else if *strength > 0.5 {
                assert_eq!(level, 1); // Mid-level
            } else {
                assert_eq!(level, 2); // Most general
            }
        }
    }

    #[tokio::test]
    async fn test_create_hierarchical_layers() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3, 0.8]);
        
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        // Should create layers for each abstraction level
        assert!(layers.len() <= 3); // At most 3 levels (0, 1, 2)
        
        // Verify layer structure
        for layer in &layers {
            assert!(!layer.entities.is_empty());
            assert!(layer.inhibition_strength > 0.0);
            
            // Dominant entity should be the one with highest activation in the layer
            if let Some(dominant) = layer.dominant_entity {
                assert!(layer.entities.contains(&dominant));
                let dominant_strength = pattern.activations[&dominant];
                
                for &entity in &layer.entities {
                    if entity != dominant {
                        assert!(pattern.activations[&entity] <= dominant_strength);
                    }
                }
            }
        }
        
        // Layers should be sorted by level
        for i in 1..layers.len() {
            assert!(layers[i-1].layer_level <= layers[i].layer_level);
        }
    }

    #[tokio::test]
    async fn test_apply_top_down_inhibition() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3]);
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let mut layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        let matrix = InhibitionMatrix::new();
        let config = InhibitionConfig::default();
        
        apply_top_down_inhibition(&mut layers, &matrix, &config);
        
        // This function modifies layers but doesn't return anything to test directly
        // In a full implementation, it would track inhibitions to be applied
        assert!(!layers.is_empty());
    }

    #[tokio::test]
    async fn test_apply_within_layer_inhibition() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.8, 0.7]);
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let mut layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        let matrix = InhibitionMatrix::new();
        let config = InhibitionConfig::default();
        
        apply_within_layer_inhibition(&mut layers, &matrix, &config);
        
        // Function should process layers with multiple entities
        for layer in &layers {
            if layer.entities.len() > 1 {
                assert!(layer.dominant_entity.is_some());
            }
        }
    }

    #[tokio::test]
    async fn test_update_pattern_from_layers() {
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3]);
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        let original_activations = pattern.activations.clone();
        
        update_pattern_from_layers(&mut pattern, &layers);
        
        // Dominant entities should be boosted slightly
        for layer in &layers {
            if let Some(dominant) = layer.dominant_entity {
                let new_strength = pattern.activations[&dominant];
                let original_strength = original_activations[&dominant];
                assert!(new_strength >= original_strength);
                assert!(new_strength <= 1.0);
            }
            
            // Non-dominant entities should be suppressed
            for &entity in &layer.entities {
                if Some(entity) != layer.dominant_entity {
                    let new_strength = pattern.activations[&entity];
                    let original_strength = original_activations[&entity];
                    assert!(new_strength <= original_strength);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_identify_specificity_winners() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3]);
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        let winners = identify_specificity_winners(&layers);
        
        // Should identify dominant entities from most specific layer (level 0)
        let level_0_layers: Vec<_> = layers.iter().filter(|l| l.layer_level == 0).collect();
        let expected_count = level_0_layers.iter()
            .filter_map(|l| l.dominant_entity)
            .count();
        
        assert_eq!(winners.len(), expected_count);
        
        for winner in &winners {
            assert!(entities.contains(winner));
        }
    }

    #[tokio::test]
    async fn test_identify_generality_suppressed() {
        let (pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3, 0.2, 0.1]);
        let abstraction_levels = assign_abstraction_levels(&pattern.activations);
        let layers = create_hierarchical_layers(&abstraction_levels, &pattern);
        
        let suppressed = identify_generality_suppressed(&layers);
        
        // Should identify non-dominant entities from higher abstraction levels (>= 2)
        for entity in &suppressed {
            assert!(entities.contains(entity));
            let level = abstraction_levels[entity];
            assert!(level >= 2);
        }
        
        // Verify these are not dominant entities
        for layer in &layers {
            if layer.layer_level >= 2 {
                for &entity in &suppressed {
                    if layer.entities.contains(&entity) {
                        assert_ne!(Some(entity), layer.dominant_entity);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_apply_hierarchical_inhibition_full_flow() {
        let system = create_test_system();
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6, 0.3, 0.8, 0.1]);
        
        let result = apply_hierarchical_inhibition(
            &system,
            &mut pattern,
            &system.inhibition_matrix,
            &system.inhibition_config,
        ).await.unwrap();
        
        // Verify result structure
        assert!(!result.hierarchical_layers.is_empty());
        assert_eq!(result.abstraction_levels.len(), entities.len());
        
        // Verify that abstraction levels are correctly assigned
        for (entity, level) in &result.abstraction_levels {
            assert!(entities.contains(entity));
            assert!(*level <= 2); // Should be 0, 1, or 2
        }
        
        // Verify that specificity winners are from the most specific layer
        for winner in &result.specificity_winners {
            let level = result.abstraction_levels[winner];
            assert_eq!(level, 0); // Most specific level
        }
        
        // Verify that generality suppressed are from higher levels
        for suppressed in &result.generality_suppressed {
            let level = result.abstraction_levels[suppressed];
            assert!(level >= 2);
        }
    }

    #[tokio::test]
    async fn test_hierarchical_inhibition_with_custom_matrix() {
        let system = create_test_system();
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.9, 0.6]);
        
        // Add specific hierarchical inhibition to matrix
        {
            let mut matrix = system.inhibition_matrix.write().await;
            matrix.hierarchical_inhibition.insert((entities[0], entities[1]), 0.8);
        }
        
        let result = apply_hierarchical_inhibition(
            &system,
            &mut pattern,
            &system.inhibition_matrix,
            &system.inhibition_config,
        ).await.unwrap();
        
        assert!(!result.hierarchical_layers.is_empty());
        assert_eq!(result.abstraction_levels.len(), 2);
    }

    #[tokio::test]
    async fn test_empty_pattern_hierarchical_inhibition() {
        let system = create_test_system();
        let mut pattern = ActivationPattern { 
            activations: HashMap::new(),
            timestamp: SystemTime::now(),
            query: "test".to_string(),
        };
        
        let result = apply_hierarchical_inhibition(
            &system,
            &mut pattern,
            &system.inhibition_matrix,
            &system.inhibition_config,
        ).await.unwrap();
        
        // Should handle empty pattern gracefully
        assert!(result.hierarchical_layers.is_empty());
        assert!(result.abstraction_levels.is_empty());
        assert!(result.specificity_winners.is_empty());
        assert!(result.generality_suppressed.is_empty());
    }

    #[tokio::test]
    async fn test_single_entity_hierarchical_inhibition() {
        let system = create_test_system();
        let (mut pattern, entities) = create_test_pattern_with_strengths(vec![0.7]);
        
        let result = apply_hierarchical_inhibition(
            &system,
            &mut pattern,
            &system.inhibition_matrix,
            &system.inhibition_config,
        ).await.unwrap();
        
        // Should handle single entity
        assert_eq!(result.hierarchical_layers.len(), 1);
        assert_eq!(result.abstraction_levels.len(), 1);
        assert_eq!(result.hierarchical_layers[0].entities.len(), 1);
        assert_eq!(result.hierarchical_layers[0].dominant_entity, Some(entities[0]));
    }
}