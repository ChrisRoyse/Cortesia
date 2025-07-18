//! Hierarchical inhibition logic

use crate::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, HierarchicalInhibitionResult,
    HierarchicalLayer, InhibitionMatrix, InhibitionConfig
};
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Apply hierarchical inhibition based on abstraction levels
pub async fn apply_hierarchical_inhibition(
    system: &CompetitiveInhibitionSystem,
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
                    
                    let total_inhibition = inhibition_strength.max(specific_inhibition);
                    
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
                    let lateral_inhibition = matrix.lateral_inhibition
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