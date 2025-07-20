//! # Quantum Knowledge Synthesizer: Activation Propagation Testing Strategies
//! 
//! This module provides comprehensive testing frameworks for activation propagation
//! in complex neural network topologies with hook-intelligent validation.
//! 
//! ## Propagation Testing Philosophy
//! - Graph topology affects propagation patterns in predictable ways
//! - Convergence behavior must be tested across diverse network structures
//! - Energy conservation and stability are critical system properties
//! - Emergent behaviors arise from complex interconnection patterns

use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, RelationType,
    ActivationPattern, LogicGate, LogicGateType, ActivationStep, ActivationOperation
};
use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::types::EntityKey;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, Instant};
use rand::Rng;

/// Advanced graph topology testing for activation propagation
#[derive(Debug, Clone)]
pub struct GraphTopologyTestHarness {
    pub max_entities: usize,
    pub convergence_threshold: f32,
    pub max_test_iterations: usize,
}

impl GraphTopologyTestHarness {
    pub fn new() -> Self {
        Self {
            max_entities: 1000,
            convergence_threshold: 0.001,
            max_test_iterations: 100,
        }
    }
    
    /// Test activation propagation in small-world networks
    pub async fn test_small_world_propagation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create small-world network: ring with random rewiring
        let ring_size = 50;
        let rewiring_probability = 0.1;
        let mut entity_keys = Vec::new();
        
        // Create ring of entities
        for i in 0..ring_size {
            let entity = BrainInspiredEntity::new(
                format!("Ring{}", i),
                EntityDirection::Hidden
            );
            entity_keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Create ring connections with random rewiring
        for i in 0..ring_size {
            let source = entity_keys[i];
            let next = entity_keys[(i + 1) % ring_size];
            
            let target = if rand::random::<f32>() < rewiring_probability {
                // Rewire to random target
                let random_idx = rand::random::<usize>() % ring_size;
                entity_keys[random_idx]
            } else {
                next
            };
            
            if source != target {
                let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
                rel.weight = 0.5;
                engine.add_relationship(rel).await?;
            }
        }
        
        // Test propagation from single source
        let mut pattern = ActivationPattern::new("small_world_test".to_string());
        pattern.activations.insert(entity_keys[0], 1.0);
        
        let result = engine.propagate_activation(&pattern).await?;
        
        // Analyze propagation characteristics
        let active_nodes = result.final_activations.iter()
            .filter(|(_, &activation)| activation > 0.01)
            .count();
        
        let total_energy: f32 = result.final_activations.values().sum();
        let propagation_efficiency = active_nodes as f32 / ring_size as f32;
        
        println!("Small-world propagation: {}/{} nodes active, efficiency: {:.3}, energy: {:.3}",
                active_nodes, ring_size, propagation_efficiency, total_energy);
        
        // Small-world networks should show efficient propagation
        assert!(propagation_efficiency > 0.3, 
            "Small-world should enable efficient propagation: {}", propagation_efficiency);
        assert!(result.converged, "Small-world propagation should converge");
        
        Ok(())
    }
    
    /// Test activation propagation in scale-free networks
    pub async fn test_scale_free_propagation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create scale-free network using preferential attachment
        let network_size = 100;
        let mut entity_keys = Vec::new();
        let mut degree_count: HashMap<EntityKey, usize> = HashMap::new();
        
        // Create initial entities
        for i in 0..network_size {
            let entity = BrainInspiredEntity::new(
                format!("ScaleFree{}", i),
                EntityDirection::Hidden
            );
            entity_keys.push(entity.id);
            degree_count.insert(entity.id, 0);
            engine.add_entity(entity).await?;
        }
        
        // Preferential attachment process
        let mut total_degree = 0;
        for i in 1..network_size {
            let new_node = entity_keys[i];
            let connections_to_make = (i.min(3)) + 1; // 1-4 connections
            
            for _ in 0..connections_to_make {
                // Select target based on degree (preferential attachment)
                let target_idx = if total_degree > 0 {
                    Self::select_preferential_target(&entity_keys[..i], &degree_count, total_degree)
                } else {
                    0 // Connect to first node if no edges exist
                };
                
                let target = entity_keys[target_idx];
                
                if new_node != target {
                    let mut rel = BrainInspiredRelationship::new(new_node, target, RelationType::RelatedTo);
                    rel.weight = 0.4;
                    engine.add_relationship(rel).await?;
                    
                    *degree_count.get_mut(&new_node).unwrap() += 1;
                    *degree_count.get_mut(&target).unwrap() += 1;
                    total_degree += 2;
                }
            }
        }
        
        // Identify hub nodes (high degree)
        let mut degrees: Vec<_> = degree_count.iter().collect();
        degrees.sort_by_key(|(_, &degree)| std::cmp::Reverse(degree));
        let hub_key = *degrees[0].0;
        
        // Test propagation from hub
        let mut hub_pattern = ActivationPattern::new("hub_propagation".to_string());
        hub_pattern.activations.insert(hub_key, 1.0);
        
        let hub_result = engine.propagate_activation(&hub_pattern).await?;
        
        // Test propagation from peripheral node
        let peripheral_key = *degrees.last().unwrap().0;
        let mut peripheral_pattern = ActivationPattern::new("peripheral_propagation".to_string());
        peripheral_pattern.activations.insert(peripheral_key, 1.0);
        
        let peripheral_result = engine.propagate_activation(&peripheral_pattern).await?;
        
        // Analyze hub vs peripheral propagation
        let hub_reach = hub_result.final_activations.iter()
            .filter(|(_, &activation)| activation > 0.01)
            .count();
        let peripheral_reach = peripheral_result.final_activations.iter()
            .filter(|(_, &activation)| activation > 0.01)
            .count();
        
        println!("Scale-free propagation: Hub reach: {}, Peripheral reach: {}", 
                hub_reach, peripheral_reach);
        
        // Hubs should have greater propagation reach
        assert!(hub_reach >= peripheral_reach,
            "Hub should have greater reach than peripheral: {} vs {}", hub_reach, peripheral_reach);
        
        Ok(())
    }
    
    /// Test activation propagation in hierarchical networks
    pub async fn test_hierarchical_propagation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create hierarchical network: tree structure with cross-level connections
        let levels = 4;
        let branching_factor = 3;
        let mut level_entities: Vec<Vec<EntityKey>> = Vec::new();
        
        // Create tree levels
        for level in 0..levels {
            let mut level_nodes = Vec::new();
            let nodes_at_level = branching_factor.pow(level as u32);
            
            for i in 0..nodes_at_level {
                let entity = BrainInspiredEntity::new(
                    format!("L{}N{}", level, i),
                    if level == 0 { EntityDirection::Input } 
                    else if level == levels - 1 { EntityDirection::Output }
                    else { EntityDirection::Hidden }
                );
                level_nodes.push(entity.id);
                engine.add_entity(entity).await?;
            }
            
            level_entities.push(level_nodes);
        }
        
        // Create hierarchical connections
        for level in 0..levels - 1 {
            for (parent_idx, &parent) in level_entities[level].iter().enumerate() {
                // Each parent connects to its children
                for child_idx in 0..branching_factor {
                    let child_global_idx = parent_idx * branching_factor + child_idx;
                    if child_global_idx < level_entities[level + 1].len() {
                        let child = level_entities[level + 1][child_global_idx];
                        
                        let mut rel = BrainInspiredRelationship::new(parent, child, RelationType::IsA);
                        rel.weight = 0.7; // Strong hierarchical connections
                        engine.add_relationship(rel).await?;
                    }
                }
            }
        }
        
        // Add cross-level connections (skip connections)
        for level in 0..levels - 2 {
            for &source in &level_entities[level] {
                if rand::random::<f32>() < 0.2 { // 20% chance of skip connection
                    let target_level = level + 2;
                    if target_level < level_entities.len() {
                        let target_idx = rand::random::<usize>() % level_entities[target_level].len();
                        let target = level_entities[target_level][target_idx];
                        
                        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
                        rel.weight = 0.3; // Weaker skip connections
                        engine.add_relationship(rel).await?;
                    }
                }
            }
        }
        
        // Test top-down propagation
        let mut top_down_pattern = ActivationPattern::new("top_down".to_string());
        top_down_pattern.activations.insert(level_entities[0][0], 1.0); // Root node
        
        let top_down_result = engine.propagate_activation(&top_down_pattern).await?;
        
        // Test bottom-up propagation
        let mut bottom_up_pattern = ActivationPattern::new("bottom_up".to_string());
        for &leaf in &level_entities[levels - 1] {
            bottom_up_pattern.activations.insert(leaf, 0.5);
        }
        
        let bottom_up_result = engine.propagate_activation(&bottom_up_pattern).await?;
        
        // Analyze hierarchical propagation patterns
        println!("Hierarchical propagation analysis:");
        for (level_idx, level_nodes) in level_entities.iter().enumerate() {
            let top_down_level_activation: f32 = level_nodes.iter()
                .map(|&key| top_down_result.final_activations.get(&key).copied().unwrap_or(0.0))
                .sum();
            let bottom_up_level_activation: f32 = level_nodes.iter()
                .map(|&key| bottom_up_result.final_activations.get(&key).copied().unwrap_or(0.0))
                .sum();
            
            println!("  Level {}: Top-down={:.3}, Bottom-up={:.3}", 
                    level_idx, top_down_level_activation, bottom_up_level_activation);
        }
        
        // Verify hierarchical propagation characteristics
        assert!(top_down_result.converged && bottom_up_result.converged,
            "Hierarchical propagation should converge");
        
        Ok(())
    }
    
    /// Test activation propagation with dynamic topology changes
    pub async fn test_dynamic_topology_propagation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create initial network
        let mut entity_keys = Vec::new();
        for i in 0..20 {
            let entity = BrainInspiredEntity::new(
                format!("Dynamic{}", i),
                EntityDirection::Hidden
            );
            entity_keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Create initial connections
        let mut active_relationships = HashSet::new();
        for i in 0..entity_keys.len() {
            for j in i + 1..entity_keys.len() {
                if rand::random::<f32>() < 0.3 {
                    let mut rel = BrainInspiredRelationship::new(
                        entity_keys[i], entity_keys[j], RelationType::RelatedTo
                    );
                    rel.weight = 0.5;
                    active_relationships.insert((entity_keys[i], entity_keys[j]));
                    engine.add_relationship(rel).await?;
                }
            }
        }
        
        // Test propagation with topology changes
        for phase in 0..5 {
            // Propagate activation
            let mut pattern = ActivationPattern::new(format!("dynamic_phase_{}", phase));
            pattern.activations.insert(entity_keys[0], 1.0);
            
            let result = engine.propagate_activation(&pattern).await?;
            
            let phase_reach = result.final_activations.iter()
                .filter(|(_, &activation)| activation > 0.01)
                .count();
            
            println!("Phase {}: Reach = {}, Convergence = {}", 
                    phase, phase_reach, result.converged);
            
            // Modify topology for next phase
            self.modify_network_topology(&engine, &entity_keys, &mut active_relationships).await?;
        }
        
        Ok(())
    }
    
    /// Test convergence properties across different network topologies
    pub async fn test_convergence_across_topologies(&self) -> Result<(), Box<dyn std::error::Error>> {
        let topologies = vec![
            ("Linear Chain", Self::create_linear_topology),
            ("Complete Graph", Self::create_complete_topology),
            ("Star Graph", Self::create_star_topology),
            ("Random Graph", Self::create_random_topology),
            ("Grid Graph", Self::create_grid_topology),
        ];
        
        for (name, topology_fn) in topologies {
            let config = ActivationConfig::default();
            let engine = ActivationPropagationEngine::new(config);
            
            let entity_keys = topology_fn(&engine, 25).await?;
            
            // Test convergence with multiple initial conditions
            let initial_conditions = vec![
                vec![0], // Single source
                vec![0, entity_keys.len() - 1], // Two sources
                (0..entity_keys.len()).step_by(5).collect(), // Multiple sources
            ];
            
            for (cond_idx, sources) in initial_conditions.iter().enumerate() {
                let mut pattern = ActivationPattern::new(
                    format!("{}_{}", name.replace(" ", "_"), cond_idx)
                );
                
                for &source_idx in sources {
                    if source_idx < entity_keys.len() {
                        pattern.activations.insert(entity_keys[source_idx], 1.0);
                    }
                }
                
                let start = Instant::now();
                let result = engine.propagate_activation(&pattern).await?;
                let duration = start.elapsed();
                
                println!("{} (condition {}): Converged={}, Iterations={}, Time={:?}",
                        name, cond_idx, result.converged, result.iterations_completed, duration);
                
                // All topologies should eventually converge
                assert!(result.converged, "{} should converge for condition {}", name, cond_idx);
            }
        }
        
        Ok(())
    }
    
    // Helper methods
    
    fn select_preferential_target(
        candidates: &[EntityKey],
        degree_count: &HashMap<EntityKey, usize>,
        total_degree: usize
    ) -> usize {
        let mut rng = rand::thread_rng();
        let random_value = rng.gen_range(0..total_degree);
        
        let mut cumulative = 0;
        for (idx, &key) in candidates.iter().enumerate() {
            cumulative += degree_count.get(&key).copied().unwrap_or(0);
            if cumulative > random_value {
                return idx;
            }
        }
        candidates.len() - 1 // Fallback
    }
    
    async fn modify_network_topology(
        &self,
        engine: &ActivationPropagationEngine,
        entity_keys: &[EntityKey],
        active_relationships: &mut HashSet<(EntityKey, EntityKey)>
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Add some new relationships
        for _ in 0..5 {
            let i = rand::random::<usize>() % entity_keys.len();
            let j = rand::random::<usize>() % entity_keys.len();
            if i != j {
                let pair = (entity_keys[i], entity_keys[j]);
                if !active_relationships.contains(&pair) {
                    let mut rel = BrainInspiredRelationship::new(
                        entity_keys[i], entity_keys[j], RelationType::RelatedTo
                    );
                    rel.weight = rand::random::<f32>() * 0.5 + 0.25; // 0.25-0.75
                    active_relationships.insert(pair);
                    engine.add_relationship(rel).await?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn create_linear_topology(
        engine: &ActivationPropagationEngine,
        size: usize
    ) -> Result<Vec<EntityKey>, Box<dyn std::error::Error>> {
        let mut keys = Vec::new();
        
        for i in 0..size {
            let entity = BrainInspiredEntity::new(
                format!("Linear{}", i),
                EntityDirection::Hidden
            );
            keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        for i in 0..size - 1 {
            let mut rel = BrainInspiredRelationship::new(keys[i], keys[i + 1], RelationType::RelatedTo);
            rel.weight = 0.6;
            engine.add_relationship(rel).await?;
        }
        
        Ok(keys)
    }
    
    async fn create_complete_topology(
        engine: &ActivationPropagationEngine,
        size: usize
    ) -> Result<Vec<EntityKey>, Box<dyn std::error::Error>> {
        let mut keys = Vec::new();
        
        for i in 0..size {
            let entity = BrainInspiredEntity::new(
                format!("Complete{}", i),
                EntityDirection::Hidden
            );
            keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        for i in 0..size {
            for j in i + 1..size {
                let mut rel = BrainInspiredRelationship::new(keys[i], keys[j], RelationType::RelatedTo);
                rel.weight = 0.3; // Lower weight to prevent oversaturation
                engine.add_relationship(rel).await?;
            }
        }
        
        Ok(keys)
    }
    
    async fn create_star_topology(
        engine: &ActivationPropagationEngine,
        size: usize
    ) -> Result<Vec<EntityKey>, Box<dyn std::error::Error>> {
        let mut keys = Vec::new();
        
        for i in 0..size {
            let entity = BrainInspiredEntity::new(
                format!("Star{}", i),
                EntityDirection::Hidden
            );
            keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Connect all nodes to center (first node)
        for i in 1..size {
            let mut rel = BrainInspiredRelationship::new(keys[0], keys[i], RelationType::RelatedTo);
            rel.weight = 0.6;
            engine.add_relationship(rel).await?;
        }
        
        Ok(keys)
    }
    
    async fn create_random_topology(
        engine: &ActivationPropagationEngine,
        size: usize
    ) -> Result<Vec<EntityKey>, Box<dyn std::error::Error>> {
        let mut keys = Vec::new();
        
        for i in 0..size {
            let entity = BrainInspiredEntity::new(
                format!("Random{}", i),
                EntityDirection::Hidden
            );
            keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Random connections with 0.2 probability
        for i in 0..size {
            for j in i + 1..size {
                if rand::random::<f32>() < 0.2 {
                    let mut rel = BrainInspiredRelationship::new(keys[i], keys[j], RelationType::RelatedTo);
                    rel.weight = rand::random::<f32>() * 0.5 + 0.25;
                    engine.add_relationship(rel).await?;
                }
            }
        }
        
        Ok(keys)
    }
    
    async fn create_grid_topology(
        engine: &ActivationPropagationEngine,
        size: usize
    ) -> Result<Vec<EntityKey>, Box<dyn std::error::Error>> {
        let grid_size = (size as f32).sqrt() as usize;
        let mut keys = Vec::new();
        
        // Create grid of entities
        for i in 0..grid_size * grid_size {
            let entity = BrainInspiredEntity::new(
                format!("Grid{}", i),
                EntityDirection::Hidden
            );
            keys.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Connect neighbors in grid
        for row in 0..grid_size {
            for col in 0..grid_size {
                let current_idx = row * grid_size + col;
                
                // Right neighbor
                if col < grid_size - 1 {
                    let right_idx = row * grid_size + (col + 1);
                    let mut rel = BrainInspiredRelationship::new(
                        keys[current_idx], keys[right_idx], RelationType::RelatedTo
                    );
                    rel.weight = 0.5;
                    engine.add_relationship(rel).await?;
                }
                
                // Bottom neighbor
                if row < grid_size - 1 {
                    let bottom_idx = (row + 1) * grid_size + col;
                    let mut rel = BrainInspiredRelationship::new(
                        keys[current_idx], keys[bottom_idx], RelationType::RelatedTo
                    );
                    rel.weight = 0.5;
                    engine.add_relationship(rel).await?;
                }
            }
        }
        
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_small_world_networks() {
        let harness = GraphTopologyTestHarness::new();
        harness.test_small_world_propagation().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_scale_free_networks() {
        let harness = GraphTopologyTestHarness::new();
        harness.test_scale_free_propagation().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_hierarchical_networks() {
        let harness = GraphTopologyTestHarness::new();
        harness.test_hierarchical_propagation().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_dynamic_topology() {
        let harness = GraphTopologyTestHarness::new();
        harness.test_dynamic_topology_propagation().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_topology_convergence() {
        let harness = GraphTopologyTestHarness::new();
        harness.test_convergence_across_topologies().await.unwrap();
    }
}