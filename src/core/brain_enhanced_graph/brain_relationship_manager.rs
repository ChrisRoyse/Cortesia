//! Relationship management for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use crate::core::types::{EntityKey, Relationship};
use crate::error::Result;
use std::collections::{HashSet, VecDeque};

impl BrainEnhancedKnowledgeGraph {
    /// Get neighbors with synaptic weights
    pub async fn get_neighbors_with_weights(&self, entity: EntityKey) -> Vec<(EntityKey, f32)> {
        let core_neighbors = self.core_graph.get_neighbors(entity);
        let mut neighbors_with_weights = Vec::new();
        
        for neighbor in core_neighbors {
            let weight = self.get_synaptic_weight(entity, neighbor).await;
            neighbors_with_weights.push((neighbor, weight));
        }
        
        // Sort by synaptic weight (descending)
        neighbors_with_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        neighbors_with_weights
    }

    /// Get parent entities (incoming connections)
    pub async fn get_parent_entities(&self, entity: EntityKey) -> Vec<(EntityKey, f32)> {
        let incoming_neighbors = self.core_graph.get_incoming_neighbors(entity);
        let mut parents_with_weights = Vec::new();
        
        for parent in incoming_neighbors {
            let weight = self.get_synaptic_weight(parent, entity).await;
            parents_with_weights.push((parent, weight));
        }
        
        // Sort by synaptic weight (descending)
        parents_with_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        parents_with_weights
    }

    /// Get child entities (outgoing connections)
    pub async fn get_child_entities(&self, entity: EntityKey) -> Vec<(EntityKey, f32)> {
        let outgoing_neighbors = self.core_graph.get_outgoing_neighbors(entity);
        let mut children_with_weights = Vec::new();
        
        for child in outgoing_neighbors {
            let weight = self.get_synaptic_weight(entity, child).await;
            children_with_weights.push((child, weight));
        }
        
        // Sort by synaptic weight (descending)
        children_with_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        children_with_weights
    }

    /// Check if relationship exists
    pub async fn has_relationship(&self, source: EntityKey, target: EntityKey) -> bool {
        self.core_graph.has_relationship(source, target)
    }

    /// Get relationship weight (from core graph)
    pub fn get_relationship_weight(&self, source: EntityKey, target: EntityKey) -> Option<f32> {
        self.core_graph.get_relationship_weight(source, target)
    }

    /// Update relationship weight in both core graph and synaptic weights
    pub async fn update_relationship_weight(&self, source: EntityKey, target: EntityKey, weight: f32) -> Result<()> {
        // Update core graph
        self.core_graph.update_relationship_weight(source, target, weight)?;
        
        // Update synaptic weight
        self.set_synaptic_weight(source, target, weight).await;
        
        Ok(())
    }

    /// Remove relationship
    pub async fn remove_relationship(&self, source: EntityKey, target: EntityKey) -> Result<bool> {
        // Remove from core graph
        let removed = self.core_graph.remove_relationship(source, target)?;
        
        if removed {
            // Remove synaptic weight
            let mut weights = self.synaptic_weights.write().await;
            weights.remove(&(source, target));
            
            // Update statistics
            self.update_learning_stats(|stats| {
                stats.relationship_count = stats.relationship_count.saturating_sub(1);
            }).await;
        }
        
        Ok(removed)
    }

    /// Find alternative paths between entities
    pub async fn find_alternative_paths(&self, source: EntityKey, target: EntityKey, max_paths: usize) -> Vec<Vec<EntityKey>> {
        let mut paths = Vec::new();
        let mut visited_paths = HashSet::new();
        
        // Use BFS to find multiple paths
        let mut queue = VecDeque::new();
        queue.push_back(vec![source]);
        
        while let Some(current_path) = queue.pop_front() {
            if paths.len() >= max_paths {
                break;
            }
            
            let current_entity = *current_path.last().unwrap();
            
            if current_entity == target {
                // Found a path
                let path_signature = self.generate_path_signature(&current_path);
                if !visited_paths.contains(&path_signature) {
                    visited_paths.insert(path_signature);
                    paths.push(current_path);
                }
                continue;
            }
            
            // Avoid cycles and limit path length
            if current_path.len() > 6 || current_path.contains(&target) {
                continue;
            }
            
            // Get neighbors with synaptic weights
            let neighbors = self.core_graph.get_neighbors(current_entity);
            
            for neighbor in neighbors {
                let weight = 1.0; // Default weight since we don't have synaptic weights here
                if !current_path.contains(&neighbor) && weight > 0.1 {
                    let mut new_path = current_path.clone();
                    new_path.push(neighbor);
                    queue.push_back(new_path);
                }
            }
        }
        
        paths
    }

    /// Get strongest path between entities
    pub async fn get_strongest_path(&self, source: EntityKey, target: EntityKey) -> Option<(Vec<EntityKey>, f32)> {
        let paths = self.find_alternative_paths(source, target, 10).await;
        
        let mut strongest_path = None;
        let mut strongest_weight = 0.0;
        
        for path in paths {
            let path_weight = self.calculate_path_weight(&path).await;
            if path_weight > strongest_weight {
                strongest_weight = path_weight;
                strongest_path = Some(path);
            }
        }
        
        strongest_path.map(|path| (path, strongest_weight))
    }

    /// Calculate path weight based on synaptic weights
    async fn calculate_path_weight(&self, path: &[EntityKey]) -> f32 {
        if path.len() < 2 {
            return 0.0;
        }
        
        let mut total_weight = 1.0;
        
        for i in 0..path.len() - 1 {
            let weight = self.get_synaptic_weight(path[i], path[i + 1]).await;
            total_weight *= weight;
        }
        
        // Apply path length penalty
        let length_penalty = 1.0 / (path.len() as f32).sqrt();
        total_weight * length_penalty
    }

    /// Generate path signature for deduplication
    fn generate_path_signature(&self, path: &[EntityKey]) -> String {
        path.iter()
            .map(|key| {
                use slotmap::{Key, KeyData};
                let key_data: KeyData = key.data();
                key_data.as_ffi().to_string()
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Create learned relationship using Hebbian learning
    pub async fn create_learned_relationship(&self, source: EntityKey, target: EntityKey) -> Result<()> {
        if !self.config.enable_hebbian_learning {
            return Ok(());
        }
        
        let source_activation = self.get_entity_activation(source).await;
        let target_activation = self.get_entity_activation(target).await;
        
        // Calculate Hebbian weight
        let hebbian_weight = source_activation * target_activation * self.config.learning_rate;
        
        if hebbian_weight > 0.1 {
            // Create relationship in core graph
            let relationship = Relationship {
                from: source,
                to: target,
                rel_type: 0, // Default relationship type
                weight: hebbian_weight,
            };
            
            self.insert_brain_relationship(relationship).await?;
        }
        
        Ok(())
    }

    /// Strengthen relationship based on co-activation
    pub async fn strengthen_relationship(&self, source: EntityKey, target: EntityKey) -> Result<()> {
        if !self.has_relationship(source, target).await {
            return Ok(());
        }
        
        let source_activation = self.get_entity_activation(source).await;
        let target_activation = self.get_entity_activation(target).await;
        let current_weight = self.get_synaptic_weight(source, target).await;
        
        // Strengthen based on co-activation
        let strengthening_factor = (source_activation * target_activation * self.config.learning_rate).min(0.1);
        let new_weight = (current_weight + strengthening_factor).clamp(0.0, 1.0);
        
        self.update_relationship_weight(source, target, new_weight).await?;
        
        Ok(())
    }

    /// Weaken relationship based on lack of co-activation
    pub async fn weaken_relationship(&self, source: EntityKey, target: EntityKey) -> Result<()> {
        if !self.has_relationship(source, target).await {
            return Ok(());
        }
        
        let current_weight = self.get_synaptic_weight(source, target).await;
        let decay_factor = 1.0 - self.config.synaptic_strength_decay;
        let new_weight = current_weight * (1.0 - decay_factor);
        
        if new_weight < 0.01 {
            // Remove very weak relationships
            self.remove_relationship(source, target).await?;
        } else {
            self.update_relationship_weight(source, target, new_weight).await?;
        }
        
        Ok(())
    }

    /// Get relationship statistics
    pub async fn get_relationship_statistics(&self) -> RelationshipStatistics {
        let core_stats = self.core_graph.get_relationship_stats();
        let synaptic_weights = self.synaptic_weights.read().await;
        
        // Calculate synaptic weight statistics
        let weights: Vec<f32> = synaptic_weights.values().cloned().collect();
        let avg_synaptic_weight = if weights.is_empty() {
            0.0
        } else {
            weights.iter().sum::<f32>() / weights.len() as f32
        };
        
        let max_synaptic_weight = weights.iter().cloned().fold(0.0, f32::max);
        let min_synaptic_weight = weights.iter().cloned().fold(1.0, f32::min);
        
        // Count strong relationships
        let strong_relationships = weights.iter().filter(|&&w| w > 0.7).count();
        let weak_relationships = weights.iter().filter(|&&w| w < 0.3).count();
        
        RelationshipStatistics {
            total_relationships: core_stats.total_relationships,
            avg_synaptic_weight,
            max_synaptic_weight,
            min_synaptic_weight,
            strong_relationships,
            weak_relationships,
            avg_degree: core_stats.average_degree,
        }
    }

    /// Prune weak relationships
    pub async fn prune_weak_relationships(&self, threshold: f32) -> Result<usize> {
        let synaptic_weights = self.synaptic_weights.read().await;
        let weak_relationships: Vec<(EntityKey, EntityKey)> = synaptic_weights
            .iter()
            .filter(|(_, &weight)| weight < threshold)
            .map(|((source, target), _)| (*source, *target))
            .collect();
        
        drop(synaptic_weights);
        
        let mut pruned_count = 0;
        for (source, target) in weak_relationships {
            if self.remove_relationship(source, target).await? {
                pruned_count += 1;
            }
        }
        
        Ok(pruned_count)
    }

    /// Get highly connected entities
    pub async fn get_highly_connected_entities(&self, min_connections: usize) -> Vec<(EntityKey, usize)> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut highly_connected = Vec::new();
        
        for entity_key in entity_keys {
            let connections = self.core_graph.get_neighbors(entity_key).len();
            if connections >= min_connections {
                highly_connected.push((entity_key, connections));
            }
        }
        
        // Sort by connection count (descending)
        highly_connected.sort_by(|a, b| b.1.cmp(&a.1));
        
        highly_connected
    }

    /// Get entities with strongest outgoing connections
    pub async fn get_strongest_senders(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut sender_strengths = Vec::new();
        
        for entity_key in entity_keys {
            let children = self.get_child_entities(entity_key).await;
            let total_outgoing_weight: f32 = children.iter().map(|(_, weight)| weight).sum();
            
            if total_outgoing_weight > 0.0 {
                sender_strengths.push((entity_key, total_outgoing_weight));
            }
        }
        
        // Sort by outgoing weight (descending)
        sender_strengths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sender_strengths.truncate(k);
        
        sender_strengths
    }

    /// Get entities with strongest incoming connections
    pub async fn get_strongest_receivers(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut receiver_strengths = Vec::new();
        
        for entity_key in entity_keys {
            let parents = self.get_parent_entities(entity_key).await;
            let total_incoming_weight: f32 = parents.iter().map(|(_, weight)| weight).sum();
            
            if total_incoming_weight > 0.0 {
                receiver_strengths.push((entity_key, total_incoming_weight));
            }
        }
        
        // Sort by incoming weight (descending)
        receiver_strengths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        receiver_strengths.truncate(k);
        
        receiver_strengths
    }

    /// Find bridge entities (entities that connect different clusters)
    pub async fn find_bridge_entities(&self) -> Vec<EntityKey> {
        let mut bridge_entities = Vec::new();
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        for entity_key in entity_keys {
            if self.is_bridge_entity(entity_key).await {
                bridge_entities.push(entity_key);
            }
        }
        
        bridge_entities
    }

    /// Check if entity is a bridge (connects different clusters)
    async fn is_bridge_entity(&self, entity: EntityKey) -> bool {
        let neighbors = self.get_neighbors_with_weights(entity).await;
        
        if neighbors.len() < 2 {
            return false;
        }
        
        // Check if removing this entity would disconnect its neighbors
        let neighbor_keys: Vec<EntityKey> = neighbors.iter().map(|(key, _)| *key).collect();
        
        // Simplified check: if neighbors are not connected to each other, entity is a bridge
        for i in 0..neighbor_keys.len() {
            for j in i + 1..neighbor_keys.len() {
                if !self.has_relationship(neighbor_keys[i], neighbor_keys[j]).await &&
                   !self.has_relationship(neighbor_keys[j], neighbor_keys[i]).await {
                    return true;
                }
            }
        }
        
        false
    }

    /// Calculate clustering coefficient for entity
    pub async fn calculate_clustering_coefficient(&self, entity: EntityKey) -> f32 {
        let neighbors = self.get_neighbors_with_weights(entity).await;
        let neighbor_keys: Vec<EntityKey> = neighbors.iter().map(|(key, _)| *key).collect();
        
        if neighbor_keys.len() < 2 {
            return 0.0;
        }
        
        let mut connected_pairs = 0;
        let total_pairs = neighbor_keys.len() * (neighbor_keys.len() - 1) / 2;
        
        for i in 0..neighbor_keys.len() {
            for j in i + 1..neighbor_keys.len() {
                if self.has_relationship(neighbor_keys[i], neighbor_keys[j]).await ||
                   self.has_relationship(neighbor_keys[j], neighbor_keys[i]).await {
                    connected_pairs += 1;
                }
            }
        }
        
        if total_pairs == 0 {
            0.0
        } else {
            connected_pairs as f32 / total_pairs as f32
        }
    }

    /// Get relationship patterns
    pub async fn get_relationship_patterns(&self) -> Vec<RelationshipPattern> {
        let mut patterns = Vec::new();
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        // Find star patterns (one entity connected to many)
        for entity_key in &entity_keys {
            let outgoing = self.get_child_entities(*entity_key).await;
            let incoming = self.get_parent_entities(*entity_key).await;
            
            if outgoing.len() > 5 && incoming.len() < 2 {
                patterns.push(RelationshipPattern::Star {
                    center: *entity_key,
                    spokes: outgoing.len(),
                });
            }
            
            if incoming.len() > 5 && outgoing.len() < 2 {
                patterns.push(RelationshipPattern::Hub {
                    center: *entity_key,
                    connections: incoming.len(),
                });
            }
        }
        
        patterns
    }
}

/// Relationship statistics
#[derive(Debug, Clone)]
pub struct RelationshipStatistics {
    pub total_relationships: usize,
    pub avg_synaptic_weight: f32,
    pub max_synaptic_weight: f32,
    pub min_synaptic_weight: f32,
    pub strong_relationships: usize,
    pub weak_relationships: usize,
    pub avg_degree: f64,
}

impl RelationshipStatistics {
    /// Get weight range
    pub fn weight_range(&self) -> f32 {
        self.max_synaptic_weight - self.min_synaptic_weight
    }
    
    /// Get strong relationship ratio
    pub fn strong_relationship_ratio(&self) -> f32 {
        if self.total_relationships == 0 {
            0.0
        } else {
            self.strong_relationships as f32 / self.total_relationships as f32
        }
    }
    
    /// Get weak relationship ratio
    pub fn weak_relationship_ratio(&self) -> f32 {
        if self.total_relationships == 0 {
            0.0
        } else {
            self.weak_relationships as f32 / self.total_relationships as f32
        }
    }
    
    /// Check if relationships are well-balanced
    pub fn is_well_balanced(&self) -> bool {
        let strong_ratio = self.strong_relationship_ratio();
        let weak_ratio = self.weak_relationship_ratio();
        
        strong_ratio > 0.1 && strong_ratio < 0.5 && weak_ratio < 0.3
    }
}

/// Relationship patterns
#[derive(Debug, Clone)]
pub enum RelationshipPattern {
    Star { center: EntityKey, spokes: usize },
    Hub { center: EntityKey, connections: usize },
    Chain { entities: Vec<EntityKey> },
    Cluster { entities: Vec<EntityKey> },
}

impl RelationshipPattern {
    /// Get pattern type name
    pub fn pattern_type(&self) -> &'static str {
        match self {
            RelationshipPattern::Star { .. } => "star",
            RelationshipPattern::Hub { .. } => "hub",
            RelationshipPattern::Chain { .. } => "chain",
            RelationshipPattern::Cluster { .. } => "cluster",
        }
    }
    
    /// Get pattern size
    pub fn size(&self) -> usize {
        match self {
            RelationshipPattern::Star { spokes, .. } => *spokes,
            RelationshipPattern::Hub { connections, .. } => *connections,
            RelationshipPattern::Chain { entities } => entities.len(),
            RelationshipPattern::Cluster { entities } => entities.len(),
        }
    }
}