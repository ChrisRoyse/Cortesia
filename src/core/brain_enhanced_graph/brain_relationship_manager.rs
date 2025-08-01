//! Relationship management for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use crate::core::types::{EntityKey, Relationship};
use crate::error::Result;
use std::collections::{HashSet, VecDeque};

/// Trait for adding relationships with different signatures
#[allow(async_fn_in_trait)]
pub trait AddRelationship<T> {
    async fn add_relationship(&self, source: T, target: T, weight: f32) -> Result<()>;
    async fn add_relationship_with_type(&self, source: T, target: T, rel_type: T, weight: f32) -> Result<()>;
}

/// Implementation for numeric IDs (i32)
impl AddRelationship<i32> for BrainEnhancedKnowledgeGraph {
    async fn add_relationship(&self, source: i32, target: i32, weight: f32) -> Result<()> {
        self.add_connection(source as u32, target as u32, weight).await
    }
    
    async fn add_relationship_with_type(&self, source: i32, target: i32, _rel_type: i32, weight: f32) -> Result<()> {
        self.add_connection(source as u32, target as u32, weight).await
    }
}

/// Implementation for string IDs (&str)
impl AddRelationship<&str> for BrainEnhancedKnowledgeGraph {
    async fn add_relationship(&self, source: &str, target: &str, weight: f32) -> Result<()> {
        self.add_relationship_with_type(source, target, "", weight).await
    }
    
    async fn add_relationship_with_type(&self, source: &str, target: &str, rel_type: &str, weight: f32) -> Result<()> {
        // Convert string IDs to numeric IDs using hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let source_id = (hasher.finish() as u32) % 1000000 + 1;
        
        let mut hasher = DefaultHasher::new();
        target.hash(&mut hasher);
        let target_id = (hasher.finish() as u32) % 1000000 + 1;
        
        // For now, ignore relationship_type as it's not stored in the current structure
        let _ = rel_type;
        
        self.add_connection(source_id, target_id, weight).await
    }
}

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

    /// Add weighted edge (test compatibility method)
    pub async fn add_weighted_edge(&self, from: EntityKey, to: EntityKey, weight: f32) -> Result<()> {
        let relationship = Relationship {
            from,
            to,
            rel_type: 0, // Default relationship type
            weight,
        };
        self.insert_brain_relationship(relationship).await
    }

    /// Add connection (test compatibility method) - takes entity IDs instead of EntityKeys
    pub async fn add_connection(&self, source_id: u32, target_id: u32, weight: f32) -> Result<()> {
        // Try to get EntityKeys from IDs using the core graph's lookup
        if let (Some(source_key), Some(target_key)) = (
            self.core_graph.get_entity_key(source_id),
            self.core_graph.get_entity_key(target_id)
        ) {
            return self.add_weighted_edge(source_key, target_key, weight).await;
        }
        
        // If entities don't exist, create them first
        let source_key = if let Some(key) = self.core_graph.get_entity_key(source_id) {
            key
        } else {
            // Create a default entity with the ID
            let entity_data = crate::core::types::EntityData {
                type_id: 1,
                embedding: vec![0.5; self.embedding_dimension()],
                properties: format!("{{\"id\": {source_id}}}"),
            };
            self.insert_brain_entity(source_id, entity_data).await?
        };
        
        let target_key = if let Some(key) = self.core_graph.get_entity_key(target_id) {
            key
        } else {
            // Create a default entity with the ID
            let entity_data = crate::core::types::EntityData {
                type_id: 1,
                embedding: vec![0.5; self.embedding_dimension()],
                properties: format!("{{\"id\": {target_id}}}"),
            };
            self.insert_brain_entity(target_id, entity_data).await?
        };
        
        self.add_weighted_edge(source_key, target_key, weight).await
    }

    /// Add relationship with entity keys directly  
    pub async fn add_relationship_keys(&self, source_key: EntityKey, target_key: EntityKey, weight: f32) -> Result<()> {
        self.add_weighted_edge(source_key, target_key, weight).await
    }

    /// Add relationship with string IDs and type - now handled by trait (4 parameters: source, target, rel_type, weight)
    pub async fn add_relationship_str(&self, source_str: &str, target_str: &str, relationship_type: &str, weight: f32) -> Result<()> {
        // Convert string IDs to numeric IDs using hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source_str.hash(&mut hasher);
        let source_id = (hasher.finish() as u32) % 1000000 + 1;
        
        let mut hasher = DefaultHasher::new();
        target_str.hash(&mut hasher);
        let target_id = (hasher.finish() as u32) % 1000000 + 1;
        
        // For now, ignore relationship_type as it's not stored in the current structure
        let _ = relationship_type;
        
        self.add_connection(source_id, target_id, weight).await
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

    /// Reset all entity activations to 0.0
    pub async fn reset_all_activations(&self) {
        let mut activations = self.entity_activations.write().await;
        activations.clear();
    }

    /// Get the current brain configuration
    pub async fn get_configuration(&self) -> super::brain_graph_types::BrainEnhancedConfig {
        self.config.clone()
    }

    /// Count relationships by type
    pub async fn count_relationships_by_type(&self, rel_type: u32) -> usize {
        let all_relationships = self.core_graph.get_all_relationships();
        all_relationships.iter()
            .filter(|rel| rel.rel_type == rel_type as u8)
            .count()
    }

    /// Analyze weight distribution of relationships
    pub async fn analyze_weight_distribution(&self) -> WeightDistribution {
        let weights = self.synaptic_weights.read().await;
        let weight_values: Vec<f32> = weights.values().cloned().collect();
        
        if weight_values.is_empty() {
            return WeightDistribution {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        
        let sum: f32 = weight_values.iter().sum();
        let mean = sum / weight_values.len() as f32;
        
        let variance = weight_values.iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>() / weight_values.len() as f32;
        let std_dev = variance.sqrt();
        
        let min = weight_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weight_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        WeightDistribution {
            mean,
            std_dev,
            min,
            max,
        }
    }

    /// Batch insert relationships
    pub async fn batch_insert_relationships(&self, relationships: &[crate::core::types::Relationship]) -> Result<()> {
        for relationship in relationships {
            self.insert_brain_relationship(*relationship).await?;
        }
        Ok(())
    }

    /// Batch update relationship weights
    pub async fn batch_update_relationship_weights(&self, updates: &[(EntityKey, EntityKey, f32)]) -> Result<()> {
        for &(from, to, new_weight) in updates {
            self.update_relationship_weight(from, to, new_weight).await?;
        }
        Ok(())
    }

    /// Batch strengthen relationships
    pub async fn batch_strengthen_relationships(&self, updates: &[(EntityKey, EntityKey, f32)]) -> Result<()> {
        for &(from, to, strength_delta) in updates {
            let current_weight = self.get_synaptic_weight(from, to).await;
            let new_weight = (current_weight + strength_delta).clamp(0.0, 1.0);
            self.update_relationship_weight(from, to, new_weight).await?;
        }
        Ok(())
    }

    /// Batch weaken relationships
    pub async fn batch_weaken_relationships(&self, updates: &[(EntityKey, EntityKey, f32)]) -> Result<()> {
        for &(from, to, weaken_delta) in updates {
            let current_weight = self.get_synaptic_weight(from, to).await;
            let new_weight = (current_weight - weaken_delta).max(0.0);
            
            if new_weight < 0.01 {
                self.remove_relationship(from, to).await?;
            } else {
                self.update_relationship_weight(from, to, new_weight).await?;
            }
        }
        Ok(())
    }

    /// Batch remove relationships
    pub async fn batch_remove_relationships(&self, pairs: &[(EntityKey, EntityKey)]) -> Result<()> {
        for &(from, to) in pairs {
            self.remove_relationship(from, to).await?;
        }
        Ok(())
    }
}

/// Weight distribution statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeightDistribution {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{EntityData, Relationship};
    
    
    use tokio;

    /// Helper function to create a test brain graph
    async fn create_test_graph() -> Result<BrainEnhancedKnowledgeGraph> {
        BrainEnhancedKnowledgeGraph::new_for_test()
    }

    /// Helper function to create test entities
    async fn create_test_entities(graph: &BrainEnhancedKnowledgeGraph, count: usize) -> Result<Vec<EntityKey>> {
        let mut entity_keys = Vec::new();
        
        for i in 0..count {
            let entity_data = EntityData {
                type_id: 1,
                embedding: vec![0.1 * (i as f32 + 1.0); 96],
                properties: format!("{{\"name\": \"test_entity_{i}\"}}"),
            };
            
            let entity_key = graph.insert_brain_entity(i as u32, entity_data).await?;
            entity_keys.push(entity_key);
        }
        
        Ok(entity_keys)
    }

    /// Helper function to create a test relationship
    fn create_test_relationship(from: EntityKey, to: EntityKey, weight: f32) -> Relationship {
        Relationship {
            from,
            to,
            rel_type: 0,
            weight,
        }
    }

    #[tokio::test]
    async fn test_create_learned_relationship_with_co_activated_entities() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set high activations for both entities (co-activation)
        graph.set_entity_activation(entities[0], 0.8).await;
        graph.set_entity_activation(entities[1], 0.9).await;
        
        // Test creating learned relationship
        let result = graph.create_learned_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify relationship was created
        assert!(graph.has_relationship(entities[0], entities[1]).await);
        
        // Check synaptic weight was set correctly
        let weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        let expected_weight = 0.8 * 0.9 * graph.config.learning_rate; // source * target * learning_rate
        assert!((weight - expected_weight).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_create_learned_relationship_with_non_co_activated_entities() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set low activations (no co-activation)
        graph.set_entity_activation(entities[0], 0.1).await;
        graph.set_entity_activation(entities[1], 0.05).await;
        
        // Test creating learned relationship
        let result = graph.create_learned_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify no relationship was created due to low Hebbian weight
        let has_relationship = graph.has_relationship(entities[0], entities[1]).await;
        assert!(!has_relationship);
    }

    #[tokio::test]
    async fn test_create_learned_relationship_with_hebbian_learning_disabled() {
        let mut graph = create_test_graph().await.unwrap();
        
        // Disable Hebbian learning
        let mut config = graph.config.clone();
        config.enable_hebbian_learning = false;
        graph.config = config;
        
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set high activations
        graph.set_entity_activation(entities[0], 0.8).await;
        graph.set_entity_activation(entities[1], 0.9).await;
        
        // Test creating learned relationship
        let result = graph.create_learned_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify no relationship was created due to disabled learning
        assert!(!graph.has_relationship(entities[0], entities[1]).await);
    }

    #[tokio::test]
    async fn test_create_learned_relationship_with_threshold_edge_case() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set activations that result in exactly threshold weight (0.1)
        let learning_rate = graph.config.learning_rate;
        let activation = (0.1 / learning_rate).sqrt(); // activation^2 * learning_rate = 0.1
        
        graph.set_entity_activation(entities[0], activation).await;
        graph.set_entity_activation(entities[1], activation).await;
        
        let result = graph.create_learned_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Should create relationship at threshold
        assert!(graph.has_relationship(entities[0], entities[1]).await);
    }

    #[tokio::test]
    async fn test_strengthen_relationship_existing_relationship() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Create initial relationship
        let relationship = create_test_relationship(entities[0], entities[1], 0.5);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        // Set high activations for strengthening
        graph.set_entity_activation(entities[0], 0.7).await;
        graph.set_entity_activation(entities[1], 0.8).await;
        
        let initial_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        
        // Test strengthening
        let result = graph.strengthen_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify weight increased
        let new_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert!(new_weight > initial_weight);
        
        // Check strengthening calculation
        let expected_strengthening = (0.7 * 0.8 * graph.config.learning_rate).min(0.1);
        let expected_weight = (initial_weight + expected_strengthening).clamp(0.0, 1.0);
        assert!((new_weight - expected_weight).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_strengthen_relationship_nonexistent_relationship() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set activations
        graph.set_entity_activation(entities[0], 0.7).await;
        graph.set_entity_activation(entities[1], 0.8).await;
        
        // Test strengthening non-existent relationship
        let result = graph.strengthen_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify no relationship was created or modified
        assert!(!graph.has_relationship(entities[0], entities[1]).await);
        
        // Synaptic weight should remain at default
        let weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert_eq!(weight, 0.1); // Default weight
    }

    #[tokio::test]
    async fn test_strengthen_relationship_weight_clamping() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Create relationship with high initial weight
        let relationship = create_test_relationship(entities[0], entities[1], 0.95);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        // Set very high activations to test clamping
        graph.set_entity_activation(entities[0], 1.0).await;
        graph.set_entity_activation(entities[1], 1.0).await;
        
        // Test strengthening
        let result = graph.strengthen_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify weight is clamped to 1.0
        let weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert!(weight <= 1.0);
    }

    #[tokio::test]
    async fn test_weaken_relationship_existing_relationship() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Create relationship with moderate weight
        let relationship = create_test_relationship(entities[0], entities[1], 0.5);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        let initial_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        
        // Test weakening
        let result = graph.weaken_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify weight decreased
        let new_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert!(new_weight < initial_weight);
        
        // Check weakening calculation
        let decay_factor = 1.0 - graph.config.synaptic_strength_decay;
        let expected_weight = initial_weight * (1.0 - decay_factor);
        assert!((new_weight - expected_weight).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_weaken_relationship_removes_very_weak_relationship() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Create relationship with very low weight
        let relationship = create_test_relationship(entities[0], entities[1], 0.005);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        // Test weakening
        let result = graph.weaken_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify relationship was removed
        assert!(!graph.has_relationship(entities[0], entities[1]).await);
    }

    #[tokio::test]
    async fn test_weaken_relationship_nonexistent_relationship() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Test weakening non-existent relationship
        let result = graph.weaken_relationship(entities[0], entities[1]).await;
        assert!(result.is_ok());
        
        // Verify no relationship exists
        assert!(!graph.has_relationship(entities[0], entities[1]).await);
    }

    #[tokio::test]
    async fn test_get_neighbors_with_weights_sorted_order() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 4).await.unwrap();
        
        // Create relationships with different weights
        let relationships = vec![
            create_test_relationship(entities[0], entities[1], 0.3),
            create_test_relationship(entities[0], entities[2], 0.8),
            create_test_relationship(entities[0], entities[3], 0.5),
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        // Get neighbors with weights
        let neighbors = graph.get_neighbors_with_weights(entities[0]).await;
        
        // Verify count
        assert_eq!(neighbors.len(), 3);
        
        // Verify sorted order (descending by weight)
        assert_eq!(neighbors[0].0, entities[2]); // Highest weight (0.8)
        assert_eq!(neighbors[1].0, entities[3]); // Medium weight (0.5)
        assert_eq!(neighbors[2].0, entities[1]); // Lowest weight (0.3)
        
        // Verify weights
        assert!((neighbors[0].1 - 0.8).abs() < 0.001);
        assert!((neighbors[1].1 - 0.5).abs() < 0.001);
        assert!((neighbors[2].1 - 0.3).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_get_neighbors_with_weights_no_neighbors() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 1).await.unwrap();
        
        // Get neighbors for entity with no relationships
        let neighbors = graph.get_neighbors_with_weights(entities[0]).await;
        
        // Verify empty result
        assert!(neighbors.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_path_weight_simple_path() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 3).await.unwrap();
        
        // Create path: entity[0] -> entity[1] -> entity[2]
        let relationships = vec![
            create_test_relationship(entities[0], entities[1], 0.6),
            create_test_relationship(entities[1], entities[2], 0.8),
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let path = vec![entities[0], entities[1], entities[2]];
        let path_weight = graph.calculate_path_weight(&path).await;
        
        // Expected weight: 0.6 * 0.8 * (1.0 / sqrt(3)) = path weight * length penalty
        let expected_weight = 0.6 * 0.8 * (1.0 / (3.0_f32).sqrt());
        assert!((path_weight - expected_weight).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_calculate_path_weight_single_entity() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 1).await.unwrap();
        
        let path = vec![entities[0]];
        let path_weight = graph.calculate_path_weight(&path).await;
        
        // Single entity path should have 0 weight
        assert_eq!(path_weight, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_path_weight_empty_path() {
        let graph = create_test_graph().await.unwrap();
        
        let path = vec![];
        let path_weight = graph.calculate_path_weight(&path).await;
        
        // Empty path should have 0 weight
        assert_eq!(path_weight, 0.0);
    }

    #[tokio::test]
    async fn test_generate_path_signature_consistency() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 3).await.unwrap();
        
        let path1 = vec![entities[0], entities[1], entities[2]];
        let path2 = vec![entities[0], entities[1], entities[2]];
        let path3 = vec![entities[2], entities[1], entities[0]];
        
        let sig1 = graph.generate_path_signature(&path1);
        let sig2 = graph.generate_path_signature(&path2);
        let sig3 = graph.generate_path_signature(&path3);
        
        // Same paths should have same signature
        assert_eq!(sig1, sig2);
        
        // Different paths should have different signatures
        assert_ne!(sig1, sig3);
        
        // Signatures should not be empty
        assert!(!sig1.is_empty());
    }

    #[tokio::test]
    async fn test_is_bridge_entity_true_case() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 4).await.unwrap();
        
        // Create bridge pattern: entity[1] - entity[0] - entity[2]
        // entity[3] connected to entity[0] but not to entity[1] or entity[2]
        let relationships = vec![
            create_test_relationship(entities[1], entities[0], 0.5),
            create_test_relationship(entities[0], entities[2], 0.5),
            create_test_relationship(entities[0], entities[3], 0.5),
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        // entity[0] should be a bridge
        let is_bridge = graph.is_bridge_entity(entities[0]).await;
        assert!(is_bridge);
    }

    #[tokio::test]
    async fn test_is_bridge_entity_false_case() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 3).await.unwrap();
        
        // Create triangle: all entities connected to each other
        let relationships = vec![
            create_test_relationship(entities[0], entities[1], 0.5),
            create_test_relationship(entities[1], entities[2], 0.5),
            create_test_relationship(entities[2], entities[0], 0.5),
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        // No entity should be a bridge in a triangle
        let is_bridge = graph.is_bridge_entity(entities[0]).await;
        assert!(!is_bridge);
    }

    #[tokio::test]
    async fn test_is_bridge_entity_insufficient_neighbors() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Create single relationship
        let relationship = create_test_relationship(entities[0], entities[1], 0.5);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        // Entity with only one neighbor cannot be a bridge
        let is_bridge = graph.is_bridge_entity(entities[0]).await;
        assert!(!is_bridge);
    }

    #[tokio::test]
    async fn test_synaptic_weight_management_internal() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Test default weight
        let default_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert_eq!(default_weight, 0.1);
        
        // Test setting weight
        graph.set_synaptic_weight(entities[0], entities[1], 0.7).await;
        let weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert!((weight - 0.7).abs() < 0.001);
        
        // Test updating weight through relationship
        graph.update_relationship_weight(entities[0], entities[1], 0.9).await.unwrap();
        let updated_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        assert!((updated_weight - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_learning_algorithm_hebbian_weight_calculation() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Test various activation combinations
        let test_cases = vec![
            (0.0, 0.0, 0.0), // No activation
            (1.0, 1.0, graph.config.learning_rate), // Full activation
            (0.5, 0.6, 0.5 * 0.6 * graph.config.learning_rate), // Partial activation
        ];
        
        for (source_activation, target_activation, expected_hebbian) in test_cases {
            graph.set_entity_activation(entities[0], source_activation).await;
            graph.set_entity_activation(entities[1], target_activation).await;
            
            // Test create_learned_relationship
            graph.create_learned_relationship(entities[0], entities[1]).await.unwrap();
            
            if expected_hebbian > 0.1 {
                assert!(graph.has_relationship(entities[0], entities[1]).await);
                let weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
                assert!((weight - expected_hebbian).abs() < 0.001);
                
                // Clean up for next test
                graph.remove_relationship(entities[0], entities[1]).await.unwrap();
            } else {
                assert!(!graph.has_relationship(entities[0], entities[1]).await);
            }
        }
    }

    #[tokio::test]
    async fn test_relationship_statistics_calculation() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 4).await.unwrap();
        
        // Create relationships with different weights
        let relationships = vec![
            create_test_relationship(entities[0], entities[1], 0.2), // Weak
            create_test_relationship(entities[1], entities[2], 0.8), // Strong
            create_test_relationship(entities[2], entities[3], 0.5), // Medium
            create_test_relationship(entities[3], entities[0], 0.9), // Strong
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let stats = graph.get_relationship_statistics().await;
        
        assert_eq!(stats.total_relationships, 4);
        assert_eq!(stats.strong_relationships, 2); // Weights > 0.7
        assert_eq!(stats.weak_relationships, 1);   // Weights < 0.3
        
        // Check average weight calculation
        let expected_avg = (0.2 + 0.8 + 0.5 + 0.9) / 4.0;
        assert!((stats.avg_synaptic_weight - expected_avg).abs() < 0.001);
        
        assert!((stats.max_synaptic_weight - 0.9).abs() < 0.001);
        assert!((stats.min_synaptic_weight - 0.2).abs() < 0.001);
        
        // Test helper methods
        assert!((stats.weight_range() - 0.7).abs() < 0.001);
        assert!((stats.strong_relationship_ratio() - 0.5).abs() < 0.001);
        assert!((stats.weak_relationship_ratio() - 0.25).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_concurrent_relationship_operations() {
        let graph = std::sync::Arc::new(create_test_graph().await.unwrap());
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Test concurrent synaptic weight operations
        let mut handles = vec![];
        
        for i in 0..10 {
            let graph_clone = graph.clone();
            let entities_clone = entities.clone();
            
            let handle = tokio::spawn(async move {
                let weight = 0.1 + (i as f32 * 0.05);
                graph_clone.set_synaptic_weight(entities_clone[0], entities_clone[1], weight).await;
                graph_clone.get_synaptic_weight(entities_clone[0], entities_clone[1]).await
            });
            
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!((0.1..=0.6).contains(&result)); // Should be within expected range
        }
    }

    #[tokio::test]
    async fn test_edge_case_zero_activations() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 2).await.unwrap();
        
        // Set zero activations
        graph.set_entity_activation(entities[0], 0.0).await;
        graph.set_entity_activation(entities[1], 0.0).await;
        
        // Test learned relationship creation
        graph.create_learned_relationship(entities[0], entities[1]).await.unwrap();
        assert!(!graph.has_relationship(entities[0], entities[1]).await);
        
        // Create relationship manually and test strengthening
        let relationship = create_test_relationship(entities[0], entities[1], 0.5);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        let initial_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        graph.strengthen_relationship(entities[0], entities[1]).await.unwrap();
        let new_weight = graph.get_synaptic_weight(entities[0], entities[1]).await;
        
        // Weight should remain the same (no strengthening with zero activations)
        assert!((new_weight - initial_weight).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_relationship_pattern_detection() {
        let graph = create_test_graph().await.unwrap();
        let entities = create_test_entities(&graph, 7).await.unwrap();
        
        // Create star pattern: entity[0] connected to entities[1-5]
        for i in 1..6 {
            let rel = create_test_relationship(entities[0], entities[i], 0.5);
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        // Create hub pattern: entities[1-5] connected to entity[6]
        for i in 1..6 {
            let rel = create_test_relationship(entities[i], entities[6], 0.5);
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let patterns = graph.get_relationship_patterns().await;
        
        // Should find both star and hub patterns
        assert!(patterns.len() >= 2);
        
        let pattern_types: Vec<&str> = patterns.iter().map(|p| p.pattern_type()).collect();
        assert!(pattern_types.contains(&"star"));
        assert!(pattern_types.contains(&"hub"));
    }
}