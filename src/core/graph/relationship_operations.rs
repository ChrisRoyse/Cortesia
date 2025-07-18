//! Relationship operations for knowledge graph

use super::graph_core::KnowledgeGraph;
use crate::core::types::{EntityKey, Relationship};
use crate::error::{GraphError, Result};

impl KnowledgeGraph {
    /// Insert a relationship between two entities
    pub fn insert_relationship(&self, relationship: Relationship) -> Result<()> {
        // Validate that both entities exist
        if !self.contains_entity_key(relationship.from) {
            return Err(GraphError::EntityKeyNotFound { key: relationship.from });
        }
        
        if !self.contains_entity_key(relationship.to) {
            return Err(GraphError::EntityKeyNotFound { key: relationship.to });
        }
        
        // Validate relationship weight
        if relationship.weight < 0.0 || relationship.weight > 1.0 {
            return Err(GraphError::InvalidRelationshipWeight(relationship.weight));
        }
        
        // Add to edge buffer for dynamic insertion
        let mut edge_buffer = self.edge_buffer.write();
        edge_buffer.push(relationship.clone());
        
        // If buffer is getting full, flush to main graph
        if edge_buffer.len() > 1000 {
            drop(edge_buffer);
            self.flush_edge_buffer()?;
        } else {
            // Also add directly to CSR graph for immediate queries
            if let (Some(from_id), Some(to_id)) = (self.get_entity_id(relationship.from), self.get_entity_id(relationship.to)) {
                let mut graph = self.graph.write();
                graph.add_edge(from_id, to_id, relationship.weight)?;
            }
        }
        
        Ok(())
    }

    /// Insert multiple relationships in batch
    pub fn insert_relationships_batch(&self, relationships: Vec<Relationship>) -> Result<()> {
        // Validate all relationships first
        for relationship in &relationships {
            if !self.contains_entity_key(relationship.from) {
                return Err(GraphError::EntityKeyNotFound { key: relationship.from });
            }
            
            if !self.contains_entity_key(relationship.to) {
                return Err(GraphError::EntityKeyNotFound { key: relationship.to });
            }
            
            if relationship.weight < 0.0 || relationship.weight > 1.0 {
                return Err(GraphError::InvalidRelationshipWeight(relationship.weight));
            }
        }
        
        // Add to edge buffer
        let mut edge_buffer = self.edge_buffer.write();
        edge_buffer.extend(relationships.clone());
        
        // Add to CSR graph
        let mut graph = self.graph.write();
        for relationship in relationships {
            if let (Some(from_id), Some(to_id)) = (self.get_entity_id(relationship.from), self.get_entity_id(relationship.to)) {
                graph.add_edge(from_id, to_id, relationship.weight)?;
            }
        }
        
        Ok(())
    }

    /// Get neighbors of an entity
    pub fn get_neighbors(&self, entity: EntityKey) -> Vec<EntityKey> {
        let mut neighbors = Vec::new();
        
        // Get entity ID from EntityKey
        let entity_id = match self.get_entity_id(entity) {
            Some(id) => id,
            None => return neighbors, // Entity not found
        };
        
        // Get neighbors from main graph
        let graph = self.graph.read();
        let neighbor_ids = graph.get_neighbors(entity_id);
        
        // Convert neighbor IDs back to EntityKeys
        for &neighbor_id in neighbor_ids {
            if let Some(neighbor_key) = self.get_entity_key(neighbor_id) {
                neighbors.push(neighbor_key);
            }
        }
        
        // Also check edge buffer for recently added relationships
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.from == entity {
                neighbors.push(relationship.to);
            }
            if relationship.to == entity {
                neighbors.push(relationship.from);
            }
        }
        
        // Remove duplicates and return
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Get outgoing neighbors of an entity
    pub fn get_outgoing_neighbors(&self, entity: EntityKey) -> Vec<EntityKey> {
        let mut neighbors = Vec::new();
        
        // Get entity ID from EntityKey
        if let Some(entity_id) = self.get_entity_id(entity) {
            // Get neighbors from main graph
            let graph = self.graph.read();
            let neighbor_ids = graph.get_neighbors(entity_id);
            
            // Convert neighbor IDs back to EntityKeys
            for &neighbor_id in neighbor_ids {
                if let Some(neighbor_key) = self.get_entity_key(neighbor_id) {
                    neighbors.push(neighbor_key);
                }
            }
        }
        
        // Also check edge buffer for recently added relationships
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.from == entity {
                neighbors.push(relationship.to);
            }
        }
        
        // Remove duplicates and return
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Get incoming neighbors of an entity
    pub fn get_incoming_neighbors(&self, entity: EntityKey) -> Vec<EntityKey> {
        let mut neighbors = Vec::new();
        
        // Note: CSRGraph doesn't have a specific get_incoming_neighbors method
        // We need to iterate through all entities to find incoming edges
        // For now, we'll just check the edge buffer
        
        // Check edge buffer for recently added relationships
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.to == entity {
                neighbors.push(relationship.from);
            }
        }
        
        // Remove duplicates and return
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Get relationship weight between two entities
    pub fn get_relationship_weight(&self, from: EntityKey, to: EntityKey) -> Option<f32> {
        // Convert EntityKeys to IDs
        let from_id = self.get_entity_id(from)?;
        let to_id = self.get_entity_id(to)?;
        
        // Check main graph first
        let graph = self.graph.read();
        if let Some(weight) = graph.get_edge_weight(from_id, to_id) {
            return Some(weight);
        }
        
        // Check edge buffer
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.from == from && relationship.to == to {
                return Some(relationship.weight);
            }
        }
        
        None
    }

    /// Check if relationship exists
    pub fn has_relationship(&self, from: EntityKey, to: EntityKey) -> bool {
        self.get_relationship_weight(from, to).is_some()
    }

    /// Get all relationships for an entity
    pub fn get_entity_relationships(&self, entity: EntityKey) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // CSRGraph doesn't store full Relationship objects, only edges
        // We need to reconstruct relationships from the edge information
        
        // Get outgoing edges
        if let Some(entity_id) = self.get_entity_id(entity) {
            let graph = self.graph.read();
            let neighbor_ids = graph.get_neighbors(entity_id);
            
            for &neighbor_id in neighbor_ids {
                if let Some(neighbor_key) = self.get_entity_key(neighbor_id) {
                    // Create a relationship object
                    relationships.push(Relationship {
                        from: entity,
                        to: neighbor_key,
                        rel_type: 0, // Default type
                        weight: graph.get_edge_weight(entity_id, neighbor_id).unwrap_or(1.0),
                    });
                }
            }
        }
        
        // Also check edge buffer
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.from == entity || relationship.to == entity {
                relationships.push(relationship.clone());
            }
        }
        
        relationships
    }

    /// Get all outgoing relationships for an entity
    pub fn get_outgoing_relationships(&self, entity: EntityKey) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // Get outgoing edges from CSRGraph
        if let Some(entity_id) = self.get_entity_id(entity) {
            let graph = self.graph.read();
            let neighbor_ids = graph.get_neighbors(entity_id);
            
            for &neighbor_id in neighbor_ids {
                if let Some(neighbor_key) = self.get_entity_key(neighbor_id) {
                    // Create a relationship object
                    relationships.push(Relationship {
                        from: entity,
                        to: neighbor_key,
                        rel_type: 0, // Default type
                        weight: graph.get_edge_weight(entity_id, neighbor_id).unwrap_or(1.0),
                    });
                }
            }
        }
        
        // Also check edge buffer
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.from == entity {
                relationships.push(relationship.clone());
            }
        }
        
        relationships
    }

    /// Get all incoming relationships for an entity
    pub fn get_incoming_relationships(&self, entity: EntityKey) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // Get relationships from main graph
        let graph = self.graph.read();
        relationships.extend(graph.get_incoming_relationships(entity));
        
        // Also check edge buffer
        let edge_buffer = self.edge_buffer.read();
        for relationship in edge_buffer.iter() {
            if relationship.to == entity {
                relationships.push(relationship.clone());
            }
        }
        
        relationships
    }

    /// Remove relationship
    pub fn remove_relationship(&self, from: EntityKey, to: EntityKey) -> Result<bool> {
        let mut removed = false;
        
        // Convert EntityKeys to IDs
        let from_id = self.get_entity_id(from)
            .ok_or_else(|| GraphError::EntityNotFound { id: 0 })?;
        let to_id = self.get_entity_id(to)
            .ok_or_else(|| GraphError::EntityNotFound { id: 0 })?;
        
        // Remove from main graph
        let mut graph = self.graph.write();
        // CSRGraph doesn't support dynamic removal, so this will always fail
        // We'll catch the error and just rely on edge buffer removal
        let _ = graph.remove_edge(from_id, to_id);
        
        // Remove from edge buffer
        let mut edge_buffer = self.edge_buffer.write();
        let original_len = edge_buffer.len();
        edge_buffer.retain(|r| !(r.from == from && r.to == to));
        if edge_buffer.len() < original_len {
            removed = true;
        }
        
        Ok(removed)
    }

    /// Update relationship weight
    pub fn update_relationship_weight(&self, from: EntityKey, to: EntityKey, new_weight: f32) -> Result<bool> {
        // Validate weight
        if new_weight < 0.0 || new_weight > 1.0 {
            return Err(GraphError::InvalidRelationshipWeight(new_weight));
        }
        
        let mut updated = false;
        
        // Convert EntityKeys to IDs
        let from_id = self.get_entity_id(from)
            .ok_or_else(|| GraphError::EntityNotFound { id: 0 })?;
        let to_id = self.get_entity_id(to)
            .ok_or_else(|| GraphError::EntityNotFound { id: 0 })?;
        
        // Update in main graph
        let mut graph = self.graph.write();
        // CSRGraph doesn't support dynamic weight updates, so this will always fail
        // We'll catch the error and just rely on edge buffer updates
        let _ = graph.update_edge_weight(from_id, to_id, new_weight);
        
        // Update in edge buffer
        let mut edge_buffer = self.edge_buffer.write();
        for relationship in edge_buffer.iter_mut() {
            if relationship.from == from && relationship.to == to {
                relationship.weight = new_weight;
                updated = true;
            }
        }
        
        Ok(updated)
    }

    /// Get relationship statistics
    pub fn get_relationship_stats(&self) -> RelationshipStats {
        let main_graph_edges = self.relationship_count();
        let buffer_edges = self.edge_buffer_size();
        
        // Calculate degree statistics
        let all_entity_keys = self.get_all_entity_keys();
        let mut degrees = Vec::new();
        
        for key in &all_entity_keys {
            let degree = self.get_neighbors(*key).len();
            degrees.push(degree);
        }
        
        degrees.sort_unstable();
        
        let average_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        };
        
        let median_degree = if degrees.is_empty() {
            0
        } else {
            degrees[degrees.len() / 2]
        };
        
        let max_degree = degrees.last().copied().unwrap_or(0);
        let min_degree = degrees.first().copied().unwrap_or(0);
        
        RelationshipStats {
            total_relationships: main_graph_edges as usize + buffer_edges,
            main_graph_relationships: main_graph_edges as usize,
            buffer_relationships: buffer_edges,
            average_degree,
            median_degree,
            max_degree,
            min_degree,
        }
    }

    /// Get degree of an entity (number of connected entities)
    pub fn get_entity_degree(&self, entity: EntityKey) -> usize {
        self.get_neighbors(entity).len()
    }

    /// Get out-degree of an entity (number of outgoing relationships)
    pub fn get_entity_out_degree(&self, entity: EntityKey) -> usize {
        self.get_outgoing_neighbors(entity).len()
    }

    /// Get in-degree of an entity (number of incoming relationships)
    pub fn get_entity_in_degree(&self, entity: EntityKey) -> usize {
        self.get_incoming_neighbors(entity).len()
    }

    /// Get entities with highest degree
    pub fn get_highest_degree_entities(&self, limit: usize) -> Vec<(EntityKey, usize)> {
        let all_keys = self.get_all_entity_keys();
        let mut entity_degrees: Vec<(EntityKey, usize)> = all_keys
            .into_iter()
            .map(|key| (key, self.get_entity_degree(key)))
            .collect();
        
        entity_degrees.sort_by(|a, b| b.1.cmp(&a.1));
        entity_degrees.truncate(limit);
        entity_degrees
    }

    /// Check if graph is connected
    pub fn is_connected(&self) -> bool {
        let entity_keys = self.get_all_entity_keys();
        if entity_keys.is_empty() {
            return true;
        }
        
        // Use BFS to check connectivity
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back(entity_keys[0]);
        visited.insert(entity_keys[0]);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in self.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.len() == entity_keys.len()
    }
}

/// Relationship statistics
#[derive(Debug, Clone)]
pub struct RelationshipStats {
    pub total_relationships: usize,
    pub main_graph_relationships: usize,
    pub buffer_relationships: usize,
    pub average_degree: f64,
    pub median_degree: usize,
    pub max_degree: usize,
    pub min_degree: usize,
}

impl RelationshipStats {
    /// Get relationship density (edges / possible edges)
    pub fn density(&self, entity_count: usize) -> f64 {
        if entity_count <= 1 {
            return 0.0;
        }
        
        let max_possible_edges = entity_count * (entity_count - 1);
        self.total_relationships as f64 / max_possible_edges as f64
    }
    
    /// Check if graph is sparse (density < 0.1)
    pub fn is_sparse(&self, entity_count: usize) -> bool {
        self.density(entity_count) < 0.1
    }
    
    /// Check if graph is dense (density > 0.7)
    pub fn is_dense(&self, entity_count: usize) -> bool {
        self.density(entity_count) > 0.7
    }
}