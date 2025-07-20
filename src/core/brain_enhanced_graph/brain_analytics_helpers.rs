//! Helper functions for brain analytics

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use crate::core::types::EntityKey;
use std::collections::HashMap;

impl BrainEnhancedKnowledgeGraph {
    /// Calculate activation distribution histogram
    pub(crate) fn calculate_activation_distribution(&self, activation_values: &[f32]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for &value in activation_values {
            let bucket = if value < 0.1 {
                "0.0-0.1"
            } else if value < 0.2 {
                "0.1-0.2"
            } else if value < 0.3 {
                "0.2-0.3"
            } else if value < 0.4 {
                "0.3-0.4"
            } else if value < 0.5 {
                "0.4-0.5"
            } else if value < 0.6 {
                "0.5-0.6"
            } else if value < 0.7 {
                "0.6-0.7"
            } else if value < 0.8 {
                "0.7-0.8"
            } else if value < 0.9 {
                "0.8-0.9"
            } else {
                "0.9-1.0"
            };
            
            *distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Calculate clustering coefficient for a single entity
    pub(crate) async fn _calculate_clustering_coefficient_helper(&self, entity_key: EntityKey) -> f32 {
        let neighbors = self.get_neighbors(entity_key);
        
        if neighbors.len() < 2 {
            return 0.0;
        }
        
        let mut triangle_count = 0;
        let mut possible_triangles = 0;
        
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                possible_triangles += 1;
                
                // Check if neighbors[i] and neighbors[j] are connected
                if self.has_relationship(neighbors[i], neighbors[j]).await ||
                   self.has_relationship(neighbors[j], neighbors[i]).await {
                    triangle_count += 1;
                }
            }
        }
        
        if possible_triangles > 0 {
            triangle_count as f32 / possible_triangles as f32
        } else {
            0.0
        }
    }

    /// Calculate shortest path between two entities using BFS
    pub(crate) async fn calculate_shortest_path(&self, start: EntityKey, end: EntityKey) -> Option<usize> {
        if start == end {
            return Some(0);
        }
        
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back((start, 0));
        visited.insert(start);
        
        while let Some((current, distance)) = queue.pop_front() {
            let neighbors = self.get_neighbors(current);
            
            for neighbor in neighbors {
                if neighbor == end {
                    return Some(distance + 1);
                }
                
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, distance + 1));
                }
            }
        }
        
        None // No path found
    }

    /// Calculate betweenness centrality for all entities
    pub(crate) async fn calculate_betweenness_centrality(&self) -> HashMap<EntityKey, f32> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut centrality: HashMap<EntityKey, f32> = HashMap::new();
        
        // Initialize centrality scores
        for &entity in &entity_keys {
            centrality.insert(entity, 0.0);
        }
        
        // For each pair of entities, find shortest paths and count how many pass through each entity
        for i in 0..entity_keys.len() {
            for j in i + 1..entity_keys.len() {
                let start = entity_keys[i];
                let end = entity_keys[j];
                
                if let Some(path_entities) = self.find_shortest_path_entities(start, end).await {
                    let path_count = path_entities.len() as f32;
                    
                    // Count each intermediate entity in the path
                    for &entity in &path_entities[1..path_entities.len()-1] { // Exclude start and end
                        if let Some(score) = centrality.get_mut(&entity) {
                            *score += 1.0 / path_count;
                        }
                    }
                }
            }
        }
        
        // Normalize by the number of entity pairs
        let normalization_factor = if entity_keys.len() > 2 {
            ((entity_keys.len() - 1) * (entity_keys.len() - 2)) as f32 / 2.0
        } else {
            1.0
        };
        
        for score in centrality.values_mut() {
            *score /= normalization_factor;
        }
        
        centrality
    }

    /// Find shortest path entities between two nodes
    async fn find_shortest_path_entities(&self, start: EntityKey, end: EntityKey) -> Option<Vec<EntityKey>> {
        if start == end {
            return Some(vec![start]);
        }
        
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent: HashMap<EntityKey, EntityKey> = HashMap::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            let neighbors = self.get_neighbors(current);
            
            for neighbor in neighbors {
                if neighbor == end {
                    // Reconstruct path
                    let mut path = Vec::new();
                    path.push(end);
                    path.push(current);
                    
                    let mut current_entity = current;
                    while let Some(&parent_entity) = parent.get(&current_entity) {
                        path.push(parent_entity);
                        current_entity = parent_entity;
                    }
                    
                    path.reverse();
                    return Some(path);
                }
                
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        
        None // No path found
    }
}