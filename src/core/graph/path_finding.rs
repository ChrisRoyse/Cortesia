//! Path finding algorithms for knowledge graph

use super::graph_core::KnowledgeGraph;
use crate::core::types::EntityKey;
// use crate::error::{GraphError, Result}; // Unused
use std::collections::{HashMap, VecDeque, HashSet};

impl KnowledgeGraph {
    /// Find path between two entities using BFS
    pub fn find_path(&self, source: EntityKey, target: EntityKey) -> Option<Vec<EntityKey>> {
        if source == target {
            return Some(vec![source]);
        }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();
        
        queue.push_back(source);
        visited.insert(source);
        
        while let Some(current) = queue.pop_front() {
            if current == target {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = target;
                
                while let Some(&prev) = parent.get(&node) {
                    path.push(node);
                    node = prev;
                }
                path.push(source);
                path.reverse();
                
                return Some(path);
            }
            
            // Check neighbors from main graph
            let graph = self.graph.read();
            for neighbor in graph.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
            
            // Also check edge buffer for recently added relationships
            let edge_buffer = self.edge_buffer.read();
            for relationship in edge_buffer.iter() {
                let neighbor = if relationship.from == current {
                    relationship.to
                } else if relationship.to == current {
                    relationship.from
                } else {
                    continue;
                };
                
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        
        None
    }

    /// Find shortest path between two entities using BFS
    pub fn find_shortest_path(&self, source: EntityKey, target: EntityKey) -> Option<Vec<EntityKey>> {
        self.find_path(source, target)
    }

    /// Find all paths between two entities within a maximum depth
    pub fn find_all_paths(&self, source: EntityKey, target: EntityKey, max_depth: usize) -> Vec<Vec<EntityKey>> {
        let mut paths = Vec::new();
        let mut current_path = vec![source];
        let mut visited = HashSet::new();
        
        visited.insert(source);
        self.find_all_paths_recursive(source, target, max_depth, &mut current_path, &mut visited, &mut paths);
        
        paths
    }

    /// Recursive helper for finding all paths
    fn find_all_paths_recursive(
        &self,
        current: EntityKey,
        target: EntityKey,
        max_depth: usize,
        current_path: &mut Vec<EntityKey>,
        visited: &mut HashSet<EntityKey>,
        paths: &mut Vec<Vec<EntityKey>>,
    ) {
        if current == target {
            paths.push(current_path.clone());
            return;
        }
        
        if current_path.len() >= max_depth {
            return;
        }
        
        // Get neighbors from main graph
        let neighbors = self.get_neighbors(current);
        
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                current_path.push(neighbor);
                
                self.find_all_paths_recursive(neighbor, target, max_depth, current_path, visited, paths);
                
                current_path.pop();
                visited.remove(&neighbor);
            }
        }
    }

    /// Find path with maximum weight (best path)
    pub fn find_best_path(&self, source: EntityKey, target: EntityKey, max_depth: usize) -> Option<(Vec<EntityKey>, f32)> {
        let mut best_path = None;
        let mut best_weight = f32::NEG_INFINITY;
        let mut current_path = vec![source];
        let mut visited = HashSet::new();
        
        visited.insert(source);
        self.find_best_path_recursive(
            source,
            target,
            max_depth,
            0.0,
            &mut current_path,
            &mut visited,
            &mut best_path,
            &mut best_weight,
        );
        
        best_path
    }

    /// Recursive helper for finding best path
    fn find_best_path_recursive(
        &self,
        current: EntityKey,
        target: EntityKey,
        max_depth: usize,
        current_weight: f32,
        current_path: &mut Vec<EntityKey>,
        visited: &mut HashSet<EntityKey>,
        best_path: &mut Option<Vec<EntityKey>>,
        best_weight: &mut f32,
    ) {
        if current == target {
            if current_weight > *best_weight {
                *best_weight = current_weight;
                *best_path = Some(current_path.clone());
            }
            return;
        }
        
        if current_path.len() >= max_depth {
            return;
        }
        
        // Get neighbors and their weights
        let outgoing_relationships = self.get_outgoing_relationships(current);
        
        for relationship in outgoing_relationships {
            let neighbor = relationship.target;
            let edge_weight = relationship.weight;
            
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                current_path.push(neighbor);
                
                self.find_best_path_recursive(
                    neighbor,
                    target,
                    max_depth,
                    current_weight + edge_weight,
                    current_path,
                    visited,
                    best_path,
                    best_weight,
                );
                
                current_path.pop();
                visited.remove(&neighbor);
            }
        }
    }

    /// Find path with minimum weight (weakest path)
    pub fn find_weakest_path(&self, source: EntityKey, target: EntityKey, max_depth: usize) -> Option<(Vec<EntityKey>, f32)> {
        let mut best_path = None;
        let mut best_weight = f32::INFINITY;
        let mut current_path = vec![source];
        let mut visited = HashSet::new();
        
        visited.insert(source);
        self.find_weakest_path_recursive(
            source,
            target,
            max_depth,
            0.0,
            &mut current_path,
            &mut visited,
            &mut best_path,
            &mut best_weight,
        );
        
        best_path
    }

    /// Recursive helper for finding weakest path
    fn find_weakest_path_recursive(
        &self,
        current: EntityKey,
        target: EntityKey,
        max_depth: usize,
        current_weight: f32,
        current_path: &mut Vec<EntityKey>,
        visited: &mut HashSet<EntityKey>,
        best_path: &mut Option<Vec<EntityKey>>,
        best_weight: &mut f32,
    ) {
        if current == target {
            if current_weight < *best_weight {
                *best_weight = current_weight;
                *best_path = Some(current_path.clone());
            }
            return;
        }
        
        if current_path.len() >= max_depth {
            return;
        }
        
        // Get neighbors and their weights
        let outgoing_relationships = self.get_outgoing_relationships(current);
        
        for relationship in outgoing_relationships {
            let neighbor = relationship.target;
            let edge_weight = relationship.weight;
            
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                current_path.push(neighbor);
                
                self.find_weakest_path_recursive(
                    neighbor,
                    target,
                    max_depth,
                    current_weight + edge_weight,
                    current_path,
                    visited,
                    best_path,
                    best_weight,
                );
                
                current_path.pop();
                visited.remove(&neighbor);
            }
        }
    }

    /// Check if there's a path between two entities
    pub fn has_path(&self, source: EntityKey, target: EntityKey) -> bool {
        self.find_path(source, target).is_some()
    }

    /// Get distance between two entities (path length)
    pub fn get_distance(&self, source: EntityKey, target: EntityKey) -> Option<usize> {
        self.find_path(source, target).map(|path| path.len() - 1)
    }

    /// Find all entities within a certain distance from source
    pub fn find_entities_within_distance(&self, source: EntityKey, max_distance: usize) -> Vec<EntityKey> {
        let mut entities = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back((source, 0));
        visited.insert(source);
        entities.push(source);
        
        while let Some((current, distance)) = queue.pop_front() {
            if distance >= max_distance {
                continue;
            }
            
            for neighbor in self.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    entities.push(neighbor);
                    queue.push_back((neighbor, distance + 1));
                }
            }
        }
        
        entities
    }

    /// Find k-nearest neighbors in graph structure (by hops, not embedding similarity)
    pub fn find_k_nearest_graph_neighbors(&self, source: EntityKey, k: usize) -> Vec<EntityKey> {
        let mut neighbors = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(source);
        visited.insert(source);
        
        while let Some(current) = queue.pop_front() {
            if neighbors.len() >= k {
                break;
            }
            
            for neighbor in self.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    neighbors.push(neighbor);
                    queue.push_back(neighbor);
                    
                    if neighbors.len() >= k {
                        break;
                    }
                }
            }
        }
        
        neighbors
    }

    /// Get path statistics
    pub fn get_path_stats(&self, source: EntityKey, target: EntityKey) -> PathStats {
        let shortest_path = self.find_shortest_path(source, target);
        let all_paths = self.find_all_paths(source, target, 5); // Limit to depth 5
        
        let shortest_distance = shortest_path.as_ref().map(|p| p.len() - 1);
        let path_count = all_paths.len();
        
        let average_path_length = if all_paths.is_empty() {
            None
        } else {
            let total_length: usize = all_paths.iter().map(|p| p.len() - 1).sum();
            Some(total_length as f64 / all_paths.len() as f64)
        };
        
        PathStats {
            shortest_distance,
            path_count,
            average_path_length,
            has_path: shortest_path.is_some(),
        }
    }

    /// Compute graph diameter (longest shortest path)
    pub fn compute_diameter(&self) -> Option<usize> {
        let entity_keys = self.get_all_entity_keys();
        let mut max_distance = 0;
        
        for &source in &entity_keys {
            for &target in &entity_keys {
                if source != target {
                    if let Some(distance) = self.get_distance(source, target) {
                        max_distance = max_distance.max(distance);
                    } else {
                        // Graph is not connected
                        return None;
                    }
                }
            }
        }
        
        Some(max_distance)
    }

    /// Compute graph radius (minimum eccentricity)
    pub fn compute_radius(&self) -> Option<usize> {
        let entity_keys = self.get_all_entity_keys();
        let mut min_eccentricity = usize::MAX;
        
        for &source in &entity_keys {
            let mut max_distance_from_source = 0;
            
            for &target in &entity_keys {
                if source != target {
                    if let Some(distance) = self.get_distance(source, target) {
                        max_distance_from_source = max_distance_from_source.max(distance);
                    } else {
                        // Graph is not connected
                        return None;
                    }
                }
            }
            
            min_eccentricity = min_eccentricity.min(max_distance_from_source);
        }
        
        if min_eccentricity == usize::MAX {
            None
        } else {
            Some(min_eccentricity)
        }
    }
}

/// Path statistics
#[derive(Debug, Clone)]
pub struct PathStats {
    pub shortest_distance: Option<usize>,
    pub path_count: usize,
    pub average_path_length: Option<f64>,
    pub has_path: bool,
}

impl PathStats {
    /// Check if entities are directly connected
    pub fn is_direct_connection(&self) -> bool {
        self.shortest_distance == Some(1)
    }
    
    /// Check if entities are closely connected (distance <= 2)
    pub fn is_close_connection(&self) -> bool {
        self.shortest_distance.map_or(false, |d| d <= 2)
    }
    
    /// Check if there are multiple paths
    pub fn has_multiple_paths(&self) -> bool {
        self.path_count > 1
    }
}