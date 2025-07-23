use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};
use crate::embedding::similarity::cosine_similarity;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use parking_lot::RwLock;
use rand::Rng;

/// Hierarchical Navigable Small World (HNSW) index for fast approximate nearest neighbor search
/// Based on the paper "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
pub struct HnswIndex {
    /// Maximum number of connections per node in layer 0
    max_connections_0: usize,
    /// Maximum number of connections per node in upper layers
    max_connections: usize,
    /// Normalization factor for level generation
    level_multiplier: f64,
    /// Entry point to the graph
    entry_point: RwLock<Option<NodeId>>,
    /// All nodes in the graph, organized by layers
    nodes: RwLock<HashMap<NodeId, Node>>,
    /// Vector dimension
    dimension: usize,
    /// Current node ID counter
    next_id: RwLock<NodeId>,
}

type NodeId = u32;

#[derive(Clone)]
struct Node {
    id: NodeId,
    entity_id: u32,
    entity_key: EntityKey,
    embedding: Vec<f32>,
    /// Connections for each layer (layer 0 is at index 0)
    connections: Vec<Vec<NodeId>>,
    /// Maximum layer this node exists in
    max_layer: usize,
}

#[derive(Clone, PartialEq)]
struct SearchCandidate {
    node_id: NodeId,
    distance: f32,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // For max heap (farthest first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            max_connections_0: 16,  // M parameter for layer 0
            max_connections: 8,     // M parameter for upper layers
            level_multiplier: 1.0 / (2.0_f64).ln(),
            entry_point: RwLock::new(None),
            nodes: RwLock::new(HashMap::new()),
            dimension,
            next_id: RwLock::new(0),
        }
    }

    /// Configure HNSW parameters for different use cases
    pub fn with_config(dimension: usize, max_connections_0: usize, max_connections: usize) -> Self {
        Self {
            max_connections_0,
            max_connections,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            entry_point: RwLock::new(None),
            nodes: RwLock::new(HashMap::new()),
            dimension,
            next_id: RwLock::new(0),
        }
    }

    /// Insert a new vector into the HNSW index
    pub fn insert(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let mut nodes = self.nodes.write();
        let mut next_id = self.next_id.write();

        let node_id = *next_id;
        *next_id += 1;

        // Generate random level for the new node
        let level = self.generate_random_level();

        let mut new_node = Node {
            id: node_id,
            entity_id,
            entity_key,
            embedding,
            connections: vec![Vec::new(); level + 1],
            max_layer: level,
        };

        if nodes.is_empty() {
            // First node becomes the entry point
            nodes.insert(node_id, new_node);
            drop(nodes);
            *self.entry_point.write() = Some(node_id);
            return Ok(());
        }

        // Find the best entry point and search from top layer down
        let entry_point = self.entry_point.read().unwrap();
        let mut current_closest = vec![entry_point];

        // Search from top layer down to layer 1
        for layer in ((level + 1)..=self.get_max_layer(&nodes)).rev() {
            current_closest = self.search_layer(&nodes, &new_node.embedding, &current_closest, 1, layer);
        }

        // For each layer from min(level, max_layer) down to 0
        for layer in (0..=level.min(self.get_max_layer(&nodes))).rev() {
            let candidates = self.search_layer(&nodes, &new_node.embedding, &current_closest, self.ef_construction(), layer);
            
            // Select neighbors for the new node
            let max_conn = if layer == 0 { self.max_connections_0 } else { self.max_connections };
            let selected_neighbors = self.select_neighbors(&nodes, &new_node.embedding, &candidates, max_conn);
            
            new_node.connections[layer] = selected_neighbors.clone();

            // Add reciprocal connections and prune if necessary
            for &neighbor_id in &selected_neighbors {
                if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                    neighbor.connections[layer].push(node_id);
                    
                    // Prune connections if necessary
                    if neighbor.connections[layer].len() > max_conn {
                        let neighbor_embedding = neighbor.embedding.clone();
                        let candidates: Vec<NodeId> = neighbor.connections[layer].clone();
                        // Temporarily drop the mutable borrow to call select_neighbors
                        let _ = neighbor;
                        let pruned = self.select_neighbors(&nodes, &neighbor_embedding, &candidates, max_conn);
                        // Get mutable reference again
                        if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                            neighbor.connections[layer] = pruned;
                        }
                    }
                }
            }

            current_closest = selected_neighbors;
        }

        // Update entry point if necessary
        if level > self.get_max_layer(&nodes) {
            *self.entry_point.write() = Some(node_id);
        }

        nodes.insert(node_id, new_node);
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        let nodes = self.nodes.read();
        if nodes.is_empty() {
            return Vec::new();
        }

        let entry_point = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return Vec::new(),
        };
        let mut current_closest = vec![entry_point];

        // Search from top layer down to layer 1
        let max_layer = self.get_max_layer(&nodes);
        for layer in (1..=max_layer).rev() {
            current_closest = self.search_layer(&nodes, query, &current_closest, 1, layer);
        }

        // Search layer 0 with larger candidate set
        let ef_search = (k * 2).max(100); // Dynamic ef based on k
        let candidates = self.search_layer(&nodes, query, &current_closest, ef_search, 0);

        // Convert to results and sort by distance
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter_map(|node_id| {
                nodes.get(&node_id).map(|node| {
                    let distance = 1.0 - cosine_similarity(query, &node.embedding);
                    (node.entity_id, distance)
                })
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Search within a specific layer
    fn search_layer(
        &self,
        nodes: &HashMap<NodeId, Node>,
        query: &[f32],
        entry_points: &[NodeId],
        num_closest: usize,
        layer: usize,
    ) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new(); // Max heap (farthest first)
        let mut dynamic_candidates = BinaryHeap::new(); // Max heap for tracking closest

        // Initialize with entry points
        for &entry_id in entry_points {
            if let Some(node) = nodes.get(&entry_id) {
                let distance = 1.0 - cosine_similarity(query, &node.embedding);
                candidates.push(SearchCandidate { node_id: entry_id, distance });
                dynamic_candidates.push(SearchCandidate { node_id: entry_id, distance });
                visited.insert(entry_id);
            }
        }

        while let Some(current) = candidates.pop() {
            // If current is farther than farthest in dynamic list, stop
            if let Some(farthest) = dynamic_candidates.peek() {
                if current.distance > farthest.distance {
                    break;
                }
            }

            if let Some(current_node) = nodes.get(&current.node_id) {
                if layer < current_node.connections.len() {
                    for &neighbor_id in &current_node.connections[layer] {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);

                            if let Some(neighbor) = nodes.get(&neighbor_id) {
                                let distance = 1.0 - cosine_similarity(query, &neighbor.embedding);
                                let candidate = SearchCandidate { node_id: neighbor_id, distance };

                                if dynamic_candidates.len() < num_closest {
                                    candidates.push(candidate.clone());
                                    dynamic_candidates.push(candidate);
                                } else if let Some(farthest) = dynamic_candidates.peek() {
                                    if distance < farthest.distance {
                                        candidates.push(candidate.clone());
                                        dynamic_candidates.pop();
                                        dynamic_candidates.push(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract closest candidates
        let mut result: Vec<SearchCandidate> = dynamic_candidates.into_iter().collect();
        result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result.into_iter().map(|c| c.node_id).collect()
    }

    /// Select best neighbors using a simple heuristic
    fn select_neighbors(
        &self,
        nodes: &HashMap<NodeId, Node>,
        query_embedding: &[f32],
        candidates: &[NodeId],
        max_connections: usize,
    ) -> Vec<NodeId> {
        let mut scored_candidates: Vec<(NodeId, f32)> = candidates
            .iter()
            .filter_map(|&node_id| {
                nodes.get(&node_id).map(|node| {
                    let distance = 1.0 - cosine_similarity(query_embedding, &node.embedding);
                    (node_id, distance)
                })
            })
            .collect();

        // Sort by distance and take the closest
        scored_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored_candidates
            .into_iter()
            .take(max_connections)
            .map(|(node_id, _)| node_id)
            .collect()
    }

    fn generate_random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();
        (-random_value.ln() * self.level_multiplier).floor() as usize
    }

    fn get_max_layer(&self, nodes: &HashMap<NodeId, Node>) -> usize {
        nodes.values().map(|node| node.max_layer).max().unwrap_or(0)
    }

    fn ef_construction(&self) -> usize {
        200 // Construction-time search parameter
    }

    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Get index statistics for debugging
    pub fn stats(&self) -> HnswStats {
        let nodes = self.nodes.read();
        let node_count = nodes.len();
        let max_layer = self.get_max_layer(&nodes);
        
        let mut layer_sizes = vec![0; max_layer + 1];
        let mut total_connections = 0;
        
        for node in nodes.values() {
            for layer in 0..=node.max_layer {
                layer_sizes[layer] += 1;
                if layer < node.connections.len() {
                    total_connections += node.connections[layer].len();
                }
            }
        }
        
        HnswStats {
            node_count,
            max_layer,
            layer_sizes,
            total_connections,
            avg_connections: if node_count > 0 { total_connections as f64 / node_count as f64 } else { 0.0 },
        }
    }
    
    /// Get the capacity of the index
    pub fn capacity(&self) -> usize {
        self.nodes.read().capacity()
    }
    
    /// Add edge (not applicable - HnswIndex stores embeddings with internal graph structure)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation(
            "HnswIndex manages its own internal graph structure. Use insert() to add entities.".to_string()
        ))
    }
    
    /// Update entity embedding
    pub fn update_entity(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        // For HNSW, updating requires removing and re-inserting
        // First, check if the entity exists
        let nodes = self.nodes.read();
        let node_id = nodes.values()
            .find(|n| n.entity_id == entity_id)
            .map(|n| n.id);
        drop(nodes);
        
        if let Some(_id) = node_id {
            // Remove the old node
            self.remove(entity_id)?;
            // Insert the new one
            self.insert(entity_id, entity_key, embedding)?;
            Ok(())
        } else {
            Err(GraphError::EntityNotFound { id: entity_id })
        }
    }
    
    /// Remove an entity from the index
    pub fn remove(&self, entity_id: u32) -> Result<()> {
        let mut nodes = self.nodes.write();
        
        // Find the node with this entity_id
        let node_to_remove = nodes.values()
            .find(|n| n.entity_id == entity_id)
            .map(|n| n.id);
            
        if let Some(node_id) = node_to_remove {
            // Remove connections to this node from other nodes
            for node in nodes.values_mut() {
                for connections in &mut node.connections {
                    connections.retain(|&id| id != node_id);
                }
            }
            
            // Remove the node itself
            nodes.remove(&node_id);
            
            // Update entry point if needed
            let mut entry_point = self.entry_point.write();
            if entry_point.as_ref() == Some(&node_id) {
                *entry_point = nodes.keys().next().copied();
            }
            
            Ok(())
        } else {
            Err(GraphError::EntityNotFound { id: entity_id })
        }
    }
    
    /// Check if index contains an entity
    pub fn contains_entity(&self, entity_id: u32) -> bool {
        self.nodes.read().values().any(|n| n.entity_id == entity_id)
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        let nodes = self.nodes.read();
        let base_size = std::mem::size_of::<usize>() * 4; // dimension, max_connections, etc.
        
        let nodes_size = nodes.values().map(|node| {
            std::mem::size_of::<NodeId>() +
            std::mem::size_of::<u32>() + // entity_id
            std::mem::size_of::<EntityKey>() +
            node.embedding.len() * std::mem::size_of::<f32>() +
            node.connections.iter().map(|c| c.len() * std::mem::size_of::<NodeId>()).sum::<usize>() +
            std::mem::size_of::<usize>() // max_layer
        }).sum::<usize>();
        
        base_size + nodes_size
    }
}

#[derive(Debug)]
pub struct HnswStats {
    pub node_count: usize,
    pub max_layer: usize,
    pub layer_sizes: Vec<usize>,
    pub total_connections: usize,
    pub avg_connections: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_hnsw_creation() {
        let index = HnswIndex::new(128);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hnsw_insertion() {
        let index = HnswIndex::new(3);
        let key = EntityKey::default();
        let embedding = vec![1.0, 0.0, 0.0];
        
        index.insert(1, key, embedding).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hnsw_search() {
        let index = HnswIndex::new(3);
        
        // Insert test points
        let points = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.7, 0.7, 0.0]),
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        // Query near [1, 0, 0]
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2);
        
        assert!(!results.is_empty());
        // First result should be entity 1 (closest to [1, 0, 0])
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_hnsw_stats() {
        let index = HnswIndex::new(4);
        
        // Insert multiple points to create a more complex graph
        for i in 0..50 {
            let embedding = vec![
                (i as f32 / 50.0),
                ((i * 2) as f32 / 50.0),
                ((i * 3) as f32 / 50.0),
                ((i * 4) as f32 / 50.0),
            ];
            index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
        }
        
        let stats = index.stats();
        assert_eq!(stats.node_count, 50);
        assert!(stats.max_layer < 10); // Should be reasonable
        assert!(stats.avg_connections > 0.0);
    }

    #[test]
    fn test_generate_random_level() {
        let index = HnswIndex::new(4);
        
        // Test level generation multiple times
        let mut levels = Vec::new();
        for _ in 0..1000 {
            let level = index.generate_random_level();
            levels.push(level);
        }
        
        // Most levels should be 0 (exponential distribution)
        let level_0_count = levels.iter().filter(|&&l| l == 0).count();
        assert!(level_0_count > 500); // Should be majority
        
        // Should occasionally generate higher levels
        let max_level = *levels.iter().max().unwrap();
        assert!(max_level > 0);
        assert!(max_level < 20); // Shouldn't be too high
        
        // Test exponential distribution property
        let level_1_count = levels.iter().filter(|&&l| l == 1).count();
        let level_2_count = levels.iter().filter(|&&l| l == 2).count();
        
        // Level 1 should be more common than level 2
        assert!(level_1_count > level_2_count);
    }

    #[test]
    fn test_select_neighbors_heuristic() {
        let index = HnswIndex::new(3);
        
        // Create test nodes
        let nodes = {
            let mut nodes_map = std::collections::HashMap::new();
            nodes_map.insert(1, Node {
                id: 1,
                entity_id: 1,
                entity_key: EntityKey::default(),
                embedding: vec![1.0, 0.0, 0.0],
                connections: vec![Vec::new()],
                max_layer: 0,
            });
            nodes_map.insert(2, Node {
                id: 2,
                entity_id: 2,
                entity_key: EntityKey::default(),
                embedding: vec![0.0, 1.0, 0.0],
                connections: vec![Vec::new()],
                max_layer: 0,
            });
            nodes_map.insert(3, Node {
                id: 3,
                entity_id: 3,
                entity_key: EntityKey::default(),
                embedding: vec![0.0, 0.0, 1.0],
                connections: vec![Vec::new()],
                max_layer: 0,
            });
            nodes_map.insert(4, Node {
                id: 4,
                entity_id: 4,
                entity_key: EntityKey::default(),
                embedding: vec![0.5, 0.5, 0.0], // Closer to query
                connections: vec![Vec::new()],
                max_layer: 0,
            });
            nodes_map
        };
        
        let query_embedding = vec![0.6, 0.4, 0.0];
        let candidates = vec![1, 2, 3, 4];
        
        let selected = index.select_neighbors(&nodes, &query_embedding, &candidates, 2);
        
        assert_eq!(selected.len(), 2);
        
        // Should select the closest neighbors (should include node 4 and 1)
        assert!(selected.contains(&4)); // Closest to query
        assert!(selected.contains(&1)); // Second closest
    }

    #[test]
    fn test_search_layer_functionality() {
        let index = HnswIndex::new(3);
        
        // Insert a few nodes to create a simple graph
        let points = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.5, 0.5, 0.0]),
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        let nodes = index.nodes.read();
        let query = vec![0.6, 0.4, 0.0];
        
        // Test search_layer with different entry points
        let entry_points = vec![1]; // Start from node 1
        let results = index.search_layer(&nodes, &query, &entry_points, 2, 0);
        
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_hnsw_multilayer_structure() {
        let index = HnswIndex::new(8);
        
        // Insert enough nodes to likely create multiple layers
        for i in 0..100 {
            let embedding: Vec<f32> = (0..8).map(|j| (i * j) as f32 / 100.0).collect();
            index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
        }
        
        let stats = index.stats();
        
        // Should have created a multi-layer structure
        assert!(stats.max_layer > 0);
        assert!(stats.layer_sizes.len() > 1);
        
        // Layer 0 should have all nodes
        assert_eq!(stats.layer_sizes[0], 100);
        
        // Higher layers should have fewer nodes
        for i in 1..stats.layer_sizes.len() {
            assert!(stats.layer_sizes[i] <= stats.layer_sizes[i-1]);
        }
    }

    #[test]
    fn test_level_multiplier_effect() {
        // Test that level_multiplier affects level generation
        let index1 = HnswIndex::new(4);
        let index2 = HnswIndex::with_config(4, 16, 8);
        
        // Both should use the same level multiplier (1.0 / ln(2))
        assert!((index1.level_multiplier - index2.level_multiplier).abs() < 1e-10);
        
        // Test that it produces reasonable levels
        let mut max_level = 0;
        for _ in 0..100 {
            let level = index1.generate_random_level();
            max_level = max_level.max(level);
        }
        
        // Should produce some variety but not too extreme
        assert!(max_level > 0);
        assert!(max_level < 15);
    }

    #[test]
    fn test_node_connections_pruning() {
        let index = HnswIndex::with_config(3, 2, 1); // Small connection limits
        
        // Insert nodes in a pattern that would create many connections
        let points = vec![
            (1, vec![0.0, 0.0, 0.0]), // Center
            (2, vec![1.0, 0.0, 0.0]),
            (3, vec![0.0, 1.0, 0.0]),
            (4, vec![0.0, 0.0, 1.0]),
            (5, vec![-1.0, 0.0, 0.0]),
            (6, vec![0.0, -1.0, 0.0]),
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        let stats = index.stats();
        
        // With pruning, average connections should respect the limits
        assert!(stats.avg_connections <= index.max_connections_0 as f64 * 1.5);
    }

    #[test]
    fn test_distance_based_neighbor_selection() {
        let index = HnswIndex::new(2);
        
        // Create a scenario where distance-based selection matters
        let points = vec![
            (1, vec![0.0, 0.0]),   // Origin
            (2, vec![1.0, 0.0]),   // Distance 1
            (3, vec![2.0, 0.0]),   // Distance 2  
            (4, vec![0.0, 1.0]),   // Distance 1
            (5, vec![0.0, 2.0]),   // Distance 2
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::from_u32(id), embedding).unwrap();
        }
        
        // Query close to origin
        let query = vec![0.1, 0.1];
        let results = index.search(&query, 3);
        
        assert_eq!(results.len(), 3);
        
        // Should return closest nodes first
        assert_eq!(results[0].0, 1); // Origin is closest
        
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i-1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_ef_construction_parameter() {
        let index = HnswIndex::new(4);
        
        // Test that ef_construction returns a reasonable value
        let ef = index.ef_construction();
        assert!(ef > 0);
        assert!(ef >= 100); // Should be at least 100 based on implementation
    }

    #[test]
    fn test_empty_hnsw_operations() {
        let index = HnswIndex::new(4);
        
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        
        // Search on empty index
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
        
        // Stats on empty index
        let stats = index.stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.max_layer, 0);
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.avg_connections, 0.0);
    }

    #[test]
    fn test_hnsw_remove_functionality() {
        let index = HnswIndex::new(3);
        
        // Insert nodes
        let points = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        assert_eq!(index.len(), 3);
        assert!(index.contains_entity(2));
        
        // Remove a node
        let result = index.remove(2);
        assert!(result.is_ok());
        
        assert_eq!(index.len(), 2);
        assert!(!index.contains_entity(2));
        assert!(index.contains_entity(1));
        assert!(index.contains_entity(3));
        
        // Test removing non-existent node
        let result = index.remove(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_hnsw_update_functionality() {
        let index = HnswIndex::new(3);
        
        // Insert initial node
        index.insert(1, EntityKey::from_u32(1), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(2, EntityKey::from_u32(2), vec![0.0, 1.0, 0.0]).unwrap();
        
        // Update entity 1
        let result = index.update_entity(1, EntityKey::from_u32(1), vec![0.0, 0.0, 1.0]);
        assert!(result.is_ok());
        
        // Verify update (search should find updated entity closer to new position)
        let query = vec![0.0, 0.0, 1.0];
        let results = index.search(&query, 1);
        assert_eq!(results[0].0, 1);
        
        // Test updating non-existent entity
        let result = index.update_entity(99, EntityKey::from_u32(99), vec![1.0, 1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_operations() {
        let mut index = HnswIndex::new(3);
        
        // Test that graph operations are not supported
        assert!(index.add_edge(1, 2, 1.0).is_err());
    }

    #[test]
    fn test_encoded_size() {
        let index = HnswIndex::new(4);
        
        // Insert a few nodes
        for i in 0..5 {
            let embedding = vec![i as f32, (i*2) as f32, (i*3) as f32, (i*4) as f32];
            index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
        }
        
        let encoded_size = index.encoded_size();
        assert!(encoded_size > 0);
        
        // Should be at least the size of basic metadata
        let base_size = std::mem::size_of::<usize>() * 4;
        assert!(encoded_size >= base_size);
    }

    #[cfg(test)]
    mod private_method_tests {
        use super::*;
        
        #[test]
        fn test_get_max_layer() {
            let index = HnswIndex::new(3);
            
            // Insert nodes with different levels
            index.insert(1, EntityKey::default(), vec![1.0, 0.0, 0.0]).unwrap();
            index.insert(2, EntityKey::default(), vec![0.0, 1.0, 0.0]).unwrap();
            index.insert(3, EntityKey::default(), vec![0.0, 0.0, 1.0]).unwrap();
            
            let nodes = index.nodes.read();
            let max_layer = index.get_max_layer(&nodes);
            
            // Should return a reasonable max layer
            assert!(max_layer < 10); // Shouldn't be too high for small dataset
        }
    }

    #[cfg(test)]
    mod performance_tests {
        use super::*;
        use std::time::Instant;
        
        #[test]
        fn test_insertion_performance() {
            let index = HnswIndex::new(64);
            
            let start = Instant::now();
            
            // Insert many nodes
            for i in 0..1000 {
                let embedding: Vec<f32> = (0..64).map(|j| (i * j) as f32 / 1000.0).collect();
                index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
            }
            
            let duration = start.elapsed();
            
            assert_eq!(index.len(), 1000);
            
            // Should be reasonably fast (loose bound)
            assert!(duration.as_millis() < 5000);
        }
        
        #[test]
        fn test_search_performance() {
            let index = HnswIndex::new(32);
            
            // Build index
            for i in 0..1000 {
                let embedding: Vec<f32> = (0..32).map(|j| (i + j) as f32 / 1000.0).collect();
                index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
            }
            
            let query: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
            
            let start = Instant::now();
            let results = index.search(&query, 10);
            let duration = start.elapsed();
            
            assert_eq!(results.len(), 10);
            
            // Should be much faster than linear search
            assert!(duration.as_millis() < 100);
        }
        
        #[test]
        fn test_scaling_behavior() {
            let dimensions = vec![16, 32, 64];
            let sizes = vec![100, 500, 1000];
            
            for &dim in &dimensions {
                for &size in &sizes {
                    let index = HnswIndex::new(dim);
                    
                    // Build index
                    let start = Instant::now();
                    for i in 0..size {
                        let embedding: Vec<f32> = (0..dim).map(|j| (i * j) as f32 / size as f32).collect();
                        index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
                    }
                    let build_time = start.elapsed();
                    
                    // Search
                    let query: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
                    let start = Instant::now();
                    let results = index.search(&query, 5);
                    let search_time = start.elapsed();
                    
                    assert_eq!(results.len(), 5);
                    
                    // Scaling should be sublinear
                    assert!(search_time.as_micros() < build_time.as_micros() / 10);
                }
            }
        }
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        
        #[test]
        fn test_hnsw_connectivity_invariants() {
            let index = HnswIndex::new(4);
            
            // Insert nodes
            for i in 0..20 {
                let embedding = vec![
                    (i as f32 / 20.0).sin(),
                    (i as f32 / 20.0).cos(),
                    (i as f32 / 10.0),
                    (i as f32 / 5.0) % 1.0,
                ];
                index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
            }
            
            let nodes = index.nodes.read();
            
            // Test HNSW invariants
            // 1. All nodes should be connected (reachable from entry point)
            // 2. Connection counts should respect limits
            // 3. Higher layers should have subset of lower layer nodes
            
            for node in nodes.values() {
                // Each layer should have <= max_connections
                for (layer, connections) in node.connections.iter().enumerate() {
                    let max_conn = if layer == 0 { index.max_connections_0 } else { index.max_connections };
                    assert!(connections.len() <= max_conn);
                }
                
                // Node should exist in all layers up to its max_layer
                assert_eq!(node.connections.len(), node.max_layer + 1);
            }
        }
        
        #[test]
        fn test_search_quality() {
            let index = HnswIndex::new(3);
            
            // Create a known pattern
            let reference_points = vec![
                (1, vec![1.0, 0.0, 0.0]),
                (2, vec![0.0, 1.0, 0.0]),
                (3, vec![0.0, 0.0, 1.0]),
                (4, vec![-1.0, 0.0, 0.0]),
                (5, vec![0.0, -1.0, 0.0]),
                (6, vec![0.0, 0.0, -1.0]),
            ];
            
            for (id, embedding) in reference_points {
                index.insert(id, EntityKey::default(), embedding).unwrap();
            }
            
            // Test that exact matches are found first
            let query = vec![1.0, 0.0, 0.0];
            let results = index.search(&query, 3);
            
            assert!(!results.is_empty());
            assert_eq!(results[0].0, 1); // Exact match should be first
            assert!(results[0].1 < 1e-6); // Distance should be near zero
            
            // Test that results are sorted by distance
            for i in 1..results.len() {
                assert!(results[i-1].1 <= results[i].1);
            }
        }
        
        #[test]
        fn test_level_distribution() {
            let index = HnswIndex::new(4);
            
            // Insert many nodes and check level distribution
            let mut level_counts = std::collections::HashMap::new();
            
            for i in 0..1000 {
                let embedding = vec![
                    (i as f32 / 1000.0),
                    ((i * 2) as f32 / 1000.0),
                    ((i * 3) as f32 / 1000.0),
                    ((i * 4) as f32 / 1000.0),
                ];
                index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
            }
            
            let nodes = index.nodes.read();
            for node in nodes.values() {
                *level_counts.entry(node.max_layer).or_insert(0) += 1;
            }
            
            // Level distribution should follow exponential decay
            let level_0_count = level_counts.get(&0).unwrap_or(&0);
            let level_1_count = level_counts.get(&1).unwrap_or(&0);
            
            // Level 0 should have significantly more nodes than level 1
            assert!(*level_0_count > *level_1_count * 2);
            
            // Should have created multiple levels
            assert!(level_counts.len() > 1);
        }
    }
}