use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};

/// Simplified spatial index for fast similarity search
/// Uses a basic k-d tree structure for O(log n) nearest neighbor queries
pub struct SpatialIndex {
    tree: Option<KDNode>,
    dimension: usize,
}

#[derive(Clone)]
struct KDNode {
    entity_id: u32,
    entity_key: EntityKey,
    embedding: Vec<f32>,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
    split_dim: usize,
}

impl SpatialIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            tree: None,
            dimension,
        }
    }

    /// Insert an entity into the spatial index
    pub fn insert(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let new_node = KDNode {
            entity_id,
            entity_key,
            embedding,
            left: None,
            right: None,
            split_dim: 0,
        };

        if self.tree.is_none() {
            self.tree = Some(new_node);
        } else {
            // For simplicity, rebuild the tree with bulk insert for now
            // In production, we'd implement proper incremental insertion
            let mut all_nodes = vec![new_node];
            self.collect_all_nodes(&self.tree, &mut all_nodes);
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = all_nodes
                .into_iter()
                .map(|node| (node.entity_id, node.entity_key, node.embedding))
                .collect();
            self.bulk_build(entities)?;
        }

        Ok(())
    }

    fn collect_all_nodes(&self, node: &Option<KDNode>, nodes: &mut Vec<KDNode>) {
        if let Some(current) = node {
            nodes.push(KDNode {
                entity_id: current.entity_id,
                entity_key: current.entity_key,
                embedding: current.embedding.clone(),
                left: None,
                right: None,
                split_dim: current.split_dim,
            });
            
            if let Some(ref left) = current.left {
                self.collect_all_nodes(&Some(left.as_ref().clone()), nodes);
            }
            if let Some(ref right) = current.right {
                self.collect_all_nodes(&Some(right.as_ref().clone()), nodes);
            }
        }
    }

    /// Find k nearest neighbors using the spatial index
    pub fn k_nearest_neighbors(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if query.len() != self.dimension || self.tree.is_none() {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut best_distances = Vec::new();

        if let Some(ref root) = self.tree {
            let boxed_root = Some(Box::new(root.clone()));
            self.search_recursive(&boxed_root, query, k, &mut candidates, &mut best_distances, 0);
        }

        // Sort by distance and return top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.into_iter().take(k).collect()
    }

    fn search_recursive(
        &self,
        node: &Option<Box<KDNode>>,
        query: &[f32],
        k: usize,
        candidates: &mut Vec<(u32, f32)>,
        best_distances: &mut Vec<f32>,
        depth: usize,
    ) {
        if let Some(current) = node {
            let split_dim = depth % self.dimension;
            
            // Calculate distance to current node
            let distance = euclidean_distance(query, &current.embedding);
            
            // Add to candidates if we have room or if it's better than our worst
            if candidates.len() < k {
                candidates.push((current.entity_id, distance));
                best_distances.push(distance);
                
                // Keep sorted for efficient worst-case lookups
                if candidates.len() == k {
                    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    best_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
            } else if distance < best_distances[k - 1] {
                // Replace worst candidate
                candidates[k - 1] = (current.entity_id, distance);
                best_distances[k - 1] = distance;
                
                // Re-sort to maintain order
                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                best_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }

            // Decide which subtree to explore first
            let diff = query[split_dim] - current.embedding[split_dim];
            let (first, second) = if diff < 0.0 {
                (&current.left, &current.right)
            } else {
                (&current.right, &current.left)
            };

            // Search the first subtree
            self.search_recursive(first, query, k, candidates, best_distances, depth + 1);

            // Check if we need to search the second subtree
            let worst_distance = if candidates.len() < k {
                f32::INFINITY
            } else {
                best_distances[k - 1]
            };

            if diff.abs() < worst_distance {
                self.search_recursive(second, query, k, candidates, best_distances, depth + 1);
            }
        }
    }

    /// Bulk build the index from a set of entities (more efficient than repeated inserts)
    pub fn bulk_build(&mut self, entities: Vec<(u32, EntityKey, Vec<f32>)>) -> Result<()> {
        if entities.is_empty() {
            return Ok(());
        }

        let nodes: Vec<KDNode> = entities
            .into_iter()
            .map(|(id, key, embedding)| KDNode {
                entity_id: id,
                entity_key: key,
                embedding,
                left: None,
                right: None,
                split_dim: 0,
            })
            .collect();

        self.tree = Some(self.build_recursive(nodes, 0));
        Ok(())
    }

    fn build_recursive(&self, mut nodes: Vec<KDNode>, depth: usize) -> KDNode {
        let split_dim = depth % self.dimension;
        
        // Sort by the splitting dimension
        nodes.sort_by(|a, b| a.embedding[split_dim].partial_cmp(&b.embedding[split_dim]).unwrap());
        
        let median = nodes.len() / 2;
        let mut root = nodes.swap_remove(median);
        root.split_dim = split_dim;

        // Recursively build left and right subtrees
        if median > 0 {
            let left_nodes = nodes.drain(..median).collect();
            root.left = Some(Box::new(self.build_recursive(left_nodes, depth + 1)));
        }

        if !nodes.is_empty() {
            root.right = Some(Box::new(self.build_recursive(nodes, depth + 1)));
        }

        root
    }

    /// Check if the index contains any entities
    pub fn is_empty(&self) -> bool {
        self.tree.is_none()
    }

    /// Get approximate count of entities in the index
    pub fn approximate_count(&self) -> usize {
        self.count_recursive(&self.tree)
    }

    fn count_recursive(&self, node: &Option<KDNode>) -> usize {
        match node {
            Some(current) => {
                let left_count = match &current.left {
                    Some(left_node) => self.count_recursive(&Some(left_node.as_ref().clone())),
                    None => 0,
                };
                let right_count = match &current.right {
                    Some(right_node) => self.count_recursive(&Some(right_node.as_ref().clone())),
                    None => 0,
                };
                1 + left_count + right_count
            }
            None => 0,
        }
    }
    
    /// Get the capacity (approximate based on node count)
    pub fn capacity(&self) -> usize {
        self.approximate_count() * 2 // Rough estimate
    }
    
    /// Add edge (not applicable - SpatialIndex stores embeddings, not edges)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation(
            "SpatialIndex stores entity embeddings, not edges. Use CSRGraph for edges.".to_string()
        ))
    }
    
    /// Update entity embedding
    pub fn update_entity(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        // For k-d tree, updating requires removing and re-inserting
        // This is a simplified implementation that rebuilds the tree
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        // Collect all nodes except the one to update
        let mut all_nodes = Vec::new();
        self.collect_all_nodes(&self.tree, &mut all_nodes);
        
        // Filter out the old entry and add the new one
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = all_nodes
            .into_iter()
            .filter(|node| node.entity_id != entity_id)
            .map(|node| (node.entity_id, node.entity_key, node.embedding))
            .chain(std::iter::once((entity_id, entity_key, embedding)))
            .collect();
            
        self.bulk_build(entities)
    }
    
    /// Remove an entity from the index
    pub fn remove(&mut self, entity_id: u32) -> Result<()> {
        // Collect all nodes except the one to remove
        let mut all_nodes = Vec::new();
        self.collect_all_nodes(&self.tree, &mut all_nodes);
        
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = all_nodes
            .into_iter()
            .filter(|node| node.entity_id != entity_id)
            .map(|node| (node.entity_id, node.entity_key, node.embedding))
            .collect();
            
        if entities.is_empty() {
            self.tree = None;
            Ok(())
        } else {
            self.bulk_build(entities)
        }
    }
    
    /// Check if index contains an entity
    pub fn contains_entity(&self, entity_id: u32) -> bool {
        self.contains_entity_recursive(&self.tree, entity_id)
    }
    
    fn contains_entity_recursive(&self, node: &Option<KDNode>, entity_id: u32) -> bool {
        match node {
            Some(current) => {
                if current.entity_id == entity_id {
                    return true;
                }
                let in_left = match &current.left {
                    Some(left_node) => self.contains_entity_recursive(&Some(left_node.as_ref().clone()), entity_id),
                    None => false,
                };
                let in_right = match &current.right {
                    Some(right_node) => self.contains_entity_recursive(&Some(right_node.as_ref().clone()), entity_id),
                    None => false,
                };
                in_left || in_right
            }
            None => false,
        }
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        // Approximate size for serialization
        std::mem::size_of::<usize>() + // dimension
        self.encoded_size_recursive(&self.tree)
    }
    
    fn encoded_size_recursive(&self, node: &Option<KDNode>) -> usize {
        match node {
            Some(current) => {
                let node_size = std::mem::size_of::<u32>() + // entity_id
                    std::mem::size_of::<EntityKey>() +
                    current.embedding.len() * std::mem::size_of::<f32>() +
                    std::mem::size_of::<usize>(); // split_dim
                    
                let left_size = match &current.left {
                    Some(left_node) => self.encoded_size_recursive(&Some(left_node.as_ref().clone())),
                    None => 0,
                };
                let right_size = match &current.right {
                    Some(right_node) => self.encoded_size_recursive(&Some(right_node.as_ref().clone())),
                    None => 0,
                };
                
                node_size + left_size + right_size
            }
            None => 0,
        }
    }
}

#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_spatial_index_creation() {
        let index = SpatialIndex::new(128);
        assert!(index.is_empty());
        assert_eq!(index.approximate_count(), 0);
    }

    #[test]
    fn test_spatial_index_insertion() {
        let mut index = SpatialIndex::new(3);
        let key = EntityKey::default();
        let embedding = vec![1.0, 2.0, 3.0];
        
        index.insert(1, key, embedding).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.approximate_count(), 1);
    }

    #[test]
    fn test_nearest_neighbor_search() {
        let mut index = SpatialIndex::new(3);
        
        // Insert test points
        let points = vec![
            (1, vec![1.0, 1.0, 1.0]),
            (2, vec![2.0, 2.0, 2.0]),
            (3, vec![3.0, 3.0, 3.0]),
            (4, vec![0.0, 0.0, 0.0]),
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        // Query near the origin
        let query = vec![0.1, 0.1, 0.1];
        let results = index.k_nearest_neighbors(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 4); // Should be closest to origin
    }

    #[test]
    fn test_bulk_build() {
        let mut index = SpatialIndex::new(2);
        
        let entities = vec![
            (1, EntityKey::default(), vec![1.0, 1.0]),
            (2, EntityKey::default(), vec![2.0, 2.0]),
            (3, EntityKey::default(), vec![3.0, 3.0]),
            (4, EntityKey::default(), vec![0.0, 0.0]),
        ];
        
        index.bulk_build(entities).unwrap();
        assert_eq!(index.approximate_count(), 4);
        
        let query = vec![1.5, 1.5];
        let results = index.k_nearest_neighbors(&query, 2);
        assert_eq!(results.len(), 2);
    }
}