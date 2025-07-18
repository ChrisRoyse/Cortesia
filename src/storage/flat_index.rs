use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};
use crate::embedding::similarity::cosine_similarity;

/// Flat vector index optimized for SIMD operations and cache efficiency
/// Trades some theoretical complexity for practical performance
pub struct FlatVectorIndex {
    // Packed entity data for cache efficiency
    entity_ids: Vec<u32>,
    entity_keys: Vec<EntityKey>,
    embeddings: Vec<f32>, // Flattened: [e1_dim0, e1_dim1, ..., e2_dim0, e2_dim1, ...]
    dimension: usize,
    count: usize,
}

impl FlatVectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            entity_ids: Vec::new(),
            entity_keys: Vec::new(),
            embeddings: Vec::new(),
            dimension,
            count: 0,
        }
    }

    /// Insert an entity into the flat index
    pub fn insert(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        self.entity_ids.push(entity_id);
        self.entity_keys.push(entity_key);
        self.embeddings.extend_from_slice(&embedding);
        self.count += 1;

        Ok(())
    }

    /// Bulk build from entities (most efficient)
    pub fn bulk_build(&mut self, entities: Vec<(u32, EntityKey, Vec<f32>)>) -> Result<()> {
        let entity_count = entities.len();
        
        // Pre-allocate for efficiency
        self.entity_ids.clear();
        self.entity_keys.clear();
        self.embeddings.clear();
        
        self.entity_ids.reserve(entity_count);
        self.entity_keys.reserve(entity_count);
        self.embeddings.reserve(entity_count * self.dimension);

        for (id, key, embedding) in entities {
            if embedding.len() != self.dimension {
                return Err(GraphError::InvalidEmbeddingDimension {
                    expected: self.dimension,
                    actual: embedding.len(),
                });
            }

            self.entity_ids.push(id);
            self.entity_keys.push(key);
            self.embeddings.extend_from_slice(&embedding);
        }

        self.count = entity_count;
        Ok(())
    }

    /// Fast k-nearest neighbors using SIMD-accelerated operations
    pub fn k_nearest_neighbors(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if query.len() != self.dimension || self.count == 0 {
            return Vec::new();
        }

        // For large datasets with suitable dimensions, use SIMD batch processing
        #[cfg(target_arch = "x86_64")]
        {
            if self.count > 100 && self.dimension >= 16 && is_x86_feature_detected!("avx2") {
                return self.k_nearest_neighbors_simd_avx2(query, k);
            }
        }

        let mut distances = Vec::with_capacity(self.count);

        // Compute distances to all entities in a cache-friendly manner
        for i in 0..self.count {
            let start_idx = i * self.dimension;
            let end_idx = start_idx + self.dimension;
            let entity_embedding = &self.embeddings[start_idx..end_idx];
            
            // Use cosine similarity (convert to distance)
            let similarity = cosine_similarity(query, entity_embedding);
            let distance = 1.0 - similarity;
            
            distances.push((self.entity_ids[i], distance));
        }

        // Use partial sort for better performance than full sort
        if k < distances.len() {
            // Use select_nth_unstable for O(n) average case
            distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
        }

        // Final sort of just the k elements
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances
    }

    /// Optimized k-nearest neighbors with early termination
    pub fn k_nearest_neighbors_fast(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if query.len() != self.dimension || self.count == 0 {
            return Vec::new();
        }

        // For small k, use a max-heap to maintain top-k
        if k <= 32 && self.count > k * 4 {
            return self.k_nearest_neighbors_heap(query, k);
        }

        // For larger k or smaller datasets, use the full approach
        self.k_nearest_neighbors(query, k)
    }

    fn k_nearest_neighbors_heap(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(PartialEq)]
        struct DistanceEntry {
            entity_id: u32,
            distance: f32,
        }

        impl Eq for DistanceEntry {}

        impl PartialOrd for DistanceEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Max heap: larger distances first
                self.distance.partial_cmp(&other.distance)
            }
        }

        impl Ord for DistanceEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap = BinaryHeap::with_capacity(k + 1);

        for i in 0..self.count {
            let start_idx = i * self.dimension;
            let end_idx = start_idx + self.dimension;
            let entity_embedding = &self.embeddings[start_idx..end_idx];
            
            let similarity = cosine_similarity(query, entity_embedding);
            let distance = 1.0 - similarity;

            if heap.len() < k {
                heap.push(DistanceEntry {
                    entity_id: self.entity_ids[i],
                    distance,
                });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(DistanceEntry {
                        entity_id: self.entity_ids[i],
                        distance,
                    });
                }
            }
        }

        // Convert heap to sorted vector (smallest distances first)
        let mut results: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|entry| (entry.entity_id, entry.distance))
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn memory_usage(&self) -> usize {
        self.entity_ids.capacity() * std::mem::size_of::<u32>() +
        self.entity_keys.capacity() * std::mem::size_of::<EntityKey>() +
        self.embeddings.capacity() * std::mem::size_of::<f32>()
    }
    
    /// Get the capacity of the index
    pub fn capacity(&self) -> usize {
        self.entity_ids.capacity()
    }
    
    /// Add edge (not applicable - FlatVectorIndex stores embeddings, not edges)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation(
            "FlatVectorIndex stores entity embeddings, not edges. Use CSRGraph for edges.".to_string()
        ))
    }
    
    /// Update entity embedding
    pub fn update_entity(&mut self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        
        // Find the entity index
        if let Some(index) = self.entity_ids.iter().position(|&id| id == entity_id) {
            // Update the embedding
            let start_idx = index * self.dimension;
            let end_idx = start_idx + self.dimension;
            self.embeddings[start_idx..end_idx].copy_from_slice(&embedding);
            self.entity_keys[index] = entity_key;
            Ok(())
        } else {
            Err(GraphError::EntityNotFound { id: entity_id })
        }
    }
    
    /// Remove an entity from the index
    pub fn remove(&mut self, entity_id: u32) -> Result<()> {
        if let Some(index) = self.entity_ids.iter().position(|&id| id == entity_id) {
            // Remove from all arrays
            self.entity_ids.remove(index);
            self.entity_keys.remove(index);
            
            // Remove embedding data
            let start_idx = index * self.dimension;
            let end_idx = start_idx + self.dimension;
            self.embeddings.drain(start_idx..end_idx);
            
            self.count -= 1;
            Ok(())
        } else {
            Err(GraphError::EntityNotFound { id: entity_id })
        }
    }
    
    /// Check if index contains an entity
    pub fn contains_entity(&self, entity_id: u32) -> bool {
        self.entity_ids.contains(&entity_id)
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        // Size needed to serialize this index
        std::mem::size_of::<usize>() * 2 + // dimension, count
        self.entity_ids.len() * std::mem::size_of::<u32>() +
        self.entity_keys.len() * std::mem::size_of::<EntityKey>() +
        self.embeddings.len() * std::mem::size_of::<f32>()
    }

    /// SIMD-accelerated k-nearest neighbors using AVX2
    #[cfg(target_arch = "x86_64")]
    fn k_nearest_neighbors_simd_avx2(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        use crate::embedding::similarity::simd;
        
        if self.count == 0 {
            return Vec::new();
        }
        
        // Pre-allocate similarity scores for all entities
        let mut similarities = vec![0.0f32; self.count];
        
        // Use SIMD batch processing for computing all similarities at once
        unsafe {
            simd::batch_cosine_similarity_avx2(
                query,
                &self.embeddings,
                self.dimension,
                &mut similarities
            );
        }
        
        // Convert similarities to distances and pair with entity IDs
        let mut distances: Vec<(u32, f32)> = similarities
            .into_iter()
            .enumerate()
            .map(|(i, similarity)| (self.entity_ids[i], 1.0 - similarity))
            .collect();
        
        // Use partial sort for better performance than full sort
        if k < distances.len() {
            distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
        }
        
        // Final sort of just the k elements
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances
    }
    
    /// SIMD-accelerated similarity search for exact similarity values
    #[cfg(target_arch = "x86_64")]
    pub fn similarity_search_simd(&self, query: &[f32], threshold: f32) -> Vec<(u32, f32)> {
        use crate::embedding::similarity::simd;
        
        if query.len() != self.dimension || self.count == 0 {
            return Vec::new();
        }
        
        let mut similarities = vec![0.0f32; self.count];
        
        // Use SIMD batch processing
        if is_x86_feature_detected!("avx2") && self.dimension >= 16 {
            unsafe {
                simd::batch_cosine_similarity_avx2(
                    query,
                    &self.embeddings,
                    self.dimension,
                    &mut similarities
                );
            }
        } else {
            // Fallback to scalar computation
            for i in 0..self.count {
                let start_idx = i * self.dimension;
                let end_idx = start_idx + self.dimension;
                let entity_embedding = &self.embeddings[start_idx..end_idx];
                similarities[i] = crate::embedding::similarity::cosine_similarity_scalar(query, entity_embedding);
            }
        }
        
        // Filter by threshold and collect results
        let mut results: Vec<(u32, f32)> = similarities
            .into_iter()
            .enumerate()
            .filter_map(|(i, similarity)| {
                if similarity >= threshold {
                    Some((self.entity_ids[i], similarity))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_flat_index_creation() {
        let index = FlatVectorIndex::new(128);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_flat_index_insertion() {
        let mut index = FlatVectorIndex::new(3);
        let key = EntityKey::default();
        let embedding = vec![1.0, 2.0, 3.0];
        
        index.insert(1, key, embedding).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_nearest_neighbor_search() {
        let mut index = FlatVectorIndex::new(3);
        
        // Insert test points
        let points = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![1.0, 1.0, 0.0]),
        ];
        
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = points
            .into_iter()
            .map(|(id, embedding)| (id, EntityKey::default(), embedding))
            .collect();
        
        index.bulk_build(entities).unwrap();
        
        // Query near [1, 0, 0]
        let query = vec![0.9, 0.1, 0.0];
        let results = index.k_nearest_neighbors(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Should be closest to [1, 0, 0]
    }

    #[test]
    fn test_heap_vs_full_search() {
        let mut index = FlatVectorIndex::new(4);
        
        // Create a larger dataset
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..1000)
            .map(|i| {
                let embedding = vec![
                    (i as f32 / 1000.0),
                    ((i * 2) as f32 / 1000.0),
                    ((i * 3) as f32 / 1000.0),
                    ((i * 4) as f32 / 1000.0),
                ];
                (i as u32, EntityKey::default(), embedding)
            })
            .collect();
        
        index.bulk_build(entities).unwrap();
        
        let query = vec![0.5, 1.0, 1.5, 2.0];
        let results_full = index.k_nearest_neighbors(&query, 10);
        let results_fast = index.k_nearest_neighbors_fast(&query, 10);
        
        // Results should be identical
        assert_eq!(results_full.len(), results_fast.len());
        for (full, fast) in results_full.iter().zip(results_fast.iter()) {
            assert_eq!(full.0, fast.0);
            assert!((full.1 - fast.1).abs() < 1e-6);
        }
    }
}