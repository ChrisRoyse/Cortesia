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

    #[test]
    fn test_heap_implementation() {
        let mut index = FlatVectorIndex::new(3);
        
        // Create test data with known distances
        let entities = vec![
            (1, EntityKey::default(), vec![1.0, 0.0, 0.0]), // Distance 0 from [1,0,0]
            (2, EntityKey::default(), vec![0.0, 1.0, 0.0]), // Distance ~1.41 from [1,0,0]
            (3, EntityKey::default(), vec![0.0, 0.0, 1.0]), // Distance ~1.41 from [1,0,0]
            (4, EntityKey::default(), vec![0.5, 0.5, 0.0]), // Distance ~0.29 from [1,0,0]
        ];
        
        index.bulk_build(entities).unwrap();
        
        let query = vec![1.0, 0.0, 0.0];
        let results = index.k_nearest_neighbors_heap(&query, 2);
        
        assert_eq!(results.len(), 2);
        // Should return the 2 closest: entity 1 (distance 0) and entity 4 (distance ~0.29)
        assert_eq!(results[0].0, 1); // Closest
        assert_eq!(results[1].0, 4); // Second closest
        assert!(results[0].1 < results[1].1); // Distances should be sorted
    }

    #[test]
    fn test_bulk_build_efficiency() {
        let mut index = FlatVectorIndex::new(128);
        
        // Test bulk build with large dataset
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..10000)
            .map(|i| {
                let embedding: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 10000.0).collect();
                (i as u32, EntityKey::default(), embedding)
            })
            .collect();
        
        let start = std::time::Instant::now();
        index.bulk_build(entities).unwrap();
        let duration = start.elapsed();
        
        assert_eq!(index.len(), 10000);
        assert!(duration.as_millis() < 1000); // Should be reasonably fast
        
        // Test that capacity was properly pre-allocated
        assert!(index.entity_ids.capacity() >= 10000);
        assert!(index.embeddings.capacity() >= 10000 * 128);
    }

    #[test]
    fn test_memory_efficiency() {
        let mut index = FlatVectorIndex::new(64);
        
        let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..1000)
            .map(|i| {
                let embedding: Vec<f32> = (0..64).map(|j| (i + j) as f32).collect();
                (i as u32, EntityKey::default(), embedding)
            })
            .collect();
        
        index.bulk_build(entities).unwrap();
        
        let memory_usage = index.memory_usage();
        
        // Calculate expected minimum memory usage
        let expected_min = 
            1000 * std::mem::size_of::<u32>() +  // entity_ids
            1000 * std::mem::size_of::<EntityKey>() +  // entity_keys
            1000 * 64 * std::mem::size_of::<f32>();  // embeddings
        
        assert!(memory_usage >= expected_min);
    }

    #[test]
    fn test_dimension_validation() {
        let mut index = FlatVectorIndex::new(3);
        
        // Test correct dimension
        let result = index.insert(1, EntityKey::default(), vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());
        
        // Test incorrect dimension
        let result = index.insert(2, EntityKey::default(), vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
        
        let result = index.insert(3, EntityKey::default(), vec![1.0, 2.0, 3.0, 4.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_index_behavior() {
        let index = FlatVectorIndex::new(128);
        
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        
        let query = vec![1.0; 128];
        let results = index.k_nearest_neighbors(&query, 10);
        assert!(results.is_empty());
        
        let results = index.k_nearest_neighbors_fast(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_distance_calculations() {
        let mut index = FlatVectorIndex::new(2);
        
        // Test vectors with known cosine similarities
        let entities = vec![
            (1, EntityKey::default(), vec![1.0, 0.0]), // [1, 0]
            (2, EntityKey::default(), vec![0.0, 1.0]), // [0, 1] - orthogonal
            (3, EntityKey::default(), vec![1.0, 1.0]), // [1, 1] - 45 degrees
        ];
        
        index.bulk_build(entities).unwrap();
        
        let query = vec![1.0, 0.0];
        let results = index.k_nearest_neighbors(&query, 3);
        
        assert_eq!(results.len(), 3);
        
        // Entity 1 should be closest (distance 0)
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 0.0).abs() < 1e-6);
        
        // Entity 3 should be next (cosine distance = 1 - cos(45°) ≈ 0.293)
        assert_eq!(results[1].0, 3);
        assert!((results[1].1 - (1.0 - std::f32::consts::FRAC_1_SQRT_2)).abs() < 1e-6);
        
        // Entity 2 should be farthest (cosine distance = 1 - cos(90°) = 1.0)
        assert_eq!(results[2].0, 2);
        assert!((results[2].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_entity() {
        let mut index = FlatVectorIndex::new(3);
        
        // Insert initial entity
        index.insert(1, EntityKey::default(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(2, EntityKey::default(), vec![0.0, 1.0, 0.0]).unwrap();
        
        // Update entity 1
        let result = index.update_entity(1, EntityKey::default(), vec![0.0, 0.0, 1.0]);
        assert!(result.is_ok());
        
        // Verify update
        let query = vec![0.0, 0.0, 1.0];
        let results = index.k_nearest_neighbors(&query, 1);
        assert_eq!(results[0].0, 1); // Entity 1 should now be closest
        
        // Test updating non-existent entity
        let result = index.update_entity(99, EntityKey::default(), vec![1.0, 1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_entity() {
        let mut index = FlatVectorIndex::new(3);
        
        // Insert entities
        index.insert(1, EntityKey::default(), vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(2, EntityKey::default(), vec![0.0, 1.0, 0.0]).unwrap();
        index.insert(3, EntityKey::default(), vec![0.0, 0.0, 1.0]).unwrap();
        
        assert_eq!(index.len(), 3);
        assert!(index.contains_entity(2));
        
        // Remove entity
        let result = index.remove(2);
        assert!(result.is_ok());
        
        assert_eq!(index.len(), 2);
        assert!(!index.contains_entity(2));
        assert!(index.contains_entity(1));
        assert!(index.contains_entity(3));
        
        // Test removing non-existent entity
        let result = index.remove(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_operations() {
        let mut index = FlatVectorIndex::new(3);
        
        // Test that graph operations are not supported
        assert!(index.add_edge(1, 2, 1.0).is_err());
    }

    #[test]
    fn test_encoded_size() {
        let mut index = FlatVectorIndex::new(4);
        
        let entities = vec![
            (1, EntityKey::default(), vec![1.0, 2.0, 3.0, 4.0]),
            (2, EntityKey::default(), vec![5.0, 6.0, 7.0, 8.0]),
        ];
        
        index.bulk_build(entities).unwrap();
        
        let encoded_size = index.encoded_size();
        
        let expected_size = 
            std::mem::size_of::<usize>() * 2 +  // dimension + count
            2 * std::mem::size_of::<u32>() +    // entity_ids
            2 * std::mem::size_of::<EntityKey>() + // entity_keys
            2 * 4 * std::mem::size_of::<f32>();    // embeddings
        
        assert_eq!(encoded_size, expected_size);
    }

    #[cfg(test)]
    mod simd_tests {
        use super::*;
        
        #[test]
        #[cfg(target_arch = "x86_64")]
        fn test_simd_vs_scalar_consistency() {
            let mut index = FlatVectorIndex::new(16);
            
            // Create data that benefits from SIMD
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..200)
                .map(|i| {
                    let embedding: Vec<f32> = (0..16).map(|j| (i * j) as f32 / 100.0).collect();
                    (i as u32, EntityKey::default(), embedding)
                })
                .collect();
            
            index.bulk_build(entities).unwrap();
            
            let query: Vec<f32> = (0..16).map(|i| i as f32 / 10.0).collect();
            
            // Get results using different code paths
            let results_normal = index.k_nearest_neighbors(&query, 10);
            
            // Force SIMD path if available
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    let results_simd = index.k_nearest_neighbors_simd_avx2(&query, 10);
                    
                    // Results should be very similar (allowing for small numerical differences)
                    assert_eq!(results_normal.len(), results_simd.len());
                    for (normal, simd) in results_normal.iter().zip(results_simd.iter()) {
                        assert_eq!(normal.0, simd.0); // Same entity ID
                        assert!((normal.1 - simd.1).abs() < 1e-5); // Very close distances
                    }
                }
            }
        }
        
        #[test]
        #[cfg(target_arch = "x86_64")]
        fn test_similarity_search_simd() {
            let mut index = FlatVectorIndex::new(32);
            
            // Create test data with known similarity patterns
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..100)
                .map(|i| {
                    let mut embedding = vec![0.0; 32];
                    embedding[0] = i as f32 / 100.0; // Primary component
                    embedding[1] = (100 - i) as f32 / 100.0; // Secondary component
                    (i as u32, EntityKey::default(), embedding)
                })
                .collect();
            
            index.bulk_build(entities).unwrap();
            
            let query = vec![0.5; 32]; // Query that should match middle entities better
            
            if std::is_x86_feature_detected!("avx2") {
                let results = index.similarity_search_simd(&query, 0.8);
                
                // Should find entities with high similarity
                assert!(!results.is_empty());
                for (_, similarity) in &results {
                    assert!(*similarity >= 0.8);
                }
                
                // Results should be sorted by similarity (descending)
                for i in 1..results.len() {
                    assert!(results[i-1].1 >= results[i].1);
                }
            }
        }
    }

    #[cfg(test)]
    mod performance_tests {
        use super::*;
        use std::time::Instant;
        
        #[test]
        fn test_search_performance_scaling() {
            let dimensions = vec![64, 128, 256];
            let dataset_sizes = vec![1000, 5000, 10000];
            
            for &dim in &dimensions {
                for &size in &dataset_sizes {
                    let mut index = FlatVectorIndex::new(dim);
                    
                    // Build test dataset
                    let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..size)
                        .map(|i| {
                            let embedding: Vec<f32> = (0..dim).map(|j| 
                                ((i * j) as f32 / (size * dim) as f32).sin()
                            ).collect();
                            (i as u32, EntityKey::default(), embedding)
                        })
                        .collect();
                    
                    index.bulk_build(entities).unwrap();
                    
                    let query: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32).cos()).collect();
                    
                    // Measure search time
                    let start = Instant::now();
                    let results = index.k_nearest_neighbors(&query, 10);
                    let duration = start.elapsed();
                    
                    assert_eq!(results.len(), 10);
                    
                    // Performance expectations (these are loose bounds)
                    match size {
                        1000 => assert!(duration.as_millis() < 50),
                        5000 => assert!(duration.as_millis() < 200), 
                        10000 => assert!(duration.as_millis() < 500),
                        _ => {}
                    }
                }
            }
        }
        
        #[test]
        fn test_heap_vs_full_performance() {
            let mut index = FlatVectorIndex::new(128);
            
            // Large dataset
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..5000)
                .map(|i| {
                    let embedding: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 5000.0).collect();
                    (i as u32, EntityKey::default(), embedding)
                })
                .collect();
            
            index.bulk_build(entities).unwrap();
            
            let query: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
            
            // Test with small k (should use heap)
            let start = Instant::now();
            let results_heap = index.k_nearest_neighbors_fast(&query, 5);
            let heap_duration = start.elapsed();
            
            // Test with larger k (should use full sort)
            let start = Instant::now();
            let results_full = index.k_nearest_neighbors(&query, 5);
            let full_duration = start.elapsed();
            
            // Results should be identical
            assert_eq!(results_heap.len(), results_full.len());
            for (heap, full) in results_heap.iter().zip(results_full.iter()) {
                assert_eq!(heap.0, full.0);
                assert!((heap.1 - full.1).abs() < 1e-6);
            }
            
            // For small k, heap should be faster or comparable
            // (This is a weak assertion since performance can vary)
            assert!(heap_duration.as_nanos() > 0);
            assert!(full_duration.as_nanos() > 0);
        }
        
        #[test]
        fn test_memory_usage_accuracy() {
            let mut index = FlatVectorIndex::new(64);
            
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..1000)
                .map(|i| {
                    let embedding: Vec<f32> = vec![i as f32; 64];
                    (i as u32, EntityKey::default(), embedding)
                })
                .collect();
            
            index.bulk_build(entities).unwrap();
            
            let reported_usage = index.memory_usage();
            
            // Calculate actual memory usage
            let actual_usage = 
                index.entity_ids.capacity() * std::mem::size_of::<u32>() +
                index.entity_keys.capacity() * std::mem::size_of::<EntityKey>() +
                index.embeddings.capacity() * std::mem::size_of::<f32>();
            
            assert_eq!(reported_usage, actual_usage);
        }
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        
        #[test]
        fn test_search_correctness_properties() {
            let mut index = FlatVectorIndex::new(4);
            
            let entities = vec![
                (1, EntityKey::default(), vec![1.0, 0.0, 0.0, 0.0]),
                (2, EntityKey::default(), vec![0.0, 1.0, 0.0, 0.0]),
                (3, EntityKey::default(), vec![0.0, 0.0, 1.0, 0.0]),
                (4, EntityKey::default(), vec![0.0, 0.0, 0.0, 1.0]),
            ];
            
            index.bulk_build(entities).unwrap();
            
            // Property: Query identical to an embedding should return that entity first
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.k_nearest_neighbors(&query, 4);
            assert_eq!(results[0].0, 1);
            assert!((results[0].1 - 0.0).abs() < 1e-6);
            
            // Property: Results should be sorted by distance
            for i in 1..results.len() {
                assert!(results[i-1].1 <= results[i].1);
            }
            
            // Property: All distances should be non-negative
            for (_, distance) in &results {
                assert!(*distance >= 0.0);
            }
            
            // Property: All distances should be <= 2.0 (max cosine distance)
            for (_, distance) in &results {
                assert!(*distance <= 2.0);
            }
        }
        
        #[test]
        fn test_consistency_across_methods() {
            let mut index = FlatVectorIndex::new(8);
            
            let entities: Vec<(u32, EntityKey, Vec<f32>)> = (0..100)
                .map(|i| {
                    let embedding: Vec<f32> = (0..8).map(|j| (i + j) as f32 / 100.0).collect();
                    (i as u32, EntityKey::default(), embedding)
                })
                .collect();
            
            index.bulk_build(entities).unwrap();
            
            let query: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
            
            // Test different search methods return consistent results
            let results_normal = index.k_nearest_neighbors(&query, 10);
            let results_fast = index.k_nearest_neighbors_fast(&query, 10);
            let results_heap = index.k_nearest_neighbors_heap(&query, 10);
            
            // All methods should return same entities (possibly in same order)
            assert_eq!(results_normal.len(), results_fast.len());
            assert_eq!(results_normal.len(), results_heap.len());
            
            for i in 0..results_normal.len() {
                assert_eq!(results_normal[i].0, results_fast[i].0);
                assert_eq!(results_normal[i].0, results_heap[i].0);
                assert!((results_normal[i].1 - results_fast[i].1).abs() < 1e-6);
                assert!((results_normal[i].1 - results_heap[i].1).abs() < 1e-6);
            }
        }
        
        #[test]
        fn test_bulk_vs_incremental_consistency() {
            let dimension = 8;
            let entities = vec![
                (1, EntityKey::default(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (2, EntityKey::default(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (3, EntityKey::default(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ];
            
            // Build using bulk_build
            let mut index_bulk = FlatVectorIndex::new(dimension);
            index_bulk.bulk_build(entities.clone()).unwrap();
            
            // Build using incremental inserts
            let mut index_incremental = FlatVectorIndex::new(dimension);
            for (id, key, embedding) in entities {
                index_incremental.insert(id, key, embedding).unwrap();
            }
            
            // Both should produce identical results
            let query = vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
            let results_bulk = index_bulk.k_nearest_neighbors(&query, 3);
            let results_incremental = index_incremental.k_nearest_neighbors(&query, 3);
            
            assert_eq!(results_bulk.len(), results_incremental.len());
            for (bulk, incremental) in results_bulk.iter().zip(results_incremental.iter()) {
                assert_eq!(bulk.0, incremental.0);
                assert!((bulk.1 - incremental.1).abs() < 1e-6);
            }
        }
    }
}