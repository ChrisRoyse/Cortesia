use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};
use crate::embedding::similarity::cosine_similarity;
use std::collections::HashMap;
use parking_lot::RwLock;
use rand::Rng;

/// Locality Sensitive Hashing (LSH) index for fast approximate similarity search
/// Uses random hyperplane hashing for cosine similarity
pub struct LshIndex {
    /// Random hyperplanes for hashing
    hyperplanes: Vec<Vec<f32>>,
    /// Hash tables storing entities by hash signature
    hash_tables: RwLock<HashMap<u64, Vec<LshEntity>>>,
    /// Vector dimension
    dimension: usize,
    /// Number of hash functions (hyperplanes)
    num_hashes: usize,
    /// Number of hash tables for multi-probing
    num_tables: usize,
}

#[derive(Clone)]
struct LshEntity {
    entity_id: u32,
    entity_key: EntityKey,
    embedding: Vec<f32>,
    hash_signature: u64,
}

impl LshIndex {
    /// Create a new LSH index
    pub fn new(dimension: usize, num_hashes: usize, num_tables: usize) -> Self {
        let mut hyperplanes = Vec::with_capacity(num_hashes * num_tables);
        let mut rng = rand::thread_rng();
        
        // Generate random hyperplanes
        for _ in 0..(num_hashes * num_tables) {
            let mut hyperplane = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                hyperplane.push(rng.gen_range(-1.0..1.0));
            }
            
            // Normalize the hyperplane
            let norm: f32 = hyperplane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut hyperplane {
                    *val /= norm;
                }
            }
            
            hyperplanes.push(hyperplane);
        }
        
        Self {
            hyperplanes,
            hash_tables: RwLock::new(HashMap::new()),
            dimension,
            num_hashes,
            num_tables,
        }
    }

    /// Create LSH index with optimized parameters for different use cases
    pub fn new_optimized(dimension: usize, target_precision: f32) -> Self {
        // Auto-configure parameters based on desired precision
        let (num_hashes, num_tables) = if target_precision > 0.9 {
            (20, 4) // High precision
        } else if target_precision > 0.8 {
            (16, 3) // Medium precision
        } else {
            (12, 2) // Fast search
        };
        
        Self::new(dimension, num_hashes, num_tables)
    }

    /// Insert a new vector into the LSH index
    pub fn insert(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let mut hash_tables = self.hash_tables.write();
        
        // Generate hash signatures for each table
        for table_idx in 0..self.num_tables {
            let hash_signature = self.compute_hash_signature(&embedding, table_idx);
            
            let entity = LshEntity {
                entity_id,
                entity_key,
                embedding: embedding.clone(),
                hash_signature,
            };
            
            hash_tables
                .entry(hash_signature)
                .or_default()
                .push(entity);
        }
        
        Ok(())
    }

    /// Search for similar vectors using LSH
    pub fn search(&self, query: &[f32], max_results: usize) -> Vec<(u32, f32)> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        let hash_tables = self.hash_tables.read();
        let mut candidates = HashMap::new();
        
        // Query each hash table
        for table_idx in 0..self.num_tables {
            let query_hash = self.compute_hash_signature(query, table_idx);
            
            // Exact hash match
            if let Some(entities) = hash_tables.get(&query_hash) {
                for entity in entities {
                    candidates.insert(entity.entity_id, entity);
                }
            }
            
            // Multi-probe LSH: also check nearby hash values
            for flip_bit in 0..self.num_hashes.min(3) { // Limit probing for performance
                let probe_hash = query_hash ^ (1u64 << flip_bit);
                if let Some(entities) = hash_tables.get(&probe_hash) {
                    for entity in entities {
                        candidates.insert(entity.entity_id, entity);
                    }
                }
            }
        }
        
        // Compute exact similarities for candidates
        let mut results: Vec<(u32, f32)> = candidates
            .values()
            .map(|entity| {
                let similarity = cosine_similarity(query, &entity.embedding);
                (entity.entity_id, similarity)
            })
            .collect();
        
        // Sort by similarity (descending) and limit results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        
        results
    }

    /// Advanced search with distance threshold
    pub fn search_threshold(&self, query: &[f32], threshold: f32) -> Vec<(u32, f32)> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        let hash_tables = self.hash_tables.read();
        let mut candidates = HashMap::new();
        
        // Query each hash table with more aggressive multi-probing for threshold queries
        for table_idx in 0..self.num_tables {
            let query_hash = self.compute_hash_signature(query, table_idx);
            
            // Check exact match and nearby hashes
            for probe_bits in 0..=2 { // More probing for threshold queries
                if probe_bits == 0 {
                    // Exact match
                    if let Some(entities) = hash_tables.get(&query_hash) {
                        for entity in entities {
                            candidates.insert(entity.entity_id, entity);
                        }
                    }
                } else {
                    // Generate all combinations of bit flips
                    for bits_to_flip in combinations(self.num_hashes.min(8), probe_bits) {
                        let mut probe_hash = query_hash;
                        for bit in bits_to_flip {
                            probe_hash ^= 1u64 << bit;
                        }
                        
                        if let Some(entities) = hash_tables.get(&probe_hash) {
                            for entity in entities {
                                candidates.insert(entity.entity_id, entity);
                            }
                        }
                    }
                }
            }
        }
        
        // Filter by threshold and compute exact similarities
        let mut results: Vec<(u32, f32)> = candidates
            .values()
            .filter_map(|entity| {
                let similarity = cosine_similarity(query, &entity.embedding);
                if similarity >= threshold {
                    Some((entity.entity_id, similarity))
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Compute hash signature for a vector using a specific table
    fn compute_hash_signature(&self, vector: &[f32], table_idx: usize) -> u64 {
        let mut signature = 0u64;
        let start_idx = table_idx * self.num_hashes;
        
        for i in 0..self.num_hashes {
            let hyperplane_idx = start_idx + i;
            if hyperplane_idx < self.hyperplanes.len() {
                let dot_product: f32 = vector
                    .iter()
                    .zip(self.hyperplanes[hyperplane_idx].iter())
                    .map(|(v, h)| v * h)
                    .sum();
                
                if dot_product >= 0.0 {
                    signature |= 1u64 << i;
                }
            }
        }
        
        signature
    }

    pub fn len(&self) -> usize {
        self.hash_tables.read().values().map(|bucket| bucket.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.hash_tables.read().is_empty()
    }

    /// Get LSH index statistics
    pub fn stats(&self) -> LshStats {
        let hash_tables = self.hash_tables.read();
        let total_entities = hash_tables.values().map(|bucket| bucket.len()).sum();
        let num_buckets = hash_tables.len();
        let max_bucket_size = hash_tables.values().map(|bucket| bucket.len()).max().unwrap_or(0);
        let avg_bucket_size = if num_buckets > 0 {
            total_entities as f64 / num_buckets as f64
        } else {
            0.0
        };
        
        // Calculate bucket size distribution
        let mut bucket_sizes: Vec<usize> = hash_tables.values().map(|bucket| bucket.len()).collect();
        bucket_sizes.sort_unstable();
        
        LshStats {
            total_entities,
            num_buckets,
            max_bucket_size,
            avg_bucket_size,
            num_tables: self.num_tables,
            num_hashes: self.num_hashes,
            bucket_size_p50: percentile(&bucket_sizes, 0.5),
            bucket_size_p95: percentile(&bucket_sizes, 0.95),
        }
    }

    /// Estimate recall for a given configuration (for tuning)
    pub fn estimate_recall(&self, sample_size: usize) -> f64 {
        let hash_tables = self.hash_tables.read();
        if hash_tables.is_empty() || sample_size == 0 {
            return 0.0;
        }
        
        // Sample entities and estimate how many would be found
        let entities: Vec<&LshEntity> = hash_tables.values().flatten().collect();
        if entities.len() < sample_size {
            return 1.0;
        }
        
        let mut total_found = 0;
        let mut rng = rand::thread_rng();
        
        for _ in 0..sample_size {
            let idx = rng.gen_range(0..entities.len());
            let test_entity = entities[idx];
            
            // Search for this entity and see if it's found
            let results = self.search(&test_entity.embedding, 50);
            if results.iter().any(|(id, _)| *id == test_entity.entity_id) {
                total_found += 1;
            }
        }
        
        total_found as f64 / sample_size as f64
    }
    
    /// Get the capacity of the index
    pub fn capacity(&self) -> usize {
        self.hash_tables.read().capacity()
    }
    
    /// Add edge (not applicable - LshIndex stores embeddings)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation(
            "LshIndex stores entity embeddings, not edges. Use CSRGraph for edges.".to_string()
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
        
        // Remove old entries
        self.remove(entity_id)?;
        
        // Insert new embedding
        self.insert(entity_id, entity_key, embedding)
    }
    
    /// Remove an entity from the index
    pub fn remove(&self, entity_id: u32) -> Result<()> {
        let mut hash_tables = self.hash_tables.write();
        let mut found = false;
        
        // Remove from all hash tables
        hash_tables.retain(|_, bucket| {
            let old_len = bucket.len();
            bucket.retain(|entity| entity.entity_id != entity_id);
            if bucket.len() < old_len {
                found = true;
            }
            !bucket.is_empty() // Remove empty buckets
        });
        
        if found {
            Ok(())
        } else {
            Err(GraphError::EntityNotFound { id: entity_id })
        }
    }
    
    /// Check if index contains an entity
    pub fn contains_entity(&self, entity_id: u32) -> bool {
        self.hash_tables.read()
            .values()
            .any(|bucket| bucket.iter().any(|entity| entity.entity_id == entity_id))
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        let hash_tables = self.hash_tables.read();
        
        let base_size = std::mem::size_of::<usize>() * 3 + // dimension, num_hashes, num_tables
            self.hyperplanes.len() * self.dimension * std::mem::size_of::<f32>();
            
        let entities_size = hash_tables.values()
            .flat_map(|bucket| bucket.iter())
            .map(|entity| {
                std::mem::size_of::<u32>() + // entity_id
                std::mem::size_of::<EntityKey>() +
                entity.embedding.len() * std::mem::size_of::<f32>() +
                std::mem::size_of::<u64>() // hash_signature
            })
            .sum::<usize>();
            
        base_size + entities_size
    }
}

/// Generate combinations of indices for multi-probing
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }
    
    let mut result = Vec::new();
    fn generate(current: &mut Vec<usize>, start: usize, n: usize, k: usize, result: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        
        for i in start..=n - (k - current.len()) {
            current.push(i);
            generate(current, i + 1, n, k, result);
            current.pop();
        }
    }
    
    let mut current = Vec::new();
    generate(&mut current, 0, n, k, &mut result);
    result
}

fn percentile(sorted_data: &[usize], p: f64) -> usize {
    if sorted_data.is_empty() {
        return 0;
    }
    let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

#[derive(Debug)]
pub struct LshStats {
    pub total_entities: usize,
    pub num_buckets: usize,
    pub max_bucket_size: usize,
    pub avg_bucket_size: f64,
    pub num_tables: usize,
    pub num_hashes: usize,
    pub bucket_size_p50: usize,
    pub bucket_size_p95: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_lsh_creation() {
        let index = LshIndex::new(128, 16, 4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_lsh_insertion() {
        let index = LshIndex::new(3, 8, 2);
        let key = EntityKey::default();
        let embedding = vec![1.0, 0.0, 0.0];
        
        index.insert(1, key, embedding).unwrap();
        assert!(!index.is_empty());
        assert!(!index.is_empty());
    }

    #[test]
    fn test_lsh_search() {
        let index = LshIndex::new(3, 12, 3);
        
        // Insert test points
        let points = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.9, 0.1, 0.0]), // Similar to point 1
        ];
        
        for (id, embedding) in points {
            index.insert(id, EntityKey::default(), embedding).unwrap();
        }
        
        // Query similar to [1, 0, 0]
        let query = vec![0.95, 0.05, 0.0];
        let results = index.search(&query, 10);
        
        assert!(!results.is_empty());
        // Should find entity 1 or 4 with high similarity
        assert!(results.iter().any(|(id, sim)| (*id == 1 || *id == 4) && *sim > 0.8));
    }

    #[test]
    fn test_lsh_threshold_search() {
        let index = LshIndex::new(4, 16, 4);
        
        // Insert test vectors
        for i in 0..20 {
            let embedding = vec![
                (i as f32 / 20.0),
                ((i * 2) as f32 / 20.0),
                ((i * 3) as f32 / 20.0),
                ((i * 4) as f32 / 20.0),
            ];
            index.insert(i as u32, EntityKey::default(), embedding).unwrap();
        }
        
        // Search with threshold
        let query = vec![0.5, 1.0, 1.5, 2.0];
        let results = index.search_threshold(&query, 0.8);
        
        // All results should have similarity >= 0.8
        for (_, similarity) in &results {
            assert!(*similarity >= 0.8);
        }
    }

    #[test]
    fn test_lsh_stats() {
        let index = LshIndex::new(4, 12, 3);
        
        // Insert multiple points
        for i in 0..100 {
            let embedding = vec![
                (i as f32 / 100.0),
                ((i * 2) as f32 / 100.0),
                ((i * 3) as f32 / 100.0),
                ((i * 4) as f32 / 100.0),
            ];
            index.insert(i as u32, EntityKey::default(), embedding).unwrap();
        }
        
        let stats = index.stats();
        assert!(stats.total_entities >= 100); // May be more due to multiple tables
        assert!(stats.num_buckets > 0);
        assert!(stats.avg_bucket_size > 0.0);
    }

    #[test]
    fn test_combinations() {
        let combos = combinations(4, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6
        assert!(combos.contains(&vec![0, 1]));
        assert!(combos.contains(&vec![2, 3]));
    }
}