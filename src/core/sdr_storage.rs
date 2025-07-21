use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use ahash::AHashMap;
use rand::prelude::*;

use crate::core::types::EntityKey;
use crate::core::sdr_types::{SDRConfig, SDR, SDRPattern, SDRStatistics, SimilaritySearchResult, SDREntry, SDRQuery};
use crate::core::sdr_index::SimilarityIndex;
use crate::error::{Result, GraphError};


/// SDR storage and retrieval system
pub struct SDRStorage {
    patterns: Arc<RwLock<AHashMap<String, SDRPattern>>>,
    entity_patterns: Arc<RwLock<AHashMap<EntityKey, String>>>, // entity -> pattern_id
    similarity_index: Arc<RwLock<SimilarityIndex>>,
    config: SDRConfig,
}

impl std::fmt::Debug for SDRStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SDRStorage")
            .field("patterns", &"Arc<RwLock<AHashMap>>")
            .field("entity_patterns", &"Arc<RwLock<AHashMap>>")
            .field("similarity_index", &"Arc<RwLock<SimilarityIndex>>")
            .field("config", &self.config)
            .finish()
    }
}

impl SDRStorage {
    pub fn new(config: SDRConfig) -> Self {
        Self {
            patterns: Arc::new(RwLock::new(AHashMap::new())),
            entity_patterns: Arc::new(RwLock::new(AHashMap::new())),
            similarity_index: Arc::new(RwLock::new(SimilarityIndex::new())),
            config,
        }
    }

    pub async fn new_default() -> Result<Self> {
        Ok(Self::new(SDRConfig::default()))
    }

    /// Store an SDR pattern
    pub async fn store_pattern(&self, pattern: SDRPattern) -> Result<()> {
        let pattern_id = pattern.pattern_id.clone();
        
        // Update similarity index
        {
            let mut index = self.similarity_index.write().await;
            index.add_pattern(&pattern)?;
        }

        // Store pattern
        {
            let mut patterns = self.patterns.write().await;
            patterns.insert(pattern_id, pattern);
        }

        Ok(())
    }

    /// Associate an entity with an SDR pattern
    pub async fn associate_entity(&self, entity_key: EntityKey, pattern_id: String) -> Result<()> {
        let mut entity_patterns = self.entity_patterns.write().await;
        entity_patterns.insert(entity_key, pattern_id);
        Ok(())
    }

    /// Get SDR pattern for an entity
    pub async fn get_entity_pattern(&self, entity_key: EntityKey) -> Result<Option<SDRPattern>> {
        let entity_patterns = self.entity_patterns.read().await;
        if let Some(pattern_id) = entity_patterns.get(&entity_key) {
            let patterns = self.patterns.read().await;
            Ok(patterns.get(pattern_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Find similar patterns based on SDR overlap
    pub async fn find_similar_patterns(
        &self,
        query_sdr: &SDR,
        max_results: usize,
    ) -> Result<Vec<(String, f32)>> {
        let index = self.similarity_index.read().await;
        index.find_similar(query_sdr, max_results, self.config.overlap_threshold)
    }

    /// Store dense vector as SDR
    pub async fn store_dense_vector(
        &self,
        entity_key: EntityKey,
        vector: &[f32],
        concept_name: String,
    ) -> Result<String> {
        let sdr = SDR::from_dense_vector(vector, &self.config);
        let pattern_id = format!("sdr_{}", uuid::Uuid::new_v4());
        
        let pattern = SDRPattern::new(pattern_id.clone(), sdr, concept_name);
        
        self.store_pattern(pattern).await?;
        self.associate_entity(entity_key, pattern_id.clone()).await?;
        
        Ok(pattern_id)
    }

    /// Retrieve dense vector from SDR
    pub async fn get_dense_vector(&self, entity_key: EntityKey) -> Result<Option<Vec<f32>>> {
        if let Some(pattern) = self.get_entity_pattern(entity_key).await? {
            Ok(Some(pattern.sdr.to_dense_vector()))
        } else {
            Ok(None)
        }
    }

    /// Get storage statistics
    pub async fn get_statistics(&self) -> Result<SDRStatistics> {
        let patterns = self.patterns.read().await;
        let entity_patterns = self.entity_patterns.read().await;
        
        let total_patterns = patterns.len();
        let total_entities = entity_patterns.len();
        
        let average_sparsity = if !patterns.is_empty() {
            patterns.values()
                .map(|p| p.sdr.sparsity())
                .sum::<f32>() / patterns.len() as f32
        } else {
            0.0
        };

        let total_active_bits: usize = patterns.values()
            .map(|p| p.sdr.active_bits.len())
            .sum();

        Ok(SDRStatistics {
            total_patterns,
            total_entities,
            average_sparsity,
            total_active_bits,
            config: self.config.clone(),
        })
    }

    /// Get entity mappings for pattern lookup
    pub async fn get_entity_mappings(&self) -> AHashMap<String, EntityKey> {
        let entity_patterns = self.entity_patterns.read().await;
        let mut result = AHashMap::new();
        for (entity_key, pattern_id) in entity_patterns.iter() {
            result.insert(pattern_id.clone(), *entity_key);
        }
        result
    }

    /// Compact storage by removing unused patterns
    pub async fn compact(&self) -> Result<usize> {
        let mut patterns = self.patterns.write().await;
        let entity_patterns = self.entity_patterns.read().await;
        
        // Find unused patterns
        let used_pattern_ids: HashSet<String> = entity_patterns.values().cloned().collect();
        let all_pattern_ids: HashSet<String> = patterns.keys().cloned().collect();
        let unused_pattern_ids: HashSet<String> = all_pattern_ids.difference(&used_pattern_ids).cloned().collect();
        
        // Remove unused patterns
        for pattern_id in &unused_pattern_ids {
            patterns.remove(pattern_id);
        }

        // Update similarity index
        {
            let mut index = self.similarity_index.write().await;
            index.remove_patterns(&unused_pattern_ids)?;
        }

        Ok(unused_pattern_ids.len())
    }

    /// Encode text to SDR (placeholder implementation)
    pub async fn encode_text(&self, text: &str) -> Result<SDR> {
        // This is a placeholder - in a real implementation you would use
        // a text encoder to convert text to dense vectors then to SDR
        let hash_seed = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let mut rng = rand::rngs::StdRng::seed_from_u64(hash_seed as u64);
        
        Ok(SDR::random_with_rng(&self.config, &mut rng))
    }

    /// Store with metadata (placeholder implementation)
    pub async fn store_with_metadata(&self, sdr: &SDR, content: String, _importance: f32) -> Result<String> {
        let pattern_id = format!("pattern_{}", uuid::Uuid::new_v4());
        let pattern = SDRPattern::new(pattern_id.clone(), sdr.clone(), content);
        self.store_pattern(pattern).await?;
        Ok(pattern_id)
    }

    /// Similarity search (placeholder implementation)
    pub async fn similarity_search(&self, query: &str, threshold: f32) -> Result<Vec<SimilaritySearchResult>> {
        // Encode query to SDR
        let query_sdr = self.encode_text(query).await?;
        
        // Find similar patterns
        let similar_patterns = {
            let index = self.similarity_index.read().await;
            index.find_similar(&query_sdr, 10, threshold)?
        };
        
        // Convert to results
        let mut results = Vec::new();
        let patterns = self.patterns.read().await;
        
        for (pattern_id, similarity) in similar_patterns {
            if let Some(pattern) = patterns.get(&pattern_id) {
                results.push(SimilaritySearchResult {
                    pattern_id,
                    content: pattern.concept_name.clone(),
                    similarity,
                });
            }
        }
        
        Ok(results)
    }

    /// Get estimated memory usage in bytes
    pub async fn memory_usage(&self) -> usize {
        let patterns = self.patterns.read().await;
        let entity_patterns = self.entity_patterns.read().await;
        
        let patterns_size = patterns.len() * (
            std::mem::size_of::<String>() + // pattern_id
            std::mem::size_of::<SDRPattern>() + // pattern data
            self.config.total_bits / 8 // estimated SDR size
        );
        
        let entity_patterns_size = entity_patterns.len() * (
            std::mem::size_of::<EntityKey>() +
            std::mem::size_of::<String>()
        );
        
        patterns_size + entity_patterns_size + 1024 // base overhead
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use slotmap::SlotMap;

    // Helper function to create test entity key
    fn create_test_entity_key() -> EntityKey {
        let mut slot_map = SlotMap::with_key();
        slot_map.insert(())
    }

    #[test]
    fn test_sdr_creation() {
        let config = SDRConfig::default();
        let sdr = SDR::random(&config);
        
        assert_eq!(sdr.total_bits, config.total_bits);
        assert_eq!(sdr.active_bits.len(), config.active_bits);
        assert!(sdr.sparsity() <= config.sparsity + 0.01); // Allow small tolerance
    }

    #[test]
    fn test_sdr_creation_empty() {
        let active_bits = HashSet::new();
        let sdr = SDR::new(active_bits, 100);
        
        assert_eq!(sdr.total_bits, 100);
        assert_eq!(sdr.active_bits.len(), 0);
        assert_eq!(sdr.sparsity(), 0.0);
    }

    #[test]
    fn test_sdr_creation_full() {
        let active_bits: HashSet<usize> = (0..100).collect();
        let sdr = SDR::new(active_bits, 100);
        
        assert_eq!(sdr.total_bits, 100);
        assert_eq!(sdr.active_bits.len(), 100);
        assert_eq!(sdr.sparsity(), 1.0);
    }

    #[test]
    fn test_sdr_from_dense_vector() {
        let config = SDRConfig {
            total_bits: 100,
            active_bits: 10,
            sparsity: 0.1,
            overlap_threshold: 0.5,
        };
        
        let dense_vector = vec![0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0];
        let sdr = SDR::from_dense_vector(&dense_vector, &config);
        
        assert_eq!(sdr.active_bits.len(), config.active_bits.min(dense_vector.len()));
        assert!(sdr.active_bits.contains(&1)); // Highest value at index 1
    }

    #[test]
    fn test_sdr_from_dense_vector_empty() {
        let config = SDRConfig::default();
        let empty_vector: Vec<f32> = vec![];
        let sdr = SDR::from_dense_vector(&empty_vector, &config);
        
        assert_eq!(sdr.active_bits.len(), 0);
        assert_eq!(sdr.sparsity(), 0.0);
    }

    #[test]
    fn test_sdr_similarity_identical() {
        let active_bits: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let sdr1 = SDR::new(active_bits.clone(), 100);
        let sdr2 = SDR::new(active_bits, 100);
        
        assert_eq!(sdr1.overlap(&sdr2), 1.0);
        assert_eq!(sdr1.jaccard_similarity(&sdr2), 1.0);
        assert_eq!(sdr1.cosine_similarity(&sdr2), 1.0);
    }

    #[test]
    fn test_sdr_similarity_disjoint() {
        let sdr1 = SDR::new([1, 2, 3].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([4, 5, 6].iter().cloned().collect(), 100);
        
        assert_eq!(sdr1.overlap(&sdr2), 0.0);
        assert_eq!(sdr1.jaccard_similarity(&sdr2), 0.0);
        assert_eq!(sdr1.cosine_similarity(&sdr2), 0.0);
    }

    #[test]
    fn test_sdr_similarity_partial_overlap() {
        let sdr1 = SDR::new([1, 2, 3, 4, 5].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([3, 4, 5, 6, 7].iter().cloned().collect(), 100);
        
        let overlap = sdr1.overlap(&sdr2);
        let jaccard = sdr1.jaccard_similarity(&sdr2);
        let cosine = sdr1.cosine_similarity(&sdr2);
        
        assert!(overlap > 0.0 && overlap < 1.0);
        assert!(jaccard > 0.0 && jaccard < 1.0);
        assert!(cosine > 0.0 && cosine < 1.0);
        
        // Verify mathematical correctness
        assert_eq!(overlap, jaccard); // For SDR, overlap and jaccard are the same
        assert!((cosine - 0.6).abs() < 0.01); // 3/(sqrt(5)*sqrt(5)) = 0.6
    }

    #[test]
    fn test_sdr_similarity_empty_sdrs() {
        let sdr1 = SDR::new(HashSet::new(), 100);
        let sdr2 = SDR::new(HashSet::new(), 100);
        
        assert_eq!(sdr1.overlap(&sdr2), 0.0);
        assert_eq!(sdr1.jaccard_similarity(&sdr2), 0.0);
        assert_eq!(sdr1.cosine_similarity(&sdr2), 0.0);
    }

    #[test]
    fn test_sdr_similarity_different_dimensions() {
        let sdr1 = SDR::new([1, 2, 3].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([1, 2, 3].iter().cloned().collect(), 200);
        
        assert_eq!(sdr1.overlap(&sdr2), 0.0);
        assert_eq!(sdr1.jaccard_similarity(&sdr2), 0.0);
        assert_eq!(sdr1.cosine_similarity(&sdr2), 0.0);
    }

    #[test]
    fn test_sdr_to_dense_vector() {
        let active_bits: HashSet<usize> = [1, 3, 5].iter().cloned().collect();
        let sdr = SDR::new(active_bits, 10);
        let dense = sdr.to_dense_vector();
        
        assert_eq!(dense.len(), 10);
        assert_eq!(dense[1], 1.0);
        assert_eq!(dense[3], 1.0);
        assert_eq!(dense[5], 1.0);
        assert_eq!(dense[0], 0.0);
        assert_eq!(dense[2], 0.0);
        assert_eq!(dense[4], 0.0);
    }

    #[test]
    fn test_sdr_union() {
        let sdr1 = SDR::new([1, 2, 3].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([3, 4, 5].iter().cloned().collect(), 100);
        
        let union = sdr1.union(&sdr2).unwrap();
        let expected_bits: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        
        assert_eq!(union.active_bits, expected_bits);
        assert_eq!(union.total_bits, 100);
    }

    #[test]
    fn test_sdr_intersection() {
        let sdr1 = SDR::new([1, 2, 3, 4].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([3, 4, 5, 6].iter().cloned().collect(), 100);
        
        let intersection = sdr1.intersection(&sdr2).unwrap();
        let expected_bits: HashSet<usize> = [3, 4].iter().cloned().collect();
        
        assert_eq!(intersection.active_bits, expected_bits);
        assert_eq!(intersection.total_bits, 100);
    }

    #[test]
    fn test_sdr_union_different_dimensions() {
        let sdr1 = SDR::new([1, 2, 3].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([3, 4, 5].iter().cloned().collect(), 200);
        
        assert!(sdr1.union(&sdr2).is_err());
    }

    #[test]
    fn test_sdr_random_with_rng() {
        let config = SDRConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        let sdr1 = SDR::random_with_rng(&config, &mut rng);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let sdr2 = SDR::random_with_rng(&config, &mut rng2);
        
        // Same seed should produce identical SDRs
        assert_eq!(sdr1.active_bits, sdr2.active_bits);
    }

    #[tokio::test]
    async fn test_sdr_storage_new() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config.clone());
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.average_sparsity, 0.0);
        assert_eq!(stats.config.total_bits, config.total_bits);
    }

    #[tokio::test]
    async fn test_sdr_storage_new_default() {
        let storage = SDRStorage::new_default().await.unwrap();
        let stats = storage.get_statistics().await.unwrap();
        
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.config.total_bits, SDRConfig::default().total_bits);
    }

    #[tokio::test]
    async fn test_sdr_storage_store_pattern() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let sdr = SDR::random(&storage.config);
        let pattern = SDRPattern::new("test_pattern".to_string(), sdr, "test_concept".to_string());
        
        storage.store_pattern(pattern).await.unwrap();
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
        assert_eq!(stats.total_entities, 0); // No entity associations yet
    }

    #[tokio::test]
    async fn test_sdr_storage_multiple_patterns() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        for i in 0..10 {
            let sdr = SDR::random(&storage.config);
            let pattern = SDRPattern::new(format!("pattern_{}", i), sdr, format!("concept_{}", i));
            storage.store_pattern(pattern).await.unwrap();
        }
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 10);
    }

    #[tokio::test]
    async fn test_entity_pattern_association() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let entity_key = create_test_entity_key();
        let dense_vector = vec![0.5; 100];
        
        let pattern_id = storage.store_dense_vector(
            entity_key,
            &dense_vector,
            "test_concept".to_string(),
        ).await.unwrap();
        
        let retrieved_pattern = storage.get_entity_pattern(entity_key).await.unwrap();
        assert!(retrieved_pattern.is_some());
        assert_eq!(retrieved_pattern.unwrap().pattern_id, pattern_id);
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
        assert_eq!(stats.total_entities, 1);
    }

    #[tokio::test]
    async fn test_get_entity_pattern_nonexistent() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let entity_key = create_test_entity_key();
        let result = storage.get_entity_pattern(entity_key).await.unwrap();
        
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_find_similar_patterns() {
        let config = SDRConfig {
            total_bits: 100,
            active_bits: 10,
            sparsity: 0.1,
            overlap_threshold: 0.3,
        };
        let storage = SDRStorage::new(config.clone());
        
        // Store some patterns with known overlaps
        let base_bits: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let sdr1 = SDR::new(base_bits.clone(), 100);
        let pattern1 = SDRPattern::new("pattern1".to_string(), sdr1, "concept1".to_string());
        storage.store_pattern(pattern1).await.unwrap();
        
        // Similar pattern (3 overlapping bits)
        let similar_bits: HashSet<usize> = [3, 4, 5, 6, 7].iter().cloned().collect();
        let sdr2 = SDR::new(similar_bits, 100);
        let pattern2 = SDRPattern::new("pattern2".to_string(), sdr2, "concept2".to_string());
        storage.store_pattern(pattern2).await.unwrap();
        
        // Dissimilar pattern (no overlap)
        let dissimilar_bits: HashSet<usize> = [10, 11, 12, 13, 14].iter().cloned().collect();
        let sdr3 = SDR::new(dissimilar_bits, 100);
        let pattern3 = SDRPattern::new("pattern3".to_string(), sdr3, "concept3".to_string());
        storage.store_pattern(pattern3).await.unwrap();
        
        // Query with pattern similar to pattern1
        let query_sdr = SDR::new(base_bits, 100);
        let results = storage.find_similar_patterns(&query_sdr, 10).await.unwrap();
        
        assert!(!results.is_empty());
        // Should find pattern1 as most similar
        assert_eq!(results[0].0, "pattern1");
        assert_eq!(results[0].1, 1.0); // Identical
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        // Store a pattern
        let sdr = SDR::random(&storage.config);
        let pattern = SDRPattern::new("test_pattern".to_string(), sdr, "test content".to_string());
        storage.store_pattern(pattern).await.unwrap();
        
        let results = storage.similarity_search("test query", 0.0).await.unwrap();
        // Since we're using a placeholder implementation, we should get some results
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_get_dense_vector() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let entity_key = create_test_entity_key();
        let original_vector = vec![0.1, 0.9, 0.8, 0.2, 0.7];
        
        storage.store_dense_vector(
            entity_key,
            &original_vector,
            "test_concept".to_string(),
        ).await.unwrap();
        
        let retrieved_vector = storage.get_dense_vector(entity_key).await.unwrap();
        assert!(retrieved_vector.is_some());
        
        let dense = retrieved_vector.unwrap();
        assert!(!dense.is_empty());
        // The retrieved vector should be a binary representation
        for value in &dense {
            assert!(*value == 0.0 || *value == 1.0);
        }
    }

    #[tokio::test]
    async fn test_get_entity_mappings() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let entity1 = create_test_entity_key();
        let entity2 = create_test_entity_key();
        
        let pattern_id1 = storage.store_dense_vector(
            entity1,
            &vec![0.5; 10],
            "concept1".to_string(),
        ).await.unwrap();
        
        let pattern_id2 = storage.store_dense_vector(
            entity2,
            &vec![0.7; 10],
            "concept2".to_string(),
        ).await.unwrap();
        
        let mappings = storage.get_entity_mappings().await;
        assert_eq!(mappings.len(), 2);
        assert_eq!(mappings.get(&pattern_id1).unwrap(), &entity1);
        assert_eq!(mappings.get(&pattern_id2).unwrap(), &entity2);
    }

    #[tokio::test]
    async fn test_compact_storage() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        // Store patterns, some with entity associations
        let entity_key = create_test_entity_key();
        let pattern_id1 = storage.store_dense_vector(
            entity_key,
            &vec![0.5; 10],
            "used_concept".to_string(),
        ).await.unwrap();
        
        // Store another pattern without entity association
        let sdr = SDR::random(&storage.config);
        let unused_pattern = SDRPattern::new("unused_pattern".to_string(), sdr, "unused_concept".to_string());
        storage.store_pattern(unused_pattern).await.unwrap();
        
        let stats_before = storage.get_statistics().await.unwrap();
        assert_eq!(stats_before.total_patterns, 2);
        
        let removed_count = storage.compact().await.unwrap();
        assert_eq!(removed_count, 1); // Should remove unused pattern
        
        let stats_after = storage.get_statistics().await.unwrap();
        assert_eq!(stats_after.total_patterns, 1);
        
        // Verify the used pattern is still there
        let retrieved = storage.get_entity_pattern(entity_key).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().pattern_id, pattern_id1);
    }

    #[tokio::test]
    async fn test_memory_usage() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let initial_usage = storage.memory_usage().await;
        assert!(initial_usage > 0); // Base overhead
        
        // Add some patterns
        for i in 0..5 {
            let entity_key = create_test_entity_key();
            storage.store_dense_vector(
                entity_key,
                &vec![0.5; 100],
                format!("concept_{}", i),
            ).await.unwrap();
        }
        
        let usage_with_data = storage.memory_usage().await;
        assert!(usage_with_data > initial_usage);
    }

    #[tokio::test]
    async fn test_encode_text_deterministic() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let text = "hello world";
        let sdr1 = storage.encode_text(text).await.unwrap();
        let sdr2 = storage.encode_text(text).await.unwrap();
        
        // Same text should produce same SDR
        assert_eq!(sdr1.active_bits, sdr2.active_bits);
    }

    #[tokio::test]
    async fn test_store_with_metadata() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let sdr = SDR::random(&storage.config);
        let pattern_id = storage.store_with_metadata(
            &sdr,
            "test content".to_string(),
            0.8,
        ).await.unwrap();
        
        assert!(pattern_id.starts_with("pattern_"));
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
    }

    #[tokio::test]
    async fn test_large_scale_storage() {
        let config = SDRConfig {
            total_bits: 1000,
            active_bits: 50,
            sparsity: 0.05,
            overlap_threshold: 0.4,
        };
        let storage = SDRStorage::new(config);
        
        // Store 100 patterns
        for i in 0..100 {
            let entity_key = create_test_entity_key();
            storage.store_dense_vector(
                entity_key,
                &vec![i as f32 / 100.0; 50],
                format!("concept_{}", i),
            ).await.unwrap();
        }
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 100);
        assert_eq!(stats.total_entities, 100);
        assert!(stats.average_sparsity > 0.0);
        assert!(stats.total_active_bits > 0);
        
        // Test similarity search with many patterns
        let query_sdr = SDR::random(&storage.config);
        let results = storage.find_similar_patterns(&query_sdr, 10).await.unwrap();
        assert!(results.len() <= 10);
    }

    // Test SimilarityIndex private methods indirectly through storage operations
    #[tokio::test]
    async fn test_similarity_index_operations() {
        let config = SDRConfig {
            total_bits: 100,
            active_bits: 5,
            sparsity: 0.05,
            overlap_threshold: 0.2,
        };
        let storage = SDRStorage::new(config);
        
        // Test adding patterns to similarity index
        let sdr1 = SDR::new([10, 20, 30, 40, 50].iter().cloned().collect(), 100);
        let pattern1 = SDRPattern::new("p1".to_string(), sdr1, "c1".to_string());
        storage.store_pattern(pattern1).await.unwrap();
        
        let sdr2 = SDR::new([30, 40, 50, 60, 70].iter().cloned().collect(), 100);
        let pattern2 = SDRPattern::new("p2".to_string(), sdr2, "c2".to_string());
        storage.store_pattern(pattern2).await.unwrap();
        
        // Test search functionality
        let query_sdr = SDR::new([10, 20, 30, 35, 45].iter().cloned().collect(), 100);
        let results = storage.find_similar_patterns(&query_sdr, 10).await.unwrap();
        
        assert_eq!(results.len(), 2);
        // Should find p1 as more similar (3 overlapping bits vs 2)
        assert_eq!(results[0].0, "p1");
        assert!(results[0].1 > results[1].1);
        
        // Test pattern removal through compaction
        let removed = storage.compact().await.unwrap();
        assert_eq!(removed, 2); // Both patterns should be removed as they have no entity associations
        
        let empty_results = storage.find_similar_patterns(&query_sdr, 10).await.unwrap();
        assert!(empty_results.is_empty());
    }

    // Test mathematical invariants and edge cases
    #[test]
    fn test_sdr_mathematical_invariants() {
        let config = SDRConfig::default();
        
        // Test sparsity calculation accuracy
        for active_count in [0, 1, 10, 100, 1000] {
            let total_bits = 1000;
            if active_count <= total_bits {
                let active_bits: HashSet<usize> = (0..active_count).collect();
                let sdr = SDR::new(active_bits, total_bits);
                let expected_sparsity = active_count as f32 / total_bits as f32;
                assert!((sdr.sparsity() - expected_sparsity).abs() < f32::EPSILON);
            }
        }
        
        // Test similarity metric bounds
        let sdr1 = SDR::new([1, 2, 3].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([4, 5, 6].iter().cloned().collect(), 100);
        
        assert!(sdr1.overlap(&sdr2) >= 0.0 && sdr1.overlap(&sdr2) <= 1.0);
        assert!(sdr1.jaccard_similarity(&sdr2) >= 0.0 && sdr1.jaccard_similarity(&sdr2) <= 1.0);
        assert!(sdr1.cosine_similarity(&sdr2) >= 0.0 && sdr1.cosine_similarity(&sdr2) <= 1.0);
    }

    #[test]
    fn test_sdr_bit_manipulation_edge_cases() {
        // Test with bits at boundary positions
        let sdr = SDR::new([0, 999].iter().cloned().collect(), 1000);
        assert_eq!(sdr.active_bits.len(), 2);
        assert!(sdr.active_bits.contains(&0));
        assert!(sdr.active_bits.contains(&999));
        
        let dense = sdr.to_dense_vector();
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[999], 1.0);
        assert_eq!(dense[500], 0.0);
    }

    #[tokio::test]
    async fn test_concurrent_storage_operations() {
        use tokio::task::JoinSet;
        
        let config = SDRConfig::default();
        let storage = Arc::new(SDRStorage::new(config));
        
        let mut join_set = JoinSet::new();
        
        // Spawn multiple concurrent storage operations
        for i in 0..10 {
            let storage_clone = storage.clone();
            join_set.spawn(async move {
                let entity_key = create_test_entity_key();
                let vector = vec![i as f32 / 10.0; 100];
                storage_clone.store_dense_vector(
                    entity_key,
                    &vector,
                    format!("concurrent_concept_{}", i),
                ).await.unwrap();
                entity_key
            });
        }
        
        let mut entity_keys = Vec::new();
        while let Some(result) = join_set.join_next().await {
            entity_keys.push(result.unwrap());
        }
        
        // Verify all operations completed successfully
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 10);
        assert_eq!(stats.total_entities, 10);
        
        // Verify all entities can be retrieved
        for entity_key in entity_keys {
            let pattern = storage.get_entity_pattern(entity_key).await.unwrap();
            assert!(pattern.is_some());
        }
    }
}