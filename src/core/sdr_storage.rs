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

    #[test]
    fn test_sdr_creation() {
        let config = SDRConfig::default();
        let sdr = SDR::random(&config);
        
        assert_eq!(sdr.total_bits, config.total_bits);
        assert_eq!(sdr.active_bits.len(), config.active_bits);
        assert!(sdr.sparsity() <= config.sparsity + 0.01); // Allow small tolerance
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
    fn test_sdr_similarity() {
        let config = SDRConfig::default();
        let sdr1 = SDR::new([1, 2, 3, 4, 5].iter().cloned().collect(), 100);
        let sdr2 = SDR::new([3, 4, 5, 6, 7].iter().cloned().collect(), 100);
        
        let overlap = sdr1.overlap(&sdr2);
        let jaccard = sdr1.jaccard_similarity(&sdr2);
        let cosine = sdr1.cosine_similarity(&sdr2);
        
        assert!(overlap > 0.0);
        assert!(jaccard > 0.0);
        assert!(cosine > 0.0);
    }

    #[tokio::test]
    async fn test_sdr_storage() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let sdr = SDR::random(&storage.config);
        let pattern = SDRPattern::new("test_pattern".to_string(), sdr, "test_concept".to_string());
        
        storage.store_pattern(pattern).await.unwrap();
        
        let stats = storage.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
    }

    #[tokio::test]
    async fn test_entity_pattern_association() {
        let config = SDRConfig::default();
        let storage = SDRStorage::new(config);
        
        let entity_key = EntityKey::default();
        let dense_vector = vec![0.5; 100];
        
        let pattern_id = storage.store_dense_vector(
            entity_key,
            &dense_vector,
            "test_concept".to_string(),
        ).await.unwrap();
        
        let retrieved_pattern = storage.get_entity_pattern(entity_key).await.unwrap();
        assert!(retrieved_pattern.is_some());
        assert_eq!(retrieved_pattern.unwrap().pattern_id, pattern_id);
    }
}