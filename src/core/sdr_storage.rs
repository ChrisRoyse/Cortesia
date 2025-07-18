use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use ahash::AHashMap;
use rand::prelude::*;

use crate::core::types::EntityKey;
use crate::error::{Result, GraphError};

/// Sparse Distributed Representation (SDR) configuration
#[derive(Debug, Clone)]
pub struct SDRConfig {
    pub total_bits: usize,
    pub active_bits: usize,
    pub sparsity: f32,
    pub overlap_threshold: f32,
}

impl Default for SDRConfig {
    fn default() -> Self {
        Self {
            total_bits: 2048,
            active_bits: 40,
            sparsity: 0.02, // 2% sparsity
            overlap_threshold: 0.5,
        }
    }
}

/// Sparse Distributed Representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDR {
    pub active_bits: HashSet<usize>,
    pub total_bits: usize,
    pub timestamp: std::time::SystemTime,
}

impl SDR {
    pub fn new(active_bits: HashSet<usize>, total_bits: usize) -> Self {
        Self {
            active_bits,
            total_bits,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Create SDR from dense vector
    pub fn from_dense_vector(vector: &[f32], config: &SDRConfig) -> Self {
        // Find indices of top-k values to create sparse representation
        let mut indexed_values: Vec<(usize, f32)> = vector.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by value (descending)
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top active_bits indices
        let active_bits: HashSet<usize> = indexed_values.iter()
            .take(config.active_bits)
            .map(|(i, _)| *i)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Create random SDR
    pub fn random(config: &SDRConfig) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let mut indices: Vec<usize> = (0..config.total_bits).collect();
        indices.shuffle(&mut thread_rng());
        
        let active_bits: HashSet<usize> = indices.into_iter()
            .take(config.active_bits)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Create random SDR with provided RNG
    pub fn random_with_rng<R: Rng>(config: &SDRConfig, rng: &mut R) -> Self {
        use rand::seq::SliceRandom;
        
        let mut indices: Vec<usize> = (0..config.total_bits).collect();
        indices.shuffle(rng);
        
        let active_bits: HashSet<usize> = indices.into_iter()
            .take(config.active_bits)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Calculate overlap with another SDR
    pub fn overlap(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let union_size = self.active_bits.union(&other.active_bits).count();
        
        if union_size == 0 {
            0.0
        } else {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Calculate Jaccard similarity
    pub fn jaccard_similarity(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let union_size = self.active_bits.union(&other.active_bits).count();
        
        if union_size == 0 {
            0.0
        } else {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Calculate cosine similarity (treating SDR as binary vector)
    pub fn cosine_similarity(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let norm_self = (self.active_bits.len() as f32).sqrt();
        let norm_other = (other.active_bits.len() as f32).sqrt();
        
        if norm_self == 0.0 || norm_other == 0.0 {
            0.0
        } else {
            intersection_size as f32 / (norm_self * norm_other)
        }
    }

    /// Convert to dense vector
    pub fn to_dense_vector(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.total_bits];
        for &bit in &self.active_bits {
            if bit < dense.len() {
                dense[bit] = 1.0;
            }
        }
        dense
    }

    /// Get sparsity (fraction of active bits)
    pub fn sparsity(&self) -> f32 {
        self.active_bits.len() as f32 / self.total_bits as f32
    }

    /// Union with another SDR
    pub fn union(&self, other: &SDR) -> Result<SDR> {
        if self.total_bits != other.total_bits {
            return Err(GraphError::InvalidInput("SDR dimensions must match".to_string()));
        }

        let active_bits = self.active_bits.union(&other.active_bits).cloned().collect();
        Ok(SDR::new(active_bits, self.total_bits))
    }

    /// Intersection with another SDR
    pub fn intersection(&self, other: &SDR) -> Result<SDR> {
        if self.total_bits != other.total_bits {
            return Err(GraphError::InvalidInput("SDR dimensions must match".to_string()));
        }

        let active_bits = self.active_bits.intersection(&other.active_bits).cloned().collect();
        Ok(SDR::new(active_bits, self.total_bits))
    }
}

/// SDR pattern for representing complex concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRPattern {
    pub pattern_id: String,
    pub sdr: SDR,
    pub concept_name: String,
    pub confidence: f32,
    pub creation_time: std::time::SystemTime,
    pub usage_count: u64,
}

impl SDRPattern {
    pub fn new(pattern_id: String, sdr: SDR, concept_name: String) -> Self {
        Self {
            pattern_id,
            sdr,
            concept_name,
            confidence: 1.0,
            creation_time: std::time::SystemTime::now(),
            usage_count: 0,
        }
    }
}

/// SDR storage and retrieval system
pub struct SDRStorage {
    patterns: Arc<RwLock<AHashMap<String, SDRPattern>>>,
    entity_patterns: Arc<RwLock<AHashMap<EntityKey, String>>>, // entity -> pattern_id
    similarity_index: Arc<RwLock<SimilarityIndex>>,
    config: SDRConfig,
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

/// Similarity index for fast SDR pattern matching
struct SimilarityIndex {
    pattern_bits: AHashMap<String, HashSet<usize>>, // pattern_id -> active bits
    bit_to_patterns: AHashMap<usize, HashSet<String>>, // bit -> pattern_ids
}

impl SimilarityIndex {
    fn new() -> Self {
        Self {
            pattern_bits: AHashMap::new(),
            bit_to_patterns: AHashMap::new(),
        }
    }

    fn add_pattern(&mut self, pattern: &SDRPattern) -> Result<()> {
        let pattern_id = pattern.pattern_id.clone();
        let active_bits = pattern.sdr.active_bits.clone();

        // Store pattern bits
        self.pattern_bits.insert(pattern_id.clone(), active_bits.clone());

        // Update inverted index
        for bit in active_bits {
            self.bit_to_patterns
                .entry(bit)
                .or_insert_with(HashSet::new)
                .insert(pattern_id.clone());
        }

        Ok(())
    }

    fn remove_patterns(&mut self, pattern_ids: &HashSet<String>) -> Result<()> {
        for pattern_id in pattern_ids {
            if let Some(active_bits) = self.pattern_bits.remove(pattern_id) {
                // Remove from inverted index
                for bit in active_bits {
                    if let Some(patterns) = self.bit_to_patterns.get_mut(&bit) {
                        patterns.remove(pattern_id);
                        if patterns.is_empty() {
                            self.bit_to_patterns.remove(&bit);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn find_similar(
        &self,
        query_sdr: &SDR,
        max_results: usize,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let mut candidate_scores = AHashMap::new();

        // Find candidate patterns using inverted index
        for &bit in &query_sdr.active_bits {
            if let Some(patterns) = self.bit_to_patterns.get(&bit) {
                for pattern_id in patterns {
                    *candidate_scores.entry(pattern_id.clone()).or_insert(0) += 1;
                }
            }
        }

        // Calculate actual similarities for candidates
        let mut similarities = Vec::new();
        
        for (pattern_id, _overlap_count) in candidate_scores {
            if let Some(pattern_bits) = self.pattern_bits.get(&pattern_id) {
                let candidate_sdr = SDR::new(pattern_bits.clone(), query_sdr.total_bits);
                let similarity = query_sdr.jaccard_similarity(&candidate_sdr);
                
                if similarity >= threshold {
                    similarities.push((pattern_id, similarity));
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(max_results);

        Ok(similarities)
    }
}

/// Statistics about SDR storage
#[derive(Debug, Clone)]
pub struct SDRStatistics {
    pub total_patterns: usize,
    pub total_entities: usize,
    pub average_sparsity: f32,
    pub total_active_bits: usize,
    pub config: SDRConfig,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilaritySearchResult {
    pub pattern_id: String,
    pub content: String,
    pub similarity: f32,
}

/// SDR Entry for storing entities with SDR representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDREntry {
    pub id: String,
    pub embedding: Vec<f32>,
    pub properties: HashMap<String, String>,
    pub activation: f32,
}

/// SDR Query for searching SDR patterns
#[derive(Debug, Clone)]
pub struct SDRQuery {
    pub query_sdr: SDR,
    pub top_k: usize,
    pub min_overlap: f32,
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