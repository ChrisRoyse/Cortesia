# Task 40: Semantic Similarity Scoring

## Metadata
- **Micro-Phase**: 2.40
- **Duration**: 15-20 minutes
- **Dependencies**: Task 39 (scoring_framework_design)
- **Output**: `src/allocation_scoring/semantic_similarity_scoring.rs`

## Description
Implement advanced semantic similarity scoring using embeddings, cosine similarity, and semantic distance calculations with intelligent caching and batch processing optimization. This component calculates semantic compatibility between concepts for allocation decisions with <0.5ms per comparison.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_scoring::{AllocationContext, ScoringStrategy};
    use crate::hierarchy_detection::ExtractedConcept;
    use std::collections::HashMap;

    #[test]
    fn test_semantic_similarity_strategy_creation() {
        let strategy = SemanticSimilarityStrategy::new();
        assert_eq!(strategy.name(), "semantic_similarity");
        assert!(strategy.supports_parallel());
        assert!(strategy.is_enabled());
        assert_eq!(strategy.embedding_dimension(), 384); // Default BERT embedding size
    }
    
    #[test]
    fn test_basic_semantic_similarity_calculation() {
        let strategy = SemanticSimilarityStrategy::new();
        
        let golden_retriever = create_test_concept("golden retriever", 0.85);
        let dog_context = create_allocation_context("dog", &["mammal", "animal"]);
        
        let score = strategy.score(&golden_retriever, &dog_context).unwrap();
        
        // Golden retriever should have high semantic similarity to dog
        assert!(score > 0.8);
        assert!(score <= 1.0);
    }
    
    #[test]
    fn test_semantic_distance_calculation() {
        let strategy = SemanticSimilarityStrategy::new();
        
        // Test semantically similar concepts
        let cat = create_test_concept("cat", 0.9);
        let dog = create_test_concept("dog", 0.9);
        let vehicle = create_test_concept("vehicle", 0.9);
        
        let mammal_context = create_allocation_context("mammal", &["animal"]);
        
        let cat_score = strategy.score(&cat, &mammal_context).unwrap();
        let dog_score = strategy.score(&dog, &mammal_context).unwrap();
        let vehicle_score = strategy.score(&vehicle, &mammal_context).unwrap();
        
        // Cat and dog should score higher than vehicle for mammal context
        assert!(cat_score > vehicle_score);
        assert!(dog_score > vehicle_score);
        
        // Cat and dog should have similar scores (both are mammals)
        assert!((cat_score - dog_score).abs() < 0.2);
    }
    
    #[test]
    fn test_semantic_embedding_generation() {
        let strategy = SemanticSimilarityStrategy::new();
        
        let concept = create_test_concept("golden retriever", 0.85);
        let embedding = strategy.generate_embedding(&concept.name).unwrap();
        
        assert_eq!(embedding.len(), 384); // Standard BERT embedding dimension
        
        // Embedding values should be normalized (roughly between -1 and 1)
        for value in &embedding {
            assert!(value.abs() <= 2.0); // Allow some tolerance
        }
        
        // Should not be all zeros
        let sum: f32 = embedding.iter().sum();
        assert!(sum.abs() > 0.1);
    }
    
    #[test]
    fn test_cosine_similarity_calculation() {
        let strategy = SemanticSimilarityStrategy::new();
        
        // Test with known vectors
        let vector1 = vec![1.0, 0.0, 0.0];
        let vector2 = vec![1.0, 0.0, 0.0]; // Identical
        let vector3 = vec![0.0, 1.0, 0.0]; // Orthogonal
        let vector4 = vec![-1.0, 0.0, 0.0]; // Opposite
        
        // Identical vectors should have similarity 1.0
        let similarity_identical = strategy.cosine_similarity(&vector1, &vector2);
        assert!((similarity_identical - 1.0).abs() < 0.001);
        
        // Orthogonal vectors should have similarity 0.0
        let similarity_orthogonal = strategy.cosine_similarity(&vector1, &vector3);
        assert!(similarity_orthogonal.abs() < 0.001);
        
        // Opposite vectors should have similarity -1.0
        let similarity_opposite = strategy.cosine_similarity(&vector1, &vector4);
        assert!((similarity_opposite + 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_semantic_caching_performance() {
        let mut strategy = SemanticSimilarityStrategy::new();
        strategy.enable_caching(true);
        
        let concept = create_test_concept("golden retriever", 0.85);
        let context = create_allocation_context("dog", &["mammal", "animal"]);
        
        // First calculation (cache miss)
        let start1 = std::time::Instant::now();
        let score1 = strategy.score(&concept, &context).unwrap();
        let elapsed1 = start1.elapsed();
        
        // Second calculation (cache hit)
        let start2 = std::time::Instant::now();
        let score2 = strategy.score(&concept, &context).unwrap();
        let elapsed2 = start2.elapsed();
        
        // Results should be identical
        assert_eq!(score1, score2);
        
        // Second call should be significantly faster
        assert!(elapsed2 < elapsed1);
        assert!(elapsed2 < std::time::Duration::from_micros(100)); // Cache hit should be very fast
        
        // Check cache statistics
        let cache_stats = strategy.get_cache_statistics();
        assert_eq!(cache_stats.hits, 1);
        assert_eq!(cache_stats.misses, 1);
    }
    
    #[test]
    fn test_batch_similarity_calculation() {
        let strategy = SemanticSimilarityStrategy::new();
        
        let concepts = vec![
            create_test_concept("golden retriever", 0.85),
            create_test_concept("labrador", 0.82),
            create_test_concept("german shepherd", 0.88),
            create_test_concept("cat", 0.80),
            create_test_concept("car", 0.75),
        ];
        
        let context = create_allocation_context("dog", &["mammal", "animal"]);
        
        let start = std::time::Instant::now();
        let scores = strategy.batch_score(&concepts, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete batch scoring in under 10ms
        assert!(elapsed < std::time::Duration::from_millis(10));
        assert_eq!(scores.len(), 5);
        
        // Dog breeds should score higher than cat and car
        assert!(scores[0] > scores[3]); // golden retriever > cat
        assert!(scores[1] > scores[4]); // labrador > car
        assert!(scores[2] > scores[3]); // german shepherd > cat
        
        // All scores should be valid
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }
    
    #[test]
    fn test_contextual_similarity_weighting() {
        let strategy = SemanticSimilarityStrategy::new();
        
        let retriever = create_test_concept("golden retriever", 0.85);
        
        // Different contexts should produce different similarity scores
        let dog_context = create_allocation_context("dog", &["mammal", "animal"]);
        let pet_context = create_allocation_context("pet", &["companion", "animal"]);
        let working_animal_context = create_allocation_context("working animal", &["trained", "animal"]);
        
        let dog_score = strategy.score(&retriever, &dog_context).unwrap();
        let pet_score = strategy.score(&retriever, &pet_context).unwrap();
        let working_score = strategy.score(&retriever, &working_animal_context).unwrap();
        
        // All should be high but with different emphases
        assert!(dog_score > 0.7);
        assert!(pet_score > 0.7);
        assert!(working_score > 0.6);
        
        // Dog context should be highest for "golden retriever"
        assert!(dog_score >= pet_score);
    }
    
    #[test]
    fn test_semantic_feature_extraction() {
        let strategy = SemanticSimilarityStrategy::new();
        
        let concept = create_test_concept("golden retriever", 0.85);
        let features = strategy.extract_semantic_features(&concept).unwrap();
        
        assert!(!features.is_empty());
        assert!(features.len() >= 3); // Should extract multiple semantic features
        
        // Should extract relevant features for golden retriever
        let feature_text = features.join(" ").to_lowercase();
        assert!(feature_text.contains("dog") || 
                feature_text.contains("breed") || 
                feature_text.contains("canine") ||
                feature_text.contains("pet"));
    }
    
    #[test]
    fn test_similarity_performance_optimization() {
        let strategy = SemanticSimilarityStrategy::new();
        
        // Test with 100 concepts for performance
        let concepts: Vec<_> = (0..100)
            .map(|i| create_test_concept(&format!("concept_{}", i), 0.8))
            .collect();
        
        let context = create_allocation_context("test_parent", &["root"]);
        
        let start = std::time::Instant::now();
        let scores = strategy.batch_score(&concepts, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should maintain <0.5ms average per concept
        let avg_time_per_concept = elapsed.as_nanos() as f64 / 100.0 / 1_000_000.0; // Convert to ms
        assert!(avg_time_per_concept < 0.5);
        
        assert_eq!(scores.len(), 100);
        
        // All scores should be valid
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }
    
    #[test]
    fn test_semantic_similarity_edge_cases() {
        let strategy = SemanticSimilarityStrategy::new();
        
        // Test with empty concept name
        let empty_concept = create_test_concept("", 0.5);
        let context = create_allocation_context("test", &[]);
        
        let score = strategy.score(&empty_concept, &context);
        assert!(score.is_ok());
        assert_eq!(score.unwrap(), 0.0); // Should return 0 for empty concept
        
        // Test with very long concept name
        let long_name = "a".repeat(1000);
        let long_concept = create_test_concept(&long_name, 0.5);
        
        let score = strategy.score(&long_concept, &context);
        assert!(score.is_ok());
        
        // Test with special characters
        let special_concept = create_test_concept("test@#$%^&*()", 0.5);
        let score = strategy.score(&special_concept, &context);
        assert!(score.is_ok());
    }
    
    #[test]
    fn test_embedding_cache_memory_management() {
        let mut strategy = SemanticSimilarityStrategy::new();
        strategy.enable_caching(true);
        strategy.set_max_cache_size(10); // Small cache for testing
        
        // Fill cache beyond capacity
        for i in 0..15 {
            let concept = create_test_concept(&format!("concept_{}", i), 0.8);
            let context = create_allocation_context("test", &[]);
            let _ = strategy.score(&concept, &context);
        }
        
        let cache_stats = strategy.get_cache_statistics();
        
        // Cache should respect max size
        assert!(cache_stats.current_size <= 10);
        assert!(cache_stats.evictions > 0);
    }
    
    fn create_test_concept(name: &str, confidence: f32) -> ExtractedConcept {
        use crate::hierarchy_detection::{ExtractedConcept, ConceptType, TextSpan};
        use std::collections::HashMap;
        
        ExtractedConcept {
            name: name.to_string(),
            concept_type: ConceptType::Entity,
            properties: HashMap::new(),
            source_span: TextSpan {
                start: 0,
                end: name.len(),
                text: name.to_string(),
            },
            confidence,
            suggested_parent: None,
            semantic_features: vec![0.5; 100],
            extracted_at: 0,
        }
    }
    
    fn create_allocation_context(target: &str, ancestors: &[&str]) -> AllocationContext {
        use crate::allocation_scoring::AllocationContext;
        
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties: HashMap::new(),
            allocation_timestamp: 0,
        }
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use rayon::prelude::*;
use crate::allocation_scoring::{AllocationContext, ScoringStrategy, ScoringError};
use crate::hierarchy_detection::ExtractedConcept;

/// Configuration for semantic similarity scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarityConfig {
    /// Embedding dimension (default: 384 for BERT)
    pub embedding_dimension: usize,
    
    /// Similarity threshold for meaningful matches
    pub similarity_threshold: f32,
    
    /// Whether to use contextual weighting
    pub use_contextual_weighting: bool,
    
    /// Maximum cache size for embeddings
    pub max_cache_size: usize,
    
    /// Whether caching is enabled
    pub caching_enabled: bool,
    
    /// Batch processing size for optimization
    pub batch_size: usize,
}

impl Default for SemanticSimilarityConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 384,
            similarity_threshold: 0.1,
            use_contextual_weighting: true,
            max_cache_size: 10000,
            caching_enabled: true,
            batch_size: 32,
        }
    }
}

/// Cached embedding with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbedding {
    /// The embedding vector
    pub embedding: Vec<f32>,
    
    /// Timestamp when created
    pub created_at: u64,
    
    /// Access count for LRU eviction
    pub access_count: u32,
    
    /// Last accessed timestamp
    pub last_accessed: u64,
}

/// Statistics for embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheStatistics {
    /// Total cache hits
    pub hits: u64,
    
    /// Total cache misses
    pub misses: u64,
    
    /// Current cache size
    pub current_size: usize,
    
    /// Maximum cache size
    pub max_size: usize,
    
    /// Number of cache evictions
    pub evictions: u64,
    
    /// Cache hit rate (0.0-1.0)
    pub hit_rate: f32,
    
    /// Average embedding generation time (nanoseconds)
    pub avg_generation_time: u64,
}

/// Semantic features extracted from concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatures {
    /// Key semantic terms
    pub key_terms: Vec<String>,
    
    /// Semantic categories
    pub categories: Vec<String>,
    
    /// Related concepts
    pub related_concepts: Vec<String>,
    
    /// Confidence in feature extraction
    pub extraction_confidence: f32,
}

/// Advanced semantic similarity scoring strategy
pub struct SemanticSimilarityStrategy {
    /// Strategy name
    name: String,
    
    /// Configuration
    config: SemanticSimilarityConfig,
    
    /// Whether strategy is enabled
    enabled: bool,
    
    /// Embedding cache for performance
    embedding_cache: Arc<DashMap<String, CachedEmbedding>>,
    
    /// Cache statistics
    cache_stats: Arc<Mutex<EmbeddingCacheStatistics>>,
    
    /// Pre-computed concept embeddings for common terms
    common_embeddings: HashMap<String, Vec<f32>>,
    
    /// Semantic feature extractor
    feature_extractor: SemanticFeatureExtractor,
}

impl SemanticSimilarityStrategy {
    /// Create a new semantic similarity strategy
    pub fn new() -> Self {
        let mut strategy = Self {
            name: "semantic_similarity".to_string(),
            config: SemanticSimilarityConfig::default(),
            enabled: true,
            embedding_cache: Arc::new(DashMap::new()),
            cache_stats: Arc::new(Mutex::new(EmbeddingCacheStatistics {
                hits: 0,
                misses: 0,
                current_size: 0,
                max_size: 10000,
                evictions: 0,
                hit_rate: 0.0,
                avg_generation_time: 0,
            })),
            common_embeddings: HashMap::new(),
            feature_extractor: SemanticFeatureExtractor::new(),
        };
        
        strategy.initialize_common_embeddings();
        strategy
    }
    
    /// Create with custom configuration
    pub fn with_config(config: SemanticSimilarityConfig) -> Self {
        let mut strategy = Self::new();
        strategy.config = config;
        strategy
    }
    
    /// Initialize pre-computed embeddings for common concepts
    fn initialize_common_embeddings(&mut self) {
        // Pre-compute embeddings for frequently used concepts
        let common_concepts = vec![
            "animal", "mammal", "bird", "fish", "reptile",
            "dog", "cat", "horse", "cow", "pig",
            "plant", "tree", "flower", "fruit", "vegetable",
            "vehicle", "car", "truck", "bike", "boat",
            "person", "human", "child", "adult", "family",
            "building", "house", "school", "hospital", "store",
            "food", "meal", "drink", "snack", "dessert",
            "tool", "machine", "device", "instrument", "equipment",
        ];
        
        for concept in common_concepts {
            if let Ok(embedding) = self.generate_embedding_internal(concept) {
                self.common_embeddings.insert(concept.to_string(), embedding);
            }
        }
    }
    
    /// Generate embedding for a concept name
    pub fn generate_embedding(&self, concept_name: &str) -> Result<Vec<f32>, ScoringError> {
        if concept_name.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dimension]);
        }
        
        // Check cache first
        if self.config.caching_enabled {
            if let Some(cached) = self.embedding_cache.get(concept_name) {
                self.update_cache_access(&cached);
                self.update_cache_stats(true, 0);
                return Ok(cached.embedding.clone());
            }
        }
        
        // Check pre-computed common embeddings
        if let Some(embedding) = self.common_embeddings.get(concept_name) {
            return Ok(embedding.clone());
        }
        
        // Generate new embedding
        let generation_start = std::time::Instant::now();
        let embedding = self.generate_embedding_internal(concept_name)?;
        let generation_time = generation_start.elapsed().as_nanos() as u64;
        
        // Cache the result
        if self.config.caching_enabled {
            self.cache_embedding(concept_name.to_string(), embedding.clone(), generation_time);
        }
        
        self.update_cache_stats(false, generation_time);
        Ok(embedding)
    }
    
    /// Internal embedding generation (simplified mock implementation)
    fn generate_embedding_internal(&self, concept_name: &str) -> Result<Vec<f32>, ScoringError> {
        // This is a simplified mock implementation
        // In a real system, this would use a pre-trained language model like BERT
        
        let mut embedding = vec![0.0; self.config.embedding_dimension];
        
        // Simple hash-based embedding generation for testing
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        concept_name.hash(&mut hasher);
        let hash_value = hasher.finish();
        
        // Fill embedding with pseudo-random values based on concept name
        for i in 0..self.config.embedding_dimension {
            let seed = hash_value.wrapping_add(i as u64);
            let value = ((seed as f32) / (u64::MAX as f32) - 0.5) * 2.0; // Scale to [-1, 1]
            embedding[i] = value;
        }
        
        // Normalize the embedding
        self.normalize_embedding(&mut embedding);
        
        Ok(embedding)
    }
    
    /// Normalize embedding vector to unit length
    fn normalize_embedding(&self, embedding: &mut Vec<f32>) {
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in embedding.iter_mut() {
                *value /= magnitude;
            }
        }
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = embedding1.iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let magnitude1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude1 > 0.0 && magnitude2 > 0.0 {
            (dot_product / (magnitude1 * magnitude2)).max(-1.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate semantic similarity with contextual weighting
    fn calculate_contextual_similarity(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        let concept_embedding = self.generate_embedding(&concept.name)?;
        
        // Calculate similarity to target concept
        let target_embedding = self.generate_embedding(&context.target_concept)?;
        let target_similarity = self.cosine_similarity(&concept_embedding, &target_embedding);
        
        if !self.config.use_contextual_weighting || context.ancestor_concepts.is_empty() {
            return Ok((target_similarity + 1.0) / 2.0); // Scale from [-1,1] to [0,1]
        }
        
        // Calculate weighted similarity including ancestors
        let mut weighted_similarity = target_similarity * 0.6; // Primary weight for target
        
        // Add ancestor similarities with decreasing weights
        let ancestor_weight = 0.4 / context.ancestor_concepts.len() as f32;
        for ancestor in &context.ancestor_concepts {
            let ancestor_embedding = self.generate_embedding(ancestor)?;
            let ancestor_similarity = self.cosine_similarity(&concept_embedding, &ancestor_embedding);
            weighted_similarity += ancestor_similarity * ancestor_weight;
        }
        
        // Scale to [0,1] range
        Ok((weighted_similarity + 1.0) / 2.0)
    }
    
    /// Extract semantic features from concept
    pub fn extract_semantic_features(&self, concept: &ExtractedConcept) -> Result<SemanticFeatures, ScoringError> {
        self.feature_extractor.extract_features(concept)
    }
    
    /// Batch score multiple concepts for efficiency
    pub fn batch_score(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<f32>, ScoringError> {
        // Process in parallel batches for optimal performance
        let scores: Result<Vec<_>, _> = concepts.par_chunks(self.config.batch_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|concept| {
                    self.calculate_contextual_similarity(concept, context)
                })
            })
            .collect();
        
        scores
    }
    
    /// Cache an embedding
    fn cache_embedding(&self, concept_name: String, embedding: Vec<f32>, generation_time: u64) {
        // Check if cache is full and evict if necessary
        if self.embedding_cache.len() >= self.config.max_cache_size {
            self.evict_least_recently_used();
        }
        
        let cached_embedding = CachedEmbedding {
            embedding,
            created_at: current_timestamp(),
            access_count: 1,
            last_accessed: current_timestamp(),
        };
        
        self.embedding_cache.insert(concept_name, cached_embedding);
        
        if let Ok(mut stats) = self.cache_stats.lock() {
            stats.current_size = self.embedding_cache.len();
            
            // Update average generation time
            let total_time = stats.avg_generation_time * stats.misses + generation_time;
            stats.avg_generation_time = total_time / (stats.misses + 1);
        }
    }
    
    /// Update cache access information
    fn update_cache_access(&self, cached: &CachedEmbedding) {
        // This would update access count and timestamp in a real implementation
        // For simplicity, we'll skip the complex update logic here
    }
    
    /// Evict least recently used cache entry
    fn evict_least_recently_used(&self) {
        let mut oldest_key = None;
        let mut oldest_time = u64::MAX;
        
        for entry in self.embedding_cache.iter() {
            if entry.value().last_accessed < oldest_time {
                oldest_time = entry.value().last_accessed;
                oldest_key = Some(entry.key().clone());
            }
        }
        
        if let Some(key) = oldest_key {
            self.embedding_cache.remove(&key);
            
            if let Ok(mut stats) = self.cache_stats.lock() {
                stats.evictions += 1;
                stats.current_size = self.embedding_cache.len();
            }
        }
    }
    
    /// Update cache statistics
    fn update_cache_stats(&self, cache_hit: bool, generation_time: u64) {
        if let Ok(mut stats) = self.cache_stats.lock() {
            if cache_hit {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
            
            let total_requests = stats.hits + stats.misses;
            if total_requests > 0 {
                stats.hit_rate = stats.hits as f32 / total_requests as f32;
            }
        }
    }
    
    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> EmbeddingCacheStatistics {
        if let Ok(stats) = self.cache_stats.lock() {
            stats.clone()
        } else {
            EmbeddingCacheStatistics {
                hits: 0,
                misses: 0,
                current_size: 0,
                max_size: self.config.max_cache_size,
                evictions: 0,
                hit_rate: 0.0,
                avg_generation_time: 0,
            }
        }
    }
    
    /// Enable or disable caching
    pub fn enable_caching(&mut self, enabled: bool) {
        self.config.caching_enabled = enabled;
        if !enabled {
            self.embedding_cache.clear();
        }
    }
    
    /// Set maximum cache size
    pub fn set_max_cache_size(&mut self, max_size: usize) {
        self.config.max_cache_size = max_size;
        
        // Evict entries if current size exceeds new limit
        while self.embedding_cache.len() > max_size {
            self.evict_least_recently_used();
        }
    }
    
    /// Get embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }
    
    /// Check if strategy is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Clear embedding cache
    pub fn clear_cache(&self) {
        self.embedding_cache.clear();
        if let Ok(mut stats) = self.cache_stats.lock() {
            stats.current_size = 0;
        }
    }
}

impl ScoringStrategy for SemanticSimilarityStrategy {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn score(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        if !self.enabled {
            return Ok(0.0);
        }
        
        self.calculate_contextual_similarity(concept, context)
    }
    
    fn supports_parallel(&self) -> bool {
        true
    }
    
    fn weight_preference(&self) -> f32 {
        0.35 // Semantic similarity is quite important
    }
}

/// Semantic feature extractor
pub struct SemanticFeatureExtractor {
    /// Common semantic patterns
    patterns: HashMap<String, Vec<String>>,
}

impl SemanticFeatureExtractor {
    pub fn new() -> Self {
        let mut extractor = Self {
            patterns: HashMap::new(),
        };
        
        extractor.initialize_patterns();
        extractor
    }
    
    /// Initialize semantic patterns for feature extraction
    fn initialize_patterns(&mut self) {
        // Animal patterns
        self.patterns.insert("animal".to_string(), vec![
            "dog".to_string(), "cat".to_string(), "bird".to_string(), 
            "mammal".to_string(), "canine".to_string(), "feline".to_string(),
            "pet".to_string(), "wild".to_string(), "domesticated".to_string(),
        ]);
        
        // Vehicle patterns
        self.patterns.insert("vehicle".to_string(), vec![
            "car".to_string(), "truck".to_string(), "bike".to_string(),
            "automobile".to_string(), "transport".to_string(), "motor".to_string(),
        ]);
        
        // Plant patterns
        self.patterns.insert("plant".to_string(), vec![
            "tree".to_string(), "flower".to_string(), "grass".to_string(),
            "botanical".to_string(), "flora".to_string(), "vegetation".to_string(),
        ]);
    }
    
    /// Extract semantic features from concept
    pub fn extract_features(&self, concept: &ExtractedConcept) -> Result<SemanticFeatures, ScoringError> {
        let concept_name_lower = concept.name.to_lowercase();
        let mut key_terms = Vec::new();
        let mut categories = Vec::new();
        let mut related_concepts = Vec::new();
        
        // Extract key terms from concept name
        let words: Vec<&str> = concept_name_lower.split_whitespace().collect();
        key_terms.extend(words.iter().map(|s| s.to_string()));
        
        // Match against semantic patterns
        for (category, patterns) in &self.patterns {
            for pattern in patterns {
                if concept_name_lower.contains(pattern) {
                    categories.push(category.clone());
                    related_concepts.push(pattern.clone());
                    break;
                }
            }
        }
        
        // Calculate extraction confidence based on matches
        let extraction_confidence = if categories.is_empty() {
            0.3 // Low confidence if no category matches
        } else {
            0.8 + (categories.len() as f32 * 0.1).min(0.2)
        };
        
        Ok(SemanticFeatures {
            key_terms,
            categories,
            related_concepts,
            extraction_confidence,
        })
    }
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for SemanticSimilarityStrategy {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create SemanticSimilarityStrategy with embedding generation and caching
2. Implement cosine similarity calculation with proper normalization
3. Add contextual weighting for ancestor concepts in hierarchy
4. Implement intelligent caching with LRU eviction for performance
5. Add batch processing capabilities for efficient parallel scoring
6. Ensure semantic feature extraction works for concept analysis

## Success Criteria
- [ ] SemanticSimilarityStrategy compiles without errors
- [ ] Cosine similarity calculation produces correct results for test vectors
- [ ] Embedding generation and caching achieves <0.5ms per comparison
- [ ] Contextual weighting improves allocation accuracy
- [ ] Batch processing handles 100+ concepts efficiently
- [ ] All tests pass with comprehensive coverage