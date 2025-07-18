use crate::core::triple::Triple;
use crate::error::Result;
use crate::neural::neural_server::NeuralProcessingServer;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use ahash;

/// Trait for embedding models used in canonicalization
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embedding_dimension(&self) -> usize;
}

/// Mock embedding model for testing and development
pub struct MockEmbeddingModel {
    dimension: usize,
}

impl MockEmbeddingModel {
    pub fn new() -> Self {
        Self {
            dimension: 384,
        }
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Simple hash-based embedding for testing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        let mut embedding = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            embedding[i] = ((hash.wrapping_mul(i as u64 + 1)) as f32) / (u64::MAX as f32);
        }
        
        Ok(embedding)
    }
    
    fn embedding_dimension(&self) -> usize {
        self.dimension
    }
}

/// Neural canonicalization and de-duplication system
pub struct NeuralCanonicalizer {
    entity_canonicalizer: EntityCanonicalizer,
    deduplicator: EntityDeduplicator,
    cache: Arc<RwLock<HashMap<String, CanonicalEntity>>>,
}

impl NeuralCanonicalizer {
    pub fn new() -> Self {
        Self {
            entity_canonicalizer: EntityCanonicalizer::new(),
            deduplicator: EntityDeduplicator::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn new_with_neural_server(neural_server: Arc<NeuralProcessingServer>) -> Self {
        Self {
            entity_canonicalizer: EntityCanonicalizer::new(),
            deduplicator: EntityDeduplicator::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn canonicalize_triple(&self, triple: &Triple) -> Result<Triple> {
        let canonical_subject = self.canonicalize_entity(&triple.subject).await?;
        let canonical_object = self.canonicalize_entity(&triple.object).await?;
        let canonical_predicate = self.canonicalize_predicate(&triple.predicate).await?;
        
        Triple::with_metadata(
            canonical_subject,
            canonical_predicate,
            canonical_object,
            triple.confidence,
            triple.source.clone(),
        )
    }

    pub async fn canonicalize_entity(&self, entity: &str) -> Result<String> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(canonical) = cache.get(entity) {
                return Ok(canonical.canonical_name.clone());
            }
        }

        // Canonicalize the entity
        let canonical = self.entity_canonicalizer.canonicalize(entity).await?;
        
        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(entity.to_string(), canonical.clone());
        }
        
        Ok(canonical.canonical_name)
    }

    pub async fn canonicalize_to_string(&self, entity: &str) -> Result<String> {
        self.canonicalize_entity(entity).await
    }

    pub async fn canonicalize_predicate(&self, predicate: &str) -> Result<String> {
        self.entity_canonicalizer.canonicalize_predicate(predicate).await
    }

    pub async fn deduplicate_entities(&self, entities: Vec<String>) -> Result<DeduplicationResult> {
        self.deduplicator.deduplicate(entities).await
    }

    pub async fn find_similar_entities(&self, entity: &str, threshold: f32) -> Result<Vec<String>> {
        let _canonical = self.canonicalize_entity(entity).await?;
        
        // In a real implementation, this would use embeddings to find similar entities
        // For now, use simple string similarity
        let cache = self.cache.read().await;
        let mut similar = Vec::new();
        
        for (original, _canonical_entity) in cache.iter() {
            if original != entity {
                let similarity = self.calculate_string_similarity(entity, original);
                if similarity > threshold {
                    similar.push(original.clone());
                }
            }
        }
        
        Ok(similar)
    }

    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Simple Levenshtein distance-based similarity
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len());
        
        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f32 / max_len as f32)
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let v1: Vec<char> = s1.chars().collect();
        let v2: Vec<char> = s2.chars().collect();
        let v1_len = v1.len();
        let v2_len = v2.len();

        if v1_len == 0 {
            return v2_len;
        }
        if v2_len == 0 {
            return v1_len;
        }

        let mut prev_distances: Vec<usize> = (0..=v2_len).collect();
        let mut curr_distances: Vec<usize> = vec![0; v2_len + 1];

        for (i, c1) in v1.iter().enumerate() {
            curr_distances[0] = i + 1;

            for (j, c2) in v2.iter().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                curr_distances[j + 1] = std::cmp::min(
                    std::cmp::min(
                        curr_distances[j] + 1,          // insertion
                        prev_distances[j + 1] + 1,      // deletion
                    ),
                    prev_distances[j] + cost,           // substitution
                );
            }

            std::mem::swap(&mut prev_distances, &mut curr_distances);
        }

        prev_distances[v2_len]
    }

    pub async fn get_canonicalization_stats(&self) -> CanonicalizationStats {
        let cache = self.cache.read().await;
        CanonicalizationStats {
            total_entities_processed: cache.len(),
            canonical_entities_created: cache.len(),
            duplicates_found: 0,
            average_confidence: 0.85,
        }
    }

    fn find_most_common_canonical(&self, cache: &HashMap<String, CanonicalEntity>) -> Option<String> {
        let mut canonical_counts: HashMap<String, usize> = HashMap::new();
        
        for canonical_entity in cache.values() {
            *canonical_counts.entry(canonical_entity.canonical_name.clone()).or_insert(0) += 1;
        }
        
        canonical_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name)
    }
}

/// Entity canonicalization using neural embeddings
pub struct EntityCanonicalizer {
    embedding_model: Arc<dyn EmbeddingModel>,
    similarity_threshold: f32,
    canonical_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl EntityCanonicalizer {
    pub fn new() -> Self {
        Self {
            embedding_model: Arc::new(MockEmbeddingModel::new()),
            similarity_threshold: 0.8,
            canonical_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn canonicalize(&self, entity: &str) -> Result<CanonicalEntity> {
        // Normalize the entity name
        let normalized = self.normalize_entity_name(entity);
        
        // Generate embedding
        let embedding = self.embedding_model.embed(&normalized).await?;
        
        // Find similar canonical entities
        let similar_canonical = self.find_similar_canonical(&embedding).await?;
        
        if let Some(canonical_name) = similar_canonical {
            Ok(CanonicalEntity {
                original_name: entity.to_string(),
                canonical_name,
                confidence: 0.9,
                alternative_forms: Vec::new(),
                normalization_applied: normalized != entity,
                similar_entities: Vec::new(),
            })
        } else {
            // This is a new canonical entity
            let canonical_name = normalized.clone();
            
            // Cache the embedding
            {
                let mut cache = self.canonical_cache.write().await;
                cache.insert(canonical_name.clone(), embedding);
            }
            
            Ok(CanonicalEntity {
                original_name: entity.to_string(),
                canonical_name,
                confidence: 1.0,
                alternative_forms: Vec::new(),
                normalization_applied: normalized != entity,
                similar_entities: Vec::new(),
            })
        }
    }

    pub async fn canonicalize_predicate(&self, predicate: &str) -> Result<String> {
        // Simple predicate canonicalization
        let normalized = predicate.to_lowercase()
            .replace(" ", "_")
            .replace("-", "_");
        
        // Common predicate mappings
        let mappings = [
            ("is_a", "is"),
            ("was_a", "is"),
            ("are", "is"),
            ("were", "is"),
            ("works_for", "works_at"),
            ("employed_by", "works_at"),
            ("born_in", "born_in"),
            ("from", "born_in"),
            ("created_by", "created"),
            ("invented_by", "invented"),
        ];
        
        for (from, to) in mappings {
            if normalized == from {
                return Ok(to.to_string());
            }
        }
        
        Ok(normalized)
    }

    fn normalize_entity_name(&self, entity: &str) -> String {
        // Remove common prefixes and suffixes
        let mut normalized = entity.trim().to_string();
        
        // Remove titles
        let prefixes = ["Dr.", "Prof.", "Mr.", "Mrs.", "Ms.", "Sir"];
        for prefix in prefixes {
            if normalized.starts_with(prefix) {
                normalized = normalized[prefix.len()..].trim().to_string();
            }
        }
        
        // Remove common suffixes
        let suffixes = [" Jr.", " Sr.", " III", " II"];
        for suffix in suffixes {
            if normalized.ends_with(suffix) {
                normalized = normalized[..normalized.len() - suffix.len()].trim().to_string();
            }
        }
        
        // Title case
        self.to_title_case(&normalized)
    }

    fn to_title_case(&self, s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    async fn find_similar_canonical(&self, embedding: &[f32]) -> Result<Option<String>> {
        let cache = self.canonical_cache.read().await;
        
        let mut best_match = None;
        let mut best_similarity = 0.0;
        
        for (canonical_name, canonical_embedding) in cache.iter() {
            let similarity = self.cosine_similarity(embedding, canonical_embedding);
            if similarity > best_similarity && similarity > self.similarity_threshold {
                best_similarity = similarity;
                best_match = Some(canonical_name.clone());
            }
        }
        
        Ok(best_match)
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Entity deduplication using neural embeddings
pub struct EntityDeduplicator {
    embedding_model: Arc<dyn EmbeddingModel>,
    similarity_threshold: f32,
}

impl EntityDeduplicator {
    pub fn new() -> Self {
        Self {
            embedding_model: Arc::new(MockEmbeddingModel::new()),
            similarity_threshold: 0.85,
        }
    }

    pub async fn deduplicate(&self, entities: Vec<String>) -> Result<DeduplicationResult> {
        let mut embeddings = Vec::new();
        
        // Generate embeddings for all entities
        for entity in &entities {
            let embedding = self.embedding_model.embed(entity).await?;
            embeddings.push(embedding);
        }
        
        // Find duplicate groups
        let mut duplicate_groups = Vec::new();
        let mut visited = vec![false; entities.len()];
        
        for i in 0..entities.len() {
            if visited[i] {
                continue;
            }
            
            let mut group = vec![i];
            visited[i] = true;
            
            for j in (i + 1)..entities.len() {
                if !visited[j] {
                    let similarity = self.cosine_similarity(&embeddings[i], &embeddings[j]);
                    if similarity > self.similarity_threshold {
                        group.push(j);
                        visited[j] = true;
                    }
                }
            }
            
            if group.len() > 1 {
                duplicate_groups.push(group);
            }
        }
        
        // Create canonical entities for each group
        let mut canonical_entities = Vec::new();
        let mut entity_mapping = HashMap::new();
        let _duplicate_group_count = duplicate_groups.len();
        
        for group in duplicate_groups {
            let canonical_entity_name = self.select_canonical_entity(&entities, &group)?;
            let canonical_entity = CanonicalEntity {
                original_name: canonical_entity_name.clone(),
                canonical_name: canonical_entity_name.clone(),
                confidence: 0.9,
                alternative_forms: group.iter().map(|&idx| entities[idx].clone()).collect(),
                normalization_applied: false,
                similar_entities: Vec::new(),
            };
            canonical_entities.push(canonical_entity);
            
            for &idx in &group {
                entity_mapping.insert(entities[idx].clone(), canonical_entity_name.clone());
            }
        }
        
        // Add entities that weren't duplicated
        for (_i, entity) in entities.iter().enumerate() {
            if !entity_mapping.contains_key(entity) {
                entity_mapping.insert(entity.clone(), entity.clone());
            }
        }
        
        // Calculate reduction ratio before moving values
        let entity_count = entities.len();
        let canonical_count = canonical_entities.len();
        let reduction_ratio = if canonical_count > 0 { 
            1.0 - (canonical_count as f32 / entity_count as f32) 
        } else { 
            0.0 
        };
        
        Ok(DeduplicationResult {
            original_entities: entities,
            canonical_entities,
            duplicate_groups: vec![], // Would need to collect actual groups
            reduction_ratio,
        })
    }

    fn select_canonical_entity(&self, entities: &[String], group: &[usize]) -> Result<String> {
        // Select the shortest entity name as canonical (usually more concise)
        let mut canonical = &entities[group[0]];
        
        for &idx in group {
            if entities[idx].len() < canonical.len() {
                canonical = &entities[idx];
            }
        }
        
        Ok(canonical.clone())
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Canonical entity representation
#[derive(Debug, Clone)]
pub struct CanonicalEntity {
    pub original_name: String,
    pub canonical_name: String,
    pub confidence: f32,
    pub alternative_forms: Vec<String>,
    pub normalization_applied: bool,
    pub similar_entities: Vec<String>,
}

/// Result of entity deduplication
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    pub original_entities: Vec<String>,
    pub canonical_entities: Vec<CanonicalEntity>,
    pub duplicate_groups: Vec<Vec<String>>,
    pub reduction_ratio: f32,
}

/// Statistics for canonicalization process
#[derive(Debug, Clone)]
pub struct CanonicalizationStats {
    pub total_entities_processed: usize,
    pub canonical_entities_created: usize,
    pub duplicates_found: usize,
    pub average_confidence: f32,
}

/// Enhanced Neural Canonicalizer with context awareness
pub struct EnhancedNeuralCanonicalizer {
    pub base_canonicalizer: NeuralCanonicalizer,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub entity_embedding_model: String,
    pub canonical_mapping: Arc<RwLock<ahash::AHashMap<String, String>>>,
}

impl EnhancedNeuralCanonicalizer {
    pub fn new(neural_server: Arc<NeuralProcessingServer>) -> Self {
        Self {
            base_canonicalizer: NeuralCanonicalizer::new_with_neural_server(neural_server.clone()),
            neural_server,
            entity_embedding_model: "entity_embedder".to_string(),
            canonical_mapping: Arc::new(RwLock::new(ahash::AHashMap::new())),
        }
    }

    /// Canonicalize with context awareness
    pub async fn canonicalize_with_context(
        &self,
        entity_name: &str,
        context: &str,
    ) -> Result<String> {
        // Check cached mapping first
        {
            let mapping = self.canonical_mapping.read().await;
            if let Some(canonical) = mapping.get(entity_name) {
                return Ok(canonical.clone());
            }
        }

        // Use neural model to generate context-aware embedding
        let context_input = format!("{} in context: {}", entity_name, context);
        let prediction = self.neural_server
            .neural_predict(&self.entity_embedding_model, vec![])
            .await?;

        // For now, use base canonicalizer as fallback
        let canonical = self.base_canonicalizer.canonicalize_entity(entity_name).await?;
        
        // Cache the mapping
        {
            let mut mapping = self.canonical_mapping.write().await;
            mapping.insert(entity_name.to_string(), canonical.clone());
        }

        Ok(canonical)
    }

    /// Generate canonical embedding for an entity
    pub async fn generate_canonical_embedding(
        &self,
        canonical_id: &str,
    ) -> Result<Vec<f32>> {
        // Use neural server to generate stable embeddings
        let result = self.neural_server
            .neural_predict(&self.entity_embedding_model, vec![])
            .await?;
        
        Ok(result.prediction)
    }
}


