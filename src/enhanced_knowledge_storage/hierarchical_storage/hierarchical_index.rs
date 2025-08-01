//! Hierarchical Index Management
//! 
//! Manages indexing structures for efficient retrieval across the hierarchical
//! knowledge layers, supporting both exact matches and semantic searches.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::enhanced_knowledge_storage::{
    model_management::ModelResourceManager,
    hierarchical_storage::types::{KnowledgeLayer, HierarchicalIndex, LayerIndexEntry, HierarchicalStorageConfig, HierarchicalStorageResult, LayerType, IndexMatch, SemanticCluster},
};

/// Manager for creating and maintaining hierarchical indexes
pub struct HierarchicalIndexManager {
    model_manager: Arc<ModelResourceManager>,
    config: HierarchicalStorageConfig,
}

impl HierarchicalIndexManager {
    /// Create new hierarchical index manager
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: HierarchicalStorageConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Build complete hierarchical index from knowledge layers
    pub async fn build_hierarchical_index(
        &self,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<HierarchicalIndex> {
        let mut index = HierarchicalIndex {
            layer_index: HashMap::new(),
            entity_index: HashMap::new(),
            concept_index: HashMap::new(),
            relationship_index: HashMap::new(),
            full_text_index: HashMap::new(),
            semantic_index: Vec::new(),
        };
        
        // Step 1: Build layer index
        self.build_layer_index(&mut index, layers)?;
        
        // Step 2: Build entity index
        self.build_entity_index(&mut index, layers)?;
        
        // Step 3: Build concept index
        self.build_concept_index(&mut index, layers)?;
        
        // Step 4: Build relationship index
        self.build_relationship_index(&mut index, layers)?;
        
        // Step 5: Build full-text index
        self.build_full_text_index(&mut index, layers)?;
        
        // Step 6: Build semantic index with clustering
        if self.config.enable_semantic_clustering {
            self.build_semantic_index(&mut index, layers).await?;
        }
        
        Ok(index)
    }
    
    /// Build layer index for fast layer lookup
    fn build_layer_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            let keywords = self.extract_layer_keywords(layer);
            
            let entry = LayerIndexEntry {
                layer_id: layer.layer_id.clone(),
                layer_type: layer.layer_type.clone(),
                parent_id: layer.parent_layer_id.clone(),
                keywords,
                entity_count: layer.entities.len(),
                relationship_count: layer.relationships.len(),
                importance_score: layer.importance_score,
                last_accessed: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                access_count: 0,
            };
            
            index.layer_index.insert(layer.layer_id.clone(), entry);
        }
        
        Ok(())
    }
    
    /// Build entity index mapping entity names to layer IDs
    fn build_entity_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            for entity in &layer.entities {
                index.entity_index
                    .entry(entity.name.clone())
                    .or_default()
                    .push(layer.layer_id.clone());
            }
        }
        
        // Remove duplicates from entity index
        for layer_ids in index.entity_index.values_mut() {
            layer_ids.sort();
            layer_ids.dedup();
        }
        
        Ok(())
    }
    
    /// Build concept index from key concepts and themes
    fn build_concept_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            // Index key phrases as concepts
            for phrase in &layer.content.key_phrases {
                index.concept_index
                    .entry(phrase.clone())
                    .or_default()
                    .push(layer.layer_id.clone());
            }
            
            // Index tags as concepts
            for tag in &layer.content.metadata.tags {
                index.concept_index
                    .entry(tag.clone())
                    .or_default()
                    .push(layer.layer_id.clone());
            }
        }
        
        // Remove duplicates
        for layer_ids in index.concept_index.values_mut() {
            layer_ids.sort();
            layer_ids.dedup();
        }
        
        Ok(())
    }
    
    /// Build relationship index mapping relationship types to layer IDs
    fn build_relationship_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            for relationship in &layer.relationships {
                let rel_type = relationship.predicate.to_string();
                index.relationship_index
                    .entry(rel_type)
                    .or_default()
                    .push(layer.layer_id.clone());
            }
        }
        
        // Remove duplicates
        for layer_ids in index.relationship_index.values_mut() {
            layer_ids.sort();
            layer_ids.dedup();
        }
        
        Ok(())
    }
    
    /// Build full-text index for keyword search
    fn build_full_text_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            let tokens = self.tokenize_text(&layer.content.processed_text);
            
            for (position, token) in tokens.iter().enumerate() {
                let matches = index.full_text_index
                    .entry(token.clone())
                    .or_default();
                
                // Find or create match for this layer
                let match_exists = matches.iter_mut().any(|m| {
                    if m.layer_id == layer.layer_id {
                        m.positions.push(position);
                        true
                    } else {
                        false
                    }
                });
                
                if !match_exists {
                    let context_start = position.saturating_sub(10);
                    let context_end = (position + 10).min(tokens.len());
                    let context_snippet = tokens[context_start..context_end].join(" ");
                    
                    matches.push(IndexMatch {
                        layer_id: layer.layer_id.clone(),
                        positions: vec![position],
                        relevance_score: self.calculate_term_relevance(token, layer),
                        context_snippet,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Build semantic index using clustering
    async fn build_semantic_index(
        &self,
        index: &mut HierarchicalIndex,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Filter layers with semantic embeddings
        let embedded_layers: Vec<&KnowledgeLayer> = layers
            .iter()
            .filter(|layer| layer.semantic_embedding.is_some())
            .collect();
        
        if embedded_layers.is_empty() {
            return Ok(());
        }
        
        // Perform clustering (simple k-means approximation)
        let num_clusters = (embedded_layers.len() as f32 / 10.0).ceil() as usize;
        let clusters = self.cluster_layers(&embedded_layers, num_clusters).await?;
        
        // Create semantic clusters
        for (cluster_id, cluster_layers) in clusters.iter().enumerate() {
            if cluster_layers.is_empty() {
                continue;
            }
            
            let center_embedding = self.calculate_cluster_center(cluster_layers);
            let cluster_label = self.generate_cluster_label(cluster_layers).await?;
            let representative_content = self.select_representative_content(cluster_layers);
            let coherence_score = self.calculate_cluster_coherence(cluster_layers);
            
            let semantic_cluster = SemanticCluster {
                cluster_id: format!("cluster_{cluster_id}"),
                center_embedding,
                member_layer_ids: cluster_layers.iter().map(|l| l.layer_id.clone()).collect(),
                cluster_label,
                coherence_score,
                representative_content,
            };
            
            index.semantic_index.push(semantic_cluster);
        }
        
        Ok(())
    }
    
    /// Update index incrementally when new layers are added
    pub async fn update_index_incremental(
        &self,
        index: &mut HierarchicalIndex,
        new_layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Update layer index
        self.build_layer_index(index, new_layers)?;
        
        // Update entity index
        self.build_entity_index(index, new_layers)?;
        
        // Update concept index
        self.build_concept_index(index, new_layers)?;
        
        // Update relationship index
        self.build_relationship_index(index, new_layers)?;
        
        // Update full-text index
        self.build_full_text_index(index, new_layers)?;
        
        // Re-cluster if semantic clustering is enabled
        if self.config.enable_semantic_clustering && !new_layers.is_empty() {
            // For now, rebuild the entire semantic index
            // In production, we'd use incremental clustering
            index.semantic_index.clear();
            let all_layers: Vec<KnowledgeLayer> = index.layer_index
                .keys()
                .filter_map(|layer_id| {
                    new_layers.iter().find(|l| &l.layer_id == layer_id).cloned()
                })
                .collect();
            
            self.build_semantic_index(index, &all_layers).await?;
        }
        
        Ok(())
    }
    
    /// Search index with multiple criteria
    pub fn search_index(
        &self,
        index: &HierarchicalIndex,
        query: &IndexQuery,
    ) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let mut layer_scores: HashMap<String, f32> = HashMap::new();
        
        // Search by keywords in full-text index
        if let Some(keywords) = &query.keywords {
            for keyword in keywords {
                if let Some(matches) = index.full_text_index.get(keyword) {
                    for match_item in matches {
                        *layer_scores.entry(match_item.layer_id.clone()).or_default() += 
                            match_item.relevance_score * query.keyword_weight;
                    }
                }
            }
        }
        
        // Search by entities
        if let Some(entities) = &query.entities {
            for entity in entities {
                if let Some(layer_ids) = index.entity_index.get(entity) {
                    for layer_id in layer_ids {
                        *layer_scores.entry(layer_id.clone()).or_default() += 
                            query.entity_weight;
                    }
                }
            }
        }
        
        // Search by concepts
        if let Some(concepts) = &query.concepts {
            for concept in concepts {
                if let Some(layer_ids) = index.concept_index.get(concept) {
                    for layer_id in layer_ids {
                        *layer_scores.entry(layer_id.clone()).or_default() += 
                            query.concept_weight;
                    }
                }
            }
        }
        
        // Search by relationship types
        if let Some(rel_types) = &query.relationship_types {
            for rel_type in rel_types {
                if let Some(layer_ids) = index.relationship_index.get(rel_type) {
                    for layer_id in layer_ids {
                        *layer_scores.entry(layer_id.clone()).or_default() += 
                            query.relationship_weight;
                    }
                }
            }
        }
        
        // Filter by layer type if specified
        if let Some(layer_types) = &query.layer_types {
            layer_scores.retain(|layer_id, _| {
                index.layer_index
                    .get(layer_id)
                    .map(|entry| layer_types.contains(&entry.layer_type))
                    .unwrap_or(false)
            });
        }
        
        // Apply importance boost
        for (layer_id, score) in layer_scores.iter_mut() {
            if let Some(entry) = index.layer_index.get(layer_id) {
                *score *= 1.0 + (entry.importance_score * query.importance_boost);
            }
        }
        
        // Convert to search results
        for (layer_id, score) in layer_scores {
            if score >= query.min_score {
                if let Some(entry) = index.layer_index.get(&layer_id) {
                    results.push(SearchResult {
                        layer_id: layer_id.clone(),
                        score,
                        layer_type: entry.layer_type.clone(),
                        matched_keywords: Vec::new(), // Could be populated with actual matches
                        matched_entities: Vec::new(),
                        matched_concepts: Vec::new(),
                    });
                }
            }
        }
        
        // Sort by score (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        results.truncate(query.max_results);
        
        results
    }
    
    /// Find semantically similar layers
    pub fn find_similar_layers(
        &self,
        index: &HierarchicalIndex,
        target_embedding: &[f32],
        max_results: usize,
    ) -> Vec<(String, f32)> {
        let mut similarities = Vec::new();
        
        // Check each semantic cluster
        for cluster in &index.semantic_index {
            let similarity = self.calculate_embedding_similarity(
                target_embedding,
                &cluster.center_embedding,
            );
            
            if similarity > self.config.semantic_similarity_threshold {
                // Add all members of this cluster
                for layer_id in &cluster.member_layer_ids {
                    similarities.push((layer_id.clone(), similarity * cluster.coherence_score));
                }
            }
        }
        
        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        similarities.truncate(max_results);
        
        similarities
    }
    
    // Helper methods
    
    /// Extract keywords from a layer
    fn extract_layer_keywords(&self, layer: &KnowledgeLayer) -> Vec<String> {
        let mut keywords = Vec::new();
        
        // Add key phrases
        keywords.extend(layer.content.key_phrases.clone());
        
        // Add entity names
        keywords.extend(layer.entities.iter().map(|e| e.name.clone()));
        
        // Add significant words from content
        let tokens = self.tokenize_text(&layer.content.processed_text);
        let significant_tokens: Vec<String> = tokens
            .into_iter()
            .filter(|token| token.len() > 4 && !self.is_stop_word(token))
            .take(10)
            .collect();
        
        keywords.extend(significant_tokens);
        
        // Remove duplicates
        keywords.sort();
        keywords.dedup();
        
        keywords
    }
    
    /// Tokenize text into words
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|word| !word.is_empty())
            .map(|word| word.to_string())
            .collect()
    }
    
    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those",
        ];
        
        STOP_WORDS.contains(&word)
    }
    
    /// Calculate term relevance for a layer
    fn calculate_term_relevance(&self, term: &str, layer: &KnowledgeLayer) -> f32 {
        let mut relevance = 0.5; // Base relevance
        
        // Boost if term is in key phrases
        if layer.content.key_phrases.iter().any(|phrase| phrase.contains(term)) {
            relevance += 0.3;
        }
        
        // Boost if term is an entity name
        if layer.entities.iter().any(|entity| entity.name.to_lowercase().contains(term)) {
            relevance += 0.2;
        }
        
        // Apply importance score
        relevance *= layer.importance_score;
        
        relevance.min(1.0)
    }
    
    /// Cluster layers using simple k-means
    async fn cluster_layers<'a>(
        &self,
        layers: &[&'a KnowledgeLayer],
        num_clusters: usize,
    ) -> HierarchicalStorageResult<Vec<Vec<&'a KnowledgeLayer>>> {
        if layers.is_empty() || num_clusters == 0 {
            return Ok(vec![]);
        }
        
        // Initialize clusters with random centers
        let mut clusters: Vec<Vec<&KnowledgeLayer>> = vec![Vec::new(); num_clusters];
        let mut centers: Vec<Vec<f32>> = Vec::new();
        
        // Initialize centers from first k layers
        for i in 0..num_clusters.min(layers.len()) {
            if let Some(embedding) = &layers[i].semantic_embedding {
                centers.push(embedding.clone());
            }
        }
        
        // Simple k-means iterations
        for _ in 0..10 {
            // Clear clusters
            for cluster in &mut clusters {
                cluster.clear();
            }
            
            // Assign layers to nearest center
            for layer in layers {
                if let Some(embedding) = &layer.semantic_embedding {
                    let mut min_distance = f32::MAX;
                    let mut best_cluster = 0;
                    
                    for (i, center) in centers.iter().enumerate() {
                        let distance = self.calculate_embedding_distance(embedding, center);
                        if distance < min_distance {
                            min_distance = distance;
                            best_cluster = i;
                        }
                    }
                    
                    clusters[best_cluster].push(layer);
                }
            }
            
            // Update centers
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() && i < centers.len() {
                    centers[i] = self.calculate_cluster_center(cluster);
                }
            }
        }
        
        Ok(clusters)
    }
    
    /// Calculate cluster center embedding
    fn calculate_cluster_center(&self, layers: &[&KnowledgeLayer]) -> Vec<f32> {
        if layers.is_empty() {
            return Vec::new();
        }
        
        let embedding_dim = layers[0]
            .semantic_embedding
            .as_ref()
            .map(|e| e.len())
            .unwrap_or(0);
        
        if embedding_dim == 0 {
            return Vec::new();
        }
        
        let mut center = vec![0.0; embedding_dim];
        let mut count = 0;
        
        for layer in layers {
            if let Some(embedding) = &layer.semantic_embedding {
                for (i, value) in embedding.iter().enumerate() {
                    if i < center.len() {
                        center[i] += value;
                    }
                }
                count += 1;
            }
        }
        
        if count > 0 {
            for value in &mut center {
                *value /= count as f32;
            }
        }
        
        center
    }
    
    /// Generate cluster label using AI
    async fn generate_cluster_label(
        &self,
        layers: &[&KnowledgeLayer],
    ) -> HierarchicalStorageResult<String> {
        if layers.is_empty() {
            return Ok("Empty cluster".to_string());
        }
        
        // Collect key phrases from cluster members
        let key_phrases: Vec<String> = layers
            .iter()
            .flat_map(|layer| &layer.content.key_phrases)
            .take(10)
            .cloned()
            .collect();
        
        if key_phrases.is_empty() {
            return Ok("Unlabeled cluster".to_string());
        }
        
        // Simple label generation from common phrases
        Ok(key_phrases[0].clone())
    }
    
    /// Select representative content from cluster
    fn select_representative_content(&self, layers: &[&KnowledgeLayer]) -> String {
        if layers.is_empty() {
            return String::new();
        }
        
        // Select layer with highest importance score
        let best_layer = layers
            .iter()
            .max_by(|a, b| a.importance_score.partial_cmp(&b.importance_score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        
        best_layer.content.summary
            .clone()
            .unwrap_or_else(|| best_layer.content.processed_text[..100.min(best_layer.content.processed_text.len())].to_string())
    }
    
    /// Calculate cluster coherence
    fn calculate_cluster_coherence(&self, layers: &[&KnowledgeLayer]) -> f32 {
        if layers.len() <= 1 {
            return 1.0;
        }
        
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        // Calculate average pairwise similarity
        for i in 0..layers.len() {
            for j in i + 1..layers.len() {
                if let (Some(emb1), Some(emb2)) = (&layers[i].semantic_embedding, &layers[j].semantic_embedding) {
                    total_similarity += self.calculate_embedding_similarity(emb1, emb2);
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            total_similarity / count as f32
        } else {
            0.5
        }
    }
    
    /// Calculate embedding similarity
    fn calculate_embedding_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        if emb1.len() != emb2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = emb1.iter().zip(emb2).map(|(a, b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }
    
    /// Calculate embedding distance
    fn calculate_embedding_distance(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        if emb1.len() != emb2.len() {
            return f32::MAX;
        }
        
        emb1.iter()
            .zip(emb2)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Query for searching the hierarchical index
#[derive(Debug, Clone)]
pub struct IndexQuery {
    pub keywords: Option<Vec<String>>,
    pub entities: Option<Vec<String>>,
    pub concepts: Option<Vec<String>>,
    pub relationship_types: Option<Vec<String>>,
    pub layer_types: Option<Vec<LayerType>>,
    pub min_score: f32,
    pub max_results: usize,
    pub keyword_weight: f32,
    pub entity_weight: f32,
    pub concept_weight: f32,
    pub relationship_weight: f32,
    pub importance_boost: f32,
}

impl Default for IndexQuery {
    fn default() -> Self {
        Self {
            keywords: None,
            entities: None,
            concepts: None,
            relationship_types: None,
            layer_types: None,
            min_score: 0.0,
            max_results: 20,
            keyword_weight: 1.0,
            entity_weight: 1.5,
            concept_weight: 1.2,
            relationship_weight: 1.3,
            importance_boost: 0.5,
        }
    }
}

/// Search result from index query
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub layer_id: String,
    pub score: f32,
    pub layer_type: LayerType,
    pub matched_keywords: Vec<String>,
    pub matched_entities: Vec<String>,
    pub matched_concepts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    use crate::enhanced_knowledge_storage::types::ModelResourceConfig;
    
    #[tokio::test]
    async fn test_index_manager_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = HierarchicalIndexManager::new(model_manager, storage_config);
        
        assert!(manager.config.enable_semantic_clustering);
    }
    
    #[tokio::test]
    async fn test_tokenization() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = HierarchicalIndexManager::new(model_manager, storage_config);
        
        let text = "The quick brown fox jumps over the lazy dog!";
        let tokens = manager.tokenize_text(text);
        
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0], "the");
        assert_eq!(tokens[1], "quick");
        assert_eq!(tokens[8], "dog");
    }
    
    #[tokio::test]
    async fn test_stop_word_detection() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = HierarchicalIndexManager::new(model_manager, storage_config);
        
        assert!(manager.is_stop_word("the"));
        assert!(manager.is_stop_word("and"));
        assert!(!manager.is_stop_word("quick"));
        assert!(!manager.is_stop_word("brown"));
    }
    
    #[tokio::test]
    async fn test_embedding_similarity() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = HierarchicalIndexManager::new(model_manager, storage_config);
        
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];
        let emb3 = vec![1.0, 0.0, 0.0];
        
        assert!((manager.calculate_embedding_similarity(&emb1, &emb2) - 0.0).abs() < 0.001);
        assert!((manager.calculate_embedding_similarity(&emb1, &emb3) - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_index_query_defaults() {
        let query = IndexQuery::default();
        
        assert_eq!(query.max_results, 20);
        assert_eq!(query.keyword_weight, 1.0);
        assert_eq!(query.entity_weight, 1.5);
        assert_eq!(query.min_score, 0.0);
    }
}