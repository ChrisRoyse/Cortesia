use crate::core::types::{EntityData, EntityKey};
use crate::core::semantic_summary::{SemanticSummary, SemanticSummarizer};
use crate::error::Result;
use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Semantic Storage that maintains both detailed summaries and fast access
/// Target: ~150-200 bytes per entity with rich semantic content for LLM understanding
pub struct SemanticStore {
    /// The semantic summarizer
    summarizer: RwLock<SemanticSummarizer>,
    
    /// Stored semantic summaries
    summaries: RwLock<HashMap<EntityKey, SemanticSummary>>,
    
    /// Entity ID to key mapping
    entity_id_map: RwLock<HashMap<u32, EntityKey>>,
    
    /// LLM-friendly text cache for quick access
    text_cache: RwLock<HashMap<EntityKey, String>>,
    
    /// Storage statistics
    stats: RwLock<SemanticStoreStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStoreStats {
    pub entity_count: usize,
    pub total_summary_bytes: usize,
    pub total_cache_bytes: usize,
    pub total_original_bytes: usize,
    pub avg_bytes_per_entity: usize,
    pub avg_compression_ratio: f32,
    pub avg_llm_comprehension_score: f32,
}

impl SemanticStore {
    pub fn new() -> Self {
        Self {
            summarizer: RwLock::new(SemanticSummarizer::new()),
            summaries: RwLock::new(HashMap::new()),
            entity_id_map: RwLock::new(HashMap::new()),
            text_cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(SemanticStoreStats::default()),
        }
    }

    /// Store an entity with semantic summarization
    pub fn store_entity(&self, entity_id: u32, entity_key: EntityKey, entity_data: &EntityData) -> Result<()> {
        // Create semantic summary
        let mut summarizer = self.summarizer.write();
        let summary = summarizer.create_summary(entity_data, entity_key)?;
        drop(summarizer);

        // Generate LLM-friendly text
        let summarizer = self.summarizer.read();
        let llm_text = summarizer.to_llm_text(&summary);
        drop(summarizer);

        // Store everything
        let mut summaries = self.summaries.write();
        let mut entity_id_map = self.entity_id_map.write();
        let mut text_cache = self.text_cache.write();

        summaries.insert(entity_key, summary.clone());
        entity_id_map.insert(entity_id, entity_key);
        text_cache.insert(entity_key, llm_text);

        drop(summaries);
        drop(entity_id_map);
        drop(text_cache);

        // Update statistics
        self.update_stats(&summary, entity_data);

        Ok(())
    }

    /// Bulk store entities for better performance
    pub fn bulk_store(&self, entities: Vec<(u32, EntityKey, EntityData)>) -> Result<()> {
        let mut summaries = self.summaries.write();
        let mut entity_id_map = self.entity_id_map.write();
        let mut text_cache = self.text_cache.write();
        let mut summarizer = self.summarizer.write();

        for (entity_id, entity_key, entity_data) in entities {
            // Create semantic summary
            let summary = summarizer.create_summary(&entity_data, entity_key)?;
            
            // Generate LLM-friendly text
            let llm_text = summarizer.to_llm_text(&summary);

            // Store
            summaries.insert(entity_key, summary.clone());
            entity_id_map.insert(entity_id, entity_key);
            text_cache.insert(entity_key, llm_text);

            // Update stats (simplified for bulk)
            self.update_stats_bulk(&summary, &entity_data);
        }

        Ok(())
    }

    /// Get the semantic summary for an entity
    pub fn get_summary(&self, entity_key: EntityKey) -> Option<SemanticSummary> {
        self.summaries.read().get(&entity_key).cloned()
    }

    /// Get the LLM-friendly text representation
    pub fn get_llm_text(&self, entity_key: EntityKey) -> Option<String> {
        self.text_cache.read().get(&entity_key).cloned()
    }

    /// Get summary by entity ID
    pub fn get_summary_by_id(&self, entity_id: u32) -> Option<SemanticSummary> {
        let entity_id_map = self.entity_id_map.read();
        if let Some(&entity_key) = entity_id_map.get(&entity_id) {
            drop(entity_id_map);
            self.get_summary(entity_key)
        } else {
            None
        }
    }

    /// Get LLM text by entity ID
    pub fn get_llm_text_by_id(&self, entity_id: u32) -> Option<String> {
        let entity_id_map = self.entity_id_map.read();
        if let Some(&entity_key) = entity_id_map.get(&entity_id) {
            drop(entity_id_map);
            self.get_llm_text(entity_key)
        } else {
            None
        }
    }

    /// Search for entities with semantic similarity
    pub fn semantic_search(&self, query_text: &str, limit: usize) -> Vec<(u32, f32, String)> {
        let summaries = self.summaries.read();
        let entity_id_map = self.entity_id_map.read();
        let text_cache = self.text_cache.read();

        let mut results = Vec::new();

        // Simple text-based search - would be more sophisticated in practice
        let query_lower = query_text.to_lowercase();
        
        for (&entity_key, _summary) in summaries.iter() {
            // Find the entity ID for this key
            let entity_id = entity_id_map.iter()
                .find(|(_, &key)| key == entity_key)
                .map(|(&id, _)| id)
                .unwrap_or(0);

            // Calculate semantic similarity (simplified)
            let similarity = if let Some(text) = text_cache.get(&entity_key) {
                self.calculate_text_similarity(&query_lower, text)
            } else {
                0.0
            };

            if similarity > 0.1 {
                let llm_text = text_cache.get(&entity_key).cloned().unwrap_or_default();
                results.push((entity_id, similarity, llm_text));
            }
        }

        // Sort by similarity and limit results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);

        results
    }

    /// Get comprehensive statistics about the semantic store
    pub fn get_stats(&self) -> SemanticStoreStats {
        self.stats.read().clone()
    }

    /// Get detailed analysis for LLM integration
    pub fn get_llm_integration_report(&self) -> String {
        let stats = self.get_stats();
        let summarizer = self.summarizer.read();
        
        let mut report = String::new();
        report.push_str("=== LLMKG Semantic Store Report ===\n\n");
        
        report.push_str(&format!("📊 Storage Statistics:\n"));
        report.push_str(&format!("  - Total entities: {}\n", stats.entity_count));
        report.push_str(&format!("  - Average bytes per entity: {} bytes\n", stats.avg_bytes_per_entity));
        report.push_str(&format!("  - Compression ratio: {:.1}x\n", stats.avg_compression_ratio));
        report.push_str(&format!("  - Total storage: {:.2} MB\n", 
                                (stats.total_summary_bytes + stats.total_cache_bytes) as f64 / 1_048_576.0));
        report.push_str("\n");
        
        report.push_str(&format!("🤖 LLM Integration Quality:\n"));
        report.push_str(&format!("  - Average comprehension score: {:.2}/1.0\n", stats.avg_llm_comprehension_score));
        report.push_str(&format!("  - Semantic richness: {}\n", 
                                if stats.avg_llm_comprehension_score > 0.8 { "Excellent" }
                                else if stats.avg_llm_comprehension_score > 0.6 { "Good" }
                                else { "Needs Improvement" }));
        
        report.push_str("\n");
        report.push_str("🎯 Summary Quality Analysis:\n");
        
        // Sample a few entities for detailed analysis
        let summaries = self.summaries.read();
        let sample_size = summaries.len().min(3);
        
        for (i, (_, summary)) in summaries.iter().take(sample_size).enumerate() {
            let comprehension = summarizer.estimate_llm_comprehension(summary);
            report.push_str(&format!("  Sample Entity {}:\n", i + 1));
            report.push_str(&format!("    - Features: {}\n", summary.key_features.len()));
            report.push_str(&format!("    - Context hints: {}\n", summary.context_hints.len()));
            report.push_str(&format!("    - LLM comprehension: {:.2}\n", comprehension));
            report.push_str(&format!("    - Original size: {} bytes\n", summary.reconstruction_metadata.original_size));
        }
        
        report.push_str("\n");
        report.push_str("✅ This semantic store provides detailed, LLM-friendly summaries that preserve\n");
        report.push_str("   essential semantic information while achieving efficient storage.\n");
        
        report
    }

    /// Calculate similarity between query and text (simplified implementation)
    fn calculate_text_similarity(&self, query: &str, text: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let query_words: Vec<&str> = query.split_whitespace().collect();
        
        let matches = query_words.iter()
            .filter(|word| text_lower.contains(*word))
            .count();
        
        if query_words.is_empty() {
            0.0
        } else {
            matches as f32 / query_words.len() as f32
        }
    }

    fn update_stats(&self, summary: &SemanticSummary, entity_data: &EntityData) {
        let mut stats = self.stats.write();
        
        // Calculate sizes
        let summary_size = self.estimate_summary_size(summary);
        let original_size = entity_data.properties.len() + entity_data.embedding.len() * 4;
        let compression_ratio = original_size as f32 / summary_size as f32;
        
        // Calculate LLM comprehension
        let summarizer = self.summarizer.read();
        let comprehension = summarizer.estimate_llm_comprehension(summary);
        drop(summarizer);
        
        // Update running averages
        let count = stats.entity_count as f32;
        stats.entity_count += 1;
        let new_count = stats.entity_count as f32;
        
        stats.total_summary_bytes += summary_size;
        stats.total_original_bytes += original_size;
        stats.avg_bytes_per_entity = ((stats.avg_bytes_per_entity as f32 * count) + summary_size as f32) as usize / stats.entity_count;
        stats.avg_compression_ratio = (stats.avg_compression_ratio * count + compression_ratio) / new_count;
        stats.avg_llm_comprehension_score = (stats.avg_llm_comprehension_score * count + comprehension) / new_count;
    }

    fn update_stats_bulk(&self, summary: &SemanticSummary, entity_data: &EntityData) {
        // Simplified stats update for bulk operations
        let mut stats = self.stats.write();
        
        let summary_size = self.estimate_summary_size(summary);
        let original_size = entity_data.properties.len() + entity_data.embedding.len() * 4;
        
        stats.entity_count += 1;
        stats.total_summary_bytes += summary_size;
        stats.total_original_bytes += original_size;
        
        // Update averages periodically
        if stats.entity_count % 100 == 0 {
            stats.avg_bytes_per_entity = stats.total_summary_bytes / stats.entity_count;
            stats.avg_compression_ratio = stats.total_original_bytes as f32 / stats.total_summary_bytes as f32;
        }
    }

    fn estimate_summary_size(&self, summary: &SemanticSummary) -> usize {
        // Rough estimate of serialized size
        let mut size = 0;
        
        // Entity type: ~8 bytes
        size += 8;
        
        // Key features: ~16 bytes per feature
        size += summary.key_features.len() * 16;
        
        // Semantic embedding: actual size
        size += summary.semantic_embedding.quantized_values.len();
        size += summary.semantic_embedding.scale_factors.len() * 4;
        size += summary.semantic_embedding.dimension_map.len();
        
        // Context hints: ~12 bytes per hint
        size += summary.context_hints.len() * 12;
        
        // Metadata: ~16 bytes
        size += 16;
        
        size
    }

    pub fn len(&self) -> usize {
        self.summaries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.summaries.read().is_empty()
    }
}

impl Default for SemanticStoreStats {
    fn default() -> Self {
        Self {
            entity_count: 0,
            total_summary_bytes: 0,
            total_cache_bytes: 0,
            total_original_bytes: 0,
            avg_bytes_per_entity: 0,
            avg_compression_ratio: 1.0,
            avg_llm_comprehension_score: 0.0,
        }
    }
}

impl Default for SemanticStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityData;

    #[test]
    fn test_semantic_store_creation() {
        let store = SemanticStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_entity_storage_and_retrieval() {
        let store = SemanticStore::new();
        let entity_key = EntityKey::default();
        
        let entity_data = EntityData {
            type_id: 1,
            properties: "This is a test entity with some meaningful content".to_string(),
            embedding: vec![0.1; 96],
        };
        
        store.store_entity(1, entity_key, &entity_data).unwrap();
        
        assert_eq!(store.len(), 1);
        
        let summary = store.get_summary(entity_key).unwrap();
        assert_eq!(summary.entity_type.type_id, 1);
        assert!(!summary.key_features.is_empty());
        
        let llm_text = store.get_llm_text(entity_key).unwrap();
        assert!(llm_text.contains("Entity Type: 1"));
    }

    #[test]
    fn test_semantic_search() {
        let store = SemanticStore::new();
        
        let entities = vec![
            (1, "This is about machine learning and AI"),
            (2, "Information about database systems"),
            (3, "Content related to machine learning algorithms"),
        ];
        
        for (id, content) in entities {
            let entity_data = EntityData {
                type_id: 1,
                properties: content.to_string(),
                embedding: vec![0.1; 96],
            };
            store.store_entity(id, EntityKey::default(), &entity_data).unwrap();
        }
        
        let results = store.semantic_search("machine learning", 10);
        assert!(!results.is_empty());
        
        // Should find entities containing "machine learning"
        assert!(results.iter().any(|(id, _, _)| *id == 1 || *id == 3));
    }

    #[test]
    fn test_storage_efficiency() {
        let store = SemanticStore::new();
        
        // Create a larger entity with substantial content
        let large_content = "This is a comprehensive entity with detailed properties and extensive metadata that would normally consume significant storage space in a traditional system".repeat(5);
        
        let entity_data = EntityData {
            type_id: 1,
            properties: large_content.clone(),
            embedding: vec![0.1; 96],
        };
        
        store.store_entity(1, EntityKey::default(), &entity_data).unwrap();
        
        let stats = store.get_stats();
        let original_size = large_content.len() + 96 * 4; // String + embedding
        
        // Should achieve reasonable compression while maintaining semantic richness
        assert!(stats.avg_bytes_per_entity < original_size);
        assert!(stats.avg_bytes_per_entity > 100); // But not too aggressive
        assert!(stats.avg_compression_ratio > 1.0);
    }
}