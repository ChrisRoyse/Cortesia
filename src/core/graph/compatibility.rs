//! Compatibility layer for legacy API support

use super::graph_core::KnowledgeGraph;
use crate::core::types::{EntityKey, EntityData};
use crate::error::Result;
use ahash::AHashMap;
use std::collections::HashMap;

impl KnowledgeGraph {
    /// Legacy entity insertion with text-based embedding generation
    pub fn insert_entity_with_text(&self, id: u32, text: &str, properties: HashMap<String, String>) -> Result<EntityKey> {
        // Generate embedding from text
        let embedding = self.generate_text_embedding(text);
        
        // Create entity data
        let entity_data = EntityData {
            type_id: 1, // Legacy entity type
            embedding,
            properties: serde_json::to_string(&properties).unwrap_or_default(),
        };
        
        // Insert entity
        self.insert_entity(id, entity_data)
    }

    /// Legacy batch entity insertion with text
    pub fn insert_entities_with_text(&self, entities: Vec<(u32, String, HashMap<String, String>)>) -> Result<Vec<EntityKey>> {
        let mut entity_data_vec = Vec::new();
        
        for (id, text, properties) in entities {
            let embedding = self.generate_text_embedding(&text);
            let entity_data = EntityData {
                type_id: 1, // Legacy entity type
                embedding,
                properties: serde_json::to_string(&properties).unwrap_or_default(),
            };
            entity_data_vec.push((id, entity_data));
        }
        
        self.insert_entities_batch(entity_data_vec)
    }

    /// Generate embedding from text (simplified implementation)
    fn generate_text_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.embedding_dim];
        
        // Simple hash-based embedding generation
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        
        // Hash text chunks
        for (i, chunk) in text.chars().collect::<Vec<_>>().chunks(16).enumerate() {
            let chunk_str: String = chunk.iter().collect();
            chunk_str.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Distribute hash across embedding dimensions
            for (j, val) in embedding.iter_mut().enumerate() {
                if (i * 16 + j) < embedding.len() {
                    *val = ((hash as u64).wrapping_mul(j as u64 + 1) % 1000) as f32 / 1000.0;
                }
            }
        }
        
        // Normalize embedding
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in embedding.iter_mut() {
                *val /= magnitude;
            }
        }
        
        embedding
    }

    /// Legacy relationship insertion with string IDs
    pub fn insert_relationship_by_id(&self, source_id: u32, target_id: u32, weight: f32) -> Result<()> {
        let source_key = self.get_entity_key(source_id)
            .ok_or_else(|| crate::error::GraphError::EntityNotFound(crate::core::types::EntityKey::new(source_id as u64)))?;
        let target_key = self.get_entity_key(target_id)
            .ok_or_else(|| crate::error::GraphError::EntityNotFound(crate::core::types::EntityKey::new(target_id as u64)))?;
        
        let relationship = crate::core::types::Relationship {
            source: source_key,
            target: target_key,
            weight,
        };
        
        self.insert_relationship(relationship)
    }

    /// Legacy neighbor lookup by ID
    pub fn get_neighbors_by_id(&self, entity_id: u32) -> Vec<u32> {
        if let Some(entity_key) = self.get_entity_key(entity_id) {
            let neighbor_keys = self.get_neighbors(entity_key);
            
            // Convert back to IDs
            let mut neighbor_ids = Vec::new();
            let id_map = self.entity_id_map.read();
            
            for neighbor_key in neighbor_keys {
                // Find ID for this key
                for (id, key) in id_map.iter() {
                    if *key == neighbor_key {
                        neighbor_ids.push(*id);
                        break;
                    }
                }
            }
            
            neighbor_ids
        } else {
            Vec::new()
        }
    }

    /// Legacy entity lookup by ID
    pub fn get_entity_by_id_legacy(&self, id: u32) -> Option<(HashMap<String, String>, Vec<f32>)> {
        if let Some((_, data)) = self.get_entity_by_id(id) {
            Some((data.properties, data.embedding))
        } else {
            None
        }
    }

    /// Legacy similarity search by text
    pub fn similarity_search_by_text(&self, query_text: &str, k: usize) -> Result<Vec<(u32, f32)>> {
        let query_embedding = self.generate_text_embedding(query_text);
        let results = self.similarity_search(&query_embedding, k)?;
        
        // Convert entity keys back to IDs
        let mut id_results = Vec::new();
        let id_map = self.entity_id_map.read();
        
        for (entity_key, similarity) in results {
            // Find ID for this key
            for (id, key) in id_map.iter() {
                if *key == entity_key {
                    id_results.push((*id, similarity));
                    break;
                }
            }
        }
        
        Ok(id_results)
    }

    /// Legacy batch similarity search
    pub fn batch_similarity_search_by_text(&self, query_texts: &[String], k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        let mut results = Vec::new();
        
        for query_text in query_texts {
            let query_result = self.similarity_search_by_text(query_text, k)?;
            results.push(query_result);
        }
        
        Ok(results)
    }

    /// Legacy bloom filter methods
    pub fn bloom_contains(&self, id: u32) -> bool {
        self.contains_entity(id)
    }

    pub fn bloom_insert(&self, id: u32) -> Result<()> {
        let mut bloom = self.bloom_filter.write();
        bloom.insert(&id);
        Ok(())
    }

    pub fn bloom_false_positive_rate(&self) -> f64 {
        let bloom = self.bloom_filter.read();
        bloom.false_positive_rate()
    }

    /// Legacy entity management methods
    pub fn get_entity_properties(&self, id: u32) -> Option<HashMap<String, String>> {
        if let Some((_, data)) = self.get_entity_by_id(id) {
            Some(data.properties)
        } else {
            None
        }
    }

    pub fn get_entity_embedding_by_id(&self, id: u32) -> Option<Vec<f32>> {
        if let Some(key) = self.get_entity_key(id) {
            self.get_entity_embedding(key)
        } else {
            None
        }
    }

    pub fn update_entity_properties(&self, id: u32, properties: HashMap<String, String>) -> Result<()> {
        if let Some(key) = self.get_entity_key(id) {
            if let Some((_, mut data)) = self.get_entity(key) {
                data.properties = properties;
                self.update_entity(key, data)?;
                Ok(())
            } else {
                Err(crate::error::GraphError::EntityNotFound(key))
            }
        } else {
            Err(crate::error::GraphError::EntityNotFound(crate::core::types::EntityKey::new(id as u64)))
        }
    }

    /// Legacy graph statistics
    pub fn get_graph_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("entity_count".to_string(), self.entity_count() as f64);
        stats.insert("relationship_count".to_string(), self.relationship_count() as f64);
        stats.insert("embedding_dimension".to_string(), self.embedding_dim as f64);
        stats.insert("memory_usage_mb".to_string(), self.memory_usage().total_bytes() as f64 / (1024.0 * 1024.0));
        
        let (cache_size, cache_capacity, cache_hit_rate) = self.cache_stats();
        stats.insert("cache_size".to_string(), cache_size as f64);
        stats.insert("cache_capacity".to_string(), cache_capacity as f64);
        stats.insert("cache_hit_rate".to_string(), cache_hit_rate);
        
        stats
    }

    /// Legacy path finding by ID
    pub fn find_path_by_id(&self, source_id: u32, target_id: u32) -> Option<Vec<u32>> {
        let source_key = self.get_entity_key(source_id)?;
        let target_key = self.get_entity_key(target_id)?;
        
        if let Some(path_keys) = self.find_path(source_key, target_key) {
            let mut path_ids = Vec::new();
            let id_map = self.entity_id_map.read();
            
            for path_key in path_keys {
                // Find ID for this key
                for (id, key) in id_map.iter() {
                    if *key == path_key {
                        path_ids.push(*id);
                        break;
                    }
                }
            }
            
            Some(path_ids)
        } else {
            None
        }
    }

    /// Legacy entity removal by ID
    pub fn remove_entity_by_id(&self, id: u32) -> Result<bool> {
        if let Some(key) = self.get_entity_key(id) {
            self.remove_entity(key)
        } else {
            Ok(false)
        }
    }

    /// Legacy relationship removal by ID
    pub fn remove_relationship_by_id(&self, source_id: u32, target_id: u32) -> Result<bool> {
        let source_key = self.get_entity_key(source_id)
            .ok_or_else(|| crate::error::GraphError::EntityNotFound(crate::core::types::EntityKey::new(source_id as u64)))?;
        let target_key = self.get_entity_key(target_id)
            .ok_or_else(|| crate::error::GraphError::EntityNotFound(crate::core::types::EntityKey::new(target_id as u64)))?;
        
        self.remove_relationship(source_key, target_key)
    }

    /// Legacy string dictionary methods
    pub fn get_string_id(&self, s: &str) -> Option<u32> {
        self.string_dictionary.read().get(s).copied()
    }

    pub fn insert_string(&self, s: String) -> u32 {
        let mut dict = self.string_dictionary.write();
        let next_id = dict.len() as u32;
        dict.insert(s, next_id);
        next_id
    }

    pub fn get_string_by_id(&self, id: u32) -> Option<String> {
        let dict = self.string_dictionary.read();
        for (string, string_id) in dict.iter() {
            if *string_id == id {
                return Some(string.clone());
            }
        }
        None
    }

    /// Legacy export/import methods
    pub fn export_entities(&self) -> Vec<(u32, HashMap<String, String>, Vec<f32>)> {
        let mut exported = Vec::new();
        let id_map = self.entity_id_map.read();
        
        for (id, key) in id_map.iter() {
            if let Some((_, data)) = self.get_entity(*key) {
                exported.push((*id, data.properties, data.embedding));
            }
        }
        
        exported
    }

    pub fn export_relationships(&self) -> Vec<(u32, u32, f32)> {
        let mut exported = Vec::new();
        let id_map = self.entity_id_map.read();
        
        // Create reverse mapping
        let mut key_to_id = AHashMap::new();
        for (id, key) in id_map.iter() {
            key_to_id.insert(*key, *id);
        }
        
        for (id, key) in id_map.iter() {
            let relationships = self.get_outgoing_relationships(*key);
            for relationship in relationships {
                if let Some(target_id) = key_to_id.get(&relationship.target) {
                    exported.push((*id, *target_id, relationship.weight));
                }
            }
        }
        
        exported
    }

    /// Legacy performance test methods
    pub fn warmup_indices(&self) -> Result<()> {
        // Perform some dummy operations to warm up indices
        let dummy_embedding = vec![0.1; self.embedding_dim];
        let _ = self.similarity_search(&dummy_embedding, 1)?;
        
        // Clear the cache after warmup
        self.clear_caches();
        
        Ok(())
    }

    /// Legacy index rebuild
    pub fn rebuild_indices(&self) -> Result<()> {
        // This would typically rebuild all indices from scratch
        // For now, we'll just clear caches
        self.clear_caches();
        Ok(())
    }

    /// Legacy entity count by type
    pub fn count_entities_by_type(&self, entity_type: &str) -> usize {
        let mut count = 0;
        let id_map = self.entity_id_map.read();
        
        for (_, key) in id_map.iter() {
            if let Some((_, data)) = self.get_entity(*key) {
                if let Some(type_value) = data.properties.get("type") {
                    if type_value == entity_type {
                        count += 1;
                    }
                }
            }
        }
        
        count
    }

    /// Legacy relationship count between types
    pub fn count_relationships_between_types(&self, source_type: &str, target_type: &str) -> usize {
        let mut count = 0;
        let id_map = self.entity_id_map.read();
        
        for (_, key) in id_map.iter() {
            if let Some((_, data)) = self.get_entity(*key) {
                if let Some(type_value) = data.properties.get("type") {
                    if type_value == source_type {
                        let outgoing = self.get_outgoing_relationships(*key);
                        for relationship in outgoing {
                            if let Some((_, target_data)) = self.get_entity(relationship.target) {
                                if let Some(target_type_value) = target_data.properties.get("type") {
                                    if target_type_value == target_type {
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        count
    }

    /// Legacy validation methods
    pub fn validate_graph_consistency(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        // Check entity-key mapping consistency
        let id_map = self.entity_id_map.read();
        let entity_count = self.entity_count();
        
        if id_map.len() != entity_count {
            issues.push(format!("Entity count mismatch: ID map has {} entries, arena has {} entities", 
                id_map.len(), entity_count));
        }
        
        // Check relationship consistency
        let relationship_count = self.relationship_count();
        let buffer_size = self.edge_buffer_size();
        
        if buffer_size > 1000 {
            issues.push(format!("Edge buffer is very large: {} entries", buffer_size));
        }
        
        // Check embedding dimension consistency
        for (id, key) in id_map.iter() {
            if let Some((_, data)) = self.get_entity(*key) {
                if data.embedding.len() != self.embedding_dim {
                    issues.push(format!("Entity {} has incorrect embedding dimension: {} (expected {})", 
                        id, data.embedding.len(), self.embedding_dim));
                }
            }
        }
        
        issues
    }
}