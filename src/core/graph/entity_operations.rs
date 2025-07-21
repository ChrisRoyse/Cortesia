//! Entity operations for knowledge graph

use super::graph_core::{KnowledgeGraph, MAX_INSERTION_TIME};
use crate::core::types::{EntityKey, EntityData, EntityMeta};
use crate::core::parallel::{ParallelProcessor, ParallelOperation};
use crate::error::{GraphError, Result};
use std::time::Instant;

impl KnowledgeGraph {
    /// Insert a single entity into the graph
    pub fn insert_entity(&self, id: u32, data: EntityData) -> Result<EntityKey> {
        let start_time = Instant::now();
        
        // Validate text size to prevent bloat
        crate::text::TextCompressor::validate_text_size(&data.properties)?;
        
        if data.embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: data.embedding.len(),
            });
        }
        
        let mut arena = self.arena.write();
        let key = arena.allocate_entity(data.clone());
        
        let mut entity_store = self.entity_store.write();
        let mut meta = entity_store.insert(key, &data)?;
        
        // Store quantized embedding
        let mut embedding_bank = self.embedding_bank.write();
        let embedding_offset = embedding_bank.len() as u32;
        
        let quantizer = self.quantizer.read();
        let quantized = quantizer.encode(&data.embedding)?;
        embedding_bank.extend_from_slice(&quantized);
        
        meta.embedding_offset = embedding_offset;
        entity_store.get_mut(key).unwrap().embedding_offset = embedding_offset;
        
        // Update indices
        let mut bloom = self.bloom_filter.write();
        bloom.insert(&id);
        
        let mut id_map = self.entity_id_map.write();
        id_map.insert(id, key);
        
        // Add to all indices for fast similarity search
        let mut spatial_index = self.spatial_index.write();
        spatial_index.insert(id, key, data.embedding.clone())?;
        
        let mut flat_index = self.flat_index.write();
        flat_index.insert(id, key, data.embedding.clone())?;
        
        let hnsw_index = self.hnsw_index.write();
        hnsw_index.insert(id, key, data.embedding.clone())?;
        
        let lsh_index = self.lsh_index.write();
        lsh_index.insert(id, key, data.embedding.clone())?;
        
        // Check if insertion took too long
        if start_time.elapsed() > MAX_INSERTION_TIME {
            #[cfg(debug_assertions)]
            log::warn!("Entity insertion took longer than expected: {:.2}s", start_time.elapsed().as_secs_f64());
        }
        
        Ok(key)
    }
    
    /// Batch insert entities for high throughput
    /// This is much more efficient than individual inserts for large datasets
    /// Uses parallel processing for validation and encoding on large batches
    pub fn insert_entities_batch(&self, entities: Vec<(u32, EntityData)>) -> Result<Vec<EntityKey>> {
        let start_time = Instant::now();
        let batch_size = entities.len();
        
        #[cfg(debug_assertions)]
        log::trace!("Starting batch insert of {} entities", batch_size);
        
        // Use parallel validation for large batches
        if ParallelProcessor::should_use_parallel(batch_size, ParallelOperation::BatchValidation) {
            #[cfg(debug_assertions)]
            log::trace!("Using parallel validation for {} entities", batch_size);
            
            ParallelProcessor::parallel_validate_entities(&entities, self.embedding_dim)?;
        } else {
            // Sequential validation for smaller batches
            for (_, data) in &entities {
                crate::text::TextCompressor::validate_text_size(&data.properties)?;
                if data.embedding.len() != self.embedding_dim {
                    return Err(GraphError::InvalidEmbeddingDimension {
                        expected: self.embedding_dim,
                        actual: data.embedding.len(),
                    });
                }
            }
        }
        
        let mut keys = Vec::with_capacity(batch_size);
        let mut spatial_entries = Vec::with_capacity(batch_size);
        
        // Allocate all entities in the arena first
        {
            let mut arena = self.arena.write();
            let mut entity_store = self.entity_store.write();
            let mut embedding_bank = self.embedding_bank.write();
            let quantizer = self.quantizer.read();
            
            for (id, data) in &entities {
                // Validate
                crate::text::TextCompressor::validate_text_size(&data.properties)?;
                
                if data.embedding.len() != self.embedding_dim {
                    return Err(GraphError::InvalidEmbeddingDimension {
                        expected: self.embedding_dim,
                        actual: data.embedding.len(),
                    });
                }
                
                // Allocate entity
                let key = arena.allocate_entity(data.clone());
                let mut meta = entity_store.insert(key, data)?;
                
                // Store quantized embedding
                let embedding_offset = embedding_bank.len() as u32;
                let quantized = quantizer.encode(&data.embedding)?;
                embedding_bank.extend_from_slice(&quantized);
                
                meta.embedding_offset = embedding_offset;
                entity_store.get_mut(key).unwrap().embedding_offset = embedding_offset;
                
                keys.push(key);
                spatial_entries.push((*id, key, data.embedding.clone()));
            }
        }
        
        // Batch update indices
        {
            let mut bloom = self.bloom_filter.write();
            let mut id_map = self.entity_id_map.write();
            
            for (id, key, _) in &spatial_entries {
                bloom.insert(id);
                id_map.insert(*id, *key);
            }
        }
        
        // Batch update spatial indices
        {
            let mut spatial_index = self.spatial_index.write();
            let mut flat_index = self.flat_index.write();
            let hnsw_index = self.hnsw_index.write();
            let lsh_index = self.lsh_index.write();
            
            for (id, key, embedding) in spatial_entries {
                spatial_index.insert(id, key, embedding.clone())?;
                flat_index.insert(id, key, embedding.clone())?;
                hnsw_index.insert(id, key, embedding.clone())?;
                lsh_index.insert(id, key, embedding)?;
            }
        }
        
        #[cfg(debug_assertions)]
        log::trace!("Batch insert completed in {:.2}ms", start_time.elapsed().as_millis());
        
        Ok(keys)
    }

    /// Get entity by key
    pub fn get_entity(&self, key: EntityKey) -> Option<(EntityMeta, EntityData)> {
        let entity_store = self.entity_store.read();
        let arena = self.arena.read();
        
        if let Some(meta) = entity_store.get(key) {
            if let Some(data) = arena.get_entity(key) {
                return Some((meta.clone(), data.clone()));
            }
        }
        
        None
    }

    /// Get entity by ID
    pub fn get_entity_by_id(&self, id: u32) -> Option<(EntityMeta, EntityData)> {
        let key = self.get_entity_key(id)?;
        self.get_entity(key)
    }

    /// Get entity data only
    pub fn get_entity_data(&self, key: EntityKey) -> Option<EntityData> {
        let arena = self.arena.read();
        arena.get_entity(key).cloned()
    }

    /// Get entity metadata only
    pub fn get_entity_meta(&self, key: EntityKey) -> Option<EntityMeta> {
        let entity_store = self.entity_store.read();
        entity_store.get(key).cloned()
    }

    /// Update entity data
    pub fn update_entity(&self, key: EntityKey, data: EntityData) -> Result<()> {
        // Validate embedding dimension
        self.validate_embedding_dimension(&data.embedding)?;
        
        // Validate text size
        crate::text::TextCompressor::validate_text_size(&data.properties)?;
        
        let mut arena = self.arena.write();
        let mut entity_store = self.entity_store.write();
        let mut embedding_bank = self.embedding_bank.write();
        
        // Update entity data in arena
        arena.update_entity(key, data.clone())?;
        
        // Update metadata
        if let Some(meta) = entity_store.get_mut(key) {
            // Store new quantized embedding
            let embedding_offset = embedding_bank.len() as u32;
            let quantizer = self.quantizer.read();
            let quantized = quantizer.encode(&data.embedding)?;
            embedding_bank.extend_from_slice(&quantized);
            
            meta.embedding_offset = embedding_offset;
            meta.last_accessed = std::time::Instant::now();
        }
        
        Ok(())
    }

    /// Remove entity
    pub fn remove_entity(&self, key: EntityKey) -> Result<bool> {
        let mut arena = self.arena.write();
        let mut entity_store = self.entity_store.write();
        let mut id_map = self.entity_id_map.write();
        
        // Remove from arena
        let removed = arena.remove_entity(key);
        
        if removed.is_some() {
            // Remove from entity store
            entity_store.remove(key);
            
            // Remove from ID mapping (need to find the ID first)
            let mut id_to_remove = None;
            for (id, stored_key) in id_map.iter() {
                if *stored_key == key {
                    id_to_remove = Some(*id);
                    break;
                }
            }
            
            if let Some(id) = id_to_remove {
                id_map.remove(&id);
                
                // Remove from bloom filter
                let mut bloom = self.bloom_filter.write();
                bloom.remove(id);
                
                // Remove from spatial indices
                let mut spatial_index = self.spatial_index.write();
                spatial_index.remove(id);
                
                let mut flat_index = self.flat_index.write();
                flat_index.remove(id);
                
                let hnsw_index = self.hnsw_index.write();
                hnsw_index.remove(id);
                
                let lsh_index = self.lsh_index.write();
                lsh_index.remove(id);
            }
        }
        
        Ok(removed.is_some())
    }

    /// Get all entity IDs
    pub fn get_all_entity_ids(&self) -> Vec<u32> {
        let id_map = self.entity_id_map.read();
        id_map.keys().cloned().collect()
    }

    /// Get all entity keys
    pub fn get_all_entity_keys(&self) -> Vec<EntityKey> {
        let id_map = self.entity_id_map.read();
        id_map.values().cloned().collect()
    }

    /// Check if entity exists by key
    pub fn contains_entity_key(&self, key: EntityKey) -> bool {
        let arena = self.arena.read();
        arena.contains_entity(key)
    }

    /// Validate entity data
    pub fn validate_entity_data(&self, data: &EntityData) -> Result<()> {
        // Validate embedding dimension
        self.validate_embedding_dimension(&data.embedding)?;
        
        // Validate text size
        crate::text::TextCompressor::validate_text_size(&data.properties)?;
        
        Ok(())
    }

    /// Get entity embedding
    pub fn get_entity_embedding(&self, key: EntityKey) -> Option<Vec<f32>> {
        let entity_store = self.entity_store.read();
        let embedding_bank = self.embedding_bank.read();
        let quantizer = self.quantizer.read();
        
        if let Some(meta) = entity_store.get(key) {
            let offset = meta.embedding_offset as usize;
            let quantized_size = quantizer.num_subspaces();
            
            if offset + quantized_size <= embedding_bank.len() {
                let quantized = &embedding_bank[offset..offset + quantized_size];
                if let Ok(embedding) = quantizer.decode(quantized) {
                    return Some(embedding);
                }
            }
        }
        
        None
    }

    /// Get entity statistics
    pub fn get_entity_stats(&self) -> EntityStats {
        let entity_count = self.entity_count();
        let id_map = self.entity_id_map.read();
        let unique_ids = id_map.len();
        
        EntityStats {
            total_entities: entity_count,
            unique_entity_ids: unique_ids,
            id_mapping_size: id_map.len(),
            embedding_bank_size: self.embedding_bank.read().len(),
        }
    }
}

/// Entity statistics
#[derive(Debug, Clone)]
pub struct EntityStats {
    pub total_entities: usize,
    pub unique_entity_ids: usize,
    pub id_mapping_size: usize,
    pub embedding_bank_size: usize,
}

impl EntityStats {
    /// Get average embedding size
    pub fn average_embedding_size(&self) -> f64 {
        if self.total_entities == 0 {
            0.0
        } else {
            self.embedding_bank_size as f64 / self.total_entities as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{EntityData, AttributeValue};
    use std::collections::HashMap;

    /// Helper function to create a test knowledge graph
    fn create_test_graph() -> KnowledgeGraph {
        KnowledgeGraph::new_with_dimension(4).expect("Failed to create test graph")
    }

    /// Helper function to create test entity data
    fn create_test_entity_data(id: u32, embedding: Vec<f32>) -> EntityData {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), AttributeValue::String(format!("entity_{}", id)));
        properties.insert("value".to_string(), AttributeValue::Number(id as f64));
        
        EntityData {
            type_id: 1,
            properties: serde_json::to_string(&properties).unwrap(),
            embedding,
        }
    }

    /// Helper function to create test entity data with empty properties
    fn create_empty_entity_data(embedding: Vec<f32>) -> EntityData {
        EntityData {
            type_id: 1,
            properties: "{}".to_string(),
            embedding,
        }
    }

    #[test]
    fn test_insert_entity_success() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let result = graph.insert_entity(1, entity_data.clone());
        assert!(result.is_ok());
        
        let key = result.unwrap();
        
        // Verify entity was inserted
        let retrieved = graph.get_entity(key);
        assert!(retrieved.is_some());
        
        let (meta, data) = retrieved.unwrap();
        assert_eq!(data.type_id, entity_data.type_id);
        assert_eq!(data.properties, entity_data.properties);
        assert_eq!(data.embedding, entity_data.embedding);
    }

    #[test]
    fn test_insert_entity_with_empty_properties() {
        let graph = create_test_graph();
        let entity_data = create_empty_entity_data(vec![1.0, 2.0, 3.0, 4.0]);
        
        let result = graph.insert_entity(1, entity_data.clone());
        assert!(result.is_ok());
        
        let key = result.unwrap();
        let retrieved = graph.get_entity(key);
        assert!(retrieved.is_some());
        
        let (_, data) = retrieved.unwrap();
        assert_eq!(data.properties, "{}");
    }

    #[test]
    fn test_insert_entity_invalid_embedding_dimension() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0]); // Wrong dimension
        
        let result = graph.insert_entity(1, entity_data);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            GraphError::InvalidEmbeddingDimension { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InvalidEmbeddingDimension error"),
        }
    }

    #[test]
    fn test_get_entity_nonexistent() {
        let graph = create_test_graph();
        let fake_key = EntityKey::default();
        
        let result = graph.get_entity(fake_key);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_entity_by_id() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(42, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(42, entity_data.clone()).unwrap();
        
        // Test get by ID
        let retrieved = graph.get_entity_by_id(42);
        assert!(retrieved.is_some());
        
        let (_, data) = retrieved.unwrap();
        assert_eq!(data.type_id, entity_data.type_id);
        assert_eq!(data.embedding, entity_data.embedding);
        
        // Test nonexistent ID
        let nonexistent = graph.get_entity_by_id(999);
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_insert_entities_batch_success() {
        let graph = create_test_graph();
        
        let entities = vec![
            (1, create_test_entity_data(1, vec![1.0, 0.0, 0.0, 0.0])),
            (2, create_test_entity_data(2, vec![0.0, 1.0, 0.0, 0.0])),
            (3, create_test_entity_data(3, vec![0.0, 0.0, 1.0, 0.0])),
        ];
        
        let result = graph.insert_entities_batch(entities.clone());
        assert!(result.is_ok());
        
        let keys = result.unwrap();
        assert_eq!(keys.len(), 3);
        
        // Verify all entities were inserted
        for (i, key) in keys.iter().enumerate() {
            let retrieved = graph.get_entity(*key);
            assert!(retrieved.is_some());
            
            let (_, data) = retrieved.unwrap();
            assert_eq!(data.type_id, entities[i].1.type_id);
            assert_eq!(data.embedding, entities[i].1.embedding);
        }
    }

    #[test]
    fn test_insert_entities_batch_with_validation_error() {
        let graph = create_test_graph();
        
        let entities = vec![
            (1, create_test_entity_data(1, vec![1.0, 0.0, 0.0, 0.0])),
            (2, create_test_entity_data(2, vec![0.0, 1.0])), // Wrong dimension
            (3, create_test_entity_data(3, vec![0.0, 0.0, 1.0, 0.0])),
        ];
        
        let result = graph.insert_entities_batch(entities);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            GraphError::InvalidEmbeddingDimension { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InvalidEmbeddingDimension error"),
        }
    }

    #[test]
    fn test_insert_entities_batch_empty() {
        let graph = create_test_graph();
        
        let result = graph.insert_entities_batch(vec![]);
        assert!(result.is_ok());
        
        let keys = result.unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_update_entity_success() {
        let graph = create_test_graph();
        let original_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, original_data).unwrap();
        
        // Update the entity
        let updated_data = create_test_entity_data(1, vec![5.0, 6.0, 7.0, 8.0]);
        let result = graph.update_entity(key, updated_data.clone());
        assert!(result.is_ok());
        
        // Verify the update
        let retrieved = graph.get_entity(key);
        assert!(retrieved.is_some());
        
        let (_, data) = retrieved.unwrap();
        assert_eq!(data.embedding, updated_data.embedding);
    }

    #[test]
    fn test_update_entity_invalid_key() {
        let graph = create_test_graph();
        let fake_key = EntityKey::default();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let result = graph.update_entity(fake_key, entity_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_entity_invalid_embedding_dimension() {
        let graph = create_test_graph();
        let original_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, original_data).unwrap();
        
        // Try to update with wrong embedding dimension
        let invalid_data = create_test_entity_data(1, vec![1.0, 2.0]); // Wrong dimension
        let result = graph.update_entity(key, invalid_data);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            GraphError::InvalidEmbeddingDimension { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InvalidEmbeddingDimension error"),
        }
    }

    #[test]
    fn test_remove_entity_success() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, entity_data).unwrap();
        
        // Verify entity exists
        assert!(graph.get_entity(key).is_some());
        assert!(graph.contains_entity_key(key));
        
        // Remove entity
        let result = graph.remove_entity(key);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true for successful removal
        
        // Verify entity is removed
        assert!(graph.get_entity(key).is_none());
        assert!(!graph.contains_entity_key(key));
        assert!(graph.get_entity_by_id(1).is_none());
    }

    #[test]
    fn test_remove_entity_nonexistent() {
        let graph = create_test_graph();
        let fake_key = EntityKey::default();
        
        let result = graph.remove_entity(fake_key);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false for non-existent entity
    }

    #[test]
    fn test_remove_entity_cleanup() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(123, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(123, entity_data).unwrap();
        
        // Verify entity is in all data structures
        assert!(graph.get_entity(key).is_some());
        assert!(graph.get_entity_by_id(123).is_some());
        
        // Remove entity
        let result = graph.remove_entity(key);
        assert!(result.is_ok());
        assert!(result.unwrap());
        
        // Verify complete cleanup
        assert!(graph.get_entity(key).is_none());
        assert!(graph.get_entity_by_id(123).is_none());
        assert!(!graph.contains_entity_key(key));
        
        // Verify ID mapping is cleaned up
        let all_ids = graph.get_all_entity_ids();
        assert!(!all_ids.contains(&123));
    }

    #[test]
    fn test_get_entity_data_only() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, entity_data.clone()).unwrap();
        
        let retrieved_data = graph.get_entity_data(key);
        assert!(retrieved_data.is_some());
        
        let data = retrieved_data.unwrap();
        assert_eq!(data.type_id, entity_data.type_id);
        assert_eq!(data.embedding, entity_data.embedding);
    }

    #[test]
    fn test_get_entity_meta_only() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, entity_data).unwrap();
        
        let retrieved_meta = graph.get_entity_meta(key);
        assert!(retrieved_meta.is_some());
        
        let meta = retrieved_meta.unwrap();
        assert_eq!(meta.type_id, 1);
    }

    #[test]
    fn test_validate_entity_data() {
        let graph = create_test_graph();
        
        // Valid data
        let valid_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        let result = graph.validate_entity_data(&valid_data);
        assert!(result.is_ok());
        
        // Invalid embedding dimension
        let invalid_data = create_test_entity_data(1, vec![1.0, 2.0]); // Wrong dimension
        let result = graph.validate_entity_data(&invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_entity_embedding() {
        let graph = create_test_graph();
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let entity_data = create_test_entity_data(1, embedding.clone());
        
        let key = graph.insert_entity(1, entity_data).unwrap();
        
        let retrieved_embedding = graph.get_entity_embedding(key);
        assert!(retrieved_embedding.is_some());
        
        // Note: Due to quantization, the embedding might not be exactly the same
        // but should be close enough for practical purposes
        let retrieved = retrieved_embedding.unwrap();
        assert_eq!(retrieved.len(), embedding.len());
    }

    #[test]
    fn test_get_all_entity_ids() {
        let graph = create_test_graph();
        
        // Initially empty
        assert_eq!(graph.get_all_entity_ids().len(), 0);
        
        // Insert some entities
        let ids = vec![1, 2, 3];
        for id in &ids {
            let entity_data = create_test_entity_data(*id, vec![1.0, 2.0, 3.0, 4.0]);
            graph.insert_entity(*id, entity_data).unwrap();
        }
        
        let all_ids = graph.get_all_entity_ids();
        assert_eq!(all_ids.len(), 3);
        
        for id in ids {
            assert!(all_ids.contains(&id));
        }
    }

    #[test]
    fn test_get_all_entity_keys() {
        let graph = create_test_graph();
        
        // Initially empty
        assert_eq!(graph.get_all_entity_keys().len(), 0);
        
        // Insert some entities
        let mut inserted_keys = Vec::new();
        for id in 1..=3 {
            let entity_data = create_test_entity_data(id, vec![1.0, 2.0, 3.0, 4.0]);
            let key = graph.insert_entity(id, entity_data).unwrap();
            inserted_keys.push(key);
        }
        
        let all_keys = graph.get_all_entity_keys();
        assert_eq!(all_keys.len(), 3);
        
        for key in inserted_keys {
            assert!(all_keys.contains(&key));
        }
    }

    #[test]
    fn test_contains_entity_key() {
        let graph = create_test_graph();
        let entity_data = create_test_entity_data(1, vec![1.0, 2.0, 3.0, 4.0]);
        
        let key = graph.insert_entity(1, entity_data).unwrap();
        
        // Should contain the inserted entity
        assert!(graph.contains_entity_key(key));
        
        // Should not contain a fake key
        let fake_key = EntityKey::default();
        assert!(!graph.contains_entity_key(fake_key));
    }

    #[test]
    fn test_get_entity_stats() {
        let graph = create_test_graph();
        
        // Initially empty
        let stats = graph.get_entity_stats();
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.unique_entity_ids, 0);
        assert_eq!(stats.average_embedding_size(), 0.0);
        
        // Insert some entities
        for id in 1..=5 {
            let entity_data = create_test_entity_data(id, vec![1.0, 2.0, 3.0, 4.0]);
            graph.insert_entity(id, entity_data).unwrap();
        }
        
        let stats = graph.get_entity_stats();
        assert_eq!(stats.total_entities, 5);
        assert_eq!(stats.unique_entity_ids, 5);
        assert!(stats.average_embedding_size() > 0.0);
    }

    #[test]
    fn test_batch_vs_individual_insert_consistency() {
        let graph1 = create_test_graph();
        let graph2 = create_test_graph();
        
        let entities = vec![
            (1, create_test_entity_data(1, vec![1.0, 0.0, 0.0, 0.0])),
            (2, create_test_entity_data(2, vec![0.0, 1.0, 0.0, 0.0])),
            (3, create_test_entity_data(3, vec![0.0, 0.0, 1.0, 0.0])),
        ];
        
        // Insert individually
        let mut individual_keys = Vec::new();
        for (id, data) in &entities {
            let key = graph1.insert_entity(*id, data.clone()).unwrap();
            individual_keys.push(key);
        }
        
        // Insert as batch
        let batch_keys = graph2.insert_entities_batch(entities.clone()).unwrap();
        
        // Both should have same number of entities
        assert_eq!(graph1.entity_count(), graph2.entity_count());
        assert_eq!(individual_keys.len(), batch_keys.len());
        
        // Verify all entities can be retrieved from both graphs
        for (i, (id, original_data)) in entities.iter().enumerate() {
            let retrieved1 = graph1.get_entity_by_id(*id);
            let retrieved2 = graph2.get_entity_by_id(*id);
            
            assert!(retrieved1.is_some());
            assert!(retrieved2.is_some());
            
            let (_, data1) = retrieved1.unwrap();
            let (_, data2) = retrieved2.unwrap();
            
            assert_eq!(data1.type_id, original_data.type_id);
            assert_eq!(data2.type_id, original_data.type_id);
            assert_eq!(data1.embedding, original_data.embedding);
            assert_eq!(data2.embedding, original_data.embedding);
        }
    }

    #[test]
    fn test_entity_lifecycle() {
        let graph = create_test_graph();
        let initial_embedding = vec![1.0, 2.0, 3.0, 4.0];
        let entity_data = create_test_entity_data(1, initial_embedding.clone());
        
        // 1. Insert
        let key = graph.insert_entity(1, entity_data).unwrap();
        assert!(graph.contains_entity_key(key));
        assert_eq!(graph.entity_count(), 1);
        
        // 2. Read
        let retrieved = graph.get_entity(key);
        assert!(retrieved.is_some());
        let (_, data) = retrieved.unwrap();
        assert_eq!(data.embedding, initial_embedding);
        
        // 3. Update
        let updated_embedding = vec![5.0, 6.0, 7.0, 8.0];
        let updated_data = create_test_entity_data(1, updated_embedding.clone());
        let update_result = graph.update_entity(key, updated_data);
        assert!(update_result.is_ok());
        
        // Verify update
        let retrieved_updated = graph.get_entity(key);
        assert!(retrieved_updated.is_some());
        let (_, updated_data_retrieved) = retrieved_updated.unwrap();
        assert_eq!(updated_data_retrieved.embedding, updated_embedding);
        
        // 4. Delete
        let delete_result = graph.remove_entity(key);
        assert!(delete_result.is_ok());
        assert!(delete_result.unwrap());
        
        // Verify deletion
        assert!(!graph.contains_entity_key(key));
        assert_eq!(graph.entity_count(), 0);
        assert!(graph.get_entity(key).is_none());
    }

    #[test]
    fn test_concurrent_entity_operations() {
        use std::sync::Arc;
        use std::thread;
        
        let graph = Arc::new(create_test_graph());
        let mut handles = Vec::new();
        
        // Spawn multiple threads to insert entities concurrently
        for thread_id in 0..4 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let id = thread_id * 10 + i;
                    let entity_data = create_test_entity_data(id as u32, vec![
                        thread_id as f32, i as f32, 0.0, 1.0
                    ]);
                    
                    let result = graph_clone.insert_entity(id as u32, entity_data);
                    assert!(result.is_ok(), "Failed to insert entity {} from thread {}", i, thread_id);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all entities were inserted
        assert_eq!(graph.entity_count(), 40);
        
        // Verify we can retrieve all entities
        for thread_id in 0..4 {
            for i in 0..10 {
                let id = thread_id * 10 + i;
                let retrieved = graph.get_entity_by_id(id as u32);
                assert!(retrieved.is_some(), "Entity {} not found", id);
            }
        }
    }
}