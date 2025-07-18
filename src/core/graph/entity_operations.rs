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
        
        if removed {
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
                bloom.remove(&id);
                
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
        
        Ok(removed)
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
            let quantized_size = quantizer.encoded_size();
            
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