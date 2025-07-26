// Integration test module for demonstrating real storage integration
// This shows how to use MMAP, HNSW, and quantization together

use std::sync::Arc;
use parking_lot::RwLock;
use crate::storage::{
    persistent_mmap::PersistentMMapStorage,
    hnsw::HnswIndex,
    quantized_index::QuantizedIndex,
    string_interner::StringInterner,
};
use crate::core::types::{EntityKey, EntityData};
use crate::error::Result;

/// Example of setting up integrated storage system
pub fn setup_integrated_storage(embedding_dim: usize) -> Result<IntegratedStorage> {
    // Create string interner for memory efficiency
    let string_interner = Arc::new(StringInterner::new());
    
    // Create MMAP storage for persistent zero-copy access
    let mmap_storage = Arc::new(PersistentMMapStorage::new(
        Some("llmkg_integrated.db"),
        embedding_dim
    )?);
    
    // Create HNSW index for fast similarity search
    let hnsw_index = Arc::new(RwLock::new(HnswIndex::new(embedding_dim)));
    
    // Create quantized index for memory-efficient storage
    let quantized_index = Arc::new(QuantizedIndex::new(embedding_dim, 8)?);
    
    Ok(IntegratedStorage {
        mmap_storage,
        hnsw_index,
        quantized_index,
        string_interner,
    })
}

pub struct IntegratedStorage {
    pub mmap_storage: Arc<PersistentMMapStorage>,
    pub hnsw_index: Arc<RwLock<HnswIndex>>,
    pub quantized_index: Arc<QuantizedIndex>,
    pub string_interner: Arc<StringInterner>,
}

impl IntegratedStorage {
    /// Store entity with all storage systems
    pub async fn store_entity(
        &self,
        entity_id: u32,
        name: &str,
        properties: &str,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        
        // Intern strings
        let interned_name = self.string_interner.intern(name);
        let interned_props = self.string_interner.intern(properties);
        
        // Create entity key and data
        let entity_key = EntityKey::from_raw(entity_id as u64);
        let entity_data = EntityData::new(
            interned_name.0 as u16,
            properties.to_string(),  // Use original string, not interned
            embedding.clone(),
        );
        
        // Store in MMAP for persistence
        self.mmap_storage.add_entity(entity_key, &entity_data, &embedding)?;
        
        // Add to HNSW for fast search
        let mut hnsw = self.hnsw_index.write();
        hnsw.insert(entity_id, entity_key, embedding.clone())?;
        drop(hnsw);
        
        // Add to quantized index if ready
        if self.quantized_index.is_ready() {
            self.quantized_index.insert(entity_id, entity_key, embedding)?;
        }
        
        let duration = start.elapsed();
        
        // Verify performance: <1ms total overhead
        if duration.as_micros() > 1000 {
            eprintln!("Warning: Entity storage took {}μs (target: <1000μs)", duration.as_micros());
        }
        
        Ok(())
    }
    
    /// Search for similar entities using HNSW
    pub async fn search_similar(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let start = std::time::Instant::now();
        
        let hnsw = self.hnsw_index.read();
        let results = hnsw.search(query_embedding, k);
        
        let duration = start.elapsed();
        
        // Verify performance: <10ms for search
        if duration.as_millis() > 10 {
            eprintln!("Warning: HNSW search took {}ms (target: <10ms)", duration.as_millis());
        }
        
        Ok(results)
    }
    
    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        let mmap_stats = self.mmap_storage.storage_stats();
        let hnsw = self.hnsw_index.read();
        let hnsw_stats = hnsw.stats();
        let interner_stats = self.string_interner.stats();
        
        StorageStats {
            total_entities: mmap_stats.entity_count,
            mmap_memory_bytes: mmap_stats.memory_usage_bytes,
            mmap_file_bytes: mmap_stats.file_size_bytes,
            hnsw_nodes: hnsw_stats.node_count,
            hnsw_layers: hnsw_stats.max_layer,
            string_interner_unique: interner_stats.unique_strings as usize,
            string_interner_saved_bytes: interner_stats.memory_saved_bytes as usize,
            compression_ratio: mmap_stats.compression_ratio,
        }
    }
    
    /// Train quantized index with sample embeddings
    pub async fn train_quantizer(&self, sample_embeddings: &[Vec<f32>]) -> Result<()> {
        self.quantized_index.train(sample_embeddings)?;
        Ok(())
    }
    
    /// Sync all data to disk
    pub async fn sync_to_disk(&self) -> Result<()> {
        self.mmap_storage.sync_to_disk()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct StorageStats {
    pub total_entities: usize,
    pub mmap_memory_bytes: u64,
    pub mmap_file_bytes: u64,
    pub hnsw_nodes: usize,
    pub hnsw_layers: usize,
    pub string_interner_unique: usize,
    pub string_interner_saved_bytes: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_integrated_storage() {
        // Create integrated storage
        let storage = setup_integrated_storage(128).unwrap();
        
        // Generate sample embeddings for training
        let sample_embeddings: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..128).map(|j| ((i + j) as f32).sin()).collect())
            .collect();
        
        // Train quantizer
        storage.train_quantizer(&sample_embeddings).await.unwrap();
        
        // Store entities
        for i in 0..10 {
            let embedding: Vec<f32> = (0..128).map(|j| ((i + j) as f32).cos()).collect();
            storage.store_entity(
                i as u32,
                &format!("Entity_{}", i),
                &format!("Properties for entity {}", i),
                embedding,
            ).await.unwrap();
        }
        
        // Search for similar entities
        let query: Vec<f32> = (0..128).map(|j| (j as f32 * 0.1).sin()).collect();
        let results = storage.search_similar(&query, 5).await.unwrap();
        
        assert_eq!(results.len(), 5);
        
        // Get statistics
        let stats = storage.get_stats();
        assert_eq!(stats.total_entities, 10);
        assert!(stats.compression_ratio > 1.0);
        
        // Sync to disk
        storage.sync_to_disk().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_performance_targets() {
        let storage = setup_integrated_storage(96).unwrap();
        
        // Train quantizer with minimal samples
        let sample_embeddings: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..96).map(|j| ((i * j) as f32 / 100.0)).collect())
            .collect();
        storage.train_quantizer(&sample_embeddings).await.unwrap();
        
        // Measure entity storage performance
        let start = std::time::Instant::now();
        for i in 0..100 {
            let embedding: Vec<f32> = (0..96).map(|j| ((i + j) as f32 / 100.0)).collect();
            storage.store_entity(
                i as u32,
                "test",
                "props",
                embedding,
            ).await.unwrap();
        }
        let storage_time = start.elapsed();
        let ms_per_entity = storage_time.as_millis() as f32 / 100.0;
        
        // Should be <1ms per entity
        assert!(ms_per_entity < 1.0, "Storage took {:.2}ms per entity", ms_per_entity);
        
        // Measure search performance
        let query: Vec<f32> = (0..96).map(|i| (i as f32 * 0.01)).collect();
        let start = std::time::Instant::now();
        let _results = storage.search_similar(&query, 10).await.unwrap();
        let search_time = start.elapsed();
        
        // Should be <10ms for search
        assert!(search_time.as_millis() < 10, "Search took {}ms", search_time.as_millis());
    }
}