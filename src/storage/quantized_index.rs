use crate::core::types::EntityKey;
use crate::error::{GraphError, Result};
use crate::embedding::quantizer::ProductQuantizer;
use std::collections::HashMap;
use parking_lot::RwLock;

/// Quantized Vector Index for memory-efficient similarity search
/// Uses Product Quantization to compress embeddings while maintaining search quality
pub struct QuantizedIndex {
    /// Product quantizer for compression
    quantizer: RwLock<ProductQuantizer>,
    /// Compressed embeddings storage
    compressed_embeddings: RwLock<Vec<u8>>,
    /// Entity metadata
    entities: RwLock<Vec<QuantizedEntity>>,
    /// Mapping from entity ID to index in entities array
    entity_id_map: RwLock<HashMap<u32, usize>>,
    /// Vector dimension
    dimension: usize,
    /// Whether the quantizer has been trained
    is_ready: RwLock<bool>,
}

#[derive(Clone)]
struct QuantizedEntity {
    entity_id: u32,
    entity_key: EntityKey,
    compressed_offset: usize,
    compressed_size: usize,
}

impl QuantizedIndex {
    pub fn new(dimension: usize, subvector_count: usize) -> Result<Self> {
        let quantizer = ProductQuantizer::new(dimension, subvector_count)?;
        
        Ok(Self {
            quantizer: RwLock::new(quantizer),
            compressed_embeddings: RwLock::new(Vec::new()),
            entities: RwLock::new(Vec::new()),
            entity_id_map: RwLock::new(HashMap::new()),
            dimension,
            is_ready: RwLock::new(false),
        })
    }

    /// Train the quantizer with a set of representative embeddings
    pub fn train(&self, training_embeddings: &[Vec<f32>]) -> Result<()> {
        if training_embeddings.is_empty() {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }

        let mut quantizer = self.quantizer.write();
        quantizer.train_adaptive(training_embeddings)?;
        drop(quantizer);

        *self.is_ready.write() = true;
        Ok(())
    }

    /// Insert a new entity with its embedding
    pub fn insert(&self, entity_id: u32, entity_key: EntityKey, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        if !*self.is_ready.read() {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }

        // Encode the embedding
        let quantizer = self.quantizer.read();
        let compressed = quantizer.encode(&embedding)?;
        drop(quantizer);

        // Store compressed embedding
        let mut compressed_embeddings = self.compressed_embeddings.write();
        let offset = compressed_embeddings.len();
        compressed_embeddings.extend_from_slice(&compressed);
        drop(compressed_embeddings);

        // Create entity metadata
        let entity = QuantizedEntity {
            entity_id,
            entity_key,
            compressed_offset: offset,
            compressed_size: compressed.len(),
        };

        // Store entity
        let mut entities = self.entities.write();
        let entity_index = entities.len();
        entities.push(entity);
        drop(entities);

        // Update ID mapping
        let mut entity_id_map = self.entity_id_map.write();
        entity_id_map.insert(entity_id, entity_index);

        Ok(())
    }

    /// Bulk insert entities for better performance
    pub fn bulk_insert(&self, entities: Vec<(u32, EntityKey, Vec<f32>)>) -> Result<()> {
        if !*self.is_ready.read() {
            return Err(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 });
        }

        let mut compressed_embeddings = self.compressed_embeddings.write();
        let mut entity_vec = self.entities.write();
        let mut entity_id_map = self.entity_id_map.write();
        let quantizer = self.quantizer.read();

        for (entity_id, entity_key, embedding) in entities {
            if embedding.len() != self.dimension {
                return Err(GraphError::InvalidEmbeddingDimension {
                    expected: self.dimension,
                    actual: embedding.len(),
                });
            }

            // Encode the embedding
            let compressed = quantizer.encode(&embedding)?;

            // Store compressed embedding
            let offset = compressed_embeddings.len();
            compressed_embeddings.extend_from_slice(&compressed);

            // Create and store entity metadata
            let entity = QuantizedEntity {
                entity_id,
                entity_key,
                compressed_offset: offset,
                compressed_size: compressed.len(),
            };

            let entity_index = entity_vec.len();
            entity_vec.push(entity);
            entity_id_map.insert(entity_id, entity_index);
        }

        Ok(())
    }

    /// Search for k-nearest neighbors using asymmetric distance computation
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if !*self.is_ready.read() {
            return Ok(Vec::new());
        }

        let entities = self.entities.read();
        let compressed_embeddings = self.compressed_embeddings.read();
        let quantizer = self.quantizer.read();

        let mut distances = Vec::with_capacity(entities.len());

        // Compute asymmetric distances to all entities
        for entity in entities.iter() {
            let compressed_codes = &compressed_embeddings[
                entity.compressed_offset..entity.compressed_offset + entity.compressed_size
            ];

            let distance = quantizer.asymmetric_distance(query, compressed_codes)?;
            distances.push((entity.entity_id, distance));
        }

        // Sort by distance and return top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Search with a distance threshold
    pub fn search_threshold(&self, query: &[f32], threshold: f32) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.dimension {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        if !*self.is_ready.read() {
            return Ok(Vec::new());
        }

        let entities = self.entities.read();
        let compressed_embeddings = self.compressed_embeddings.read();
        let quantizer = self.quantizer.read();

        let mut results = Vec::new();

        // Compute distances and filter by threshold
        for entity in entities.iter() {
            let compressed_codes = &compressed_embeddings[
                entity.compressed_offset..entity.compressed_offset + entity.compressed_size
            ];

            let distance = quantizer.asymmetric_distance(query, compressed_codes)?;
            if distance <= threshold {
                results.push((entity.entity_id, distance));
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> QuantizedIndexStats {
        let entities = self.entities.read();
        let compressed_embeddings = self.compressed_embeddings.read();
        let quantizer = self.quantizer.read();

        let entity_count = entities.len();
        let compressed_size = compressed_embeddings.len();
        let codebook_size = quantizer.memory_usage();
        let metadata_size = entities.capacity() * std::mem::size_of::<QuantizedEntity>();
        let total_size = compressed_size + codebook_size + metadata_size;

        let original_size = entity_count * self.dimension * std::mem::size_of::<f32>();
        let compression_ratio = if total_size > 0 {
            original_size as f32 / total_size as f32
        } else {
            0.0
        };

        QuantizedIndexStats {
            entity_count,
            compressed_embeddings_bytes: compressed_size,
            codebook_bytes: codebook_size,
            metadata_bytes: metadata_size,
            total_bytes: total_size,
            original_size_bytes: original_size,
            compression_ratio,
            bytes_per_entity: if entity_count > 0 { total_size / entity_count } else { 0 },
            is_trained: quantizer.is_trained(),
            training_quality: quantizer.training_quality(),
        }
    }

    /// Check if the index is ready for use
    pub fn is_ready(&self) -> bool {
        *self.is_ready.read()
    }

    /// Get the number of entities in the index
    pub fn len(&self) -> usize {
        self.entities.read().len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.entities.read().is_empty()
    }

    /// Clear all data in the index
    pub fn clear(&self) {
        self.compressed_embeddings.write().clear();
        self.entities.write().clear();
        self.entity_id_map.write().clear();
        *self.is_ready.write() = false;
    }

    /// Reconstruct an embedding from its compressed form (for testing)
    pub fn reconstruct_embedding(&self, entity_id: u32) -> Result<Vec<f32>> {
        let entity_id_map = self.entity_id_map.read();
        let entity_index = entity_id_map.get(&entity_id)
            .ok_or(GraphError::InvalidEmbeddingDimension { expected: 1, actual: 0 })?;

        let entities = self.entities.read();
        let entity = &entities[*entity_index];

        let compressed_embeddings = self.compressed_embeddings.read();
        let compressed_codes = &compressed_embeddings[
            entity.compressed_offset..entity.compressed_offset + entity.compressed_size
        ];

        let quantizer = self.quantizer.read();
        quantizer.decode(compressed_codes)
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedIndexStats {
    pub entity_count: usize,
    pub compressed_embeddings_bytes: usize,
    pub codebook_bytes: usize,
    pub metadata_bytes: usize,
    pub total_bytes: usize,
    pub original_size_bytes: usize,
    pub compression_ratio: f32,
    pub bytes_per_entity: usize,
    pub is_trained: bool,
    pub training_quality: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_quantized_index_creation() {
        let index = QuantizedIndex::new(96, 8).unwrap();
        assert!(index.is_empty());
        assert!(!index.is_ready());
    }

    #[test]
    fn test_quantized_index_training_and_insertion() {
        let index = QuantizedIndex::new(12, 4).unwrap();
        
        // Generate training data
        let training_data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..12).map(|j| (i + j) as f32 / 100.0).collect()
            })
            .collect();

        // Train the quantizer
        index.train(&training_data).unwrap();
        assert!(index.is_ready());

        // Insert an entity
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        index.insert(1, EntityKey::from_u32(1), embedding).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_quantized_search() {
        let index = QuantizedIndex::new(8, 4).unwrap();
        
        // Generate and train on data
        let training_data: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..8).map(|j| ((i + j) as f32).sin()).collect()
            })
            .collect();

        index.train(&training_data).unwrap();

        // Insert test entities
        let entities = vec![
            (1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        for (id, embedding) in entities {
            index.insert(id, EntityKey::from_u32(id), embedding).unwrap();
        }

        // Search for similar vectors
        let query = vec![0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // Entity 3 should be closest, followed by entity 1
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_compression_ratio() {
        let index = QuantizedIndex::new(96, 8).unwrap();
        
        // Train with random data
        let training_data: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                (0..96).map(|j| ((i * j) as f32).sin()).collect()
            })
            .collect();

        index.train(&training_data).unwrap();

        // Insert entities
        for i in 0..100 {
            let embedding: Vec<f32> = (0..96).map(|j| ((i + j) as f32) / 100.0).collect();
            index.insert(i as u32, EntityKey::from_u32(i as u32), embedding).unwrap();
        }

        let stats = index.memory_usage();
        
        // Should achieve significant compression
        assert!(stats.compression_ratio > 2.0);
        assert!(stats.bytes_per_entity < 200); // Much less than 96 * 4 = 384 bytes
        assert_eq!(stats.entity_count, 100);
    }
}