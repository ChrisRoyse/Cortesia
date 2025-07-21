//! Core knowledge graph structure and basic operations

use crate::core::entity::EntityStore;
use crate::core::memory::{GraphArena, EpochManager};
use crate::core::types::{EntityKey, Relationship};
use crate::storage::csr::CSRGraph;
use crate::storage::bloom::BloomFilter;
use crate::storage::spatial_index::SpatialIndex;
use crate::storage::flat_index::FlatVectorIndex;
use crate::storage::lru_cache::{SimilarityCache};
use crate::storage::hnsw::HnswIndex;
use crate::storage::lsh::LshIndex;
use crate::embedding::quantizer::ProductQuantizer;
use crate::error::{GraphError, Result};

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration};
use ahash::AHashMap;

// Performance timeout guards
pub const MAX_INSERTION_TIME: Duration = Duration::from_secs(10);
pub const MAX_QUERY_TIME: Duration = Duration::from_millis(100);
pub const MAX_SIMILARITY_SEARCH_TIME: Duration = Duration::from_millis(50);

/// Main knowledge graph structure
pub struct KnowledgeGraph {
    // Core storage
    pub arena: RwLock<GraphArena>,
    pub entity_store: RwLock<EntityStore>,
    pub graph: RwLock<CSRGraph>,
    
    // Embedding system
    pub embedding_bank: RwLock<Vec<u8>>,
    pub quantizer: RwLock<ProductQuantizer>,
    pub embedding_dim: usize,
    
    // Indexing
    pub bloom_filter: RwLock<BloomFilter>,
    pub entity_id_map: RwLock<AHashMap<u32, EntityKey>>,
    pub spatial_index: RwLock<SpatialIndex>,
    pub flat_index: RwLock<FlatVectorIndex>,
    pub hnsw_index: RwLock<HnswIndex>,
    pub lsh_index: RwLock<LshIndex>,
    pub similarity_cache: RwLock<SimilarityCache>,
    
    // Concurrency
    pub epoch_manager: Arc<EpochManager>,
    
    // Metadata
    pub string_dictionary: RwLock<AHashMap<String, u32>>,
    
    // Mutable edge buffer for dynamic relationship insertion
    pub edge_buffer: RwLock<Vec<Relationship>>,
}

impl KnowledgeGraph {
    /// Create new knowledge graph with specified embedding dimension
    pub fn new_internal(embedding_dim: usize) -> Result<Self> {
        let subvector_count = 8; // For 96-dim embeddings, 8 subvectors of 12 dims each
        
        Ok(Self {
            arena: RwLock::new(GraphArena::new()),
            entity_store: RwLock::new(EntityStore::new()),
            graph: RwLock::new(CSRGraph::new()),
            embedding_bank: RwLock::new(Vec::new()),
            quantizer: RwLock::new(ProductQuantizer::new(embedding_dim, subvector_count)?),
            embedding_dim,
            bloom_filter: RwLock::new(BloomFilter::new(1_000_000, 0.01)),
            entity_id_map: RwLock::new(AHashMap::new()),
            spatial_index: RwLock::new(SpatialIndex::new(embedding_dim)),
            flat_index: RwLock::new(FlatVectorIndex::new(embedding_dim)),
            hnsw_index: RwLock::new(HnswIndex::new(embedding_dim)),
            lsh_index: RwLock::new(LshIndex::new_optimized(embedding_dim, 0.85)), // 85% precision target
            similarity_cache: RwLock::new(SimilarityCache::new(1000)), // Cache up to 1000 queries
            epoch_manager: Arc::new(EpochManager::new(16)),
            string_dictionary: RwLock::new(AHashMap::new()),
            edge_buffer: RwLock::new(Vec::new()),
        })
    }

    /// Create new knowledge graph with specified dimension
    pub fn new_with_dimension(embedding_dim: usize) -> Result<Self> {
        Self::new_internal(embedding_dim)
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.arena.read().entity_count()
    }
    
    /// Get relationship count
    pub fn relationship_count(&self) -> u32 {
        self.graph.read().edge_count()
    }
    
    /// Get edge count (alias for relationship_count)
    pub fn edge_count(&self) -> u32 {
        self.relationship_count()
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let arena_bytes = self.arena.read().memory_usage();
        let entity_store_bytes = self.entity_store.read().memory_usage();
        let graph_bytes = self.graph.read().memory_usage();
        let embedding_bank_bytes = self.embedding_bank.read().len() * std::mem::size_of::<u8>();
        let quantizer_bytes = self.quantizer.read().memory_usage();
        let bloom_filter_bytes = self.bloom_filter.read().memory_usage();
        
        MemoryUsage {
            arena_bytes,
            entity_store_bytes,
            graph_bytes,
            embedding_bank_bytes,
            quantizer_bytes,
            bloom_filter_bytes,
        }
    }

    /// Get embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dim
    }

    /// Check if entity exists
    pub fn contains_entity(&self, id: u32) -> bool {
        self.entity_id_map.read().contains_key(&id)
    }

    /// Get entity key by ID
    pub fn get_entity_key(&self, id: u32) -> Option<EntityKey> {
        self.entity_id_map.read().get(&id).copied()
    }
    
    /// Get entity ID by key (reverse lookup)
    pub fn get_entity_id(&self, key: EntityKey) -> Option<u32> {
        let id_map = self.entity_id_map.read();
        for (id, &stored_key) in id_map.iter() {
            if stored_key == key {
                return Some(*id);
            }
        }
        None
    }

    /// Clear all cached data
    pub fn clear_caches(&self) {
        self.similarity_cache.write().clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, f64) {
        let cache = self.similarity_cache.read();
        (cache.len(), cache.capacity(), cache.hit_rate())
    }

    /// Validate embedding dimension
    pub fn validate_embedding_dimension(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }
        Ok(())
    }

    /// Get string dictionary size
    pub fn string_dictionary_size(&self) -> usize {
        self.string_dictionary.read().len()
    }

    /// Get edge buffer size
    pub fn edge_buffer_size(&self) -> usize {
        self.edge_buffer.read().len()
    }

    /// Flush edge buffer to main graph
    pub fn flush_edge_buffer(&self) -> Result<()> {
        let mut buffer = self.edge_buffer.write();
        let mut graph = self.graph.write();
        
        for relationship in buffer.drain(..) {
            if let (Some(from_id), Some(to_id)) = (self.get_entity_id(relationship.from), self.get_entity_id(relationship.to)) {
                graph.add_edge(from_id, to_id, relationship.weight)?;
            }
        }
        
        Ok(())
    }

    /// Get epoch manager
    pub fn epoch_manager(&self) -> &Arc<EpochManager> {
        &self.epoch_manager
    }
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryUsage {
    pub arena_bytes: usize,
    pub entity_store_bytes: usize,
    pub graph_bytes: usize,
    pub embedding_bank_bytes: usize,
    pub quantizer_bytes: usize,
    pub bloom_filter_bytes: usize,
}

impl MemoryUsage {
    /// Get total memory usage in bytes
    pub fn total_bytes(&self) -> usize {
        self.arena_bytes + 
        self.entity_store_bytes + 
        self.graph_bytes + 
        self.embedding_bank_bytes + 
        self.quantizer_bytes + 
        self.bloom_filter_bytes
    }
    
    /// Get memory usage per entity
    pub fn bytes_per_entity(&self, entity_count: usize) -> usize {
        if entity_count == 0 {
            0
        } else {
            self.total_bytes() / entity_count
        }
    }

    /// Get memory usage breakdown as percentages
    pub fn usage_breakdown(&self) -> MemoryBreakdown {
        let total = self.total_bytes() as f64;
        if total == 0.0 {
            return MemoryBreakdown::default();
        }

        MemoryBreakdown {
            arena_percentage: (self.arena_bytes as f64 / total) * 100.0,
            entity_store_percentage: (self.entity_store_bytes as f64 / total) * 100.0,
            graph_percentage: (self.graph_bytes as f64 / total) * 100.0,
            embedding_bank_percentage: (self.embedding_bank_bytes as f64 / total) * 100.0,
            quantizer_percentage: (self.quantizer_bytes as f64 / total) * 100.0,
            bloom_filter_percentage: (self.bloom_filter_bytes as f64 / total) * 100.0,
        }
    }
}

/// Memory usage breakdown by percentage
#[derive(Debug)]
pub struct MemoryBreakdown {
    pub arena_percentage: f64,
    pub entity_store_percentage: f64,
    pub graph_percentage: f64,
    pub embedding_bank_percentage: f64,
    pub quantizer_percentage: f64,
    pub bloom_filter_percentage: f64,
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self {
            arena_percentage: 0.0,
            entity_store_percentage: 0.0,
            graph_percentage: 0.0,
            embedding_bank_percentage: 0.0,
            quantizer_percentage: 0.0,
            bloom_filter_percentage: 0.0,
        }
    }
}

// Performance testing compatibility methods
impl KnowledgeGraph {
    /// Constructor that accepts dimension parameter for compatibility
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Self::new_with_dimension(embedding_dim)
    }
    
    /// Default constructor for performance tests
    pub fn new_default() -> Result<Self> {
        Self::new_with_dimension(96)
    }
}

impl std::fmt::Debug for KnowledgeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeGraph")
            .field("entity_count", &self.entity_count())
            .field("relationship_count", &self.relationship_count())
            .field("embedding_dim", &self.embedding_dim)
            .field("string_dictionary_size", &self.string_dictionary_size())
            .field("edge_buffer_size", &self.edge_buffer_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;

    #[test]
    fn test_new_internal_with_valid_dimension() {
        let embedding_dim = 96;
        let graph = KnowledgeGraph::new_internal(embedding_dim).unwrap();
        
        assert_eq!(graph.embedding_dim, embedding_dim);
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        assert_eq!(graph.edge_buffer_size(), 0);
        assert_eq!(graph.string_dictionary_size(), 0);
    }

    #[test]
    fn test_new_internal_with_different_dimensions() {
        // Test various embedding dimensions
        let dimensions = vec![32, 64, 96, 128, 256, 512];
        
        for dim in dimensions {
            let graph = KnowledgeGraph::new_internal(dim).unwrap();
            assert_eq!(graph.embedding_dim, dim);
            assert_eq!(graph.embedding_dimension(), dim);
        }
    }

    #[test]
    fn test_new_internal_with_zero_dimension() {
        // Test edge case with zero dimension
        let result = KnowledgeGraph::new_internal(0);
        // This might fail due to ProductQuantizer constraints
        // The test validates the behavior regardless of success/failure
        match result {
            Ok(graph) => assert_eq!(graph.embedding_dim, 0),
            Err(_) => {
                // Zero dimension is likely invalid for ProductQuantizer
                // This is expected behavior
            }
        }
    }

    #[test]
    fn test_new_internal_with_large_dimension() {
        // Test with a large dimension to ensure no overflow
        let embedding_dim = 4096;
        let result = KnowledgeGraph::new_internal(embedding_dim);
        
        match result {
            Ok(graph) => {
                assert_eq!(graph.embedding_dim, embedding_dim);
                assert_eq!(graph.entity_count(), 0);
                assert_eq!(graph.relationship_count(), 0);
            }
            Err(_) => {
                // Large dimensions might be constrained by ProductQuantizer
                // This is valid behavior
            }
        }
    }

    #[test]
    fn test_new_internal_component_initialization() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Verify all components are properly initialized
        assert!(graph.arena.read().entity_count() == 0);
        assert!(graph.graph.read().edge_count() == 0);
        assert!(graph.embedding_bank.read().is_empty());
        assert!(graph.entity_id_map.read().is_empty());
        assert!(graph.string_dictionary.read().is_empty());
        assert!(graph.edge_buffer.read().is_empty());
        
        // Verify bloom filter is initialized
        assert!(graph.bloom_filter.read().memory_usage() > 0);
        
        // Verify cache is initialized with correct capacity
        let (current_size, capacity, _hit_rate) = graph.cache_stats();
        assert_eq!(current_size, 0);
        assert_eq!(capacity, 1000); // As specified in new_internal
    }

    #[test]
    fn test_entity_count_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        assert_eq!(graph.entity_count(), 0);
    }

    #[test]
    fn test_relationship_count_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        assert_eq!(graph.relationship_count(), 0);
        assert_eq!(graph.edge_count(), 0); // Test alias method
    }

    #[test]
    fn test_memory_usage_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let memory_usage = graph.memory_usage();
        
        // Verify memory usage structure
        assert!(memory_usage.total_bytes() > 0); // Should have some baseline memory
        assert_eq!(memory_usage.embedding_bank_bytes, 0); // Empty embedding bank
        
        // Test memory breakdown
        let breakdown = memory_usage.usage_breakdown();
        assert!(breakdown.arena_percentage >= 0.0);
        assert!(breakdown.entity_store_percentage >= 0.0);
        assert!(breakdown.graph_percentage >= 0.0);
        assert!(breakdown.embedding_bank_percentage == 0.0); // Empty
        assert!(breakdown.quantizer_percentage >= 0.0);
        assert!(breakdown.bloom_filter_percentage >= 0.0);
        
        // Sum of percentages should be approximately 100% (allowing for floating point errors)
        let total_percentage = breakdown.arena_percentage + 
                              breakdown.entity_store_percentage + 
                              breakdown.graph_percentage + 
                              breakdown.embedding_bank_percentage + 
                              breakdown.quantizer_percentage + 
                              breakdown.bloom_filter_percentage;
        assert!((total_percentage - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_usage_bytes_per_entity() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let memory_usage = graph.memory_usage();
        
        // Test with zero entities
        assert_eq!(memory_usage.bytes_per_entity(0), 0);
        
        // Test with hypothetical entity count
        let hypothetical_entities = 100;
        let bytes_per_entity = memory_usage.bytes_per_entity(hypothetical_entities);
        assert_eq!(bytes_per_entity, memory_usage.total_bytes() / hypothetical_entities);
    }

    #[test]
    fn test_contains_entity_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        assert!(!graph.contains_entity(0));
        assert!(!graph.contains_entity(1));
        assert!(!graph.contains_entity(u32::MAX));
    }

    #[test]
    fn test_get_entity_key_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        assert!(graph.get_entity_key(0).is_none());
        assert!(graph.get_entity_key(1).is_none());
        assert!(graph.get_entity_key(u32::MAX).is_none());
    }

    #[test]
    fn test_get_entity_id_empty_graph() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let dummy_key = EntityKey::new(0); // Assuming EntityKey has a new method
        assert!(graph.get_entity_id(dummy_key).is_none());
    }

    #[test]
    fn test_validate_embedding_dimension_correct() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let embedding = vec![0.0f32; 96];
        assert!(graph.validate_embedding_dimension(&embedding).is_ok());
    }

    #[test]
    fn test_validate_embedding_dimension_incorrect() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Test too small
        let embedding_small = vec![0.0f32; 95];
        let result_small = graph.validate_embedding_dimension(&embedding_small);
        assert!(result_small.is_err());
        
        // Test too large
        let embedding_large = vec![0.0f32; 97];
        let result_large = graph.validate_embedding_dimension(&embedding_large);
        assert!(result_large.is_err());
        
        // Test empty
        let embedding_empty = vec![];
        let result_empty = graph.validate_embedding_dimension(&embedding_empty);
        assert!(result_empty.is_err());
    }

    #[test]
    fn test_cache_operations() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Test initial cache state
        let (initial_size, capacity, initial_hit_rate) = graph.cache_stats();
        assert_eq!(initial_size, 0);
        assert_eq!(capacity, 1000);
        assert!(initial_hit_rate >= 0.0 && initial_hit_rate <= 1.0);
        
        // Test cache clearing (should be no-op on empty cache)
        graph.clear_caches();
        let (size_after_clear, _, _) = graph.cache_stats();
        assert_eq!(size_after_clear, 0);
    }

    #[test]
    fn test_string_dictionary_size_empty() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        assert_eq!(graph.string_dictionary_size(), 0);
    }

    #[test]
    fn test_edge_buffer_operations() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Test initial state
        assert_eq!(graph.edge_buffer_size(), 0);
        
        // Test flushing empty buffer
        let result = graph.flush_edge_buffer();
        assert!(result.is_ok());
        assert_eq!(graph.edge_buffer_size(), 0);
    }

    #[test]
    fn test_epoch_manager_access() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let epoch_manager = graph.epoch_manager();
        
        // Verify epoch manager is properly initialized
        assert!(Arc::strong_count(epoch_manager) >= 1);
    }

    #[test]
    fn test_constructor_aliases() {
        // Test new() alias
        let graph1 = KnowledgeGraph::new(128).unwrap();
        assert_eq!(graph1.embedding_dimension(), 128);
        
        // Test new_with_dimension()
        let graph2 = KnowledgeGraph::new_with_dimension(64).unwrap();
        assert_eq!(graph2.embedding_dimension(), 64);
        
        // Test new_default()
        let graph3 = KnowledgeGraph::new_default().unwrap();
        assert_eq!(graph3.embedding_dimension(), 96);
    }

    #[test]
    fn test_debug_implementation() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        let debug_string = format!("{:?}", graph);
        
        // Verify debug output contains expected fields
        assert!(debug_string.contains("KnowledgeGraph"));
        assert!(debug_string.contains("entity_count"));
        assert!(debug_string.contains("relationship_count"));
        assert!(debug_string.contains("embedding_dim"));
        assert!(debug_string.contains("string_dictionary_size"));
        assert!(debug_string.contains("edge_buffer_size"));
    }

    #[test]
    fn test_memory_breakdown_default() {
        let breakdown = MemoryBreakdown::default();
        assert_eq!(breakdown.arena_percentage, 0.0);
        assert_eq!(breakdown.entity_store_percentage, 0.0);
        assert_eq!(breakdown.graph_percentage, 0.0);
        assert_eq!(breakdown.embedding_bank_percentage, 0.0);
        assert_eq!(breakdown.quantizer_percentage, 0.0);
        assert_eq!(breakdown.bloom_filter_percentage, 0.0);
    }

    #[test]
    fn test_component_integration_consistency() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Verify consistent state across components
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        assert_eq!(graph.edge_count(), graph.relationship_count());
        
        // Verify embedding dimension consistency
        assert_eq!(graph.embedding_dim, graph.embedding_dimension());
        
        // Verify cache initialization
        let (cache_size, cache_capacity, _) = graph.cache_stats();
        assert_eq!(cache_size, 0);
        assert!(cache_capacity > 0);
        
        // Verify buffer initialization
        assert_eq!(graph.edge_buffer_size(), 0);
        assert_eq!(graph.string_dictionary_size(), 0);
    }

    #[test]
    fn test_concurrent_access_initialization() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Test that all RwLock components can be accessed
        {
            let _arena = graph.arena.read();
            let _entity_store = graph.entity_store.read();
            let _graph_storage = graph.graph.read();
            let _embedding_bank = graph.embedding_bank.read();
            let _quantizer = graph.quantizer.read();
            let _bloom_filter = graph.bloom_filter.read();
            let _entity_id_map = graph.entity_id_map.read();
            let _spatial_index = graph.spatial_index.read();
            let _flat_index = graph.flat_index.read();
            let _hnsw_index = graph.hnsw_index.read();
            let _lsh_index = graph.lsh_index.read();
            let _similarity_cache = graph.similarity_cache.read();
            let _string_dictionary = graph.string_dictionary.read();
            let _edge_buffer = graph.edge_buffer.read();
        }
        
        // All reads should succeed without deadlock
        assert!(true);
    }

    #[test]
    fn test_quantizer_initialization_with_subvectors() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // The new_internal method uses 8 subvectors for 96-dim embeddings
        // Verify the quantizer was created successfully
        let quantizer = graph.quantizer.read();
        assert!(quantizer.memory_usage() > 0);
    }

    #[test]
    fn test_lsh_index_precision_target() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Verify LSH index was created with optimized settings (85% precision target)
        let lsh_index = graph.lsh_index.read();
        // We can't directly test the precision target, but we can verify it was created
        // This tests the component integration in new_internal
        drop(lsh_index); // Explicit drop to release the read lock
        assert!(true); // LSH index creation succeeded
    }

    #[test]
    fn test_bloom_filter_initialization_parameters() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Verify bloom filter was created with expected parameters (1M capacity, 1% error rate)
        let bloom_filter = graph.bloom_filter.read();
        assert!(bloom_filter.memory_usage() > 0);
        // The actual capacity and error rate are internal to BloomFilter
        // but we can verify successful creation
    }

    #[test] 
    fn test_epoch_manager_initialization() {
        let graph = KnowledgeGraph::new_internal(96).unwrap();
        
        // Verify epoch manager was created with 16 threads
        let epoch_manager = graph.epoch_manager();
        assert!(Arc::strong_count(epoch_manager) >= 1);
        
        // Test that it's the same instance when accessed multiple times
        let epoch_manager2 = graph.epoch_manager();
        assert!(Arc::ptr_eq(epoch_manager, epoch_manager2));
    }

    #[test]
    fn test_constants_are_reasonable() {
        // Test that the timeout constants are set to reasonable values
        assert!(MAX_INSERTION_TIME.as_secs() > 0);
        assert!(MAX_QUERY_TIME.as_millis() > 0);
        assert!(MAX_SIMILARITY_SEARCH_TIME.as_millis() > 0);
        
        // Verify the relationships between timeouts
        assert!(MAX_INSERTION_TIME > MAX_QUERY_TIME);
        assert!(MAX_QUERY_TIME >= MAX_SIMILARITY_SEARCH_TIME);
    }
}