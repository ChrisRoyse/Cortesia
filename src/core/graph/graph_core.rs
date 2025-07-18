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