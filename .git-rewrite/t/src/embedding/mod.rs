pub mod quantizer;
pub mod store;
pub mod store_compat;
pub mod similarity;
pub mod simd_search;

// Additional types for streaming and performance monitoring
pub use store::EmbeddingStore;
pub use store_compat::EmbeddingStore as CompatEmbeddingStore;

/// Filtered embedding for optimized queries
#[derive(Debug, Clone)]
pub struct FilteredEmbedding {
    pub embedding: Vec<f32>,
    pub entity_id: u32,
    pub confidence: f32,
    pub filter_score: f32,
}

/// Entity embedding with metadata
#[derive(Debug, Clone)]
pub struct EntityEmbedding {
    pub entity_id: u32,
    pub embedding: Vec<f32>,
    pub timestamp: u64,
}

/// Embedding vector type alias
pub type EmbeddingVector = Vec<f32>;

/// Result type for throughput measurements
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    pub operations_per_second: f64,
    pub total_operations: u64,
    pub duration_ms: u128,
    pub memory_usage_mb: f64,
}

impl FilteredEmbedding {
    pub fn new(embedding: Vec<f32>, entity_id: u32, confidence: f32, filter_score: f32) -> Self {
        Self {
            embedding,
            entity_id,
            confidence,
            filter_score,
        }
    }
}

impl EntityEmbedding {
    pub fn new(entity_id: u32, embedding: Vec<f32>) -> Self {
        Self {
            entity_id,
            embedding,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    pub fn to_vector(&self) -> EmbeddingVector {
        self.embedding.clone()
    }
}

impl ThroughputResult {
    pub fn new(operations_per_second: f64, total_operations: u64, duration_ms: u128) -> Self {
        Self {
            operations_per_second,
            total_operations,
            duration_ms,
            memory_usage_mb: 0.0,
        }
    }
}

// Conversion helper functions to fix type mismatches
impl From<EntityEmbedding> for EmbeddingVector {
    fn from(entity_embedding: EntityEmbedding) -> Self {
        entity_embedding.embedding
    }
}

impl From<&EntityEmbedding> for EmbeddingVector {
    fn from(entity_embedding: &EntityEmbedding) -> Self {
        entity_embedding.embedding.clone()
    }
}

/// Helper function to convert Vec<EntityEmbedding> to Vec<EmbeddingVector>
pub fn entity_embeddings_to_vectors(entity_embeddings: Vec<EntityEmbedding>) -> Vec<EmbeddingVector> {
    entity_embeddings.into_iter().map(|e| e.embedding).collect()
}

/// Helper function to convert &[EntityEmbedding] to Vec<EmbeddingVector>
pub fn entity_embeddings_slice_to_vectors(entity_embeddings: &[EntityEmbedding]) -> Vec<EmbeddingVector> {
    entity_embeddings.iter().map(|e| e.embedding.clone()).collect()
}