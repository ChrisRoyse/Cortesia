pub mod update_handler;
pub mod incremental_indexing;
pub mod temporal_updates;

pub use update_handler::{
    StreamingUpdateHandler,
    UpdateStream,
    UpdateBatch,
    UpdateResult,
    ConflictResolver,
    ConflictResolution,
    UpdateStats,
};

pub use incremental_indexing::{
    IncrementalIndexer,
    IndexUpdate,
    IndexUpdateResult,
    BloomFilterUpdater,
    CSRUpdater,
    EmbeddingUpdater,
};

use crate::core::types::Relationship;
use crate::error::Result;
use async_trait::async_trait;

/// Trait for streaming edge operations
#[async_trait]
pub trait EdgeStreamer {
    /// Stream edges from the graph in batches
    async fn stream_edges(&self, batch_size: usize) -> Result<Vec<Relationship>>;
    
    /// Stream edges matching a filter
    async fn stream_filtered_edges<F>(&self, filter: F, batch_size: usize) -> Result<Vec<Relationship>>
    where
        F: Fn(&Relationship) -> bool + Send + Sync;
    
    /// Get total edge count for streaming
    async fn total_edge_count(&self) -> Result<usize>;
    
    /// Reset streaming position
    async fn reset_stream(&mut self) -> Result<()>;
}

/// Simple implementation of EdgeStreamer for basic graph structures
pub struct SimpleEdgeStreamer {
    edges: Vec<Relationship>,
    current_position: usize,
}

impl SimpleEdgeStreamer {
    pub fn new(edges: Vec<Relationship>) -> Self {
        Self {
            edges,
            current_position: 0,
        }
    }
}

#[async_trait]
impl EdgeStreamer for SimpleEdgeStreamer {
    async fn stream_edges(&self, batch_size: usize) -> Result<Vec<Relationship>> {
        let start = self.current_position;
        let end = (start + batch_size).min(self.edges.len());
        Ok(self.edges[start..end].to_vec())
    }
    
    async fn stream_filtered_edges<F>(&self, filter: F, batch_size: usize) -> Result<Vec<Relationship>>
    where
        F: Fn(&Relationship) -> bool + Send + Sync,
    {
        let filtered: Vec<Relationship> = self.edges
            .iter()
            .filter(|&edge| filter(edge))
            .take(batch_size)
            .cloned()
            .collect();
        Ok(filtered)
    }
    
    async fn total_edge_count(&self) -> Result<usize> {
        Ok(self.edges.len())
    }
    
    async fn reset_stream(&mut self) -> Result<()> {
        self.current_position = 0;
        Ok(())
    }
}