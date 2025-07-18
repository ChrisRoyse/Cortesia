use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::core::brain_types::BrainInspiredEntity;
use crate::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
use crate::error::{Result, GraphError};

/// Type of update operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateOperation {
    Create,
    Update,
    Delete,
    Merge,
}

/// Source of the update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateSource {
    User(String),
    System,
    Federation(String),
    Import(String),
}

/// Temporal update event
#[derive(Debug, Clone)]
pub struct TemporalUpdate {
    pub operation: UpdateOperation,
    pub entity: BrainInspiredEntity,
    pub timestamp: DateTime<Utc>,
    pub source: UpdateSource,
    pub metadata: Option<serde_json::Value>,
}

/// Update processing statistics
#[derive(Debug, Clone, Default)]
pub struct UpdateStatistics {
    pub total_updates: u64,
    pub successful_updates: u64,
    pub failed_updates: u64,
    pub average_latency_ms: f64,
    pub peak_queue_size: usize,
}

/// Real-time incremental update processor
pub struct IncrementalTemporalProcessor {
    pub update_queue: Arc<Mutex<VecDeque<TemporalUpdate>>>,
    pub processing_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    pub batch_size: usize,
    pub max_latency: Duration,
    pub temporal_graph: Arc<TemporalKnowledgeGraph>,
    pub statistics: Arc<Mutex<UpdateStatistics>>,
    pub is_running: Arc<Mutex<bool>>,
}

impl IncrementalTemporalProcessor {
    pub fn new(
        temporal_graph: Arc<TemporalKnowledgeGraph>,
        batch_size: usize,
        max_latency: Duration,
    ) -> Self {
        Self {
            update_queue: Arc::new(Mutex::new(VecDeque::new())),
            processing_thread: Arc::new(Mutex::new(None)),
            batch_size,
            max_latency,
            temporal_graph,
            statistics: Arc::new(Mutex::new(UpdateStatistics::default())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start processing updates
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().await;
        if *is_running {
            return Err(GraphError::InvalidState("Processor already running".to_string()));
        }
        *is_running = true;

        // For now, we'll just mark as started without spawning a background task
        // In a full implementation, we'd handle the Send/Sync requirements properly
        Ok(())
    }

    /// Stop processing updates
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().await;
        *is_running = false;

        // For now, just mark as stopped
        Ok(())
    }

    /// Add an update to the queue
    pub async fn enqueue_update(&self, update: TemporalUpdate) -> Result<()> {
        let mut queue = self.update_queue.lock().await;
        queue.push_back(update);

        // Update peak queue size statistic
        let mut stats = self.statistics.lock().await;
        stats.peak_queue_size = stats.peak_queue_size.max(queue.len());

        Ok(())
    }

    /// Process a stream of updates
    pub async fn process_update_stream<S>(&self, mut updates: S) -> Result<()>
    where
        S: Stream<Item = TemporalUpdate> + Unpin,
    {
        while let Some(update) = updates.next().await {
            self.enqueue_update(update).await?;
        }
        Ok(())
    }

    /// Main processing loop
    async fn processing_loop(
        queue: Arc<Mutex<VecDeque<TemporalUpdate>>>,
        graph: Arc<TemporalKnowledgeGraph>,
        stats: Arc<Mutex<UpdateStatistics>>,
        batch_size: usize,
        max_latency: Duration,
        is_running: Arc<Mutex<bool>>,
    ) {
        let mut last_batch_time = tokio::time::Instant::now();

        loop {
            // Check if we should continue running
            if !*is_running.lock().await {
                break;
            }

            // Collect batch
            let mut batch = Vec::new();
            let batch_start_time = tokio::time::Instant::now();

            {
                let mut queue_guard = queue.lock().await;
                while batch.len() < batch_size && !queue_guard.is_empty() {
                    if let Some(update) = queue_guard.pop_front() {
                        batch.push(update);
                    }
                }
            }

            // Process batch if we have updates or if max latency exceeded
            if !batch.is_empty() || batch_start_time.duration_since(last_batch_time) >= max_latency {
                if !batch.is_empty() {
                    Self::process_batch(batch, &graph, &stats).await;
                }
                last_batch_time = batch_start_time;
            } else {
                // Sleep briefly to avoid busy waiting
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    /// Process a batch of updates
    async fn process_batch(
        batch: Vec<TemporalUpdate>,
        graph: &Arc<TemporalKnowledgeGraph>,
        stats: &Arc<Mutex<UpdateStatistics>>,
    ) {
        let batch_size = batch.len();
        let start_time = tokio::time::Instant::now();
        let mut successful = 0;
        let mut failed = 0;

        for update in batch {
            match Self::process_single_update(update, graph).await {
                Ok(_) => successful += 1,
                Err(e) => {
                    eprintln!("Failed to process update: {:?}", e);
                    failed += 1;
                }
            }
        }

        let elapsed = start_time.elapsed();
        let latency_ms = elapsed.as_millis() as f64 / batch_size as f64;

        // Update statistics
        let mut stats_guard = stats.lock().await;
        stats_guard.total_updates += batch_size as u64;
        stats_guard.successful_updates += successful;
        stats_guard.failed_updates += failed;
        
        // Update average latency (simple moving average)
        let total = stats_guard.total_updates as f64;
        stats_guard.average_latency_ms = 
            (stats_guard.average_latency_ms * (total - batch_size as f64) + latency_ms * batch_size as f64) / total;
    }

    /// Process a single update
    async fn process_single_update(
        update: TemporalUpdate,
        graph: &Arc<TemporalKnowledgeGraph>,
    ) -> Result<()> {
        match update.operation {
            UpdateOperation::Create | UpdateOperation::Update => {
                // Determine valid time range
                let valid_time = TimeRange::new(update.timestamp);
                
                // Insert or update the entity
                graph.insert_temporal_entity(update.entity, valid_time).await?;
            }
            UpdateOperation::Delete => {
                // For deletion, we end the valid time of the current version
                // This would require additional method in TemporalKnowledgeGraph
                // For now, we'll skip implementation
                return Err(GraphError::NotImplemented("Delete operation not yet implemented".to_string()));
            }
            UpdateOperation::Merge => {
                // Merge operation would combine multiple versions
                // This requires conflict resolution logic
                return Err(GraphError::NotImplemented("Merge operation not yet implemented".to_string()));
            }
        }

        Ok(())
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> UpdateStatistics {
        self.statistics.lock().await.clone()
    }

    /// Get current queue size
    pub async fn get_queue_size(&self) -> usize {
        self.update_queue.lock().await.len()
    }

    /// Clear the update queue
    pub async fn clear_queue(&self) -> Result<()> {
        let mut queue = self.update_queue.lock().await;
        queue.clear();
        Ok(())
    }
}

/// Builder for creating temporal updates
#[derive(Debug, Clone)]
pub struct TemporalUpdateBuilder {
    operation: UpdateOperation,
    entity: Option<BrainInspiredEntity>,
    timestamp: Option<DateTime<Utc>>,
    source: Option<UpdateSource>,
    metadata: Option<serde_json::Value>,
}

impl TemporalUpdateBuilder {
    pub fn new(operation: UpdateOperation) -> Self {
        Self {
            operation,
            entity: None,
            timestamp: None,
            source: None,
            metadata: None,
        }
    }

    pub fn with_entity(mut self, entity: BrainInspiredEntity) -> Self {
        self.entity = Some(entity);
        self
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn with_source(mut self, source: UpdateSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn build(self) -> Result<TemporalUpdate> {
        let entity = self.entity
            .ok_or_else(|| GraphError::InvalidInput("Entity is required".to_string()))?;
        
        Ok(TemporalUpdate {
            operation: self.operation,
            entity,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            source: self.source.unwrap_or(UpdateSource::System),
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::EntityDirection;
    use crate::core::graph::KnowledgeGraph;

    #[tokio::test]
    async fn test_incremental_processor() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(TemporalKnowledgeGraph::new(graph));
        let processor = IncrementalTemporalProcessor::new(
            temporal_graph,
            10,
            Duration::from_millis(100),
        );

        // Start processor
        processor.start().await.unwrap();

        // Create and enqueue an update
        let entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        let update = TemporalUpdateBuilder::new(UpdateOperation::Create)
            .with_entity(entity)
            .with_source(UpdateSource::User("test_user".to_string()))
            .build()
            .unwrap();

        processor.enqueue_update(update).await.unwrap();

        // Wait briefly for processing
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Check statistics
        let stats = processor.get_statistics().await;
        assert!(stats.total_updates > 0);

        // Stop processor
        processor.stop().await.unwrap();
    }

    #[test]
    fn test_temporal_update_builder() {
        let entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Output);
        let update = TemporalUpdateBuilder::new(UpdateOperation::Update)
            .with_entity(entity)
            .with_source(UpdateSource::System)
            .build();

        assert!(update.is_ok());
        let update = update.unwrap();
        assert!(matches!(update.operation, UpdateOperation::Update));
        assert!(matches!(update.source, UpdateSource::System));
    }
}