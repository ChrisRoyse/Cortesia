// Multi-database versioning and temporal tracking module
// Implements comprehensive version management with anchor+delta strategy

pub mod version_store;
pub mod version_graph;
pub mod temporal_query;
pub mod merger;
pub mod types;

pub use version_store::VersionStore;
pub use version_graph::{VersionGraph, VersionNode, VersionEdge};
pub use temporal_query::TemporalQueryEngine;
pub use merger::VersionMerger;
pub use types::*;

use crate::error::{GraphError, Result};
use crate::federation::DatabaseId;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Multi-database version manager that handles versioning across the federation
pub struct MultiDatabaseVersionManager {
    /// Version stores for each database
    version_stores: Arc<RwLock<HashMap<DatabaseId, Arc<VersionStore>>>>,
    /// Global version graph tracking cross-database relationships
    global_version_graph: Arc<RwLock<VersionGraph>>,
    /// Temporal query engine for time-based queries
    temporal_engine: Arc<TemporalQueryEngine>,
    /// Version merger for conflict resolution
    merger: Arc<VersionMerger>,
}

impl MultiDatabaseVersionManager {
    pub fn new() -> Result<Self> {
        Ok(Self {
            version_stores: Arc::new(RwLock::new(HashMap::new())),
            global_version_graph: Arc::new(RwLock::new(VersionGraph::new())),
            temporal_engine: Arc::new(TemporalQueryEngine::new()?),
            merger: Arc::new(VersionMerger::new()),
        })
    }

    /// Register a new database for version management
    pub async fn register_database(&self, database_id: DatabaseId) -> Result<()> {
        let version_store = Arc::new(VersionStore::new(database_id.clone())?);
        let mut stores = self.version_stores.write().await;
        stores.insert(database_id.clone(), version_store);

        // Add database to global version graph
        let mut graph = self.global_version_graph.write().await;
        graph.add_database(database_id)?;

        Ok(())
    }

    /// Create a new version for an entity
    pub async fn create_version(
        &self,
        database_id: &DatabaseId,
        entity_id: &str,
        changes: Vec<FieldChange>,
        author: Option<String>,
        message: Option<String>,
    ) -> Result<VersionId> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        let version_id = store.create_version(entity_id, changes, author, message).await?;

        // Update global version graph
        let graph = self.global_version_graph.write().await;
        graph.add_version(database_id.clone(), entity_id.to_string(), version_id.clone()).await?;

        Ok(version_id)
    }

    /// Get version history for an entity
    pub async fn get_version_history(
        &self,
        database_id: &DatabaseId,
        entity_id: &str,
    ) -> Result<Vec<VersionEntry>> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        store.get_version_history(entity_id).await
    }

    /// Compare two versions of an entity
    pub async fn compare_versions(
        &self,
        database_id: &DatabaseId,
        entity_id: &str,
        version1: &VersionId,
        version2: &VersionId,
    ) -> Result<VersionComparison> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        store.compare_versions(entity_id, version1, version2).await
    }

    /// Compare the same entity across different databases
    pub async fn compare_across_databases(
        &self,
        entity_id: &str,
        database_versions: Vec<(DatabaseId, Option<VersionId>)>,
    ) -> Result<CrossDatabaseComparison> {
        let mut entity_versions = Vec::new();

        for (database_id, version_id) in database_versions {
            let stores = self.version_stores.read().await;
            let store = stores.get(&database_id)
                .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

            let version = if let Some(vid) = version_id {
                store.get_version(entity_id, &vid).await?
            } else {
                store.get_latest_version(entity_id).await?
            };

            entity_versions.push((database_id, version));
        }

        self.merger.compare_across_databases(entity_id, entity_versions).await
    }

    /// Execute a temporal query
    pub async fn temporal_query(&self, query: TemporalQuery) -> Result<TemporalResult> {
        self.temporal_engine.execute_query(query, &self.version_stores).await
    }

    /// Create a database snapshot
    pub async fn create_snapshot(
        &self,
        database_id: &DatabaseId,
        snapshot_name: String,
        description: Option<String>,
    ) -> Result<SnapshotId> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        store.create_snapshot(snapshot_name, description).await
    }

    /// Restore from a snapshot
    pub async fn restore_snapshot(
        &self,
        database_id: &DatabaseId,
        snapshot_id: &SnapshotId,
    ) -> Result<()> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        store.restore_snapshot(snapshot_id).await
    }

    /// Merge versions with conflict resolution
    pub async fn merge_versions(
        &self,
        database_id: &DatabaseId,
        entity_id: &str,
        base_version: &VersionId,
        version1: &VersionId,
        version2: &VersionId,
        resolution_strategy: ConflictResolution,
    ) -> Result<MergeResult> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        self.merger.three_way_merge(
            store.as_ref(),
            entity_id,
            base_version,
            version1,
            version2,
            resolution_strategy,
        ).await
    }

    /// Get global version statistics
    pub async fn get_version_statistics(&self) -> Result<VersionStatistics> {
        let stores = self.version_stores.read().await;
        let mut total_versions = 0;
        let mut total_entities = 0;
        let mut per_database_stats = HashMap::new();

        for (database_id, store) in stores.iter() {
            let stats = store.get_statistics().await?;
            total_versions += stats.total_versions;
            total_entities += stats.total_entities;
            per_database_stats.insert(database_id.clone(), stats);
        }

        Ok(VersionStatistics {
            total_versions,
            total_entities,
            total_databases: stores.len(),
            per_database_stats,
        })
    }

    /// Cleanup old versions based on retention policy
    pub async fn cleanup_old_versions(
        &self,
        database_id: &DatabaseId,
        retention_policy: RetentionPolicy,
    ) -> Result<CleanupResult> {
        let stores = self.version_stores.read().await;
        let store = stores.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", database_id.as_str())))?;

        store.cleanup_versions(retention_policy).await
    }
}