use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use ahash::AHashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::brain_types::{BrainInspiredEntity, BrainInspiredRelationship};
use crate::core::graph::KnowledgeGraph;
use crate::core::types::EntityKey;
use crate::error::{Result, GraphError};

/// Time range for bi-temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,  // None = current
}

impl TimeRange {
    pub fn new(start: DateTime<Utc>) -> Self {
        Self { start, end: None }
    }

    pub fn with_end(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end: Some(end) }
    }

    pub fn contains(&self, time: &DateTime<Utc>) -> bool {
        if time < &self.start {
            return false;
        }
        match self.end {
            Some(end) => time <= &end,
            None => true,
        }
    }
}

/// Temporal entity with bi-temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEntity {
    pub entity: BrainInspiredEntity,
    pub valid_time: TimeRange,      // When fact was true in real world
    pub transaction_time: TimeRange, // When fact was stored in system
    pub version_id: u64,
    pub supersedes: Option<EntityKey>, // Previous version reference
}

/// Temporal relationship with versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRelationship {
    pub relationship: BrainInspiredRelationship,
    pub valid_time: TimeRange,
    pub transaction_time: TimeRange,
    pub version_id: u64,
    pub supersedes: Option<(EntityKey, EntityKey)>, // Previous relationship
}

/// Bi-temporal index for efficient temporal queries
pub struct BiTemporalIndex {
    // Valid time index: time -> entities valid at that time
    valid_time_index: BTreeMap<DateTime<Utc>, Vec<EntityKey>>,
    // Transaction time index: time -> entities stored at that time
    transaction_time_index: BTreeMap<DateTime<Utc>, Vec<EntityKey>>,
    // Entity version chains: entity_key -> list of versions
    version_chains: AHashMap<EntityKey, Vec<u64>>,
}

impl BiTemporalIndex {
    pub fn new() -> Self {
        Self {
            valid_time_index: BTreeMap::new(),
            transaction_time_index: BTreeMap::new(),
            version_chains: AHashMap::new(),
        }
    }

    pub fn index_entity(&mut self, entity_key: EntityKey, temporal_entity: &TemporalEntity) {
        // Index by valid time start
        self.valid_time_index
            .entry(temporal_entity.valid_time.start)
            .or_insert_with(Vec::new)
            .push(entity_key);

        // Index by transaction time start
        self.transaction_time_index
            .entry(temporal_entity.transaction_time.start)
            .or_insert_with(Vec::new)
            .push(entity_key);

        // Track version chain
        self.version_chains
            .entry(entity_key)
            .or_insert_with(Vec::new)
            .push(temporal_entity.version_id);
    }

    pub fn find_valid_at(&self, time: DateTime<Utc>) -> Vec<EntityKey> {
        let mut results = Vec::new();
        
        // Find all entities with valid_time.start <= time
        for (start_time, entities) in self.valid_time_index.range(..=time).rev() {
            results.extend(entities.iter().cloned());
        }
        
        results
    }
}

/// Temporal storage backend
pub struct TemporalStore {
    entities: AHashMap<EntityKey, Vec<TemporalEntity>>,
    relationships: AHashMap<(EntityKey, EntityKey), Vec<TemporalRelationship>>,
    next_version_id: u64,
}

impl TemporalStore {
    pub fn new() -> Self {
        Self {
            entities: AHashMap::new(),
            relationships: AHashMap::new(),
            next_version_id: 1,
        }
    }

    pub fn store_entity_version(&mut self, key: EntityKey, temporal_entity: TemporalEntity) {
        self.entities
            .entry(key)
            .or_insert_with(Vec::new)
            .push(temporal_entity);
    }

    pub fn store_relationship_version(
        &mut self,
        source: EntityKey,
        target: EntityKey,
        temporal_rel: TemporalRelationship,
    ) {
        self.relationships
            .entry((source, target))
            .or_insert_with(Vec::new)
            .push(temporal_rel);
    }

    pub fn get_next_version_id(&mut self) -> u64 {
        let id = self.next_version_id;
        self.next_version_id += 1;
        id
    }

    pub fn get_entity_at_time(
        &self,
        key: EntityKey,
        valid_time: DateTime<Utc>,
        transaction_time: DateTime<Utc>,
    ) -> Option<&TemporalEntity> {
        self.entities.get(&key).and_then(|versions| {
            versions.iter()
                .filter(|v| v.valid_time.contains(&valid_time))
                .filter(|v| v.transaction_time.contains(&transaction_time))
                .max_by_key(|v| v.version_id)
        })
    }
}

/// Temporal knowledge graph with bi-temporal data model
#[derive(Clone)]
pub struct TemporalKnowledgeGraph {
    pub current_graph: Arc<RwLock<KnowledgeGraph>>,
    pub temporal_store: Arc<RwLock<TemporalStore>>,
    pub bi_temporal_index: Arc<RwLock<BiTemporalIndex>>,
}

impl TemporalKnowledgeGraph {
    pub fn new(current_graph: KnowledgeGraph) -> Self {
        Self {
            current_graph: Arc::new(RwLock::new(current_graph)),
            temporal_store: Arc::new(RwLock::new(TemporalStore::new())),
            bi_temporal_index: Arc::new(RwLock::new(BiTemporalIndex::new())),
        }
    }

    /// Create a new temporal graph with default configuration
    pub fn new_default() -> Self {
        let default_graph = KnowledgeGraph::new(384).unwrap();
        Self::new(default_graph)
    }

    /// Insert a temporal entity with bi-temporal tracking
    pub async fn insert_temporal_entity(
        &self,
        entity: BrainInspiredEntity,
        valid_time: TimeRange,
    ) -> Result<EntityKey> {
        let transaction_time = TimeRange::new(Utc::now());
        let mut temporal_store = self.temporal_store.write().await;
        
        // Check for existing versions
        let supersedes = if let Some(existing_versions) = temporal_store.entities.get(&entity.id) {
            // Find the current version
            existing_versions.iter()
                .filter(|v| v.valid_time.end.is_none())
                .map(|v| v.entity.id)
                .last()
        } else {
            None
        };

        // Create new version
        let version_id = temporal_store.get_next_version_id();
        let temporal_entity = TemporalEntity {
            entity: entity.clone(),
            valid_time,
            transaction_time,
            version_id,
            supersedes,
        };

        // Store the temporal entity
        let entity_key = entity.id;
        temporal_store.store_entity_version(entity_key, temporal_entity.clone());

        // Update bi-temporal index
        let mut index = self.bi_temporal_index.write().await;
        index.index_entity(entity_key, &temporal_entity);

        // Update current graph with latest version
        let mut current = self.current_graph.write().await;
        // Note: We'll need to adapt this to work with the existing KnowledgeGraph structure
        // For now, we'll store the entity ID for reference
        
        Ok(entity_key)
    }

    /// Insert multiple temporal entities with shared metadata
    pub async fn insert_temporal_entities(
        &self,
        entities: Vec<BrainInspiredEntity>,
        metadata: AHashMap<String, String>,
    ) -> Result<Vec<EntityKey>> {
        let mut entity_keys = Vec::new();
        let valid_time = TimeRange::new(Utc::now());
        
        for entity in entities {
            let entity_key = self.insert_temporal_entity(entity, valid_time.clone()).await?;
            entity_keys.push(entity_key);
        }
        
        // Store metadata separately if needed
        // For now, we'll just return the entity keys
        
        Ok(entity_keys)
    }

    /// Query entities at a specific point in time
    pub async fn query_at_time(
        &self,
        valid_time: DateTime<Utc>,
        transaction_time: DateTime<Utc>,
    ) -> Result<Vec<TemporalEntity>> {
        let store = self.temporal_store.read().await;
        let index = self.bi_temporal_index.read().await;
        
        // Find all entities valid at the given time
        let valid_entities = index.find_valid_at(valid_time);
        
        let mut results = Vec::new();
        for entity_key in valid_entities {
            if let Some(temporal_entity) = store.get_entity_at_time(
                entity_key,
                valid_time,
                transaction_time,
            ) {
                results.push(temporal_entity.clone());
            }
        }
        
        Ok(results)
    }

    /// Get the full history of an entity
    pub async fn get_entity_history(
        &self,
        entity_key: EntityKey,
    ) -> Result<Vec<TemporalEntity>> {
        let store = self.temporal_store.read().await;
        
        store.entities.get(&entity_key)
            .map(|versions| versions.clone())
            .ok_or_else(|| GraphError::InvalidInput(format!("Entity {:?} not found", entity_key)))
    }

    /// Insert a temporal relationship
    pub async fn insert_temporal_relationship(
        &self,
        relationship: BrainInspiredRelationship,
        valid_time: TimeRange,
    ) -> Result<()> {
        let transaction_time = TimeRange::new(Utc::now());
        let mut temporal_store = self.temporal_store.write().await;
        
        let version_id = temporal_store.get_next_version_id();
        let key = (relationship.source, relationship.target);
        
        // Check for superseded relationship
        let supersedes = temporal_store.relationships.get(&key)
            .and_then(|versions| {
                versions.iter()
                    .filter(|v| v.valid_time.end.is_none())
                    .map(|v| (v.relationship.source, v.relationship.target))
                    .last()
            });

        let temporal_rel = TemporalRelationship {
            relationship,
            valid_time,
            transaction_time,
            version_id,
            supersedes,
        };

        temporal_store.store_relationship_version(key.0, key.1, temporal_rel);
        
        Ok(())
    }

    /// Time travel query - get the state of the graph at a specific time
    pub async fn time_travel_query(
        &self,
        query: &str,
        valid_time: DateTime<Utc>,
        transaction_time: DateTime<Utc>,
    ) -> Result<Vec<TemporalEntity>> {
        // This would integrate with the existing query engine
        // For now, we'll implement a simple temporal entity lookup
        self.query_at_time(valid_time, transaction_time).await
    }

    /// Find temporal patterns across time ranges
    pub async fn find_temporal_patterns(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        pattern: &str,
    ) -> Result<Vec<(DateTime<Utc>, Vec<TemporalEntity>)>> {
        let store = self.temporal_store.read().await;
        let mut temporal_results = Vec::new();
        
        // Sample at regular intervals
        let mut current_time = start_time;
        let interval = chrono::Duration::hours(1); // Configurable
        
        while current_time <= end_time {
            let entities = self.query_at_time(current_time, Utc::now()).await?;
            
            // Filter entities matching the pattern
            let matching_entities: Vec<_> = entities.into_iter()
                .filter(|e| e.entity.concept_id.contains(pattern))
                .collect();
            
            if !matching_entities.is_empty() {
                temporal_results.push((current_time, matching_entities));
            }
            
            current_time = current_time + interval;
        }
        
        Ok(temporal_results)
    }

    /// Get all entities in the graph (for cognitive pattern access)
    pub async fn get_all_entities(&self) -> Vec<BrainInspiredEntity> {
        let store = self.temporal_store.read().await;
        let mut entities = Vec::new();
        
        for versions in store.entities.values() {
            // Get the latest version of each entity
            if let Some(latest) = versions.iter().max_by_key(|v| v.version_id) {
                entities.push(latest.entity.clone());
            }
        }
        
        entities
    }

    /// Search entities by concept pattern
    pub async fn search_entities(&self, pattern: &str) -> Vec<BrainInspiredEntity> {
        let store = self.temporal_store.read().await;
        let mut entities = Vec::new();
        
        for versions in store.entities.values() {
            // Get the latest version of each entity
            if let Some(latest) = versions.iter().max_by_key(|v| v.version_id) {
                if latest.entity.concept_id.contains(pattern) {
                    entities.push(latest.entity.clone());
                }
            }
        }
        
        entities
    }
}

// Safety: TemporalKnowledgeGraph contains only Arc<RwLock<T>> which are Send + Sync
unsafe impl Send for TemporalKnowledgeGraph {}
unsafe impl Sync for TemporalKnowledgeGraph {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::EntityDirection;

    #[tokio::test]
    async fn test_temporal_entity_insertion() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = TemporalKnowledgeGraph::new(graph);
        
        let mut entity = BrainInspiredEntity::new("test_concept".to_string(), EntityDirection::Input);
        entity.id = EntityKey::default();
        
        let valid_time = TimeRange::new(Utc::now());
        let result = temporal_graph.insert_temporal_entity(entity, valid_time).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_time_travel_query() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = TemporalKnowledgeGraph::new(graph);
        
        // Insert an entity
        let mut entity = BrainInspiredEntity::new("historical_fact".to_string(), EntityDirection::Input);
        entity.id = EntityKey::default();
        
        let valid_time = TimeRange::new(Utc::now() - chrono::Duration::days(7));
        temporal_graph.insert_temporal_entity(entity, valid_time).await.unwrap();
        
        // Query at different times
        let past_time = Utc::now() - chrono::Duration::days(3);
        let results = temporal_graph.query_at_time(past_time, Utc::now()).await.unwrap();
        
        assert!(!results.is_empty());
    }

    #[test]
    fn test_time_range_contains() {
        let start = Utc::now() - chrono::Duration::days(7);
        let end = Utc::now() - chrono::Duration::days(1);
        let range = TimeRange::with_end(start, end);
        
        let test_time = Utc::now() - chrono::Duration::days(3);
        assert!(range.contains(&test_time));
        
        let future_time = Utc::now() + chrono::Duration::days(1);
        assert!(!range.contains(&future_time));
    }
}