# Task 21: Temporal Versioning System
**Estimated Time**: 15-20 minutes
**Dependencies**: 20_connection_pooling.md
**Stage**: Advanced Features

## Objective
Implement a comprehensive temporal versioning system that tracks all changes to nodes, relationships, and properties over time, enabling point-in-time queries, change tracking, and temporal consistency validation.

## Specific Requirements

### 1. Temporal Node Versioning
- Version tracking for all node properties and metadata
- Immutable version history with efficient storage
- Temporal indexing for fast point-in-time queries
- Version branching support for experimental changes

### 2. Relationship Temporal Tracking
- Track creation, modification, and deletion of relationships
- Temporal relationship traversal capabilities
- Version consistency across connected nodes
- Efficient temporal join operations

### 3. Property Change History
- Fine-grained property-level versioning
- Change attribution and metadata tracking
- Efficient diff computation between versions
- Temporal property inheritance resolution

## Implementation Steps

### 1. Create Temporal Versioning Core System
```rust
// src/inheritance/temporal/versioning_system.rs
use std::collections::{HashMap, BTreeMap};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVersioningSystem {
    node_versions: Arc<RwLock<HashMap<String, NodeVersionHistory>>>,
    relationship_versions: Arc<RwLock<HashMap<String, RelationshipVersionHistory>>>,
    temporal_index: Arc<TemporalIndex>,
    version_metadata: Arc<RwLock<VersionMetadataStore>>,
    consistency_validator: Arc<TemporalConsistencyValidator>,
    config: TemporalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVersionHistory {
    pub node_id: String,
    pub versions: BTreeMap<DateTime<Utc>, NodeVersion>,
    pub current_version: DateTime<Utc>,
    pub branch_points: Vec<BranchPoint>,
    pub merge_history: Vec<MergeRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVersion {
    pub version_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub properties: HashMap<String, PropertyValue>,
    pub metadata: VersionMetadata,
    pub parent_version: Option<Uuid>,
    pub branch_id: Option<String>,
    pub change_summary: ChangeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyValue {
    pub value: serde_json::Value,
    pub change_type: ChangeType,
    pub change_metadata: PropertyChangeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
    Inherited,
    Merged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    pub author: Option<String>,
    pub commit_message: Option<String>,
    pub tags: Vec<String>,
    pub source_operation: OperationType,
    pub consistency_hash: String,
}

impl TemporalVersioningSystem {
    pub async fn new(config: TemporalConfig) -> Result<Self, TemporalError> {
        let node_versions = Arc::new(RwLock::new(HashMap::new()));
        let relationship_versions = Arc::new(RwLock::new(HashMap::new()));
        let temporal_index = Arc::new(TemporalIndex::new(config.index_config.clone()));
        let version_metadata = Arc::new(RwLock::new(VersionMetadataStore::new()));
        let consistency_validator = Arc::new(TemporalConsistencyValidator::new());
        
        Ok(Self {
            node_versions,
            relationship_versions,
            temporal_index,
            version_metadata,
            consistency_validator,
            config,
        })
    }
    
    pub async fn create_node_version(
        &self,
        node_id: &str,
        properties: HashMap<String, serde_json::Value>,
        metadata: VersionMetadata,
    ) -> Result<NodeVersion, VersioningError> {
        let timestamp = Utc::now();
        let version_id = Uuid::new_v4();
        
        // Convert properties to versioned format
        let versioned_properties: HashMap<String, PropertyValue> = properties
            .into_iter()
            .map(|(key, value)| {
                let property_value = PropertyValue {
                    value,
                    change_type: ChangeType::Created,
                    change_metadata: PropertyChangeMetadata {
                        timestamp,
                        previous_value: None,
                        change_reason: None,
                    },
                };
                (key, property_value)
            })
            .collect();
        
        // Create new version
        let new_version = NodeVersion {
            version_id,
            timestamp,
            properties: versioned_properties,
            metadata,
            parent_version: None,
            branch_id: None,
            change_summary: ChangeSummary::new(),
        };
        
        // Update version history
        let mut node_versions = self.node_versions.write().await;
        let history = node_versions.entry(node_id.to_string())
            .or_insert_with(|| NodeVersionHistory {
                node_id: node_id.to_string(),
                versions: BTreeMap::new(),
                current_version: timestamp,
                branch_points: Vec::new(),
                merge_history: Vec::new(),
            });
        
        history.versions.insert(timestamp, new_version.clone());
        history.current_version = timestamp;
        
        // Update temporal index
        self.temporal_index.index_node_version(node_id, &new_version).await?;
        
        // Validate temporal consistency
        self.consistency_validator.validate_node_version(node_id, &new_version).await?;
        
        Ok(new_version)
    }
    
    pub async fn update_node_version(
        &self,
        node_id: &str,
        property_updates: HashMap<String, serde_json::Value>,
        metadata: VersionMetadata,
    ) -> Result<NodeVersion, VersioningError> {
        let timestamp = Utc::now();
        let version_id = Uuid::new_v4();
        
        // Get current version
        let node_versions = self.node_versions.read().await;
        let current_history = node_versions.get(node_id)
            .ok_or_else(|| VersioningError::NodeNotFound(node_id.to_string()))?;
        
        let current_version = current_history.versions.get(&current_history.current_version)
            .ok_or_else(|| VersioningError::VersionNotFound)?;
        
        // Create new properties map with updates
        let mut new_properties = current_version.properties.clone();
        for (key, new_value) in property_updates {
            let previous_value = new_properties.get(&key).map(|pv| pv.value.clone());
            
            let property_value = PropertyValue {
                value: new_value,
                change_type: if previous_value.is_some() {
                    ChangeType::Modified
                } else {
                    ChangeType::Created
                },
                change_metadata: PropertyChangeMetadata {
                    timestamp,
                    previous_value,
                    change_reason: metadata.commit_message.clone(),
                },
            };
            
            new_properties.insert(key, property_value);
        }
        
        // Create new version
        let new_version = NodeVersion {
            version_id,
            timestamp,
            properties: new_properties,
            metadata,
            parent_version: Some(current_version.version_id),
            branch_id: current_version.branch_id.clone(),
            change_summary: self.compute_change_summary(current_version, &new_properties),
        };
        
        drop(node_versions); // Release read lock
        
        // Update version history
        let mut node_versions = self.node_versions.write().await;
        let history = node_versions.get_mut(node_id).unwrap();
        history.versions.insert(timestamp, new_version.clone());
        history.current_version = timestamp;
        
        // Update temporal index
        self.temporal_index.index_node_version(node_id, &new_version).await?;
        
        Ok(new_version)
    }
    
    pub async fn get_node_at_time(
        &self,
        node_id: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<NodeVersion>, VersioningError> {
        let node_versions = self.node_versions.read().await;
        let history = match node_versions.get(node_id) {
            Some(h) => h,
            None => return Ok(None),
        };
        
        // Find the most recent version at or before the requested timestamp
        let version = history.versions
            .range(..=timestamp)
            .next_back()
            .map(|(_, version)| version.clone());
        
        Ok(version)
    }
    
    pub async fn get_property_history(
        &self,
        node_id: &str,
        property_name: &str,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<PropertyHistoryEntry>, VersioningError> {
        let node_versions = self.node_versions.read().await;
        let history = node_versions.get(node_id)
            .ok_or_else(|| VersioningError::NodeNotFound(node_id.to_string()))?;
        
        let start = start_time.unwrap_or(DateTime::<Utc>::MIN_UTC);
        let end = end_time.unwrap_or(Utc::now());
        
        let mut property_history = Vec::new();
        
        for (timestamp, version) in history.versions.range(start..=end) {
            if let Some(property_value) = version.properties.get(property_name) {
                property_history.push(PropertyHistoryEntry {
                    timestamp: *timestamp,
                    value: property_value.value.clone(),
                    change_type: property_value.change_type.clone(),
                    metadata: property_value.change_metadata.clone(),
                });
            }
        }
        
        Ok(property_history)
    }
    
    fn compute_change_summary(
        &self,
        previous_version: &NodeVersion,
        new_properties: &HashMap<String, PropertyValue>,
    ) -> ChangeSummary {
        let mut added_properties = Vec::new();
        let mut modified_properties = Vec::new();
        let mut deleted_properties = Vec::new();
        
        // Find added and modified properties
        for (key, new_prop) in new_properties {
            match previous_version.properties.get(key) {
                Some(old_prop) => {
                    if old_prop.value != new_prop.value {
                        modified_properties.push(key.clone());
                    }
                },
                None => {
                    added_properties.push(key.clone());
                }
            }
        }
        
        // Find deleted properties
        for key in previous_version.properties.keys() {
            if !new_properties.contains_key(key) {
                deleted_properties.push(key.clone());
            }
        }
        
        ChangeSummary {
            added_properties,
            modified_properties,
            deleted_properties,
            total_changes: added_properties.len() + modified_properties.len() + deleted_properties.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub value: serde_json::Value,
    pub change_type: ChangeType,
    pub metadata: PropertyChangeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyChangeMetadata {
    pub timestamp: DateTime<Utc>,
    pub previous_value: Option<serde_json::Value>,
    pub change_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSummary {
    pub added_properties: Vec<String>,
    pub modified_properties: Vec<String>,
    pub deleted_properties: Vec<String>,
    pub total_changes: usize,
}

impl ChangeSummary {
    pub fn new() -> Self {
        Self {
            added_properties: Vec::new(),
            modified_properties: Vec::new(),
            deleted_properties: Vec::new(),
            total_changes: 0,
        }
    }
}
```

### 2. Implement Temporal Index System
```rust
// src/inheritance/temporal/temporal_index.rs
#[derive(Debug)]
pub struct TemporalIndex {
    time_index: Arc<RwLock<BTreeMap<DateTime<Utc>, Vec<String>>>>,
    node_index: Arc<RwLock<HashMap<String, Vec<DateTime<Utc>>>>>,
    property_index: Arc<RwLock<HashMap<String, PropertyTimeIndex>>>,
    bloom_filters: Arc<RwLock<HashMap<String, BloomFilter>>>,
    config: TemporalIndexConfig,
}

#[derive(Debug)]
pub struct PropertyTimeIndex {
    property_name: String,
    value_timeline: BTreeMap<DateTime<Utc>, HashMap<String, serde_json::Value>>,
    node_timelines: HashMap<String, BTreeMap<DateTime<Utc>, serde_json::Value>>,
}

impl TemporalIndex {
    pub async fn index_node_version(
        &self,
        node_id: &str,
        version: &NodeVersion,
    ) -> Result<(), IndexingError> {
        // Update time index
        let mut time_index = self.time_index.write().await;
        time_index.entry(version.timestamp)
            .or_insert_with(Vec::new)
            .push(node_id.to_string());
        
        // Update node index
        let mut node_index = self.node_index.write().await;
        node_index.entry(node_id.to_string())
            .or_insert_with(Vec::new)
            .push(version.timestamp);
        
        // Update property indices
        let mut property_index = self.property_index.write().await;
        for (prop_name, prop_value) in &version.properties {
            let prop_time_index = property_index.entry(prop_name.clone())
                .or_insert_with(|| PropertyTimeIndex {
                    property_name: prop_name.clone(),
                    value_timeline: BTreeMap::new(),
                    node_timelines: HashMap::new(),
                });
            
            // Update value timeline
            prop_time_index.value_timeline
                .entry(version.timestamp)
                .or_insert_with(HashMap::new)
                .insert(node_id.to_string(), prop_value.value.clone());
            
            // Update node timeline
            prop_time_index.node_timelines
                .entry(node_id.to_string())
                .or_insert_with(BTreeMap::new)
                .insert(version.timestamp, prop_value.value.clone());
        }
        
        Ok(())
    }
    
    pub async fn find_nodes_at_time(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<String>, IndexingError> {
        let time_index = self.time_index.read().await;
        
        // Find all nodes that have versions at or before the timestamp
        let mut result_nodes = Vec::new();
        for (time, nodes) in time_index.range(..=timestamp) {
            result_nodes.extend(nodes.clone());
        }
        
        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        result_nodes.retain(|node| seen.insert(node.clone()));
        
        Ok(result_nodes)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Complete node versioning with immutable history
- [ ] Property-level change tracking and diff computation
- [ ] Point-in-time query capabilities for nodes and relationships
- [ ] Temporal index for efficient time-based queries
- [ ] Version consistency validation and conflict detection

### Performance Requirements
- [ ] Version creation time < 10ms for standard nodes
- [ ] Point-in-time query response < 50ms for recent versions
- [ ] Temporal index lookup time < 5ms
- [ ] Version storage overhead < 30% of original data size
- [ ] Property history retrieval < 20ms for 1000 changes

### Testing Requirements
- [ ] Unit tests for version creation and retrieval
- [ ] Integration tests for temporal queries
- [ ] Performance tests for version storage efficiency
- [ ] Consistency validation tests

## Validation Steps

1. **Test version creation and retrieval**:
   ```rust
   let versioning = TemporalVersioningSystem::new(config).await?;
   let version = versioning.create_node_version("test_node", properties, metadata).await?;
   let retrieved = versioning.get_node_at_time("test_node", version.timestamp).await?;
   assert_eq!(retrieved.unwrap().version_id, version.version_id);
   ```

2. **Test temporal queries**:
   ```rust
   let history = versioning.get_property_history("test_node", "name", None, None).await?;
   assert!(!history.is_empty());
   ```

3. **Run temporal versioning tests**:
   ```bash
   cargo test temporal_versioning_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/temporal/versioning_system.rs` - Core temporal versioning
- `src/inheritance/temporal/temporal_index.rs` - Temporal indexing system
- `src/inheritance/temporal/consistency_validator.rs` - Version consistency validation
- `src/inheritance/temporal/mod.rs` - Module exports
- `tests/inheritance/temporal_tests.rs` - Temporal versioning test suite

## Success Metrics
- Version creation latency: <10ms average
- Point-in-time query performance: <50ms for recent versions
- Storage efficiency: <30% overhead for versioned data
- Index lookup performance: <5ms average

## Next Task
Upon completion, proceed to **22_branch_management.md** to implement version branching and merging capabilities.