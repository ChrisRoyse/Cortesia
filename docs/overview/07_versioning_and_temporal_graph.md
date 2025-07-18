# Versioning and Temporal Graph Capabilities

## Overview

The LLMKG system implements a sophisticated versioning and temporal graph system that enables tracking of knowledge evolution over time. This bi-temporal data model provides comprehensive version control for entities and relationships, supporting both valid time (when facts were true in the real world) and transaction time (when facts were stored in the system). The system enables time-travel queries, change tracking, conflict resolution, and sophisticated temporal analysis.

## Core Temporal Architecture

### Bi-Temporal Data Model (`src/versioning/temporal_graph.rs`)

The system implements a bi-temporal data model that tracks two different time dimensions:

#### Time Range Structure:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: Option<DateTime<Utc>>,  // None = current
}
```

**Valid Time vs Transaction Time**:
- **Valid Time**: When a fact was true in the real world
- **Transaction Time**: When a fact was stored in the system
- **Bi-temporal Tracking**: Combination of both time dimensions for complete temporal tracking

#### Temporal Entity Structure:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEntity {
    pub entity: BrainInspiredEntity,
    pub valid_time: TimeRange,      // When fact was true in real world
    pub transaction_time: TimeRange, // When fact was stored in system
    pub version_id: u64,
    pub supersedes: Option<EntityKey>, // Previous version reference
}
```

#### Temporal Relationship Structure:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRelationship {
    pub relationship: BrainInspiredRelationship,
    pub valid_time: TimeRange,
    pub transaction_time: TimeRange,
    pub version_id: u64,
    pub supersedes: Option<(EntityKey, EntityKey)>, // Previous relationship
}
```

### Bi-Temporal Indexing

The system maintains specialized indices for efficient temporal queries:

```rust
pub struct BiTemporalIndex {
    // Valid time index: time -> entities valid at that time
    valid_time_index: BTreeMap<DateTime<Utc>, Vec<EntityKey>>,
    // Transaction time index: time -> entities stored at that time
    transaction_time_index: BTreeMap<DateTime<Utc>, Vec<EntityKey>>,
    // Entity version chains: entity_key -> list of versions
    version_chains: AHashMap<EntityKey, Vec<u64>>,
}
```

**Index Operations**:
- **Valid Time Lookup**: Find entities valid at a specific time
- **Transaction Time Lookup**: Find entities stored at a specific time
- **Version Chain Tracking**: Track all versions of an entity
- **Range Queries**: Efficient queries over time ranges

### Temporal Storage Backend

```rust
pub struct TemporalStore {
    entities: AHashMap<EntityKey, Vec<TemporalEntity>>,
    relationships: AHashMap<(EntityKey, EntityKey), Vec<TemporalRelationship>>,
    next_version_id: u64,
}
```

**Storage Features**:
- **Version Lists**: Multiple versions per entity/relationship
- **Efficient Lookup**: Direct access to specific versions
- **Version Management**: Automatic version ID generation
- **Conflict Resolution**: Handling of concurrent modifications

## Temporal Knowledge Graph (`src/versioning/temporal_graph.rs`)

### Core Architecture

```rust
pub struct TemporalKnowledgeGraph {
    pub current_graph: Arc<RwLock<KnowledgeGraph>>,
    pub temporal_store: Arc<RwLock<TemporalStore>>,
    pub bi_temporal_index: Arc<RwLock<BiTemporalIndex>>,
}
```

### Key Operations

#### Entity Insertion with Temporal Tracking:
```rust
pub async fn insert_temporal_entity(
    &self,
    entity: BrainInspiredEntity,
    valid_time: TimeRange,
) -> Result<EntityKey> {
    let transaction_time = TimeRange::new(Utc::now());
    
    // Check for existing versions
    let supersedes = self.find_superseded_version(&entity.id).await?;
    
    // Create new version
    let version_id = self.generate_version_id();
    let temporal_entity = TemporalEntity {
        entity,
        valid_time,
        transaction_time,
        version_id,
        supersedes,
    };
    
    // Store and index
    self.store_temporal_entity(temporal_entity).await?;
    self.update_indices(temporal_entity).await?;
    
    Ok(entity_key)
}
```

#### Time-Travel Queries:
```rust
pub async fn query_at_time(
    &self,
    valid_time: DateTime<Utc>,
    transaction_time: DateTime<Utc>,
) -> Result<Vec<TemporalEntity>> {
    let index = self.bi_temporal_index.read().await;
    let valid_entities = index.find_valid_at(valid_time);
    
    let mut results = Vec::new();
    for entity_key in valid_entities {
        if let Some(temporal_entity) = self.get_entity_at_time(
            entity_key,
            valid_time,
            transaction_time,
        ).await? {
            results.push(temporal_entity);
        }
    }
    
    Ok(results)
}
```

#### Temporal Pattern Analysis:
```rust
pub async fn find_temporal_patterns(
    &self,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    pattern: &str,
) -> Result<Vec<(DateTime<Utc>, Vec<TemporalEntity>)>> {
    let mut temporal_results = Vec::new();
    let interval = chrono::Duration::hours(1);
    
    let mut current_time = start_time;
    while current_time <= end_time {
        let entities = self.query_at_time(current_time, Utc::now()).await?;
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
```

## Version Management System (`src/versioning/version_store.rs`)

### Version Store Architecture

```rust
pub struct VersionStore {
    database_id: DatabaseId,
    entity_versions: Arc<RwLock<HashMap<String, Vec<VersionEntry>>>>,
    version_index: Arc<RwLock<HashMap<VersionId, VersionEntry>>>,
    snapshots: Arc<RwLock<HashMap<SnapshotId, SnapshotData>>>,
    stats: Arc<RwLock<DatabaseVersionStats>>,
}
```

### Version Entry Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version_id: VersionId,
    pub entity_id: String,
    pub database_id: DatabaseId,
    pub timestamp: SystemTime,
    pub author: Option<String>,
    pub message: Option<String>,
    pub parent_versions: Vec<VersionId>,
    pub changes: Vec<FieldChange>,
    pub is_anchor: bool,  // True if this is an anchor version (full state)
    pub compressed_size: usize,
    pub metadata: VersionMetadata,
}
```

### Change Tracking

#### Field Change Structure:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldChange {
    pub field_name: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    Added,      // Field was added
    Modified,   // Field value was changed
    Removed,    // Field was removed
    Renamed,    // Field was renamed
}
```

#### Version Creation:
```rust
pub async fn create_version(
    &self,
    entity_id: &str,
    changes: Vec<FieldChange>,
    author: Option<String>,
    message: Option<String>,
) -> Result<VersionId> {
    let version_id = VersionId::generate(&self.database_id, entity_id);
    let timestamp = SystemTime::now();
    
    // Determine if this should be an anchor version
    let is_anchor = self.should_create_anchor_version(entity_id).await?;
    
    // Get parent versions
    let parent_versions = self.get_parent_versions(entity_id).await?;
    
    // Create version entry
    let version_entry = VersionEntry {
        version_id: version_id.clone(),
        entity_id: entity_id.to_string(),
        database_id: self.database_id.clone(),
        timestamp,
        author,
        message,
        parent_versions,
        changes: changes.clone(),
        is_anchor,
        compressed_size: self.estimate_compressed_size(&changes),
        metadata: self.create_version_metadata(&changes),
    };
    
    // Store the version
    self.store_version(version_entry).await?;
    
    Ok(version_id)
}
```

### Version Comparison

#### Version Comparison Structure:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    pub entity_id: String,
    pub version1: VersionId,
    pub version2: VersionId,
    pub differences: Vec<FieldDifference>,
    pub similarity_score: f32,
    pub change_summary: ChangeSummary,
}
```

#### Field Difference Analysis:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDifference {
    pub field_name: String,
    pub difference_type: DifferenceType,
    pub version1_value: Option<serde_json::Value>,
    pub version2_value: Option<serde_json::Value>,
    pub similarity_score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferenceType {
    ValueChanged,
    FieldAdded,
    FieldRemoved,
    TypeChanged,
    FormatChanged,
    StructureChanged,
}
```

## Temporal Query System (`src/versioning/types.rs`)

### Temporal Query Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalQuery {
    /// Get entity state at a specific point in time
    PointInTime {
        entity_id: String,
        database_id: DatabaseId,
        timestamp: SystemTime,
    },
    /// Get all changes in a time range
    TimeRange {
        entity_id: Option<String>,
        database_id: DatabaseId,
        start_time: SystemTime,
        end_time: SystemTime,
        include_metadata: bool,
    },
    /// Track evolution of a specific field over time
    FieldEvolution {
        entity_id: String,
        database_id: DatabaseId,
        field_name: String,
        time_range: Option<(SystemTime, SystemTime)>,
    },
    /// Compare versions across time
    TemporalComparison {
        entity_id: String,
        database_id: DatabaseId,
        timestamps: Vec<SystemTime>,
    },
    /// Find entities that changed during a time period
    ChangedEntities {
        database_id: DatabaseId,
        start_time: SystemTime,
        end_time: SystemTime,
        change_types: Option<Vec<ChangeType>>,
    },
}
```

### Query Result Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalResultData {
    PointInTimeResult {
        entity_state: HashMap<String, serde_json::Value>,
        version_id: VersionId,
        timestamp: SystemTime,
    },
    TimeRangeResult {
        changes: Vec<VersionEntry>,
        summary: ChangeSummary,
    },
    FieldEvolutionResult {
        field_history: Vec<(SystemTime, serde_json::Value, VersionId)>,
        trend_analysis: TrendAnalysis,
    },
    TemporalComparisonResult {
        timeline: Vec<(SystemTime, VersionEntry)>,
        differences: Vec<VersionComparison>,
    },
    ChangedEntitiesResult {
        entities: Vec<(String, Vec<VersionEntry>)>,
        change_statistics: HashMap<ChangeType, usize>,
    },
}
```

### Trend Analysis

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub change_frequency: f32,
    pub stability_score: f32,
    pub pattern_detected: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}
```

## Snapshot Management

### Snapshot Creation and Restoration

```rust
pub async fn create_snapshot(
    &self,
    name: String,
    description: Option<String>,
) -> Result<SnapshotId> {
    let snapshot_id = SnapshotId::new(format!("{}:{}", self.database_id.as_str(), uuid::Uuid::new_v4()));
    
    // Capture current state of all entities
    let entity_versions = self.entity_versions.read().await;
    let mut entity_states = HashMap::new();
    
    for (entity_id, versions) in entity_versions.iter() {
        if let Some(latest_version) = versions.last() {
            entity_states.insert(entity_id.clone(), latest_version.version_id.clone());
        }
    }
    
    let snapshot = SnapshotData {
        id: snapshot_id.clone(),
        name,
        description,
        created_at: SystemTime::now(),
        entity_states,
    };
    
    self.store_snapshot(snapshot).await?;
    
    Ok(snapshot_id)
}
```

### Snapshot Restoration

```rust
pub async fn restore_snapshot(&self, snapshot_id: &SnapshotId) -> Result<()> {
    let snapshot = self.get_snapshot(snapshot_id).await?;
    
    // Restore entity states from snapshot
    for (entity_id, version_id) in snapshot.entity_states {
        let version = self.get_version(&entity_id, &version_id).await?;
        self.restore_entity_state(entity_id, version).await?;
    }
    
    Ok(())
}
```

## Conflict Resolution and Merging

### Merge Conflict Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    pub field_name: String,
    pub base_value: Option<serde_json::Value>,
    pub version1_value: Option<serde_json::Value>,
    pub version2_value: Option<serde_json::Value>,
    pub conflict_type: ConflictType,
    pub suggested_resolution: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    ValueConflict,      // Different values for same field
    StructuralConflict, // Structural changes conflict
    TypeConflict,       // Type changes conflict
    RenameConflict,     // Field rename conflicts
}
```

### Conflict Resolution Strategies

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictResolution {
    TakeVersion1,       // Always use version 1
    TakeVersion2,       // Always use version 2
    TakeNewest,         // Use the most recent change
    TakeHighestConfidence, // Use version with highest confidence
    ManualResolution,   // Require manual intervention
    SmartMerge,         // Apply intelligent merging rules
}
```

### Merge Result

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub success: bool,
    pub merged_version: Option<VersionId>,
    pub conflicts: Vec<MergeConflict>,
    pub resolution_applied: ConflictResolution,
    pub merge_statistics: MergeStatistics,
}
```

## Version Management Features

### Anchor vs Delta Versions

**Anchor Versions**:
- **Complete State**: Full entity state stored
- **Compression**: Baseline for delta compression
- **Recovery**: Fast recovery point for version reconstruction
- **Frequency**: Every 10th version (configurable)

**Delta Versions**:
- **Change-only**: Only stores changes from previous version
- **Compression**: Efficient storage using deltas
- **Reconstruction**: Requires walking back to anchor version
- **Performance**: Faster storage, slower reconstruction

### Version Metadata

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    pub branch_name: Option<String>,
    pub tags: Vec<String>,
    pub confidence_score: f32,
    pub validation_status: ValidationStatus,
    pub source: VersionSource,
    pub checksum: String,
}
```

**Metadata Features**:
- **Branching**: Support for version branches
- **Tagging**: Version labels for important milestones
- **Confidence**: Quality assessment of version
- **Validation**: Automatic validation status
- **Source Tracking**: Origin of version changes
- **Integrity**: Checksum for version integrity

### Version Cleanup and Retention

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub max_versions_per_entity: Option<usize>,
    pub max_age_days: Option<u32>,
    pub preserve_anchors: bool,
    pub preserve_tagged_versions: bool,
    pub min_versions_to_keep: usize,
}
```

**Cleanup Process**:
1. **Policy Evaluation**: Check retention policy constraints
2. **Version Filtering**: Identify versions to keep vs remove
3. **Anchor Preservation**: Always preserve anchor versions
4. **Tag Preservation**: Preserve tagged versions
5. **Minimum Retention**: Ensure minimum versions are kept
6. **Storage Cleanup**: Remove unused versions from storage

## Cross-Database Versioning

### Cross-Database Comparison

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseComparison {
    pub entity_id: String,
    pub database_versions: Vec<(DatabaseId, VersionEntry)>,
    pub consensus_fields: HashMap<String, serde_json::Value>,
    pub conflicting_fields: HashMap<String, Vec<(DatabaseId, serde_json::Value)>>,
    pub similarity_matrix: Vec<Vec<f32>>,
    pub recommendations: Vec<String>,
}
```

**Comparison Features**:
- **Multi-database Analysis**: Compare versions across databases
- **Consensus Detection**: Identify agreed-upon values
- **Conflict Identification**: Find conflicting values
- **Similarity Assessment**: Measure cross-database similarity
- **Recommendations**: Suggest resolution strategies

### Federated Versioning

**Distributed Version Control**:
- **Database-specific Versions**: Each database maintains own versions
- **Global Version Coordination**: Coordinate versions across databases
- **Conflict Resolution**: Resolve conflicts between databases
- **Synchronization**: Sync versions across federated databases

## Performance Optimizations

### Indexing Strategies

**Temporal Indices**:
- **B-Tree Indices**: Efficient range queries on time dimensions
- **Hash Indices**: Fast lookup for specific versions
- **Composite Indices**: Multi-dimensional temporal queries
- **Bloom Filters**: Efficient existence checks

**Query Optimization**:
- **Index Selection**: Choose optimal indices for queries
- **Query Rewriting**: Optimize temporal query execution
- **Caching**: Cache frequent temporal queries
- **Parallel Processing**: Parallel execution of temporal operations

### Storage Optimization

**Compression Techniques**:
- **Delta Compression**: Store only changes between versions
- **Field-level Compression**: Compress individual fields
- **Temporal Compression**: Compress time-series data
- **Metadata Compression**: Compress version metadata

**Storage Layout**:
- **Columnar Storage**: Efficient for temporal analytics
- **Time-based Partitioning**: Partition by time ranges
- **Version Clustering**: Cluster related versions
- **Hot/Cold Storage**: Separate frequently vs rarely accessed data

## Advanced Temporal Features

### Temporal Constraints

**Referential Integrity**:
- **Temporal Foreign Keys**: References valid at specific times
- **Temporal Constraints**: Constraints that vary over time
- **Consistency Enforcement**: Maintain temporal consistency
- **Validation Rules**: Temporal validation rules

### Temporal Aggregation

**Time-based Aggregation**:
- **Window Functions**: Aggregate over time windows
- **Temporal Joins**: Join across time dimensions
- **Trend Analysis**: Analyze trends over time
- **Seasonal Patterns**: Detect seasonal patterns

### Temporal Indexing Advanced Features

**Specialized Indices**:
- **R-Tree Indices**: For temporal-spatial queries
- **Interval Indices**: For time interval queries
- **Segment Indices**: For time segment queries
- **Hybrid Indices**: Combined temporal-spatial-attribute indices

## Integration with LLMKG Components

### Knowledge Graph Integration

**Temporal Knowledge Evolution**:
- **Entity Evolution**: Track entity changes over time
- **Relationship Evolution**: Track relationship changes
- **Schema Evolution**: Track schema changes
- **Concept Evolution**: Track concept development

### Cognitive Pattern Integration

**Temporal Reasoning**:
- **Historical Analysis**: Analyze historical patterns
- **Trend Prediction**: Predict future trends
- **Causal Analysis**: Identify causal relationships over time
- **Temporal Correlation**: Find temporal correlations

### Neural Network Integration

**Temporal Embeddings**:
- **Time-aware Embeddings**: Embeddings that capture temporal aspects
- **Temporal Similarity**: Similarity measures across time
- **Temporal Clustering**: Cluster entities by temporal patterns
- **Temporal Prediction**: Predict future states

## Use Cases and Applications

### Historical Analysis

**Knowledge Archaeology**:
- **Version Reconstruction**: Reconstruct historical states
- **Change Analysis**: Analyze historical changes
- **Pattern Detection**: Detect historical patterns
- **Trend Analysis**: Analyze long-term trends

### Audit and Compliance

**Audit Trail**:
- **Complete History**: Full audit trail of changes
- **Compliance Reporting**: Generate compliance reports
- **Change Attribution**: Track change authors
- **Regulatory Compliance**: Meet regulatory requirements

### Collaborative Editing

**Multi-user Editing**:
- **Concurrent Editing**: Support concurrent edits
- **Conflict Resolution**: Resolve edit conflicts
- **Merge Strategies**: Intelligent merge strategies
- **Version Control**: Git-like version control

### Experimental Research

**Research Versioning**:
- **Experiment Tracking**: Track experimental changes
- **Hypothesis Testing**: Test hypotheses over time
- **Result Comparison**: Compare experimental results
- **Reproducibility**: Ensure reproducible research

## Future Enhancements

### Planned Features

**Advanced Temporal Analytics**:
- **Machine Learning Integration**: ML-powered temporal analysis
- **Predictive Analytics**: Predict future changes
- **Anomaly Detection**: Detect temporal anomalies
- **Pattern Recognition**: Recognize complex temporal patterns

**Enhanced Collaboration**:
- **Branching Strategies**: Advanced branching support
- **Merge Algorithms**: Sophisticated merge algorithms
- **Conflict Prevention**: Prevent conflicts before they occur
- **Collaborative Features**: Real-time collaboration features

**Performance Improvements**:
- **Distributed Storage**: Distributed temporal storage
- **Parallel Processing**: Parallel temporal processing
- **GPU Acceleration**: GPU-accelerated temporal operations
- **Memory Optimization**: Advanced memory optimization

The versioning and temporal graph capabilities in LLMKG provide a comprehensive solution for tracking knowledge evolution, supporting complex temporal queries, and enabling sophisticated analysis of how knowledge changes over time. This system forms the foundation for advanced applications in research, compliance, collaboration, and historical analysis.