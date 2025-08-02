# Directory Overview: Versioning

## 1. High-Level Summary

The versioning directory implements a comprehensive multi-database versioning and temporal tracking system for the LLMKG knowledge graph. This module provides advanced version management capabilities including bi-temporal tracking, cross-database conflict resolution, temporal querying, and intelligent merging strategies. The system uses an anchor+delta strategy for efficient storage and supports complex operations like time travel queries, version lineage tracking, and automated conflict resolution.

## 2. Tech Stack

- **Language:** Rust
- **Key Dependencies:** 
  - `tokio` - Async runtime for concurrent operations
  - `serde` - Serialization/deserialization for version data
  - `chrono` - Date/time handling for temporal operations
  - `ahash` - High-performance hashmap implementation
  - `uuid` - Unique identifier generation
- **Storage:** In-memory with HashMap-based indexing
- **Concurrency:** Arc<RwLock<T>> for thread-safe access

## 3. Directory Structure

The module is organized into specialized components:
- **Core Types** (`types.rs`) - Fundamental data structures and enums
- **Storage Layer** (`version_store.rs`) - Per-database version storage
- **Graph Layer** (`version_graph.rs`) - Cross-database relationship tracking
- **Temporal Layer** (`temporal_graph.rs`, `temporal_query.rs`) - Time-based operations
- **Merge Engine** (`merger.rs`) - Conflict detection and resolution
- **Main Interface** (`mod.rs`) - High-level orchestration

## 4. File Breakdown

### `mod.rs`
- **Purpose:** Main entry point providing the `MultiDatabaseVersionManager` orchestrator
- **Classes:**
  - `MultiDatabaseVersionManager`
    - **Description:** Central coordinator for multi-database version management
    - **Key Fields:**
      - `version_stores: Arc<RwLock<HashMap<DatabaseId, Arc<VersionStore>>>>` - Per-database version storage
      - `global_version_graph: Arc<RwLock<VersionGraph>>` - Cross-database relationship tracking
      - `temporal_engine: Arc<TemporalQueryEngine>` - Temporal query processor
      - `merger: Arc<VersionMerger>` - Conflict resolution engine
    - **Methods:**
      - `register_database(database_id)` - Registers a new database for version management
      - `create_version(database_id, entity_id, changes, author, message)` - Creates a new version for an entity
      - `get_version_history(database_id, entity_id)` - Retrieves complete version history
      - `compare_versions(database_id, entity_id, version1, version2)` - Compares two versions
      - `compare_across_databases(entity_id, database_versions)` - Cross-database comparison
      - `temporal_query(query)` - Executes temporal queries
      - `merge_versions(database_id, entity_id, base_version, version1, version2, strategy)` - Merges conflicting versions

### `types.rs`
- **Purpose:** Core type definitions for the versioning system
- **Key Types:**
  - `VersionId` - Unique identifier for versions with timestamp-based generation
  - `SnapshotId` - Unique identifier for database snapshots
  - `FieldChange` - Represents a change to a field with old/new values and change type
  - `VersionEntry` - Complete version record with metadata, changes, and lineage
  - `VersionComparison` - Result of comparing two versions with differences and similarity
  - `TemporalQuery` - Union type for different temporal query operations
  - `MergeResult` - Result of merge operations with conflict information
- **Enums:**
  - `ChangeType` - Added, Modified, Removed, Renamed
  - `ValidationStatus` - Valid, Invalid, Pending, Warning
  - `ConflictResolution` - TakeVersion1, TakeVersion2, TakeNewest, TakeHighestConfidence, ManualResolution, SmartMerge
  - `TrendDirection` - Increasing, Decreasing, Stable, Oscillating, Unknown

### `version_store.rs`
- **Purpose:** Per-database version storage and management
- **Classes:**
  - `VersionStore`
    - **Description:** Manages version data for a specific database with anchor+delta strategy
    - **Key Fields:**
      - `database_id: DatabaseId` - Associated database identifier
      - `entity_versions: Arc<RwLock<HashMap<String, Vec<VersionEntry>>>>` - Entity to versions mapping
      - `version_index: Arc<RwLock<HashMap<VersionId, VersionEntry>>>` - Version ID to version data mapping
      - `snapshots: Arc<RwLock<HashMap<SnapshotId, SnapshotData>>>` - Database snapshots
    - **Methods:**
      - `create_version(entity_id, changes, author, message)` - Creates new version with automatic anchor detection
      - `get_version_history(entity_id)` - Returns all versions for an entity
      - `get_version(entity_id, version_id)` - Retrieves specific version
      - `compare_versions(entity_id, version1, version2)` - Compares two versions with detailed diff
      - `create_snapshot(name, description)` - Creates database snapshot
      - `cleanup_versions(policy)` - Removes old versions based on retention policy

### `version_graph.rs`
- **Purpose:** Tracks relationships between versions and databases
- **Classes:**
  - `VersionGraph`
    - **Description:** Graph structure tracking version relationships across databases
    - **Key Fields:**
      - `nodes: Arc<RwLock<HashMap<VersionNodeId, VersionNode>>>` - Version nodes
      - `edges: Arc<RwLock<HashMap<VersionNodeId, Vec<VersionEdge>>>>` - Version relationships
      - `database_nodes: Arc<RwLock<HashMap<DatabaseId, HashSet<VersionNodeId>>>>` - Database to nodes mapping
    - **Methods:**
      - `add_version(database_id, entity_id, version_id)` - Adds version node to graph
      - `add_edge(from_node, to_node, edge_type)` - Creates relationship between versions
      - `get_entity_version_history(entity_id)` - Gets cross-database version history
      - `find_cross_database_relationships(entity_id)` - Discovers cross-database relationships
      - `detect_conflicts(entity_id)` - Identifies version conflicts
      - `find_merge_candidates(entity_id)` - Suggests merge opportunities
  - `VersionNode` - Graph node representing a version with lineage information
  - `VersionEdge` - Graph edge representing relationship between versions

### `temporal_graph.rs`
- **Purpose:** Bi-temporal graph implementation with time-based entity tracking
- **Classes:**
  - `TemporalKnowledgeGraph`
    - **Description:** Bi-temporal knowledge graph supporting valid time and transaction time
    - **Key Fields:**
      - `current_graph: Arc<RwLock<KnowledgeGraph>>` - Current state graph
      - `temporal_store: Arc<RwLock<TemporalStore>>` - Temporal version storage
      - `bi_temporal_index: Arc<RwLock<BiTemporalIndex>>` - Time-based indexing
    - **Methods:**
      - `insert_temporal_entity(entity, valid_time)` - Inserts entity with temporal tracking
      - `query_at_time(valid_time, transaction_time)` - Queries graph state at specific time
      - `get_entity_history(entity_key)` - Gets complete temporal history of entity
      - `time_travel_query(query, valid_time, transaction_time)` - Executes time travel queries
      - `find_temporal_patterns(start_time, end_time, pattern)` - Discovers temporal patterns
  - `TemporalEntity` - Entity with bi-temporal tracking (valid time + transaction time)
  - `BiTemporalIndex` - Efficient indexing for temporal queries

### `temporal_query.rs`
- **Purpose:** Temporal query engine for time-based version analysis
- **Classes:**
  - `TemporalQueryEngine`
    - **Description:** Executes complex temporal queries with caching and statistics
    - **Key Fields:**
      - `query_cache: Arc<RwLock<HashMap<String, (TemporalResult, SystemTime)>>>` - Query result cache
      - `stats: Arc<RwLock<TemporalQueryStats>>` - Query execution statistics
    - **Methods:**
      - `execute_query(query, version_stores)` - Main query execution with caching
      - `execute_point_in_time_query(entity_id, database_id, timestamp, stores)` - Point-in-time entity state
      - `execute_time_range_query(entity_id, database_id, start_time, end_time, stores)` - Changes in time range
      - `execute_field_evolution_query(entity_id, database_id, field_name, time_range, stores)` - Field change tracking
      - `execute_temporal_comparison_query(entity_id, database_id, timestamps, stores)` - Multi-timestamp comparison
      - `get_statistics()` - Returns query performance metrics

### `merger.rs`
- **Purpose:** Version merger for handling conflicts and merging versions
- **Classes:**
  - `VersionMerger`
    - **Description:** Intelligent version merging with conflict resolution strategies
    - **Key Fields:**
      - `strategy_weights: HashMap<ConflictResolution, f32>` - Resolution strategy preferences
      - `merge_stats: MergeOperationStats` - Merge operation statistics
    - **Methods:**
      - `three_way_merge(store, entity_id, base_version, version1, version2, strategy)` - Three-way merge with base
      - `compare_across_databases(entity_id, database_versions)` - Cross-database comparison
      - `consensus_merge(entity_id, versions, threshold)` - Consensus-based merging
      - `detect_merge_conflicts(base_state, v1_state, v2_state)` - Conflict detection
      - `resolve_conflicts(conflicts, states, strategy)` - Applies resolution strategy
      - `suggest_conflict_resolution(base_value, v1_value, v2_value, conflict_type)` - Intelligent suggestions

## 5. Key Data Structures

### Version Entry Structure
```rust
pub struct VersionEntry {
    pub version_id: VersionId,           // Unique version identifier
    pub entity_id: String,              // Associated entity
    pub database_id: DatabaseId,        // Source database
    pub timestamp: SystemTime,          // Creation time
    pub author: Option<String>,         // Version author
    pub message: Option<String>,        // Commit message
    pub parent_versions: Vec<VersionId>, // Parent version lineage
    pub changes: Vec<FieldChange>,      // Actual changes
    pub is_anchor: bool,                // Full state vs delta
    pub compressed_size: usize,         // Storage efficiency
    pub metadata: VersionMetadata,      // Additional metadata
}
```

### Temporal Entity Structure
```rust
pub struct TemporalEntity {
    pub entity: BrainInspiredEntity,    // The actual entity
    pub valid_time: TimeRange,          // When fact was true in real world
    pub transaction_time: TimeRange,    // When fact was stored in system
    pub version_id: u64,                // Version identifier
    pub supersedes: Option<EntityKey>,  // Previous version reference
}
```

## 6. Core Algorithms

### Anchor+Delta Strategy
- Every 10th version is stored as a full anchor (complete state)
- Intermediate versions stored as deltas (changes only)
- Enables efficient storage while maintaining fast reconstruction

### Three-Way Merge Algorithm
1. Reconstruct full states for base, version1, and version2
2. Detect conflicts by comparing field values across all three versions
3. Apply resolution strategy (newest, highest confidence, smart merge, etc.)
4. Generate merged version with conflict metadata

### Bi-Temporal Indexing
- **Valid Time:** When the fact was true in the real world
- **Transaction Time:** When the fact was recorded in the system
- Enables complex temporal queries like "What did we know about X at time T?"

### Conflict Detection Logic
```rust
fn determine_conflict_type(base, v1, v2) -> Option<ConflictType> {
    match (base, v1, v2) {
        (_, Some(v1), Some(v2)) if v1 == v2 => None,                    // No conflict
        (Some(_), Some(v1), Some(v2)) if v1 != v2 => ValueConflict,     // Different changes
        (None, Some(_), None) | (None, None, Some(_)) => StructuralConflict, // Add/remove
        (_, Some(v1), Some(v2)) if different_types(v1, v2) => TypeConflict,  // Type changes
        _ => None,
    }
}
```

## 7. API Patterns

### Temporal Query Types
- **PointInTime:** Get entity state at specific timestamp
- **TimeRange:** Get all changes within time period
- **FieldEvolution:** Track how specific field changed over time
- **TemporalComparison:** Compare entity across multiple timestamps
- **ChangedEntities:** Find all entities that changed in time period

### Conflict Resolution Strategies
- **TakeVersion1/TakeVersion2:** Simple preference-based resolution
- **TakeNewest:** Use most recent change
- **TakeHighestConfidence:** Use version with highest confidence score
- **SmartMerge:** Apply intelligent merging rules
- **ManualResolution:** Require human intervention

## 8. Dependencies

### Internal Dependencies
- `crate::error::{GraphError, Result}` - Error handling
- `crate::federation::DatabaseId` - Multi-database support
- `crate::core::brain_types::{BrainInspiredEntity, BrainInspiredRelationship}` - Core entity types
- `crate::core::graph::KnowledgeGraph` - Base knowledge graph

### External Dependencies
- `tokio::sync::RwLock` - Async-safe reader-writer locks
- `serde::{Serialize, Deserialize}` - Data serialization
- `chrono::{DateTime, Utc}` - Date/time handling
- `ahash::AHashMap` - High-performance hash maps
- `std::collections::{HashMap, HashSet, BTreeMap}` - Standard collections

## 9. Performance Characteristics

### Time Complexity
- **Version Creation:** O(1) for delta versions, O(n) for anchor versions
- **Version Lookup:** O(1) with version index
- **History Retrieval:** O(k) where k is number of versions
- **Temporal Queries:** O(log n) with temporal indexing
- **Conflict Detection:** O(n*m) where n,m are number of changes

### Space Complexity
- **Storage:** O(n) with compression via anchor+delta strategy
- **Indexing:** O(n) for version index, O(t) for temporal index
- **Cache:** Bounded with LRU eviction (max 1000 entries)

### Optimization Features
- Query result caching with TTL
- Anchor+delta compression strategy
- Bi-temporal indexing for fast temporal queries
- Concurrent processing with RwLock
- Batch operations for bulk updates

## 10. Usage Examples

### Creating and Managing Versions
```rust
// Register database and create version
manager.register_database(database_id).await?;
let version_id = manager.create_version(
    &database_id,
    "entity_123", 
    vec![FieldChange { /* ... */ }],
    Some("author".to_string()),
    Some("Updated entity data".to_string())
).await?;
```

### Temporal Queries
```rust
// Query entity state at specific time
let temporal_query = TemporalQuery::PointInTime {
    entity_id: "entity_123".to_string(),
    database_id: database_id.clone(),
    timestamp: specific_time,
};
let result = manager.temporal_query(temporal_query).await?;
```

### Conflict Resolution
```rust
// Merge conflicting versions
let merge_result = manager.merge_versions(
    &database_id,
    "entity_123",
    &base_version,
    &version1,
    &version2,
    ConflictResolution::SmartMerge
).await?;
```

This versioning system provides enterprise-grade version management with advanced features like bi-temporal tracking, intelligent conflict resolution, and comprehensive temporal querying capabilities.