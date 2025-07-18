// Core types for versioning and temporal functionality

use crate::federation::DatabaseId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Unique identifier for a version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionId(pub String);

impl VersionId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Generate a new version ID based on timestamp and database
    pub fn generate(database_id: &DatabaseId, entity_id: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        Self(format!("{}:{}:{}", database_id.as_str(), entity_id, timestamp))
    }
}

/// Unique identifier for a snapshot
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(pub String);

impl SnapshotId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Represents a change to a field in an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldChange {
    pub field_name: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub change_type: ChangeType,
}

/// Types of changes that can occur to a field
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    Added,      // Field was added
    Modified,   // Field value was changed
    Removed,    // Field was removed
    Renamed,    // Field was renamed
}

/// Version entry with metadata and changes
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

/// Additional metadata for a version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    pub branch_name: Option<String>,
    pub tags: Vec<String>,
    pub confidence_score: f32,
    pub validation_status: ValidationStatus,
    pub source: VersionSource,
    pub checksum: String,
}

/// Status of version validation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Pending,
    Warning,
}

/// Source of the version
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionSource {
    UserEdit,
    AutomaticExtraction,
    Import,
    Merge,
    Migration,
}

/// Comparison result between two versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    pub entity_id: String,
    pub version1: VersionId,
    pub version2: VersionId,
    pub differences: Vec<FieldDifference>,
    pub similarity_score: f32,
    pub change_summary: ChangeSummary,
}

/// Difference between two field values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDifference {
    pub field_name: String,
    pub difference_type: DifferenceType,
    pub version1_value: Option<serde_json::Value>,
    pub version2_value: Option<serde_json::Value>,
    pub similarity_score: f32,
}

/// Types of differences between field values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferenceType {
    ValueChanged,
    FieldAdded,
    FieldRemoved,
    TypeChanged,
    FormatChanged,
    StructureChanged,
}

/// Summary of changes between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSummary {
    pub fields_added: usize,
    pub fields_modified: usize,
    pub fields_removed: usize,
    pub major_changes: Vec<String>,
    pub change_categories: HashMap<String, usize>,
}

/// Comparison across multiple databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseComparison {
    pub entity_id: String,
    pub database_versions: Vec<(DatabaseId, VersionEntry)>,
    pub consensus_fields: HashMap<String, serde_json::Value>,
    pub conflicting_fields: HashMap<String, Vec<(DatabaseId, serde_json::Value)>>,
    pub similarity_matrix: Vec<Vec<f32>>,
    pub recommendations: Vec<String>,
}

/// Temporal query types for time-based analysis
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

/// Result of a temporal query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalResult {
    pub query_type: String,
    pub execution_time_ms: u64,
    pub result_data: TemporalResultData,
}

/// Different types of temporal query results
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

/// Trend analysis for field evolution
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

/// Merge result with conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub success: bool,
    pub merged_version: Option<VersionId>,
    pub conflicts: Vec<MergeConflict>,
    pub resolution_applied: ConflictResolution,
    pub merge_statistics: MergeStatistics,
}

/// Represents a merge conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    pub field_name: String,
    pub base_value: Option<serde_json::Value>,
    pub version1_value: Option<serde_json::Value>,
    pub version2_value: Option<serde_json::Value>,
    pub conflict_type: ConflictType,
    pub suggested_resolution: Option<serde_json::Value>,
}

/// Types of merge conflicts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    ValueConflict,      // Different values for same field
    StructuralConflict, // Structural changes conflict
    TypeConflict,       // Type changes conflict
    RenameConflict,     // Field rename conflicts
}

/// Conflict resolution strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictResolution {
    TakeVersion1,       // Always use version 1
    TakeVersion2,       // Always use version 2
    TakeNewest,         // Use the most recent change
    TakeHighestConfidence, // Use version with highest confidence
    ManualResolution,   // Require manual intervention
    SmartMerge,         // Apply intelligent merging rules
}

/// Statistics about a merge operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStatistics {
    pub total_fields: usize,
    pub fields_merged_automatically: usize,
    pub fields_requiring_resolution: usize,
    pub merge_time_ms: u64,
}

/// Version statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStatistics {
    pub total_versions: usize,
    pub total_entities: usize,
    pub total_databases: usize,
    pub per_database_stats: HashMap<DatabaseId, DatabaseVersionStats>,
}

/// Version statistics for a specific database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseVersionStats {
    pub total_versions: usize,
    pub total_entities: usize,
    pub anchor_versions: usize,
    pub delta_versions: usize,
    pub total_storage_bytes: usize,
    pub compression_ratio: f32,
    pub average_version_size: f32,
}

/// Retention policy for version cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub max_versions_per_entity: Option<usize>,
    pub max_age_days: Option<u32>,
    pub preserve_anchors: bool,
    pub preserve_tagged_versions: bool,
    pub min_versions_to_keep: usize,
}

/// Result of a cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    pub versions_removed: usize,
    pub storage_freed_bytes: usize,
    pub entities_affected: usize,
    pub cleanup_time_ms: u64,
}