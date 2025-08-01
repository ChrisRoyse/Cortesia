// Core types for multi-database federation

use crate::core::types::EntityKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for a database in the federation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatabaseId(pub String);

impl DatabaseId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Federated entity key that includes database context
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FederatedEntityKey {
    pub database_id: DatabaseId,
    pub entity_key: EntityKey,
}

impl FederatedEntityKey {
    pub fn new(database_id: DatabaseId, entity_key: EntityKey) -> Self {
        Self { database_id, entity_key }
    }
}

/// Cross-database relationship representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseRelationship {
    pub from_db: DatabaseId,
    pub from_entity: String,  // Entity ID as string for cross-database compatibility
    pub to_db: DatabaseId,
    pub to_entity: String,    // Entity ID as string for cross-database compatibility
    pub rel_type: String,
    pub confidence: f32,
    pub metadata: serde_json::Value,
}

/// Database capabilities that determine what operations are supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseCapabilities {
    pub supports_versioning: bool,
    pub supports_vector_similarity: bool,
    pub supports_temporal_queries: bool,
    pub supports_graph_algorithms: bool,
    pub supported_math_operations: Vec<MathOperation>,
    pub max_entities: Option<usize>,
    pub max_query_complexity: Option<u32>,
    pub supports_transactions: bool,
    pub supports_batch_operations: bool,
}

impl Default for DatabaseCapabilities {
    fn default() -> Self {
        Self {
            supports_versioning: true,
            supports_vector_similarity: true,
            supports_temporal_queries: true,
            supports_graph_algorithms: true,
            supported_math_operations: vec![
                MathOperation::CosineSimilarity,
                MathOperation::EuclideanDistance,
            ],
            max_entities: Some(1_000_000),
            max_query_complexity: Some(1000),
            supports_transactions: true,
            supports_batch_operations: true,
        }
    }
}

/// Mathematical operations that can be performed across databases
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathOperation {
    CosineSimilarity,
    EuclideanDistance,
    JaccardSimilarity,
    PageRank,
    ShortestPath,
    ClusteringCoefficient,
    BetweennessCentrality,
    EigenvectorCentrality,
}

/// Types of queries that can be executed in a federated manner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    /// Find similar entities across multiple databases
    CrossDatabaseSimilarity {
        query_vector: Vec<f32>,
        databases: Vec<DatabaseId>,
        similarity_threshold: f32,
        max_results: usize,
    },
    /// Similarity search using text or embeddings
    SimilaritySearch(SimilarityQuery),
    /// Compare the same entity across different databases
    EntityComparison {
        entity_id: String,
        databases: Vec<DatabaseId>,
        comparison_fields: Vec<String>,
    },
    /// Find relationships between entities in different databases
    CrossDatabaseRelationship {
        source_db: DatabaseId,
        source_entity: EntityKey,
        target_db: DatabaseId,
        relationship_types: Vec<String>,
        max_hops: u8,
    },
    /// Execute mathematical operations across databases
    MathematicalOperation {
        operation: MathOperation,
        databases: Vec<DatabaseId>,
        parameters: HashMap<String, serde_json::Value>,
    },
    /// Aggregate data from multiple databases
    AggregateQuery {
        operation: AggregateFunction,
        databases: Vec<DatabaseId>,
        filter_criteria: FilterCriteria,
    },
}

/// Similarity query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityQuery {
    pub query_text: String,
    pub threshold: f32,
    pub max_results: usize,
    pub embedding_model: String,
}

/// Federated query structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQuery {
    pub query_id: String,
    pub query_type: QueryType,
    pub target_databases: Vec<DatabaseId>,
    pub merge_strategy: MergeStrategy,
    pub timeout_ms: u64,
}

impl FederatedQuery {
    /// Get the merge strategy for this query type
    pub fn merge_strategy(&self) -> MergeStrategy {
        match &self.query_type {
            QueryType::CrossDatabaseSimilarity { .. } => MergeStrategy::SimilarityMerge,
            QueryType::SimilaritySearch(_) => MergeStrategy::SimilarityMerge,
            QueryType::EntityComparison { .. } => MergeStrategy::ComparisonMerge,
            QueryType::CrossDatabaseRelationship { .. } => MergeStrategy::RelationshipMerge,
            QueryType::MathematicalOperation { .. } => MergeStrategy::MathematicalMerge,
            QueryType::AggregateQuery { .. } => MergeStrategy::AggregationMerge,
        }
    }

    /// Get the list of databases involved in this query
    pub fn target_databases(&self) -> Vec<DatabaseId> {
        self.target_databases.clone()
    }
}

/// Strategies for merging results from multiple databases
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MergeStrategy {
    SimilarityMerge,    // Merge by similarity scores
    ComparisonMerge,    // Side-by-side comparison
    RelationshipMerge,  // Merge relationship graphs
    MathematicalMerge,  // Combine mathematical results
    AggregationMerge,   // Aggregate numerical values
    UnionMerge,         // Simple union of results
    IntersectionMerge,  // Only common results
}

/// Aggregate functions for multi-database operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Average,
    Min,
    Max,
    StandardDeviation,
    Median,
}

/// Filter criteria for aggregate queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCriteria {
    pub entity_types: Option<Vec<String>>,
    pub relationship_types: Option<Vec<String>>,
    pub confidence_threshold: Option<f32>,
    pub time_range: Option<(SystemTime, SystemTime)>,
    pub custom_filters: HashMap<String, serde_json::Value>,
}

/// Result of a federated query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQueryResult {
    pub query_id: String,
    pub execution_time_ms: u64,
    pub databases_queried: Vec<DatabaseId>,
    pub total_results: usize,
    pub results: QueryResultData,
    pub metadata: QueryMetadata,
}

/// Different types of query result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResultData {
    SimilarityResults(Vec<SimilarityMatch>),
    ComparisonResults(Vec<EntityComparison>),
    RelationshipResults(Vec<CrossDatabaseRelationship>),
    MathematicalResults(MathematicalResult),
    AggregateResults(AggregateResult),
}

/// Similarity match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    pub entity: String,  // Entity ID as string for flexibility
    pub similarity_score: f32,
    pub metadata: serde_json::Value,  // Flexible metadata
}

/// Entity comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityComparison {
    pub entity_id: String,
    pub database_versions: Vec<EntityVersion>,
    pub differences: Vec<FieldDifference>,
    pub similarity_score: f32,
}

/// Version of an entity in a specific database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityVersion {
    pub database_id: DatabaseId,
    pub version_id: String,
    pub timestamp: u64,  // Unix timestamp for compatibility
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Difference between entity fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDifference {
    pub field_name: String,
    pub database_values: HashMap<DatabaseId, serde_json::Value>,
    pub difference_type: DifferenceType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferenceType {
    ValueMismatch,
    MissingInSome,
    TypeMismatch,
    FormatDifference,
}

/// Mathematical operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalResult {
    pub operation: MathOperation,
    pub result_type: MathResultType,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathResultType {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Graph(Vec<(FederatedEntityKey, FederatedEntityKey, f64)>),
    Rankings(Vec<(FederatedEntityKey, f64)>),
}

/// Aggregate operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateResult {
    pub function: AggregateFunction,
    pub value: f64,
    pub count: usize,
    pub per_database: HashMap<DatabaseId, f64>,
}

/// Query execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub query_plan: String,
    pub optimization_used: Vec<String>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub network_round_trips: usize,
    pub data_transferred_bytes: usize,
}

/// Database health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseHealth {
    pub database_id: DatabaseId,
    pub is_healthy: bool,
    pub response_time_ms: Option<u64>,
    pub last_error: Option<String>,
    pub capabilities: DatabaseCapabilities,
    pub entity_count: Option<usize>,
    pub memory_usage_mb: Option<f64>,
}

/// Generate a unique query ID
pub fn generate_query_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("query_{timestamp}")
}