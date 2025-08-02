// Core types for mathematical operations

use crate::federation::{DatabaseId, FederatedEntityKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mathematical operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalResult {
    pub operation_type: String,
    pub execution_time_ms: u64,
    pub databases_involved: Vec<DatabaseId>,
    pub result_data: MathResultData,
    pub metadata: MathMetadata,
}

/// Different types of mathematical result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathResultData {
    Scalar {
        value: f64,
        unit: Option<String>,
    },
    Vector {
        values: Vec<f64>,
        labels: Option<Vec<String>>,
    },
    Matrix {
        values: Vec<Vec<f64>>,
        row_labels: Option<Vec<String>>,
        col_labels: Option<Vec<String>>,
    },
    Graph {
        edges: Vec<(FederatedEntityKey, FederatedEntityKey, f64)>,
        node_properties: HashMap<FederatedEntityKey, HashMap<String, f64>>,
    },
    Rankings {
        rankings: Vec<(FederatedEntityKey, f64)>,
        ranking_type: String,
    },
    Similarity {
        pairs: Vec<SimilarityPair>,
        similarity_type: String,
    },
}

/// Metadata for mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathMetadata {
    pub algorithm_used: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub convergence_iterations: Option<usize>,
    pub approximation_error: Option<f64>,
    pub memory_usage_mb: f64,
    pub network_operations: usize,
}

/// Similarity pair result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityPair {
    pub entity1: FederatedEntityKey,
    pub entity2: FederatedEntityKey,
    pub similarity_score: f32,
    pub similarity_breakdown: Option<HashMap<String, f32>>,
}

/// Relationship strength calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipStrength {
    pub entity1: FederatedEntityKey,
    pub entity2: FederatedEntityKey,
    pub overall_strength: f32,
    pub individual_metrics: HashMap<String, f32>,
    pub confidence_score: f32,
    pub relationship_types: Vec<String>,
}

/// PageRank calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResult {
    pub rankings: Vec<(FederatedEntityKey, f32)>,
    pub iterations_to_convergence: usize,
    pub convergence_threshold: f32,
    pub damping_factor: f32,
    pub total_entities: usize,
    pub execution_time_ms: u64,
}

/// Shortest path calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathResult {
    pub source: FederatedEntityKey,
    pub target: FederatedEntityKey,
    pub path_found: bool,
    pub path_length: usize,
    pub path: Vec<FederatedEntityKey>,
    pub path_weights: Vec<f32>,
    pub total_weight: f32,
}

/// Centrality calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityResult {
    pub entity: FederatedEntityKey,
    pub centrality_type: CentralityType,
    pub score: f32,
    pub rank: usize,
    pub percentile: f32,
}

/// Types of centrality measures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentralityType {
    Betweenness,
    Closeness,
    Eigenvector,
    PageRank,
    Degree,
}

/// Clustering coefficient result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    pub entity: FederatedEntityKey,
    pub clustering_coefficient: f32,
    pub local_clustering: f32,
    pub global_clustering: f32,
    pub neighbor_count: usize,
}

/// Similarity metrics available
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Jaccard,
    Pearson,
    Spearman,
    Manhattan,
    Hamming,
}

/// Similarity calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub entity1: FederatedEntityKey,
    pub entity2: FederatedEntityKey,
    pub similarity_score: f32,
    pub metric_used: SimilarityMetric,
    pub confidence: f32,
    pub metadata: SimilarityMetadata,
}

/// Metadata for similarity calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetadata {
    pub vector_dimensions: Option<usize>,
    pub missing_values_handled: bool,
    pub normalization_applied: bool,
    pub computation_method: String,
}

/// Distributed operation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedOperation {
    pub operation_type: DistributedOperationType,
    pub target_entities: Vec<FederatedEntityKey>,
    pub databases: Vec<DatabaseId>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub parallel_workers: Option<usize>,
    pub timeout_seconds: Option<u64>,
}

/// Types of distributed operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedOperationType {
    PageRank,
    BetweennessCentrality,
    ClosenessCentrality,
    CommunityDetection,
    ShortestPaths,
    SimilarityMatrix,
    GraphStatistics,
}

/// Graph statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub total_entities: usize,
    pub total_relationships: usize,
    pub databases_included: Vec<DatabaseId>,
    pub average_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub density: f32,
    pub connected_components: usize,
    pub diameter: Option<usize>,
    pub clustering_coefficient: f32,
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    pub communities: Vec<Community>,
    pub modularity_score: f32,
    pub algorithm_used: String,
    pub total_entities: usize,
    pub cross_database_communities: usize,
}

/// A detected community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: String,
    pub entities: Vec<FederatedEntityKey>,
    pub internal_density: f32,
    pub external_connectivity: f32,
    pub dominant_databases: Vec<DatabaseId>,
    pub representative_entities: Vec<FederatedEntityKey>,
}

/// Performance metrics for mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation_name: String,
    pub total_time_ms: u64,
    pub computation_time_ms: u64,
    pub network_time_ms: u64,
    pub memory_peak_mb: f64,
    pub memory_average_mb: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub entities_processed: usize,
    pub relationships_processed: usize,
}

/// Optimization parameters for mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    pub use_approximation: bool,
    pub approximation_factor: Option<f32>,
    pub max_iterations: Option<usize>,
    pub convergence_threshold: Option<f32>,
    pub parallel_execution: bool,
    pub cache_results: bool,
    pub batch_size: Option<usize>,
    pub memory_limit_mb: Option<usize>,
}

/// Error types specific to mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathError {
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InsufficientData {
        required: usize,
        available: usize,
    },
    ConvergenceFailure {
        iterations: usize,
        final_error: f64,
    },
    InvalidParameters {
        parameter: String,
        message: String,
    },
    DatabaseUnavailable {
        database_id: DatabaseId,
    },
    ComputationTimeout {
        operation: String,
        timeout_seconds: u64,
    },
}

/// Result wrapper for mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathOperationResult<T> {
    pub success: bool,
    pub result: Option<T>,
    pub error: Option<MathError>,
    pub performance: PerformanceMetrics,
    pub warnings: Vec<String>,
}

impl<T> MathOperationResult<T> {
    pub fn success(result: T, performance: PerformanceMetrics) -> Self {
        Self {
            success: true,
            result: Some(result),
            error: None,
            performance,
            warnings: Vec::new(),
        }
    }

    pub fn failure(error: MathError, performance: PerformanceMetrics) -> Self {
        Self {
            success: false,
            result: None,
            error: Some(error),
            performance,
            warnings: Vec::new(),
        }
    }

    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}