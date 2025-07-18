use crate::core::types::{EntityKey, GraphQuery, TraversalParams};
use crate::cognitive::types::{Pattern, Path, Subgraph, StructureAnalysis};
use crate::error::Result;
use std::collections::HashMap;

/// Graph-based query engine for cognitive operations
/// Replaces NeuralProcessingServer with pure graph operations
#[async_trait::async_trait]
pub trait GraphQueryEngine: Send + Sync {
    /// Find patterns in graph structure
    async fn find_patterns(&self, query: GraphQuery) -> Result<Vec<Pattern>>;
    
    /// Traverse paths through the graph
    async fn traverse_paths(&self, start: EntityKey, params: TraversalParams) -> Result<Vec<Path>>;
    
    /// Analyze subgraph structure
    async fn analyze_structure(&self, subgraph: &Subgraph) -> Result<StructureAnalysis>;
    
    /// Compute similarity between entities based on graph structure
    async fn compute_similarity(&self, entity_a: EntityKey, entity_b: EntityKey) -> Result<f32>;
    
    /// Find intersection nodes between themes/patterns
    async fn find_intersection_nodes(&self, themes: &[Pattern]) -> Result<Vec<EntityKey>>;
    
    /// Analyze shared edges between patterns
    async fn analyze_shared_edges(&self, themes: &[Pattern]) -> Result<Vec<(EntityKey, EntityKey)>>;
    
    /// Calculate convergence point from nodes and relationships
    async fn calculate_convergence(&self, nodes: &[EntityKey], edges: &[(EntityKey, EntityKey)]) -> Result<EntityKey>;
    
    /// Calculate confidence score for patterns
    async fn calculate_confidence(&self, patterns: &[Pattern]) -> Result<f32>;
    
    /// Get entity subgraph representation
    async fn get_entity_subgraph(&self, entity: EntityKey) -> Result<Subgraph>;
    
    /// Calculate activation state for working memory
    async fn calculate_activation_state(&self, subgraph: &Subgraph) -> Result<f32>;
    
    /// Compute entity vector using graph structure (replaces neural embeddings)
    async fn compute_entity_vector(&self, entity: EntityKey) -> Result<Vec<f32>>;
    
    /// Traverse reasoning path through relationships (replaces neural reasoning)
    async fn traverse_reasoning_path(&self, start: EntityKey, goal: EntityKey) -> Result<Vec<Path>>;
    
    /// Generate patterns from graph context (replaces neural completion)
    async fn generate_from_patterns(&self, context: &Subgraph) -> Result<Vec<Pattern>>;
    
    /// Classify entities by graph topology (replaces neural classification)
    async fn classify_by_graph_topology(&self, entity: EntityKey) -> Result<String>;
    
    /// Find concept clusters in the graph
    async fn find_concept_clusters(&self, min_size: usize) -> Result<Vec<Vec<EntityKey>>>;
    
    /// Detect cycles in the graph (for systems thinking)
    async fn detect_cycles(&self, max_length: usize) -> Result<Vec<Vec<EntityKey>>>;
    
    /// Find bridge nodes that connect different domains
    async fn find_bridge_nodes(&self, domain_a: &str, domain_b: &str) -> Result<Vec<EntityKey>>;
    
    /// Calculate centrality of nodes
    async fn calculate_centrality(&self, entities: &[EntityKey]) -> Result<HashMap<EntityKey, f32>>;
    
    /// Find analogies through structural similarity
    async fn find_analogies(&self, source: EntityKey, target_domain: &str) -> Result<Vec<(EntityKey, f32)>>;
    
    /// Expand concepts through graph exploration
    async fn expand_concepts(&self, seed: EntityKey, max_depth: usize) -> Result<Vec<EntityKey>>;
}

/// Parameters for graph traversal
#[derive(Debug, Clone)]
pub struct GraphTraversalParams {
    pub max_depth: usize,
    pub max_paths: usize,
    pub include_bidirectional: bool,
    pub edge_weight_threshold: Option<f32>,
    pub required_edge_types: Option<Vec<String>>,
    pub excluded_nodes: Option<Vec<EntityKey>>,
}

impl Default for GraphTraversalParams {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_paths: 100,
            include_bidirectional: true,
            edge_weight_threshold: None,
            required_edge_types: None,
            excluded_nodes: None,
        }
    }
}

/// Result of structural analysis
#[derive(Debug, Clone)]
pub struct GraphStructureAnalysis {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f32,
    pub clustering_coefficient: f32,
    pub connected_components: usize,
    pub average_path_length: f32,
    pub diameter: usize,
    pub modularity: f32,
}

/// Synthesis result from convergent thinking
#[derive(Debug, Clone)]
pub struct GraphSynthesis {
    pub central_concept: EntityKey,
    pub supporting_patterns: Vec<EntityKey>,
    pub confidence: f32,
    pub explanation: String,
}