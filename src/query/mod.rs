pub mod rag;
pub mod optimizer;
pub mod clustering;
pub mod summarization;
pub mod two_tier;

use crate::core::graph::KnowledgeGraph;
use std::sync::Arc;

/// Real-time query structure with required fields
#[derive(Debug, Clone)]
pub struct RealtimeQuery {
    pub query_text: String,
    pub initial_nodes: Vec<u32>,
    pub max_depth: u32,
    pub similarity_threshold: f32,
    pub max_results: usize,
}

/// Partitioned Graph RAG system for distributed querying
#[derive(Debug, Clone)]
pub struct PartitionedGraphRAG {
    pub partitions: Vec<Arc<KnowledgeGraph>>,
    pub partition_strategy: PartitionStrategy,
    pub merge_strategy: MergeStrategy,
}

#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    ByNodeId,
    ByEntityType,
    ByEmbeddingCluster,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    UnionAll,
    IntersectAll,
    RankByScore,
    Custom(String),
}

impl RealtimeQuery {
    pub fn new(query_text: String) -> Self {
        Self {
            query_text,
            initial_nodes: Vec::new(),
            max_depth: 3,
            similarity_threshold: 0.7,
            max_results: 10,
        }
    }
    
    pub fn with_initial_nodes(mut self, nodes: Vec<u32>) -> Self {
        self.initial_nodes = nodes;
        self
    }
    
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }
}

impl PartitionedGraphRAG {
    pub fn new() -> Self {
        Self {
            partitions: Vec::new(),
            partition_strategy: PartitionStrategy::ByNodeId,
            merge_strategy: MergeStrategy::RankByScore,
        }
    }
    
    pub fn add_partition(&mut self, graph: Arc<KnowledgeGraph>) {
        self.partitions.push(graph);
    }
    
    pub fn with_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.partition_strategy = strategy;
        self
    }
}

impl Default for PartitionedGraphRAG {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Arc<KnowledgeGraph>> for PartitionedGraphRAG {
    fn from(graph: Arc<KnowledgeGraph>) -> Self {
        let mut rag = Self::new();
        rag.add_partition(graph);
        rag
    }
}