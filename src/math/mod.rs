// Mathematical operations module for multi-database systems
// Implements distributed graph algorithms and similarity computations

pub mod similarity;
pub mod graph_algorithms;
pub mod distributed_math;
pub mod types;

pub use similarity::{SimilarityEngine, SimilarityMetric};
pub use graph_algorithms::{GraphAlgorithms, PageRankResult, DijkstraResult as ShortestPathResult};
pub use distributed_math::DistributedMathEngine;
pub use types::*;

use crate::error::Result;

/// Main mathematical operations coordinator for multi-database systems
pub struct MathEngine {
    similarity_engine: SimilarityEngine,
    graph_algorithm_engine: GraphAlgorithms,
    distributed_engine: DistributedMathEngine,
}

impl MathEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            similarity_engine: SimilarityEngine::new(),
            graph_algorithm_engine: GraphAlgorithms::new(),
            distributed_engine: DistributedMathEngine::new(),
        })
    }

    /// Calculate cosine similarity between vectors
    pub fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        self.similarity_engine.cosine_similarity(vec1, vec2)
    }

    /// Calculate Euclidean distance between vectors
    pub fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        self.similarity_engine.euclidean_distance(vec1, vec2)
    }

    /// Calculate Jaccard similarity for sets
    pub fn jaccard_similarity<T>(&self, set1: &[T], set2: &[T]) -> f32
    where
        T: PartialEq + Clone + std::hash::Hash + Eq,
    {
        self.similarity_engine.jaccard_similarity(set1, set2)
    }

    /// Get graph algorithms engine
    pub fn graph_algorithms(&self) -> &GraphAlgorithms {
        &self.graph_algorithm_engine
    }

    /// Get distributed math engine
    pub fn distributed_engine(&self) -> &DistributedMathEngine {
        &self.distributed_engine
    }
}