use std::collections::{HashSet, HashMap};
use serde::{Deserialize, Serialize};

/// Sparse Distributed Representation (SDR) configuration
#[derive(Debug, Clone)]
pub struct SDRConfig {
    pub total_bits: usize,
    pub active_bits: usize,
    pub sparsity: f32,
    pub overlap_threshold: f32,
}

impl Default for SDRConfig {
    fn default() -> Self {
        Self {
            total_bits: 2048,
            active_bits: 40,
            sparsity: 0.02, // 2% sparsity
            overlap_threshold: 0.5,
        }
    }
}

/// Sparse Distributed Representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDR {
    pub active_bits: HashSet<usize>,
    pub total_bits: usize,
    pub timestamp: std::time::SystemTime,
}

/// SDR pattern for representing complex concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRPattern {
    pub pattern_id: String,
    pub sdr: SDR,
    pub concept_name: String,
    pub confidence: f32,
    pub creation_time: std::time::SystemTime,
    pub usage_count: u64,
}

impl SDRPattern {
    pub fn new(pattern_id: String, sdr: SDR, concept_name: String) -> Self {
        Self {
            pattern_id,
            sdr,
            concept_name,
            confidence: 1.0,
            creation_time: std::time::SystemTime::now(),
            usage_count: 0,
        }
    }
}

/// Statistics about SDR storage
#[derive(Debug, Clone)]
pub struct SDRStatistics {
    pub total_patterns: usize,
    pub total_entities: usize,
    pub average_sparsity: f32,
    pub total_active_bits: usize,
    pub config: SDRConfig,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilaritySearchResult {
    pub pattern_id: String,
    pub content: String,
    pub similarity: f32,
}

/// SDR Entry for storing entities with SDR representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDREntry {
    pub id: String,
    pub embedding: Vec<f32>,
    pub properties: HashMap<String, String>,
    pub activation: f32,
}

/// SDR Query for searching SDR patterns
#[derive(Debug, Clone)]
pub struct SDRQuery {
    pub query_sdr: SDR,
    pub top_k: usize,
    pub min_overlap: f32,
}