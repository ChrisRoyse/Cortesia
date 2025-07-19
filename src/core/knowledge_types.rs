use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::core::triple::Triple;

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_nodes: usize,
    pub total_triples: usize,
    pub total_bytes: usize,
    pub bytes_per_node: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Search parameters optimized for LLM queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleQuery {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub limit: usize,
    pub min_confidence: f32,
    pub include_chunks: bool,
}

/// LLM-friendly search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub nodes: Vec<crate::core::triple::KnowledgeNode>,
    pub triples: Vec<Triple>,
    pub entity_context: HashMap<String, EntityContext>,
    pub query_time_ms: u64,
    pub total_found: usize,
}

impl KnowledgeResult {
    pub fn iter(&self) -> std::slice::Iter<Triple> {
        self.triples.iter()
    }
    
    pub fn len(&self) -> usize {
        self.triples.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

impl IntoIterator for KnowledgeResult {
    type Item = Triple;
    type IntoIter = std::vec::IntoIter<Triple>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.triples.into_iter()
    }
}

impl<'a> IntoIterator for &'a KnowledgeResult {
    type Item = &'a Triple;
    type IntoIter = std::slice::Iter<'a, Triple>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.triples.iter()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityContext {
    pub entity_name: String,
    pub entity_type: String,
    pub description: String,
    pub related_triples: Vec<Triple>,
    pub confidence_score: f32,
}