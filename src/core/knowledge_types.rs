use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::core::triple::Triple;
use uuid::Uuid;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

// Enhanced entity structure for Phase 1
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entity {
    pub id: Uuid,
    pub name: String,
    pub entity_type: EntityType,
    pub aliases: Vec<String>,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Concept,
    Event,
    Time,
    Quantity,
    Unknown,
}

// Question parsing types
#[derive(Debug, Clone)]
pub struct QuestionIntent {
    pub question_type: QuestionType,
    pub entities: Vec<String>,
    pub expected_answer_type: AnswerType,
    pub temporal_context: Option<TimeRange>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuestionType {
    What,
    Who,
    When,
    Where,
    Why,
    How,
    Which,
    Is,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnswerType {
    Entity,
    Fact,
    List,
    Boolean,
    Number,
    Text,
    Time,
    Location,
}

#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: Option<String>,
    pub end: Option<String>,
}

// Answer generation types
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    pub confidence: f32,
    pub facts: Vec<Triple>,
    pub entities: Vec<String>,
}