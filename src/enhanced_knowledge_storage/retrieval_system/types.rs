//! Retrieval System Types
//! 
//! Core types for the advanced retrieval system with multi-hop reasoning
//! and context-aware search capabilities.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::enhanced_knowledge_storage::{
    types::*,
    hierarchical_storage::types::*,
    knowledge_processing::types::EntityType,
};

/// Query for the retrieval system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub natural_language_query: String,
    pub structured_constraints: Option<StructuredConstraints>,
    pub retrieval_mode: RetrievalMode,
    pub max_results: usize,
    pub min_relevance_score: f32,
    pub enable_multi_hop: bool,
    pub max_reasoning_hops: u32,
    pub context_window_size: usize,
    pub enable_query_expansion: bool,
    pub enable_temporal_filtering: bool,
    pub time_range: Option<TimeRange>,
}

impl Default for RetrievalQuery {
    fn default() -> Self {
        Self {
            natural_language_query: String::new(),
            structured_constraints: None,
            retrieval_mode: RetrievalMode::Hybrid,
            max_results: 10,
            min_relevance_score: 0.5,
            enable_multi_hop: true,
            max_reasoning_hops: 3,
            context_window_size: 1000,
            enable_query_expansion: true,
            enable_temporal_filtering: false,
            time_range: None,
        }
    }
}

/// Structured constraints for precise retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredConstraints {
    pub required_entities: Vec<String>,
    pub required_relationships: Vec<String>,
    pub required_concepts: Vec<String>,
    pub layer_types: Vec<LayerType>,
    pub exclude_patterns: Vec<String>,
    pub metadata_filters: HashMap<String, String>,
}

/// Retrieval modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    Exact,           // Exact keyword matching
    Semantic,        // Semantic similarity based
    Hybrid,          // Combination of exact and semantic
    GraphTraversal,  // Follow semantic links
    MultiHop,        // Multi-hop reasoning
}

/// Time range for temporal filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

/// Result from retrieval system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub query_id: String,
    pub retrieved_items: Vec<RetrievedItem>,
    pub reasoning_chain: Option<ReasoningChain>,
    pub query_understanding: QueryUnderstanding,
    pub total_matches: usize,
    pub retrieval_time_ms: u64,
    pub confidence_score: f32,
}

/// Individual retrieved item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedItem {
    pub layer_id: String,
    pub document_id: String,
    pub content: String,
    pub relevance_score: f32,
    pub match_explanation: MatchExplanation,
    pub context_before: Option<String>,
    pub context_after: Option<String>,
    pub layer_type: LayerType,
    pub importance_score: f32,
    pub semantic_links: Vec<LinkedLayer>,
}

/// Explanation of why an item matched
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchExplanation {
    pub matched_keywords: Vec<String>,
    pub matched_entities: Vec<String>,
    pub matched_concepts: Vec<String>,
    pub semantic_similarity: Option<f32>,
    pub reasoning_steps: Vec<String>,
    pub match_type: MatchType,
}

/// Types of matches
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    ExactKeyword,
    SemanticSimilarity,
    EntityReference,
    ConceptMatch,
    RelationshipMatch,
    MultiHopInference,
    ContextualRelevance,
}

/// Linked layer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedLayer {
    pub layer_id: String,
    pub link_type: SemanticLinkType,
    pub link_strength: f32,
    pub summary: String,
}

/// Multi-hop reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub reasoning_steps: Vec<ReasoningStep>,
    pub final_conclusion: String,
    pub confidence: f32,
    pub evidence_strength: f32,
    pub reasoning_type: ReasoningType,
}

/// Individual reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: u32,
    pub hypothesis: String,
    pub supporting_evidence: Vec<String>,
    pub layer_ids: Vec<String>,
    pub inference: String,
    pub confidence: f32,
    pub step_type: StepType,
}

/// Types of reasoning
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasoningType {
    Deductive,
    Inductive,
    Abductive,
    Analogical,
    Causal,
    Temporal,
}

/// Types of reasoning steps
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepType {
    DirectEvidence,
    InferredConnection,
    TransitiveRelation,
    CausalLink,
    TemporalSequence,
    ConceptualBridge,
}

/// Query understanding analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstanding {
    pub intent: QueryIntent,
    pub extracted_entities: Vec<String>,
    pub extracted_concepts: Vec<String>,
    pub temporal_context: Option<String>,
    pub complexity_level: ComplexityLevel,
    pub suggested_expansions: Vec<String>,
    pub ambiguities: Vec<String>,
}

/// Query intent types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryIntent {
    FactualLookup,      // Looking for specific facts
    ConceptExploration, // Exploring concepts
    RelationshipQuery,  // Finding relationships
    CausalAnalysis,     // Understanding cause-effect
    TemporalSequence,   // Time-based queries
    Comparison,         // Comparing entities/concepts
    Definition,         // Seeking definitions
    Example,           // Looking for examples
    Aggregation,       // Aggregating information
}

/// Configuration for retrieval system
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub embedding_model_id: String,
    pub reasoning_model_id: String,
    pub max_parallel_searches: usize,
    pub cache_search_results: bool,
    pub cache_ttl_seconds: u64,
    pub enable_fuzzy_matching: bool,
    pub fuzzy_threshold: f32,
    pub context_overlap_tokens: usize,
    pub enable_result_reranking: bool,
    pub reranking_model_id: Option<String>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            embedding_model_id: "minilm_l6_v2".to_string(),
            reasoning_model_id: "smollm2_360m".to_string(),
            max_parallel_searches: 5,
            cache_search_results: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_fuzzy_matching: true,
            fuzzy_threshold: 0.8,
            context_overlap_tokens: 50,
            enable_result_reranking: true,
            reranking_model_id: Some("smollm2_135m".to_string()),
        }
    }
}

/// Search cache entry
#[derive(Debug, Clone)]
pub struct SearchCacheEntry {
    pub query_hash: u64,
    pub results: Vec<RetrievedItem>,
    pub timestamp: u64,
    pub access_count: u32,
}

/// Context aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedContext {
    pub primary_content: String,
    pub supporting_contexts: Vec<SupportingContext>,
    pub entity_summary: HashMap<String, EntityContext>,
    pub relationship_summary: Vec<RelationshipContext>,
    pub temporal_flow: Option<TemporalFlow>,
    pub coherence_score: f32,
}

/// Supporting context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportingContext {
    pub layer_id: String,
    pub content_snippet: String,
    pub relevance_to_primary: f32,
    pub link_path: Vec<SemanticLinkType>,
}

/// Entity context summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityContext {
    pub entity_name: String,
    pub entity_type: EntityType,
    pub occurrences: Vec<EntityOccurrence>,
    pub relationships: Vec<String>,
    pub importance_in_context: f32,
}

/// Entity occurrence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityOccurrence {
    pub layer_id: String,
    pub context: String,
    pub confidence: f32,
}

/// Relationship context summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipContext {
    pub source: String,
    pub predicate: String,
    pub target: String,
    pub occurrences: Vec<String>, // Layer IDs
    pub strength: f32,
}

/// Temporal flow information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFlow {
    pub events: Vec<TemporalEvent>,
    pub sequence_confidence: f32,
}

/// Temporal event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    pub description: String,
    pub layer_ids: Vec<String>,
    pub temporal_marker: Option<String>,
    pub sequence_position: u32,
}

/// Query expansion result
#[derive(Debug, Clone)]
pub struct QueryExpansion {
    pub original_query: String,
    pub expanded_terms: Vec<ExpandedTerm>,
    pub related_entities: Vec<String>,
    pub related_concepts: Vec<String>,
    pub suggested_filters: Vec<String>,
}

/// Expanded query term
#[derive(Debug, Clone)]
pub struct ExpandedTerm {
    pub term: String,
    pub expansion_type: ExpansionType,
    pub confidence: f32,
    pub source: String,
}

/// Types of query expansion
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpansionType {
    Synonym,
    Hypernym,   // More general term
    Hyponym,    // More specific term
    Related,
    Acronym,
    Spelling,
}

/// Result type for retrieval operations
pub type RetrievalResult2<T> = std::result::Result<T, RetrievalError>;

/// Error types for retrieval operations
#[derive(Debug, Clone)]
pub enum RetrievalError {
    QueryProcessingError(String),
    StorageAccessError(String),
    ReasoningError(String),
    ContextAggregationError(String),
    CacheError(String),
    ModelError(String),
    InvalidQuery(String),
    TimeoutError(String),
}

impl std::fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrievalError::QueryProcessingError(msg) => 
                write!(f, "Query processing error: {}", msg),
            RetrievalError::StorageAccessError(msg) => 
                write!(f, "Storage access error: {}", msg),
            RetrievalError::ReasoningError(msg) => 
                write!(f, "Reasoning error: {}", msg),
            RetrievalError::ContextAggregationError(msg) => 
                write!(f, "Context aggregation error: {}", msg),
            RetrievalError::CacheError(msg) => 
                write!(f, "Cache error: {}", msg),
            RetrievalError::ModelError(msg) => 
                write!(f, "Model error: {}", msg),
            RetrievalError::InvalidQuery(msg) => 
                write!(f, "Invalid query: {}", msg),
            RetrievalError::TimeoutError(msg) => 
                write!(f, "Timeout error: {}", msg),
        }
    }
}

impl std::error::Error for RetrievalError {}

/// Statistics about retrieval performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalStats {
    pub total_queries: usize,
    pub average_response_time_ms: f64,
    pub cache_hit_rate: f32,
    pub multi_hop_usage_rate: f32,
    pub average_results_per_query: f32,
    pub query_intent_distribution: HashMap<QueryIntent, usize>,
    pub reasoning_type_distribution: HashMap<ReasoningType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_retrieval_query_defaults() {
        let query = RetrievalQuery::default();
        
        assert_eq!(query.retrieval_mode, RetrievalMode::Hybrid);
        assert!(query.enable_multi_hop);
        assert_eq!(query.max_reasoning_hops, 3);
        assert_eq!(query.max_results, 10);
    }
    
    #[test]
    fn test_query_intent_types() {
        let intents = vec![
            QueryIntent::FactualLookup,
            QueryIntent::ConceptExploration,
            QueryIntent::RelationshipQuery,
            QueryIntent::CausalAnalysis,
        ];
        
        for intent in intents {
            match intent {
                QueryIntent::FactualLookup => assert!(true),
                QueryIntent::ConceptExploration => assert!(true),
                QueryIntent::RelationshipQuery => assert!(true),
                QueryIntent::CausalAnalysis => assert!(true),
                _ => assert!(true),
            }
        }
    }
    
    #[test]
    fn test_retrieval_config_defaults() {
        let config = RetrievalConfig::default();
        
        assert_eq!(config.embedding_model_id, "minilm_l6_v2");
        assert_eq!(config.reasoning_model_id, "smollm2_360m");
        assert!(config.cache_search_results);
        assert!(config.enable_fuzzy_matching);
    }
}