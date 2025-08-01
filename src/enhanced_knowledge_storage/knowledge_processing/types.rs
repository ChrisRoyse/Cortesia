//! Knowledge Processing Types
//! 
//! Core types and structures for AI-powered knowledge processing.

// use std::time::Instant; // Removed unused import
use serde::{Serialize, Deserialize};
use crate::enhanced_knowledge_storage::types::ComplexityLevel;

/// Contextual entity with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualEntity {
    pub name: String,
    pub entity_type: EntityType,
    pub context: String,
    pub confidence: f32,
    pub span: Option<TextSpan>,
    pub attributes: std::collections::HashMap<String, String>,
    pub extracted_at: u64, // timestamp
}

/// Types of entities that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Concept,
    Event, 
    Technology,
    Method,
    Measurement,
    TimeExpression,
    Other(String),
}

impl EntityType {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "person" | "people" => EntityType::Person,
            "organization" | "company" | "institution" => EntityType::Organization,
            "location" | "place" | "country" | "city" => EntityType::Location,
            "concept" | "idea" | "theory" => EntityType::Concept,
            "event" | "occurrence" | "happening" => EntityType::Event,
            "technology" | "tech" | "software" | "hardware" => EntityType::Technology,
            "method" | "approach" | "technique" => EntityType::Method,
            "measurement" | "metric" | "quantity" => EntityType::Measurement,
            "time" | "date" | "period" => EntityType::TimeExpression,
            _ => EntityType::Other(s.to_string()),
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            EntityType::Person => "Person".to_string(),
            EntityType::Organization => "Organization".to_string(),
            EntityType::Location => "Location".to_string(),
            EntityType::Concept => "Concept".to_string(),
            EntityType::Event => "Event".to_string(),
            EntityType::Technology => "Technology".to_string(),
            EntityType::Method => "Method".to_string(),
            EntityType::Measurement => "Measurement".to_string(),
            EntityType::TimeExpression => "TimeExpression".to_string(),
            EntityType::Other(s) => s.clone(),
        }
    }
}

/// Text span indicating position in original text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

/// Complex relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexRelationship {
    pub source: String,
    pub predicate: RelationshipType,
    pub target: String,
    pub context: String,
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
    pub relationship_strength: f32,
    pub temporal_info: Option<TemporalInfo>,
}

/// Types of relationships between entities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RelationshipType {
    // Direct relationships
    CreatedBy,
    WorksFor,
    LocatedIn,
    PartOf,
    InstanceOf,
    
    // Causal relationships
    Causes,
    ResultsIn,
    EnabledBy,
    PreventedBy,
    
    // Temporal relationships
    Before,
    After,
    During,
    
    // Hierarchical relationships
    ParentOf,
    ChildOf,
    SuperiorTo,
    SubordinateTo,
    
    // Semantic relationships
    SimilarTo,
    OppositeOf,
    RelatedTo,
    InfluencedBy,
    
    // Custom relationship
    Custom(String),
}

impl RelationshipType {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "created" | "created_by" | "authored" => RelationshipType::CreatedBy,
            "works_for" | "employed_by" | "member_of" => RelationshipType::WorksFor,
            "located_in" | "based_in" | "situated_in" => RelationshipType::LocatedIn,
            "part_of" | "component_of" | "belongs_to" => RelationshipType::PartOf,
            "instance_of" | "type_of" | "kind_of" => RelationshipType::InstanceOf,
            "causes" | "leads_to" | "triggers" => RelationshipType::Causes,
            "results_in" | "produces" | "generates" => RelationshipType::ResultsIn,
            "enabled_by" | "facilitated_by" | "supported_by" => RelationshipType::EnabledBy,
            "prevented_by" | "blocked_by" | "hindered_by" => RelationshipType::PreventedBy,
            "before" | "precedes" | "earlier_than" => RelationshipType::Before,
            "after" | "follows" | "later_than" => RelationshipType::After,
            "during" | "concurrent_with" | "simultaneous" => RelationshipType::During,
            "parent_of" | "contains" | "includes" => RelationshipType::ParentOf,
            "child_of" | "contained_in" | "included_in" => RelationshipType::ChildOf,
            "superior_to" | "greater_than" | "dominates" => RelationshipType::SuperiorTo,
            "subordinate_to" | "less_than" | "dominated_by" => RelationshipType::SubordinateTo,
            "similar_to" | "like" | "resembles" => RelationshipType::SimilarTo,
            "opposite_of" | "contrary_to" | "inverse_of" => RelationshipType::OppositeOf,
            "related_to" | "associated_with" | "connected_to" => RelationshipType::RelatedTo,
            "influenced_by" | "affected_by" | "shaped_by" => RelationshipType::InfluencedBy,
            _ => RelationshipType::Custom(s.to_string()),
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            RelationshipType::CreatedBy => "created_by".to_string(),
            RelationshipType::WorksFor => "works_for".to_string(),
            RelationshipType::LocatedIn => "located_in".to_string(),
            RelationshipType::PartOf => "part_of".to_string(),
            RelationshipType::InstanceOf => "instance_of".to_string(),
            RelationshipType::Causes => "causes".to_string(),
            RelationshipType::ResultsIn => "results_in".to_string(),
            RelationshipType::EnabledBy => "enabled_by".to_string(),
            RelationshipType::PreventedBy => "prevented_by".to_string(),
            RelationshipType::Before => "before".to_string(),
            RelationshipType::After => "after".to_string(),
            RelationshipType::During => "during".to_string(),
            RelationshipType::ParentOf => "parent_of".to_string(),
            RelationshipType::ChildOf => "child_of".to_string(),
            RelationshipType::SuperiorTo => "superior_to".to_string(),
            RelationshipType::SubordinateTo => "subordinate_to".to_string(),
            RelationshipType::SimilarTo => "similar_to".to_string(),
            RelationshipType::OppositeOf => "opposite_of".to_string(),
            RelationshipType::RelatedTo => "related_to".to_string(),
            RelationshipType::InfluencedBy => "influenced_by".to_string(),
            RelationshipType::Custom(s) => s.clone(),
        }
    }
}

/// Temporal information for relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub duration: Option<String>,
    pub frequency: Option<String>,
}

/// Semantic chunk with intelligent boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    pub id: String,
    pub content: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub semantic_coherence: f32,
    pub key_concepts: Vec<String>,
    pub entities: Vec<ContextualEntity>,
    pub relationships: Vec<ComplexRelationship>,
    pub chunk_type: ChunkType,
    pub overlap_with_previous: Option<String>,
    pub overlap_with_next: Option<String>,
}

/// Types of semantic chunks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkType {
    Paragraph,
    Section,
    Topic,
    Dialogue,
    List,
    Code,
    Table,
    Other,
}

/// Semantic boundary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticBoundary {
    pub position: usize,
    pub boundary_type: BoundaryType,
    pub confidence: f32,
    pub reason: String,
}

/// Types of semantic boundaries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BoundaryType {
    TopicShift,      // Change in topic or theme
    SentenceEnd,     // Natural sentence ending
    ParagraphBreak,  // Paragraph boundary
    SectionBreak,    // Section or chapter boundary
    EntityBoundary,  // Entity relationship boundary
    ConceptBoundary, // Conceptual idea boundary
}

/// Document structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructure {
    pub sections: Vec<DocumentSection>,
    pub overall_topic: Option<String>,
    pub key_themes: Vec<String>,
    pub complexity_level: ComplexityLevel,
    pub estimated_reading_time: std::time::Duration,
}

/// Document section information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    pub title: Option<String>,
    pub start_pos: usize,
    pub end_pos: usize,
    pub section_type: SectionType,
    pub key_points: Vec<String>,
}

/// Types of document sections
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionType {
    Title,
    Abstract,
    Introduction,
    Body,
    Conclusion,
    References,
    Appendix,
    Other,
}

/// Configuration for knowledge processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProcessingConfig {
    /// Model to use for entity extraction
    pub entity_extraction_model: String,
    /// Model to use for relationship extraction
    pub relationship_extraction_model: String,
    /// Model to use for semantic analysis
    pub semantic_analysis_model: String,
    /// Minimum confidence threshold for entities
    pub min_entity_confidence: f32,
    /// Minimum confidence threshold for relationships
    pub min_relationship_confidence: f32,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// Minimum chunk size in characters
    pub min_chunk_size: usize,
    /// Overlap size between chunks
    pub chunk_overlap_size: usize,
    /// Enable context preservation
    pub preserve_context: bool,
}

impl Default for KnowledgeProcessingConfig {
    fn default() -> Self {
        Self {
            entity_extraction_model: "smollm2_360m".to_string(),
            relationship_extraction_model: "smollm2_360m".to_string(),
            semantic_analysis_model: "smollm2_360m".to_string(),
            min_entity_confidence: 0.7,
            min_relationship_confidence: 0.6,
            max_chunk_size: 2048,  // 2KB default
            min_chunk_size: 128,   // 128 chars minimum
            chunk_overlap_size: 256, // 256 chars overlap
            preserve_context: true,
        }
    }
}

/// Processing result with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProcessingResult {
    pub document_id: String,
    pub chunks: Vec<SemanticChunk>,
    pub global_entities: Vec<ContextualEntity>,
    pub global_relationships: Vec<ComplexRelationship>,
    pub document_structure: DocumentStructure,
    pub processing_metadata: ProcessingMetadata,
    pub quality_metrics: QualityMetrics,
}

/// Metadata about the processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time: std::time::Duration,
    pub models_used: Vec<String>,
    pub total_tokens_processed: usize,
    pub chunks_created: usize,
    pub entities_extracted: usize,
    pub relationships_extracted: usize,
    pub memory_usage_peak: u64,
}

/// Quality metrics for processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub entity_extraction_quality: f32,
    pub relationship_extraction_quality: f32,
    pub semantic_coherence: f32,
    pub context_preservation: f32,
    pub overall_quality: f32,
}

impl QualityMetrics {
    pub fn calculate_overall_quality(&mut self) {
        self.overall_quality = (
            self.entity_extraction_quality * 0.3 +
            self.relationship_extraction_quality * 0.3 +
            self.semantic_coherence * 0.2 +
            self.context_preservation * 0.2
        ).clamp(0.0, 1.0);
    }
}

/// Error types specific to knowledge processing
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeProcessingError {
    #[error("Entity extraction failed: {0}")]
    EntityExtractionFailed(String),
    
    #[error("Relationship extraction failed: {0}")]
    RelationshipExtractionFailed(String),
    
    #[error("Semantic analysis failed: {0}")]
    SemanticAnalysisFailed(String),
    
    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type KnowledgeProcessingResult2<T> = std::result::Result<T, KnowledgeProcessingError>;