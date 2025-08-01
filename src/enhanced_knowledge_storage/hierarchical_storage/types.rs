//! Hierarchical Storage Types
//! 
//! Core types for the hierarchical knowledge storage system that organizes
//! information in layers: Document → Section → Paragraph → Sentence → Entity.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::enhanced_knowledge_storage::{
    types::*,
    knowledge_processing::types::*,
};

/// Hierarchical knowledge structure containing multiple layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalKnowledge {
    pub document_id: String,
    pub global_context: GlobalContext,
    pub knowledge_layers: Vec<KnowledgeLayer>,
    pub semantic_links: SemanticLinkGraph,
    pub retrieval_index: HierarchicalIndex,
    pub created_at: u64,
    pub last_updated: u64,
}

/// A single layer in the knowledge hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeLayer {
    pub layer_id: String,
    pub layer_type: LayerType,
    pub parent_layer_id: Option<String>,
    pub child_layer_ids: Vec<String>,
    pub content: LayerContent,
    pub entities: Vec<ContextualEntity>,
    pub relationships: Vec<ComplexRelationship>,
    pub semantic_embedding: Option<Vec<f32>>,
    pub importance_score: f32,
    pub coherence_score: f32,
    pub position: LayerPosition,
}

/// Types of layers in the hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    Document,      // Top-level document
    Section,       // Major sections (chapters, parts)
    Paragraph,     // Paragraph-level content
    Sentence,      // Individual sentences
    Entity,        // Entity-focused content
    Relationship,  // Relationship-focused content
    Concept,       // Conceptual groupings
}

/// Content within a knowledge layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerContent {
    pub raw_text: String,
    pub processed_text: String,
    pub key_phrases: Vec<String>,
    pub summary: Option<String>,
    pub metadata: LayerMetadata,
}

/// Position information for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPosition {
    pub start_offset: usize,
    pub end_offset: usize,
    pub depth_level: u32,
    pub sequence_number: u32,
}

/// Metadata for a knowledge layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub word_count: usize,
    pub character_count: usize,
    pub complexity_level: ComplexityLevel,
    pub reading_time: Duration,
    pub tags: Vec<String>,
    pub custom_attributes: HashMap<String, String>,
}

/// Graph structure for semantic links between layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticLinkGraph {
    pub nodes: HashMap<String, SemanticNode>,
    pub edges: Vec<SemanticEdge>,
    pub link_types: HashMap<String, LinkTypeInfo>,
}

impl SemanticLinkGraph {
    /// Get the total number of semantic links (edges) in the graph
    pub fn len(&self) -> usize {
        self.edges.len()
    }
    
    /// Check if the semantic link graph is empty
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.nodes.is_empty()
    }
}

/// A node in the semantic link graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    pub node_id: String,
    pub layer_id: String,
    pub node_type: SemanticNodeType,
    pub importance_weight: f32,
    pub centrality_scores: CentralityScores,
    pub connected_nodes: Vec<String>,
}

/// Types of semantic nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticNodeType {
    LayerNode,        // Represents a knowledge layer
    EntityNode,       // Represents an entity
    ConceptNode,      // Represents a concept
    RelationshipNode, // Represents a relationship
    TopicNode,        // Represents a topic cluster
}

/// Centrality scores for graph analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityScores {
    pub degree_centrality: f32,
    pub betweenness_centrality: f32,
    pub closeness_centrality: f32,
    pub eigenvector_centrality: f32,
}

/// An edge in the semantic link graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEdge {
    pub edge_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub link_type: SemanticLinkType,
    pub weight: f32,
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
    pub created_at: u64,
}

/// Types of semantic links
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticLinkType {
    Hierarchical,      // Parent-child relationships
    Sequential,        // Ordered sequence relationships
    Referential,       // Cross-references and mentions
    Semantic,          // Semantic similarity
    Causal,           // Cause-effect relationships
    Temporal,         // Time-based relationships
    Categorical,      // Category membership
    Comparative,      // Comparison relationships
    Definitional,     // Definition relationships
    Explanatory,      // Explanation relationships
}

/// Information about link types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkTypeInfo {
    pub link_type: SemanticLinkType,
    pub description: String,
    pub is_directional: bool,
    pub typical_strength_range: (f32, f32),
    pub color_code: String, // For visualization
}

/// Hierarchical index for efficient retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalIndex {
    pub layer_index: HashMap<String, LayerIndexEntry>,
    pub entity_index: HashMap<String, Vec<String>>, // entity_name -> layer_ids
    pub concept_index: HashMap<String, Vec<String>>, // concept -> layer_ids
    pub relationship_index: HashMap<String, Vec<String>>, // relationship_type -> layer_ids
    pub full_text_index: HashMap<String, Vec<IndexMatch>>, // term -> matches
    pub semantic_index: Vec<SemanticCluster>,
}

/// Entry in the layer index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerIndexEntry {
    pub layer_id: String,
    pub layer_type: LayerType,
    pub parent_id: Option<String>,
    pub keywords: Vec<String>,
    pub entity_count: usize,
    pub relationship_count: usize,
    pub importance_score: f32,
    pub last_accessed: u64,
    pub access_count: u64,
}

/// Match result from full-text index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMatch {
    pub layer_id: String,
    pub positions: Vec<usize>,
    pub relevance_score: f32,
    pub context_snippet: String,
}

/// Semantic cluster for grouping related content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    pub cluster_id: String,
    pub center_embedding: Vec<f32>,
    pub member_layer_ids: Vec<String>,
    pub cluster_label: String,
    pub coherence_score: f32,
    pub representative_content: String,
}

/// Global context extracted from document analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalContext {
    pub document_theme: String,
    pub key_entities: Vec<String>,
    pub main_relationships: Vec<String>,
    pub conceptual_framework: Vec<String>,
    pub context_preservation_score: f32,
    pub domain_classification: Vec<String>,
    pub complexity_indicators: ComplexityIndicators,
}

/// Indicators of document complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityIndicators {
    pub vocabulary_complexity: f32,
    pub syntactic_complexity: f32,
    pub conceptual_density: f32,
    pub relationship_complexity: f32,
    pub overall_complexity: ComplexityLevel,
}

/// Configuration for hierarchical storage
#[derive(Debug, Clone)]
pub struct HierarchicalStorageConfig {
    pub max_layers_per_document: usize,
    pub max_nodes_per_layer: usize,
    pub semantic_similarity_threshold: f32,
    pub importance_score_threshold: f32,
    pub enable_semantic_clustering: bool,
    pub enable_cross_document_links: bool,
    pub index_update_frequency: Duration,
    pub compression_threshold: f32,
    pub cache_size_limit: usize,
}

impl Default for HierarchicalStorageConfig {
    fn default() -> Self {
        Self {
            max_layers_per_document: 1000,
            max_nodes_per_layer: 50,
            semantic_similarity_threshold: 0.7,
            importance_score_threshold: 0.3,
            enable_semantic_clustering: true,
            enable_cross_document_links: true,
            index_update_frequency: Duration::from_secs(300), // 5 minutes
            compression_threshold: 0.8,
            cache_size_limit: 10000, // Number of layers to cache
        }
    }
}

/// Result type for hierarchical storage operations
pub type HierarchicalStorageResult<T> = std::result::Result<T, HierarchicalStorageError>;

/// Error types for hierarchical storage operations
#[derive(Debug, Clone)]
pub enum HierarchicalStorageError {
    LayerNotFound(String),
    InvalidLayerStructure(String),
    IndexingError(String),
    SemanticAnalysisError(String),
    StorageError(String),
    RetrievalError(String),
    GraphError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for HierarchicalStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HierarchicalStorageError::LayerNotFound(id) => 
                write!(f, "Knowledge layer not found: {}", id),
            HierarchicalStorageError::InvalidLayerStructure(msg) => 
                write!(f, "Invalid layer structure: {}", msg),
            HierarchicalStorageError::IndexingError(msg) => 
                write!(f, "Indexing error: {}", msg),
            HierarchicalStorageError::SemanticAnalysisError(msg) => 
                write!(f, "Semantic analysis error: {}", msg),
            HierarchicalStorageError::StorageError(msg) => 
                write!(f, "Storage error: {}", msg),
            HierarchicalStorageError::RetrievalError(msg) => 
                write!(f, "Retrieval error: {}", msg),
            HierarchicalStorageError::GraphError(msg) => 
                write!(f, "Graph operation error: {}", msg),
            HierarchicalStorageError::ConfigurationError(msg) => 
                write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for HierarchicalStorageError {}

/// Statistics about hierarchical storage system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalStorageStats {
    pub total_documents: usize,
    pub total_layers: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub average_depth: f32,
    pub layer_type_distribution: HashMap<LayerType, usize>,
    pub link_type_distribution: HashMap<SemanticLinkType, usize>,
    pub storage_efficiency: f32,
    pub index_coverage: f32,
    pub cache_hit_rate: f32,
}

// Utility functions for working with hierarchical types

impl LayerType {
    /// Get the typical depth level for this layer type
    pub fn typical_depth(&self) -> u32 {
        match self {
            LayerType::Document => 0,
            LayerType::Section => 1,
            LayerType::Paragraph => 2,
            LayerType::Sentence => 3,
            LayerType::Entity => 4,
            LayerType::Relationship => 4,
            LayerType::Concept => 3,
        }
    }
    
    /// Check if this layer type can contain the specified child type
    pub fn can_contain(&self, child_type: &LayerType) -> bool {
        match (self, child_type) {
            (LayerType::Document, LayerType::Section) => true,
            (LayerType::Section, LayerType::Paragraph) => true,
            (LayerType::Paragraph, LayerType::Sentence) => true,
            (LayerType::Sentence, LayerType::Entity) => true,
            (LayerType::Sentence, LayerType::Relationship) => true,
            (LayerType::Section, LayerType::Concept) => true,
            (LayerType::Paragraph, LayerType::Concept) => true,
            _ => false,
        }
    }
}

impl SemanticLinkType {
    /// Check if this link type is directional
    pub fn is_directional(&self) -> bool {
        matches!(
            self,
            SemanticLinkType::Hierarchical |
            SemanticLinkType::Sequential |
            SemanticLinkType::Causal |
            SemanticLinkType::Temporal |
            SemanticLinkType::Definitional |
            SemanticLinkType::Explanatory
        )
    }
    
    /// Get typical strength range for this link type
    pub fn strength_range(&self) -> (f32, f32) {
        match self {
            SemanticLinkType::Hierarchical => (0.8, 1.0),
            SemanticLinkType::Sequential => (0.7, 0.9),
            SemanticLinkType::Causal => (0.6, 0.9),
            SemanticLinkType::Temporal => (0.5, 0.8),
            SemanticLinkType::Semantic => (0.4, 0.8),
            SemanticLinkType::Referential => (0.3, 0.7),
            SemanticLinkType::Categorical => (0.5, 0.8),
            SemanticLinkType::Comparative => (0.4, 0.7),
            SemanticLinkType::Definitional => (0.7, 0.9),
            SemanticLinkType::Explanatory => (0.6, 0.8),
        }
    }
}

impl HierarchicalKnowledge {
    /// Create a new hierarchical knowledge structure
    pub fn new(document_id: String, global_context: GlobalContext) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            document_id,
            global_context,
            knowledge_layers: Vec::new(),
            semantic_links: SemanticLinkGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
                link_types: HashMap::new(),
            },
            retrieval_index: HierarchicalIndex {
                layer_index: HashMap::new(),
                entity_index: HashMap::new(),
                concept_index: HashMap::new(),
                relationship_index: HashMap::new(),
                full_text_index: HashMap::new(),
                semantic_index: Vec::new(),
            },
            created_at: timestamp,
            last_updated: timestamp,
        }
    }
    
    /// Get layers by type
    pub fn get_layers_by_type(&self, layer_type: &LayerType) -> Vec<&KnowledgeLayer> {
        self.knowledge_layers
            .iter()
            .filter(|layer| &layer.layer_type == layer_type)
            .collect()
    }
    
    /// Get layer by ID
    pub fn get_layer(&self, layer_id: &str) -> Option<&KnowledgeLayer> {
        self.knowledge_layers
            .iter()
            .find(|layer| layer.layer_id == layer_id)
    }
    
    /// Calculate total importance score
    pub fn total_importance(&self) -> f32 {
        self.knowledge_layers
            .iter()
            .map(|layer| layer.importance_score)
            .sum()
    }
    
    /// Get maximum depth in the hierarchy
    pub fn max_depth(&self) -> u32 {
        self.knowledge_layers
            .iter()
            .map(|layer| layer.position.depth_level)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_type_hierarchy() {
        assert!(LayerType::Document.can_contain(&LayerType::Section));
        assert!(LayerType::Section.can_contain(&LayerType::Paragraph));
        assert!(LayerType::Paragraph.can_contain(&LayerType::Sentence));
        assert!(!LayerType::Sentence.can_contain(&LayerType::Section));
    }
    
    #[test]
    fn test_layer_type_depths() {
        assert_eq!(LayerType::Document.typical_depth(), 0);
        assert_eq!(LayerType::Section.typical_depth(), 1);
        assert_eq!(LayerType::Paragraph.typical_depth(), 2);
        assert_eq!(LayerType::Sentence.typical_depth(), 3);
    }
    
    #[test]
    fn test_semantic_link_properties() {
        assert!(SemanticLinkType::Hierarchical.is_directional());
        assert!(SemanticLinkType::Causal.is_directional());
        assert!(!SemanticLinkType::Semantic.is_directional());
        
        let (min, max) = SemanticLinkType::Hierarchical.strength_range();
        assert!(min < max);
        assert!(min >= 0.0 && max <= 1.0);
    }
    
    #[test]
    fn test_hierarchical_knowledge_creation() {
        let global_context = GlobalContext {
            document_theme: "Test Document".to_string(),
            key_entities: vec!["Entity1".to_string()],
            main_relationships: vec!["Relationship1".to_string()],
            conceptual_framework: vec!["Concept1".to_string()],
            context_preservation_score: 0.8,
            domain_classification: vec!["Technology".to_string()],
            complexity_indicators: ComplexityIndicators {
                vocabulary_complexity: 0.5,
                syntactic_complexity: 0.6,
                conceptual_density: 0.7,
                relationship_complexity: 0.8,
                overall_complexity: ComplexityLevel::Medium,
            },
        };
        
        let knowledge = HierarchicalKnowledge::new("doc_1".to_string(), global_context);
        
        assert_eq!(knowledge.document_id, "doc_1");
        assert_eq!(knowledge.global_context.document_theme, "Test Document");
        assert_eq!(knowledge.knowledge_layers.len(), 0);
        assert_eq!(knowledge.max_depth(), 0);
    }
    
    #[test]
    fn test_storage_config_defaults() {
        let config = HierarchicalStorageConfig::default();
        
        assert_eq!(config.max_layers_per_document, 1000);
        assert_eq!(config.semantic_similarity_threshold, 0.7);
        assert!(config.enable_semantic_clustering);
        assert!(config.enable_cross_document_links);
    }
}