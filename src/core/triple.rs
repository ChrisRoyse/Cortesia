use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::error::{GraphError, Result};

/// Maximum chunk size based on research: 512 tokens (~400 words, ~2KB)
pub const MAX_CHUNK_SIZE_BYTES: usize = 2048;

/// Maximum predicate length (research shows 1-3 words optimal)
pub const MAX_PREDICATE_LENGTH: usize = 64;

/// Maximum entity name length for anti-bloat
pub const MAX_ENTITY_NAME_LENGTH: usize = 128;

/// Core triple structure: Subject-Predicate-Object
/// Optimized for LLM comprehension and minimal memory footprint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Triple {
    /// Subject: The entity being described (e.g., "Einstein", "Python", "AI")
    pub subject: String,
    
    /// Predicate: The relationship (max 3 words, e.g., "invented", "is_type", "works_at")
    pub predicate: String,
    
    /// Object: The target entity or literal value
    pub object: String,
    
    /// Confidence score for this triple (0.0-1.0)
    pub confidence: f32,
    
    /// Optional source/provenance information
    pub source: Option<String>,
}

// Manual Hash implementation to handle f32 confidence field
impl Hash for Triple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.subject.hash(state);
        self.predicate.hash(state);
        self.object.hash(state);
        // Convert f32 to bits for consistent hashing
        self.confidence.to_bits().hash(state);
        self.source.hash(state);
    }
}

/// Knowledge node that can store either a simple triple or a data chunk
/// Designed to be LLM-friendly while preventing data bloat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Unique identifier generated from content hash
    pub id: String,
    
    /// Node type for LLM understanding
    pub node_type: NodeType,
    
    /// The primary content (triple or chunk)
    pub content: NodeContent,
    
    /// Embedding vector for similarity search
    pub embedding: Vec<f32>,
    
    /// Metadata for optimization
    pub metadata: NodeMetadata,
}

/// Types of knowledge nodes for LLM clarity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    /// Simple fact: "Einstein invented relativity"
    Triple,
    
    /// Detailed information chunk (max 512 tokens)
    Chunk,
    
    /// Entity definition/description
    Entity,
    
    /// Relationship definition
    Relationship,
    
    /// Conceptual node for abstract ideas
    Concept,
}

/// Content variants optimized for different knowledge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeContent {
    /// Single SPO triple
    Triple(Triple),
    
    /// Text chunk with size limit
    Chunk {
        text: String,
        /// Extracted triples from this chunk
        extracted_triples: Vec<Triple>,
        /// Word count for LLM context planning
        word_count: usize,
    },
    
    /// Entity definition
    Entity {
        name: String,
        description: String,
        entity_type: String,
        properties: HashMap<String, String>,
    },
    
    /// Relationship schema
    Relationship {
        predicate: String,
        description: String,
        domain: String,  // Subject type
        range: String,   // Object type
    },
}

/// Metadata for memory optimization and LLM usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: u64,
    
    /// Last access time for cache optimization  
    pub last_accessed: u64,
    
    /// Usage frequency for importance scoring
    pub access_count: u32,
    
    /// Memory size in bytes
    pub size_bytes: usize,
    
    /// Quality score (0.0-1.0) for ranking
    pub quality_score: f32,
    
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Triple {
    /// Create a new triple with validation for anti-bloat
    pub fn new(subject: String, predicate: String, object: String) -> Result<Self> {
        // Validate lengths to prevent bloat
        if subject.len() > MAX_ENTITY_NAME_LENGTH {
            return Err(GraphError::SerializationError(
                format!("Subject too long: {} > {}", subject.len(), MAX_ENTITY_NAME_LENGTH)
            ));
        }
        
        if predicate.len() > MAX_PREDICATE_LENGTH {
            return Err(GraphError::SerializationError(
                format!("Predicate too long: {} > {}", predicate.len(), MAX_PREDICATE_LENGTH)
            ));
        }
        
        if object.len() > MAX_ENTITY_NAME_LENGTH {
            return Err(GraphError::SerializationError(
                format!("Object too long: {} > {}", object.len(), MAX_ENTITY_NAME_LENGTH)
            ));
        }
        
        // Validate predicate format (1-3 words, underscore separated)
        let word_count = predicate.split('_').count();
        if word_count > 3 {
            return Err(GraphError::SerializationError(
                format!("Predicate has too many words: {} > 3", word_count)
            ));
        }
        
        Ok(Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_lowercase(),
            object: object.trim().to_string(),
            confidence: 1.0,
            source: None,
        })
    }
    
    /// Create triple with confidence and source
    pub fn with_metadata(
        subject: String, 
        predicate: String, 
        object: String, 
        confidence: f32,
        source: Option<String>
    ) -> Result<Self> {
        let mut triple = Self::new(subject, predicate, object)?;
        triple.confidence = confidence.clamp(0.0, 1.0);
        triple.source = source;
        Ok(triple)
    }
    
    /// Generate a unique ID for this triple
    pub fn id(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        
        format!("triple_{:x}", hasher.finish())
    }
    
    /// Convert to natural language for LLM consumption
    pub fn to_natural_language(&self) -> String {
        match self.predicate.as_str() {
            "is" | "is_a" | "type" => format!("{} is {}", self.subject, self.object),
            "has" | "has_property" => format!("{} has {}", self.subject, self.object),
            "located_in" | "in" => format!("{} is located in {}", self.subject, self.object),
            "created_by" | "invented_by" => format!("{} was created by {}", self.subject, self.object),
            "works_at" | "employed_by" => format!("{} works at {}", self.subject, self.object),
            "connected_to" | "related_to" => format!("{} is connected to {}", self.subject, self.object),
            _ => format!("{} {} {}", self.subject, self.predicate.replace('_', " "), self.object),
        }
    }
    
    /// Estimate memory footprint for anti-bloat monitoring
    pub fn memory_footprint(&self) -> usize {
        self.subject.len() + 
        self.predicate.len() + 
        self.object.len() + 
        self.source.as_ref().map(|s| s.len()).unwrap_or(0) +
        std::mem::size_of::<f32>() // confidence
    }
}

impl KnowledgeNode {
    /// Create a new triple node
    pub fn new_triple(triple: Triple, embedding: Vec<f32>) -> Self {
        let size_bytes = triple.memory_footprint() + embedding.len() * 4;
        let id = triple.id();
        
        Self {
            id,
            node_type: NodeType::Triple,
            content: NodeContent::Triple(triple),
            embedding,
            metadata: NodeMetadata {
                created_at: current_timestamp(),
                last_accessed: current_timestamp(),
                access_count: 0,
                size_bytes,
                quality_score: 1.0,
                tags: Vec::new(),
            },
        }
    }
    
    /// Create a new chunk node with automatic triple extraction
    pub fn new_chunk(
        text: String, 
        embedding: Vec<f32>,
        extracted_triples: Vec<Triple>
    ) -> Result<Self> {
        // Validate chunk size to prevent bloat
        if text.len() > MAX_CHUNK_SIZE_BYTES {
            return Err(GraphError::SerializationError(
                format!("Chunk too large: {} > {}", text.len(), MAX_CHUNK_SIZE_BYTES)
            ));
        }
        
        let word_count = text.split_whitespace().count();
        let size_bytes = text.len() + 
                        extracted_triples.iter().map(|t| t.memory_footprint()).sum::<usize>() +
                        embedding.len() * 4;
        
        // Generate ID from content hash
        let id = Self::generate_content_id(&text);
        
        Ok(Self {
            id,
            node_type: NodeType::Chunk,
            content: NodeContent::Chunk {
                text,
                extracted_triples,
                word_count,
            },
            embedding,
            metadata: NodeMetadata {
                created_at: current_timestamp(),
                last_accessed: current_timestamp(),
                access_count: 0,
                size_bytes,
                quality_score: 1.0,
                tags: Vec::new(),
            },
        })
    }
    
    /// Create an entity node
    pub fn new_entity(
        name: String,
        description: String,
        entity_type: String,
        properties: HashMap<String, String>,
        embedding: Vec<f32>
    ) -> Result<Self> {
        // Validate entity data size
        let total_size = name.len() + description.len() + entity_type.len() +
                        properties.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>();
        
        if total_size > MAX_CHUNK_SIZE_BYTES {
            return Err(GraphError::SerializationError(
                format!("Entity data too large: {} > {}", total_size, MAX_CHUNK_SIZE_BYTES)
            ));
        }
        
        let id = Self::generate_content_id(&name);
        let size_bytes = total_size + embedding.len() * 4;
        
        Ok(Self {
            id,
            node_type: NodeType::Entity,
            content: NodeContent::Entity {
                name,
                description,
                entity_type,
                properties,
            },
            embedding,
            metadata: NodeMetadata {
                created_at: current_timestamp(),
                last_accessed: current_timestamp(),
                access_count: 0,
                size_bytes,
                quality_score: 1.0,
                tags: Vec::new(),
            },
        })
    }
    
    /// Generate content-based ID
    fn generate_content_id(content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("node_{:x}", hasher.finish())
    }
    
    /// Update access metadata for cache optimization
    pub fn mark_accessed(&mut self) {
        self.metadata.last_accessed = current_timestamp();
        self.metadata.access_count += 1;
    }
    
    /// Get all triples from this node (direct or extracted)
    pub fn get_triples(&self) -> Vec<&Triple> {
        match &self.content {
            NodeContent::Triple(triple) => vec![triple],
            NodeContent::Chunk { extracted_triples, .. } => extracted_triples.iter().collect(),
            _ => Vec::new(),
        }
    }
    
    /// Convert to LLM-friendly representation
    pub fn to_llm_format(&self) -> String {
        match &self.content {
            NodeContent::Triple(triple) => {
                format!("FACT: {}", triple.to_natural_language())
            },
            NodeContent::Chunk { text, word_count, .. } => {
                format!("KNOWLEDGE ({} words): {}", word_count, text)
            },
            NodeContent::Entity { name, description, entity_type, .. } => {
                format!("ENTITY: {} ({})\nDescription: {}", name, entity_type, description)
            },
            NodeContent::Relationship { predicate, description, domain, range } => {
                format!("RELATIONSHIP: {} (from {} to {})\nDescription: {}", 
                       predicate, domain, range, description)
            },
        }
    }
    
    /// Calculate quality score based on usage and content
    pub fn calculate_quality_score(&mut self) {
        let recency_factor = 1.0 - ((current_timestamp() - self.metadata.created_at) as f32 / (86400.0 * 365.0)); // Decay over year
        let usage_factor = ((self.metadata.access_count as f32).ln() as f32).max(0.0) / 10.0; // Log scale
        let size_efficiency = 1.0 - (self.metadata.size_bytes as f32 / MAX_CHUNK_SIZE_BYTES as f32).min(1.0f32);
        
        self.metadata.quality_score = (recency_factor * 0.3 + usage_factor * 0.5 + size_efficiency * 0.2).clamp(0.0, 1.0);
    }
}

/// Optimized predicate vocabulary for common relationships
/// Helps LLMs use consistent, short predicates
pub struct PredicateVocabulary {
    predicates: HashMap<String, PredicateInfo>,
}

#[derive(Debug, Clone)]
pub struct PredicateInfo {
    pub canonical_form: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub typical_domain: String,
    pub typical_range: String,
}

impl PredicateVocabulary {
    pub fn new() -> Self {
        let mut predicates = HashMap::new();
        
        // Core predicates optimized for LLM understanding
        let core_predicates = vec![
            ("is", "Entity classification", "Entity", "Type"),
            ("has", "Property ownership", "Entity", "Property"),
            ("located_in", "Spatial relationship", "Entity", "Location"),
            ("part_of", "Composition relationship", "Entity", "Entity"),
            ("created_by", "Creation relationship", "Work", "Creator"),
            ("works_at", "Employment relationship", "Person", "Organization"),
            ("connected_to", "General connection", "Entity", "Entity"),
            ("similar_to", "Similarity relationship", "Entity", "Entity"),
            ("causes", "Causal relationship", "Event", "Event"),
            ("follows", "Temporal sequence", "Event", "Event"),
        ];
        
        for (pred, desc, domain, range) in core_predicates {
            predicates.insert(pred.to_string(), PredicateInfo {
                canonical_form: pred.to_string(),
                aliases: Vec::new(),
                description: desc.to_string(),
                typical_domain: domain.to_string(),
                typical_range: range.to_string(),
            });
        }
        
        Self { predicates }
    }
    
    /// Normalize predicate to canonical form
    pub fn normalize(&self, predicate: &str) -> String {
        let clean = predicate.trim().to_lowercase().replace(' ', "_");
        
        // Check if it's already canonical
        if self.predicates.contains_key(&clean) {
            return clean;
        }
        
        // Check aliases
        for (canonical, info) in &self.predicates {
            if info.aliases.contains(&clean) {
                return canonical.clone();
            }
        }
        
        // Return cleaned version if not found
        clean
    }
    
    /// Get suggestions for LLM predicate selection
    pub fn suggest_predicates(&self, context: &str) -> Vec<String> {
        // Simple keyword-based suggestions
        let context_lower = context.to_lowercase();
        let mut suggestions = Vec::new();
        
        for (predicate, info) in &self.predicates {
            if context_lower.contains(&info.description.to_lowercase()) ||
               context_lower.contains(predicate) {
                suggestions.push(predicate.clone());
            }
        }
        
        if suggestions.is_empty() {
            suggestions = vec!["is".to_string(), "has".to_string(), "connected_to".to_string()];
        }
        
        suggestions
    }
}

impl Default for PredicateVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Test constants and helpers
    const TEST_EMBEDDING: [f32; 3] = [0.1, 0.2, 0.3];

    fn create_test_embedding() -> Vec<f32> {
        TEST_EMBEDDING.to_vec()
    }

    fn create_large_string(size: usize) -> String {
        "a".repeat(size)
    }

    #[test]
    fn test_triple_new_valid_creation() {
        let result = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        );
        
        assert!(result.is_ok());
        let triple = result.unwrap();
        assert_eq!(triple.subject, "Einstein");
        assert_eq!(triple.predicate, "invented");
        assert_eq!(triple.object, "relativity");
        assert_eq!(triple.confidence, 1.0);
        assert!(triple.source.is_none());
    }

    #[test]
    fn test_triple_new_trims_whitespace() {
        let result = Triple::new(
            "  Einstein  ".to_string(),
            "  invented  ".to_string(),
            "  relativity  ".to_string(),
        );
        
        assert!(result.is_ok());
        let triple = result.unwrap();
        assert_eq!(triple.subject, "Einstein");
        assert_eq!(triple.predicate, "invented");
        assert_eq!(triple.object, "relativity");
    }

    #[test]
    fn test_triple_new_normalizes_predicate_case() {
        let result = Triple::new(
            "Einstein".to_string(),
            "INVENTED".to_string(),
            "relativity".to_string(),
        );
        
        assert!(result.is_ok());
        let triple = result.unwrap();
        assert_eq!(triple.predicate, "invented");
    }

    #[test]
    fn test_triple_new_subject_too_long() {
        let long_subject = create_large_string(MAX_ENTITY_NAME_LENGTH + 1);
        let result = Triple::new(
            long_subject,
            "invented".to_string(),
            "relativity".to_string(),
        );
        
        assert!(result.is_err());
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("Subject too long"));
        } else {
            panic!("Expected SerializationError for long subject");
        }
    }

    #[test]
    fn test_triple_new_predicate_too_long() {
        let long_predicate = create_large_string(MAX_PREDICATE_LENGTH + 1);
        let result = Triple::new(
            "Einstein".to_string(),
            long_predicate,
            "relativity".to_string(),
        );
        
        assert!(result.is_err());
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("Predicate too long"));
        } else {
            panic!("Expected SerializationError for long predicate");
        }
    }

    #[test]
    fn test_triple_new_object_too_long() {
        let long_object = create_large_string(MAX_ENTITY_NAME_LENGTH + 1);
        let result = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            long_object,
        );
        
        assert!(result.is_err());
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("Object too long"));
        } else {
            panic!("Expected SerializationError for long object");
        }
    }

    #[test]
    fn test_triple_new_predicate_too_many_words() {
        let result = Triple::new(
            "Einstein".to_string(),
            "very_long_complex_predicate_name".to_string(),
            "relativity".to_string(),
        );
        
        assert!(result.is_err());
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("too many words"));
        } else {
            panic!("Expected SerializationError for predicate with too many words");
        }
    }

    #[test]
    fn test_triple_new_predicate_max_words_allowed() {
        let result = Triple::new(
            "Einstein".to_string(),
            "is_known_for".to_string(), // 3 words - should be allowed
            "relativity".to_string(),
        );
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_triple_new_at_size_limits() {
        let max_subject = create_large_string(MAX_ENTITY_NAME_LENGTH);
        let max_predicate = create_large_string(MAX_PREDICATE_LENGTH);
        let max_object = create_large_string(MAX_ENTITY_NAME_LENGTH);
        
        let result = Triple::new(max_subject, max_predicate, max_object);
        assert!(result.is_ok());
    }

    #[test]
    fn test_triple_with_metadata() {
        let result = Triple::with_metadata(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
            0.8,
            Some("Wikipedia".to_string()),
        );
        
        assert!(result.is_ok());
        let triple = result.unwrap();
        assert_eq!(triple.confidence, 0.8);
        assert_eq!(triple.source, Some("Wikipedia".to_string()));
    }

    #[test]
    fn test_triple_with_metadata_clamps_confidence() {
        // Test confidence > 1.0
        let result = Triple::with_metadata(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
            1.5,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().confidence, 1.0);

        // Test confidence < 0.0
        let result = Triple::with_metadata(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
            -0.5,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().confidence, 0.0);
    }

    #[test]
    fn test_triple_id_generation() {
        let triple1 = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        ).unwrap();
        
        let triple2 = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        ).unwrap();
        
        let triple3 = Triple::new(
            "Newton".to_string(),
            "invented".to_string(),
            "calculus".to_string(),
        ).unwrap();
        
        // Same triples should have same ID
        assert_eq!(triple1.id(), triple2.id());
        
        // Different triples should have different IDs
        assert_ne!(triple1.id(), triple3.id());
        
        // ID should have correct format
        assert!(triple1.id().starts_with("triple_"));
    }

    #[test]
    fn test_triple_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let triple1 = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        ).unwrap();
        
        let triple2 = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        ).unwrap();
        
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        
        triple1.hash(&mut hasher1);
        triple2.hash(&mut hasher2);
        
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_triple_memory_footprint() {
        let triple = Triple::new(
            "Einstein".to_string(),
            "invented".to_string(),
            "relativity".to_string(),
        ).unwrap();
        
        let expected_size = "Einstein".len() + "invented".len() + "relativity".len() + std::mem::size_of::<f32>();
        assert_eq!(triple.memory_footprint(), expected_size);
        
        // Test with source
        let mut triple_with_source = triple.clone();
        triple_with_source.source = Some("Wikipedia".to_string());
        
        let expected_with_source = expected_size + "Wikipedia".len();
        assert_eq!(triple_with_source.memory_footprint(), expected_with_source);
    }

    #[test]
    fn test_triple_to_natural_language() {
        let test_cases = vec![
            ("Einstein", "is", "scientist", "Einstein is scientist"),
            ("Python", "has", "syntax", "Python has syntax"),
            ("Paris", "located_in", "France", "Paris is located in France"),
            ("Theory", "created_by", "Einstein", "Theory was created by Einstein"),
            ("Employee", "works_at", "Company", "Employee works at Company"),
            ("A", "connected_to", "B", "A is connected to B"),
            ("Custom", "custom_pred", "Value", "Custom custom pred Value"),
        ];
        
        for (subject, predicate, object, expected) in test_cases {
            let triple = Triple::new(
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            ).unwrap();
            
            assert_eq!(triple.to_natural_language(), expected);
        }
    }

    #[test]
    fn test_knowledge_node_new_chunk_valid() {
        let text = "This is a test chunk with valid size.".to_string();
        let embedding = create_test_embedding();
        let extracted_triples = vec![
            Triple::new("test".to_string(), "is".to_string(), "chunk".to_string()).unwrap()
        ];
        
        let result = KnowledgeNode::new_chunk(text.clone(), embedding, extracted_triples);
        assert!(result.is_ok());
        
        let node = result.unwrap();
        assert_eq!(node.node_type, NodeType::Chunk);
        assert!(node.id.starts_with("node_"));
        
        if let NodeContent::Chunk { text: chunk_text, word_count, .. } = &node.content {
            assert_eq!(chunk_text, &text);
            assert_eq!(*word_count, 8); // "This is a test chunk with valid size."
        } else {
            panic!("Expected Chunk content");
        }
    }

    #[test]
    fn test_knowledge_node_new_chunk_too_large() {
        let large_text = create_large_string(MAX_CHUNK_SIZE_BYTES + 1);
        let embedding = create_test_embedding();
        let extracted_triples = Vec::new();
        
        let result = KnowledgeNode::new_chunk(large_text, embedding, extracted_triples);
        assert!(result.is_err());
        
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("Chunk too large"));
        } else {
            panic!("Expected SerializationError for large chunk");
        }
    }

    #[test]
    fn test_knowledge_node_new_chunk_at_size_limit() {
        let text = create_large_string(MAX_CHUNK_SIZE_BYTES);
        let embedding = create_test_embedding();
        let extracted_triples = Vec::new();
        
        let result = KnowledgeNode::new_chunk(text, embedding, extracted_triples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_knowledge_node_new_chunk_word_count_calculation() {
        let text = "One two three four five".to_string();
        let embedding = create_test_embedding();
        let extracted_triples = Vec::new();
        
        let result = KnowledgeNode::new_chunk(text, embedding, extracted_triples);
        assert!(result.is_ok());
        
        let node = result.unwrap();
        if let NodeContent::Chunk { word_count, .. } = &node.content {
            assert_eq!(*word_count, 5);
        } else {
            panic!("Expected Chunk content");
        }
    }

    #[test]
    fn test_knowledge_node_new_chunk_size_calculation() {
        let text = "Test".to_string();
        let embedding = vec![0.1, 0.2]; // 2 * 4 bytes = 8 bytes
        let extracted_triples = vec![
            Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap()
        ];
        
        let result = KnowledgeNode::new_chunk(text.clone(), embedding.clone(), extracted_triples.clone());
        assert!(result.is_ok());
        
        let node = result.unwrap();
        let expected_size = text.len() + 
                           extracted_triples[0].memory_footprint() + 
                           embedding.len() * 4;
        assert_eq!(node.metadata.size_bytes, expected_size);
    }

    #[test]
    fn test_knowledge_node_new_entity_valid() {
        let name = "Einstein".to_string();
        let description = "Famous physicist".to_string();
        let entity_type = "Person".to_string();
        let mut properties = HashMap::new();
        properties.insert("birth_year".to_string(), "1879".to_string());
        let embedding = create_test_embedding();
        
        let result = KnowledgeNode::new_entity(
            name.clone(),
            description.clone(),
            entity_type.clone(),
            properties.clone(),
            embedding
        );
        
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(node.node_type, NodeType::Entity);
        
        if let NodeContent::Entity { name: entity_name, .. } = &node.content {
            assert_eq!(entity_name, &name);
        } else {
            panic!("Expected Entity content");
        }
    }

    #[test]
    fn test_knowledge_node_new_entity_too_large() {
        let large_name = create_large_string(MAX_CHUNK_SIZE_BYTES);
        let description = "Description".to_string();
        let entity_type = "Type".to_string();
        let properties = HashMap::new();
        let embedding = create_test_embedding();
        
        let result = KnowledgeNode::new_entity(
            large_name,
            description,
            entity_type,
            properties,
            embedding
        );
        
        assert!(result.is_err());
        if let Err(GraphError::SerializationError(msg)) = result {
            assert!(msg.contains("Entity data too large"));
        } else {
            panic!("Expected SerializationError for large entity");
        }
    }

    #[test]
    fn test_knowledge_node_generate_content_id() {
        let content1 = "test content";
        let content2 = "test content";
        let content3 = "different content";
        
        let id1 = KnowledgeNode::generate_content_id(content1);
        let id2 = KnowledgeNode::generate_content_id(content2);
        let id3 = KnowledgeNode::generate_content_id(content3);
        
        // Same content should generate same ID
        assert_eq!(id1, id2);
        
        // Different content should generate different ID
        assert_ne!(id1, id3);
        
        // ID should have correct format
        assert!(id1.starts_with("node_"));
    }

    #[test]
    fn test_knowledge_node_mark_accessed() {
        let triple = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let mut node = KnowledgeNode::new_triple(triple, create_test_embedding());
        
        let initial_access_count = node.metadata.access_count;
        let initial_last_accessed = node.metadata.last_accessed;
        
        // Wait a moment to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        node.mark_accessed();
        
        assert_eq!(node.metadata.access_count, initial_access_count + 1);
        assert!(node.metadata.last_accessed >= initial_last_accessed);
    }

    #[test]
    fn test_knowledge_node_get_triples() {
        // Test triple node
        let triple = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let triple_node = KnowledgeNode::new_triple(triple.clone(), create_test_embedding());
        
        let triples = triple_node.get_triples();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0], &triple);
        
        // Test chunk node
        let text = "Test chunk".to_string();
        let embedding = create_test_embedding();
        let extracted_triples = vec![
            Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap(),
            Triple::new("C".to_string(), "has".to_string(), "D".to_string()).unwrap(),
        ];
        
        let chunk_node = KnowledgeNode::new_chunk(text, embedding, extracted_triples.clone()).unwrap();
        let chunk_triples = chunk_node.get_triples();
        assert_eq!(chunk_triples.len(), 2);
        
        // Test entity node (should return empty)
        let entity_node = KnowledgeNode::new_entity(
            "Test".to_string(),
            "Description".to_string(),
            "Type".to_string(),
            HashMap::new(),
            create_test_embedding()
        ).unwrap();
        
        let entity_triples = entity_node.get_triples();
        assert_eq!(entity_triples.len(), 0);
    }

    #[test]
    fn test_knowledge_node_to_llm_format() {
        // Test triple node
        let triple = Triple::new("Einstein".to_string(), "is".to_string(), "scientist".to_string()).unwrap();
        let triple_node = KnowledgeNode::new_triple(triple, create_test_embedding());
        
        let llm_format = triple_node.to_llm_format();
        assert!(llm_format.starts_with("FACT:"));
        assert!(llm_format.contains("Einstein is scientist"));
        
        // Test chunk node
        let text = "This is a test chunk".to_string();
        let chunk_node = KnowledgeNode::new_chunk(text.clone(), create_test_embedding(), Vec::new()).unwrap();
        
        let chunk_format = chunk_node.to_llm_format();
        assert!(chunk_format.starts_with("KNOWLEDGE"));
        assert!(chunk_format.contains("5 words")); // "This is a test chunk"
        assert!(chunk_format.contains(&text));
        
        // Test entity node
        let entity_node = KnowledgeNode::new_entity(
            "Einstein".to_string(),
            "Famous physicist".to_string(),
            "Person".to_string(),
            HashMap::new(),
            create_test_embedding()
        ).unwrap();
        
        let entity_format = entity_node.to_llm_format();
        assert!(entity_format.starts_with("ENTITY:"));
        assert!(entity_format.contains("Einstein"));
        assert!(entity_format.contains("Person"));
        assert!(entity_format.contains("Famous physicist"));
    }

    #[test]
    fn test_knowledge_node_calculate_quality_score() {
        let triple = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let mut node = KnowledgeNode::new_triple(triple, create_test_embedding());
        
        // Initial quality score should be 1.0
        assert_eq!(node.metadata.quality_score, 1.0);
        
        // Simulate some access
        node.metadata.access_count = 10;
        node.calculate_quality_score();
        
        // Quality score should be between 0.0 and 1.0
        assert!(node.metadata.quality_score >= 0.0);
        assert!(node.metadata.quality_score <= 1.0);
    }

    #[test]
    fn test_predicate_vocabulary_new() {
        let vocab = PredicateVocabulary::new();
        
        // Should contain core predicates
        assert!(vocab.predicates.contains_key("is"));
        assert!(vocab.predicates.contains_key("has"));
        assert!(vocab.predicates.contains_key("located_in"));
        assert!(vocab.predicates.contains_key("created_by"));
    }

    #[test]
    fn test_predicate_vocabulary_normalize_canonical() {
        let vocab = PredicateVocabulary::new();
        
        // Test canonical forms
        assert_eq!(vocab.normalize("is"), "is");
        assert_eq!(vocab.normalize("has"), "has");
        assert_eq!(vocab.normalize("located_in"), "located_in");
    }

    #[test]
    fn test_predicate_vocabulary_normalize_case_and_spaces() {
        let vocab = PredicateVocabulary::new();
        
        // Test case normalization
        assert_eq!(vocab.normalize("IS"), "is");
        assert_eq!(vocab.normalize("HAS"), "has");
        
        // Test space to underscore conversion
        assert_eq!(vocab.normalize("located in"), "located_in");
        assert_eq!(vocab.normalize("created by"), "created_by");
        
        // Test trimming
        assert_eq!(vocab.normalize("  is  "), "is");
    }

    #[test]
    fn test_predicate_vocabulary_normalize_unknown() {
        let vocab = PredicateVocabulary::new();
        
        // Unknown predicates should be cleaned and returned
        assert_eq!(vocab.normalize("custom_predicate"), "custom_predicate");
        assert_eq!(vocab.normalize("UNKNOWN PRED"), "unknown_pred");
    }

    #[test]
    fn test_predicate_vocabulary_suggest_predicates() {
        let vocab = PredicateVocabulary::new();
        
        // Test context-based suggestions
        let suggestions = vocab.suggest_predicates("Entity classification");
        assert!(suggestions.contains(&"is".to_string()));
        
        let suggestions = vocab.suggest_predicates("Property ownership");
        assert!(suggestions.contains(&"has".to_string()));
        
        let suggestions = vocab.suggest_predicates("Spatial relationship");
        assert!(suggestions.contains(&"located_in".to_string()));
    }

    #[test]
    fn test_predicate_vocabulary_suggest_predicates_fallback() {
        let vocab = PredicateVocabulary::new();
        
        // Test fallback for unknown context
        let suggestions = vocab.suggest_predicates("completely unknown context");
        assert_eq!(suggestions.len(), 3);
        assert!(suggestions.contains(&"is".to_string()));
        assert!(suggestions.contains(&"has".to_string()));
        assert!(suggestions.contains(&"connected_to".to_string()));
    }

    #[test]
    fn test_predicate_vocabulary_suggest_predicates_predicate_name_match() {
        let vocab = PredicateVocabulary::new();
        
        // Test direct predicate name matching
        let suggestions = vocab.suggest_predicates("This causes that");
        assert!(suggestions.contains(&"causes".to_string()));
        
        let suggestions = vocab.suggest_predicates("A follows B");
        assert!(suggestions.contains(&"follows".to_string()));
    }

    #[test]
    fn test_predicate_vocabulary_default() {
        let vocab1 = PredicateVocabulary::new();
        let vocab2 = PredicateVocabulary::default();
        
        // Default should be equivalent to new()
        assert_eq!(vocab1.predicates.len(), vocab2.predicates.len());
        
        for (key, _) in &vocab1.predicates {
            assert!(vocab2.predicates.contains_key(key));
        }
    }

    #[test]
    fn test_predicate_info_structure() {
        let vocab = PredicateVocabulary::new();
        
        if let Some(info) = vocab.predicates.get("is") {
            assert_eq!(info.canonical_form, "is");
            assert_eq!(info.description, "Entity classification");
            assert_eq!(info.typical_domain, "Entity");
            assert_eq!(info.typical_range, "Type");
            assert!(info.aliases.is_empty()); // Initially no aliases
        } else {
            panic!("Expected 'is' predicate to exist");
        }
    }

    #[test]
    fn test_current_timestamp() {
        let timestamp1 = current_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let timestamp2 = current_timestamp();
        
        // Timestamps should be reasonable Unix timestamps
        assert!(timestamp1 > 1_600_000_000); // After 2020
        assert!(timestamp2 >= timestamp1);
    }

    #[test]
    fn test_node_metadata_initialization() {
        let triple = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let node = KnowledgeNode::new_triple(triple, create_test_embedding());
        
        assert!(node.metadata.created_at > 0);
        assert!(node.metadata.last_accessed > 0);
        assert_eq!(node.metadata.access_count, 0);
        assert!(node.metadata.size_bytes > 0);
        assert_eq!(node.metadata.quality_score, 1.0);
        assert!(node.metadata.tags.is_empty());
    }

    #[test]
    fn test_triple_equality_and_clone() {
        let triple1 = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let triple2 = triple1.clone();
        let triple3 = Triple::new("A".to_string(), "is".to_string(), "C".to_string()).unwrap();
        
        assert_eq!(triple1, triple2);
        assert_ne!(triple1, triple3);
    }

    #[test]
    fn test_node_type_equality() {
        assert_eq!(NodeType::Triple, NodeType::Triple);
        assert_eq!(NodeType::Chunk, NodeType::Chunk);
        assert_ne!(NodeType::Triple, NodeType::Chunk);
    }

    // Edge case tests for validation logic
    #[test]
    fn test_triple_validation_edge_cases() {
        // Empty strings
        let result = Triple::new("".to_string(), "is".to_string(), "something".to_string());
        assert!(result.is_ok()); // Empty should be allowed after trimming
        
        // Single character
        let result = Triple::new("A".to_string(), "i".to_string(), "B".to_string());
        assert!(result.is_ok());
        
        // Predicate with single word
        let result = Triple::new("A".to_string(), "is".to_string(), "B".to_string());
        assert!(result.is_ok());
        
        // Predicate with two words
        let result = Triple::new("A".to_string(), "is_a".to_string(), "B".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_chunk_validation_edge_cases() {
        // Empty text
        let result = KnowledgeNode::new_chunk("".to_string(), create_test_embedding(), Vec::new());
        assert!(result.is_ok());
        
        // Single character
        let result = KnowledgeNode::new_chunk("a".to_string(), create_test_embedding(), Vec::new());
        assert!(result.is_ok());
        
        // Text with only whitespace
        let result = KnowledgeNode::new_chunk("   ".to_string(), create_test_embedding(), Vec::new());
        assert!(result.is_ok());
        
        if let Ok(node) = result {
            if let NodeContent::Chunk { word_count, .. } = &node.content {
                assert_eq!(*word_count, 0); // Only whitespace should count as 0 words
            }
        }
    }

    #[test]
    fn test_memory_footprint_edge_cases() {
        // Triple with no source
        let triple = Triple::new("A".to_string(), "is".to_string(), "B".to_string()).unwrap();
        let expected = "A".len() + "is".len() + "B".len() + std::mem::size_of::<f32>();
        assert_eq!(triple.memory_footprint(), expected);
        
        // Triple with empty source
        let mut triple_empty_source = triple.clone();
        triple_empty_source.source = Some("".to_string());
        assert_eq!(triple_empty_source.memory_footprint(), expected); // Empty string adds 0
    }
}

