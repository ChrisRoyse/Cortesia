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

