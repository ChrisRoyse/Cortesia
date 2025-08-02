# API Reference

Complete reference documentation for the Enhanced Knowledge Storage System API.

## Table of Contents

- [Core Components](#core-components)
- [Configuration Types](#configuration-types)
- [Processing Types](#processing-types)
- [Storage Types](#storage-types)
- [Error Types](#error-types)
- [Usage Examples](#usage-examples)

## Core Components

### IntelligentKnowledgeProcessor

The main entry point for processing knowledge documents with AI-powered enhancement.

#### Constructor

```rust
pub fn new(
    model_manager: Arc<ModelResourceManager>,
    config: KnowledgeProcessingConfig,
) -> Self
```

Creates a new intelligent knowledge processor with the specified model manager and configuration.

**Parameters:**
- `model_manager: Arc<ModelResourceManager>` - Shared model resource manager for AI models
- `config: KnowledgeProcessingConfig` - Processing configuration parameters

**Returns:**
- `IntelligentKnowledgeProcessor` - New processor instance

**Example:**
```rust
let model_config = ModelResourceConfig::default();
let model_manager = Arc::new(ModelResourceManager::new(model_config));
let processing_config = KnowledgeProcessingConfig::default();

let processor = IntelligentKnowledgeProcessor::new(
    model_manager,
    processing_config
);
```

#### Methods

##### `process_knowledge`

```rust
pub async fn process_knowledge(
    &self,
    content: &str,
    title: &str,
) -> Result<KnowledgeProcessingResult, EnhancedStorageError>
```

Processes a document using the full AI-powered pipeline including entity extraction, relationship mapping, and semantic chunking.

**Parameters:**
- `content: &str` - The raw text content to process
- `title: &str` - A descriptive title for the document

**Returns:**
- `Result<KnowledgeProcessingResult, EnhancedStorageError>` - Processing results or error

**Processing Pipeline:**
1. Global context analysis
2. Semantic chunking with boundary detection
3. Entity extraction using SmolLM models
4. Relationship mapping and validation
5. Quality metrics calculation

**Example:**
```rust
let result = processor.process_knowledge(
    "Einstein developed the theory of relativity in 1905...",
    "Physics History"
).await?;

println!("Document ID: {}", result.document_id);
println!("Quality Score: {:.2}", result.quality_metrics.overall_quality);
```

##### `get_processing_stats`

```rust
pub fn get_processing_stats(&self, result: &KnowledgeProcessingResult) -> ProcessingStats
```

Extracts processing statistics from a processing result.

**Parameters:**
- `result: &KnowledgeProcessingResult` - Processing result to analyze

**Returns:**
- `ProcessingStats` - Performance and quality statistics

**Example:**
```rust
let result = processor.process_knowledge(content, title).await?;
let stats = processor.get_processing_stats(&result);

println!("Total chunks: {}", stats.total_chunks);
println!("Processing time: {:?}", stats.processing_time);
println!("Quality score: {:.2}", stats.quality_score);
```

##### `validate_processing_result`

```rust
pub fn validate_processing_result(&self, result: &KnowledgeProcessingResult) -> ProcessingValidation
```

Validates processing results against quality thresholds and configuration requirements.

**Parameters:**
- `result: &KnowledgeProcessingResult` - Processing result to validate

**Returns:**
- `ProcessingValidation` - Validation results with errors and warnings

**Example:**
```rust
let validation = processor.validate_processing_result(&result);
if !validation.is_valid {
    for error in &validation.errors {
        eprintln!("Error: {}", error);
    }
}
```

### ModelResourceManager

Manages AI model loading, caching, and resource allocation.

#### Constructor

```rust
pub fn new(config: ModelResourceConfig) -> Self
```

Creates a new model resource manager with the specified configuration.

**Parameters:**
- `config: ModelResourceConfig` - Configuration for resource management

**Returns:**
- `ModelResourceManager` - New manager instance

#### Methods

##### `load_model`

```rust
pub async fn load_model(&self, model_id: &str) -> Result<Arc<dyn Model>, EnhancedStorageError>
```

Loads and caches an AI model for processing tasks.

**Parameters:**
- `model_id: &str` - Identifier of the model to load

**Returns:**
- `Result<Arc<dyn Model>, EnhancedStorageError>` - Loaded model or error

##### `get_model_info`

```rust
pub fn get_model_info(&self, model_id: &str) -> Option<ModelMetadata>
```

Retrieves metadata about a model including memory requirements and capabilities.

**Parameters:**
- `model_id: &str` - Model identifier

**Returns:**
- `Option<ModelMetadata>` - Model metadata if available

##### `evict_idle_models`

```rust
pub async fn evict_idle_models(&self) -> usize
```

Removes idle models from cache to free memory.

**Returns:**
- `usize` - Number of models evicted

## Configuration Types

### KnowledgeProcessingConfig

Configuration parameters for knowledge processing.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProcessingConfig {
    /// Model ID for entity extraction
    pub entity_extraction_model: String,
    
    /// Model ID for relationship extraction  
    pub relationship_extraction_model: String,
    
    /// Model ID for semantic analysis
    pub semantic_analysis_model: String,
    
    /// Maximum size for semantic chunks (characters)
    pub max_chunk_size: usize,
    
    /// Minimum size for semantic chunks (characters)
    pub min_chunk_size: usize,
    
    /// Overlap size between chunks (characters)
    pub chunk_overlap_size: usize,
    
    /// Minimum confidence threshold for entities
    pub min_entity_confidence: f32,
    
    /// Minimum confidence threshold for relationships
    pub min_relationship_confidence: f32,
    
    /// Whether to preserve global context across chunks
    pub preserve_context: bool,
    
    /// Enable quality validation during processing
    pub enable_quality_validation: bool,
}
```

#### Default Values

```rust
impl Default for KnowledgeProcessingConfig {
    fn default() -> Self {
        Self {
            entity_extraction_model: "smollm2_360m".to_string(),
            relationship_extraction_model: "smollm_360m_instruct".to_string(),
            semantic_analysis_model: "smollm2_135m".to_string(),
            max_chunk_size: 2048,
            min_chunk_size: 128,
            chunk_overlap_size: 64,
            min_entity_confidence: 0.6,
            min_relationship_confidence: 0.5,
            preserve_context: true,
            enable_quality_validation: true,
        }
    }
}
```

### ModelResourceConfig

Configuration for model resource management.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResourceConfig {
    /// Maximum total memory usage across all loaded models (bytes)
    pub max_memory_usage: u64,
    
    /// Maximum number of models that can be loaded simultaneously
    pub max_concurrent_models: usize,
    
    /// Time after which idle models are eligible for eviction
    pub idle_timeout: Duration,
    
    /// Minimum memory threshold below which no eviction occurs
    pub min_memory_threshold: u64,
}
```

#### Default Values

```rust
impl Default for ModelResourceConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 2_000_000_000, // 2GB
            max_concurrent_models: 3,
            idle_timeout: Duration::from_secs(300), // 5 minutes
            min_memory_threshold: 100_000_000, // 100MB
        }
    }
}
```

## Processing Types

### KnowledgeProcessingResult

Complete result of knowledge processing operation.

```rust
#[derive(Debug, Clone)]
pub struct KnowledgeProcessingResult {
    /// Unique identifier for the processed document
    pub document_id: String,
    
    /// Semantic chunks with preserved context
    pub chunks: Vec<SemanticChunk>,
    
    /// All entities extracted from the document
    pub global_entities: Vec<ContextualEntity>,
    
    /// All relationships found in the document
    pub global_relationships: Vec<ComplexRelationship>,
    
    /// Document structure and organization
    pub document_structure: DocumentStructure,
    
    /// Processing metadata and statistics
    pub processing_metadata: ProcessingMetadata,
    
    /// Quality metrics for the processing result
    pub quality_metrics: QualityMetrics,
}
```

### SemanticChunk

A semantically coherent portion of a document with extracted knowledge.

```rust
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// Unique identifier for the chunk
    pub id: String,
    
    /// The text content of the chunk
    pub content: String,
    
    /// Starting character position in the original document
    pub start_pos: usize,
    
    /// Ending character position in the original document
    pub end_pos: usize,
    
    /// Semantic coherence score (0.0 to 1.0)
    pub semantic_coherence: f32,
    
    /// Key concepts identified in the chunk
    pub key_concepts: Vec<String>,
    
    /// Entities found within this chunk
    pub entities: Vec<ContextualEntity>,
    
    /// Relationships found within this chunk
    pub relationships: Vec<ComplexRelationship>,
    
    /// Type of chunk (paragraph, section, etc.)
    pub chunk_type: ChunkType,
    
    /// Overlap information with previous chunk
    pub overlap_with_previous: Option<ChunkOverlap>,
    
    /// Overlap information with next chunk
    pub overlap_with_next: Option<ChunkOverlap>,
}
```

### ContextualEntity

An entity extracted from text with contextual information.

```rust
#[derive(Debug, Clone)]
pub struct ContextualEntity {
    /// The name/text of the entity
    pub name: String,
    
    /// Classification of the entity type
    pub entity_type: EntityType,
    
    /// Surrounding context where entity was found
    pub context: String,
    
    /// Confidence score for the extraction (0.0 to 1.0)
    pub confidence: f32,
    
    /// Character span in the original text
    pub span: Option<(usize, usize)>,
    
    /// Additional attributes associated with the entity
    pub attributes: HashMap<String, String>,
    
    /// Timestamp when entity was extracted
    pub extracted_at: u64,
}
```

### ComplexRelationship

A relationship between entities with contextual information.

```rust
#[derive(Debug, Clone)]
pub struct ComplexRelationship {
    /// Source entity in the relationship
    pub source: String,
    
    /// Type of relationship
    pub predicate: RelationshipType,
    
    /// Target entity in the relationship
    pub target: String,
    
    /// Context where relationship was found
    pub context: String,
    
    /// Confidence score for the relationship (0.0 to 1.0)
    pub confidence: f32,
    
    /// Supporting evidence for the relationship
    pub supporting_evidence: Vec<String>,
    
    /// Strength of the relationship (0.0 to 1.0)
    pub relationship_strength: f32,
    
    /// Temporal information if applicable
    pub temporal_info: Option<TemporalInfo>,
}
```

### QualityMetrics

Quality assessment metrics for processing results.

```rust
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Quality score for entity extraction (0.0 to 1.0)
    pub entity_extraction_quality: f32,
    
    /// Quality score for relationship extraction (0.0 to 1.0)
    pub relationship_extraction_quality: f32,
    
    /// Semantic coherence across chunks (0.0 to 1.0)
    pub semantic_coherence: f32,
    
    /// Context preservation score (0.0 to 1.0)
    pub context_preservation: f32,
    
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
}

impl QualityMetrics {
    /// Calculate overall quality from component scores
    pub fn calculate_overall_quality(&mut self) {
        self.overall_quality = (
            self.entity_extraction_quality * 0.3 +
            self.relationship_extraction_quality * 0.25 +
            self.semantic_coherence * 0.25 +
            self.context_preservation * 0.2
        ).min(1.0);
    }
}
```

## Storage Types

### DocumentStructure

Structural representation of a processed document.

```rust
#[derive(Debug, Clone)]
pub struct DocumentStructure {
    /// Logical sections within the document
    pub sections: Vec<DocumentSection>,
    
    /// Overall topic or theme of the document
    pub overall_topic: Option<String>,
    
    /// Key themes identified in the document
    pub key_themes: Vec<String>,
    
    /// Assessed complexity level
    pub complexity_level: ComplexityLevel,
    
    /// Estimated reading time
    pub estimated_reading_time: Duration,
}
```

### DocumentSection

A logical section within a document.

```rust
#[derive(Debug, Clone)]
pub struct DocumentSection {
    /// Optional title for the section
    pub title: Option<String>,
    
    /// Starting character position
    pub start_pos: usize,
    
    /// Ending character position
    pub end_pos: usize,
    
    /// Type of section (header, body, conclusion, etc.)
    pub section_type: SectionType,
    
    /// Key points identified in this section
    pub key_points: Vec<String>,
}
```

## Enumeration Types

### EntityType

Classification of extracted entities.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person names
    Person,
    
    /// Organizations, companies, institutions
    Organization,
    
    /// Geographic locations
    Location,
    
    /// Dates and time references
    Date,
    
    /// Abstract concepts and ideas
    Concept,
    
    /// Technologies, tools, systems
    Technology,
    
    /// Events and occurrences
    Event,
    
    /// Measurements and quantities
    Measurement,
    
    /// Other entity types
    Other(String),
}
```

### RelationshipType

Types of relationships between entities.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Entity created or authored by another
    CreatedBy,
    
    /// Entity is part of another
    PartOf,
    
    /// Entity is located in another
    LocatedIn,
    
    /// Event occurred at a specific time/place
    OccurredAt,
    
    /// Entity influenced by another
    InfluencedBy,
    
    /// Event caused by another
    CausedBy,
    
    /// Entities are similar
    SimilarTo,
    
    /// Entities are opposite
    OppositeOf,
    
    /// Custom relationship type
    Custom(String),
}
```

### ChunkType

Types of semantic chunks.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkType {
    /// Single paragraph
    Paragraph,
    
    /// Document section
    Section,
    
    /// List item
    ListItem,
    
    /// Table content
    Table,
    
    /// Code block
    Code,
    
    /// Quote or citation
    Quote,
    
    /// Header or title
    Header,
    
    /// Other chunk types
    Other,
}
```

### ComplexityLevel

Assessment of processing complexity.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Simple text processing, basic entity extraction
    Low,
    
    /// Moderate complexity analysis, relationship extraction
    Medium,
    
    /// Complex reasoning, multi-step processing
    High,
}
```

## Error Types

### EnhancedStorageError

All possible errors in the enhanced storage system.

```rust
#[derive(Debug, thiserror::Error)]
pub enum EnhancedStorageError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
```

## Result Types

### ProcessingStats

Performance statistics for processing operations.

```rust
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Total number of chunks created
    pub total_chunks: usize,
    
    /// Total number of entities extracted
    pub total_entities: usize,
    
    /// Total number of relationships found
    pub total_relationships: usize,
    
    /// Total processing time
    pub processing_time: Duration,
    
    /// Average chunk size in characters
    pub average_chunk_size: f32,
    
    /// Overall quality score
    pub quality_score: f32,
    
    /// List of AI models used
    pub models_used: Vec<String>,
}
```

### ProcessingValidation

Validation results for processing quality.

```rust
#[derive(Debug, Clone)]
pub struct ProcessingValidation {
    /// Whether the processing result passes validation
    pub is_valid: bool,
    
    /// Overall quality score
    pub quality_score: f32,
    
    /// Critical errors that prevent usage
    pub errors: Vec<String>,
    
    /// Warnings about quality or configuration issues
    pub warnings: Vec<String>,
}
```

## Usage Examples

### Basic Processing

```rust
use llmkg::enhanced_knowledge_storage::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup
    let model_manager = Arc::new(ModelResourceManager::new(
        ModelResourceConfig::default()
    ));
    let processor = IntelligentKnowledgeProcessor::new(
        model_manager,
        KnowledgeProcessingConfig::default()
    );
    
    // Process knowledge
    let content = "Albert Einstein developed the theory of relativity...";
    let result = processor.process_knowledge(content, "Physics").await?;
    
    // Access results
    println!("Processed {} chunks", result.chunks.len());
    println!("Found {} entities", result.global_entities.len());
    
    Ok(())
}
```

### Custom Configuration

```rust
let custom_config = KnowledgeProcessingConfig {
    entity_extraction_model: "smollm2_360m".to_string(),
    max_chunk_size: 4096,
    min_entity_confidence: 0.7,
    preserve_context: true,
    ..Default::default()
};

let processor = IntelligentKnowledgeProcessor::new(
    model_manager,
    custom_config
);
```

### Quality Validation

```rust
let result = processor.process_knowledge(content, title).await?;
let validation = processor.validate_processing_result(&result);

if !validation.is_valid {
    eprintln!("Processing failed validation:");
    for error in &validation.errors {
        eprintln!("  Error: {}", error);
    }
    for warning in &validation.warnings {
        eprintln!("  Warning: {}", warning);
    }
} else {
    println!("Processing quality: {:.2}", validation.quality_score);
}
```

### Performance Monitoring

```rust
let start = std::time::Instant::now();
let result = processor.process_knowledge(content, title).await?;
let processing_time = start.elapsed();

let stats = processor.get_processing_stats(&result);
println!("Performance Summary:");
println!("  Processing time: {:?}", processing_time);
println!("  Average chunk size: {:.0} chars", stats.average_chunk_size);
println!("  Quality score: {:.2}", stats.quality_score);
println!("  Models used: {:?}", stats.models_used);
```

### Error Handling

```rust
match processor.process_knowledge(content, title).await {
    Ok(result) => {
        println!("Successfully processed: {}", result.document_id);
    },
    Err(EnhancedStorageError::ModelNotFound(model)) => {
        eprintln!("Required model not available: {}", model);
    },
    Err(EnhancedStorageError::InsufficientResources(msg)) => {
        eprintln!("Not enough resources: {}", msg);
    },
    Err(e) => {
        eprintln!("Processing failed: {}", e);
    }
}
```

This API reference provides comprehensive documentation for all public interfaces in the Enhanced Knowledge Storage System, enabling developers to effectively integrate and use the system in their applications.