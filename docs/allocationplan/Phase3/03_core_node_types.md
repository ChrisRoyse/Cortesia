# Task 03: Core Node Type Definitions

**Estimated Time**: 20-30 minutes  
**Dependencies**: 02_schema_constraints.md  
**Stage**: Foundation  

## Objective
Implement the core node type definitions (Concept, Memory, Property, Exception, Version, NeuralPathway) with proper data structures and validation.

## Specific Requirements

### 1. Rust Data Structures
- Define structs for all node types
- Implement serialization/deserialization
- Add field validation and business rules
- Support for Neo4j integration

### 2. Node Type Implementations
- Concept nodes with semantic properties
- Memory nodes with temporal properties
- Property nodes with inheritance support
- Exception nodes for override handling
- Version nodes for temporal versioning
- NeuralPathway nodes for neural metadata

### 3. Business Logic
- Node creation and validation
- Property assignment and inheritance
- Temporal versioning support
- Neural pathway integration

## Implementation Steps

### 1. Define Core Data Structures
```rust
// src/storage/node_types.rs
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub name: String,
    pub concept_type: String,
    pub ttfs_encoding: Option<f32>,
    pub inheritance_depth: i32,
    pub property_count: i32,
    pub inherited_property_count: i32,
    pub semantic_embedding: Vec<f32>,
    pub creation_timestamp: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_frequency: i32,
    pub confidence_score: f32,
    pub source_attribution: Option<String>,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub context: Option<String>,
    pub memory_type: MemoryType,
    pub strength: f32,
    pub decay_rate: f32,
    pub consolidation_level: ConsolidationLevel,
    pub neural_pattern: Option<String>,
    pub retrieval_count: i32,
    pub last_strengthened: DateTime<Utc>,
    pub associated_emotions: Vec<String>,
    pub sensory_modalities: Vec<SensoryModality>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyNode {
    pub id: String,
    pub name: String,
    pub value: PropertyValue,
    pub data_type: DataType,
    pub inheritance_priority: i32,
    pub is_inheritable: bool,
    pub is_overridable: bool,
    pub validation_rules: Option<String>,
    pub default_value: Option<PropertyValue>,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceptionNode {
    pub id: String,
    pub property_name: String,
    pub original_value: PropertyValue,
    pub exception_value: PropertyValue,
    pub exception_reason: String,
    pub confidence: f32,
    pub evidence_sources: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub validated_at: Option<DateTime<Utc>>,
    pub validation_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNode {
    pub id: String,
    pub branch_name: String,
    pub version_number: i32,
    pub parent_version: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub change_summary: String,
    pub node_count: i32,
    pub relationship_count: i32,
    pub memory_usage_bytes: i64,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPathwayNode {
    pub id: String,
    pub pathway_type: PathwayType,
    pub activation_pattern: Vec<f32>,
    pub cortical_columns: Vec<String>,
    pub processing_time_ms: f32,
    pub network_types_used: Vec<String>,
    pub ttfs_timings: Vec<f32>,
    pub lateral_inhibition_events: i32,
    pub stdp_weight_changes: Vec<f32>,
    pub confidence_score: f32,
    pub created_at: DateTime<Utc>,
}
```

### 2. Define Supporting Enums
```rust
// src/storage/node_enums.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Validated,
    Pending,
    Conflicted,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    WorkingMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsolidationLevel {
    Working,
    ShortTerm,
    LongTerm,
    Permanent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Tactile,
    Olfactory,
    Gustatory,
    Proprioceptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathwayType {
    Allocation,
    Retrieval,
    Update,
    Validation,
    Consolidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<PropertyValue>),
    Object(std::collections::HashMap<String, PropertyValue>),
    Embedding(Vec<f32>),
}
```

### 3. Implement Node Builders
```rust
// src/storage/node_builders.rs
pub struct ConceptNodeBuilder {
    concept: ConceptNode,
}

impl ConceptNodeBuilder {
    pub fn new(name: &str, concept_type: &str) -> Self {
        Self {
            concept: ConceptNode {
                id: Uuid::new_v4().to_string(),
                name: name.to_string(),
                concept_type: concept_type.to_string(),
                ttfs_encoding: None,
                inheritance_depth: 0,
                property_count: 0,
                inherited_property_count: 0,
                semantic_embedding: Vec::new(),
                creation_timestamp: Utc::now(),
                last_accessed: Utc::now(),
                access_frequency: 0,
                confidence_score: 1.0,
                source_attribution: None,
                validation_status: ValidationStatus::Pending,
            },
        }
    }
    
    pub fn with_ttfs_encoding(mut self, encoding: f32) -> Self {
        self.concept.ttfs_encoding = Some(encoding);
        self
    }
    
    pub fn with_semantic_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.concept.semantic_embedding = embedding;
        self
    }
    
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.concept.confidence_score = confidence.clamp(0.0, 1.0);
        self
    }
    
    pub fn build(self) -> Result<ConceptNode, BuilderError> {
        self.validate()?;
        Ok(self.concept)
    }
    
    fn validate(&self) -> Result<(), BuilderError> {
        if self.concept.name.is_empty() {
            return Err(BuilderError::EmptyName);
        }
        
        if self.concept.confidence_score < 0.0 || self.concept.confidence_score > 1.0 {
            return Err(BuilderError::InvalidConfidence);
        }
        
        Ok(())
    }
}

// Similar builders for other node types...
```

### 4. Implement Node Operations
```rust
// src/storage/node_operations.rs
pub trait NodeOperations<T> {
    async fn create(&self, node: &T) -> Result<String, NodeError>;
    async fn read(&self, id: &str) -> Result<Option<T>, NodeError>;
    async fn update(&self, id: &str, node: &T) -> Result<(), NodeError>;
    async fn delete(&self, id: &str) -> Result<(), NodeError>;
    async fn list(&self, filters: &NodeFilters) -> Result<Vec<T>, NodeError>;
}

pub struct ConceptOperations {
    connection_manager: Arc<Neo4jConnectionManager>,
}

impl NodeOperations<ConceptNode> for ConceptOperations {
    async fn create(&self, concept: &ConceptNode) -> Result<String, NodeError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            CREATE (c:Concept {
                id: $id,
                name: $name,
                type: $concept_type,
                ttfs_encoding: $ttfs_encoding,
                inheritance_depth: $inheritance_depth,
                property_count: $property_count,
                inherited_property_count: $inherited_property_count,
                semantic_embedding: $semantic_embedding,
                creation_timestamp: $creation_timestamp,
                last_accessed: $last_accessed,
                access_frequency: $access_frequency,
                confidence_score: $confidence_score,
                source_attribution: $source_attribution,
                validation_status: $validation_status
            })
            RETURN c.id as id
        "#;
        
        let result = session.run(query, Some(concept.to_neo4j_parameters())).await?;
        
        // Extract and return the created ID
        Ok(concept.id.clone())
    }
    
    async fn read(&self, id: &str) -> Result<Option<ConceptNode>, NodeError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = "MATCH (c:Concept {id: $id}) RETURN c";
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("id".to_string(), id.into());
        
        let result = session.run(query, Some(parameters)).await?;
        
        // Parse result and convert to ConceptNode
        // Implementation details...
        
        Ok(None) // Placeholder
    }
}
```

### 5. Add Validation and Error Handling
```rust
// src/storage/node_errors.rs
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("Database connection error: {0}")]
    Connection(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Node not found: {0}")]
    NotFound(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}

#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("Name cannot be empty")]
    EmptyName,
    
    #[error("Confidence score must be between 0.0 and 1.0")]
    InvalidConfidence,
    
    #[error("Invalid semantic embedding dimension")]
    InvalidEmbedding,
    
    #[error("Required field missing: {0}")]
    MissingField(String),
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All node types are properly defined with required fields
- [ ] Node builders create valid instances with validation
- [ ] CRUD operations work for all node types
- [ ] Serialization/deserialization works correctly
- [ ] Business rule validation is enforced

### Performance Requirements
- [ ] Node creation time < 5ms
- [ ] Node retrieval time < 2ms
- [ ] Batch operations support 100+ nodes
- [ ] Memory usage is optimized for large embeddings

### Testing Requirements
- [ ] Unit tests for all node types and builders
- [ ] Integration tests with database operations
- [ ] Validation tests for all business rules
- [ ] Serialization round-trip tests

## Validation Steps

1. **Test node creation**:
   ```rust
   let concept = ConceptNodeBuilder::new("test_concept", "Entity")
       .with_confidence(0.95)
       .build()?;
   ```

2. **Test database operations**:
   ```rust
   let concept_ops = ConceptOperations::new(connection_manager);
   let id = concept_ops.create(&concept).await?;
   let retrieved = concept_ops.read(&id).await?;
   ```

3. **Run validation tests**:
   ```bash
   cargo test node_types_tests
   ```

## Files to Create/Modify
- `src/storage/node_types.rs` - Core node type definitions
- `src/storage/node_enums.rs` - Supporting enums
- `src/storage/node_builders.rs` - Node builder patterns
- `src/storage/node_operations.rs` - CRUD operations
- `src/storage/node_errors.rs` - Error types
- `tests/storage/node_tests.rs` - Comprehensive test suite

## Error Handling
- Invalid field values
- Missing required fields
- Constraint violations
- Serialization failures
- Database operation errors

## Success Metrics
- Node creation success rate: 100%
- Validation error detection: 100%
- CRUD operation performance meets requirements
- Zero data corruption incidents

## Next Task
Upon completion, proceed to **04_relationship_types.md** to implement relationship type definitions.