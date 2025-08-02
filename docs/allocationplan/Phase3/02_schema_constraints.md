# Task 02: Schema Constraints and Indices

**Estimated Time**: 15-25 minutes  
**Dependencies**: 01_neo4j_setup.md  
**Stage**: Foundation  

## Objective
Create database constraints, unique indices, and performance indices for the knowledge graph schema.

## Specific Requirements

### 1. Unique Constraints
- Ensure unique identifiers for all node types
- Prevent duplicate nodes with same ID
- Maintain referential integrity

### 2. Performance Indices
- Create indices for frequently queried properties
- Optimize inheritance traversal queries
- Support semantic similarity searches
- Enable temporal versioning queries

### 3. Validation Rules
- Implement schema validation at database level
- Add property type constraints where possible
- Ensure required fields are present

## Implementation Steps

### 1. Create Core Constraints
```cypher
// Unique constraints for core node types
CREATE CONSTRAINT concept_id_unique FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT memory_id_unique FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT property_id_unique FOR (p:Property) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT exception_id_unique FOR (e:Exception) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT version_id_unique FOR (v:Version) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT neural_pathway_id_unique FOR (n:NeuralPathway) REQUIRE n.id IS UNIQUE;

// Composite constraints for relationships
CREATE CONSTRAINT version_branch_unique FOR (v:Version) REQUIRE (v.branch_name, v.version_number) IS UNIQUE;
```

### 2. Create Performance Indices
```cypher
// Semantic and similarity indices
CREATE INDEX concept_ttfs_index FOR (c:Concept) ON (c.ttfs_encoding);
CREATE INDEX concept_semantic_embedding FOR (c:Concept) ON (c.semantic_embedding);
CREATE INDEX concept_type_index FOR (c:Concept) ON (c.type);
CREATE INDEX concept_name_index FOR (c:Concept) ON (c.name);

// Inheritance and hierarchy indices
CREATE INDEX inheritance_depth_index FOR (c:Concept) ON (c.inheritance_depth);
CREATE INDEX property_inheritance_index FOR (p:Property) ON (p.is_inheritable);
CREATE INDEX property_priority_index FOR (p:Property) ON (p.inheritance_priority);

// Temporal indices
CREATE INDEX memory_timestamp_index FOR (m:Memory) ON (m.created_at);
CREATE INDEX concept_created_index FOR (c:Concept) ON (c.creation_timestamp);
CREATE INDEX concept_accessed_index FOR (c:Concept) ON (c.last_accessed);
CREATE INDEX version_timestamp_index FOR (v:Version) ON (v.created_at);

// Performance optimization indices
CREATE INDEX concept_frequency_index FOR (c:Concept) ON (c.access_frequency);
CREATE INDEX memory_strength_index FOR (m:Memory) ON (m.strength);
CREATE INDEX neural_pathway_index FOR (n:NeuralPathway) ON (n.activation_pattern);
CREATE INDEX pathway_type_index FOR (n:NeuralPathway) ON (n.pathway_type);

// Relationship indices
CREATE INDEX inherits_depth_index FOR ()-[r:INHERITS_FROM]-() ON (r.inheritance_depth);
CREATE INDEX property_source_index FOR ()-[r:HAS_PROPERTY]-() ON (r.property_source);
CREATE INDEX semantic_similarity_index FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity_score);
```

### 3. Create Range Indices for Numeric Properties
```cypher
// Numeric range indices for performance queries
CREATE RANGE INDEX concept_confidence_range FOR (c:Concept) ON (c.confidence_score);
CREATE RANGE INDEX memory_decay_range FOR (m:Memory) ON (m.decay_rate);
CREATE RANGE INDEX exception_confidence_range FOR (e:Exception) ON (e.confidence);
CREATE RANGE INDEX similarity_score_range FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity_score);
CREATE RANGE INDEX inheritance_priority_range FOR (p:Property) ON (p.inheritance_priority);
```

### 4. Implement Schema Validation in Rust
```rust
// src/storage/schema_validator.rs
pub struct SchemaValidator {
    required_fields: HashMap<String, Vec<String>>,
    field_types: HashMap<String, HashMap<String, FieldType>>,
}

impl SchemaValidator {
    pub fn new() -> Self {
        let mut validator = Self {
            required_fields: HashMap::new(),
            field_types: HashMap::new(),
        };
        
        validator.initialize_concept_rules();
        validator.initialize_memory_rules();
        validator.initialize_property_rules();
        
        validator
    }
    
    pub fn validate_concept(&self, concept: &ConceptNode) -> Result<(), ValidationError> {
        // Validate required fields
        self.check_required_fields("Concept", concept)?;
        
        // Validate field types
        self.check_field_types("Concept", concept)?;
        
        // Validate business rules
        self.validate_concept_rules(concept)?;
        
        Ok(())
    }
    
    fn initialize_concept_rules(&mut self) {
        self.required_fields.insert("Concept".to_string(), vec![
            "id".to_string(),
            "name".to_string(),
            "type".to_string(),
            "creation_timestamp".to_string(),
        ]);
        
        let mut concept_types = HashMap::new();
        concept_types.insert("id".to_string(), FieldType::String);
        concept_types.insert("confidence_score".to_string(), FieldType::Float);
        concept_types.insert("inheritance_depth".to_string(), FieldType::Integer);
        
        self.field_types.insert("Concept".to_string(), concept_types);
    }
}
```

### 5. Create Index Management Service
```rust
// src/storage/index_manager.rs
pub struct IndexManager {
    connection: Arc<Neo4jConnectionManager>,
}

impl IndexManager {
    pub async fn ensure_all_indices(&self) -> Result<(), IndexError> {
        // Create all required indices and constraints
        self.create_constraints().await?;
        self.create_performance_indices().await?;
        self.create_range_indices().await?;
        
        Ok(())
    }
    
    pub async fn get_index_status(&self) -> Result<IndexStatus, IndexError> {
        // Query database for index status
        let query = "SHOW INDEXES YIELD name, state, populationPercent";
        // Implementation
    }
    
    pub async fn rebuild_index(&self, index_name: &str) -> Result<(), IndexError> {
        // Drop and recreate specific index
        // Implementation
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All unique constraints are created and enforced
- [ ] Performance indices are created for all specified properties
- [ ] Range indices are created for numeric properties
- [ ] Schema validation prevents invalid data insertion
- [ ] Index status monitoring is functional

### Performance Requirements
- [ ] Index creation completes within 30 seconds for empty database
- [ ] Constraint validation adds <1ms overhead per operation
- [ ] Query performance improves by >50% with indices

### Testing Requirements
- [ ] Unit tests for schema validator
- [ ] Integration tests for constraint enforcement
- [ ] Performance tests for indexed vs non-indexed queries
- [ ] Tests for constraint violation handling

## Validation Steps

1. **Verify constraint creation**:
   ```cypher
   SHOW CONSTRAINTS
   ```

2. **Verify index creation**:
   ```cypher
   SHOW INDEXES
   ```

3. **Test constraint enforcement**:
   ```cypher
   // Should fail - duplicate ID
   CREATE (c1:Concept {id: "test_id", name: "Test 1"})
   CREATE (c2:Concept {id: "test_id", name: "Test 2"})
   ```

4. **Run schema validation tests**:
   ```bash
   cargo test schema_validation_tests
   ```

## Files to Create/Modify
- `src/storage/schema_validator.rs` - Schema validation logic
- `src/storage/index_manager.rs` - Index management service
- `src/storage/constraints.cypher` - Constraint creation queries
- `src/storage/indices.cypher` - Index creation queries
- `tests/storage/schema_tests.rs` - Schema validation tests
- `tests/storage/constraint_tests.rs` - Constraint enforcement tests

## Error Handling
- Constraint violation errors
- Index creation failures
- Schema validation errors
- Duplicate constraint creation attempts
- Index corruption detection

## Success Metrics
- Constraint creation success rate: 100%
- Index creation time: <30 seconds
- Query performance improvement: >50%
- Schema validation error detection: 100%

## Next Task
Upon completion, proceed to **03_core_node_types.md** to implement basic node type definitions.