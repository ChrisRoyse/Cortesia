# Task 04: Relationship Type Definitions

**Estimated Time**: 25-30 minutes  
**Dependencies**: 03_core_node_types.md  
**Stage**: Foundation  

## Objective
Implement relationship type definitions and operations for connecting nodes in the knowledge graph with proper metadata and validation.

## Specific Requirements

### 1. Relationship Data Structures
- INHERITS_FROM relationships for hierarchy
- HAS_PROPERTY relationships for ownership
- HAS_EXCEPTION relationships for overrides
- SEMANTICALLY_RELATED relationships for similarity
- TEMPORAL_SEQUENCE relationships for time ordering
- NEURAL_PATHWAY relationships for neural metadata
- VERSION_OF relationships for temporal versioning

### 2. Relationship Operations
- Create and validate relationships
- Query relationships with filters
- Update relationship properties
- Handle relationship constraints

### 3. Metadata Management
- Store relationship-specific metadata
- Support weighted relationships
- Track relationship creation and modification
- Validate relationship consistency

## Implementation Steps

### 1. Define Relationship Structures
```rust
// src/storage/relationship_types.rs
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritsFromRelationship {
    pub inheritance_type: InheritanceType,
    pub inheritance_depth: i32,
    pub property_mask: Vec<String>,
    pub exception_count: i32,
    pub strength: f32,
    pub established_at: DateTime<Utc>,
    pub last_validated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasPropertyRelationship {
    pub property_source: PropertySource,
    pub inheritance_path: Vec<String>,
    pub override_level: i32,
    pub is_default: bool,
    pub confidence: f32,
    pub established_by: String,
    pub last_modified: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasExceptionRelationship {
    pub exception_type: ExceptionType,
    pub precedence: i32,
    pub scope: ExceptionScope,
    pub validation_status: ValidationStatus,
    pub supporting_evidence: Vec<String>,
    pub confidence_score: f32,
    pub detected_by: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticallyRelatedRelationship {
    pub relationship_type: SemanticRelationType,
    pub similarity_score: f32,
    pub distance_metric: DistanceMetric,
    pub computed_by: String,
    pub vector_distance: f32,
    pub context_dependent: bool,
    pub temporal_stability: f32,
    pub established_at: DateTime<Utc>,
    pub last_verified: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSequenceRelationship {
    pub sequence_type: TemporalType,
    pub temporal_distance: i32,
    pub confidence: f32,
    pub precision: TemporalPrecision,
    pub established_by: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPathwayRelationship {
    pub pathway_id: String,
    pub pathway_strength: f32,
    pub activation_frequency: i32,
    pub stdp_weight: f32,
    pub refractory_period: i32,
    pub last_activation: DateTime<Utc>,
    pub pathway_efficiency: f32,
    pub cortical_column_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionOfRelationship {
    pub version_id: String,
    pub change_type: ChangeType,
    pub diff_data: String,
    pub rollback_data: String,
    pub change_confidence: f32,
    pub created_by: String,
    pub change_timestamp: DateTime<Utc>,
}
```

### 2. Define Supporting Enums
```rust
// src/storage/relationship_enums.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InheritanceType {
    Direct,
    Transitive,
    Multiple,
    Virtual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertySource {
    Direct,
    Inherited,
    Computed,
    Exception,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionType {
    PropertyOverride,
    ConceptException,
    InheritanceException,
    ValidationException,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionScope {
    Local,
    Inherited,
    Global,
    Contextual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticRelationType {
    Similar,
    Opposite,
    PartOf,
    InstanceOf,
    RelatedTo,
    Synonym,
    Antonym,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Jaccard,
    Manhattan,
    Hamming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalType {
    Before,
    After,
    During,
    Overlaps,
    Concurrent,
    Sequential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPrecision {
    Exact,
    Approximate,
    Relative,
    Fuzzy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
    Moved,
    Merged,
    Split,
}
```

### 3. Implement Relationship Operations
```rust
// src/storage/relationship_operations.rs
pub struct RelationshipOperations {
    connection_manager: Arc<Neo4jConnectionManager>,
    relationship_cache: Arc<RwLock<HashMap<String, CachedRelationship>>>,
}

impl RelationshipOperations {
    pub async fn create_inheritance_relationship(
        &self,
        child_id: &str,
        parent_id: &str,
        relationship: InheritsFromRelationship,
    ) -> Result<(), RelationshipError> {
        // Validate relationship constraints
        self.validate_inheritance_constraints(child_id, parent_id, &relationship).await?;
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (child:Concept {id: $child_id})
            MATCH (parent:Concept {id: $parent_id})
            CREATE (child)-[r:INHERITS_FROM {
                inheritance_type: $inheritance_type,
                inheritance_depth: $inheritance_depth,
                property_mask: $property_mask,
                exception_count: $exception_count,
                strength: $strength,
                established_at: $established_at,
                last_validated: $last_validated
            }]->(parent)
            RETURN r
        "#;
        
        let parameters = self.build_inheritance_parameters(child_id, parent_id, &relationship);
        session.run(query, Some(parameters)).await?;
        
        // Update inheritance depth for child and descendants
        self.update_inheritance_depths(child_id).await?;
        
        // Invalidate affected caches
        self.invalidate_inheritance_cache(child_id).await;
        
        Ok(())
    }
    
    pub async fn create_property_relationship(
        &self,
        concept_id: &str,
        property_id: &str,
        relationship: HasPropertyRelationship,
    ) -> Result<(), RelationshipError> {
        // Validate property assignment rules
        self.validate_property_assignment(concept_id, property_id, &relationship).await?;
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $concept_id})
            MATCH (p:Property {id: $property_id})
            CREATE (c)-[r:HAS_PROPERTY {
                property_source: $property_source,
                inheritance_path: $inheritance_path,
                override_level: $override_level,
                is_default: $is_default,
                confidence: $confidence,
                established_by: $established_by,
                last_modified: $last_modified
            }]->(p)
            RETURN r
        "#;
        
        let parameters = self.build_property_parameters(concept_id, property_id, &relationship);
        session.run(query, Some(parameters)).await?;
        
        Ok(())
    }
    
    pub async fn create_semantic_relationship(
        &self,
        source_id: &str,
        target_id: &str,
        relationship: SemanticallyRelatedRelationship,
    ) -> Result<(), RelationshipError> {
        // Validate semantic similarity constraints
        self.validate_semantic_relationship(&relationship).await?;
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (source:Concept {id: $source_id})
            MATCH (target:Concept {id: $target_id})
            CREATE (source)-[r:SEMANTICALLY_RELATED {
                relationship_type: $relationship_type,
                similarity_score: $similarity_score,
                distance_metric: $distance_metric,
                computed_by: $computed_by,
                vector_distance: $vector_distance,
                context_dependent: $context_dependent,
                temporal_stability: $temporal_stability,
                established_at: $established_at,
                last_verified: $last_verified
            }]->(target)
            RETURN r
        "#;
        
        let parameters = self.build_semantic_parameters(source_id, target_id, &relationship);
        session.run(query, Some(parameters)).await?;
        
        Ok(())
    }
}
```

### 4. Add Relationship Validation
```rust
// src/storage/relationship_validator.rs
pub struct RelationshipValidator {
    max_inheritance_depth: i32,
    min_similarity_threshold: f32,
    max_property_overrides: i32,
}

impl RelationshipValidator {
    pub fn new() -> Self {
        Self {
            max_inheritance_depth: 10,
            min_similarity_threshold: 0.1,
            max_property_overrides: 5,
        }
    }
    
    pub async fn validate_inheritance_constraints(
        &self,
        child_id: &str,
        parent_id: &str,
        relationship: &InheritsFromRelationship,
    ) -> Result<(), ValidationError> {
        // Check for circular inheritance
        if self.would_create_cycle(child_id, parent_id).await? {
            return Err(ValidationError::CircularInheritance);
        }
        
        // Check inheritance depth limits
        if relationship.inheritance_depth > self.max_inheritance_depth {
            return Err(ValidationError::ExceedsMaxDepth);
        }
        
        // Validate inheritance strength
        if relationship.strength < 0.0 || relationship.strength > 1.0 {
            return Err(ValidationError::InvalidStrength);
        }
        
        Ok(())
    }
    
    pub async fn validate_semantic_relationship(
        &self,
        relationship: &SemanticallyRelatedRelationship,
    ) -> Result<(), ValidationError> {
        // Validate similarity score
        if relationship.similarity_score < self.min_similarity_threshold {
            return Err(ValidationError::SimilarityTooLow);
        }
        
        // Validate temporal stability
        if relationship.temporal_stability < 0.0 || relationship.temporal_stability > 1.0 {
            return Err(ValidationError::InvalidStability);
        }
        
        Ok(())
    }
    
    async fn would_create_cycle(
        &self,
        child_id: &str,
        parent_id: &str,
    ) -> Result<bool, ValidationError> {
        // Check if parent_id is already a descendant of child_id
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (parent:Concept {id: $parent_id})-[:INHERITS_FROM*]->(ancestor:Concept {id: $child_id})
            RETURN COUNT(path) > 0 as would_cycle
        "#;
        
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("parent_id".to_string(), parent_id.into());
        parameters.insert("child_id".to_string(), child_id.into());
        
        let result = session.run(query, Some(parameters)).await?;
        
        // Parse result to determine if cycle would be created
        Ok(false) // Placeholder
    }
}
```

### 5. Implement Relationship Queries
```rust
// src/storage/relationship_queries.rs
pub struct RelationshipQueries {
    connection_manager: Arc<Neo4jConnectionManager>,
}

impl RelationshipQueries {
    pub async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<Vec<InheritanceNode>, QueryError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (child:Concept {id: $concept_id})-[:INHERITS_FROM*]->(ancestor:Concept)
            RETURN nodes(path) as inheritance_path, relationships(path) as inheritance_rels
            ORDER BY length(path)
        "#;
        
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("concept_id".to_string(), concept_id.into());
        
        let result = session.run(query, Some(parameters)).await?;
        
        // Parse and return inheritance chain
        Ok(Vec::new()) // Placeholder
    }
    
    pub async fn find_semantic_neighbors(
        &self,
        concept_id: &str,
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<SemanticNeighbor>, QueryError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (source:Concept {id: $concept_id})-[r:SEMANTICALLY_RELATED]->(target:Concept)
            WHERE r.similarity_score >= $min_similarity
            RETURN target, r
            ORDER BY r.similarity_score DESC
            LIMIT $limit
        "#;
        
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("concept_id".to_string(), concept_id.into());
        parameters.insert("min_similarity".to_string(), min_similarity.into());
        parameters.insert("limit".to_string(), (limit as i64).into());
        
        let result = session.run(query, Some(parameters)).await?;
        
        // Parse and return semantic neighbors
        Ok(Vec::new()) // Placeholder
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All relationship types are properly defined and functional
- [ ] Relationship creation includes proper validation
- [ ] Circular inheritance detection works correctly
- [ ] Relationship queries return accurate results
- [ ] Relationship metadata is stored and retrievable

### Performance Requirements
- [ ] Relationship creation time < 5ms
- [ ] Inheritance chain queries < 10ms for depth 10
- [ ] Semantic similarity queries < 20ms
- [ ] Cycle detection queries < 15ms

### Testing Requirements
- [ ] Unit tests for all relationship types
- [ ] Integration tests with database operations
- [ ] Validation tests for constraint enforcement
- [ ] Performance tests for query operations

## Validation Steps

1. **Test relationship creation**:
   ```rust
   let rel_ops = RelationshipOperations::new(connection_manager);
   rel_ops.create_inheritance_relationship("child", "parent", inheritance_rel).await?;
   ```

2. **Test cycle detection**:
   ```rust
   // Should detect and prevent circular inheritance
   rel_ops.create_inheritance_relationship("parent", "child", inheritance_rel).await?;
   ```

3. **Run relationship tests**:
   ```bash
   cargo test relationship_tests
   ```

## Files to Create/Modify
- `src/storage/relationship_types.rs` - Relationship definitions
- `src/storage/relationship_enums.rs` - Supporting enums
- `src/storage/relationship_operations.rs` - CRUD operations
- `src/storage/relationship_validator.rs` - Validation logic
- `src/storage/relationship_queries.rs` - Query operations
- `tests/storage/relationship_tests.rs` - Test suite

## Error Handling
- Circular inheritance prevention
- Invalid relationship constraints
- Missing node references
- Validation rule violations
- Query execution errors

## Success Metrics
- Relationship creation success rate: 100%
- Cycle detection accuracy: 100%
- Query performance meets requirements
- Zero constraint violations

## Next Task
Upon completion, proceed to **05_basic_crud_operations.md** to implement basic CRUD operations.