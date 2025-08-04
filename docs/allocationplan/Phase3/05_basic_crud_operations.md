# Task 05: Basic CRUD Operations

**Estimated Time**: 20-25 minutes  
**Dependencies**: 04_relationship_types.md  
**Stage**: Foundation  

## Objective
Implement comprehensive CRUD (Create, Read, Update, Delete) operations for all node and relationship types with proper error handling and validation.

## Specific Requirements

### 1. Node CRUD Operations
- Create nodes with validation
- Read nodes by ID and filters
- Update node properties
- Delete nodes with cascade handling
- Batch operations for performance

### 2. Relationship CRUD Operations
- Create relationships with constraint checking
- Query relationships by type and properties
- Update relationship metadata
- Delete relationships safely

### 3. Transaction Management
- Atomic operations for complex changes
- Rollback capability for failed operations
- Deadlock detection and handling

## Implementation Steps

### 1. Create CRUD Service Interface
```rust
// src/storage/crud_service.rs
use async_trait::async_trait;

#[async_trait]
pub trait CrudService<T> {
    async fn create(&self, item: &T) -> Result<String, CrudError>;
    async fn read(&self, id: &str) -> Result<Option<T>, CrudError>;
    async fn update(&self, id: &str, item: &T) -> Result<(), CrudError>;
    async fn delete(&self, id: &str) -> Result<(), CrudError>;
    async fn list(&self, filters: &FilterCriteria) -> Result<Vec<T>, CrudError>;
    async fn exists(&self, id: &str) -> Result<bool, CrudError>;
    async fn count(&self, filters: &FilterCriteria) -> Result<usize, CrudError>;
}

pub struct NodeCrudService<T> {
    connection_manager: Arc<Neo4jConnectionManager>,
    node_type: String,
    validator: Arc<dyn NodeValidator<T>>,
    cache: Arc<RwLock<LRUCache<String, T>>>,
}

impl<T> NodeCrudService<T> 
where 
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    pub fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
        node_type: String,
        validator: Arc<dyn NodeValidator<T>>,
    ) -> Self {
        Self {
            connection_manager,
            node_type,
            validator,
            cache: Arc::new(RwLock::new(LRUCache::new(1000))),
        }
    }
}
```

### 2. Implement Concept CRUD Operations
```rust
// src/storage/concept_crud.rs
#[async_trait]
impl CrudService<ConceptNode> for NodeCrudService<ConceptNode> {
    async fn create(&self, concept: &ConceptNode) -> Result<String, CrudError> {
        // Validate the concept
        self.validator.validate(concept).await?;
        
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
        
        let parameters = concept.to_neo4j_parameters()?;
        
        let mut result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            let id: String = record.get("id")?;
            
            // Cache the created concept
            self.cache.write().await.put(id.clone(), concept.clone());
            
            Ok(id)
        } else {
            Err(CrudError::CreateFailed("No ID returned".to_string()))
        }
    }
    
    async fn read(&self, id: &str) -> Result<Option<ConceptNode>, CrudError> {
        // Check cache first
        if let Some(cached_concept) = self.cache.read().await.get(id) {
            return Ok(Some(cached_concept.clone()));
        }
        
        let session = self.connection_manager.get_session().await?;
        
        let query = "MATCH (c:Concept {id: $id}) RETURN c";
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("id".to_string(), id.into());
        
        let mut result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            let concept_data: Value = record.get("c")?;
            let concept = ConceptNode::from_neo4j_value(concept_data)?;
            
            // Update cache
            self.cache.write().await.put(id.to_string(), concept.clone());
            
            // Update access tracking
            self.update_access_tracking(id).await?;
            
            Ok(Some(concept))
        } else {
            Ok(None)
        }
    }
    
    async fn update(&self, id: &str, concept: &ConceptNode) -> Result<(), CrudError> {
        // Validate the updated concept
        self.validator.validate(concept).await?;
        
        // Check if concept exists
        if !self.exists(id).await? {
            return Err(CrudError::NotFound(id.to_string()));
        }
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $id})
            SET c += $properties,
                c.last_modified = datetime()
            RETURN c
        "#;
        
        let mut parameters = concept.to_neo4j_parameters()?;
        parameters.insert("id".to_string(), id.into());
        
        session.run(query, Some(parameters)).await?;
        
        // Invalidate cache
        self.cache.write().await.pop(id);
        
        Ok(())
    }
    
    async fn delete(&self, id: &str) -> Result<(), CrudError> {
        // Check dependencies before deletion
        self.check_deletion_constraints(id).await?;
        
        let session = self.connection_manager.get_session().await?;
        
        // Start transaction for cascade deletion
        let transaction = session.begin_transaction().await?;
        
        // Delete related relationships first
        let delete_rels_query = r#"
            MATCH (c:Concept {id: $id})-[r]-()
            DELETE r
        "#;
        
        transaction.run(delete_rels_query, Some(hashmap!["id".to_string() => id.into()])).await?;
        
        // Delete the concept node
        let delete_node_query = r#"
            MATCH (c:Concept {id: $id})
            DELETE c
        "#;
        
        transaction.run(delete_node_query, Some(hashmap!["id".to_string() => id.into()])).await?;
        
        transaction.commit().await?;
        
        // Remove from cache
        self.cache.write().await.pop(id);
        
        Ok(())
    }
    
    async fn list(&self, filters: &FilterCriteria) -> Result<Vec<ConceptNode>, CrudError> {
        let session = self.connection_manager.get_session().await?;
        
        let (query, parameters) = self.build_list_query(filters)?;
        
        let result = session.run(&query, Some(parameters)).await?;
        
        let mut concepts = Vec::new();
        for record in result {
            let concept_data: Value = record.get("c")?;
            let concept = ConceptNode::from_neo4j_value(concept_data)?;
            concepts.push(concept);
        }
        
        Ok(concepts)
    }
}
```

### 3. Implement Batch Operations
```rust
// src/storage/batch_operations.rs
pub struct BatchOperations {
    connection_manager: Arc<Neo4jConnectionManager>,
    batch_size: usize,
}

impl BatchOperations {
    pub async fn batch_create_concepts(
        &self,
        concepts: Vec<ConceptNode>,
    ) -> Result<Vec<String>, BatchError> {
        let session = self.connection_manager.get_session().await?;
        let mut created_ids = Vec::new();
        
        for chunk in concepts.chunks(self.batch_size) {
            let transaction = session.begin_transaction().await?;
            
            for concept in chunk {
                let query = r#"
                    CREATE (c:Concept $properties)
                    RETURN c.id as id
                "#;
                
                let parameters = hashmap!["properties".to_string() => concept.to_neo4j_value()?];
                
                let mut result = transaction.run(query, Some(parameters)).await?;
                
                if let Some(record) = result.next().await? {
                    let id: String = record.get("id")?;
                    created_ids.push(id);
                } else {
                    transaction.rollback().await?;
                    return Err(BatchError::PartialFailure(created_ids));
                }
            }
            
            transaction.commit().await?;
        }
        
        Ok(created_ids)
    }
    
    pub async fn batch_update_properties(
        &self,
        updates: Vec<(String, HashMap<String, Value>)>,
    ) -> Result<(), BatchError> {
        let session = self.connection_manager.get_session().await?;
        
        for chunk in updates.chunks(self.batch_size) {
            let transaction = session.begin_transaction().await?;
            
            for (id, properties) in chunk {
                let query = r#"
                    MATCH (c:Concept {id: $id})
                    SET c += $properties
                "#;
                
                let parameters = hashmap![
                    "id".to_string() => id.into(),
                    "properties".to_string() => properties.clone().into()
                ];
                
                transaction.run(query, Some(parameters)).await?;
            }
            
            transaction.commit().await?;
        }
        
        Ok(())
    }
}
```

### 4. Add Query Builder for Complex Filters
```rust
// src/storage/query_builder.rs
pub struct QueryBuilder {
    node_type: String,
    filters: Vec<FilterCondition>,
    sorting: Vec<SortCondition>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl QueryBuilder {
    pub fn new(node_type: &str) -> Self {
        Self {
            node_type: node_type.to_string(),
            filters: Vec::new(),
            sorting: Vec::new(),
            limit: None,
            offset: None,
        }
    }
    
    pub fn filter(mut self, condition: FilterCondition) -> Self {
        self.filters.push(condition);
        self
    }
    
    pub fn sort_by(mut self, field: &str, direction: SortDirection) -> Self {
        self.sorting.push(SortCondition {
            field: field.to_string(),
            direction,
        });
        self
    }
    
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub fn build(&self) -> Result<(String, HashMap<String, Value>), QueryBuilderError> {
        let mut query = format!("MATCH (n:{}) ", self.node_type);
        let mut parameters = HashMap::new();
        
        // Build WHERE clause
        if !self.filters.is_empty() {
            query.push_str("WHERE ");
            let conditions: Vec<String> = self.filters
                .iter()
                .enumerate()
                .map(|(i, filter)| {
                    let param_name = format!("filter_{}", i);
                    parameters.insert(param_name.clone(), filter.value.clone());
                    filter.to_cypher_condition(&param_name)
                })
                .collect();
            query.push_str(&conditions.join(" AND "));
        }
        
        query.push_str(" RETURN n");
        
        // Build ORDER BY clause
        if !self.sorting.is_empty() {
            query.push_str(" ORDER BY ");
            let sorts: Vec<String> = self.sorting
                .iter()
                .map(|sort| format!("n.{} {}", sort.field, sort.direction.to_cypher()))
                .collect();
            query.push_str(&sorts.join(", "));
        }
        
        // Add LIMIT and SKIP
        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }
        
        if let Some(offset) = self.offset {
            query.push_str(&format!(" SKIP {}", offset));
        }
        
        Ok((query, parameters))
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All CRUD operations work for all node types
- [ ] Batch operations handle large datasets efficiently
- [ ] Transaction management ensures data consistency
- [ ] Complex queries with filters and sorting work correctly
- [ ] Cache operations improve read performance

### Performance Requirements
- [ ] Single node create/read/update/delete < 5ms
- [ ] Batch operations handle 1000+ items efficiently
- [ ] Cache hit rate > 80% for frequent reads
- [ ] Query builder generates optimized Cypher

### Testing Requirements
- [ ] Unit tests for all CRUD operations
- [ ] Integration tests with real database
- [ ] Performance tests for batch operations
- [ ] Transaction rollback tests

## Validation Steps

1. **Test basic CRUD operations**:
   ```rust
   let crud_service = NodeCrudService::new(connection_manager, "Concept".to_string(), validator);
   let id = crud_service.create(&concept).await?;
   let retrieved = crud_service.read(&id).await?;
   ```

2. **Test batch operations**:
   ```rust
   let batch_ops = BatchOperations::new(connection_manager);
   let ids = batch_ops.batch_create_concepts(concepts).await?;
   ```

3. **Run comprehensive tests**:
   ```bash
   cargo test crud_operations_tests
   ```

## Files to Create/Modify
- `src/storage/crud_service.rs` - CRUD service interface
- `src/storage/concept_crud.rs` - Concept CRUD implementation
- `src/storage/batch_operations.rs` - Batch operation support
- `src/storage/query_builder.rs` - Query building utilities
- `tests/storage/crud_tests.rs` - Comprehensive test suite

## Error Handling
- Validation failures
- Constraint violations
- Transaction rollbacks
- Cache inconsistencies
- Network/connection errors

## Success Metrics
- CRUD operation success rate: 100%
- Performance requirements met
- Zero data inconsistencies
- Cache efficiency > 80%

## Next Task
Upon completion, proceed to **06_ttfs_encoding_integration.md** to integrate TTFS encoding from Phase 2.