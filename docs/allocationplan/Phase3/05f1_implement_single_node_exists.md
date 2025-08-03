# Task 05f1: Implement Single Node Exists Check

**Estimated Time**: 6 minutes  
**Dependencies**: 05e_implement_node_deletion.md  
**Next Task**: 05f2_implement_batch_nodes_exist.md  

## Objective
Replace the TODO in node_exists method with basic existence check.

## Single Action
Replace the core node_exists method only.

## Code to Replace
Replace the `node_exists` method in `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    pub async fn node_exists(&self, id: &str, node_type: &str) -> Result<bool, CrudError> {
        if id.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node ID cannot be empty".to_string(),
            });
        }
        
        if node_type.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node type cannot be empty".to_string(),
            });
        }
        
        // Create optimized Cypher query that only checks existence
        let cypher = format!(
            "MATCH (n:{}) WHERE n.id = $id RETURN count(n) > 0 as exists",
            node_type
        );
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("id", id);
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Extract existence result
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::Boolean(exists)) = record.get("exists") {
                return Ok(*exists);
            }
        }
        
        // If we can't get a boolean result, assume node doesn't exist
        Ok(false)
    }
}
```

## Success Check
```bash
cargo check
cargo test node_exists
```

## Acceptance Criteria
- [ ] node_exists method compiles
- [ ] Uses optimized count query
- [ ] Validates input parameters
- [ ] Returns boolean result

## Duration
4-6 minutes for basic existence check.