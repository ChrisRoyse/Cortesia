# Task 05f2: Implement Batch Node Existence Check

**Estimated Time**: 8 minutes  
**Dependencies**: 05f1_implement_single_node_exists.md  
**Next Task**: 05f3_implement_property_exists_check.md  

## Objective
Add batch existence check for multiple nodes at once.

## Single Action
Add nodes_exist method for batch checking.

## Code to Add
Add to `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    // Batch existence check for multiple nodes
    pub async fn nodes_exist(
        &self, 
        ids: &[String], 
        node_type: &str
    ) -> Result<HashMap<String, bool>, CrudError> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        
        if node_type.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node type cannot be empty".to_string(),
            });
        }
        
        // Create batch query to check multiple nodes at once
        let cypher = format!(
            "UNWIND $ids as node_id MATCH (n:{}) WHERE n.id = node_id RETURN n.id as found_id",
            node_type
        );
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("ids", ids.to_vec());
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Build result map
        let mut existence_map = HashMap::new();
        
        // Initialize all IDs as not existing
        for id in ids {
            existence_map.insert(id.clone(), false);
        }
        
        // Mark found IDs as existing
        for record in result.records() {
            if let Some(BoltValue::String(found_id)) = record.get("found_id") {
                existence_map.insert(found_id.to_string(), true);
            }
        }
        
        Ok(existence_map)
    }
}
```

## Success Check
```bash
cargo check
cargo test batch_exists
```

## Acceptance Criteria
- [ ] Batch method compiles
- [ ] Uses UNWIND for efficiency
- [ ] Returns HashMap of results
- [ ] Handles empty input gracefully

## Duration
6-8 minutes for batch existence check.