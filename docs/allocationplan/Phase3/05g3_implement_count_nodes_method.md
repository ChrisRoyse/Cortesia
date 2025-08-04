# Task 05g3: Implement Count Nodes Method

**Estimated Time**: 8 minutes  
**Dependencies**: 05g2_add_filter_criteria_helpers.md  
**Next Task**: 05g4_test_listing_functionality.md  

## Objective
Add helper method to count nodes matching filters.

## Single Action
Add count_nodes method to BasicNodeOperations.

## Code to Add
Add to `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    // Helper method to count nodes matching filters
    pub async fn count_nodes(&self, filters: &FilterCriteria) -> Result<usize, CrudError> {
        let node_type = filters.node_type.as_ref()
            .ok_or_else(|| CrudError::ValidationError {
                message: "Node type must be specified for counting".to_string(),
            })?;
        
        // Build WHERE clause from property filters
        let mut where_clauses = Vec::new();
        let mut params = BoltMap::new();
        
        for (i, (key, value)) in filters.properties.iter().enumerate() {
            let param_name = format!("prop_{}", i);
            where_clauses.push(format!("n.{} = ${}", key, param_name));
            params.insert(param_name.into(), BoltValue::String(value.clone().into()));
        }
        
        let where_part = if where_clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_clauses.join(" AND "))
        };
        
        let cypher = format!("MATCH (n:{}){} RETURN count(n) as node_count", node_type, where_part);
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        
        // Add parameters
        for (key, value) in params {
            query.param(&key, value);
        }
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Extract count
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::Integer(count)) = record.get("node_count") {
                return Ok(*count as usize);
            }
        }
        
        Ok(0)
    }
}
```

## Success Check
```bash
cargo check
cargo test count_nodes
```

## Acceptance Criteria
- [ ] count_nodes method compiles
- [ ] Uses same filter logic as list_nodes
- [ ] Returns count as usize
- [ ] Handles empty results

## Duration
6-8 minutes for count method.