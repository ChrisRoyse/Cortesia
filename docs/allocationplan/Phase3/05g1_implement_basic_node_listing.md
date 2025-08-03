# Task 05g1: Implement Basic Node Listing

**Estimated Time**: 10 minutes  
**Dependencies**: 05f_implement_node_exists_check.md  
**Next Task**: 05g2_add_filter_criteria_helpers.md  

## Objective
Replace the TODO in list_nodes method with basic Neo4j query implementation.

## Single Action
Replace the core list_nodes method only.

## Code to Replace
Replace the `list_nodes` method in `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    pub async fn list_nodes<T>(
        &self,
        filters: &FilterCriteria,
    ) -> Result<Vec<T>, CrudError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Validate filters
        if let Some(ref node_type) = filters.node_type {
            if node_type.is_empty() {
                return Err(CrudError::ValidationError {
                    message: "Node type cannot be empty".to_string(),
                });
            }
        } else {
            return Err(CrudError::ValidationError {
                message: "Node type must be specified for listing".to_string(),
            });
        }
        
        let node_type = filters.node_type.as_ref().unwrap();
        
        // Build WHERE clause from property filters
        let mut where_clauses = Vec::new();
        let mut params = BoltMap::new();
        
        for (i, (key, value)) in filters.properties.iter().enumerate() {
            let param_name = format!("prop_{}", i);
            where_clauses.push(format!("n.{} = ${}", key, param_name));
            params.insert(param_name.into(), BoltValue::String(value.clone().into()));
        }
        
        // Build ORDER BY clause
        let order_clause = if let Some(ref order_by) = filters.order_by {
            if filters.ascending {
                format!(" ORDER BY n.{} ASC", order_by)
            } else {
                format!(" ORDER BY n.{} DESC", order_by)
            }
        } else {
            " ORDER BY n.id ASC".to_string()
        };
        
        // Build LIMIT and SKIP clauses
        let limit_clause = if let Some(limit) = filters.limit {
            format!(" LIMIT {}", limit)
        } else {
            String::new()
        };
        
        let skip_clause = if let Some(offset) = filters.offset {
            format!(" SKIP {}", offset)
        } else {
            String::new()
        };
        
        // Construct complete Cypher query
        let where_part = if where_clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_clauses.join(" AND "))
        };
        
        let cypher = format!(
            "MATCH (n:{}){}{}{}{} RETURN n",
            node_type, where_part, order_clause, skip_clause, limit_clause
        );
        
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
        
        // Convert results to nodes
        let mut nodes = Vec::new();
        
        for record in result.records() {
            if let Some(BoltValue::Node(node)) = record.get("n") {
                // Convert node properties to JSON
                let properties_json = self.bolt_map_to_json(&node.properties())?;
                
                // Deserialize to target type
                let node_instance: T = serde_json::from_value(properties_json)
                    .map_err(|e| CrudError::SerializationError { source: e })?;
                
                nodes.push(node_instance);
            }
        }
        
        Ok(nodes)
    }
}
```

## Success Check
```bash
cargo check
```

## Acceptance Criteria
- [ ] list_nodes method compiles
- [ ] Handles filtering by properties
- [ ] Supports ordering and pagination
- [ ] Returns deserialized nodes

## Duration
8-10 minutes for core listing logic.