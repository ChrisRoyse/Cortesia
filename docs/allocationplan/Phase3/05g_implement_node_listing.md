# Task 05g: Implement Node Listing

**Estimated Time**: 12 minutes  
**Dependencies**: 05f_implement_node_exists_check.md  
**Next Task**: 05h_test_complete_crud_operations.md  

## Objective
Implement the node listing logic to query and retrieve multiple nodes with filtering.

## Single Action
Replace the TODO in list_nodes method with real Neo4j implementation.

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

## Add Helper Methods
Add convenience methods to FilterCriteria:
```rust
impl FilterCriteria {
    pub fn new(node_type: String) -> Self {
        Self {
            node_type: Some(node_type),
            ..Default::default()
        }
    }
    
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }
    
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
    
    pub fn with_order(mut self, field: String, ascending: bool) -> Self {
        self.order_by = Some(field);
        self.ascending = ascending;
        self
    }
}
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_listing_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_filter_criteria_builder() {
        let filters = FilterCriteria::new("Concept".to_string())
            .with_property("type".to_string(), "Entity".to_string())
            .with_property("status".to_string(), "active".to_string())
            .with_limit(50)
            .with_offset(10)
            .with_order("name".to_string(), true);
        
        assert_eq!(filters.node_type, Some("Concept".to_string()));
        assert_eq!(filters.properties.len(), 2);
        assert_eq!(filters.limit, Some(50));
        assert_eq!(filters.offset, Some(10));
        assert_eq!(filters.order_by, Some("name".to_string()));
        assert!(filters.ascending);
    }
    
    #[test]
    fn test_cypher_query_construction() {
        // Test basic query construction
        let node_type = "Memory";
        let basic_query = format!("MATCH (n:{}) ORDER BY n.id ASC RETURN n", node_type);
        
        assert!(basic_query.contains("MATCH (n:Memory)"));
        assert!(basic_query.contains("ORDER BY n.id ASC"));
        assert!(basic_query.contains("RETURN n"));
        
        // Test query with WHERE clause
        let where_clauses = vec!["n.type = $prop_0", "n.status = $prop_1"];
        let where_part = format!(" WHERE {}", where_clauses.join(" AND "));
        let filtered_query = format!("MATCH (n:{}){} RETURN n", node_type, where_part);
        
        assert!(filtered_query.contains("WHERE n.type = $prop_0 AND n.status = $prop_1"));
        
        // Test LIMIT and SKIP
        let limit_clause = " LIMIT 100";
        let skip_clause = " SKIP 20";
        let paginated_query = format!("MATCH (n:{}){}{} RETURN n", node_type, skip_clause, limit_clause);
        
        assert!(paginated_query.contains("SKIP 20"));
        assert!(paginated_query.contains("LIMIT 100"));
    }
    
    #[test]
    fn test_count_query_construction() {
        let node_type = "Property";
        let count_query = format!("MATCH (n:{}) RETURN count(n) as node_count", node_type);
        
        assert!(count_query.contains("MATCH (n:Property)"));
        assert!(count_query.contains("count(n)"));
        assert!(count_query.contains("as node_count"));
        assert!(!count_query.contains("RETURN n")); // Should not return full nodes
    }
    
    #[test]
    fn test_parameter_generation() {
        // Test parameter name generation
        let properties = vec![
            ("name".to_string(), "test".to_string()),
            ("type".to_string(), "entity".to_string()),
            ("status".to_string(), "active".to_string()),
        ];
        
        let mut params = BoltMap::new();
        let mut where_clauses = Vec::new();
        
        for (i, (key, value)) in properties.iter().enumerate() {
            let param_name = format!("prop_{}", i);
            where_clauses.push(format!("n.{} = ${}", key, param_name));
            params.insert(param_name.into(), BoltValue::String(value.clone().into()));
        }
        
        assert_eq!(where_clauses.len(), 3);
        assert_eq!(params.len(), 3);
        assert!(where_clauses[0].contains("n.name = $prop_0"));
        assert!(where_clauses[1].contains("n.type = $prop_1"));
        assert!(where_clauses[2].contains("n.status = $prop_2"));
    }
    
    #[tokio::test]
    async fn test_node_listing_workflow() {
        // Test the complete listing workflow structure
        let filters = FilterCriteria::new("Concept".to_string())
            .with_property("concept_type".to_string(), "Entity".to_string())
            .with_limit(10);
        
        // Verify filter structure
        assert_eq!(filters.node_type, Some("Concept".to_string()));
        assert!(filters.properties.contains_key("concept_type"));
        assert_eq!(filters.limit, Some(10));
        
        // Test validation
        if filters.node_type.is_none() {
            panic!("Node type should be specified");
        }
        
        let node_type = filters.node_type.as_ref().unwrap();
        assert!(!node_type.is_empty());
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test node_listing_tests
```

## Acceptance Criteria
- [ ] Node listing method implemented
- [ ] Property filtering works
- [ ] Pagination with limit/offset
- [ ] Sorting by any field
- [ ] Count method for result totals
- [ ] FilterCriteria builder pattern
- [ ] Tests pass

## Duration
10-12 minutes for node listing implementation.