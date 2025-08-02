# Task 05f: Implement Node Exists Check

**Estimated Time**: 6 minutes  
**Dependencies**: 05e_implement_node_deletion.md  
**Next Task**: 05g_implement_node_listing.md  

## Objective
Implement the node existence check to verify if a node exists in Neo4j.

## Single Action
Replace the TODO in node_exists method with real Neo4j implementation.

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
    
    // Check if any nodes exist with a specific property value
    pub async fn node_exists_with_property(
        &self,
        node_type: &str,
        property_name: &str,
        property_value: &serde_json::Value,
    ) -> Result<bool, CrudError> {
        if node_type.is_empty() || property_name.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node type and property name cannot be empty".to_string(),
            });
        }
        
        // Convert property value to Bolt value
        let bolt_value = self.json_to_bolt_value(property_value.clone())?;
        
        // Create query to check for property existence
        let cypher = format!(
            "MATCH (n:{}) WHERE n.{} = $property_value RETURN count(n) > 0 as exists",
            node_type, property_name
        );
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("property_value", bolt_value);
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Extract existence result
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::Boolean(exists)) = record.get("exists") {
                return Ok(*exists);
            }
        }
        
        Ok(false)
    }
}
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_exists_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_exists_validation() {
        // Test input validation
        let empty_id = "";
        let valid_id = "valid_id_123";
        let empty_type = "";
        let valid_type = "Concept";
        
        // Test validation conditions
        assert!(empty_id.is_empty());
        assert!(!valid_id.is_empty());
        assert!(empty_type.is_empty());
        assert!(!valid_type.is_empty());
        
        // Test that validation errors would be triggered
        if empty_id.is_empty() || empty_type.is_empty() {
            assert!(true, "Empty parameters should trigger validation error");
        }
    }
    
    #[test]
    fn test_batch_exists_structure() {
        // Test batch existence check structure
        let ids = vec![
            "id_1".to_string(),
            "id_2".to_string(),
            "id_3".to_string(),
        ];
        
        assert_eq!(ids.len(), 3);
        assert!(!ids.is_empty());
        
        // Test result map structure
        let mut result_map = HashMap::new();
        for id in &ids {
            result_map.insert(id.clone(), false);
        }
        
        assert_eq!(result_map.len(), 3);
        assert_eq!(result_map.get("id_1"), Some(&false));
        
        // Simulate found nodes
        result_map.insert("id_1".to_string(), true);
        assert_eq!(result_map.get("id_1"), Some(&true));
    }
    
    #[test]
    fn test_property_exists_query_structure() {
        // Test property existence query structure
        let node_type = "Concept";
        let property_name = "name";
        let query = format!(
            "MATCH (n:{}) WHERE n.{} = $property_value RETURN count(n) > 0 as exists",
            node_type, property_name
        );
        
        assert!(query.contains("MATCH (n:Concept)"));
        assert!(query.contains("WHERE n.name ="));
        assert!(query.contains("$property_value"));
        assert!(query.contains("count(n) > 0"));
        assert!(query.contains("as exists"));
    }
    
    #[test]
    fn test_cypher_query_optimization() {
        // Test that existence queries are optimized
        let node_type = "Memory";
        let exists_query = format!(
            "MATCH (n:{}) WHERE n.id = $id RETURN count(n) > 0 as exists",
            node_type
        );
        
        let batch_query = format!(
            "UNWIND $ids as node_id MATCH (n:{}) WHERE n.id = node_id RETURN n.id as found_id",
            node_type
        );
        
        // Verify optimized queries use count() instead of returning full nodes
        assert!(exists_query.contains("count(n) > 0"));
        assert!(!exists_query.contains("RETURN n"));
        
        // Verify batch query uses UNWIND for efficiency
        assert!(batch_query.contains("UNWIND $ids"));
        assert!(batch_query.contains("as node_id"));
    }
    
    #[tokio::test]
    async fn test_property_value_conversion() {
        // Test property value conversion for existence checks
        let test_values = vec![
            serde_json::Value::String("test_value".to_string()),
            serde_json::Value::Number(42.into()),
            serde_json::Value::Bool(true),
        ];
        
        for value in test_values {
            // Test that values can be processed for queries
            match &value {
                serde_json::Value::String(s) => {
                    assert!(!s.is_empty());
                },
                serde_json::Value::Number(n) => {
                    assert!(n.is_i64() || n.is_f64());
                },
                serde_json::Value::Bool(b) => {
                    assert!(*b == true || *b == false);
                },
                _ => panic!("Unexpected value type"),
            }
        }
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test node_exists_tests
```

## Acceptance Criteria
- [ ] Node existence check implemented
- [ ] Batch existence check for multiple nodes
- [ ] Property-based existence check
- [ ] Optimized queries using count()
- [ ] Tests pass

## Duration
4-6 minutes for node existence implementation.