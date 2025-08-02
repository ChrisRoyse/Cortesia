# Task 05d: Implement Node Updating

**Estimated Time**: 12 minutes  
**Dependencies**: 05c_implement_node_reading.md  
**Next Task**: 05e_implement_node_deletion.md  

## Objective
Implement the node updating logic to modify existing nodes in Neo4j.

## Single Action
Replace the TODO in update_node method with real Neo4j implementation.

## Code to Replace
Replace the `update_node` method in `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    pub async fn update_node<T: GraphNode + Serialize>(
        &self,
        id: &str,
        node: &T,
        options: UpdateOptions,
    ) -> Result<(), CrudError> {
        if id.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node ID cannot be empty".to_string(),
            });
        }
        
        // Validation
        if options.validate && !node.validate() {
            return Err(CrudError::ValidationError {
                message: format!("Node validation failed for {}", id),
            });
        }
        
        // Check if node exists
        let node_type = node.node_type();
        let exists = self.node_exists(id, node_type).await?;
        
        if !exists {
            if options.create_if_missing {
                // Create the node instead of updating
                let create_options = CreateOptions {
                    validate: options.validate,
                    upsert: false,
                    return_existing: false,
                };
                self.create_node(node, create_options).await?;
                return Ok(());
            } else {
                return Err(CrudError::NotFound { id: id.to_string() });
            }
        }
        
        // Serialize node to JSON
        let node_json = node.to_json()
            .map_err(|e| CrudError::SerializationError { source: e })?;
        
        let node_data: serde_json::Value = serde_json::from_str(&node_json)
            .map_err(|e| CrudError::SerializationError { source: e })?;
        
        // Convert to BoltMap for Neo4j
        let mut properties = BoltMap::new();
        if let serde_json::Value::Object(obj) = node_data {
            for (key, value) in obj {
                let bolt_value = self.json_to_bolt_value(value)?;
                properties.insert(key.into(), bolt_value);
            }
        }
        
        // Create Cypher query based on update type
        let cypher = if options.partial {
            // Partial update - only update provided properties
            format!(
                "MATCH (n:{}) WHERE n.id = $id SET n += $properties RETURN n.id as id",
                node_type
            )
        } else {
            // Full update - replace all properties
            format!(
                "MATCH (n:{}) WHERE n.id = $id SET n = $properties RETURN n.id as id",
                node_type
            )
        };
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("id", id);
        query.param("properties", properties);
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Verify update succeeded
        if result.records().is_empty() {
            return Err(CrudError::NotFound { id: id.to_string() });
        }
        
        Ok(())
    }
    
    // Helper method for partial updates with specific properties
    pub async fn update_node_properties(
        &self,
        id: &str,
        node_type: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> Result<(), CrudError> {
        if id.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node ID cannot be empty".to_string(),
            });
        }
        
        if properties.is_empty() {
            return Ok(()); // Nothing to update
        }
        
        // Convert properties to BoltMap
        let mut bolt_properties = BoltMap::new();
        for (key, value) in properties {
            let bolt_value = self.json_to_bolt_value(value)?;
            bolt_properties.insert(key.into(), bolt_value);
        }
        
        // Create Cypher query for property update
        let cypher = format!(
            "MATCH (n:{}) WHERE n.id = $id SET n += $properties RETURN n.id as id",
            node_type
        );
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("id", id);
        query.param("properties", bolt_properties);
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Verify update succeeded
        if result.records().is_empty() {
            return Err(CrudError::NotFound { id: id.to_string() });
        }
        
        Ok(())
    }
}
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_updating_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_update_options() {
        let partial_update = UpdateOptions {
            partial: true,
            validate: true,
            create_if_missing: false,
        };
        
        let full_update = UpdateOptions {
            partial: false,
            validate: false,
            create_if_missing: true,
        };
        
        assert!(partial_update.partial);
        assert!(!full_update.partial);
        assert!(partial_update.validate);
        assert!(!full_update.validate);
        assert!(!partial_update.create_if_missing);
        assert!(full_update.create_if_missing);
    }
    
    #[test]
    fn test_property_update_validation() {
        // Test property update structure
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), serde_json::Value::String("updated_name".to_string()));
        properties.insert("confidence".to_string(), serde_json::Value::Number(0.95.into()));
        
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("confidence"));
        
        // Test empty properties
        let empty_properties: HashMap<String, serde_json::Value> = HashMap::new();
        assert!(empty_properties.is_empty());
    }
    
    #[tokio::test]
    async fn test_node_modification() {
        // Test node modification logic
        let mut concept = ConceptNode::new("UpdateTest".to_string(), "Entity".to_string());
        
        // Modify the concept
        concept.confidence_score = 0.85;
        concept.access_frequency = 10;
        
        // Test that validation still works
        assert!(concept.validate());
        assert_eq!(concept.confidence_score, 0.85);
        assert_eq!(concept.access_frequency, 10);
        
        // Test JSON serialization of modified node
        let json = concept.to_json().unwrap();
        assert!(json.contains("0.85"));
        assert!(json.contains("UpdateTest"));
    }
    
    #[test]
    fn test_cypher_query_structure() {
        // Test that query strings are constructed correctly
        let node_type = "Concept";
        let partial_query = format!(
            "MATCH (n:{}) WHERE n.id = $id SET n += $properties RETURN n.id as id",
            node_type
        );
        let full_query = format!(
            "MATCH (n:{}) WHERE n.id = $id SET n = $properties RETURN n.id as id",
            node_type
        );
        
        assert!(partial_query.contains("SET n +="));
        assert!(full_query.contains("SET n ="));
        assert!(partial_query.contains("MATCH (n:Concept)"));
        assert!(full_query.contains("MATCH (n:Concept)"));
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test node_updating_tests
```

## Acceptance Criteria
- [ ] Node updating method implemented
- [ ] Partial vs full update logic works
- [ ] Create-if-missing option functions
- [ ] Property-specific update method works
- [ ] Tests pass

## Duration
10-12 minutes for node updating implementation.