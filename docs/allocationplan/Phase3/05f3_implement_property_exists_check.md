# Task 05f3: Implement Property-Based Existence Check

**Estimated Time**: 9 minutes  
**Dependencies**: 05f2_implement_batch_nodes_exist.md  
**Next Task**: 05f4_test_existence_functionality.md  

## Objective
Add method to check if nodes exist with specific property values.

## Single Action
Add node_exists_with_property method.

## Code to Add
Add to `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
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

## Success Check
```bash
cargo check
cargo test property_exists
```

## Acceptance Criteria
- [ ] Property existence method compiles
- [ ] Handles JSON to Bolt conversion
- [ ] Uses parameterized queries safely
- [ ] Returns boolean result

## Duration
7-9 minutes for property-based existence.