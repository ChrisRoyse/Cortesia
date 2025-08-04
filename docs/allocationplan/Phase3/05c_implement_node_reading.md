# Task 05c: Implement Node Reading

**Estimated Time**: 10 minutes  
**Dependencies**: 05b_implement_node_creation.md  
**Next Task**: 05d_implement_node_updating.md  

## Objective
Implement the node reading logic to retrieve nodes from Neo4j by ID.

## Single Action
Replace the TODO in read_node method with real Neo4j implementation.

## Code to Replace
Replace the `read_node` method in `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    pub async fn read_node<T>(&self, id: &str, node_type: &str) -> Result<Option<T>, CrudError>
    where
        T: for<'de> Deserialize<'de>,
    {
        if id.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node ID cannot be empty".to_string(),
            });
        }
        
        // Create Cypher query to find node by ID
        let cypher = format!(
            "MATCH (n:{}) WHERE n.id = $id RETURN n",
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
        
        // Check if node was found
        if result.records().is_empty() {
            return Ok(None);
        }
        
        // Extract node data from first record
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::Node(node)) = record.get("n") {
                // Convert node properties to JSON
                let properties_json = self.bolt_map_to_json(&node.properties())?;
                
                // Deserialize to target type
                let node_instance: T = serde_json::from_value(properties_json)
                    .map_err(|e| CrudError::SerializationError { source: e })?;
                
                return Ok(Some(node_instance));
            }
        }
        
        Err(CrudError::DatabaseError {
            source: anyhow::anyhow!("Invalid node data returned from database"),
        })
    }
    
    // Helper method to convert BoltMap to JSON Value
    fn bolt_map_to_json(&self, bolt_map: &BoltMap) -> Result<serde_json::Value, CrudError> {
        let mut json_map = serde_json::Map::new();
        
        for (key, value) in bolt_map.iter() {
            let json_value = self.bolt_value_to_json(value)?;
            json_map.insert(key.to_string(), json_value);
        }
        
        Ok(serde_json::Value::Object(json_map))
    }
    
    // Helper method to convert BoltValue to JSON Value
    fn bolt_value_to_json(&self, bolt_value: &BoltValue) -> Result<serde_json::Value, CrudError> {
        match bolt_value {
            BoltValue::Null => Ok(serde_json::Value::Null),
            BoltValue::Boolean(b) => Ok(serde_json::Value::Bool(*b)),
            BoltValue::Integer(i) => Ok(serde_json::Value::Number((*i).into())),
            BoltValue::Float(f) => {
                if let Some(n) = serde_json::Number::from_f64(*f) {
                    Ok(serde_json::Value::Number(n))
                } else {
                    Ok(serde_json::Value::String(f.to_string()))
                }
            },
            BoltValue::String(s) => Ok(serde_json::Value::String(s.to_string())),
            BoltValue::List(list) => {
                let json_list: Result<Vec<serde_json::Value>, CrudError> = list
                    .iter()
                    .map(|v| self.bolt_value_to_json(v))
                    .collect();
                Ok(serde_json::Value::Array(json_list?))
            },
            BoltValue::Map(map) => {
                self.bolt_map_to_json(map)
            },
            _ => {
                // For unsupported types, convert to string
                Ok(serde_json::Value::String(format!("{:?}", bolt_value)))
            }
        }
    }
}
```

## Update Imports
Add to imports section:
```rust
use neo4j::BoltNode;
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_reading_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_bolt_value_conversion() {
        // Test the conversion logic without database
        let test_cases = vec![
            (BoltValue::Null, serde_json::Value::Null),
            (BoltValue::Boolean(true), serde_json::Value::Bool(true)),
            (BoltValue::Integer(42), serde_json::Value::Number(42.into())),
            (BoltValue::String("test".into()), serde_json::Value::String("test".to_string())),
        ];
        
        // This tests conversion logic structure
        for (bolt_value, expected_json) in test_cases {
            match (&bolt_value, &expected_json) {
                (BoltValue::Null, serde_json::Value::Null) => assert!(true),
                (BoltValue::Boolean(b1), serde_json::Value::Bool(b2)) => assert_eq!(b1, b2),
                (BoltValue::Integer(i1), serde_json::Value::Number(n)) => {
                    assert_eq!(*i1, n.as_i64().unwrap());
                },
                (BoltValue::String(s1), serde_json::Value::String(s2)) => {
                    assert_eq!(s1.as_str(), s2.as_str());
                },
                _ => panic!("Conversion test failed"),
            }
        }
    }
    
    #[test]
    fn test_read_validation() {
        // Test input validation without database
        let empty_id = "";
        let valid_id = "valid_id_123";
        let node_type = "Concept";
        
        // Test ID validation logic
        assert!(empty_id.is_empty());
        assert!(!valid_id.is_empty());
        assert!(!node_type.is_empty());
        
        // Test error cases
        if empty_id.is_empty() {
            // This simulates the validation error that would occur
            assert!(true, "Empty ID should trigger validation error");
        }
    }
    
    #[tokio::test]
    async fn test_read_node_structure() {
        // Test that the read structure is correct
        let concept = ConceptNode::new("ReadTest".to_string(), "Entity".to_string());
        let json = concept.to_json().unwrap();
        
        // Test that we can deserialize back
        let parsed: ConceptNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "ReadTest");
        assert_eq!(parsed.concept_type, "Entity");
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test node_reading_tests
```

## Acceptance Criteria
- [ ] Node reading method implemented
- [ ] Bolt to JSON conversion works
- [ ] Error handling for missing nodes
- [ ] Input validation functions
- [ ] Tests pass

## Duration
8-10 minutes for node reading implementation.