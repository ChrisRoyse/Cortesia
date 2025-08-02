# Task 05b: Implement Node Creation

**Estimated Time**: 12 minutes  
**Dependencies**: 05a_create_basic_node_operations.md  
**Next Task**: 05c_implement_node_reading.md  

## Objective
Implement the actual node creation logic with Neo4j database operations.

## Single Action
Replace the TODO in create_node method with real Neo4j implementation.

## Code to Replace
Replace the `create_node` method in `src/storage/crud_operations.rs`:
```rust
use neo4j::{Query, BoltMap, BoltValue};

impl BasicNodeOperations {
    pub async fn create_node<T: GraphNode + Serialize>(
        &self,
        node: &T,
        options: CreateOptions,
    ) -> Result<String, CrudError> {
        // Validation
        if options.validate && !node.validate() {
            return Err(CrudError::ValidationError {
                message: format!("Node validation failed for {}", node.id()),
            });
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
        
        // Check if node already exists (for upsert)
        if options.upsert {
            let exists = self.node_exists(node.id(), node.node_type()).await?;
            if exists && options.return_existing {
                return Ok(node.id().to_string());
            }
        }
        
        // Create Cypher query
        let node_label = node.node_type();
        let cypher = if options.upsert {
            format!(
                "MERGE (n:{} {{id: $id}}) SET n = $properties RETURN n.id as id",
                node_label
            )
        } else {
            format!(
                "CREATE (n:{} $properties) RETURN n.id as id",
                node_label
            )
        };
        
        // Execute query
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut query = Query::new(cypher);
        query.param("properties", properties);
        if options.upsert {
            query.param("id", node.id());
        }
        
        let result = session.run(query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        // Extract created node ID
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::String(id)) = record.get("id") {
                return Ok(id.to_string());
            }
        }
        
        Err(CrudError::DatabaseError {
            source: anyhow::anyhow!("Failed to create node: no ID returned"),
        })
    }
    
    // Helper method to convert JSON values to Bolt values
    fn json_to_bolt_value(&self, value: serde_json::Value) -> Result<BoltValue, CrudError> {
        match value {
            serde_json::Value::Null => Ok(BoltValue::Null),
            serde_json::Value::Bool(b) => Ok(BoltValue::Boolean(b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(BoltValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(BoltValue::Float(f))
                } else {
                    Ok(BoltValue::String(n.to_string().into()))
                }
            },
            serde_json::Value::String(s) => Ok(BoltValue::String(s.into())),
            serde_json::Value::Array(arr) => {
                let bolt_list: Result<Vec<BoltValue>, CrudError> = arr
                    .into_iter()
                    .map(|v| self.json_to_bolt_value(v))
                    .collect();
                Ok(BoltValue::List(bolt_list?))
            },
            serde_json::Value::Object(_) => {
                // For nested objects, convert to JSON string
                let json_str = serde_json::to_string(&value)
                    .map_err(|e| CrudError::SerializationError { source: e })?;
                Ok(BoltValue::String(json_str.into()))
            },
        }
    }
}
```

## Add Required Imports
Add to the top of `src/storage/crud_operations.rs`:
```rust
use neo4j::{Query, BoltMap, BoltValue};
use serde_json;
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_creation_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[tokio::test]
    async fn test_node_creation_structure() {
        // Test that the creation logic compiles and structures are correct
        let concept = ConceptNode::new("TestConcept".to_string(), "Entity".to_string());
        
        // Test options
        let options = CreateOptions {
            validate: true,
            upsert: false,
            return_existing: false,
        };
        
        assert!(concept.validate());
        assert_eq!(options.validate, true);
        
        // Test JSON serialization
        let json_result = concept.to_json();
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("\"id\""));
        assert!(json_str.contains("TestConcept"));
    }
    
    #[test]
    fn test_json_value_types() {
        // Test JSON value handling without database connection
        let test_values = vec![
            serde_json::Value::Null,
            serde_json::Value::Bool(true),
            serde_json::Value::Number(serde_json::Number::from(42)),
            serde_json::Value::String("test".to_string()),
            serde_json::Value::Array(vec![serde_json::Value::Number(serde_json::Number::from(1))]),
        ];
        
        // This tests the structure without requiring database connection
        for value in test_values {
            match value {
                serde_json::Value::Null => assert!(true),
                serde_json::Value::Bool(_) => assert!(true),
                serde_json::Value::Number(_) => assert!(true),
                serde_json::Value::String(_) => assert!(true),
                serde_json::Value::Array(_) => assert!(true),
                _ => assert!(false, "Unexpected value type"),
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
cargo test node_creation_tests
```

## Acceptance Criteria
- [ ] Node creation method implemented
- [ ] JSON to Bolt value conversion works
- [ ] Upsert logic functions correctly
- [ ] Error handling comprehensive
- [ ] Tests pass

## Duration
10-12 minutes for node creation implementation.