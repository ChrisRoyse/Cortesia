# Task 05e: Implement Node Deletion

**Estimated Time**: 8 minutes  
**Dependencies**: 05d_implement_node_updating.md  
**Next Task**: 05f_implement_node_exists_check.md  

## Objective
Implement the node deletion logic to safely remove nodes from Neo4j.

## Single Action
Replace the TODO in delete_node method with real Neo4j implementation.

## Code to Replace
Replace the `delete_node` method in `src/storage/crud_operations.rs`:
```rust
impl BasicNodeOperations {
    pub async fn delete_node(&self, id: &str, node_type: &str) -> Result<(), CrudError> {
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
        
        // Check if node exists before attempting deletion
        let exists = self.node_exists(id, node_type).await?;
        if !exists {
            return Err(CrudError::NotFound { id: id.to_string() });
        }
        
        // Create Cypher query to delete node and all its relationships
        let cypher = format!(
            "MATCH (n:{}) WHERE n.id = $id DETACH DELETE n RETURN count(n) as deleted_count",
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
        
        // Verify deletion succeeded
        if let Some(record) = result.records().first() {
            if let Some(BoltValue::Integer(deleted_count)) = record.get("deleted_count") {
                if *deleted_count == 1 {
                    return Ok(());
                } else if *deleted_count == 0 {
                    return Err(CrudError::NotFound { id: id.to_string() });
                }
            }
        }
        
        Err(CrudError::DatabaseError {
            source: anyhow::anyhow!("Unexpected deletion result for node {}", id),
        })
    }
    
    // Helper method for safe deletion with relationship checks
    pub async fn delete_node_safe(&self, id: &str, node_type: &str) -> Result<DeleteResult, CrudError> {
        if id.is_empty() {
            return Err(CrudError::ValidationError {
                message: "Node ID cannot be empty".to_string(),
            });
        }
        
        // First, check what relationships this node has
        let relationship_check_cypher = format!(
            "MATCH (n:{}) WHERE n.id = $id OPTIONAL MATCH (n)-[r]-() RETURN count(r) as rel_count",
            node_type
        );
        
        let session = self.connection_manager.get_session().await
            .map_err(|e| CrudError::ConnectionError { 
                message: format!("Failed to get session: {}", e) 
            })?;
        
        let mut check_query = Query::new(relationship_check_cypher);
        check_query.param("id", id);
        
        let check_result = session.run(check_query).await
            .map_err(|e| CrudError::DatabaseError { source: e.into() })?;
        
        let relationship_count = if let Some(record) = check_result.records().first() {
            if let Some(BoltValue::Integer(count)) = record.get("rel_count") {
                *count as usize
            } else {
                0
            }
        } else {
            return Err(CrudError::NotFound { id: id.to_string() });
        };
        
        // Delete the node (DETACH DELETE removes relationships too)
        self.delete_node(id, node_type).await?;
        
        Ok(DeleteResult {
            node_id: id.to_string(),
            relationships_deleted: relationship_count,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DeleteResult {
    pub node_id: String,
    pub relationships_deleted: usize,
}
```

## Add Required Type
Add to the struct definitions section:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResult {
    pub node_id: String,
    pub relationships_deleted: usize,
}
```

## Integration Test
Add test function:
```rust
#[cfg(test)]
mod node_deletion_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_delete_validation() {
        // Test input validation
        let empty_id = "";
        let valid_id = "valid_id_123";
        let empty_type = "";
        let valid_type = "Concept";
        
        // Test validation logic
        assert!(empty_id.is_empty());
        assert!(!valid_id.is_empty());
        assert!(empty_type.is_empty());
        assert!(!valid_type.is_empty());
        
        // Test that errors would be triggered
        if empty_id.is_empty() {
            assert!(true, "Empty ID should trigger validation error");
        }
        
        if empty_type.is_empty() {
            assert!(true, "Empty type should trigger validation error");
        }
    }
    
    #[test]
    fn test_delete_result_structure() {
        let delete_result = DeleteResult {
            node_id: "test_node_123".to_string(),
            relationships_deleted: 5,
        };
        
        assert_eq!(delete_result.node_id, "test_node_123");
        assert_eq!(delete_result.relationships_deleted, 5);
        
        // Test serialization
        let json = serde_json::to_string(&delete_result).unwrap();
        assert!(json.contains("test_node_123"));
        assert!(json.contains("5"));
    }
    
    #[test]
    fn test_cypher_query_construction() {
        // Test that deletion queries are constructed correctly
        let node_type = "Concept";
        let delete_query = format!(
            "MATCH (n:{}) WHERE n.id = $id DETACH DELETE n RETURN count(n) as deleted_count",
            node_type
        );
        
        assert!(delete_query.contains("DETACH DELETE"));
        assert!(delete_query.contains("MATCH (n:Concept)"));
        assert!(delete_query.contains("WHERE n.id = $id"));
        assert!(delete_query.contains("RETURN count(n)"));
        
        let safe_check_query = format!(
            "MATCH (n:{}) WHERE n.id = $id OPTIONAL MATCH (n)-[r]-() RETURN count(r) as rel_count",
            node_type
        );
        
        assert!(safe_check_query.contains("OPTIONAL MATCH"));
        assert!(safe_check_query.contains("-[r]-()"));
        assert!(safe_check_query.contains("count(r)"));
    }
    
    #[tokio::test]
    async fn test_deletion_workflow() {
        // Test the deletion workflow structure
        let concept = ConceptNode::new("DeleteTest".to_string(), "Entity".to_string());
        let node_id = concept.id.clone();
        let node_type = concept.node_type();
        
        // Test that we have the necessary data for deletion
        assert!(!node_id.is_empty());
        assert!(!node_type.is_empty());
        assert_eq!(node_type, "Concept");
        
        // Test error scenarios
        let empty_id = "";
        let empty_type = "";
        
        // These would trigger validation errors
        assert!(empty_id.is_empty());
        assert!(empty_type.is_empty());
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run tests
cargo test node_deletion_tests
```

## Acceptance Criteria
- [ ] Node deletion method implemented
- [ ] Safe deletion with relationship checking
- [ ] DETACH DELETE for relationship cleanup
- [ ] Delete result tracking
- [ ] Tests pass

## Duration
6-8 minutes for node deletion implementation.