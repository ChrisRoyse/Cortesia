# Task 05a: Create Basic Node Operations

**Estimated Time**: 10 minutes  
**Dependencies**: 04f_create_relationship_trait.md  
**Next Task**: 05b_implement_node_creation.md  

## Objective
Create the basic node operations interface and error types for CRUD operations.

## Single Action
Create the foundation structs and traits for node CRUD operations.

## File to Create
File: `src/storage/crud_operations.rs`
```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::storage::{Neo4jConnectionManager, GraphNode};

#[derive(Debug, Error)]
pub enum CrudError {
    #[error("Node not found: {id}")]
    NotFound { id: String },
    
    #[error("Validation failed: {message}")]
    ValidationError { message: String },
    
    #[error("Database error: {source}")]
    DatabaseError { source: anyhow::Error },
    
    #[error("Constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },
    
    #[error("Serialization error: {source}")]
    SerializationError { source: serde_json::Error },
    
    #[error("Connection error: {message}")]
    ConnectionError { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCriteria {
    pub node_type: Option<String>,
    pub properties: HashMap<String, String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub order_by: Option<String>,
    pub ascending: bool,
}

impl Default for FilterCriteria {
    fn default() -> Self {
        Self {
            node_type: None,
            properties: HashMap::new(),
            limit: Some(100),
            offset: None,
            order_by: None,
            ascending: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreateOptions {
    pub validate: bool,
    pub upsert: bool,
    pub return_existing: bool,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            validate: true,
            upsert: false,
            return_existing: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UpdateOptions {
    pub partial: bool,
    pub validate: bool,
    pub create_if_missing: bool,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            partial: true,
            validate: true,
            create_if_missing: false,
        }
    }
}

pub struct BasicNodeOperations {
    connection_manager: Arc<Neo4jConnectionManager>,
}

impl BasicNodeOperations {
    pub fn new(connection_manager: Arc<Neo4jConnectionManager>) -> Self {
        Self {
            connection_manager,
        }
    }
    
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
        
        // TODO: Implementation in next task
        todo!("Implementation in task 05b")
    }
    
    pub async fn read_node<T>(&self, id: &str, node_type: &str) -> Result<Option<T>, CrudError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // TODO: Implementation in next task
        todo!("Implementation in task 05c")
    }
    
    pub async fn update_node<T: GraphNode + Serialize>(
        &self,
        id: &str,
        node: &T,
        options: UpdateOptions,
    ) -> Result<(), CrudError> {
        // TODO: Implementation in next task
        todo!("Implementation in task 05d")
    }
    
    pub async fn delete_node(&self, id: &str, node_type: &str) -> Result<(), CrudError> {
        // TODO: Implementation in next task
        todo!("Implementation in task 05e")
    }
    
    pub async fn node_exists(&self, id: &str, node_type: &str) -> Result<bool, CrudError> {
        // TODO: Implementation in task 05f
        todo!("Implementation in task 05f")
    }
    
    pub async fn list_nodes<T>(
        &self,
        filters: &FilterCriteria,
    ) -> Result<Vec<T>, CrudError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // TODO: Implementation in task 05g
        todo!("Implementation in task 05g")
    }
}

#[cfg(test)]
mod crud_operations_tests {
    use super::*;
    
    #[test]
    fn test_filter_criteria_default() {
        let filter = FilterCriteria::default();
        
        assert_eq!(filter.limit, Some(100));
        assert!(filter.ascending);
        assert!(filter.properties.is_empty());
        assert!(filter.node_type.is_none());
    }
    
    #[test]
    fn test_create_options() {
        let options = CreateOptions::default();
        
        assert!(options.validate);
        assert!(!options.upsert);
        assert!(!options.return_existing);
        
        let custom_options = CreateOptions {
            validate: false,
            upsert: true,
            return_existing: true,
        };
        
        assert!(!custom_options.validate);
        assert!(custom_options.upsert);
        assert!(custom_options.return_existing);
    }
    
    #[test]
    fn test_update_options() {
        let options = UpdateOptions::default();
        
        assert!(options.partial);
        assert!(options.validate);
        assert!(!options.create_if_missing);
    }
}
```

## Module Update
Add to `src/storage/mod.rs`:
```rust
pub mod crud_operations;

pub use crud_operations::*;
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully with TODO warnings

# Run tests
cargo test crud_operations_tests
```

## Acceptance Criteria
- [ ] CrudError enum defines all error types
- [ ] FilterCriteria struct for query filtering
- [ ] CreateOptions and UpdateOptions for operation control
- [ ] BasicNodeOperations struct with method signatures
- [ ] Tests pass

## Duration
8-10 minutes for basic operations foundation.