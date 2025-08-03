# Task 12i: Create Property Exception Handler

**Time**: 6 minutes
**Dependencies**: 12h_property_exceptions_types.md
**Stage**: Inheritance System

## Objective
Create handler for property exceptions with database operations.

## Implementation
Create `src/inheritance/property_exceptions.rs`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::property_types::*;

pub struct PropertyExceptionHandler {
    connection_manager: Arc<Neo4jConnectionManager>,
    exception_cache: Arc<RwLock<HashMap<String, Vec<ExceptionNode>>>>,
}

impl PropertyExceptionHandler {
    pub fn new(connection_manager: Arc<Neo4jConnectionManager>) -> Self {
        Self {
            connection_manager,
            exception_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_property_exception(
        &self,
        concept_id: &str,
        property_name: &str,
        original_value: PropertyValue,
        exception_value: PropertyValue,
        reason: &str,
        confidence: f32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let exception = ExceptionNodeBuilder::new()
            .property_name(property_name)
            .original_value(original_value)
            .exception_value(exception_value)
            .exception_reason(reason)
            .confidence(confidence)
            .build()?;
        
        let exception_id = self.create_exception_node(&exception).await?;
        
        // Create relationship to concept
        self.create_exception_relationship(concept_id, &exception_id).await?;
        
        // Invalidate related caches
        self.invalidate_exception_cache(concept_id, property_name).await;
        
        Ok(exception_id)
    }
}
```

## Success Criteria
- Exception handler structure compiles
- Constructor and main methods work

## Next Task
12j_exception_database_operations.md