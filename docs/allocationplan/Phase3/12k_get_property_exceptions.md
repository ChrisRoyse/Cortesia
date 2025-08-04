# Task 12k: Implement Get Property Exceptions

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 12j_exception_database_operations.md
**Stage**: Inheritance System

## Objective
Add method to retrieve property exceptions with caching.

## Implementation
Add to `src/inheritance/property_exceptions.rs`:

```rust
impl PropertyExceptionHandler {
    pub async fn get_property_exceptions(
        &self,
        concept_id: &str,
        property_name: &str,
    ) -> Result<Vec<ExceptionNode>, Box<dyn std::error::Error>> {
        let cache_key = format!("{}:{}", concept_id, property_name);
        
        // Check cache first
        if let Some(cached_exceptions) = self.exception_cache.read().await.get(&cache_key) {
            return Ok(cached_exceptions.clone());
        }
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $concept_id})-[:HAS_EXCEPTION]->(e:Exception)
            WHERE e.property_name = $property_name
            RETURN e.id as exception_id,
                   e.property_name as property_name,
                   e.original_value as original_value,
                   e.exception_value as exception_value,
                   e.exception_reason as exception_reason,
                   e.confidence as confidence,
                   e.precedence as precedence,
                   e.created_at as created_at
            ORDER BY e.precedence DESC, e.confidence DESC
        "#;
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
            "property_name".to_string() => property_name.into()
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut exceptions = Vec::new();
        for record in result {
            let exception = ExceptionNode {
                id: record.get("exception_id")?,
                property_name: record.get("property_name")?,
                original_value: PropertyValue::Text(record.get("original_value")?),
                exception_value: PropertyValue::Text(record.get("exception_value")?),
                exception_reason: record.get("exception_reason")?,
                confidence: record.get("confidence")?,
                precedence: record.get("precedence").unwrap_or(0),
                created_at: record.get("created_at")?,
            };
            exceptions.push(exception);
        }
        
        // Cache the exceptions
        self.exception_cache.write().await.insert(cache_key, exceptions.clone());
        
        Ok(exceptions)
    }
}
```

## Success Criteria
- Retrieves exceptions from database
- Caching mechanism works properly

## Next Task
12l_integration_with_inheritance.md