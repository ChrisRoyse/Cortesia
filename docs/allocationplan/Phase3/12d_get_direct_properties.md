# Task 12d: Implement Get Direct Properties

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 12c_resolve_properties_method.md
**Stage**: Inheritance System

## Objective
Add method to retrieve direct properties from database.

## Implementation
Add to `src/inheritance/property_inheritance_engine.rs`:

```rust
impl PropertyInheritanceEngine {
    async fn get_direct_properties(
        &self,
        concept_id: &str,
    ) -> Result<Vec<PropertyNode>, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $concept_id})-[:HAS_PROPERTY]->(p:Property)
            RETURN p.name as property_name,
                   p.value as property_value,
                   p.is_inheritable as is_inheritable,
                   p.inheritance_priority as inheritance_priority
            ORDER BY p.inheritance_priority
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        let mut properties = Vec::new();
        for record in result {
            let property = PropertyNode {
                name: record.get("property_name")?,
                value: self.parse_property_value(record.get("property_value")?)?,
                is_inheritable: record.get("is_inheritable").unwrap_or(true),
                inheritance_priority: record.get("inheritance_priority").unwrap_or(0),
            };
            properties.push(property);
        }
        
        Ok(properties)
    }

    fn parse_property_value(&self, value: String) -> Result<PropertyValue, Box<dyn std::error::Error>> {
        // Simple text value for now
        Ok(PropertyValue::Text(value))
    }
}
```

## Success Criteria
- Database query returns property data
- PropertyNode objects are correctly built

## Next Task
12e_inheritance_chain_resolution.md