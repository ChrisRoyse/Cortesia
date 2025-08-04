# Task 11j: Implement Cycle Detection

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 11i_get_all_descendants.md
**Stage**: Inheritance System

## Objective
Complete the cycle detection logic for inheritance relationships.

## Implementation
Replace the placeholder `would_create_cycle` method:

```rust
impl InheritanceHierarchyManager {
    async fn would_create_cycle(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Check if parent_concept_id is already a descendant of child_concept_id
        let descendants = self.get_all_descendants(child_concept_id, Some(20)).await?;
        Ok(descendants.contains(&parent_concept_id.to_string()))
    }

    pub async fn detect_cycles_in_hierarchy(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        // Use graph algorithms to detect cycles
        let query = r#"
            MATCH (c:Concept)-[r:INHERITS_FROM*]->(c)
            WHERE r.is_active = true
            RETURN c.id as concept_with_cycle
        "#;
        
        let result = session.run(query, None).await?;
        
        let mut cycles = Vec::new();
        for record in result {
            cycles.push(record.get("concept_with_cycle")?);
        }
        
        Ok(cycles)
    }
}
```

## Success Criteria
- Cycle detection prevents invalid relationships
- Can identify existing cycles in hierarchy

## Next Task
12a_property_inheritance_types.md