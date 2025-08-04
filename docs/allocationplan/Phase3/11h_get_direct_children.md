# Task 11h: Implement Get Direct Children

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 11g_build_chain_from_db.md
**Stage**: Inheritance System

## Objective
Add method to get direct children of a concept.

## Implementation
Add to `src/inheritance/hierarchy_manager.rs`:

```rust
impl InheritanceHierarchyManager {
    pub async fn get_direct_children(
        &self,
        parent_concept_id: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (parent:Concept {id: $parent_id})<-[r:INHERITS_FROM]-(child:Concept)
            WHERE r.is_active = true
            RETURN child.id as child_id
            ORDER BY r.created_at
        "#;
        
        let parameters = hashmap!["parent_id".to_string() => parent_concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        let mut children = Vec::new();
        for record in result {
            children.push(record.get("child_id")?);
        }
        
        Ok(children)
    }
}
```

## Success Criteria
- Method returns list of child concept IDs
- Query filters active relationships only

## Next Task
11i_get_all_descendants.md