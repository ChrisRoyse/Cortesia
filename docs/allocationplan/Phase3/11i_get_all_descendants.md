# Task 11i: Implement Get All Descendants

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 11h_get_direct_children.md
**Stage**: Inheritance System

## Objective
Add method to get all descendants with optional depth limit.

## Implementation
Add to `src/inheritance/hierarchy_manager.rs`:

```rust
impl InheritanceHierarchyManager {
    pub async fn get_all_descendants(
        &self,
        ancestor_concept_id: &str,
        max_depth: Option<u32>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let depth_filter = match max_depth {
            Some(depth) => format!("WHERE length(path) <= {}", depth),
            None => String::new(),
        };
        
        let query = format!(
            r#"
            MATCH path = (ancestor:Concept {{id: $ancestor_id}})<-[r:INHERITS_FROM*]-(descendant:Concept)
            {}
            RETURN DISTINCT descendant.id as descendant_id,
                   length(path) as inheritance_depth
            ORDER BY inheritance_depth, descendant_id
            "#,
            depth_filter
        );
        
        let parameters = hashmap!["ancestor_id".to_string() => ancestor_concept_id.into()];
        let result = session.run(&query, Some(parameters)).await?;
        
        let mut descendants = Vec::new();
        for record in result {
            descendants.push(record.get("descendant_id")?);
        }
        
        Ok(descendants)
    }
}
```

## Success Criteria
- Returns all descendants at any depth
- Respects max_depth parameter when provided

## Next Task
11j_cycle_detection.md