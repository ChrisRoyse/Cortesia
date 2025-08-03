# Task 12e: Implement Inheritance Chain Resolution

**Time**: 7 minutes
**Dependencies**: 12d_get_direct_properties.md
**Stage**: Inheritance System

## Objective
Add logic to resolve inheritance chains for property resolution.

## Implementation
Add to `src/inheritance/property_inheritance_engine.rs`:

```rust
impl PropertyInheritanceEngine {
    async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (child:Concept {id: $concept_id})-[:INHERITS_FROM*]->(ancestor:Concept)
            WITH path, length(path) as depth
            ORDER BY depth
            RETURN 
                [node in nodes(path) | node.id] as node_ids,
                depth
            LIMIT 1
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        let mut chain = InheritanceChain {
            child_concept_id: concept_id.to_string(),
            chain: Vec::new(),
            total_depth: 0,
            is_valid: true,
            has_cycles: false,
        };
        
        for record in result {
            let node_ids: Vec<String> = record.get("node_ids")?;
            let depth: i32 = record.get("depth")?;
            
            // Build inheritance links from the path
            for (i, ancestor_id) in node_ids.iter().enumerate() {
                if i == 0 { continue; } // Skip the child itself
                
                chain.chain.push(crate::inheritance::hierarchy_types::InheritanceLink {
                    parent_concept_id: ancestor_id.clone(),
                    relationship_id: format!("inh_{}", i),
                    inheritance_type: crate::inheritance::hierarchy_types::InheritanceType::ClassInheritance,
                    depth_from_child: i as u32,
                    weight: 1.0,
                });
            }
            
            chain.total_depth = depth as u32;
        }
        
        Ok(chain)
    }
}
```

## Success Criteria
- Returns proper inheritance chain structure
- Handles paths correctly from database

## Next Task
12f_resolve_inherited_properties.md