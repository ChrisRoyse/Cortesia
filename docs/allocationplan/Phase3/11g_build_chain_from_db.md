# Task 11g: Implement Build Chain From Database

**Time**: 9 minutes
**Dependencies**: 11f_get_inheritance_chain.md
**Stage**: Inheritance System

## Objective
Implement database query to build inheritance chains.

## Implementation
Replace TODO in `build_inheritance_chain_from_db`:

```rust
let session = self.connection_manager.get_session().await?;

// Query to get full inheritance chain
let query = r#"
    MATCH path = (child:Concept {id: $concept_id})-[r:INHERITS_FROM*]->(ancestor:Concept)
    WITH relationships(path) as rels, nodes(path) as concepts
    UNWIND range(0, length(rels)-1) as i
    WITH rels[i] as rel, concepts[i+1] as parent_concept, i
    RETURN parent_concept.id as parent_id,
           rel.relationship_id as relationship_id,
           rel.inheritance_type as inheritance_type,
           rel.inheritance_weight as weight,
           i as depth_from_child
    ORDER BY depth_from_child
"#;

let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
let result = session.run(query, Some(parameters)).await?;

let mut chain_links = Vec::new();
let mut max_depth = 0;

for record in result {
    let depth: i64 = record.get("depth_from_child")?;
    let link = InheritanceLink {
        parent_concept_id: record.get("parent_id")?,
        relationship_id: record.get("relationship_id")?,
        inheritance_type: InheritanceType::ClassInheritance, // TODO: Parse properly
        depth_from_child: depth as u32,
        weight: record.get("weight")?,
    };
    
    max_depth = max_depth.max(depth as u32);
    chain_links.push(link);
}
```

## Success Criteria
- Database query returns proper chain data
- InheritanceLinks are correctly built

## Next Task
11h_get_direct_children.md