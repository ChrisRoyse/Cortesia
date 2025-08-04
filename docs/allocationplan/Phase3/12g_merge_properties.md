# Task 12g: Implement Merge Properties

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 12f_resolve_inherited_properties.md
**Stage**: Inheritance System

## Objective
Complete the property resolution by merging direct and inherited properties.

## Implementation
Replace TODO in `resolve_properties` method with inheritance logic:

```rust
let resolved_properties = if include_inherited {
    // Get inheritance chain
    let inheritance_chain = self.get_inheritance_chain(concept_id).await?;
    
    // Resolve inherited properties
    let inherited_properties = self.resolve_inherited_properties(
        &inheritance_chain,
        &direct_properties,
    ).await?;
    
    // Merge direct and inherited properties
    let total_count = direct_properties.len() + inherited_properties.len();
    
    ResolvedProperties {
        concept_id: concept_id.to_string(),
        direct_properties,
        inherited_properties,
        resolution_time: chrono::Utc::now(),
        total_property_count: total_count,
    }
} else {
    ResolvedProperties {
        concept_id: concept_id.to_string(),
        direct_properties: direct_properties.clone(),
        inherited_properties: Vec::new(),
        resolution_time: chrono::Utc::now(),
        total_property_count: direct_properties.len(),
    }
};
```

## Success Criteria
- Properly merges direct and inherited properties
- Calculates correct property counts

## Next Task
12h_property_exceptions_types.md