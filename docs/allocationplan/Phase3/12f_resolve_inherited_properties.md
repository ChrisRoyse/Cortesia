# Task 12f: Implement Resolve Inherited Properties

**Time**: 8 minutes
**Dependencies**: 12e_inheritance_chain_resolution.md
**Stage**: Inheritance System

## Objective
Add logic to resolve properties from inheritance chain.

## Implementation
Add to `src/inheritance/property_inheritance_engine.rs`:

```rust
impl PropertyInheritanceEngine {
    async fn resolve_inherited_properties(
        &self,
        inheritance_chain: &InheritanceChain,
        direct_properties: &[PropertyNode],
    ) -> Result<Vec<InheritedProperty>, Box<dyn std::error::Error>> {
        let mut inherited_properties = Vec::new();
        let direct_property_names: std::collections::HashSet<String> = direct_properties
            .iter()
            .map(|p| p.name.clone())
            .collect();
        
        // Process inheritance chain in order (closest ancestor first)
        for link in &inheritance_chain.chain {
            let ancestor_properties = self.get_direct_properties(&link.parent_concept_id).await?;
            
            for property in ancestor_properties {
                // Skip if property is already defined directly or by closer ancestor
                if direct_property_names.contains(&property.name) ||
                   inherited_properties.iter().any(|ip: &InheritedProperty| ip.property.name == property.name) {
                    continue;
                }
                
                // Check if property is inheritable
                if !property.is_inheritable {
                    continue;
                }
                
                inherited_properties.push(InheritedProperty {
                    property,
                    source_concept_id: link.parent_concept_id.clone(),
                    inheritance_depth: link.depth_from_child as i32,
                    inheritance_strength: link.weight,
                    has_exception: false,
                    exception_reason: None,
                });
            }
        }
        
        // Sort by inheritance priority and depth
        inherited_properties.sort_by(|a, b| {
            a.property.inheritance_priority
                .cmp(&b.property.inheritance_priority)
                .then(a.inheritance_depth.cmp(&b.inheritance_depth))
        });
        
        Ok(inherited_properties)
    }
}
```

## Success Criteria
- Resolves properties from inheritance chain
- Properly handles property conflicts

## Next Task
12g_merge_properties.md