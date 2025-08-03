# Task 12c: Implement Resolve Properties Method

**Time**: 6 minutes
**Dependencies**: 12b_inheritance_engine_struct.md
**Stage**: Inheritance System

## Objective
Add main method to resolve properties with inheritance.

## Implementation
Add to `src/inheritance/property_inheritance_engine.rs`:

```rust
impl PropertyInheritanceEngine {
    pub async fn resolve_properties(
        &self,
        concept_id: &str,
        include_inherited: bool,
    ) -> Result<ResolvedProperties, Box<dyn std::error::Error>> {
        let resolution_start = Instant::now();
        
        // Check resolution cache first
        let cache_key = format!("{}:{}", concept_id, include_inherited);
        if let Some(cached_properties) = self.resolution_cache.read().await.get(&cache_key) {
            return Ok(cached_properties.clone());
        }
        
        // Get direct properties
        let direct_properties = self.get_direct_properties(concept_id).await?;
        
        let resolved_properties = if include_inherited {
            // TODO: Resolve inherited properties (next task)
            ResolvedProperties {
                concept_id: concept_id.to_string(),
                direct_properties,
                inherited_properties: Vec::new(),
                resolution_time: chrono::Utc::now(),
                total_property_count: 0,
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
        
        // Cache the resolved properties
        self.resolution_cache.write().await.put(cache_key, resolved_properties.clone());
        
        Ok(resolved_properties)
    }
}
```

## Success Criteria
- Method compiles and returns ResolvedProperties
- Caching mechanism works

## Next Task
12d_get_direct_properties.md