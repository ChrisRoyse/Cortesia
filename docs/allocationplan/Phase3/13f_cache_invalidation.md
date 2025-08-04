# Task 13f: Implement Cache Invalidation

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 13e_cache_property_operations.md
**Stage**: Inheritance System

## Objective
Add cache invalidation logic for concept and property changes.

## Implementation
Add to `src/inheritance/cache/inheritance_cache_manager.rs`:

```rust
impl InheritanceCacheManager {
    pub async fn invalidate_concept(&self, concept_id: &str) {
        let start_time = std::time::Instant::now();
        
        // Invalidate the concept itself
        {
            let mut chain_cache = self.chain_cache.write().await;
            chain_cache.remove(concept_id);
        }
        
        // Invalidate property resolutions for this concept
        {
            let mut property_cache = self.property_cache.write().await;
            let keys_to_remove: Vec<_> = property_cache.keys()
                .filter(|key| key.starts_with(&format!("{}:", concept_id)))
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                property_cache.remove(&key);
            }
        }
        
        // Invalidate all descendants (they might inherit from this concept)
        self.invalidate_descendants(concept_id).await;
        
        // Update statistics
        {
            let mut stats = self.cache_stats.write().await;
            stats.invalidation_count += 1;
        }
    }

    async fn invalidate_descendants(&self, concept_id: &str) {
        // Find all concepts that have this concept in their inheritance chain
        let descendant_concepts = self.find_dependent_concepts(concept_id).await;
        
        for descendant in descendant_concepts {
            {
                let mut chain_cache = self.chain_cache.write().await;
                chain_cache.remove(&descendant);
            }
            
            {
                let mut property_cache = self.property_cache.write().await;
                let keys_to_remove: Vec<_> = property_cache.keys()
                    .filter(|key| key.starts_with(&format!("{}:", descendant)))
                    .cloned()
                    .collect();
                
                for key in keys_to_remove {
                    property_cache.remove(&key);
                }
            }
        }
    }

    async fn find_dependent_concepts(&self, concept_id: &str) -> Vec<String> {
        // Find all concepts that have this concept in their inheritance chain
        let chain_cache = self.chain_cache.read().await;
        
        chain_cache
            .iter()
            .filter_map(|(child_id, cached_chain)| {
                let has_dependency = cached_chain.chain.chain.iter()
                    .any(|link| link.parent_concept_id == concept_id);
                
                if has_dependency {
                    Some(child_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub async fn invalidate_property(&self, concept_id: &str, property_name: &str) {
        // Remove property-specific resolutions
        {
            let mut property_cache = self.property_cache.write().await;
            let property_key = format!("{}:{}", concept_id, property_name);
            property_cache.remove(&property_key);
        }
        
        // Invalidate descendants that might inherit this property
        let descendants = self.find_dependent_concepts(concept_id).await;
        {
            let mut property_cache = self.property_cache.write().await;
            for descendant in descendants {
                let descendant_key = format!("{}:{}", descendant, property_name);
                property_cache.remove(&descendant_key);
            }
        }
        
        // Update statistics
        {
            let mut stats = self.cache_stats.write().await;
            stats.invalidation_count += 1;
        }
    }
}
```

## Success Criteria
- Concept invalidation removes all related entries
- Property invalidation handles descendants correctly

## Next Task
13g_cache_warming.md