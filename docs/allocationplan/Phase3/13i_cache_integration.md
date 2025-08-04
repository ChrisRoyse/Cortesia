# Task 13i: Integrate Cache with Inheritance System

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 13h_cache_metrics.md
**Stage**: Inheritance System

## Objective
Integrate cache manager with existing inheritance components.

## Implementation
Modify `src/inheritance/hierarchy_manager.rs` to use cache:

```rust
use crate::inheritance::cache::inheritance_cache_manager::InheritanceCacheManager;

// Add cache manager to the struct
pub struct InheritanceHierarchyManager {
    connection_manager: Arc<Neo4jConnectionManager>,
    cache_manager: Arc<InheritanceCacheManager>,
    // Remove the old cache fields
}

impl InheritanceHierarchyManager {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cache_config = crate::inheritance::cache::cache_types::CacheConfig::default();
        let cache_manager = Arc::new(InheritanceCacheManager::new(cache_config));
        
        Ok(Self {
            connection_manager,
            cache_manager,
        })
    }

    // Update get_inheritance_chain to use new cache
    pub async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, Box<dyn std::error::Error>> {
        // Check cache first
        if let Some(cached_chain) = self.cache_manager.get_cached_chain(concept_id).await {
            return Ok(cached_chain.chain);
        }
        
        // Build inheritance chain from database
        let chain = self.build_inheritance_chain_from_db(concept_id).await?;
        
        // Cache the result
        self.cache_manager.store_chain(concept_id.to_string(), chain.clone()).await;
        
        Ok(chain)
    }
}
```

Modify `src/inheritance/property_inheritance_engine.rs`:

```rust
// Add cache manager to the struct
pub struct PropertyInheritanceEngine {
    connection_manager: Arc<Neo4jConnectionManager>,
    cache_manager: Arc<InheritanceCacheManager>,
    config: InheritanceConfig,
}

impl PropertyInheritanceEngine {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
        config: InheritanceConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cache_config = crate::inheritance::cache::cache_types::CacheConfig::default();
        let cache_manager = Arc::new(InheritanceCacheManager::new(cache_config));
        
        Ok(Self {
            connection_manager,
            cache_manager,
            config,
        })
    }

    // Update resolve_properties to use new cache
    pub async fn resolve_properties(
        &self,
        concept_id: &str,
        include_inherited: bool,
    ) -> Result<ResolvedProperties, Box<dyn std::error::Error>> {
        let cache_key = format!("{}:{}", concept_id, include_inherited);
        
        // Check cache first
        if let Some(cached_properties) = self.cache_manager.get_cached_properties(&cache_key).await {
            return Ok(cached_properties.resolved_properties);
        }
        
        // Resolve properties as before...
        let resolved_properties = // ... existing logic
        
        // Cache the result
        self.cache_manager.store_properties(cache_key, resolved_properties.clone()).await;
        
        Ok(resolved_properties)
    }
}
```

## Success Criteria
- Cache manager is properly integrated
- All inheritance operations use centralized caching

## Next Task
13j_cache_mod_file.md