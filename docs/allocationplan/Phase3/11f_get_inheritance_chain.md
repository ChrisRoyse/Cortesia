# Task 11f: Implement Get Inheritance Chain

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 11e_inheritance_database_creation.md
**Stage**: Inheritance System

## Objective
Add method to retrieve inheritance chains with caching.

## Implementation
Add to `src/inheritance/hierarchy_manager.rs`:

```rust
impl InheritanceHierarchyManager {
    pub async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, Box<dyn std::error::Error>> {
        // Check cache first
        if let Some(cached_chain) = self.hierarchy_cache.read().await.get(concept_id) {
            return Ok(cached_chain.clone());
        }
        
        // Build inheritance chain from database
        let chain = self.build_inheritance_chain_from_db(concept_id).await?;
        
        // Cache the result
        self.hierarchy_cache.write().await.put(concept_id.to_string(), chain.clone());
        
        Ok(chain)
    }

    async fn build_inheritance_chain_from_db(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, Box<dyn std::error::Error>> {
        // TODO: Implement database query (next task)
        Ok(InheritanceChain {
            child_concept_id: concept_id.to_string(),
            chain: Vec::new(),
            total_depth: 0,
            is_valid: true,
            has_cycles: false,
        })
    }
}
```

## Success Criteria
- Method compiles with basic caching
- Returns valid InheritanceChain structure

## Next Task
11g_build_chain_from_db.md