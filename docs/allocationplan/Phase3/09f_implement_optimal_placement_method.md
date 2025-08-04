# Task 09f: Implement Optimal Placement Method

**Estimated Time**: 10 minutes  
**Dependencies**: 09e_implement_cache_key_generator.md  
**Stage**: Neural Integration - Core Placement Logic

## Objective
Implement main method to determine optimal node placement.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
use chrono::Utc;

impl AllocationGuidedPlacement {
    pub async fn determine_optimal_placement(
        &self,
        content: &str,
        spike_pattern: &TTFSSpikePattern,
    ) -> Result<PlacementDecision, PlacementError> {
        let placement_start = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_placement_cache_key(content, spike_pattern);
        if let Some(cached_decision) = self.placement_cache.read().await.get(&cache_key) {
            return Ok(cached_decision.clone());
        }
        
        // Process through cortical columns for placement guidance
        let cortical_consensus = self.multi_column_processor
            .process_for_knowledge_graph(spike_pattern)
            .await
            .map_err(|e| PlacementError::CorticalError(e.to_string()))?;
        
        // Use allocation engine to determine placement
        let allocation_result = self.allocation_engine
            .determine_allocation(&cortical_consensus)
            .await
            .map_err(|e| PlacementError::AllocationError(e.to_string()))?;
        
        // Convert allocation result to placement decision
        let placement_decision = self.convert_allocation_to_placement(
            allocation_result,
            cortical_consensus,
            placement_start.elapsed(),
        ).await?;
        
        // Cache the result
        self.placement_cache.write().await.put(cache_key, placement_decision.clone());
        
        Ok(placement_decision)
    }
}
```

## Acceptance Criteria
- [ ] Main placement method compiles
- [ ] Cache lookup implemented
- [ ] Cortical processing integrated
- [ ] Allocation engine called
- [ ] Error handling present

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09g_implement_allocation_to_placement_converter.md**