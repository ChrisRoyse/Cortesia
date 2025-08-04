# Task 08h: Implement Pathway Activation Method

**Estimated Time**: 6 minutes  
**Dependencies**: 08g_implement_pathway_retrieval_methods.md  
**Stage**: Neural Integration - Activation Logic

## Objective
Implement pathway activation with reinforcement and decay.

## Implementation

Add to `src/neural_pathways/pathway_storage.rs`:
```rust
use std::time::Instant;

impl PathwayStorageService {
    pub async fn activate_pathway(
        &self,
        pathway_id: &str,
    ) -> Result<PathwayActivationResult, PathwayStorageError> {
        let activation_start = Instant::now();
        
        let mut pathways = self.pathways.write().await;
        let pathway = pathways
            .get_mut(pathway_id)
            .ok_or_else(|| PathwayStorageError::PathwayNotFound(pathway_id.to_string()))?;
        
        // Apply reinforcement
        let reinforcement = pathway.strength * pathway.reinforcement_factor;
        pathway.strength = (pathway.strength + reinforcement).min(10.0);
        
        // Update usage statistics
        pathway.usage_count += 1;
        pathway.last_activated = Utc::now();
        pathway.modified_at = Utc::now();
        
        let traversal_time = activation_start.elapsed();
        let activation_strength = pathway.strength;
        
        Ok(PathwayActivationResult {
            pathway_id: pathway_id.to_string(),
            activation_strength,
            traversal_time,
            reinforcement_applied: reinforcement,
        })
    }
    
    pub async fn apply_decay(&self, pathway_id: &str) -> Result<(), PathwayStorageError> {
        let mut pathways = self.pathways.write().await;
        let pathway = pathways
            .get_mut(pathway_id)
            .ok_or_else(|| PathwayStorageError::PathwayNotFound(pathway_id.to_string()))?;
        
        // Apply decay
        pathway.strength = (pathway.strength * (1.0 - pathway.decay_rate)).max(0.01);
        pathway.modified_at = Utc::now();
        
        Ok(())
    }
}
```

## Acceptance Criteria
- [ ] Activation method compiles
- [ ] Reinforcement logic implemented
- [ ] Usage statistics updated
- [ ] Decay mechanism working
- [ ] Proper bounds checking

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **08i_implement_pathway_stats_method.md**