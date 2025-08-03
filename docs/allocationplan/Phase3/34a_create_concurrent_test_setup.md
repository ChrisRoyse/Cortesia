# Task 34a: Create Concurrent Test Setup

**Estimated Time**: 4 minutes  
**Dependencies**: 33p  
**Stage**: Concurrency Testing  

## Objective
Create basic test setup infrastructure for concurrent access testing.

## Implementation Steps

1. Create `tests/concurrency/mod.rs`:
```rust
use std::sync::Arc;
use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;

pub async fn setup_concurrency_test_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_testing());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_testing());
    let memory_pool = Arc::new(MemoryPool::new_for_testing());
    
    Arc::new(
        BrainEnhancedGraphCore::new_with_test_config(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create test brain graph")
    )
}

pub fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}
```

## Acceptance Criteria
- [ ] Test module created
- [ ] Setup function implemented
- [ ] Helper functions available

## Success Metrics
- Setup completes in under 2 seconds
- Test graph is properly configured

## Next Task
34b_test_basic_thread_safety.md