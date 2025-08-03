# Task 33a: Create Integrity Test Setup

**Estimated Time**: 4 minutes  
**Dependencies**: 32p  
**Stage**: Data Integrity Testing  

## Objective
Create basic test setup infrastructure for data integrity validation.

## Implementation Steps

1. Create `tests/integrity/mod.rs`:
```rust
use std::sync::Arc;
use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::types::*;

pub async fn setup_integrity_test_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_testing());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_testing());
    let memory_pool = Arc::new(MemoryPool::new_for_testing());
    
    Arc::new(
        BrainEnhancedGraphCore::new_with_integrity_checking(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create integrity test graph")
    )
}

pub fn create_test_concept_request(concept_id: &str) -> MemoryAllocationRequest {
    MemoryAllocationRequest {
        concept_id: concept_id.to_string(),
        concept_type: ConceptType::Semantic,
        content: format!("Test content for {}", concept_id),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "integrity_test_user".to_string(),
        request_id: format!("integrity_req_{}", concept_id),
        version_info: None,
    }
}

pub fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}
```

## Acceptance Criteria
- [ ] Integrity test module created
- [ ] Setup function with integrity checking enabled
- [ ] Helper functions for test data creation

## Success Metrics
- Setup completes in under 3 seconds
- Integrity checking is properly enabled

## Next Task
33b_test_referential_integrity.md