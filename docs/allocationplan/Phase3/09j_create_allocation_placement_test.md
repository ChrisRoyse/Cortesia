# Task 09j: Create Allocation Placement Test

**Estimated Time**: 8 minutes  
**Dependencies**: 09i_implement_neighbor_suggestion_method.md  
**Stage**: Neural Integration - Testing

## Objective
Create comprehensive test for allocation-guided placement functionality.

## Implementation

Create `tests/integration/allocation_placement_basic_test.rs`:
```rust
#[cfg(test)]
mod tests {
    use crate::integration::allocation_placement::*;
    
    #[tokio::test]
    async fn test_cache_key_generation() {
        let mock_allocation_engine = Arc::new(MockAllocationEngine::new());
        let mock_processor = Arc::new(MockMultiColumnProcessor::new());
        let mock_connection = Arc::new(MockConnectionManager::new());
        
        let placement = AllocationGuidedPlacement::new(
            mock_allocation_engine,
            mock_processor,
            mock_connection,
        ).await.unwrap();
        
        let spike_pattern = TTFSSpikePattern {
            first_spike_time: 1.0,
            pattern_hash: "test_hash".to_string(),
        };
        
        let key1 = placement.generate_placement_cache_key("test content", &spike_pattern);
        let key2 = placement.generate_placement_cache_key("test content", &spike_pattern);
        let key3 = placement.generate_placement_cache_key("different content", &spike_pattern);
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
    
    #[tokio::test]
    async fn test_allocation_to_placement_conversion() {
        let mock_allocation_engine = Arc::new(MockAllocationEngine::new());
        let mock_processor = Arc::new(MockMultiColumnProcessor::new());
        let mock_connection = Arc::new(MockConnectionManager::new());
        
        let placement = AllocationGuidedPlacement::new(
            mock_allocation_engine,
            mock_processor,
            mock_connection,
        ).await.unwrap();
        
        let allocation_result = AllocationResult {
            hierarchy_level: 2,
            semantic_cluster: "test_cluster".to_string(),
            primary_column_id: Some(1),
            confidence: 0.8,
            related_concepts: vec!["concept1".to_string()],
            parent_suggestions: vec!["parent1".to_string()],
            child_suggestions: vec!["child1".to_string()],
            exception_flags: Vec::new(),
        };
        
        let cortical_consensus = CorticalConsensus {
            consensus_strength: 0.9,
        };
        
        let placement_decision = placement.convert_allocation_to_placement(
            allocation_result,
            cortical_consensus,
            std::time::Duration::from_millis(100),
        ).await.unwrap();
        
        assert_eq!(placement_decision.target_location.graph_level, 2);
        assert_eq!(placement_decision.target_location.semantic_cluster, "test_cluster");
        assert_eq!(placement_decision.confidence_score, 0.8);
        assert_eq!(placement_decision.parent_candidates.len(), 1);
    }
}

// Mock implementations for testing
struct MockAllocationEngine;
struct MockMultiColumnProcessor;
struct MockConnectionManager;

impl MockAllocationEngine {
    fn new() -> Self { Self }
}

impl MockMultiColumnProcessor {
    fn new() -> Self { Self }
}

impl MockConnectionManager {
    fn new() -> Self { Self }
}
```

## Acceptance Criteria
- [ ] All tests compile and pass
- [ ] Cache key generation tested
- [ ] Conversion logic tested
- [ ] Mock implementations present

## Validation Steps
```bash
cargo test allocation_placement_basic_test
```

## Next Task
Proceed to **09k_create_allocation_module_exports.md**