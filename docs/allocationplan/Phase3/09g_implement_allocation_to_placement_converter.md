# Task 09g: Implement Allocation to Placement Converter

**Estimated Time**: 8 minutes  
**Dependencies**: 09f_implement_optimal_placement_method.md  
**Stage**: Neural Integration - Conversion Logic

## Objective
Implement converter from allocation results to placement decisions.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
impl AllocationGuidedPlacement {
    async fn convert_allocation_to_placement(
        &self,
        allocation_result: AllocationResult,
        cortical_consensus: CorticalConsensus,
        processing_time: std::time::Duration,
    ) -> Result<PlacementDecision, PlacementError> {
        // Determine placement location based on allocation
        let placement_location = PlacementLocation {
            graph_level: allocation_result.hierarchy_level,
            semantic_cluster: allocation_result.semantic_cluster.clone(),
            cortical_column_id: allocation_result.primary_column_id,
            suggested_neighbors: allocation_result.related_concepts.clone(),
        };
        
        // Create allocation metadata
        let allocation_metadata = AllocationMetadata {
            cortical_consensus_score: cortical_consensus.consensus_strength,
            allocation_strength: allocation_result.confidence,
            exception_indicators: allocation_result.exception_flags.clone(),
            processing_time_ms: processing_time.as_millis() as u64,
            timestamp: Utc::now(),
        };
        
        // Generate placement rationale
        let placement_rationale = format!(
            "Allocated to level {} in cluster '{}' with {:.2} confidence based on cortical consensus {:.2}",
            allocation_result.hierarchy_level,
            allocation_result.semantic_cluster,
            allocation_result.confidence,
            cortical_consensus.consensus_strength
        );
        
        Ok(PlacementDecision {
            target_location: placement_location,
            confidence_score: allocation_result.confidence,
            parent_candidates: allocation_result.parent_suggestions.clone(),
            child_suggestions: allocation_result.child_suggestions.clone(),
            allocation_metadata,
            placement_rationale,
        })
    }
}
```

## Acceptance Criteria
- [ ] Conversion method compiles
- [ ] All fields properly mapped
- [ ] Rationale generation working
- [ ] Metadata creation complete

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09h_implement_hierarchical_placement_method.md**