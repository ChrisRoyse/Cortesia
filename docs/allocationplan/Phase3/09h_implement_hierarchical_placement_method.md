# Task 09h: Implement Hierarchical Placement Method

**Estimated Time**: 7 minutes  
**Dependencies**: 09g_implement_allocation_to_placement_converter.md  
**Stage**: Neural Integration - Hierarchy Logic

## Objective
Implement method to place nodes hierarchically based on allocation.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
impl AllocationGuidedPlacement {
    pub async fn place_node_hierarchically(
        &self,
        concept_id: &str,
        placement_decision: &PlacementDecision,
    ) -> Result<(), PlacementError> {
        let session = self.connection_manager.get_session().await
            .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
        
        // Create parent relationships if parent candidates exist
        for parent_candidate in &placement_decision.parent_candidates {
            let create_parent_rel_query = r#"
                MATCH (parent:Concept {id: $parent_id})
                MATCH (child:Concept {id: $child_id})
                MERGE (parent)-[r:CONTAINS]->(child)
                SET r.confidence = $confidence,
                    r.allocation_based = true,
                    r.created_at = datetime()
            "#;
            
            let params = hashmap![
                "parent_id".to_string() => parent_candidate.clone().into(),
                "child_id".to_string() => concept_id.to_string().into(),
                "confidence".to_string() => placement_decision.confidence_score.into()
            ];
            
            session.run(create_parent_rel_query, Some(params)).await
                .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
        }
        
        // Update node with allocation metadata
        let update_node_query = r#"
            MATCH (c:Concept {id: $concept_id})
            SET c.graph_level = $graph_level,
                c.semantic_cluster = $semantic_cluster,
                c.cortical_column_id = $cortical_column_id,
                c.allocation_confidence = $confidence,
                c.last_allocation_update = datetime()
        "#;
        
        let update_params = hashmap![
            "concept_id".to_string() => concept_id.to_string().into(),
            "graph_level".to_string() => (placement_decision.target_location.graph_level as i64).into(),
            "semantic_cluster".to_string() => placement_decision.target_location.semantic_cluster.clone().into(),
            "cortical_column_id".to_string() => placement_decision.target_location.cortical_column_id.map(|id| id as i64).into(),
            "confidence".to_string() => placement_decision.confidence_score.into()
        ];
        
        session.run(update_node_query, Some(update_params)).await
            .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
        
        Ok(())
    }
}
```

## Acceptance Criteria
- [ ] Hierarchical placement compiles
- [ ] Parent relationships created
- [ ] Node metadata updated
- [ ] Proper error handling

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09i_implement_neighbor_suggestion_method.md**