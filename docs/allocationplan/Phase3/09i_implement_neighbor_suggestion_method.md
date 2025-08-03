# Task 09i: Implement Neighbor Suggestion Method

**Estimated Time**: 6 minutes  
**Dependencies**: 09h_implement_hierarchical_placement_method.md  
**Stage**: Neural Integration - Neighbor Logic

## Objective
Implement method to suggest and create neighbor relationships.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
impl AllocationGuidedPlacement {
    pub async fn create_suggested_neighbor_relationships(
        &self,
        concept_id: &str,
        suggested_neighbors: &[String],
        confidence_threshold: f32,
    ) -> Result<usize, PlacementError> {
        let session = self.connection_manager.get_session().await
            .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
        
        let mut relationships_created = 0;
        
        for neighbor_id in suggested_neighbors {
            // Check if relationship already exists
            let check_query = r#"
                MATCH (a:Concept {id: $concept_id})
                MATCH (b:Concept {id: $neighbor_id})
                RETURN EXISTS((a)-[:RELATED_TO]-(b)) as exists
            "#;
            
            let check_params = hashmap![
                "concept_id".to_string() => concept_id.to_string().into(),
                "neighbor_id".to_string() => neighbor_id.clone().into()
            ];
            
            let result = session.run(check_query, Some(check_params)).await
                .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
            
            if let Some(record) = result.into_iter().next() {
                let exists: bool = record.get("exists").unwrap_or(false);
                if exists {
                    continue; // Skip if relationship already exists
                }
            }
            
            // Create new neighbor relationship
            let create_query = r#"
                MATCH (a:Concept {id: $concept_id})
                MATCH (b:Concept {id: $neighbor_id})
                CREATE (a)-[r:RELATED_TO]->(b)
                SET r.confidence = $confidence,
                    r.allocation_suggested = true,
                    r.created_at = datetime()
            "#;
            
            let create_params = hashmap![
                "concept_id".to_string() => concept_id.to_string().into(),
                "neighbor_id".to_string() => neighbor_id.clone().into(),
                "confidence".to_string() => confidence_threshold.into()
            ];
            
            session.run(create_query, Some(create_params)).await
                .map_err(|e| PlacementError::PlacementFailed(e.to_string()))?;
            
            relationships_created += 1;
        }
        
        Ok(relationships_created)
    }
}
```

## Acceptance Criteria
- [ ] Neighbor suggestion method compiles
- [ ] Duplicate relationship checking
- [ ] Relationship creation working
- [ ] Returns count of created relationships

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09j_create_allocation_placement_test.md**