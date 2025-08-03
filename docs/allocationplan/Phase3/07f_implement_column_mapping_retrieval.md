# Task 07f: Implement Column Mapping Retrieval

**Estimated Time**: 5 minutes  
**Dependencies**: 07e_implement_optimal_column_finder.md  
**Stage**: Neural Integration - Mapping Queries

## Objective
Implement methods to retrieve column mappings.

## Implementation

Add to `src/integration/cortical_column_integration.rs`:
```rust
impl CorticalColumnIntegration {
    pub async fn get_concept_column(&self, concept_id: &str) -> Option<ColumnId> {
        self.column_mappings.read().await.get(concept_id).copied()
    }
    
    pub async fn get_concepts_in_column(&self, column_id: ColumnId) -> Vec<String> {
        let mappings = self.column_mappings.read().await;
        mappings
            .iter()
            .filter(|(_, &col_id)| col_id == column_id)
            .map(|(concept_id, _)| concept_id.clone())
            .collect()
    }
    
    pub async fn remove_concept_mapping(&self, concept_id: &str) -> Option<ColumnId> {
        self.column_mappings.write().await.remove(concept_id)
    }
}
```

## Acceptance Criteria
- [ ] Mapping retrieval methods compile
- [ ] Concept-to-column lookup working
- [ ] Column-to-concepts lookup working
- [ ] Mapping removal implemented

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **07g_create_cortical_integration_test.md**