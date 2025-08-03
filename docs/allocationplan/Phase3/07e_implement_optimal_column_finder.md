# Task 07e: Implement Optimal Column Finder

**Estimated Time**: 6 minutes  
**Dependencies**: 07d_implement_column_assignment_method.md  
**Stage**: Neural Integration - Column Selection

## Objective
Implement helper method to find optimal column for content.

## Implementation

Add to `src/integration/cortical_column_integration.rs`:
```rust
impl CorticalColumnIntegration {
    async fn find_optimal_column(
        &self,
        concept_type: &str,
        content_features: &[f32],
    ) -> Result<ColumnId, ColumnAssignmentError> {
        // Query column manager for best match
        let available_columns = self.column_manager.get_available_columns().await
            .map_err(|e| ColumnAssignmentError::ManagerError(e.to_string()))?;
        
        if available_columns.is_empty() {
            return Err(ColumnAssignmentError::NoAvailableColumns);
        }
        
        // Simple heuristic: hash concept type to column
        let type_hash = self.hash_concept_type(concept_type);
        let column_index = type_hash % available_columns.len();
        
        Ok(available_columns[column_index])
    }
    
    fn hash_concept_type(&self, concept_type: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        concept_type.hash(&mut hasher);
        hasher.finish() as usize
    }
}
```

## Acceptance Criteria
- [ ] Column finding logic implemented
- [ ] Hash-based selection working
- [ ] Error handling for empty columns
- [ ] Proper return types

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **07f_implement_column_mapping_retrieval.md**