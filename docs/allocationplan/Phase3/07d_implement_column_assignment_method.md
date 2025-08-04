# Task 07d: Implement Column Assignment Method

**Estimated Time**: 8 minutes  
**Dependencies**: 07c_implement_cortical_constructor.md  
**Stage**: Neural Integration - Assignment Logic

## Objective
Implement method to assign concepts to cortical columns.

## Implementation

Add to `src/integration/cortical_column_integration.rs`:
```rust
impl CorticalColumnIntegration {
    pub async fn assign_concept_to_column(
        &self,
        concept_id: &str,
        concept_type: &str,
        content_features: &[f32],
    ) -> Result<ColumnId, ColumnAssignmentError> {
        // Find optimal column based on content features
        let optimal_column = self.find_optimal_column(
            concept_type,
            content_features,
        ).await?;
        
        // Check column capacity
        if !self.column_manager.has_capacity(optimal_column).await
            .map_err(|e| ColumnAssignmentError::ManagerError(e.to_string()))? {
            return Err(ColumnAssignmentError::NoAvailableColumns);
        }
        
        // Assign concept to column
        self.column_manager.assign_concept(optimal_column, concept_id).await
            .map_err(|e| ColumnAssignmentError::AssignmentFailed(e.to_string()))?;
        
        // Update mappings
        self.column_mappings.write().await.insert(
            concept_id.to_string(),
            optimal_column,
        );
        
        Ok(optimal_column)
    }
}
```

## Acceptance Criteria
- [ ] Method compiles without errors
- [ ] Capacity checking implemented
- [ ] Mapping updates included
- [ ] Proper error handling

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **07e_implement_optimal_column_finder.md**