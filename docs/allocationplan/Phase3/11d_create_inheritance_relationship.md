# Task 11d: Implement Create Inheritance Relationship

**Time**: 8 minutes (1.5 min read, 5 min implement, 1.5 min verify)
**Dependencies**: 11c_hierarchy_manager_struct.md
**Stage**: Inheritance System

## Objective
Add method to create inheritance relationships with cycle detection.

## Implementation
Add to `src/inheritance/hierarchy_manager.rs`:

```rust
impl InheritanceHierarchyManager {
    pub async fn create_inheritance_relationship(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
        inheritance_type: InheritanceType,
        inheritance_weight: f32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Validate that this relationship won't create cycles
        if self.would_create_cycle(parent_concept_id, child_concept_id).await? {
            return Err(format!("Cycle detected: {} -> {}", parent_concept_id, child_concept_id).into());
        }
        
        // Generate relationship ID
        let relationship_id = format!("inh_{}_{}", parent_concept_id, child_concept_id);
        
        // TODO: Create in database (next task)
        
        Ok(relationship_id)
    }

    async fn would_create_cycle(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Simplified cycle check - implement in next tasks
        Ok(false)
    }
}
```

## Success Criteria
- Method compiles and basic validation works
- Cycle detection placeholder exists

## Next Task
11e_inheritance_database_creation.md