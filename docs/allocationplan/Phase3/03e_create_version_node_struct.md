# Task 03e: Create Version Node Struct

**Estimated Time**: 8 minutes  
**Dependencies**: 03d_create_exception_node_struct.md  
**Next Task**: 03f_create_neural_pathway_struct.md  

## Objective
Create the VersionNode data structure for temporal versioning system.

## Single Action
Add VersionNode struct and version management types to node_types.rs.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionNode {
    pub id: String,
    pub branch_name: String,
    pub version_number: i32,
    pub parent_version: Option<String>,
    pub change_summary: String,
    pub change_type: ChangeType,
    pub affected_entities: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub is_current: bool,
    pub is_stable: bool,
    pub merge_conflicts: Vec<MergeConflict>,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    Create,
    Update,
    Delete,
    Merge,
    Branch,
    Rollback,
    Compress,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MergeConflict {
    pub entity_id: String,
    pub property_name: String,
    pub source_value: PropertyValue,
    pub target_value: PropertyValue,
    pub resolution_strategy: Option<ConflictResolution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictResolution {
    UseSource,
    UseTarget,
    Merge,
    Manual,
}

impl VersionNode {
    pub fn new(
        branch_name: String,
        version_number: i32,
        change_summary: String,
        change_type: ChangeType,
        created_by: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            branch_name,
            version_number,
            parent_version: None,
            change_summary,
            change_type,
            affected_entities: Vec::new(),
            created_at: Utc::now(),
            created_by,
            is_current: true,
            is_stable: false,
            merge_conflicts: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_version = Some(parent_id);
        self
    }
    
    pub fn add_affected_entity(&mut self, entity_id: String) {
        if !self.affected_entities.contains(&entity_id) {
            self.affected_entities.push(entity_id);
        }
    }
    
    pub fn mark_stable(&mut self) {
        self.is_stable = true;
    }
    
    pub fn add_merge_conflict(&mut self, conflict: MergeConflict) {
        self.merge_conflicts.push(conflict);
    }
    
    pub fn has_conflicts(&self) -> bool {
        !self.merge_conflicts.is_empty()
    }
    
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;
    
    #[test]
    fn test_version_node_creation() {
        let version = VersionNode::new(
            "main".to_string(),
            1,
            "Initial version".to_string(),
            ChangeType::Create,
            "system".to_string(),
        );
        
        assert_eq!(version.branch_name, "main");
        assert_eq!(version.version_number, 1);
        assert_eq!(version.change_type, ChangeType::Create);
        assert!(version.is_current);
        assert!(!version.is_stable);
        assert!(!version.has_conflicts());
    }
    
    #[test]
    fn test_version_with_conflicts() {
        let mut version = VersionNode::new(
            "feature".to_string(),
            2,
            "Feature branch".to_string(),
            ChangeType::Branch,
            "developer".to_string(),
        );
        
        let conflict = MergeConflict {
            entity_id: "entity_1".to_string(),
            property_name: "name".to_string(),
            source_value: PropertyValue::String("source".to_string()),
            target_value: PropertyValue::String("target".to_string()),
            resolution_strategy: None,
        };
        
        version.add_merge_conflict(conflict);
        assert!(version.has_conflicts());
        assert_eq!(version.merge_conflicts.len(), 1);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run version tests
cargo test version_tests
```

## Acceptance Criteria
- [ ] VersionNode struct compiles without errors
- [ ] Change tracking system works
- [ ] Conflict detection mechanisms function
- [ ] Metadata storage works
- [ ] Tests pass

## Duration
6-8 minutes for version node implementation and testing.