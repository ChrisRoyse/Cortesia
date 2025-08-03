# Task 07g: Create Cortical Integration Test

**Estimated Time**: 7 minutes  
**Dependencies**: 07f_implement_column_mapping_retrieval.md  
**Stage**: Neural Integration - Testing

## Objective
Create basic test for cortical column integration functionality.

## Implementation

Create `tests/integration/cortical_integration_basic_test.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cortical_integration_creation() {
        let mock_column_manager = Arc::new(MockColumnManager::new());
        let integration = CorticalColumnIntegration::new(mock_column_manager).await;
        
        assert!(integration.is_ok());
    }
    
    #[tokio::test]
    async fn test_concept_type_hashing() {
        let mock_column_manager = Arc::new(MockColumnManager::new());
        let integration = CorticalColumnIntegration::new(mock_column_manager).await.unwrap();
        
        let hash1 = integration.hash_concept_type("Entity");
        let hash2 = integration.hash_concept_type("Entity");
        let hash3 = integration.hash_concept_type("Relationship");
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
    
    #[tokio::test]
    async fn test_mapping_operations() {
        let mock_column_manager = Arc::new(MockColumnManager::new());
        let integration = CorticalColumnIntegration::new(mock_column_manager).await.unwrap();
        
        // Test initial empty state
        assert!(integration.get_concept_column("concept1").await.is_none());
        
        // Test mapping storage (would be set during assignment)
        integration.column_mappings.write().await.insert("concept1".to_string(), 1);
        
        assert_eq!(integration.get_concept_column("concept1").await, Some(1));
        assert_eq!(integration.get_concepts_in_column(1).await, vec!["concept1".to_string()]);
        
        // Test removal
        assert_eq!(integration.remove_concept_mapping("concept1").await, Some(1));
        assert!(integration.get_concept_column("concept1").await.is_none());
    }
}

// Mock implementation for testing
struct MockColumnManager;

impl MockColumnManager {
    fn new() -> Self {
        Self
    }
}
```

## Acceptance Criteria
- [ ] Test file compiles
- [ ] Creation test present
- [ ] Hashing test present
- [ ] Mapping operations test present
- [ ] Mock manager implemented

## Validation Steps
```bash
cargo test cortical_integration_basic_test
```

## Next Task
Proceed to **07h_create_cortical_module_exports.md**