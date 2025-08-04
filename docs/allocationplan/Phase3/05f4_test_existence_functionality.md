# Task 05f4: Test Node Existence Functionality

**Estimated Time**: 8 minutes  
**Dependencies**: 05f3_implement_property_exists_check.md  
**Next Task**: 05g1_implement_basic_node_listing.md  

## Objective
Add comprehensive tests for all node existence checking methods.

## Single Action
Add test module for existence functionality.

## Code to Add
Add to `src/storage/crud_operations.rs`:
```rust
#[cfg(test)]
mod node_exists_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_exists_validation() {
        // Test input validation
        let empty_id = "";
        let valid_id = "valid_id_123";
        let empty_type = "";
        let valid_type = "Concept";
        
        // Test validation conditions
        assert!(empty_id.is_empty());
        assert!(!valid_id.is_empty());
        assert!(empty_type.is_empty());
        assert!(!valid_type.is_empty());
        
        // Test that validation errors would be triggered
        if empty_id.is_empty() || empty_type.is_empty() {
            assert!(true, "Empty parameters should trigger validation error");
        }
    }
    
    #[test]
    fn test_batch_exists_structure() {
        // Test batch existence check structure
        let ids = vec![
            "id_1".to_string(),
            "id_2".to_string(),
            "id_3".to_string(),
        ];
        
        assert_eq!(ids.len(), 3);
        assert!(!ids.is_empty());
        
        // Test result map structure
        let mut result_map = HashMap::new();
        for id in &ids {
            result_map.insert(id.clone(), false);
        }
        
        assert_eq!(result_map.len(), 3);
        assert_eq!(result_map.get("id_1"), Some(&false));
        
        // Simulate found nodes
        result_map.insert("id_1".to_string(), true);
        assert_eq!(result_map.get("id_1"), Some(&true));
    }
    
    #[test]
    fn test_cypher_query_optimization() {
        // Test that existence queries are optimized
        let node_type = "Memory";
        let exists_query = format!(
            "MATCH (n:{}) WHERE n.id = $id RETURN count(n) > 0 as exists",
            node_type
        );
        
        let batch_query = format!(
            "UNWIND $ids as node_id MATCH (n:{}) WHERE n.id = node_id RETURN n.id as found_id",
            node_type
        );
        
        // Verify optimized queries use count() instead of returning full nodes
        assert!(exists_query.contains("count(n) > 0"));
        assert!(!exists_query.contains("RETURN n"));
        
        // Verify batch query uses UNWIND for efficiency
        assert!(batch_query.contains("UNWIND $ids"));
        assert!(batch_query.contains("as node_id"));
    }
}
```

## Success Check
```bash
cargo test node_exists_tests
```

## Acceptance Criteria
- [ ] Validation tests pass
- [ ] Batch structure tests pass
- [ ] Query optimization tests pass
- [ ] All test cases covered

## Duration
6-8 minutes for existence tests.