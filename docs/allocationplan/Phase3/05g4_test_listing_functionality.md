# Task 05g4: Test Node Listing Functionality

**Estimated Time**: 10 minutes  
**Dependencies**: 05g3_implement_count_nodes_method.md  
**Next Task**: 05h1_create_crud_test_structure.md  

## Objective
Add comprehensive tests for node listing and filtering.

## Single Action
Add test module for node listing functionality.

## Code to Add
Add to `src/storage/crud_operations.rs`:
```rust
#[cfg(test)]
mod node_listing_tests {
    use super::*;
    use crate::storage::node_types::*;
    
    #[test]
    fn test_filter_criteria_builder() {
        let filters = FilterCriteria::new("Concept".to_string())
            .with_property("type".to_string(), "Entity".to_string())
            .with_property("status".to_string(), "active".to_string())
            .with_limit(50)
            .with_offset(10)
            .with_order("name".to_string(), true);
        
        assert_eq!(filters.node_type, Some("Concept".to_string()));
        assert_eq!(filters.properties.len(), 2);
        assert_eq!(filters.limit, Some(50));
        assert_eq!(filters.offset, Some(10));
        assert_eq!(filters.order_by, Some("name".to_string()));
        assert!(filters.ascending);
    }
    
    #[test]
    fn test_cypher_query_construction() {
        // Test basic query construction
        let node_type = "Memory";
        let basic_query = format!("MATCH (n:{}) ORDER BY n.id ASC RETURN n", node_type);
        
        assert!(basic_query.contains("MATCH (n:Memory)"));
        assert!(basic_query.contains("ORDER BY n.id ASC"));
        assert!(basic_query.contains("RETURN n"));
        
        // Test query with WHERE clause
        let where_clauses = vec!["n.type = $prop_0", "n.status = $prop_1"];
        let where_part = format!(" WHERE {}", where_clauses.join(" AND "));
        let filtered_query = format!("MATCH (n:{}){} RETURN n", node_type, where_part);
        
        assert!(filtered_query.contains("WHERE n.type = $prop_0 AND n.status = $prop_1"));
    }
    
    #[test]
    fn test_count_query_construction() {
        let node_type = "Property";
        let count_query = format!("MATCH (n:{}) RETURN count(n) as node_count", node_type);
        
        assert!(count_query.contains("MATCH (n:Property)"));
        assert!(count_query.contains("count(n)"));
        assert!(count_query.contains("as node_count"));
        assert!(!count_query.contains("RETURN n")); // Should not return full nodes
    }
}
```

## Success Check
```bash
cargo test node_listing_tests
```

## Acceptance Criteria
- [ ] Filter builder tests pass
- [ ] Query construction tests pass
- [ ] Count query tests pass
- [ ] All edge cases covered

## Duration
8-10 minutes for comprehensive tests.