# Task 05g2: Add Filter Criteria Helper Methods

**Estimated Time**: 6 minutes  
**Dependencies**: 05g1_implement_basic_node_listing.md  
**Next Task**: 05g3_implement_count_nodes_method.md  

## Objective
Add convenience builder methods to FilterCriteria struct.

## Single Action
Add builder pattern methods to FilterCriteria.

## Code to Add
Add to FilterCriteria impl in `src/storage/crud_operations.rs`:
```rust
impl FilterCriteria {
    pub fn new(node_type: String) -> Self {
        Self {
            node_type: Some(node_type),
            ..Default::default()
        }
    }
    
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }
    
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
    
    pub fn with_order(mut self, field: String, ascending: bool) -> Self {
        self.order_by = Some(field);
        self.ascending = ascending;
        self
    }
}
```

## Success Check
```bash
cargo check
cargo test filter_criteria
```

## Acceptance Criteria
- [ ] Builder methods compile
- [ ] Fluent API works correctly
- [ ] Methods are chainable
- [ ] All fields can be set

## Duration
4-6 minutes for builder methods.