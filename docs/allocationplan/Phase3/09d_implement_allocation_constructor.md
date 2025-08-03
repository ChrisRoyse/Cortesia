# Task 09d: Implement Allocation Constructor

**Estimated Time**: 4 minutes  
**Dependencies**: 09c_create_allocation_placement_struct.md  
**Stage**: Neural Integration - Constructor

## Objective
Implement constructor for AllocationGuidedPlacement.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
impl AllocationGuidedPlacement {
    pub async fn new(
        allocation_engine: Arc<AllocationEngine>,
        multi_column_processor: Arc<MultiColumnProcessor>,
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, PlacementError> {
        Ok(Self {
            allocation_engine,
            multi_column_processor,
            connection_manager,
            placement_cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(5000).unwrap()
            ))),
        })
    }
}
```

## Acceptance Criteria
- [ ] Constructor compiles without errors
- [ ] Proper initialization of all fields
- [ ] Cache size configured correctly
- [ ] Returns Result type

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09e_implement_cache_key_generator.md**