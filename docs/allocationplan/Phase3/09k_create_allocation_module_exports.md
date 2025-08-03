# Task 09k: Create Allocation Module Exports

**Estimated Time**: 3 minutes  
**Dependencies**: 09j_create_allocation_placement_test.md  
**Stage**: Neural Integration - Module Setup

## Objective
Export allocation placement modules properly.

## Implementation

Update `src/integration/mod.rs`:
```rust
pub mod ttfs_integration;
pub mod ttfs_concept_integration;
pub mod cortical_column_integration;
pub mod allocation_placement;

pub use ttfs_integration::{
    TTFSIntegrationService, 
    TTFSIntegrationError,
    TTFSSimilarConcept,
    TTFSCluster,
    TTFSQueryResult
};

pub use ttfs_concept_integration::{
    TTFSConceptIntegration,
    ConceptCreationError,
    ConceptUpdateError
};

pub use cortical_column_integration::{
    CorticalColumnIntegration,
    ColumnAssignmentError,
    IntegrationError,
    ColumnId
};

pub use allocation_placement::{
    AllocationGuidedPlacement,
    PlacementDecision,
    PlacementLocation,
    AllocationMetadata,
    PlacementError
};
```

## Acceptance Criteria
- [ ] All allocation modules exported
- [ ] Public types accessible
- [ ] No compilation errors
- [ ] Clean module structure

## Validation Steps
```bash
cargo check
cargo doc --no-deps
```

## Success Metrics
- All allocation placement components properly modularized
- Documentation builds without warnings
- Module imports work correctly

## Next Task
Allocation-guided placement micro-tasks complete. Proceed to **10a_add_spike_pattern_imports.md** for spike pattern processing.