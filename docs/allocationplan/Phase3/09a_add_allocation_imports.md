# Task 09a: Add Allocation Imports

**Estimated Time**: 3 minutes  
**Dependencies**: 08k_create_pathway_module_exports.md  
**Stage**: Neural Integration - Allocation Imports

## Objective
Add required imports for Phase 2 allocation engine components.

## Implementation

Add to `src/integration/mod.rs`:
```rust
pub mod allocation_placement;
```

Create `src/integration/allocation_placement.rs`:
```rust
use crate::phase2::allocation::{AllocationEngine, CorticalConsensus, AllocationResult};
use crate::phase2::cortical::{MultiColumnProcessor, ColumnResult};
use crate::phase2::ttfs::TTFSSpikePattern;
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;
use std::time::Instant;
```

## Acceptance Criteria
- [ ] Imports added without compilation errors
- [ ] Module structure prepared
- [ ] `cargo check` passes

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09b_create_placement_data_types.md**