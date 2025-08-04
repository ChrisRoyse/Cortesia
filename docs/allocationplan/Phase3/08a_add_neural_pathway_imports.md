# Task 08a: Add Neural Pathway Imports

**Estimated Time**: 3 minutes  
**Dependencies**: 07h_create_cortical_module_exports.md  
**Stage**: Neural Integration - Pathway Imports

## Objective
Add required imports for neural pathway storage components.

## Implementation

Add to `src/lib.rs`:
```rust
pub mod neural_pathways {
    pub mod pathway_types;
    pub mod pathway_storage;
}
```

Create `src/neural_pathways/mod.rs`:
```rust
pub mod pathway_types;
pub mod pathway_storage;

pub use pathway_types::*;
pub use pathway_storage::*;
```

## Acceptance Criteria
- [ ] Module structure created
- [ ] Imports added without compilation errors
- [ ] `cargo check` passes

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **08b_create_pathway_data_types.md**