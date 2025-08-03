# Task 09b: Create Placement Data Types

**Estimated Time**: 6 minutes  
**Dependencies**: 09a_add_allocation_imports.md  
**Stage**: Neural Integration - Data Structures

## Objective
Create data types for allocation-guided placement operations.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementDecision {
    pub target_location: PlacementLocation,
    pub confidence_score: f32,
    pub parent_candidates: Vec<String>,
    pub child_suggestions: Vec<String>,
    pub allocation_metadata: AllocationMetadata,
    pub placement_rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementLocation {
    pub graph_level: u32,
    pub semantic_cluster: String,
    pub cortical_column_id: Option<u32>,
    pub suggested_neighbors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationMetadata {
    pub cortical_consensus_score: f32,
    pub allocation_strength: f32,
    pub exception_indicators: Vec<String>,
    pub processing_time_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlacementError {
    #[error("Allocation engine error: {0}")]
    AllocationError(String),
    #[error("Cortical processing error: {0}")]
    CorticalError(String),
    #[error("Placement determination failed: {0}")]
    PlacementFailed(String),
}
```

## Acceptance Criteria
- [ ] All placement types compile
- [ ] Serde serialization traits present
- [ ] Error enum defined
- [ ] Proper field types

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09c_create_allocation_placement_struct.md**