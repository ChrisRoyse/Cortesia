# Task 08b: Create Pathway Data Types

**Estimated Time**: 6 minutes  
**Dependencies**: 08a_add_neural_pathway_imports.md  
**Stage**: Neural Integration - Data Structures

## Objective
Create core data structures for neural pathway representation.

## Implementation

Create `src/neural_pathways/pathway_types.rs`:
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::integration::cortical_column_integration::ColumnId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPathway {
    pub id: String,
    pub source_concept_id: String,
    pub target_concept_id: String,
    pub pathway_type: PathwayType,
    pub activation_pattern: Vec<f32>,
    pub strength: f32,
    pub usage_count: u64,
    pub last_activated: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub decay_rate: f32,
    pub reinforcement_factor: f32,
    pub metadata: PathwayMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathwayType {
    Association,
    Causality,
    Similarity,
    Hierarchy,
    Temporal,
    Spatial,
    Custom(String),
}

impl Default for PathwayType {
    fn default() -> Self {
        PathwayType::Association
    }
}
```

## Acceptance Criteria
- [ ] All types compile without errors
- [ ] Serde serialization traits present
- [ ] Proper field types defined
- [ ] Default implementation for PathwayType

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **08c_create_pathway_metadata_types.md**