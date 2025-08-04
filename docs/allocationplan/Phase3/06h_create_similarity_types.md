# Task 06h: Create Similarity Types

**Estimated Time**: 4 minutes  
**Dependencies**: 06g_implement_similarity_calculation.md  
**Stage**: Neural Integration - Data Types

## Objective
Create data types for similarity operations.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
#[derive(Debug, Clone)]
pub struct TTFSSimilarConcept {
    pub concept_id: String,
    pub concept_name: String,
    pub encoding: f32,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct TTFSCluster {
    pub center_encoding: f32,
    pub concept_ids: Vec<String>,
    pub average_similarity: f32,
}

#[derive(Debug, Clone)]
pub struct TTFSQueryResult {
    pub concept: ConceptNode,
    pub ttfs_similarity: f32,
    pub encoding_difference: f32,
}
```

Add necessary imports:
```rust
use crate::core::types::ConceptNode;
```

## Acceptance Criteria
- [ ] All types compile
- [ ] Derive traits present
- [ ] Appropriate field types
- [ ] Import added for ConceptNode

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06i_create_concept_integration_struct.md**