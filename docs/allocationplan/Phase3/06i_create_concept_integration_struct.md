# Task 06i: Create Concept Integration Struct

**Estimated Time**: 5 minutes  
**Dependencies**: 06h_create_similarity_types.md  
**Stage**: Neural Integration - Concept Integration

## Objective
Create struct for TTFS-concept integration operations.

## Implementation

Create `src/integration/ttfs_concept_integration.rs`:
```rust
use crate::integration::ttfs_integration::TTFSIntegrationService;
use crate::core::types::ConceptNode;
use crate::core::graph::entity_operations::NodeCrudService;
use std::sync::Arc;

pub struct TTFSConceptIntegration {
    ttfs_service: Arc<TTFSIntegrationService>,
    concept_crud: Arc<NodeCrudService<ConceptNode>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConceptCreationError {
    #[error("TTFS encoding failed: {0}")]
    TTFSError(String),
    #[error("Concept creation failed: {0}")]
    CreationFailed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ConceptUpdateError {
    #[error("Concept not found: {0}")]
    ConceptNotFound(String),
    #[error("TTFS encoding failed: {0}")]
    TTFSError(String),
    #[error("Update failed: {0}")]
    UpdateFailed(String),
}
```

Add to `src/integration/mod.rs`:
```rust
pub mod ttfs_concept_integration;
```

## Acceptance Criteria
- [ ] Struct compiles
- [ ] Error types defined
- [ ] Module exported
- [ ] Proper imports

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06j_implement_concept_creation_method.md**