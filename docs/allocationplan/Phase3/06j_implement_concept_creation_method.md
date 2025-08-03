# Task 06j: Implement Concept Creation Method

**Estimated Time**: 8 minutes  
**Dependencies**: 06i_create_concept_integration_struct.md  
**Stage**: Neural Integration - Concept Creation

## Objective
Implement method to create concepts with TTFS encoding.

## Implementation

Add to `src/integration/ttfs_concept_integration.rs`:
```rust
use chrono::Utc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl TTFSConceptIntegration {
    pub async fn new(
        ttfs_service: Arc<TTFSIntegrationService>,
        concept_crud: Arc<NodeCrudService<ConceptNode>>,
    ) -> Self {
        Self {
            ttfs_service,
            concept_crud,
        }
    }

    pub async fn create_concept_with_ttfs(
        &self,
        name: &str,
        content: &str,
        concept_type: &str,
    ) -> Result<ConceptNode, ConceptCreationError> {
        // Generate TTFS encoding for the content
        let ttfs_encoding = self.ttfs_service.encode_content(content).await
            .map_err(|e| ConceptCreationError::TTFSError(e.to_string()))?;
        
        // Create concept with TTFS encoding
        let concept = ConceptNode {
            id: String::new(), // Will be set by CRUD service
            name: name.to_string(),
            concept_type: concept_type.to_string(),
            ttfs_encoding: Some(ttfs_encoding),
            content_hash: self.generate_content_hash(content),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
        };
        
        // Store in knowledge graph
        let concept_id = self.concept_crud.create(&concept).await
            .map_err(|e| ConceptCreationError::CreationFailed(e.to_string()))?;
        
        // Update the concept with the actual ID
        let mut final_concept = concept;
        final_concept.id = concept_id;
        
        Ok(final_concept)
    }

    fn generate_content_hash(&self, content: &str) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
```

## Acceptance Criteria
- [ ] Method compiles
- [ ] TTFS encoding integrated
- [ ] Error handling implemented
- [ ] Content hash generated

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06k_create_basic_integration_test.md**