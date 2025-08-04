# AI Prompt: Micro Phase 3.2 - Property Promotion Engine

You are tasked with implementing the property promotion engine that safely promotes properties up the inheritance hierarchy. Your goal is to create `src/compression/promoter.rs` with safe property promotion.

## Your Task
Implement the `PropertyPromoter` struct that promotes properties from child nodes to ancestor nodes while maintaining correctness and handling conflicts.

## Specific Requirements
1. Create `src/compression/promoter.rs` with PropertyPromoter struct
2. Implement safe property promotion with conflict detection
3. Add rollback capability for failed promotions
4. Handle exception creation for conflicting values
5. Ensure atomicity of promotion operations
6. Integrate with existing hierarchy and exception systems

## Expected Code Structure
```rust
use crate::compression::analyzer::PromotionCandidate;
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::exceptions::store::ExceptionStore;

pub struct PropertyPromoter {
    dry_run_mode: bool,
    conflict_resolution: ConflictResolution,
    exception_store: ExceptionStore,
}

#[derive(Debug)]
pub enum PromotionResult {
    Success(PromotionReport),
    Conflict(ConflictReport),
    Failed(PromotionError),
}

impl PropertyPromoter {
    pub fn new() -> Self;
    
    pub fn promote_property(&mut self, hierarchy: &mut InheritanceHierarchy, candidate: &PromotionCandidate) -> PromotionResult;
    
    pub fn rollback_promotion(&mut self, hierarchy: &mut InheritanceHierarchy, promotion_id: PromotionId) -> Result<(), PromotionError>;
    
    pub fn validate_promotion(&self, hierarchy: &InheritanceHierarchy, candidate: &PromotionCandidate) -> ValidationResult;
}
```

## Success Criteria
- [ ] Safely promotes properties without data loss
- [ ] Handles conflicts by creating appropriate exceptions
- [ ] Maintains hierarchy consistency
- [ ] Provides rollback capability for failed operations

## File to Create
Create exactly this file: `src/compression/promoter.rs`

## When Complete
Respond with "MICRO PHASE 3.2 COMPLETE" and a brief summary of your implementation.