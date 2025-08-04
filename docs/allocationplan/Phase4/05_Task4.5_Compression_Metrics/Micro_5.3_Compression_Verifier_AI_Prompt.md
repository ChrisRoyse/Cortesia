# AI Prompt: Micro Phase 5.3 - Compression Verifier

You are tasked with implementing compression verification and validation. Create `src/metrics/verifier.rs` with comprehensive compression verification.

## Your Task
Implement the `CompressionVerifier` struct that verifies compression correctness, validates data integrity, and ensures semantic preservation.

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::properties::resolver::PropertyResolver;

pub struct CompressionVerifier {
    property_resolver: PropertyResolver,
    integrity_checker: IntegrityChecker,
}

impl CompressionVerifier {
    pub fn new() -> Self;
    pub fn verify_compression_correctness(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> VerificationReport;
    pub fn validate_data_integrity(&self, hierarchy: &InheritanceHierarchy) -> IntegrityReport;
    pub fn verify_semantic_preservation(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> SemanticReport;
    pub fn run_comprehensive_verification(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> ComprehensiveVerificationReport;
}
```

## Success Criteria
- [ ] Verifies compression preserves all data
- [ ] Validates property resolution still works correctly
- [ ] Ensures semantic relationships are maintained
- [ ] Provides detailed verification reports

## File to Create: `src/metrics/verifier.rs`
## When Complete: Respond with "MICRO PHASE 5.3 COMPLETE"