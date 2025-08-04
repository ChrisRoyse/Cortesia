# AI Prompt: Micro Phase 3.5 - Compression Validation System

You are tasked with implementing the compression validation system that verifies correctness and measures effectiveness. Your goal is to create `src/compression/validator.rs` with comprehensive validation.

## Your Task
Implement the `CompressionValidator` struct that validates compression results, ensures data integrity, and measures compression effectiveness with detailed reporting.

## Specific Requirements
1. Create `src/compression/validator.rs` with CompressionValidator struct
2. Validate that compression preserves all property values
3. Verify that property resolution still works correctly
4. Measure actual compression ratios and memory savings
5. Detect any data inconsistencies or corruption
6. Provide detailed validation reports

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::properties::resolver::PropertyResolver;
use crate::compression::orchestrator::CompressionReport;

pub struct CompressionValidator {
    property_resolver: PropertyResolver,
    validation_depth: ValidationDepth,
}

#[derive(Debug)]
pub struct ValidationReport {
    pub validation_passed: bool,
    pub property_resolution_tests: usize,
    pub integrity_violations: Vec<IntegrityViolation>,
    pub actual_compression_ratio: f32,
    pub measured_bytes_saved: usize,
    pub validation_duration: std::time::Duration,
}

impl CompressionValidator {
    pub fn new() -> Self;
    
    pub fn validate_compression(&self, hierarchy: &InheritanceHierarchy, before_snapshot: &HierarchySnapshot) -> ValidationReport;
    
    pub fn verify_property_integrity(&self, hierarchy: &InheritanceHierarchy) -> Vec<IntegrityViolation>;
    
    pub fn measure_compression_effectiveness(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> CompressionMeasurement;
}
```

## Success Criteria
- [ ] Validates compression preserves all data correctly
- [ ] Accurately measures compression effectiveness
- [ ] Detects any integrity violations
- [ ] Provides comprehensive validation reporting

## File to Create
Create exactly this file: `src/compression/validator.rs`

## When Complete
Respond with "MICRO PHASE 3.5 COMPLETE" and a brief summary of your implementation.