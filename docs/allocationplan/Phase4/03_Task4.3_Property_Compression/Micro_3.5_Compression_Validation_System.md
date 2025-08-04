# Micro Phase 3.5: Compression Validation System

**Estimated Time**: 30 minutes
**Dependencies**: Micro 3.4 (Iterative Compression Algorithm)
**Objective**: Ensure semantic correctness and data integrity after compression operations

## Task Description

Implement a comprehensive validation system that verifies compression operations maintain semantic correctness, detect any data corruption, and ensure the inheritance hierarchy remains functionally equivalent after compression.

## Deliverables

Create `src/compression/validator.rs` with:

1. **CompressionValidator struct**: Core validation engine
2. **Semantic verification**: Ensure no meaning is lost during compression
3. **Integrity checking**: Verify data consistency across the hierarchy
4. **Property resolution**: Validate inheritance still works correctly
5. **Exception validation**: Ensure exceptions are properly created and function
6. **Performance validation**: Verify compression doesn't break query performance

## Success Criteria

- [ ] Detects 100% of semantic violations during compression
- [ ] Validates property resolution matches pre-compression behavior
- [ ] Verifies all exceptions are semantically correct
- [ ] Ensures no data corruption or loss occurs
- [ ] Completes validation of 10,000 nodes in < 50ms
- [ ] Provides detailed error reports for any issues found

## Implementation Requirements

```rust
pub struct CompressionValidator {
    validation_level: ValidationLevel,
    enable_performance_checks: bool,
    max_validation_time: Duration,
    error_tolerance: ErrorTolerance,
}

#[derive(Debug, Clone)]
pub enum ValidationLevel {
    Basic,          // Essential checks only
    Standard,       // Most common validation scenarios
    Comprehensive,  // Full semantic and integrity validation
    Paranoid,       // Exhaustive validation with cross-checks
}

#[derive(Debug, Clone)]
pub enum ErrorTolerance {
    None,           // Any error fails validation
    Minor,          // Allow minor inconsistencies
    Moderate,       // Allow some data loss if semantically equivalent
}

#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub validation_level: ValidationLevel,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub performance_metrics: ValidationMetrics,
    pub semantic_integrity_score: f32,
    pub data_consistency_score: f32,
}

#[derive(Debug)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub severity: ErrorSeverity,
    pub description: String,
    pub affected_nodes: Vec<NodeId>,
    pub property_name: Option<String>,
    pub expected_value: Option<PropertyValue>,
    pub actual_value: Option<PropertyValue>,
}

#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    SemanticViolation,
    DataCorruption,
    InheritanceBreakage,
    ExceptionMalformation,
    PerformanceDegradation,
    InconsistentState,
}

impl CompressionValidator {
    pub fn new(level: ValidationLevel) -> Self;
    
    pub fn validate_compression(
        &self,
        original_hierarchy: &InheritanceHierarchy,
        compressed_hierarchy: &InheritanceHierarchy,
        compression_operations: &[CompressionOperation]
    ) -> ValidationResult;
    
    pub fn validate_semantic_equivalence(
        &self,
        original: &InheritanceHierarchy,
        compressed: &InheritanceHierarchy
    ) -> SemanticValidationResult;
    
    pub fn validate_property_resolution(
        &self,
        hierarchy: &InheritanceHierarchy,
        test_nodes: &[NodeId]
    ) -> ResolutionValidationResult;
    
    pub fn validate_exceptions(
        &self,
        hierarchy: &InheritanceHierarchy,
        exceptions: &[Exception]
    ) -> ExceptionValidationResult;
    
    pub fn validate_compression_integrity(
        &self,
        hierarchy: &InheritanceHierarchy,
        operations: &[CompressionOperation]
    ) -> IntegrityValidationResult;
    
    pub fn benchmark_query_performance(
        &self,
        original: &InheritanceHierarchy,
        compressed: &InheritanceHierarchy
    ) -> PerformanceComparisonResult;
}

#[derive(Debug)]
pub struct SemanticValidationResult {
    pub nodes_checked: usize,
    pub properties_verified: usize,
    pub semantic_violations: Vec<SemanticViolation>,
    pub equivalence_score: f32,
}

#[derive(Debug)]
pub struct CompressionOperation {
    pub operation_type: OperationType,
    pub source_node: NodeId,
    pub target_node: NodeId,
    pub property_name: String,
    pub original_value: PropertyValue,
    pub promoted_value: PropertyValue,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    PropertyPromotion,
    ExceptionCreation,
    PropertyRemoval,
    ValueConsolidation,
}
```

## Test Requirements

Must pass compression validation tests:
```rust
#[test]
fn test_semantic_equivalence_validation() {
    let original = create_animal_hierarchy();
    let mut compressed = original.clone();
    
    // Apply valid compression
    promote_property(&mut compressed, "warm_blooded", "Mammal");
    
    let validator = CompressionValidator::new(ValidationLevel::Standard);
    let result = validator.validate_semantic_equivalence(&original, &compressed);
    
    assert!(result.equivalence_score >= 0.99); // Should be nearly identical semantically
    assert!(result.semantic_violations.is_empty());
    
    // Test all nodes still resolve properties correctly
    for node_id in original.get_all_nodes() {
        for property in original.get_all_property_names() {
            let original_value = original.get_property(node_id, property);
            let compressed_value = compressed.get_property(node_id, property);
            assert_eq!(original_value, compressed_value, 
                      "Property '{}' differs for node {:?}", property, node_id);
        }
    }
}

#[test]
fn test_invalid_compression_detection() {
    let original = create_animal_hierarchy();
    let mut corrupted = original.clone();
    
    // Simulate corruption: remove property without proper exception
    let dog_node = corrupted.get_node_by_name("Dog").unwrap();
    corrupted.remove_property(dog_node, "loyal"); // This breaks semantics
    
    let validator = CompressionValidator::new(ValidationLevel::Comprehensive);
    let operations = vec![]; // Empty operations - this is the problem
    
    let result = validator.validate_compression(&original, &corrupted, &operations);
    
    assert!(!result.is_valid);
    assert!(!result.errors.is_empty());
    assert!(result.errors.iter().any(|e| matches!(e.error_type, ValidationErrorType::SemanticViolation)));
    assert!(result.semantic_integrity_score < 0.9);
}

#[test]
fn test_exception_validation() {
    let mut hierarchy = create_bird_hierarchy();
    
    // Create proper exception for penguin not flying
    let bird_node = hierarchy.get_node_by_name("Bird").unwrap();
    let penguin_node = hierarchy.get_node_by_name("Penguin").unwrap();
    
    hierarchy.set_property(bird_node, "can_fly", PropertyValue::Boolean(true));
    
    let exception = Exception {
        node_id: penguin_node,
        property_name: "can_fly".to_string(),
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        reason: "Penguins are flightless birds".to_string(),
    };
    
    hierarchy.add_exception(exception.clone());
    
    let validator = CompressionValidator::new(ValidationLevel::Standard);
    let exception_result = validator.validate_exceptions(&hierarchy, &[exception]);
    
    assert!(exception_result.is_valid);
    assert!(exception_result.malformed_exceptions.is_empty());
    
    // Verify penguin resolves to false, other birds to true
    assert_eq!(hierarchy.get_property(penguin_node, "can_fly"), Some(PropertyValue::Boolean(false)));
    
    let eagle_node = hierarchy.get_node_by_name("Eagle").unwrap();
    assert_eq!(hierarchy.get_property(eagle_node, "can_fly"), Some(PropertyValue::Boolean(true)));
}

#[test]
fn test_compression_integrity_check() {
    let original = create_complex_hierarchy(1000);
    let mut compressed = original.clone();
    
    let operations = perform_compression(&mut compressed);
    
    let validator = CompressionValidator::new(ValidationLevel::Comprehensive);
    let integrity_result = validator.validate_compression_integrity(&compressed, &operations);
    
    assert!(integrity_result.is_consistent);
    assert!(integrity_result.operation_consistency_score >= 0.95);
    
    // Verify all operations are accounted for
    for operation in &operations {
        assert!(integrity_result.verified_operations.contains(&operation.operation_type));
    }
}

#[test]
fn test_performance_regression_detection() {
    let original = create_realistic_hierarchy(5000);
    let mut compressed = original.clone();
    
    perform_aggressive_compression(&mut compressed);
    
    let validator = CompressionValidator::new(ValidationLevel::Standard);
    validator.enable_performance_checks = true;
    
    let perf_result = validator.benchmark_query_performance(&original, &compressed);
    
    // Compressed version should be faster or at least not significantly slower
    assert!(perf_result.query_performance_ratio >= 0.8); // At most 20% slower
    assert!(perf_result.memory_efficiency_ratio >= 5.0); // At least 5x less memory
    
    // Test specific query types
    assert!(perf_result.property_lookup_performance >= 0.9);
    assert!(perf_result.inheritance_traversal_performance >= 0.8);
}

#[test]
fn test_validation_performance() {
    let original = create_large_hierarchy(10000);
    let mut compressed = original.clone();
    
    let operations = perform_standard_compression(&mut compressed);
    
    let validator = CompressionValidator::new(ValidationLevel::Standard);
    
    let start = Instant::now();
    let result = validator.validate_compression(&original, &compressed, &operations);
    let validation_time = start.elapsed();
    
    assert!(validation_time < Duration::from_millis(50)); // < 50ms for 10k nodes
    assert!(result.is_valid);
    assert!(result.performance_metrics.nodes_per_second > 200000); // > 200k nodes/sec
}

#[test]
fn test_comprehensive_validation_levels() {
    let original = create_animal_hierarchy();
    let mut compressed = original.clone();
    let operations = perform_minor_compression(&mut compressed);
    
    // Test different validation levels
    let levels = [
        ValidationLevel::Basic,
        ValidationLevel::Standard,
        ValidationLevel::Comprehensive,
        ValidationLevel::Paranoid,
    ];
    
    for level in levels {
        let validator = CompressionValidator::new(level);
        let result = validator.validate_compression(&original, &compressed, &operations);
        
        assert!(result.is_valid);
        
        // More comprehensive levels should find more details (warnings, not errors)
        match level {
            ValidationLevel::Basic => assert!(result.warnings.len() <= 5),
            ValidationLevel::Standard => assert!(result.warnings.len() <= 10),
            ValidationLevel::Comprehensive => assert!(result.warnings.len() <= 20),
            ValidationLevel::Paranoid => assert!(result.warnings.len() <= 50),
        }
    }
}

#[test]
fn test_error_tolerance_levels() {
    let original = create_test_hierarchy();
    let mut slightly_corrupted = original.clone();
    
    // Introduce minor inconsistency
    introduce_minor_corruption(&mut slightly_corrupted);
    
    let strict_validator = CompressionValidator::new(ValidationLevel::Standard);
    strict_validator.error_tolerance = ErrorTolerance::None;
    
    let lenient_validator = CompressionValidator::new(ValidationLevel::Standard);
    lenient_validator.error_tolerance = ErrorTolerance::Moderate;
    
    let operations = vec![];
    
    let strict_result = strict_validator.validate_compression(&original, &slightly_corrupted, &operations);
    let lenient_result = lenient_validator.validate_compression(&original, &slightly_corrupted, &operations);
    
    // Strict should fail, lenient should pass
    assert!(!strict_result.is_valid);
    assert!(lenient_result.is_valid);
    
    // But lenient should still report the issue as a warning
    assert!(!lenient_result.warnings.is_empty());
}
```

## File Location
`src/compression/validator.rs`

## Next Micro Phase
After completion, proceed to Micro 3.6: Compression Performance Tests