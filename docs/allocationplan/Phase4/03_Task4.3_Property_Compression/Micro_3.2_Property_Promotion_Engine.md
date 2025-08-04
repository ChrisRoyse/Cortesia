# Micro Phase 3.2: Property Promotion Engine

**Estimated Time**: 50 minutes
**Dependencies**: Micro 3.1 (Property Analysis Engine)
**Objective**: Execute property promotions while maintaining semantic correctness and creating exceptions

## Task Description

Implement the core promotion engine that takes analysis results and safely promotes properties up the hierarchy, handling conflicts and creating exceptions where necessary.

## Deliverables

Create `src/compression/promoter.rs` with:

1. **PropertyPromoter struct**: Core promotion execution engine
2. **Safe promotion logic**: Validate promotions before execution
3. **Exception creation**: Generate exceptions for conflicting values
4. **Rollback capability**: Undo promotions if problems arise
5. **Batch processing**: Efficiently process multiple promotions
6. **Semantic validation**: Ensure no meaning is lost

## Success Criteria

- [ ] Promotes properties without losing semantic information
- [ ] Creates appropriate exceptions for all conflicts
- [ ] Rollback works correctly for failed promotions
- [ ] Batch processing is atomic (all succeed or all fail)
- [ ] Processes 1000 promotions in < 10ms
- [ ] Validation catches 100% of semantic violations

## Implementation Requirements

```rust
pub struct PropertyPromoter {
    validation_enabled: bool,
    auto_rollback: bool,
    max_exceptions_per_property: usize,
    promotion_strategy: PromotionStrategy,
}

#[derive(Debug, Clone)]
pub enum PromotionStrategy {
    Conservative,  // Only promote if 100% agreement
    Aggressive,    // Promote with majority agreement
    Balanced,      // Promote if >80% agreement
}

#[derive(Debug)]
pub struct PromotionResult {
    pub successful_promotions: usize,
    pub failed_promotions: usize,
    pub exceptions_created: usize,
    pub bytes_saved: usize,
    pub properties_removed: usize,
    pub rollbacks_performed: usize,
    pub execution_time: Duration,
}

impl PropertyPromoter {
    pub fn new(strategy: PromotionStrategy) -> Self;
    
    pub fn promote_properties(
        &self,
        hierarchy: &mut InheritanceHierarchy,
        candidates: Vec<PromotionCandidate>
    ) -> PromotionResult;
    
    pub fn promote_single_property(
        &self,
        hierarchy: &mut InheritanceHierarchy,
        candidate: &PromotionCandidate
    ) -> Result<SinglePromotionResult, PromotionError>;
    
    pub fn validate_promotion(
        &self,
        hierarchy: &InheritanceHierarchy,
        candidate: &PromotionCandidate
    ) -> ValidationResult;
    
    pub fn rollback_promotion(
        &self,
        hierarchy: &mut InheritanceHierarchy,
        promotion_record: &PromotionRecord
    ) -> Result<(), RollbackError>;
}

#[derive(Debug)]
struct PromotionRecord {
    target_node: NodeId,
    property_name: String,
    promoted_value: PropertyValue,
    removed_from_nodes: Vec<(NodeId, PropertyValue)>,
    exceptions_created: Vec<(NodeId, Exception)>,
    timestamp: Instant,
}

#[derive(Debug)]
pub struct SinglePromotionResult {
    pub bytes_saved: usize,
    pub nodes_affected: usize,
    pub exceptions_created: usize,
    pub promotion_record: PromotionRecord,
}
```

## Test Requirements

Must pass property promotion tests:
```rust
#[test]
fn test_successful_promotion() {
    let mut hierarchy = create_animal_hierarchy();
    let promoter = PropertyPromoter::new(PromotionStrategy::Balanced);
    
    // All dogs are loyal
    let candidate = PromotionCandidate {
        property_name: "loyal".to_string(),
        property_value: PropertyValue::Boolean(true),
        source_nodes: vec![
            hierarchy.get_node_by_name("Golden Retriever").unwrap(),
            hierarchy.get_node_by_name("Labrador").unwrap(),
            hierarchy.get_node_by_name("Beagle").unwrap(),
        ],
        target_ancestor: hierarchy.get_node_by_name("Dog").unwrap(),
        frequency_in_subtree: 1.0,
        estimated_bytes_saved: 150,
        confidence_score: 0.95,
        conflicts: vec![],
    };
    
    let result = promoter.promote_single_property(&mut hierarchy, &candidate).unwrap();
    
    // Verify promotion
    let dog_node = hierarchy.get_node_by_name("Dog").unwrap();
    assert_eq!(
        hierarchy.get_local_property(dog_node, "loyal"),
        Some(PropertyValue::Boolean(true))
    );
    
    // Verify removal from children
    let golden_retriever = hierarchy.get_node_by_name("Golden Retriever").unwrap();
    assert_eq!(hierarchy.get_local_property(golden_retriever, "loyal"), None);
    
    // But they should still inherit it
    assert_eq!(
        hierarchy.get_property(golden_retriever, "loyal"),
        Some(PropertyValue::Boolean(true))
    );
    
    assert!(result.bytes_saved > 0);
    assert_eq!(result.exceptions_created, 0);
}

#[test]
fn test_promotion_with_exceptions() {
    let mut hierarchy = create_bird_hierarchy();
    let promoter = PropertyPromoter::new(PromotionStrategy::Aggressive);
    
    // Most birds can fly, but penguin cannot
    let candidate = PromotionCandidate {
        property_name: "can_fly".to_string(),
        property_value: PropertyValue::Boolean(true),
        source_nodes: vec![
            hierarchy.get_node_by_name("Eagle").unwrap(),
            hierarchy.get_node_by_name("Sparrow").unwrap(),
            hierarchy.get_node_by_name("Penguin").unwrap(), // Exception!
        ],
        target_ancestor: hierarchy.get_node_by_name("Bird").unwrap(),
        frequency_in_subtree: 0.66, // 2/3 can fly
        estimated_bytes_saved: 120,
        confidence_score: 0.8,
        conflicts: vec![],
    };
    
    // Set penguin as non-flying
    let penguin = hierarchy.get_node_by_name("Penguin").unwrap();
    hierarchy.set_local_property(penguin, "can_fly", PropertyValue::Boolean(false));
    
    let result = promoter.promote_single_property(&mut hierarchy, &candidate).unwrap();
    
    // Verify promotion to Bird
    let bird_node = hierarchy.get_node_by_name("Bird").unwrap();
    assert_eq!(
        hierarchy.get_local_property(bird_node, "can_fly"),
        Some(PropertyValue::Boolean(true))
    );
    
    // Verify exception created for penguin
    assert_eq!(result.exceptions_created, 1);
    
    let penguin_exception = hierarchy.get_exception(penguin, "can_fly").unwrap();
    assert_eq!(penguin_exception.inherited_value, PropertyValue::Boolean(true));
    assert_eq!(penguin_exception.actual_value, PropertyValue::Boolean(false));
}

#[test]
fn test_promotion_validation() {
    let hierarchy = create_test_hierarchy();
    let promoter = PropertyPromoter::new(PromotionStrategy::Conservative);
    
    // Invalid candidate: trying to promote to non-ancestor
    let invalid_candidate = PromotionCandidate {
        property_name: "invalid".to_string(),
        property_value: PropertyValue::String("test".to_string()),
        source_nodes: vec![NodeId(1), NodeId(2)],
        target_ancestor: NodeId(999), // Non-existent node
        frequency_in_subtree: 1.0,
        estimated_bytes_saved: 100,
        confidence_score: 0.9,
        conflicts: vec![],
    };
    
    let validation = promoter.validate_promotion(&hierarchy, &invalid_candidate);
    assert!(!validation.is_valid);
    assert!(!validation.errors.is_empty());
}

#[test]
fn test_batch_promotion_atomicity() {
    let mut hierarchy = create_large_hierarchy(1000);
    let promoter = PropertyPromoter::new(PromotionStrategy::Balanced);
    
    let candidates = vec![
        create_valid_candidate(&hierarchy, "prop1"),
        create_invalid_candidate(), // This one will fail
        create_valid_candidate(&hierarchy, "prop2"),
    ];
    
    let result = promoter.promote_properties(&mut hierarchy, candidates);
    
    // All should fail due to one invalid candidate (atomic operation)
    assert_eq!(result.successful_promotions, 0);
    assert_eq!(result.rollbacks_performed, 0); // Nothing to rollback
}

#[test]
fn test_promotion_performance() {
    let mut hierarchy = create_large_hierarchy(10000);
    let promoter = PropertyPromoter::new(PromotionStrategy::Balanced);
    
    let candidates: Vec<_> = (0..1000)
        .map(|i| create_promotion_candidate(&hierarchy, &format!("prop_{}", i)))
        .collect();
    
    let start = Instant::now();
    let result = promoter.promote_properties(&mut hierarchy, candidates);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(10)); // < 10ms for 1000 promotions
    assert!(result.successful_promotions > 900); // Most should succeed
}
```

## File Location
`src/compression/promoter.rs`

## Next Micro Phase
After completion, proceed to Micro 3.3: Compression Orchestrator