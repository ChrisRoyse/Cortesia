# Micro Phase 1.4: Property Resolution Engine

**Estimated Time**: 50 minutes
**Dependencies**: Micro 1.3 (Hierarchy Tree Structure)
**Objective**: Implement the core property resolution algorithm that walks inheritance chains

## Task Description

Create the property resolution system that can efficiently lookup properties through inheritance chains, handling both single and multiple inheritance scenarios.

## Deliverables

Create `src/properties/resolver.rs` with:

1. **PropertyResolver struct**: Core resolution engine
2. **Resolution strategies**: Depth-first and breadth-first traversal
3. **Cycle detection**: Prevent infinite loops in complex hierarchies
4. **Performance optimization**: Early termination and path caching
5. **Multiple inheritance handling**: Deterministic conflict resolution

## Success Criteria

- [ ] Resolves properties through arbitrary depth inheritance chains
- [ ] Handles multiple inheritance with deterministic results
- [ ] Detects and prevents infinite loops from circular references
- [ ] Resolution time < 100μs for chains up to 20 levels deep
- [ ] Memory usage bounded (no unbounded recursion)
- [ ] Supports both depth-first and breadth-first strategies

## Implementation Requirements

```rust
#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    DepthFirst,
    BreadthFirst,
    // C3 linearization for multiple inheritance
    C3Linearization,
}

pub struct PropertyResolver {
    strategy: ResolutionStrategy,
    max_depth: usize,
    cycle_detector: CycleDetector,
}

impl PropertyResolver {
    pub fn new(strategy: ResolutionStrategy) -> Self;
    
    pub fn resolve_property(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> PropertyResolution;
    
    pub fn get_all_properties(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId
    ) -> HashMap<String, PropertyValue>;
    
    pub fn resolve_with_path(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> (Option<PropertyValue>, Vec<NodeId>);
}

#[derive(Debug)]
pub struct PropertyResolution {
    pub value: Option<PropertyValue>,
    pub source_node: Option<NodeId>,
    pub resolution_path: Vec<NodeId>,
    pub resolution_time: Duration,
}

struct CycleDetector {
    visited: RefCell<HashSet<NodeId>>,
}
```

## Test Requirements

Must pass property resolution tests:
```rust
#[test]
fn test_single_inheritance_resolution() {
    let hierarchy = create_test_hierarchy();
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    
    // Animal -> Mammal -> Dog
    let dog = hierarchy.get_node_by_name("Dog").unwrap();
    
    let resolution = resolver.resolve_property(&hierarchy, dog, "alive");
    assert_eq!(resolution.value, Some(PropertyValue::Boolean(true)));
    assert_eq!(resolution.source_node, hierarchy.get_node_by_name("Animal"));
}

#[test]
fn test_multiple_inheritance_resolution() {
    let hierarchy = create_diamond_hierarchy();
    let resolver = PropertyResolver::new(ResolutionStrategy::C3Linearization);
    
    // Device -> (Phone, Computer) -> Smartphone
    let smartphone = hierarchy.get_node_by_name("Smartphone").unwrap();
    
    let resolution = resolver.resolve_property(&hierarchy, smartphone, "electronic");
    assert_eq!(resolution.value, Some(PropertyValue::Boolean(true)));
    
    // Should have deterministic resolution for conflicts
    let portable = resolver.resolve_property(&hierarchy, smartphone, "portable");
    assert!(portable.value.is_some());
    assert!(portable.resolution_path.len() <= 3);
}

#[test]
fn test_cycle_detection() {
    let mut hierarchy = InheritanceHierarchy::new();
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    
    // Create circular reference: A -> B -> C -> A
    let a = hierarchy.create_node("A");
    let b = hierarchy.create_child("B", a);
    let c = hierarchy.create_child("C", b);
    hierarchy.add_parent(a, c).unwrap(); // Creates cycle
    
    let resolution = resolver.resolve_property(&hierarchy, a, "test_prop");
    // Should complete without infinite loop
    assert!(resolution.resolution_time < Duration::from_millis(100));
}

#[test]
fn test_resolution_performance() {
    let hierarchy = create_deep_hierarchy(20); // 20 levels deep
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    let leaf = NodeId(19);
    
    let start = Instant::now();
    for _ in 0..1000 {
        resolver.resolve_property(&hierarchy, leaf, "root_property");
    }
    let elapsed = start.elapsed();
    
    let per_resolution = elapsed / 1000;
    assert!(per_resolution < Duration::from_micros(100));
}
```

## File Location
`src/properties/resolver.rs`

## Next Micro Phase
After completion, proceed to Micro 1.5: Property Cache System