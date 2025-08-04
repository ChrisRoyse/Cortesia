# AI Prompt: Micro Phase 1.7 - Integration Tests

You are tasked with creating comprehensive integration tests for the complete hierarchical node system. Your goal is to create `tests/integration/task_4_1_hierarchy_nodes.rs` with end-to-end tests that verify all components work together correctly.

## Your Task
Implement comprehensive integration tests that verify the full Task 4.1 hierarchical node system including single inheritance, multiple inheritance, property resolution, caching, and DAG management.

## Specific Requirements
1. Create `tests/integration/task_4_1_hierarchy_nodes.rs` with full system integration tests
2. Test single inheritance workflows end-to-end
3. Test multiple inheritance scenarios including diamond patterns
4. Verify property resolution performance benchmarks
5. Test memory usage and ensure no leaks
6. Verify thread safety under concurrent access
7. Test cache effectiveness and invalidation
8. Test DAG validation and cycle prevention

## Expected Code Structure
You must implement these exact test functions:

```rust
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// Import all the modules we've built
use llmkg::hierarchy::tree::InheritanceHierarchy;
use llmkg::hierarchy::node::{NodeId, InheritanceNode};
use llmkg::properties::value::PropertyValue;
use llmkg::properties::resolver::{PropertyResolver, ResolutionStrategy};
use llmkg::properties::cache::PropertyCache;
use llmkg::hierarchy::dag::DAGManager;

#[test]
fn test_single_inheritance_full_workflow() {
    // Test complete single inheritance workflow
    // Create Animal -> Mammal -> Dog hierarchy
    // Add properties at each level
    // Test property resolution through the chain
    // Verify caching improves performance
}

#[test]
fn test_multiple_inheritance_diamond_pattern() {
    // Test diamond inheritance pattern
    // Device -> (Phone, Computer) -> Smartphone
    // Test property resolution with conflicts
    // Verify C3 linearization works correctly
    // Test conflict resolution strategies
}

#[test]
fn test_property_resolution_performance() {
    // Create deep hierarchy (20+ levels)
    // Measure property resolution time
    // Verify performance < 100μs per lookup
    // Test with and without caching
}

#[test]
fn test_cache_effectiveness() {
    // Test cache hit rates
    // Verify cache invalidation works
    // Test cache under various access patterns
    // Verify memory bounds are respected
}

#[test]
fn test_concurrent_access_safety() {
    // Test with 10+ concurrent threads
    // Verify no data races or deadlocks
    // Test concurrent reads and writes
    // Verify cache thread safety
}

#[test]
fn test_dag_cycle_prevention() {
    // Test DAG manager prevents cycles
    // Test various cycle scenarios
    // Verify error handling is correct
}

#[test]
fn test_memory_usage_stress() {
    // Create large hierarchies
    // Monitor memory usage
    // Verify no memory leaks
    // Test garbage collection
}

#[test]
fn test_real_world_animal_taxonomy() {
    // Create realistic animal taxonomy
    // Test complex inheritance patterns
    // Verify property inheritance works correctly
}

#[test]
fn test_real_world_software_class_hierarchy() {
    // Create software class hierarchy
    // Test method resolution order
    // Verify multiple inheritance scenarios
}

#[test]
fn test_edge_cases_and_error_handling() {
    // Test invalid node IDs
    // Test missing properties
    // Test empty hierarchies
    // Verify graceful error handling
}

#[test]
fn test_serialization_roundtrip() {
    // Test that hierarchies can be serialized and deserialized
    // Verify all data is preserved
    // Test cache state handling
}

#[test]
fn test_hierarchy_modification_operations() {
    // Test adding/removing nodes
    // Test adding/removing relationships
    // Verify cache invalidation on changes
    // Test DAG consistency after changes
}
```

## Detailed Test Implementation

### Single Inheritance Test
```rust
#[test]
fn test_single_inheritance_full_workflow() {
    let hierarchy = InheritanceHierarchy::new();
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    let cache = PropertyCache::new(1000, Duration::from_secs(60));
    
    // Build Animal -> Mammal -> Dog -> Golden Retriever hierarchy
    let animal = hierarchy.create_node("Animal").unwrap();
    let mammal = hierarchy.create_child("Mammal", animal).unwrap();
    let dog = hierarchy.create_child("Dog", mammal).unwrap();
    let golden = hierarchy.create_child("Golden Retriever", dog).unwrap();
    
    // Add properties at each level
    if let Some(mut animal_node) = hierarchy.get_node(animal) {
        animal_node.add_property("alive".to_string(), PropertyValue::Boolean(true));
        animal_node.add_property("kingdom".to_string(), PropertyValue::String("Animalia".to_string()));
    }
    
    if let Some(mut mammal_node) = hierarchy.get_node(mammal) {
        mammal_node.add_property("warm_blooded".to_string(), PropertyValue::Boolean(true));
        mammal_node.add_property("has_fur".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut dog_node) = hierarchy.get_node(dog) {
        dog_node.add_property("domesticated".to_string(), PropertyValue::Boolean(true));
        dog_node.add_property("barks".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut golden_node) = hierarchy.get_node(golden) {
        golden_node.add_property("retrieves".to_string(), PropertyValue::Boolean(true));
        golden_node.add_property("color".to_string(), PropertyValue::String("golden".to_string()));
    }
    
    // Test property resolution
    let alive_resolution = resolver.resolve_property(&hierarchy, golden, "alive");
    assert_eq!(alive_resolution.value, Some(PropertyValue::Boolean(true)));
    assert_eq!(alive_resolution.source_node, Some(animal));
    
    let retrieves_resolution = resolver.resolve_property(&hierarchy, golden, "retrieves");
    assert_eq!(retrieves_resolution.value, Some(PropertyValue::Boolean(true)));
    assert_eq!(retrieves_resolution.source_node, Some(golden));
    
    // Test caching performance improvement
    let start = Instant::now();
    for _ in 0..100 {
        resolver.resolve_property(&hierarchy, golden, "alive");
    }
    let uncached_time = start.elapsed();
    
    // Cache the result
    cache.insert(golden, "alive", Some(PropertyValue::Boolean(true)), Some(animal));
    
    let start = Instant::now();
    for _ in 0..100 {
        cache.get(golden, "alive");
    }
    let cached_time = start.elapsed();
    
    assert!(cached_time < uncached_time / 5); // Should be at least 5x faster
    assert!(cache.hit_rate() > 0.9); // Should have high hit rate
}
```

### Multiple Inheritance Test
```rust
#[test]
fn test_multiple_inheritance_diamond_pattern() {
    let hierarchy = InheritanceHierarchy::new();
    let resolver = PropertyResolver::new(ResolutionStrategy::C3Linearization);
    let dag_manager = DAGManager::new();
    
    // Create diamond pattern: Device -> (Phone, Computer) -> Smartphone
    let device = hierarchy.create_node("Device").unwrap();
    let phone = hierarchy.create_child("Phone", device).unwrap();
    let computer = hierarchy.create_child("Computer", device).unwrap();
    let smartphone = hierarchy.create_node("Smartphone").unwrap();
    
    // Add smartphone as child of both phone and computer
    dag_manager.add_parent(&hierarchy, smartphone, phone).unwrap();
    dag_manager.add_parent(&hierarchy, smartphone, computer).unwrap();
    
    // Add properties with potential conflicts
    if let Some(mut device_node) = hierarchy.get_node(device) {
        device_node.add_property("electronic".to_string(), PropertyValue::Boolean(true));
        device_node.add_property("portable".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut phone_node) = hierarchy.get_node(phone) {
        phone_node.add_property("makes_calls".to_string(), PropertyValue::Boolean(true));
        phone_node.add_property("screen_size".to_string(), PropertyValue::Float(5.0));
    }
    
    if let Some(mut computer_node) = hierarchy.get_node(computer) {
        computer_node.add_property("runs_software".to_string(), PropertyValue::Boolean(true));
        computer_node.add_property("screen_size".to_string(), PropertyValue::Float(15.0)); // Conflict!
    }
    
    // Test MRO computation
    let mro = dag_manager.compute_mro(&hierarchy, smartphone).unwrap();
    assert!(mro.len() >= 4); // smartphone, phone, computer, device (at minimum)
    assert_eq!(mro[0], smartphone); // smartphone should be first
    
    // Test property resolution
    let electronic = resolver.resolve_property(&hierarchy, smartphone, "electronic");
    assert_eq!(electronic.value, Some(PropertyValue::Boolean(true)));
    
    // Test conflict detection
    let conflicts = dag_manager.detect_conflicts(&hierarchy, smartphone);
    assert!(!conflicts.is_empty());
    
    let screen_conflict = conflicts.iter()
        .find(|c| c.property_name == "screen_size")
        .expect("Should find screen_size conflict");
    
    assert!(screen_conflict.conflicting_values.len() > 1);
    
    // Test deterministic resolution
    let screen_resolution1 = resolver.resolve_property(&hierarchy, smartphone, "screen_size");
    let screen_resolution2 = resolver.resolve_property(&hierarchy, smartphone, "screen_size");
    assert_eq!(screen_resolution1.value, screen_resolution2.value); // Should be deterministic
}
```

## Success Criteria (You must verify these)
- [ ] All single inheritance scenarios pass end-to-end
- [ ] Multiple inheritance with conflicts resolves correctly and deterministically
- [ ] Property lookup performance < 100μs for 20-level depth hierarchies
- [ ] Cache hit rate > 80% for repeated lookups in realistic scenarios
- [ ] No memory leaks under stress testing with large hierarchies
- [ ] Thread-safe operation under 10+ concurrent threads
- [ ] DAG cycle prevention works correctly
- [ ] Cache invalidation maintains consistency
- [ ] Error handling provides meaningful feedback
- [ ] All tests pass consistently

## File to Create
Create exactly this file: `tests/integration/task_4_1_hierarchy_nodes.rs`

## When Complete
Respond with "MICRO PHASE 1.7 COMPLETE" and a brief summary of what you implemented, including:
- Which integration scenarios were tested
- Performance benchmarks achieved
- Thread safety verification results
- Memory usage characteristics observed
- Confirmation that all tests pass