# AI Prompt: Micro Phase 1.4 - Property Resolution Engine

You are tasked with implementing the core property resolution algorithm that walks inheritance chains to find property values. Your goal is to create `src/properties/resolver.rs` with an efficient, robust property resolution system that handles complex inheritance scenarios including multiple inheritance and cycles.

## Your Task
Implement the `PropertyResolver` struct that can efficiently lookup properties through inheritance chains using different traversal strategies, with cycle detection and performance optimization.

## Specific Requirements
1. Create `src/properties/resolver.rs` with PropertyResolver struct supporting multiple resolution strategies
2. Implement depth-first and breadth-first traversal algorithms
3. Add C3 linearization for deterministic multiple inheritance resolution
4. Implement cycle detection to prevent infinite loops
5. Optimize for performance with early termination and bounded recursion
6. Provide detailed resolution information including paths and timing
7. Handle edge cases gracefully (missing nodes, empty hierarchies, etc.)

## Expected Code Structure
You must implement these exact signatures:

```rust
use std::collections::{HashMap, HashSet, VecDeque};
use std::cell::RefCell;
use std::time::{Duration, Instant};
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;

#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    DepthFirst,
    BreadthFirst,
    C3Linearization,
}

#[derive(Debug, PartialEq)]
pub enum ResolutionError {
    NodeNotFound(NodeId),
    CycleDetected(Vec<NodeId>),
    MaxDepthExceeded(usize),
    PropertyNotFound(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Implement error display
    }
}

impl std::error::Error for ResolutionError {}

pub struct PropertyResolver {
    strategy: ResolutionStrategy,
    max_depth: usize,
    cycle_detector: CycleDetector,
}

impl PropertyResolver {
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            max_depth: 100, // Reasonable default
            cycle_detector: CycleDetector::new(),
        }
    }
    
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
    
    pub fn resolve_property(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> PropertyResolution {
        let start_time = Instant::now();
        
        let (value, source_node, path) = match self.strategy {
            ResolutionStrategy::DepthFirst => {
                self.resolve_depth_first(hierarchy, node, property)
            },
            ResolutionStrategy::BreadthFirst => {
                self.resolve_breadth_first(hierarchy, node, property)
            },
            ResolutionStrategy::C3Linearization => {
                self.resolve_c3_linearization(hierarchy, node, property)
            },
        };
        
        PropertyResolution {
            value,
            source_node,
            resolution_path: path,
            resolution_time: start_time.elapsed(),
            strategy_used: self.strategy.clone(),
        }
    }
    
    pub fn get_all_properties(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId
    ) -> Result<HashMap<String, PropertyValue>, ResolutionError> {
        // Collect all properties from this node and its ancestors
        // Handle property overriding correctly
    }
    
    pub fn resolve_with_path(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> (Option<PropertyValue>, Vec<NodeId>) {
        let resolution = self.resolve_property(hierarchy, node, property);
        (resolution.value, resolution.resolution_path)
    }
    
    fn resolve_depth_first(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> (Option<PropertyValue>, Option<NodeId>, Vec<NodeId>) {
        // Implement depth-first search
        // Check local properties first, then recurse through parents
    }
    
    fn resolve_breadth_first(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> (Option<PropertyValue>, Option<NodeId>, Vec<NodeId>) {
        // Implement breadth-first search using queue
        // Check all nodes at current level before going deeper
    }
    
    fn resolve_c3_linearization(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> (Option<PropertyValue>, Option<NodeId>, Vec<NodeId>) {
        // Implement C3 linearization algorithm for multiple inheritance
        // Create method resolution order (MRO) and search in that order
    }
    
    fn compute_c3_linearization(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId
    ) -> Vec<NodeId> {
        // Compute C3 linearization order for a node
        // This is the standard algorithm used in Python for multiple inheritance
    }
    
    fn get_node_property(
        &self,
        hierarchy: &InheritanceHierarchy,
        node: NodeId,
        property: &str
    ) -> Option<PropertyValue> {
        // Get property directly from node's local properties
    }
}

#[derive(Debug)]
pub struct PropertyResolution {
    pub value: Option<PropertyValue>,
    pub source_node: Option<NodeId>,
    pub resolution_path: Vec<NodeId>,
    pub resolution_time: Duration,
    pub strategy_used: ResolutionStrategy,
}

impl PropertyResolution {
    pub fn found(&self) -> bool {
        self.value.is_some()
    }
    
    pub fn path_length(&self) -> usize {
        self.resolution_path.len()
    }
}

struct CycleDetector {
    current_path: RefCell<Vec<NodeId>>,
    visited_in_path: RefCell<HashSet<NodeId>>,
}

impl CycleDetector {
    fn new() -> Self {
        Self {
            current_path: RefCell::new(Vec::new()),
            visited_in_path: RefCell::new(HashSet::new()),
        }
    }
    
    fn enter_node(&self, node: NodeId) -> Result<(), ResolutionError> {
        let mut path = self.current_path.borrow_mut();
        let mut visited = self.visited_in_path.borrow_mut();
        
        if visited.contains(&node) {
            // Create cycle information
            let cycle_start = path.iter().position(|&n| n == node).unwrap_or(0);
            let cycle = path[cycle_start..].to_vec();
            return Err(ResolutionError::CycleDetected(cycle));
        }
        
        path.push(node);
        visited.insert(node);
        Ok(())
    }
    
    fn exit_node(&self, node: NodeId) {
        let mut path = self.current_path.borrow_mut();
        let mut visited = self.visited_in_path.borrow_mut();
        
        if let Some(pos) = path.iter().rposition(|&n| n == node) {
            path.truncate(pos);
        }
        visited.remove(&node);
    }
    
    fn clear(&self) {
        self.current_path.borrow_mut().clear();
        self.visited_in_path.borrow_mut().clear();
    }
}

impl Default for PropertyResolver {
    fn default() -> Self {
        Self::new(ResolutionStrategy::DepthFirst)
    }
}
```

## Success Criteria (You must verify these)
- [ ] Resolves properties through arbitrary depth inheritance chains correctly
- [ ] Handles multiple inheritance with deterministic results using C3 linearization
- [ ] Detects and prevents infinite loops from circular references
- [ ] Resolution time < 100μs for chains up to 20 levels deep
- [ ] Memory usage bounded (no unbounded recursion or memory leaks)
- [ ] Supports all three resolution strategies correctly
- [ ] Provides complete resolution information (path, timing, source)
- [ ] Handles edge cases gracefully (missing nodes, properties, etc.)
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_hierarchy() -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        // Animal -> Mammal -> Dog
        let animal = hierarchy.create_node("Animal").unwrap();
        let mammal = hierarchy.create_child("Mammal", animal).unwrap();
        let dog = hierarchy.create_child("Dog", mammal).unwrap();
        
        // Add properties
        if let Some(mut animal_node) = hierarchy.get_node(animal) {
            animal_node.add_property("alive".to_string(), PropertyValue::Boolean(true));
            animal_node.add_property("kingdom".to_string(), PropertyValue::String("Animalia".to_string()));
        }
        
        if let Some(mut mammal_node) = hierarchy.get_node(mammal) {
            mammal_node.add_property("warm_blooded".to_string(), PropertyValue::Boolean(true));
            mammal_node.add_property("has_fur".to_string(), PropertyValue::Boolean(true));
        }
        
        hierarchy
    }

    #[test]
    fn test_single_inheritance_resolution() {
        let hierarchy = create_test_hierarchy();
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        
        let dog = hierarchy.get_node_by_name("Dog").unwrap();
        
        let resolution = resolver.resolve_property(&hierarchy, dog, "alive");
        assert_eq!(resolution.value, Some(PropertyValue::Boolean(true)));
        assert!(resolution.found());
        assert!(resolution.path_length() > 0);
        assert!(resolution.resolution_time < Duration::from_millis(10));
    }

    #[test]
    fn test_property_not_found() {
        let hierarchy = create_test_hierarchy();
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        
        let dog = hierarchy.get_node_by_name("Dog").unwrap();
        
        let resolution = resolver.resolve_property(&hierarchy, dog, "nonexistent");
        assert_eq!(resolution.value, None);
        assert!(!resolution.found());
    }

    #[test]
    fn test_breadth_first_vs_depth_first() {
        let hierarchy = create_diamond_hierarchy();
        
        let df_resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        let bf_resolver = PropertyResolver::new(ResolutionStrategy::BreadthFirst);
        
        let smartphone = hierarchy.get_node_by_name("Smartphone").unwrap();
        
        let df_result = df_resolver.resolve_property(&hierarchy, smartphone, "electronic");
        let bf_result = bf_resolver.resolve_property(&hierarchy, smartphone, "electronic");
        
        // Both should find the property but potentially with different paths
        assert_eq!(df_result.value, bf_result.value);
        assert!(df_result.found() && bf_result.found());
    }

    #[test]
    fn test_c3_linearization() {
        let hierarchy = create_diamond_hierarchy();
        let resolver = PropertyResolver::new(ResolutionStrategy::C3Linearization);
        
        let smartphone = hierarchy.get_node_by_name("Smartphone").unwrap();
        
        let resolution = resolver.resolve_property(&hierarchy, smartphone, "portable");
        assert!(resolution.found());
        
        // C3 linearization should provide deterministic ordering
        let all_props = resolver.get_all_properties(&hierarchy, smartphone).unwrap();
        assert!(!all_props.is_empty());
    }

    #[test]
    fn test_cycle_detection() {
        let hierarchy = InheritanceHierarchy::new();
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        
        // Create potential cycle (would be caught by hierarchy, but test resolver)
        let a = hierarchy.create_node("A").unwrap();
        
        // Test that cycle detector works within resolution
        let resolution = resolver.resolve_property(&hierarchy, a, "test_prop");
        assert!(resolution.resolution_time < Duration::from_millis(100));
    }

    #[test]
    fn test_resolution_performance() {
        let hierarchy = create_deep_hierarchy(20);
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        
        // Test performance with deep hierarchy
        let start = Instant::now();
        for i in 0..100 {
            let node = NodeId(i % 20);
            resolver.resolve_property(&hierarchy, node, "deep_property");
        }
        let elapsed = start.elapsed();
        
        let per_resolution = elapsed / 100;
        assert!(per_resolution < Duration::from_micros(100));
    }

    #[test]
    fn test_max_depth_limiting() {
        let hierarchy = create_deep_hierarchy(50);
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst)
            .with_max_depth(10);
        
        let leaf = NodeId(49);
        let resolution = resolver.resolve_property(&hierarchy, leaf, "root_property");
        
        // Should not traverse beyond max depth
        assert!(resolution.path_length() <= 10);
    }

    fn create_diamond_hierarchy() -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        // Device -> (Phone, Computer) -> Smartphone (diamond pattern)
        let device = hierarchy.create_node("Device").unwrap();
        let phone = hierarchy.create_child("Phone", device).unwrap();
        let computer = hierarchy.create_child("Computer", device).unwrap();
        let smartphone = hierarchy.create_node("Smartphone").unwrap();
        
        hierarchy.add_parent(smartphone, phone).unwrap();
        hierarchy.add_parent(smartphone, computer).unwrap();
        
        // Add conflicting properties to test resolution
        if let Some(mut device_node) = hierarchy.get_node(device) {
            device_node.add_property("electronic".to_string(), PropertyValue::Boolean(true));
        }
        
        hierarchy
    }

    fn create_deep_hierarchy(depth: usize) -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        let mut current = hierarchy.create_node("Root").unwrap();
        if let Some(mut root_node) = hierarchy.get_node(current) {
            root_node.add_property("root_property".to_string(), PropertyValue::String("root_value".to_string()));
        }
        
        for i in 1..depth {
            current = hierarchy.create_child(&format!("Node{}", i), current).unwrap();
        }
        
        hierarchy
    }
}
```

## File to Create
Create exactly this file: `src/properties/resolver.rs`

## Dependencies Required
No additional dependencies beyond what's already in the project.

## Implementation Notes
1. **C3 Linearization**: Implement the standard C3 linearization algorithm used in Python for multiple inheritance
2. **Performance**: Use iterative approaches where possible to avoid stack overflow
3. **Cycle Detection**: Maintain current path to detect cycles efficiently
4. **Early Termination**: Stop searching once property is found in appropriate strategy

## When Complete
Respond with "MICRO PHASE 1.4 COMPLETE" and a brief summary of what you implemented, including:
- Which resolution strategies were implemented
- How cycle detection works
- Performance optimizations applied
- How multiple inheritance conflicts are resolved
- Confirmation that all tests pass