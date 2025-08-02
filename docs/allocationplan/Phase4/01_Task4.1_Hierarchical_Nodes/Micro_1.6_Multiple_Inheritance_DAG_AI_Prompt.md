# AI Prompt: Micro Phase 1.6 - Multiple Inheritance DAG Support

You are tasked with extending the hierarchy system to properly support multiple inheritance using Directed Acyclic Graphs (DAGs). Your goal is to create `src/hierarchy/dag.rs` with robust DAG validation, C3 linearization, and diamond problem resolution.

## Your Task
Implement the `DAGManager` struct that manages multiple inheritance scenarios, ensures DAG integrity, computes method resolution orders, and handles property conflicts deterministically.

## Specific Requirements
1. Create `src/hierarchy/dag.rs` with DAGManager for multiple inheritance support
2. Implement cycle detection and prevention when adding multiple parents
3. Add C3 linearization algorithm for consistent method resolution order
4. Handle diamond inheritance pattern with deterministic resolution
5. Implement topological sorting for stable property resolution order
6. Add conflict detection and resolution following established rules
7. Maintain performance characteristics (O(log n) property lookup)

## Expected Code Structure
You must implement these exact signatures:

```rust
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;

#[derive(Debug, PartialEq)]
pub enum DAGError {
    CycleDetected(Vec<NodeId>),
    InvalidNode(NodeId),
    C3LinearizationFailed,
    ConflictResolutionFailed,
}

impl std::fmt::Display for DAGError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Implement error display
    }
}

impl std::error::Error for DAGError {}

#[derive(Debug, Clone)]
pub struct PropertyConflict {
    pub property_name: String,
    pub conflicting_values: Vec<(NodeId, PropertyValue)>,
    pub resolution_strategy: ConflictResolutionStrategy,
}

#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    FirstFound,      // Use first occurrence in MRO
    LastFound,       // Use last occurrence in MRO
    Explicit,        // Require explicit resolution
    Merge,           // Attempt to merge values
}

#[derive(Debug)]
pub struct DAGValidationResult {
    pub is_valid: bool,
    pub cycles: Vec<Vec<NodeId>>,
    pub conflicts: Vec<PropertyConflict>,
    pub linearization_errors: Vec<NodeId>,
}

pub struct DAGManager {
    topology_cache: DashMap<NodeId, Vec<NodeId>>, // MRO cache
    conflict_resolver: ConflictResolver,
    cycle_detector: CycleDetector,
    c3_cache: DashMap<NodeId, Vec<NodeId>>, // C3 linearization cache
}

impl DAGManager {
    pub fn new() -> Self {
        Self {
            topology_cache: DashMap::new(),
            conflict_resolver: ConflictResolver::new(),
            cycle_detector: CycleDetector::new(),
            c3_cache: DashMap::new(),
        }
    }
    
    pub fn add_parent(
        &self, 
        hierarchy: &InheritanceHierarchy, 
        child: NodeId, 
        parent: NodeId
    ) -> Result<(), DAGError> {
        // Check if adding this parent would create a cycle
        if self.would_create_cycle(hierarchy, child, parent)? {
            return Err(DAGError::CycleDetected(self.find_cycle_path(hierarchy, child, parent)));
        }
        
        // Clear affected caches
        self.invalidate_caches_for_node(child);
        
        Ok(())
    }
    
    pub fn compute_mro(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Result<Vec<NodeId>, DAGError> {
        // Check cache first
        if let Some(cached_mro) = self.c3_cache.get(&node) {
            return Ok(cached_mro.clone());
        }
        
        let mro = self.c3_linearization(hierarchy, node)?;
        self.c3_cache.insert(node, mro.clone());
        Ok(mro)
    }
    
    pub fn detect_conflicts(
        &self, 
        hierarchy: &InheritanceHierarchy, 
        node: NodeId
    ) -> Vec<PropertyConflict> {
        let mut conflicts = Vec::new();
        let mro = match self.compute_mro(hierarchy, node) {
            Ok(mro) => mro,
            Err(_) => return conflicts,
        };
        
        // Collect all properties from MRO
        let mut property_sources: HashMap<String, Vec<(NodeId, PropertyValue)>> = HashMap::new();
        
        for &ancestor in &mro {
            if let Some(ancestor_node) = hierarchy.get_node(ancestor) {
                for (prop_name, prop_value) in &ancestor_node.local_properties {
                    property_sources
                        .entry(prop_name.clone())
                        .or_insert_with(Vec::new)
                        .push((ancestor, prop_value.clone()));
                }
            }
        }
        
        // Find conflicts (multiple different values for same property)
        for (property_name, sources) in property_sources {
            if sources.len() > 1 {
                let unique_values: HashSet<_> = sources.iter().map(|(_, v)| v).collect();
                if unique_values.len() > 1 {
                    conflicts.push(PropertyConflict {
                        property_name,
                        conflicting_values: sources,
                        resolution_strategy: ConflictResolutionStrategy::FirstFound,
                    });
                }
            }
        }
        
        conflicts
    }
    
    pub fn validate_dag(&self, hierarchy: &InheritanceHierarchy) -> DAGValidationResult {
        let mut result = DAGValidationResult {
            is_valid: true,
            cycles: Vec::new(),
            conflicts: Vec::new(),
            linearization_errors: Vec::new(),
        };
        
        // Check all nodes for cycles and conflicts
        for node_ref in hierarchy.get_all_nodes() {
            let node_id = node_ref.key().clone();
            
            // Check for cycles
            if let Some(cycle) = self.detect_cycle_from_node(hierarchy, node_id) {
                result.cycles.push(cycle);
                result.is_valid = false;
            }
            
            // Check for linearization errors
            if self.compute_mro(hierarchy, node_id).is_err() {
                result.linearization_errors.push(node_id);
                result.is_valid = false;
            }
            
            // Check for conflicts
            let conflicts = self.detect_conflicts(hierarchy, node_id);
            result.conflicts.extend(conflicts);
        }
        
        result
    }
    
    fn c3_linearization(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Result<Vec<NodeId>, DAGError> {
        // Implement C3 linearization algorithm
        // This is the same algorithm used in Python for method resolution order
        
        let mut result = vec![node];
        
        if let Some(node_ref) = hierarchy.get_node(node) {
            let parents = &node_ref.parents;
            
            if parents.is_empty() {
                return Ok(result);
            }
            
            // Get linearizations of all parents
            let mut parent_linearizations = Vec::new();
            for &parent in parents {
                let parent_mro = self.c3_linearization(hierarchy, parent)?;
                parent_linearizations.push(parent_mro);
            }
            
            // Add parent list itself
            parent_linearizations.push(parents.clone());
            
            // Merge linearizations using C3 algorithm
            let merged = self.merge_linearizations(parent_linearizations)?;
            result.extend(merged);
        }
        
        Ok(result)
    }
    
    fn merge_linearizations(&self, mut linearizations: Vec<Vec<NodeId>>) -> Result<Vec<NodeId>, DAGError> {
        let mut result = Vec::new();
        
        while !linearizations.is_empty() {
            linearizations.retain(|lin| !lin.is_empty());
            
            if linearizations.is_empty() {
                break;
            }
            
            // Find a good head (appears first in some list and doesn't appear in tail of any other)
            let mut good_head = None;
            
            for linearization in &linearizations {
                if let Some(&candidate) = linearization.first() {
                    let is_good = linearizations.iter().all(|other_lin| {
                        other_lin.is_empty() || 
                        other_lin[0] == candidate || 
                        !other_lin[1..].contains(&candidate)
                    });
                    
                    if is_good {
                        good_head = Some(candidate);
                        break;
                    }
                }
            }
            
            match good_head {
                Some(head) => {
                    result.push(head);
                    // Remove head from all linearizations
                    for linearization in &mut linearizations {
                        if let Some(pos) = linearization.iter().position(|&x| x == head) {
                            if pos == 0 {
                                linearization.remove(0);
                            }
                        }
                    }
                }
                None => {
                    return Err(DAGError::C3LinearizationFailed);
                }
            }
        }
        
        Ok(result)
    }
    
    fn would_create_cycle(&self, hierarchy: &InheritanceHierarchy, child: NodeId, parent: NodeId) -> Result<bool, DAGError> {
        // Use DFS to check if parent is reachable from child
        self.is_ancestor(hierarchy, parent, child)
    }
    
    fn is_ancestor(&self, hierarchy: &InheritanceHierarchy, ancestor: NodeId, descendant: NodeId) -> Result<bool, DAGError> {
        if ancestor == descendant {
            return Ok(true);
        }
        
        let mut visited = HashSet::new();
        let mut stack = vec![descendant];
        
        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            
            if let Some(node_ref) = hierarchy.get_node(current) {
                for &parent in &node_ref.parents {
                    if parent == ancestor {
                        return Ok(true);
                    }
                    if !visited.contains(&parent) {
                        stack.push(parent);
                    }
                }
            }
        }
        
        Ok(false)
    }
    
    fn find_cycle_path(&self, hierarchy: &InheritanceHierarchy, start: NodeId, target: NodeId) -> Vec<NodeId> {
        // Find the actual cycle path for error reporting
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        
        if self.find_cycle_dfs(hierarchy, start, target, &mut path, &mut visited) {
            path
        } else {
            vec![]
        }
    }
    
    fn find_cycle_dfs(
        &self,
        hierarchy: &InheritanceHierarchy,
        current: NodeId,
        target: NodeId,
        path: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>
    ) -> bool {
        if current == target && !path.is_empty() {
            path.push(current);
            return true;
        }
        
        if visited.contains(&current) {
            return false;
        }
        
        visited.insert(current);
        path.push(current);
        
        if let Some(node_ref) = hierarchy.get_node(current) {
            for &parent in &node_ref.parents {
                if self.find_cycle_dfs(hierarchy, parent, target, path, visited) {
                    return true;
                }
            }
        }
        
        path.pop();
        false
    }
    
    fn detect_cycle_from_node(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> Option<Vec<NodeId>> {
        // Detect any cycle starting from this node
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        
        if self.has_cycle_dfs(hierarchy, node, &mut visited, &mut path) {
            Some(path)
        } else {
            None
        }
    }
    
    fn has_cycle_dfs(
        &self,
        hierarchy: &InheritanceHierarchy,
        current: NodeId,
        visited: &mut HashSet<NodeId>,
        path: &mut Vec<NodeId>
    ) -> bool {
        if path.contains(&current) {
            // Found cycle
            let cycle_start = path.iter().position(|&n| n == current).unwrap();
            path.truncate(cycle_start);
            path.push(current);
            return true;
        }
        
        if visited.contains(&current) {
            return false;
        }
        
        visited.insert(current);
        path.push(current);
        
        if let Some(node_ref) = hierarchy.get_node(current) {
            for &parent in &node_ref.parents {
                if self.has_cycle_dfs(hierarchy, parent, visited, path) {
                    return true;
                }
            }
        }
        
        path.pop();
        false
    }
    
    fn invalidate_caches_for_node(&self, node: NodeId) {
        self.topology_cache.remove(&node);
        self.c3_cache.remove(&node);
        
        // Also invalidate caches for descendants since their MRO might change
        // This would require walking the hierarchy, simplified for now
    }
}

pub struct ConflictResolver {
    default_strategy: ConflictResolutionStrategy,
}

impl ConflictResolver {
    fn new() -> Self {
        Self {
            default_strategy: ConflictResolutionStrategy::FirstFound,
        }
    }
    
    pub fn resolve_conflict(
        &self,
        conflict: &PropertyConflict,
        mro: &[NodeId]
    ) -> Option<PropertyValue> {
        match conflict.resolution_strategy {
            ConflictResolutionStrategy::FirstFound => {
                // Use the first occurrence in MRO order
                for &node in mro {
                    if let Some((_, value)) = conflict.conflicting_values.iter()
                        .find(|(source, _)| *source == node) {
                        return Some(value.clone());
                    }
                }
                None
            }
            ConflictResolutionStrategy::LastFound => {
                // Use the last occurrence in MRO order
                for &node in mro.iter().rev() {
                    if let Some((_, value)) = conflict.conflicting_values.iter()
                        .find(|(source, _)| *source == node) {
                        return Some(value.clone());
                    }
                }
                None
            }
            ConflictResolutionStrategy::Explicit => {
                // Require explicit resolution - return None to indicate this
                None
            }
            ConflictResolutionStrategy::Merge => {
                // Attempt to merge values (implementation depends on value type)
                self.attempt_merge(&conflict.conflicting_values)
            }
        }
    }
    
    fn attempt_merge(&self, values: &[(NodeId, PropertyValue)]) -> Option<PropertyValue> {
        // Simple merge strategy - could be made more sophisticated
        if values.is_empty() {
            return None;
        }
        
        // For now, just return the first value
        // In a real implementation, this could merge arrays, combine strings, etc.
        Some(values[0].1.clone())
    }
}

struct CycleDetector {
    // Could add more sophisticated cycle detection state here
}

impl CycleDetector {
    fn new() -> Self {
        Self {}
    }
}

impl Default for DAGManager {
    fn default() -> Self {
        Self::new()
    }
}
```

## Success Criteria (You must verify these)
- [ ] Prevents cycle creation when adding multiple parents
- [ ] C3 linearization produces consistent method resolution order
- [ ] Diamond inheritance resolves deterministically
- [ ] Topological sort is stable and reproducible
- [ ] Conflict resolution follows established rules (first-found strategy)
- [ ] Performance remains reasonable for complex hierarchies
- [ ] Cache invalidation works correctly when DAG structure changes
- [ ] Error handling provides clear feedback for invalid operations
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_prevention() {
        let hierarchy = InheritanceHierarchy::new();
        let dag_manager = DAGManager::new();
        
        let a = hierarchy.create_node("A").unwrap();
        let b = hierarchy.create_child("B", a).unwrap();
        let c = hierarchy.create_child("C", b).unwrap();
        
        // Try to create cycle: A -> B -> C -> A
        let result = dag_manager.add_parent(&hierarchy, a, c);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DAGError::CycleDetected(_)));
    }

    #[test]
    fn test_diamond_inheritance() {
        let hierarchy = create_diamond_hierarchy();
        let dag_manager = DAGManager::new();
        
        // D inherits from both B and C, which both inherit from A
        let d = hierarchy.get_node_by_name("D").unwrap();
        
        let mro = dag_manager.compute_mro(&hierarchy, d).unwrap();
        
        // MRO should be deterministic and include all ancestors
        assert!(mro.contains(&d));
        assert!(mro.len() >= 4); // D, B, C, A at minimum
        
        // D should come first in its own MRO
        assert_eq!(mro[0], d);
    }

    #[test]
    fn test_c3_linearization() {
        let hierarchy = create_complex_hierarchy();
        let dag_manager = DAGManager::new();
        
        let node = hierarchy.get_node_by_name("ComplexNode").unwrap();
        let mro1 = dag_manager.compute_mro(&hierarchy, node).unwrap();
        let mro2 = dag_manager.compute_mro(&hierarchy, node).unwrap();
        
        // MRO should be consistent across calls
        assert_eq!(mro1, mro2);
        
        // Node should appear first in its own MRO
        assert_eq!(mro1[0], node);
    }

    #[test]
    fn test_conflict_detection() {
        let hierarchy = create_conflicting_hierarchy();
        let dag_manager = DAGManager::new();
        
        let child = hierarchy.get_node_by_name("Child").unwrap();
        let conflicts = dag_manager.detect_conflicts(&hierarchy, child);
        
        assert!(!conflicts.is_empty());
        
        // Should detect property conflicts from multiple parents
        let conflict = &conflicts[0];
        assert!(conflict.conflicting_values.len() > 1);
    }

    #[test]
    fn test_dag_validation() {
        let hierarchy = create_invalid_hierarchy();
        let dag_manager = DAGManager::new();
        
        let validation = dag_manager.validate_dag(&hierarchy);
        
        if !validation.is_valid {
            assert!(!validation.cycles.is_empty() || !validation.linearization_errors.is_empty());
        }
    }

    #[test]
    fn test_performance_with_deep_hierarchy() {
        let hierarchy = create_deep_multiple_inheritance_hierarchy(100);
        let dag_manager = DAGManager::new();
        
        let leaf = hierarchy.get_node_by_name("Leaf").unwrap();
        
        let start = std::time::Instant::now();
        let mro = dag_manager.compute_mro(&hierarchy, leaf);
        let elapsed = start.elapsed();
        
        assert!(mro.is_ok());
        assert!(elapsed < std::time::Duration::from_millis(100)); // Should be fast
    }

    fn create_diamond_hierarchy() -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        let a = hierarchy.create_node("A").unwrap();
        let b = hierarchy.create_child("B", a).unwrap();
        let c = hierarchy.create_child("C", a).unwrap();
        let d = hierarchy.create_node("D").unwrap();
        
        hierarchy.add_parent(d, b).unwrap();
        hierarchy.add_parent(d, c).unwrap();
        
        hierarchy
    }

    fn create_complex_hierarchy() -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        // Create a more complex inheritance pattern
        let base = hierarchy.create_node("Base").unwrap();
        let mixin1 = hierarchy.create_child("Mixin1", base).unwrap();
        let mixin2 = hierarchy.create_child("Mixin2", base).unwrap();
        let complex = hierarchy.create_node("ComplexNode").unwrap();
        
        hierarchy.add_parent(complex, mixin1).unwrap();
        hierarchy.add_parent(complex, mixin2).unwrap();
        
        hierarchy
    }

    fn create_conflicting_hierarchy() -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        let parent1 = hierarchy.create_node("Parent1").unwrap();
        let parent2 = hierarchy.create_node("Parent2").unwrap();
        let child = hierarchy.create_node("Child").unwrap();
        
        // Add conflicting properties
        if let Some(mut p1) = hierarchy.get_node(parent1) {
            p1.add_property("color".to_string(), PropertyValue::String("red".to_string()));
        }
        if let Some(mut p2) = hierarchy.get_node(parent2) {
            p2.add_property("color".to_string(), PropertyValue::String("blue".to_string()));
        }
        
        hierarchy.add_parent(child, parent1).unwrap();
        hierarchy.add_parent(child, parent2).unwrap();
        
        hierarchy
    }

    fn create_invalid_hierarchy() -> InheritanceHierarchy {
        // This would normally be prevented, but for testing purposes
        let hierarchy = InheritanceHierarchy::new();
        let a = hierarchy.create_node("A").unwrap();
        hierarchy
    }

    fn create_deep_multiple_inheritance_hierarchy(depth: usize) -> InheritanceHierarchy {
        let hierarchy = InheritanceHierarchy::new();
        
        let root = hierarchy.create_node("Root").unwrap();
        let mut current_level = vec![root];
        
        for level in 1..depth {
            let mut next_level = Vec::new();
            for i in 0..2 {
                let node = hierarchy.create_node(&format!("Node_{}_{}", level, i)).unwrap();
                // Add multiple parents from previous level
                for &parent in &current_level {
                    hierarchy.add_parent(node, parent).unwrap();
                }
                next_level.push(node);
            }
            current_level = next_level;
        }
        
        // Create leaf with multiple inheritance
        let leaf = hierarchy.create_node("Leaf").unwrap();
        for &parent in &current_level {
            hierarchy.add_parent(leaf, parent).unwrap();
        }
        
        hierarchy
    }
}
```

## File to Create
Create exactly this file: `src/hierarchy/dag.rs`

## When Complete
Respond with "MICRO PHASE 1.6 COMPLETE" and a brief summary of what you implemented, including:
- C3 linearization algorithm approach
- Cycle detection strategy
- Conflict resolution mechanisms
- Performance optimizations for complex hierarchies
- Confirmation that all tests pass