# AI Prompt: Micro Phase 4.1 - Hierarchy Reorganizer

You are tasked with implementing the hierarchy reorganizer that optimizes inheritance structure. Create `src/optimization/reorganizer.rs` with intelligent hierarchy restructuring.

## Your Task
Implement the `HierarchyReorganizer` struct that analyzes and reorganizes inheritance structures for optimal performance and memory usage.

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::hierarchy::node::NodeId;

pub struct HierarchyReorganizer {
    optimization_strategy: OptimizationStrategy,
    performance_threshold: f32,
}

impl HierarchyReorganizer {
    pub fn new() -> Self;
    pub fn reorganize_hierarchy(&self, hierarchy: &mut InheritanceHierarchy) -> ReorganizationReport;
    pub fn analyze_structure(&self, hierarchy: &InheritanceHierarchy) -> StructureAnalysis;
    pub fn suggest_improvements(&self, hierarchy: &InheritanceHierarchy) -> Vec<ImprovementSuggestion>;
}
```

## Success Criteria
- [ ] Improves hierarchy performance through restructuring
- [ ] Maintains semantic correctness of inheritance
- [ ] Reduces average path length for property resolution
- [ ] Optimizes memory layout for better cache performance

## File to Create: `src/optimization/reorganizer.rs`
## When Complete: Respond with "MICRO PHASE 4.1 COMPLETE"