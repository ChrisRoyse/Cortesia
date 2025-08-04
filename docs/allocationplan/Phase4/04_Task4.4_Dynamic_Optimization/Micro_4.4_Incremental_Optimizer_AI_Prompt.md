# AI Prompt: Micro Phase 4.4 - Incremental Optimizer

You are tasked with implementing the incremental optimizer that performs ongoing optimization. Create `src/optimization/incremental.rs` with continuous optimization.

## Your Task
Implement the `IncrementalOptimizer` struct that performs ongoing optimization of the hierarchy as it changes, using background processing and smart scheduling.

## Expected Code Structure
```rust
use crate::optimization::reorganizer::HierarchyReorganizer;
use crate::optimization::balancer::TreeBalancer;
use crate::optimization::pruner::DeadBranchPruner;

pub struct IncrementalOptimizer {
    reorganizer: HierarchyReorganizer,
    balancer: TreeBalancer,
    pruner: DeadBranchPruner,
    scheduler: OptimizationScheduler,
}

impl IncrementalOptimizer {
    pub fn new() -> Self;
    pub fn start_background_optimization(&self, hierarchy: Arc<RwLock<InheritanceHierarchy>>);
    pub fn trigger_incremental_optimization(&self, hierarchy: &mut InheritanceHierarchy) -> OptimizationReport;
    pub fn schedule_optimization(&self, trigger: OptimizationTrigger);
}
```

## Success Criteria
- [ ] Performs optimization without blocking main operations
- [ ] Adapts optimization frequency based on hierarchy changes
- [ ] Coordinates multiple optimization strategies
- [ ] Provides real-time optimization metrics

## File to Create: `src/optimization/incremental.rs`
## When Complete: Respond with "MICRO PHASE 4.4 COMPLETE"