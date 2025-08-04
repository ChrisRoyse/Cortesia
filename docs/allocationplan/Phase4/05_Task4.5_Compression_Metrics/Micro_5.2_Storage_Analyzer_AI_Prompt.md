# AI Prompt: Micro Phase 5.2 - Storage Analyzer

You are tasked with implementing storage analysis for hierarchy memory usage. Create `src/metrics/storage.rs` with detailed storage analysis.

## Your Task
Implement the `StorageAnalyzer` struct that analyzes memory usage patterns, storage efficiency, and identifies optimization opportunities.

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;

pub struct StorageAnalyzer {
    analysis_depth: AnalysisDepth,
    memory_profiler: MemoryProfiler,
}

impl StorageAnalyzer {
    pub fn new() -> Self;
    pub fn analyze_memory_usage(&self, hierarchy: &InheritanceHierarchy) -> MemoryUsageReport;
    pub fn identify_memory_hotspots(&self, hierarchy: &InheritanceHierarchy) -> Vec<MemoryHotspot>;
    pub fn calculate_storage_efficiency(&self, hierarchy: &InheritanceHierarchy) -> StorageEfficiency;
    pub fn suggest_storage_optimizations(&self, hierarchy: &InheritanceHierarchy) -> Vec<StorageOptimization>;
}
```

## Success Criteria
- [ ] Accurately analyzes memory usage patterns
- [ ] Identifies memory hotspots and inefficiencies
- [ ] Suggests concrete storage optimizations
- [ ] Provides actionable storage recommendations

## File to Create: `src/metrics/storage.rs`
## When Complete: Respond with "MICRO PHASE 5.2 COMPLETE"