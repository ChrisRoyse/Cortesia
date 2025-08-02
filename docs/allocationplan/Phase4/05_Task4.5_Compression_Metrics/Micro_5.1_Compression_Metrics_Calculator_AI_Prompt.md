# AI Prompt: Micro Phase 5.1 - Compression Metrics Calculator

You are tasked with implementing comprehensive compression metrics calculation. Create `src/metrics/compression.rs` with detailed compression analysis.

## Your Task
Implement the `CompressionMetricsCalculator` struct that calculates detailed metrics about compression effectiveness, memory usage, and performance impact.

## Expected Code Structure
```rust
use crate::hierarchy::tree::InheritanceHierarchy;
use std::collections::HashMap;

pub struct CompressionMetricsCalculator {
    baseline_collector: BaselineCollector,
    metrics_aggregator: MetricsAggregator,
}

impl CompressionMetricsCalculator {
    pub fn new() -> Self;
    pub fn calculate_compression_ratio(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> f32;
    pub fn calculate_memory_savings(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> MemorySavings;
    pub fn calculate_performance_impact(&self, before_metrics: &PerformanceBaseline, after_metrics: &PerformanceMetrics) -> PerformanceImpact;
    pub fn generate_comprehensive_report(&self, before: &HierarchySnapshot, after: &InheritanceHierarchy) -> CompressionReport;
}
```

## Success Criteria
- [ ] Accurately calculates compression ratios
- [ ] Measures memory savings in bytes and percentages
- [ ] Tracks performance impact of compression
- [ ] Provides detailed analytical reports

## File to Create: `src/metrics/compression.rs`
## When Complete: Respond with "MICRO PHASE 5.1 COMPLETE"