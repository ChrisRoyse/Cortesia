# AI Prompt: Micro Phase 2.5 - Exception Storage Optimization

You are tasked with implementing storage optimization for exceptions to minimize memory usage and improve access performance. Your goal is to create `src/exceptions/optimizer.rs` with compression and deduplication strategies.

## Your Task
Implement the `ExceptionOptimizer` struct that optimizes exception storage through compression, deduplication, and efficient indexing to reduce memory footprint.

## Specific Requirements
1. Create `src/exceptions/optimizer.rs` with ExceptionOptimizer struct
2. Implement exception deduplication to eliminate redundant data
3. Add compression for frequently occurring exception patterns
4. Optimize storage layout for better cache performance
5. Implement garbage collection for outdated exceptions
6. Provide metrics on storage efficiency improvements

## Expected Code Structure
```rust
use crate::exceptions::store::{Exception, ExceptionStore};
use crate::hierarchy::node::NodeId;
use std::collections::HashMap;

pub struct ExceptionOptimizer {
    compression_threshold: usize,
    deduplication_enabled: bool,
    gc_interval: std::time::Duration,
    last_optimization: std::time::Instant,
}

impl ExceptionOptimizer {
    pub fn new() -> Self;
    
    pub fn optimize_storage(&self, store: &mut ExceptionStore) -> OptimizationReport;
    
    pub fn deduplicate_exceptions(&self, store: &mut ExceptionStore) -> usize;
    
    pub fn compress_patterns(&self, store: &mut ExceptionStore) -> usize;
    
    pub fn garbage_collect(&self, store: &mut ExceptionStore) -> usize;
}

#[derive(Debug)]
pub struct OptimizationReport {
    pub bytes_saved: usize,
    pub exceptions_deduplicated: usize,
    pub patterns_compressed: usize,
    pub items_garbage_collected: usize,
    pub optimization_time: std::time::Duration,
}
```

## Success Criteria
- [ ] Reduces memory usage through effective compression
- [ ] Eliminates duplicate exceptions successfully
- [ ] Maintains access performance after optimization
- [ ] Provides meaningful optimization metrics

## Test Requirements
```rust
#[test]
fn test_storage_optimization() {
    let optimizer = ExceptionOptimizer::new();
    let mut store = ExceptionStore::new();
    
    // Add redundant exceptions
    for i in 0..100 {
        let exception = Exception::new(/* same exception pattern */);
        store.add_exception(NodeId(i), "test".to_string(), exception);
    }
    
    let report = optimizer.optimize_storage(&mut store);
    assert!(report.bytes_saved > 0);
    assert!(report.exceptions_deduplicated > 0);
}
```

## File to Create
Create exactly this file: `src/exceptions/optimizer.rs`

## When Complete
Respond with "MICRO PHASE 2.5 COMPLETE" and a brief summary of your implementation.