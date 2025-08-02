# AI Prompt: Micro Phase 3.4 - Iterative Compression Algorithm

You are tasked with implementing the iterative compression algorithm that repeatedly analyzes and compresses until optimal compression is achieved. Your goal is to create `src/compression/iterative.rs` with smart iteration control.

## Your Task
Implement the `IterativeCompressor` struct that performs multiple rounds of analysis and promotion to achieve optimal compression with convergence detection and adaptive strategies.

## Specific Requirements
1. Create `src/compression/iterative.rs` with IterativeCompressor struct
2. Implement multi-round compression with convergence detection
3. Add adaptive strategies that adjust based on previous results
4. Optimize iteration order for faster convergence
5. Prevent infinite loops with proper termination conditions
6. Track compression metrics across iterations

## Expected Code Structure
```rust
use crate::compression::orchestrator::{CompressionOrchestrator, CompressionReport};
use crate::hierarchy::tree::InheritanceHierarchy;

pub struct IterativeCompressor {
    max_iterations: usize,
    convergence_threshold: f32,
    adaptive_strategy: AdaptiveStrategy,
    orchestrator: CompressionOrchestrator,
}

#[derive(Debug)]
pub struct IterativeCompressionReport {
    pub iterations_performed: usize,
    pub convergence_achieved: bool,
    pub final_compression_ratio: f32,
    pub total_bytes_saved: usize,
    pub iteration_reports: Vec<CompressionReport>,
}

impl IterativeCompressor {
    pub fn new() -> Self;
    
    pub fn compress_until_convergence(&mut self, hierarchy: &mut InheritanceHierarchy) -> IterativeCompressionReport;
    
    pub fn compress_n_iterations(&mut self, hierarchy: &mut InheritanceHierarchy, n: usize) -> IterativeCompressionReport;
    
    pub fn detect_convergence(&self, reports: &[CompressionReport]) -> bool;
}
```

## Success Criteria
- [ ] Achieves optimal compression through iteration
- [ ] Detects convergence to prevent unnecessary work
- [ ] Adapts strategy based on compression progress
- [ ] Prevents infinite loops with proper termination

## File to Create
Create exactly this file: `src/compression/iterative.rs`

## When Complete
Respond with "MICRO PHASE 3.4 COMPLETE" and a brief summary of your implementation.