# Task 32a: Create Benchmark Setup

**Estimated Time**: 4 minutes  
**Dependencies**: 31p  
**Stage**: Performance Benchmarking  

## Objective
Create basic benchmarking infrastructure and setup utilities.

## Implementation Steps

1. Create `tests/benchmarks/mod.rs`:
```rust
use criterion::{Criterion, BatchSize};
use std::sync::Arc;
use tokio::runtime::Runtime;
use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::types::*;

pub async fn setup_benchmark_brain_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_benchmarks());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_benchmarks());
    let memory_pool = Arc::new(MemoryPool::new_for_benchmarks());
    
    Arc::new(
        BrainEnhancedGraphCore::new_optimized_for_performance(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create benchmark brain graph")
    )
}

pub fn create_benchmark_allocation_request(id: &str) -> MemoryAllocationRequest {
    MemoryAllocationRequest {
        concept_id: id.to_string(),
        concept_type: ConceptType::Semantic,
        content: format!("Benchmark content for {}", id),
        semantic_embedding: Some(generate_benchmark_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "benchmark_user".to_string(),
        request_id: format!("bench_req_{}", id),
        version_info: None,
    }
}

pub fn generate_benchmark_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 + 1.0) / (size as f32)).collect()
}

pub struct BenchmarkConfig {
    pub sample_size: usize,
    pub warmup_iterations: usize,
    pub measurement_time_ms: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sample_size: 100,
            warmup_iterations: 3,
            measurement_time_ms: 2000,
        }
    }
}
```

2. Add Criterion dependency to `Cargo.toml`:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
tokio-test = "0.4"
```

## Acceptance Criteria
- [ ] Benchmark setup module created
- [ ] Helper functions for test data generation
- [ ] Criterion configuration ready

## Success Metrics
- Setup completes in under 2 seconds
- Benchmark environment properly configured

## Next Task
32b_benchmark_memory_allocation.md