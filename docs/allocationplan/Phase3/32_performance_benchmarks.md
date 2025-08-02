# Task 32: Performance Benchmarks

**Estimated Time**: 15-20 minutes  
**Dependencies**: 31_phase2_integration_tests.md  
**Stage**: Integration & Testing  

## Objective
Execute comprehensive performance benchmarks across all Phase 3 knowledge graph components to validate performance requirements, identify bottlenecks, establish baseline metrics, and ensure production readiness under various load conditions.

## Specific Requirements

### 1. Core Operation Benchmarks
- Measure memory allocation performance across different concept types
- Benchmark search operations (semantic, TTFS, spreading activation, hierarchical)
- Test inheritance resolution performance with varying chain depths
- Validate query optimization effectiveness for complex graph operations

### 2. Scalability Testing
- Test performance with increasing graph size (1K, 10K, 100K, 1M nodes)
- Measure concurrent operation performance (10, 100, 1000 simultaneous operations)
- Benchmark memory usage growth patterns under load
- Test cache effectiveness and hit rates under various access patterns

### 3. Integration Performance
- Measure end-to-end operation latency including Phase 2 integration
- Test API endpoint response times under load
- Benchmark database connection pooling effectiveness
- Validate temporal versioning performance impact

## Implementation Steps

### 1. Create Core Performance Benchmark Suite
```rust
// tests/benchmarks/core_performance_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use std::sync::Arc;
use tokio::runtime::Runtime;

use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::types::*;

fn benchmark_memory_allocation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let mut group = c.benchmark_group("memory_allocation");
    group.sample_size(1000);
    
    // Benchmark different concept types
    group.bench_function("episodic_allocation", |b| {
        b.to_async(&rt).iter_batched(
            || generate_episodic_allocation_request(),
            |request| async {
                black_box(
                    brain_graph
                        .allocate_memory_with_cortical_coordination(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("semantic_allocation", |b| {
        b.to_async(&rt).iter_batched(
            || generate_semantic_allocation_request(),
            |request| async {
                black_box(
                    brain_graph
                        .allocate_memory_with_cortical_coordination(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("abstract_allocation", |b| {
        b.to_async(&rt).iter_batched(
            || generate_abstract_allocation_request(),
            |request| async {
                black_box(
                    brain_graph
                        .allocate_memory_with_cortical_coordination(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.finish();
}

fn benchmark_search_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        // Populate with test data
        populate_graph_for_search_benchmarks(&graph).await;
        graph
    });
    
    let mut group = c.benchmark_group("search_operations");
    group.sample_size(500);
    
    group.bench_function("semantic_search", |b| {
        b.to_async(&rt).iter_batched(
            || SearchRequest {
                query_text: "artificial intelligence machine learning".to_string(),
                search_type: SearchType::Semantic,
                similarity_threshold: Some(0.8),
                limit: Some(10),
                user_context: UserContext::default(),
                use_ttfs_encoding: false,
                cortical_area_filter: None,
            },
            |request| async {
                black_box(
                    brain_graph
                        .search_memory_with_semantic_similarity(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("ttfs_search", |b| {
        b.to_async(&rt).iter_batched(
            || SearchRequest {
                query_text: "neural network processing".to_string(),
                search_type: SearchType::TTFS,
                similarity_threshold: Some(0.7),
                limit: Some(10),
                user_context: UserContext::default(),
                use_ttfs_encoding: true,
                cortical_area_filter: Some(vec!["computational_cortex".to_string()]),
            },
            |request| async {
                black_box(
                    brain_graph
                        .search_memory_with_ttfs(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("spreading_activation", |b| {
        b.to_async(&rt).iter_batched(
            || SpreadingActivationRequest {
                seed_concept_ids: vec!["concept_1".to_string(), "concept_5".to_string()],
                activation_threshold: 0.5,
                max_hops: 3,
                decay_factor: 0.8,
                max_results: 20,
            },
            |request| async {
                black_box(
                    brain_graph
                        .perform_spreading_activation_search(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("hierarchical_search", |b| {
        b.to_async(&rt).iter_batched(
            || HierarchicalSearchRequest {
                concept_id: "root_concept".to_string(),
                search_direction: SearchDirection::Descendants,
                max_depth: 5,
                include_properties: true,
                filter_criteria: None,
            },
            |request| async {
                black_box(
                    brain_graph
                        .perform_hierarchical_search(request)
                        .await
                        .unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });
    
    group.finish();
}

fn benchmark_inheritance_resolution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        // Create inheritance hierarchy for benchmarking
        create_inheritance_hierarchy_for_benchmarks(&graph).await;
        graph
    });
    
    let mut group = c.benchmark_group("inheritance_resolution");
    group.sample_size(300);
    
    // Test different inheritance chain depths
    for depth in [1, 3, 5, 10].iter() {
        group.bench_with_input(
            format!("resolve_properties_depth_{}", depth),
            depth,
            |b, &depth| {
                b.to_async(&rt).iter_batched(
                    || format!("concept_depth_{}", depth),
                    |concept_id| async {
                        black_box(
                            brain_graph
                                .resolve_inherited_properties(&concept_id, true)
                                .await
                                .unwrap()
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let mut group = c.benchmark_group("concurrent_operations");
    group.sample_size(100);
    
    // Test different concurrency levels
    for concurrent_ops in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            format!("concurrent_allocations_{}", concurrent_ops),
            concurrent_ops,
            |b, &concurrent_ops| {
                b.to_async(&rt).iter_batched(
                    || {
                        (0..concurrent_ops)
                            .map(|i| generate_allocation_request_with_id(i))
                            .collect::<Vec<_>>()
                    },
                    |requests| async {
                        let tasks: Vec<_> = requests
                            .into_iter()
                            .map(|request| {
                                let graph = brain_graph.clone();
                                tokio::spawn(async move {
                                    graph
                                        .allocate_memory_with_cortical_coordination(request)
                                        .await
                                        .unwrap()
                                })
                            })
                            .collect();
                        
                        let results = futures::future::join_all(tasks).await;
                        black_box(results)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

fn benchmark_graph_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("graph_scalability");
    group.sample_size(50);
    
    // Test with different graph sizes
    for graph_size in [1000, 10000, 100000].iter() {
        let brain_graph = rt.block_on(async {
            let graph = setup_benchmark_brain_graph().await;
            populate_graph_with_size(&graph, *graph_size).await;
            graph
        });
        
        group.bench_with_input(
            format!("search_in_graph_size_{}", graph_size),
            graph_size,
            |b, _| {
                b.to_async(&rt).iter_batched(
                    || SearchRequest {
                        query_text: "benchmark query".to_string(),
                        search_type: SearchType::Semantic,
                        similarity_threshold: Some(0.8),
                        limit: Some(10),
                        user_context: UserContext::default(),
                        use_ttfs_encoding: false,
                        cortical_area_filter: None,
                    },
                    |request| async {
                        black_box(
                            brain_graph
                                .search_memory_with_semantic_similarity(request)
                                .await
                                .unwrap()
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

// Helper functions
async fn setup_benchmark_brain_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_benchmarks());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_benchmarks());
    let memory_pool = Arc::new(MemoryPool::new_for_benchmarks());
    
    Arc::new(
        BrainEnhancedGraphCore::new_with_phase2_integration(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create benchmark brain graph")
    )
}

async fn populate_graph_for_search_benchmarks(graph: &BrainEnhancedGraphCore) {
    // Populate with realistic test data for search benchmarks
    let concepts = vec![
        ("concept_1", "artificial intelligence machine learning", ConceptType::Abstract),
        ("concept_2", "neural networks deep learning", ConceptType::Semantic),
        ("concept_3", "natural language processing", ConceptType::Specific),
        ("concept_4", "computer vision image recognition", ConceptType::Semantic),
        ("concept_5", "reinforcement learning algorithms", ConceptType::Abstract),
        // Add more test concepts...
    ];
    
    for (id, content, concept_type) in concepts {
        let request = MemoryAllocationRequest {
            concept_id: id.to_string(),
            concept_type,
            content: content.to_string(),
            semantic_embedding: Some(generate_realistic_embedding()),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "benchmark_user".to_string(),
            request_id: format!("benchmark_req_{}", id),
            version_info: None,
        };
        
        graph
            .allocate_memory_with_cortical_coordination(request)
            .await
            .expect("Failed to populate benchmark data");
    }
}

async fn create_inheritance_hierarchy_for_benchmarks(graph: &BrainEnhancedGraphCore) {
    // Create hierarchical relationships for inheritance benchmarks
    for depth in 1..=10 {
        let parent_id = if depth == 1 {
            "root_concept".to_string()
        } else {
            format!("concept_depth_{}", depth - 1)
        };
        
        let child_id = format!("concept_depth_{}", depth);
        
        // Allocate child concept
        let request = MemoryAllocationRequest {
            concept_id: child_id.clone(),
            concept_type: ConceptType::Specific,
            content: format!("Concept at depth {}", depth),
            semantic_embedding: Some(generate_realistic_embedding()),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "benchmark_user".to_string(),
            request_id: format!("benchmark_hierarchy_{}", depth),
            version_info: None,
        };
        
        graph
            .allocate_memory_with_cortical_coordination(request)
            .await
            .expect("Failed to create hierarchy concept");
        
        // Create inheritance relationship
        if depth > 1 {
            graph
                .create_inheritance_relationship(
                    &parent_id,
                    &child_id,
                    InheritanceType::DirectSubclass,
                )
                .await
                .expect("Failed to create inheritance relationship");
        }
    }
}

criterion_group!(
    benches,
    benchmark_memory_allocation,
    benchmark_search_operations,
    benchmark_inheritance_resolution,
    benchmark_concurrent_operations,
    benchmark_graph_scalability
);
criterion_main!(benches);
```

### 2. Create System Performance Monitor
```rust
// tests/benchmarks/system_performance_monitor.rs
use std::time::{Duration, Instant};
use tokio::time::interval;
use sysinfo::{System, SystemExt, ProcessExt};

pub struct SystemPerformanceMonitor {
    start_time: Instant,
    system: System,
    metrics_history: Vec<SystemMetrics>,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub timestamp: Instant,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub memory_total_mb: u64,
    pub active_allocations: usize,
    pub cache_hit_rate: f64,
    pub avg_response_time_ms: f64,
    pub operations_per_second: f64,
}

impl SystemPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            system: System::new_all(),
            metrics_history: Vec::new(),
        }
    }
    
    pub async fn start_monitoring(&mut self, brain_graph: Arc<BrainEnhancedGraphCore>) {
        let mut interval = interval(Duration::from_millis(100));
        
        loop {
            interval.tick().await;
            
            self.system.refresh_all();
            
            let metrics = SystemMetrics {
                timestamp: Instant::now(),
                cpu_usage_percent: self.system.global_processor_info().cpu_usage(),
                memory_usage_mb: self.get_process_memory_usage(),
                memory_total_mb: self.system.total_memory() / 1024 / 1024,
                active_allocations: brain_graph.get_active_allocation_count().await,
                cache_hit_rate: brain_graph.get_cache_hit_rate().await,
                avg_response_time_ms: brain_graph.get_avg_response_time_ms().await,
                operations_per_second: brain_graph.get_operations_per_second().await,
            };
            
            self.metrics_history.push(metrics);
            
            // Log critical performance thresholds
            if metrics.cpu_usage_percent > 80.0 {
                println!("⚠️ High CPU usage detected: {:.1}%", metrics.cpu_usage_percent);
            }
            
            if metrics.memory_usage_mb > 1024 {
                println!("⚠️ High memory usage detected: {} MB", metrics.memory_usage_mb);
            }
            
            if metrics.avg_response_time_ms > 50.0 {
                println!("⚠️ High response time detected: {:.1} ms", metrics.avg_response_time_ms);
            }
        }
    }
    
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let total_runtime = self.start_time.elapsed();
        
        let avg_cpu = self.metrics_history.iter()
            .map(|m| m.cpu_usage_percent)
            .sum::<f32>() / self.metrics_history.len() as f32;
        
        let peak_memory = self.metrics_history.iter()
            .map(|m| m.memory_usage_mb)
            .max()
            .unwrap_or(0);
        
        let avg_response_time = self.metrics_history.iter()
            .map(|m| m.avg_response_time_ms)
            .sum::<f64>() / self.metrics_history.len() as f64;
        
        let avg_ops_per_sec = self.metrics_history.iter()
            .map(|m| m.operations_per_second)
            .sum::<f64>() / self.metrics_history.len() as f64;
        
        PerformanceReport {
            total_runtime,
            avg_cpu_usage: avg_cpu,
            peak_memory_usage_mb: peak_memory,
            avg_response_time_ms: avg_response_time,
            avg_operations_per_second: avg_ops_per_sec,
            metrics_count: self.metrics_history.len(),
        }
    }
    
    fn get_process_memory_usage(&self) -> u64 {
        if let Some(process) = self.system.process(sysinfo::get_current_pid().unwrap()) {
            process.memory() / 1024 / 1024 // Convert to MB
        } else {
            0
        }
    }
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub total_runtime: Duration,
    pub avg_cpu_usage: f32,
    pub peak_memory_usage_mb: u64,
    pub avg_response_time_ms: f64,
    pub avg_operations_per_second: f64,
    pub metrics_count: usize,
}

impl PerformanceReport {
    pub fn meets_performance_requirements(&self) -> bool {
        self.avg_response_time_ms < 10.0
            && self.avg_operations_per_second > 1000.0
            && self.peak_memory_usage_mb < 2048
            && self.avg_cpu_usage < 70.0
    }
    
    pub fn print_summary(&self) {
        println!("\n📊 Performance Benchmark Report");
        println!("================================");
        println!("Total Runtime: {:.2}s", self.total_runtime.as_secs_f64());
        println!("Average CPU Usage: {:.1}%", self.avg_cpu_usage);
        println!("Peak Memory Usage: {} MB", self.peak_memory_usage_mb);
        println!("Average Response Time: {:.2} ms", self.avg_response_time_ms);
        println!("Average Operations/Second: {:.0}", self.avg_operations_per_second);
        println!("Metrics Collected: {}", self.metrics_count);
        
        if self.meets_performance_requirements() {
            println!("✅ All performance requirements met!");
        } else {
            println!("❌ Some performance requirements not met");
        }
    }
}
```

### 3. Create Benchmark Runner
```rust
// tests/benchmarks/benchmark_runner.rs
use std::sync::Arc;
use tokio::task::JoinHandle;

pub struct BenchmarkRunner {
    brain_graph: Arc<BrainEnhancedGraphCore>,
    performance_monitor: SystemPerformanceMonitor,
    benchmark_results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub duration: Duration,
    pub operations_count: usize,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
}

impl BenchmarkRunner {
    pub async fn new() -> Self {
        let brain_graph = setup_benchmark_brain_graph().await;
        let performance_monitor = SystemPerformanceMonitor::new();
        
        Self {
            brain_graph,
            performance_monitor,
            benchmark_results: Vec::new(),
        }
    }
    
    pub async fn run_full_benchmark_suite(&mut self) -> BenchmarkSuiteResult {
        println!("🚀 Starting comprehensive performance benchmark suite...");
        
        // Start system monitoring
        let monitor_handle = self.start_monitoring().await;
        
        // Run individual benchmarks
        self.run_allocation_benchmarks().await;
        self.run_search_benchmarks().await;
        self.run_inheritance_benchmarks().await;
        self.run_concurrency_benchmarks().await;
        self.run_scalability_benchmarks().await;
        self.run_api_benchmarks().await;
        
        // Stop monitoring
        monitor_handle.abort();
        
        let performance_report = self.performance_monitor.generate_performance_report();
        
        BenchmarkSuiteResult {
            individual_results: self.benchmark_results.clone(),
            overall_performance: performance_report,
            meets_requirements: self.validate_requirements(),
        }
    }
    
    async fn run_allocation_benchmarks(&mut self) {
        println!("📝 Running memory allocation benchmarks...");
        
        let start_time = Instant::now();
        let mut successful_ops = 0;
        let mut latencies = Vec::new();
        
        for i in 0..1000 {
            let op_start = Instant::now();
            
            let request = generate_allocation_request_with_id(i);
            match self.brain_graph.allocate_memory_with_cortical_coordination(request).await {
                Ok(_) => {
                    successful_ops += 1;
                    latencies.push(op_start.elapsed().as_millis() as f64);
                }
                Err(_) => {}
            }
        }
        
        let total_duration = start_time.elapsed();
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let throughput = successful_ops as f64 / total_duration.as_secs_f64();
        
        let result = BenchmarkResult {
            test_name: "memory_allocation".to_string(),
            duration: total_duration,
            operations_count: 1000,
            success_rate: successful_ops as f64 / 1000.0,
            avg_latency_ms: avg_latency,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: self.get_current_memory_usage(),
            cpu_usage_percent: self.get_current_cpu_usage(),
        };
        
        println!("  ✓ Allocation benchmark completed: {:.0} ops/sec, {:.2}ms avg latency", 
                throughput, avg_latency);
        
        self.benchmark_results.push(result);
    }
    
    async fn run_concurrency_benchmarks(&mut self) {
        println!("🔄 Running concurrency benchmarks...");
        
        let concurrency_levels = vec![10, 50, 100, 500, 1000];
        
        for level in concurrency_levels {
            let start_time = Instant::now();
            
            let tasks: Vec<JoinHandle<Result<_, _>>> = (0..level)
                .map(|i| {
                    let graph = self.brain_graph.clone();
                    tokio::spawn(async move {
                        let request = generate_allocation_request_with_id(i);
                        graph.allocate_memory_with_cortical_coordination(request).await
                    })
                })
                .collect();
            
            let results = futures::future::join_all(tasks).await;
            let successful_ops = results.iter().filter(|r| r.is_ok()).count();
            
            let total_duration = start_time.elapsed();
            let throughput = successful_ops as f64 / total_duration.as_secs_f64();
            
            let result = BenchmarkResult {
                test_name: format!("concurrency_level_{}", level),
                duration: total_duration,
                operations_count: level,
                success_rate: successful_ops as f64 / level as f64,
                avg_latency_ms: total_duration.as_millis() as f64 / level as f64,
                throughput_ops_per_sec: throughput,
                memory_usage_mb: self.get_current_memory_usage(),
                cpu_usage_percent: self.get_current_cpu_usage(),
            };
            
            println!("  ✓ Concurrency {} completed: {:.0} ops/sec, {:.1}% success", 
                    level, throughput, result.success_rate * 100.0);
            
            self.benchmark_results.push(result);
        }
    }
    
    fn validate_requirements(&self) -> bool {
        // Check if all benchmarks meet performance requirements
        self.benchmark_results.iter().all(|result| {
            match result.test_name.as_str() {
                "memory_allocation" => {
                    result.avg_latency_ms < 10.0 
                        && result.throughput_ops_per_sec > 100.0
                        && result.success_rate > 0.99
                }
                name if name.starts_with("concurrency_level_") => {
                    result.success_rate > 0.95
                        && result.throughput_ops_per_sec > 50.0
                }
                _ => true
            }
        })
    }
}

#[derive(Debug)]
pub struct BenchmarkSuiteResult {
    pub individual_results: Vec<BenchmarkResult>,
    pub overall_performance: PerformanceReport,
    pub meets_requirements: bool,
}

impl BenchmarkSuiteResult {
    pub fn print_comprehensive_report(&self) {
        println!("\n🎯 Comprehensive Benchmark Results");
        println!("==================================");
        
        for result in &self.individual_results {
            println!("\n📋 {}", result.test_name);
            println!("   Duration: {:.2}s", result.duration.as_secs_f64());
            println!("   Operations: {}", result.operations_count);
            println!("   Success Rate: {:.1}%", result.success_rate * 100.0);
            println!("   Avg Latency: {:.2}ms", result.avg_latency_ms);
            println!("   Throughput: {:.0} ops/sec", result.throughput_ops_per_sec);
            println!("   Memory Usage: {}MB", result.memory_usage_mb);
            println!("   CPU Usage: {:.1}%", result.cpu_usage_percent);
        }
        
        self.overall_performance.print_summary();
        
        if self.meets_requirements {
            println!("\n🎉 All performance requirements successfully met!");
        } else {
            println!("\n⚠️ Some performance requirements not met - review individual results");
        }
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All core operations (allocation, search, inheritance) benchmarked
- [ ] Concurrency testing validates 1000+ simultaneous operations
- [ ] Scalability testing confirms performance at 100K+ nodes
- [ ] Memory usage remains stable under sustained load
- [ ] API endpoint response times measured and validated

### Performance Requirements
- [ ] Memory allocation: < 10ms average latency
- [ ] Search operations: < 50ms for complex queries
- [ ] Inheritance resolution: < 5ms for 6-hop chains
- [ ] Concurrent operations: > 1000 ops/second
- [ ] Memory usage: < 2GB for 100K nodes

### Testing Requirements
- [ ] Comprehensive benchmark suite runs successfully
- [ ] Performance monitoring captures accurate metrics
- [ ] Benchmark results demonstrate requirement compliance
- [ ] Scalability tests validate production readiness

## Validation Steps

1. **Run comprehensive benchmark suite**:
   ```bash
   cargo bench --bench core_performance_benchmarks
   ```

2. **Execute system performance tests**:
   ```bash
   cargo test --test system_performance_monitor --release
   ```

3. **Run scalability benchmarks**:
   ```bash
   cargo bench --bench scalability_benchmarks -- --sample-size 50
   ```

4. **Generate performance report**:
   ```bash
   cargo run --bin benchmark_runner --release
   ```

## Files to Create/Modify
- `tests/benchmarks/core_performance_benchmarks.rs` - Main benchmark suite
- `tests/benchmarks/system_performance_monitor.rs` - System monitoring
- `tests/benchmarks/benchmark_runner.rs` - Benchmark orchestration
- `benches/main.rs` - Criterion benchmark definitions
- `Cargo.toml` - Add criterion and sysinfo dependencies

## Success Metrics
- Memory allocation latency: < 10ms (95th percentile)
- Search performance: < 50ms for 6-hop queries
- Concurrent throughput: > 1000 operations/second
- Memory efficiency: < 2GB for 100K nodes
- CPU utilization: < 70% under normal load

## Next Task
Upon completion, proceed to **33_data_integrity_tests.md** to validate data integrity and consistency across all components.