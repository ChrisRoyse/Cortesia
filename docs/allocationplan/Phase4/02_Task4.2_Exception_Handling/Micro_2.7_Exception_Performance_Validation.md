# Micro Phase 2.7: Exception Performance Validation

**Estimated Time**: 20 minutes
**Dependencies**: Micro 2.6 (Exception Integration Tests)
**Objective**: Implement comprehensive performance benchmarks and validation for the exception system

## Task Description

Create a comprehensive performance validation suite that benchmarks all aspects of the exception handling system. This includes micro-benchmarks for individual components and macro-benchmarks for realistic usage scenarios.

The validation suite provides performance regression detection, optimization guidance, and ensures the system meets performance requirements under various load conditions.

## Deliverables

Create `benches/task_4_2_exception_performance.rs` with:

1. **Component benchmarks**: Individual performance tests for each exception component
2. **End-to-end benchmarks**: Full workflow performance under realistic loads
3. **Scalability tests**: Performance characteristics as data size increases
4. **Memory benchmarks**: Memory usage and allocation patterns
5. **Regression detection**: Automated performance regression detection

## Success Criteria

- [ ] Exception detection performs <1ms per node with 1000 properties
- [ ] Exception storage operations complete <100μs per exception
- [ ] Pattern learning processes 1000+ examples per second
- [ ] Storage optimization reduces memory usage >50%
- [ ] Concurrent operations scale linearly with thread count
- [ ] Benchmarks provide <5% variance between runs

## Implementation Requirements

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;

use llmkg::exceptions::{
    ExceptionStore, ExceptionDetector, ExceptionHandler, PatternLearner, StorageOptimizer
};
use llmkg::core::{InheritanceNode, NodeId, PropertyValue};

// Benchmark data generators
fn generate_test_nodes(count: usize, properties_per_node: usize) -> Vec<InheritanceNode>;
fn generate_exception_patterns(count: usize) -> Vec<(ExceptionContext, Exception)>;
fn generate_realistic_inheritance_hierarchy(depth: usize, width: usize) -> Vec<InheritanceNode>;

// Performance measurement utilities
struct PerformanceProfiler {
    start_time: Instant,
    checkpoints: Vec<(String, Duration)>,
    memory_samples: Vec<(String, usize)>,
}

impl PerformanceProfiler {
    fn new() -> Self;
    fn checkpoint(&mut self, name: &str);
    fn memory_checkpoint(&mut self, name: &str);
    fn finish(self) -> PerformanceReport;
}

#[derive(Debug)]
struct PerformanceReport {
    total_duration: Duration,
    checkpoints: Vec<(String, Duration)>,
    memory_usage: Vec<(String, usize)>,
    peak_memory: usize,
}

// Benchmark configuration
struct BenchmarkConfig {
    warm_up_iterations: usize,
    measurement_iterations: usize,
    sample_size: usize,
    timeout: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warm_up_iterations: 100,
            measurement_iterations: 1000,
            sample_size: 100,
            timeout: Duration::from_secs(30),
        }
    }
}
```

## Test Requirements

Must pass comprehensive performance benchmarks:
```rust
fn bench_exception_store_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("exception_store");
    
    for size in [100, 1000, 10000].iter() {
        let store = ExceptionStore::new();
        let exceptions: Vec<_> = (0..*size).map(|i| {
            (
                NodeId(i),
                format!("prop_{}", i % 100),
                Exception {
                    inherited_value: PropertyValue::Boolean(true),
                    actual_value: PropertyValue::Boolean(false),
                    reason: format!("Exception {}", i),
                    source: ExceptionSource::Detected,
                    created_at: Instant::now(),
                    confidence: 0.8,
                }
            )
        }).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark store operations
        group.bench_with_input(
            BenchmarkId::new("add_exception", size),
            size,
            |b, &size| {
                b.iter(|| {
                    for (node_id, prop, exception) in &exceptions[0..size.min(100)] {
                        store.add_exception(*node_id, prop.clone(), exception.clone());
                    }
                });
            }
        );
        
        // Setup data for retrieval benchmark
        for (node_id, prop, exception) in &exceptions {
            store.add_exception(*node_id, prop.clone(), exception.clone());
        }
        
        // Benchmark retrieval operations
        group.bench_with_input(
            BenchmarkId::new("get_exception", size),
            size,
            |b, &size| {
                b.iter(|| {
                    for (node_id, prop, _) in &exceptions[0..size.min(100)] {
                        black_box(store.get_exception(*node_id, prop));
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn bench_exception_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("exception_detection");
    
    let detector = ExceptionDetector::new(0.8, 0.7);
    
    for node_size in [10, 100, 1000].iter() {
        let nodes = generate_test_nodes(1, *node_size);
        let node = &nodes[0];
        
        let inherited_properties: HashMap<String, PropertyValue> = (0..*node_size).map(|i| {
            (format!("prop_{}", i), PropertyValue::Boolean(true))
        }).collect();
        
        group.throughput(Throughput::Elements(*node_size as u64));
        group.bench_with_input(
            BenchmarkId::new("detect_node_exceptions", node_size),
            node_size,
            |b, _| {
                b.iter(|| {
                    black_box(detector.detect_node_exceptions(
                        black_box(node),
                        black_box(&inherited_properties)
                    ));
                });
            }
        );
    }
    
    group.finish();
}

fn bench_exception_handler_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("exception_handler");
    
    let store = Arc::new(ExceptionStore::new());
    let handler = ExceptionHandler::new(Arc::clone(&store));
    
    // Pre-populate with exceptions
    for i in 0..1000 {
        let exception = Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: format!("Exception {}", i),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        store.add_exception(NodeId(i), "test_prop".to_string(), exception);
    }
    
    for cache_hit_rate in [0.1, 0.5, 0.9].iter() {
        let cache_hits = (*cache_hit_rate * 1000.0) as usize;
        
        group.bench_with_input(
            BenchmarkId::new("resolve_property", format!("{}% cache hit", cache_hit_rate * 100.0)),
            cache_hit_rate,
            |b, _| {
                b.iter(|| {
                    // Mix of cache hits and misses
                    for i in 0..100 {
                        let node_id = if i < cache_hits / 10 {
                            NodeId(i as u64) // Cache hit
                        } else {
                            NodeId(1000 + i as u64) // Cache miss
                        };
                        
                        black_box(handler.resolve_property(
                            node_id,
                            "test_prop",
                            Some(PropertyValue::Boolean(true))
                        ));
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn bench_pattern_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_learning");
    
    for batch_size in [10, 100, 1000].iter() {
        let mut learner = PatternLearner::new();
        let patterns = generate_exception_patterns(*batch_size);
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("learn_patterns", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    for (context, exception) in &patterns {
                        learner.learn_from_exception(black_box(exception), black_box(context));
                    }
                });
            }
        );
        
        // Benchmark prediction after learning
        let test_context = &patterns[0].0;
        group.bench_with_input(
            BenchmarkId::new("predict_likelihood", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(learner.predict_exception_likelihood(black_box(test_context)));
                });
            }
        );
    }
    
    group.finish();
}

fn bench_storage_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_optimization");
    
    for data_size in [1000, 10000, 100000].iter() {
        let mut optimizer = StorageOptimizer::new(*data_size);
        
        // Pre-populate with data
        let exceptions: Vec<_> = (0..*data_size).map(|i| {
            let key = ExceptionKey {
                node_id: NodeId(i),
                property_hash: (i % 100) as u64, // Create similarity for compression
            };
            let exception = Exception {
                inherited_value: PropertyValue::String("default".to_string()),
                actual_value: PropertyValue::String(format!("value_{}", i)),
                reason: "Test exception".to_string(),
                source: ExceptionSource::Detected,
                created_at: Instant::now(),
                confidence: 0.8,
            };
            (key, exception)
        }).collect();
        
        for (key, exception) in &exceptions {
            optimizer.store_exception(*key, exception.clone()).unwrap();
        }
        
        group.throughput(Throughput::Elements(*data_size as u64));
        
        // Benchmark optimization operations
        group.bench_with_input(
            BenchmarkId::new("optimize_storage", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    black_box(optimizer.optimize_storage());
                });
            }
        );
        
        // Benchmark bulk operations
        let test_keys: Vec<_> = exceptions.iter().take(1000).map(|(k, _)| *k).collect();
        group.bench_with_input(
            BenchmarkId::new("bulk_retrieve", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    black_box(optimizer.bulk_retrieve(black_box(&test_keys)));
                });
            }
        );
    }
    
    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    
    for thread_count in [1, 2, 4, 8].iter() {
        let store = Arc::new(ExceptionStore::new());
        let operations_per_thread = 1000;
        
        group.throughput(Throughput::Elements((thread_count * operations_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_store_operations", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count).map(|thread_id| {
                        let store = Arc::clone(&store);
                        thread::spawn(move || {
                            for i in 0..operations_per_thread {
                                let node_id = NodeId(thread_id * operations_per_thread + i);
                                let exception = Exception {
                                    inherited_value: PropertyValue::Boolean(true),
                                    actual_value: PropertyValue::Boolean(false),
                                    reason: format!("Thread {} Exception {}", thread_id, i),
                                    source: ExceptionSource::Detected,
                                    created_at: Instant::now(),
                                    confidence: 0.8,
                                };
                                store.add_exception(node_id, "test_prop".to_string(), exception);
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn bench_end_to_end_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    
    for scenario_size in [100, 1000].iter() {
        let hierarchy = generate_realistic_inheritance_hierarchy(5, *scenario_size / 5);
        
        group.throughput(Throughput::Elements(*scenario_size as u64));
        group.bench_with_input(
            BenchmarkId::new("complete_exception_workflow", scenario_size),
            scenario_size,
            |b, _| {
                b.iter(|| {
                    let store = Arc::new(ExceptionStore::new());
                    let detector = ExceptionDetector::new(0.8, 0.7);
                    let handler = ExceptionHandler::new(Arc::clone(&store));
                    let mut learner = PatternLearner::new();
                    let mut optimizer = StorageOptimizer::new(*scenario_size);
                    
                    for node in &hierarchy {
                        // Simulate inherited properties
                        let inherited_properties: HashMap<String, PropertyValue> = 
                            node.local_properties.iter()
                                .map(|(k, _)| (k.clone(), PropertyValue::Boolean(true)))
                                .collect();
                        
                        // Detect exceptions
                        let exceptions = detector.detect_node_exceptions(node, &inherited_properties);
                        
                        // Store exceptions
                        for (prop, exception) in &exceptions {
                            store.add_exception(node.id, prop.clone(), exception.clone());
                        }
                        
                        // Apply exceptions through handler
                        let mut resolved_props = inherited_properties.clone();
                        handler.apply_exceptions_to_node(node.id, &mut resolved_props);
                        
                        // Learn patterns
                        for (prop, exception) in &exceptions {
                            let context = ExceptionContext {
                                property_name: prop.clone(),
                                inherited_value: exception.inherited_value.clone(),
                                actual_value: exception.actual_value.clone(),
                                node_type: node.node_type.clone(),
                                inheritance_depth: 2,
                                sibling_properties: node.local_properties.clone(),
                            };
                            learner.learn_from_exception(exception, &context);
                        }
                    }
                    
                    // Final optimization
                    black_box(optimizer.optimize_storage());
                });
            }
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    for data_size in [1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_efficiency", data_size),
            data_size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    
                    for _ in 0..iters {
                        let start = Instant::now();
                        
                        let store = ExceptionStore::new();
                        let mut profiler = PerformanceProfiler::new();
                        
                        profiler.memory_checkpoint("baseline");
                        
                        // Add exceptions and measure memory growth
                        for i in 0..size {
                            let exception = Exception {
                                inherited_value: PropertyValue::String(format!("inherited_{}", i)),
                                actual_value: PropertyValue::String(format!("actual_{}", i)),
                                reason: format!("Reason {}", i),
                                source: ExceptionSource::Detected,
                                created_at: Instant::now(),
                                confidence: 0.8,
                            };
                            store.add_exception(NodeId(i), format!("prop_{}", i), exception);
                            
                            if i % (size / 10) == 0 {
                                profiler.memory_checkpoint(&format!("after_{}_exceptions", i));
                            }
                        }
                        
                        profiler.memory_checkpoint("final");
                        let report = profiler.finish();
                        
                        // Validate memory usage is reasonable
                        let final_memory = report.memory_usage.last().unwrap().1;
                        let baseline_memory = report.memory_usage.first().unwrap().1;
                        let memory_per_exception = (final_memory - baseline_memory) / size;
                        
                        // Should be less than 500 bytes per exception on average
                        assert!(memory_per_exception < 500, 
                               "Memory usage too high: {} bytes per exception", memory_per_exception);
                        
                        total_duration += start.elapsed();
                    }
                    
                    total_duration
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_exception_store_operations,
    bench_exception_detection,
    bench_exception_handler_resolution,
    bench_pattern_learning,
    bench_storage_optimization,
    bench_concurrent_operations,
    bench_end_to_end_workflow,
    bench_memory_usage
);

criterion_main!(benches);

// Helper function implementations
fn generate_test_nodes(count: usize, properties_per_node: usize) -> Vec<InheritanceNode> {
    (0..count).map(|i| {
        let mut node = InheritanceNode::new(NodeId(i as u64), &format!("TestNode_{}", i));
        
        for j in 0..properties_per_node {
            let prop_name = format!("prop_{}", j);
            let value = match j % 3 {
                0 => PropertyValue::Boolean(i % 2 == 0),
                1 => PropertyValue::String(format!("value_{}_{}", i, j)),
                _ => PropertyValue::Number(i as f64 + j as f64),
            };
            node.local_properties.insert(prop_name, value);
        }
        
        node
    }).collect()
}

fn generate_exception_patterns(count: usize) -> Vec<(ExceptionContext, Exception)> {
    (0..count).map(|i| {
        let context = ExceptionContext {
            property_name: format!("prop_{}", i % 10),
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            node_type: format!("Type_{}", i % 5),
            inheritance_depth: (i % 3) as u32 + 1,
            sibling_properties: HashMap::new(),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: format!("Pattern exception {}", i),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        
        (context, exception)
    }).collect()
}

fn generate_realistic_inheritance_hierarchy(depth: usize, width: usize) -> Vec<InheritanceNode> {
    let mut nodes = Vec::new();
    let mut node_id = 0;
    
    for level in 0..depth {
        for node_in_level in 0..width {
            let mut node = InheritanceNode::new(NodeId(node_id), 
                                             &format!("Level{}_Node{}", level, node_in_level));
            
            // Add parent references (except for root level)
            if level > 0 {
                let parent_level_start = (level - 1) * width;
                let parent_id = parent_level_start + (node_in_level % width);
                node.parent_ids.push(NodeId(parent_id as u64));
            }
            
            // Add some properties
            for prop_idx in 0..5 {
                let prop_name = format!("prop_{}", prop_idx);
                let value = if node_id % 3 == 0 {
                    PropertyValue::Boolean(false) // Create some exceptions
                } else {
                    PropertyValue::Boolean(true)
                };
                node.local_properties.insert(prop_name, value);
            }
            
            nodes.push(node);
            node_id += 1;
        }
    }
    
    nodes
}
```

## File Location
`benches/task_4_2_exception_performance.rs`

## Next Micro Phase
Task 4.2 Exception Handling System is now complete. All micro phases have been implemented and documented. Proceed to Task 4.3 if available, or begin implementation of the documented exception system components.