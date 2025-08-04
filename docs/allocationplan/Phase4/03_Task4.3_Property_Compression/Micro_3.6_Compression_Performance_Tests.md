# Micro Phase 3.6: Compression Performance Tests

**Estimated Time**: 25 minutes
**Dependencies**: Micro 3.5 (Compression Validation System)
**Objective**: Create comprehensive performance benchmarks and regression detection for compression system

## Task Description

Implement a complete performance testing suite that benchmarks compression operations, detects performance regressions, and validates that the system meets the 10x compression target under various workloads and data patterns.

## Deliverables

Create `benches/task_4_3_compression_performance.rs` with:

1. **Compression benchmarks**: Time and memory usage for various hierarchy sizes
2. **Throughput testing**: Nodes processed per second under different loads
3. **Scalability analysis**: Performance characteristics as data size increases
4. **Regression detection**: Automated detection of performance degradation
5. **Memory efficiency**: Track memory usage during compression operations
6. **Real-world scenarios**: Benchmarks based on realistic data patterns

## Success Criteria

- [ ] Benchmarks 10,000 node compression in < 100ms consistently
- [ ] Achieves 10x compression ratio on highly redundant hierarchies
- [ ] Memory usage grows linearly with hierarchy size
- [ ] Detects 5%+ performance regressions automatically
- [ ] Throughput exceeds 100,000 nodes/second for analysis phase
- [ ] Provides comprehensive performance reporting

## Implementation Requirements

```rust
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use std::time::Duration;

mod compression_benchmarks {
    use super::*;
    use llmkg::compression::*;
    use llmkg::core::*;

    pub struct CompressionBenchmark {
        pub name: String,
        pub hierarchy_size: usize,
        pub redundancy_factor: f32,
        pub expected_compression_ratio: f32,
        pub max_execution_time: Duration,
    }

    pub fn benchmark_compression_analysis(c: &mut Criterion) {
        let benchmarks = vec![
            CompressionBenchmark {
                name: "Small Hierarchy".to_string(),
                hierarchy_size: 1000,
                redundancy_factor: 0.8,
                expected_compression_ratio: 5.0,
                max_execution_time: Duration::from_millis(10),
            },
            CompressionBenchmark {
                name: "Medium Hierarchy".to_string(),
                hierarchy_size: 5000,
                redundancy_factor: 0.85,
                expected_compression_ratio: 8.0,
                max_execution_time: Duration::from_millis(50),
            },
            CompressionBenchmark {
                name: "Large Hierarchy".to_string(),
                hierarchy_size: 10000,
                redundancy_factor: 0.9,
                expected_compression_ratio: 12.0,
                max_execution_time: Duration::from_millis(100),
            },
            CompressionBenchmark {
                name: "XLarge Hierarchy".to_string(),
                hierarchy_size: 50000,
                redundancy_factor: 0.95,
                expected_compression_ratio: 15.0,
                max_execution_time: Duration::from_millis(500),
            },
        ];

        for benchmark in benchmarks {
            let mut group = c.benchmark_group(&format!("compression_analysis_{}", benchmark.name.replace(" ", "_").to_lowercase()));
            group.throughput(Throughput::Elements(benchmark.hierarchy_size as u64));
            group.measurement_time(Duration::from_secs(10));

            group.bench_function("property_analysis", |b| {
                b.iter_batched(
                    || create_hierarchy_with_redundancy(benchmark.hierarchy_size, benchmark.redundancy_factor),
                    |hierarchy| {
                        let analyzer = PropertyAnalyzer::new(0.7, 5);
                        black_box(analyzer.analyze_hierarchy(&hierarchy))
                    },
                    BatchSize::SmallInput,
                );
            });

            group.bench_function("full_compression", |b| {
                b.iter_batched(
                    || {
                        let hierarchy = create_hierarchy_with_redundancy(benchmark.hierarchy_size, benchmark.redundancy_factor);
                        let config = CompressionConfig::default();
                        let orchestrator = CompressionOrchestrator::new(config);
                        (hierarchy, orchestrator)
                    },
                    |(mut hierarchy, orchestrator)| {
                        black_box(orchestrator.compress_hierarchy(&mut hierarchy))
                    },
                    BatchSize::SmallInput,
                );
            });

            group.finish();
        }
    }

    pub fn benchmark_compression_throughput(c: &mut Criterion) {
        let mut group = c.benchmark_group("compression_throughput");
        group.throughput(Throughput::Elements(10000));

        group.bench_function("analysis_throughput", |b| {
            b.iter_batched(
                || create_realistic_hierarchy(10000),
                |hierarchy| {
                    let analyzer = PropertyAnalyzer::new(0.6, 3);
                    let start = std::time::Instant::now();
                    let result = analyzer.analyze_hierarchy(&hierarchy);
                    let elapsed = start.elapsed();
                    
                    // Calculate throughput
                    let nodes_per_second = (10000 as f64) / elapsed.as_secs_f64();
                    assert!(nodes_per_second > 100000.0, "Throughput below target: {} nodes/sec", nodes_per_second);
                    
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("promotion_throughput", |b| {
            b.iter_batched(
                || {
                    let hierarchy = create_realistic_hierarchy(10000);
                    let analyzer = PropertyAnalyzer::new(0.7, 5);
                    let candidates = analyzer.find_promotion_candidates(&hierarchy);
                    (hierarchy, candidates)
                },
                |(mut hierarchy, candidates)| {
                    let promoter = PropertyPromoter::new(PromotionStrategy::Balanced);
                    let start = std::time::Instant::now();
                    let result = promoter.promote_properties(&mut hierarchy, candidates);
                    let elapsed = start.elapsed();
                    
                    // Ensure reasonable promotion speed
                    assert!(elapsed < Duration::from_millis(50), "Promotion too slow: {:?}", elapsed);
                    
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }

    pub fn benchmark_memory_efficiency(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory_efficiency");

        group.bench_function("memory_usage_scaling", |b| {
            let sizes = vec![1000, 2000, 5000, 10000];
            
            for size in sizes {
                let initial_memory = get_current_memory_usage();
                
                b.iter_batched(
                    || create_hierarchy_with_redundancy(size, 0.8),
                    |mut hierarchy| {
                        let config = CompressionConfig::default();
                        let orchestrator = CompressionOrchestrator::new(config);
                        
                        let memory_before = get_current_memory_usage();
                        let result = orchestrator.compress_hierarchy(&mut hierarchy);
                        let memory_after = get_current_memory_usage();
                        
                        let memory_used = memory_after - memory_before;
                        let memory_per_node = memory_used as f64 / size as f64;
                        
                        // Memory usage should be roughly linear
                        assert!(memory_per_node < 1000.0, "Memory usage per node too high: {} bytes", memory_per_node);
                        
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            }
        });

        group.finish();
    }

    pub fn benchmark_compression_ratios(c: &mut Criterion) {
        let mut group = c.benchmark_group("compression_ratios");

        let test_scenarios = vec![
            ("highly_redundant", 0.95, 10.0),
            ("moderately_redundant", 0.8, 6.0),
            ("low_redundancy", 0.6, 3.0),
            ("minimal_redundancy", 0.4, 1.5),
        ];

        for (scenario_name, redundancy, min_expected_ratio) in test_scenarios {
            group.bench_function(&format!("ratio_{}", scenario_name), |b| {
                b.iter_batched(
                    || create_hierarchy_with_redundancy(5000, redundancy),
                    |mut hierarchy| {
                        let initial_size = hierarchy.calculate_total_size_bytes();
                        
                        let config = CompressionConfig::default();
                        let orchestrator = CompressionOrchestrator::new(config);
                        let result = orchestrator.compress_hierarchy(&mut hierarchy).unwrap();
                        
                        let final_size = hierarchy.calculate_total_size_bytes();
                        let actual_ratio = initial_size as f32 / final_size as f32;
                        
                        assert!(actual_ratio >= min_expected_ratio * 0.8, 
                               "Compression ratio too low for {}: {:.2}x (expected >= {:.2}x)", 
                               scenario_name, actual_ratio, min_expected_ratio);
                        
                        black_box((result, actual_ratio))
                    },
                    BatchSize::SmallInput,
                );
            });
        }

        group.finish();
    }

    pub fn benchmark_iterative_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("iterative_compression");

        group.bench_function("convergence_speed", |b| {
            b.iter_batched(
                || {
                    let hierarchy = create_complex_hierarchy(3000);
                    let orchestrator = CompressionOrchestrator::new(CompressionConfig::default());
                    (hierarchy, orchestrator)
                },
                |(mut hierarchy, orchestrator)| {
                    let compressor = IterativeCompressor::new(10, 8.0, OptimizationStrategy::Adaptive);
                    
                    let start = std::time::Instant::now();
                    let progress = compressor.compress_iteratively(&mut hierarchy, &orchestrator).unwrap();
                    let elapsed = start.elapsed();
                    
                    // Should converge quickly
                    assert!(progress.completed_iterations.len() <= 6, "Too many iterations: {}", progress.completed_iterations.len());
                    assert!(elapsed < Duration::from_millis(200), "Iterative compression too slow: {:?}", elapsed);
                    
                    // Should achieve reasonable compression
                    assert!(progress.current_ratio >= 5.0, "Insufficient compression ratio: {:.2}x", progress.current_ratio);
                    
                    black_box(progress)
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }

    pub fn benchmark_validation_performance(c: &mut Criterion) {
        let mut group = c.benchmark_group("validation_performance");

        group.bench_function("comprehensive_validation", |b| {
            b.iter_batched(
                || {
                    let original = create_realistic_hierarchy(10000);
                    let mut compressed = original.clone();
                    let operations = perform_standard_compression(&mut compressed);
                    (original, compressed, operations)
                },
                |(original, compressed, operations)| {
                    let validator = CompressionValidator::new(ValidationLevel::Comprehensive);
                    
                    let start = std::time::Instant::now();
                    let result = validator.validate_compression(&original, &compressed, &operations);
                    let elapsed = start.elapsed();
                    
                    assert!(elapsed < Duration::from_millis(50), "Validation too slow: {:?}", elapsed);
                    assert!(result.is_valid, "Validation failed: {:?}", result.errors);
                    
                    black_box(result)
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }

    // Performance regression detection
    pub fn detect_performance_regressions() -> PerformanceReport {
        let baseline_metrics = load_baseline_performance_metrics();
        let current_metrics = run_current_performance_tests();
        
        let mut regressions = Vec::new();
        
        for (test_name, baseline_time) in baseline_metrics.execution_times {
            if let Some(current_time) = current_metrics.execution_times.get(&test_name) {
                let regression_ratio = current_time.as_secs_f64() / baseline_time.as_secs_f64();
                
                if regression_ratio > 1.05 { // 5% threshold
                    regressions.push(PerformanceRegression {
                        test_name: test_name.clone(),
                        baseline_time,
                        current_time: *current_time,
                        regression_percentage: (regression_ratio - 1.0) * 100.0,
                    });
                }
            }
        }
        
        PerformanceReport {
            total_tests_run: current_metrics.execution_times.len(),
            regressions_detected: regressions.len(),
            regressions,
            overall_performance_score: calculate_performance_score(&current_metrics),
            compression_ratio_consistency: validate_compression_ratio_consistency(&current_metrics),
        }
    }

    // Helper functions for creating test data
    fn create_hierarchy_with_redundancy(size: usize, redundancy_factor: f32) -> InheritanceHierarchy {
        let mut hierarchy = InheritanceHierarchy::new();
        
        // Create root
        let root = hierarchy.create_node("Root".to_string());
        
        // Build tree structure with specified redundancy
        build_redundant_tree(&mut hierarchy, root, size - 1, redundancy_factor, 0, 4);
        
        hierarchy
    }

    fn build_redundant_tree(
        hierarchy: &mut InheritanceHierarchy,
        parent: NodeId,
        remaining_nodes: usize,
        redundancy_factor: f32,
        depth: usize,
        max_depth: usize,
    ) {
        if remaining_nodes == 0 || depth >= max_depth {
            return;
        }
        
        let children_count = std::cmp::min(remaining_nodes, 4);
        let nodes_per_child = remaining_nodes / children_count;
        
        for i in 0..children_count {
            let child_name = format!("Node_{}_{}", depth, i);
            let child = hierarchy.create_node(child_name);
            hierarchy.add_inheritance(child, parent);
            
            // Add redundant properties based on redundancy factor
            add_redundant_properties(hierarchy, child, parent, redundancy_factor);
            
            // Recursively build subtree
            let child_nodes = if i == children_count - 1 {
                remaining_nodes - (nodes_per_child * i)
            } else {
                nodes_per_child
            };
            
            if child_nodes > 1 {
                build_redundant_tree(hierarchy, child, child_nodes - 1, redundancy_factor, depth + 1, max_depth);
            }
        }
    }

    fn add_redundant_properties(
        hierarchy: &mut InheritanceHierarchy,
        node: NodeId,
        parent: NodeId,
        redundancy_factor: f32,
    ) {
        // Add properties that are shared with siblings to create compression opportunities
        let property_count = 10;
        
        for i in 0..property_count {
            let should_add_redundant = fastrand::f32() < redundancy_factor;
            
            if should_add_redundant {
                let property_name = format!("shared_prop_{}", i);
                let property_value = PropertyValue::String(format!("shared_value_{}", i));
                hierarchy.set_property(node, &property_name, property_value);
            } else {
                let property_name = format!("unique_prop_{}_{}", node.0, i);
                let property_value = PropertyValue::String(format!("unique_value_{}_{}", node.0, i));
                hierarchy.set_property(node, &property_name, property_value);
            }
        }
    }
}

criterion_group!(
    compression_benches,
    compression_benchmarks::benchmark_compression_analysis,
    compression_benchmarks::benchmark_compression_throughput,
    compression_benchmarks::benchmark_memory_efficiency,
    compression_benchmarks::benchmark_compression_ratios,
    compression_benchmarks::benchmark_iterative_performance,
    compression_benchmarks::benchmark_validation_performance
);

criterion_main!(compression_benches);

#[derive(Debug)]
pub struct PerformanceReport {
    pub total_tests_run: usize,
    pub regressions_detected: usize,
    pub regressions: Vec<PerformanceRegression>,
    pub overall_performance_score: f32,
    pub compression_ratio_consistency: bool,
}

#[derive(Debug)]
pub struct PerformanceRegression {
    pub test_name: String,
    pub baseline_time: Duration,
    pub current_time: Duration,
    pub regression_percentage: f64,
}
```

## Test Requirements

Must pass performance benchmark validation:
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_benchmark_consistency() {
        // Run benchmarks multiple times to ensure consistency
        let mut results = Vec::new();
        
        for _ in 0..5 {
            let hierarchy = create_hierarchy_with_redundancy(1000, 0.8);
            let config = CompressionConfig::default();
            let orchestrator = CompressionOrchestrator::new(config);
            
            let start = std::time::Instant::now();
            let mut hierarchy_copy = hierarchy.clone();
            let result = orchestrator.compress_hierarchy(&mut hierarchy_copy).unwrap();
            let elapsed = start.elapsed();
            
            results.push((elapsed, result.compression_ratio));
        }
        
        // Check consistency (coefficient of variation < 20%)
        let times: Vec<f64> = results.iter().map(|(t, _)| t.as_secs_f64()).collect();
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean_time;
        
        assert!(coefficient_of_variation < 0.2, "Performance too inconsistent: CV = {:.3}", coefficient_of_variation);
    }

    #[test]
    fn test_compression_ratio_targets() {
        let test_cases = vec![
            (0.9, 8.0),   // 90% redundancy -> 8x compression
            (0.8, 5.0),   // 80% redundancy -> 5x compression
            (0.7, 3.0),   // 70% redundancy -> 3x compression
        ];
        
        for (redundancy, min_ratio) in test_cases {
            let hierarchy = create_hierarchy_with_redundancy(2000, redundancy);
            let initial_size = hierarchy.calculate_total_size_bytes();
            
            let config = CompressionConfig::default();
            let orchestrator = CompressionOrchestrator::new(config);
            let mut hierarchy_copy = hierarchy.clone();
            
            let result = orchestrator.compress_hierarchy(&mut hierarchy_copy).unwrap();
            let final_size = hierarchy_copy.calculate_total_size_bytes();
            let actual_ratio = initial_size as f32 / final_size as f32;
            
            assert!(actual_ratio >= min_ratio * 0.9, 
                   "Compression ratio {:.2}x below target {:.2}x for redundancy {:.0}%", 
                   actual_ratio, min_ratio, redundancy * 100.0);
        }
    }

    #[test]
    fn test_scalability_characteristics() {
        let sizes = vec![1000, 2000, 5000, 10000];
        let mut execution_times = Vec::new();
        
        for size in sizes {
            let hierarchy = create_hierarchy_with_redundancy(size, 0.8);
            let config = CompressionConfig::default();
            let orchestrator = CompressionOrchestrator::new(config);
            
            let start = std::time::Instant::now();
            let mut hierarchy_copy = hierarchy.clone();
            orchestrator.compress_hierarchy(&mut hierarchy_copy).unwrap();
            let elapsed = start.elapsed();
            
            execution_times.push((size, elapsed.as_secs_f64()));
        }
        
        // Check that time complexity is reasonable (not exponential)
        for i in 1..execution_times.len() {
            let (prev_size, prev_time) = execution_times[i - 1];
            let (curr_size, curr_time) = execution_times[i];
            
            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time / prev_time;
            
            // Time should not grow faster than O(n log n)
            let expected_max_ratio = size_ratio * size_ratio.log2();
            assert!(time_ratio <= expected_max_ratio * 1.5, 
                   "Poor time complexity: {}x size increase caused {}x time increase", 
                   size_ratio, time_ratio);
        }
    }

    #[test]
    fn test_memory_regression_detection() {
        let baseline_memory = measure_baseline_memory_usage();
        let current_memory = measure_current_memory_usage();
        
        let memory_increase_ratio = current_memory as f64 / baseline_memory as f64;
        
        assert!(memory_increase_ratio <= 1.1, 
               "Memory usage regression detected: {:.1}% increase", 
               (memory_increase_ratio - 1.0) * 100.0);
    }

    fn measure_baseline_memory_usage() -> usize {
        // This would load from a stored baseline file in a real implementation
        1024 * 1024 // 1MB baseline
    }

    fn measure_current_memory_usage() -> usize {
        let hierarchy = create_hierarchy_with_redundancy(5000, 0.8);
        let config = CompressionConfig::default();
        let orchestrator = CompressionOrchestrator::new(config);
        
        let memory_before = get_current_memory_usage();
        let mut hierarchy_copy = hierarchy.clone();
        orchestrator.compress_hierarchy(&mut hierarchy_copy).unwrap();
        let memory_after = get_current_memory_usage();
        
        memory_after - memory_before
    }
}
```

## File Location
`benches/task_4_3_compression_performance.rs`

## Task 4.3 Completion
This completes all micro phases for Task 4.3: Property Compression Engine. The system now includes:
- Analysis engine for identifying compression opportunities
- Promotion engine for safe property movement
- Orchestrator for coordinating the complete workflow
- Iterative algorithm for optimal compression ratios
- Validation system for ensuring correctness
- Comprehensive performance testing and regression detection

Total estimated time: 185 minutes (40+50+35+45+30+25)