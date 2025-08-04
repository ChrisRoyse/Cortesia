# Micro Task 41: Benchmarking

**Priority**: HIGH  
**Estimated Time**: 40 minutes  
**Dependencies**: 40_integration_tests.md completed  
**Skills Required**: Performance analysis, benchmarking methodologies

## Objective

Implement comprehensive benchmarking suite to validate Phase 7 performance targets and establish baseline metrics for future optimization.

## Context

Benchmarking provides quantitative validation that the query through activation system meets all performance requirements and establishes metrics for monitoring production performance and regression detection.

## Specifications

### Benchmark Categories

1. **Core Performance Metrics**
   - Query processing latency (< 50ms target)
   - Activation spreading speed (< 10ms target)
   - Intent recognition time (< 200ms target)
   - Memory usage efficiency (< 50MB target)

2. **Scalability Benchmarks**
   - Concurrent query throughput (> 100/s target)
   - Graph size scaling behavior
   - Memory usage under load
   - Cache hit rate optimization

3. **Quality Metrics**
   - Intent recognition accuracy (> 90% target)
   - Activation accuracy (> 95% target)
   - Explanation quality (> 85% target)
   - Result relevance scores

## Implementation Guide

### Step 1: Benchmark Infrastructure
```rust
// File: benches/query_system_benchmarks.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use LLMKG::query::*;
use LLMKG::core::*;

pub struct BenchmarkSuite {
    small_processor: QueryProcessor,      // 1K nodes
    medium_processor: QueryProcessor,     // 100K nodes  
    large_processor: QueryProcessor,      // 1M nodes
    test_queries: Vec<BenchmarkQuery>,
}

#[derive(Clone)]
pub struct BenchmarkQuery {
    pub text: String,
    pub expected_intent: QueryIntentType,
    pub complexity: QueryComplexity,
    pub expected_results: usize,
}

#[derive(Clone, Debug)]
pub enum QueryComplexity {
    Simple,    // Direct lookup
    Medium,    // 2-3 hops
    Complex,   // Multi-hop with reasoning
    Compound,  // Multiple sub-queries
}

impl BenchmarkSuite {
    pub async fn new() -> Self {
        Self {
            small_processor: QueryProcessor::new_with_test_graph(GraphSize::Small).await,
            medium_processor: QueryProcessor::new_with_test_graph(GraphSize::Medium).await,
            large_processor: QueryProcessor::new_with_test_graph(GraphSize::Large).await,
            test_queries: Self::create_benchmark_queries(),
        }
    }
    
    fn create_benchmark_queries() -> Vec<BenchmarkQuery> {
        vec![
            BenchmarkQuery {
                text: "What animals can fly?".to_string(),
                expected_intent: QueryIntentType::Filter,
                complexity: QueryComplexity::Simple,
                expected_results: 15,
            },
            BenchmarkQuery {
                text: "How are mammals related to reptiles through evolution?".to_string(),
                expected_intent: QueryIntentType::Relationship,
                complexity: QueryComplexity::Complex,
                expected_results: 8,
            },
            BenchmarkQuery {
                text: "Compare the cardiovascular systems of birds and mammals".to_string(),
                expected_intent: QueryIntentType::Comparison,
                complexity: QueryComplexity::Compound,
                expected_results: 12,
            },
        ]
    }
}
```

### Step 2: Core Performance Benchmarks
```rust
fn benchmark_query_processing_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let suite = rt.block_on(BenchmarkSuite::new());
    
    let mut group = c.benchmark_group("query_processing_latency");
    group.significance_level(0.1).sample_size(100);
    
    for query in &suite.test_queries {
        for (size_name, processor) in [
            ("small", &suite.small_processor),
            ("medium", &suite.medium_processor),
            ("large", &suite.large_processor),
        ] {
            group.bench_with_input(
                BenchmarkId::new(&query.complexity, size_name),
                &(processor, query),
                |b, (processor, query)| {
                    b.to_async(&rt).iter(|| async {
                        let result = processor.process_query(
                            black_box(&query.text),
                            black_box(&QueryContext::default())
                        ).await;
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_activation_spreading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let suite = rt.block_on(BenchmarkSuite::new());
    
    let mut group = c.benchmark_group("activation_spreading");
    group.significance_level(0.1).sample_size(50);
    
    // Test with different graph sizes
    for (size_name, node_count) in [
        ("1k", 1_000),
        ("10k", 10_000),
        ("100k", 100_000),
        ("1m", 1_000_000),
    ] {
        group.bench_with_input(
            BenchmarkId::new("spreading_time", size_name),
            &node_count,
            |b, &node_count| {
                b.to_async(&rt).iter(|| async {
                    let graph = create_test_graph(node_count).await;
                    let spreader = ActivationSpreader::new();
                    
                    let mut initial_state = ActivationState::new();
                    initial_state.set_activation(NodeId(0), 1.0);
                    
                    let result = spreader.spread_activation(
                        black_box(&initial_state),
                        black_box(&graph)
                    ).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_intent_recognition(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let parser = rt.block_on(QueryIntentParser::new());
    
    let mut group = c.benchmark_group("intent_recognition");
    group.significance_level(0.1).sample_size(200);
    
    let test_queries = vec![
        "What animals can fly?",
        "How are dogs related to wolves through evolution?",
        "Compare the digestive systems of carnivores and herbivores",
        "Why do birds migrate south in winter?",
        "What is the definition of photosynthesis?",
    ];
    
    for query in test_queries {
        group.bench_with_input(
            BenchmarkId::new("parse_time", query.len()),
            query,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let result = parser.parse_intent(black_box(query)).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}
```

### Step 3: Scalability Benchmarks  
```rust
fn benchmark_concurrent_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let suite = rt.block_on(BenchmarkSuite::new());
    
    let mut group = c.benchmark_group("concurrent_throughput");
    group.significance_level(0.1).sample_size(10);
    
    for concurrent_count in [1, 10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_queries", concurrent_count),
            &concurrent_count,
            |b, &concurrent_count| {
                b.to_async(&rt).iter(|| async {
                    let processor = &suite.medium_processor;
                    let query = "What animals live in water?";
                    
                    let handles: Vec<_> = (0..concurrent_count).map(|_| {
                        let p = processor.clone();
                        tokio::spawn(async move {
                            p.process_query(query, &QueryContext::default()).await
                        })
                    }).collect();
                    
                    let results = futures::future::try_join_all(handles).await;
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.significance_level(0.1).sample_size(20);
    
    for node_count in [1_000, 10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("memory_per_node", node_count),
            &node_count,
            |b, &node_count| {
                b.iter(|| {
                    let initial_memory = get_memory_usage();
                    
                    let state = black_box({
                        let mut state = ActivationState::with_capacity(node_count);
                        for i in 0..node_count {
                            if i % 100 == 0 { // Sparse activation
                                state.set_activation(NodeId(i), 0.5);
                            }
                        }
                        state
                    });
                    
                    let final_memory = get_memory_usage();
                    let memory_used = final_memory - initial_memory;
                    
                    drop(state);
                    
                    memory_used
                });
            },
        );
    }
    
    group.finish();
}
```

### Step 4: Quality Benchmarks
```rust
fn benchmark_accuracy_metrics(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let suite = rt.block_on(BenchmarkSuite::new());
    
    let mut group = c.benchmark_group("accuracy_metrics");
    group.significance_level(0.1).sample_size(30);
    
    // Intent recognition accuracy
    group.bench_function("intent_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            let test_cases = create_labeled_test_cases();
            let mut correct = 0;
            let total = test_cases.len();
            
            for (query, expected_intent) in test_cases {
                match suite.small_processor.process_query(&query, &QueryContext::default()).await {
                    Ok(result) => {
                        if std::mem::discriminant(&result.intent.intent_type) == 
                           std::mem::discriminant(&expected_intent) {
                            correct += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            
            let accuracy = correct as f64 / total as f64;
            black_box(accuracy)
        });
    });
    
    // Activation accuracy
    group.bench_function("activation_accuracy", |b| {
        b.to_async(&rt).iter(|| async {
            let accuracy = measure_activation_accuracy(&suite.medium_processor).await;
            black_box(accuracy)
        });
    });
    
    group.finish();
}

async fn measure_activation_accuracy(processor: &QueryProcessor) -> f64 {
    let test_queries = vec![
        ("Animals that fly", vec!["bird", "bat", "insect"]),
        ("Predators", vec!["lion", "tiger", "shark", "eagle"]),
        ("Aquatic animals", vec!["fish", "whale", "dolphin"]),
    ];
    
    let mut total_precision = 0.0;
    let mut count = 0;
    
    for (query, expected_results) in test_queries {
        if let Ok(result) = processor.process_query(query, &QueryContext::default()).await {
            let returned_entities: Vec<String> = result.results.iter()
                .map(|r| r.entity_name.to_lowercase())
                .collect();
            
            let relevant_found = expected_results.iter()
                .filter(|expected| {
                    returned_entities.iter().any(|returned| returned.contains(*expected))
                })
                .count();
            
            let precision = relevant_found as f64 / returned_entities.len().max(1) as f64;
            total_precision += precision;
            count += 1;
        }
    }
    
    if count > 0 {
        total_precision / count as f64
    } else {
        0.0
    }
}
```

### Step 5: Performance Report Generation
```rust
pub struct BenchmarkReport {
    pub timestamp: SystemTime,
    pub performance_metrics: PerformanceMetrics,
    pub scalability_metrics: ScalabilityMetrics,
    pub quality_metrics: QualityMetrics,
    pub targets_met: bool,
}

#[derive(Debug)]
pub struct PerformanceMetrics {
    pub avg_query_time: Duration,
    pub p95_query_time: Duration,
    pub avg_activation_time: Duration,
    pub avg_intent_recognition_time: Duration,
    pub memory_usage_mb: f64,
}

impl BenchmarkReport {
    pub fn generate_from_results(results: &BenchmarkResults) -> Self {
        let performance = PerformanceMetrics {
            avg_query_time: results.calculate_average_query_time(),
            p95_query_time: results.calculate_p95_query_time(),
            avg_activation_time: results.calculate_average_activation_time(),
            avg_intent_recognition_time: results.calculate_average_intent_time(),
            memory_usage_mb: results.calculate_peak_memory_mb(),
        };
        
        let targets_met = Self::validate_targets(&performance, &results.quality);
        
        Self {
            timestamp: SystemTime::now(),
            performance_metrics: performance,
            scalability_metrics: results.scalability.clone(),
            quality_metrics: results.quality.clone(),
            targets_met,
        }
    }
    
    fn validate_targets(perf: &PerformanceMetrics, quality: &QualityMetrics) -> bool {
        perf.avg_query_time < Duration::from_millis(50) &&
        perf.avg_activation_time < Duration::from_millis(10) &&
        perf.avg_intent_recognition_time < Duration::from_millis(200) &&
        perf.memory_usage_mb < 50.0 &&
        quality.intent_accuracy > 0.90 &&
        quality.activation_accuracy > 0.95 &&
        quality.explanation_quality > 0.85
    }
    
    pub fn print_summary(&self) {
        println!("=== Phase 7 Benchmark Report ===");
        println!("Generated: {:?}", self.timestamp);
        println!();
        
        println!("Performance Metrics:");
        println!("  Average Query Time: {:?} (target: <50ms)", self.performance_metrics.avg_query_time);
        println!("  P95 Query Time: {:?}", self.performance_metrics.p95_query_time);
        println!("  Average Activation Time: {:?} (target: <10ms)", self.performance_metrics.avg_activation_time);
        println!("  Intent Recognition Time: {:?} (target: <200ms)", self.performance_metrics.avg_intent_recognition_time);
        println!("  Memory Usage: {:.1}MB (target: <50MB)", self.performance_metrics.memory_usage_mb);
        println!();
        
        println!("Quality Metrics:");
        println!("  Intent Accuracy: {:.1}% (target: >90%)", self.quality_metrics.intent_accuracy * 100.0);
        println!("  Activation Accuracy: {:.1}% (target: >95%)", self.quality_metrics.activation_accuracy * 100.0);
        println!("  Explanation Quality: {:.1}% (target: >85%)", self.quality_metrics.explanation_quality * 100.0);
        println!();
        
        println!("Overall Status: {}", if self.targets_met { "✅ ALL TARGETS MET" } else { "❌ TARGETS NOT MET" });
    }
}
```

## File Locations

- `benches/query_system_benchmarks.rs` - Main benchmark suite
- `benches/performance_benchmarks.rs` - Performance-focused tests
- `benches/scalability_benchmarks.rs` - Scalability validation
- `benches/quality_benchmarks.rs` - Quality metrics
- `src/benchmarks/report.rs` - Report generation

## Success Criteria

- [ ] All performance targets met
- [ ] Scalability benchmarks pass
- [ ] Quality metrics above thresholds
- [ ] Benchmark suite runs without errors
- [ ] Report generation functional
- [ ] Regression detection working

## Execution Commands

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark categories
cargo bench query_processing_latency
cargo bench activation_spreading
cargo bench concurrent_throughput

# Generate detailed report
cargo bench --bench query_system_benchmarks -- --output-format html

# Regression testing
cargo bench --save-baseline main
cargo bench --baseline main
```

## Performance Targets Validation

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Query Processing | < 50ms | TBD | ⚪ |
| Activation Spreading | < 10ms | TBD | ⚪ |
| Intent Recognition | < 200ms | TBD | ⚪ |
| Memory Usage | < 50MB | TBD | ⚪ |
| Concurrent Throughput | > 100/s | TBD | ⚪ |
| Intent Accuracy | > 90% | TBD | ⚪ |
| Activation Accuracy | > 95% | TBD | ⚪ |
| Explanation Quality | > 85% | TBD | ⚪ |

## Quality Gates

- [ ] No performance regressions detected
- [ ] Memory usage stable across runs
- [ ] Quality metrics reproducible
- [ ] Benchmark reliability > 95%
- [ ] All targets achieved

## Final Deliverable

Upon completion of this task, Phase 7 is complete and ready for integration with Phase 8 (MCP Complete Implementation). The benchmarking results will provide baseline metrics for production monitoring and future optimization efforts.