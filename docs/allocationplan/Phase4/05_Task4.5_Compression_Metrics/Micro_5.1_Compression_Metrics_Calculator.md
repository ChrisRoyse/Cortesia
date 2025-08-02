# Micro Phase 5.1: Compression Metrics Calculator

**Estimated Time**: 35 minutes
**Dependencies**: Task 4.4 Complete (Dynamic Optimization)
**Objective**: Implement comprehensive metrics calculation for inheritance system compression analysis

## Task Description

Create a detailed metrics calculation system that can accurately measure compression ratios, storage efficiency, inheritance patterns, and performance characteristics of the compressed hierarchy system.

## Deliverables

Create `src/compression/metrics.rs` with:

1. **CompressionMetrics struct**: Complete metrics collection
2. **Storage calculation**: Accurate byte-level storage measurement
3. **Compression analysis**: Ratio and efficiency calculations
4. **Inheritance metrics**: Property inheritance patterns and rates
5. **Performance metrics**: Query time and cache efficiency

## Success Criteria

- [ ] Calculates compression ratio accurate to within 2%
- [ ] Measures storage usage down to the byte level
- [ ] Tracks inheritance rates across all hierarchy levels
- [ ] Reports cache hit rates and query performance
- [ ] Completes metrics calculation for 10,000 nodes in < 50ms
- [ ] Provides breakdown by node type, property type, and depth level

## Implementation Requirements

```rust
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    // Overall compression
    pub compression_ratio: f64,
    pub total_bytes_saved: usize,
    pub storage_efficiency: f64,
    
    // Node statistics
    pub total_nodes: usize,
    pub hierarchy_depth: u32,
    pub average_fanout: f64,
    pub leaf_nodes: usize,
    
    // Property statistics  
    pub total_properties: usize,
    pub unique_property_names: usize,
    pub inherited_properties: usize,
    pub local_properties: usize,
    pub property_inheritance_rate: f64,
    
    // Exception statistics
    pub total_exceptions: usize,
    pub exception_rate: f64,
    pub exceptions_by_type: HashMap<ExceptionSource, usize>,
    
    // Storage breakdown
    pub storage_breakdown: StorageBreakdown,
    
    // Performance metrics
    pub query_performance: QueryPerformanceMetrics,
    
    // Analysis metadata
    pub calculation_time: Duration,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct StorageBreakdown {
    pub node_storage: usize,
    pub property_storage: usize,
    pub exception_storage: usize,
    pub index_storage: usize,
    pub cache_storage: usize,
    pub metadata_storage: usize,
}

#[derive(Debug, Clone)]
pub struct QueryPerformanceMetrics {
    pub average_query_time: Duration,
    pub cache_hit_rate: f64,
    pub inheritance_chain_length_avg: f32,
    pub queries_per_second: f64,
}

pub struct CompressionMetricsCalculator {
    include_cache_analysis: bool,
    include_performance_analysis: bool,
    sampling_size: usize,
}

impl CompressionMetricsCalculator {
    pub fn new() -> Self;
    
    pub fn calculate_comprehensive_metrics(&self, hierarchy: &InheritanceHierarchy) -> CompressionMetrics;
    
    pub fn calculate_compression_ratio(&self, hierarchy: &InheritanceHierarchy) -> f64;
    
    pub fn calculate_storage_breakdown(&self, hierarchy: &InheritanceHierarchy) -> StorageBreakdown;
    
    pub fn calculate_inheritance_metrics(&self, hierarchy: &InheritanceHierarchy) -> InheritanceMetrics;
    
    pub fn benchmark_query_performance(&self, hierarchy: &InheritanceHierarchy) -> QueryPerformanceMetrics;
    
    pub fn estimate_uncompressed_size(&self, hierarchy: &InheritanceHierarchy) -> usize;
}

#[derive(Debug, Clone)]
struct InheritanceMetrics {
    pub inheritance_depth_distribution: HashMap<u32, usize>,
    pub property_frequency_distribution: HashMap<String, usize>,
    pub inheritance_efficiency_by_level: HashMap<u32, f64>,
}
```

## Test Requirements

Must pass comprehensive metrics calculation tests:
```rust
#[test]
fn test_compression_ratio_accuracy() {
    let hierarchy = create_test_hierarchy_with_known_compression();
    let calculator = CompressionMetricsCalculator::new();
    
    // Test hierarchy has known properties:
    // - 1000 nodes
    // - 50 unique properties  
    // - 80% inheritance rate
    // - Expected 12x compression
    
    let metrics = calculator.calculate_comprehensive_metrics(&hierarchy);
    
    // Verify compression ratio within 2% of expected
    assert!((metrics.compression_ratio - 12.0).abs() < 0.24); // Within 2%
    
    // Verify storage calculations are reasonable
    assert!(metrics.total_bytes_saved > 0);
    assert!(metrics.storage_efficiency > 0.9); // >90% efficiency
}

#[test]
fn test_storage_breakdown_accuracy() {
    let hierarchy = create_large_hierarchy(5000);
    let calculator = CompressionMetricsCalculator::new();
    
    let storage = calculator.calculate_storage_breakdown(&hierarchy);
    
    // All components should have non-zero storage
    assert!(storage.node_storage > 0);
    assert!(storage.property_storage > 0);
    assert!(storage.index_storage > 0);
    
    // Total should equal sum of parts
    let calculated_total = storage.node_storage + 
                          storage.property_storage + 
                          storage.exception_storage + 
                          storage.index_storage + 
                          storage.cache_storage + 
                          storage.metadata_storage;
    
    let measured_total = hierarchy.calculate_actual_memory_usage();
    
    // Should be within 5% of actual measured size
    let diff_ratio = (calculated_total as f64 - measured_total as f64).abs() / measured_total as f64;
    assert!(diff_ratio < 0.05);
}

#[test]
fn test_inheritance_metrics_calculation() {
    let hierarchy = create_multi_level_hierarchy(); // 10 levels deep
    let calculator = CompressionMetricsCalculator::new();
    
    let metrics = calculator.calculate_inheritance_metrics(&hierarchy);
    
    // Should have nodes at multiple depth levels
    assert!(metrics.inheritance_depth_distribution.len() >= 5);
    
    // Inheritance efficiency should generally decrease with depth
    let mut prev_efficiency = 1.0;
    for depth in 0..5 {
        if let Some(&efficiency) = metrics.inheritance_efficiency_by_level.get(&depth) {
            assert!(efficiency <= prev_efficiency + 0.1); // Allow some variance
            prev_efficiency = efficiency;
        }
    }
    
    // Property frequency should include common properties
    assert!(metrics.property_frequency_distribution.len() > 0);
}

#[test]
fn test_query_performance_benchmarking() {
    let hierarchy = create_large_hierarchy(10000);
    let calculator = CompressionMetricsCalculator::new();
    
    let perf_metrics = calculator.benchmark_query_performance(&hierarchy);
    
    // Average query time should meet requirements
    assert!(perf_metrics.average_query_time < Duration::from_micros(100));
    
    // Cache hit rate should be reasonable for repeated queries
    assert!(perf_metrics.cache_hit_rate >= 0.7); // >70% hit rate
    
    // Queries per second should be high
    assert!(perf_metrics.queries_per_second > 10000.0);
    
    // Average inheritance chain should be reasonable
    assert!(perf_metrics.inheritance_chain_length_avg > 1.0);
    assert!(perf_metrics.inheritance_chain_length_avg < 20.0);
}

#[test]
fn test_metrics_calculation_performance() {
    let hierarchy = create_large_hierarchy(10000);
    let calculator = CompressionMetricsCalculator::new();
    
    let start = Instant::now();
    let metrics = calculator.calculate_comprehensive_metrics(&hierarchy);
    let elapsed = start.elapsed();
    
    // Should complete in < 50ms for 10k nodes
    assert!(elapsed < Duration::from_millis(50));
    
    // Metrics should be populated
    assert!(metrics.total_nodes == 10000);
    assert!(metrics.compression_ratio > 1.0);
    assert!(metrics.calculation_time <= elapsed);
}

#[test]
fn test_exception_metrics_accuracy() {
    let hierarchy = create_hierarchy_with_known_exceptions();
    let calculator = CompressionMetricsCalculator::new();
    
    // Hierarchy has 100 nodes, 20 exceptions of known types
    let metrics = calculator.calculate_comprehensive_metrics(&hierarchy);
    
    assert_eq!(metrics.total_exceptions, 20);
    assert_eq!(metrics.exception_rate, 0.2); // 20/100
    
    // Should break down exceptions by type
    let total_by_type: usize = metrics.exceptions_by_type.values().sum();
    assert_eq!(total_by_type, 20);
    
    // Each exception type should have reasonable counts
    for (&source, &count) in &metrics.exceptions_by_type {
        assert!(count > 0);
        assert!(count <= 20);
    }
}

#[test]
fn test_uncompressed_size_estimation() {
    let compressed_hierarchy = create_compressed_hierarchy();
    let calculator = CompressionMetricsCalculator::new();
    
    let estimated_uncompressed = calculator.estimate_uncompressed_size(&compressed_hierarchy);
    let actual_compressed = compressed_hierarchy.calculate_actual_memory_usage();
    
    // Estimated uncompressed should be significantly larger
    assert!(estimated_uncompressed > actual_compressed * 5); // At least 5x larger
    
    // But not unrealistically large
    assert!(estimated_uncompressed < actual_compressed * 100); // Less than 100x
}
```

## File Location
`src/compression/metrics.rs`

## Next Micro Phase
After completion, proceed to Micro 5.2: Storage Analyzer