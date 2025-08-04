# Micro Phase 5.2: Storage Analyzer

**Estimated Time**: 30 minutes
**Dependencies**: Micro 5.1 Complete (Compression Metrics Calculator)
**Objective**: Implement detailed storage usage pattern analysis for deep insights into memory allocation and efficiency

## Task Description

Create a comprehensive storage analysis system that provides detailed breakdown of memory usage patterns, identifies optimization opportunities, and tracks storage efficiency across different components of the inheritance hierarchy system.

## Deliverables

Create `src/compression/storage_analyzer.rs` with:

1. **StorageAnalyzer struct**: Advanced storage pattern analysis
2. **Memory usage tracking**: Detailed component-level analysis
3. **Pattern detection**: Identify storage inefficiencies and bottlenecks
4. **Optimization recommendations**: Actionable insights for improvement
5. **Historical tracking**: Storage usage trends over time

## Success Criteria

- [ ] Analyzes storage patterns down to individual property level
- [ ] Identifies memory fragmentation and waste within 1% accuracy
- [ ] Detects storage hotspots and inefficient allocations
- [ ] Provides optimization recommendations with projected savings
- [ ] Completes analysis of 10,000 nodes in < 25ms
- [ ] Tracks storage trends and predicts future usage patterns

## Implementation Requirements

```rust
#[derive(Debug, Clone)]
pub struct StorageAnalyzer {
    track_historical_data: bool,
    enable_fragmentation_analysis: bool,
    optimization_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct StorageAnalysis {
    // Detailed component breakdown
    pub component_breakdown: ComponentStorageBreakdown,
    
    // Memory efficiency metrics
    pub fragmentation_analysis: FragmentationAnalysis,
    pub allocation_efficiency: f64,
    pub waste_percentage: f64,
    
    // Pattern analysis
    pub storage_patterns: Vec<StoragePattern>,
    pub hotspots: Vec<StorageHotspot>,
    pub inefficiencies: Vec<StorageInefficiency>,
    
    // Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub projected_savings: usize,
    
    // Trend analysis
    pub historical_trends: Option<StorageTrends>,
    
    // Analysis metadata
    pub analysis_time: Duration,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct ComponentStorageBreakdown {
    // Node storage details
    pub node_headers: usize,
    pub node_metadata: usize,
    pub node_indices: usize,
    
    // Property storage details
    pub property_names: usize,
    pub property_values: usize,
    pub property_type_info: usize,
    pub property_inheritance_data: usize,
    
    // Exception storage details
    pub exception_metadata: usize,
    pub exception_data: usize,
    pub exception_indices: usize,
    
    // Cache storage details
    pub cache_entries: usize,
    pub cache_metadata: usize,
    pub cache_overhead: usize,
    
    // System overhead
    pub alignment_padding: usize,
    pub struct_overhead: usize,
    pub pointer_storage: usize,
}

#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    pub internal_fragmentation: f64,
    pub external_fragmentation: f64,
    pub fragmented_blocks: usize,
    pub largest_free_block: usize,
    pub fragmentation_impact: FragmentationImpact,
}

#[derive(Debug, Clone)]
pub enum FragmentationImpact {
    Minimal,
    Moderate { estimated_overhead: usize },
    Severe { recommended_defrag: bool },
}

#[derive(Debug, Clone)]
pub struct StoragePattern {
    pub pattern_type: StoragePatternType,
    pub frequency: usize,
    pub impact: StorageImpact,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone)]
pub enum StoragePatternType {
    DuplicateProperties,
    SparseInheritance,
    OverindexedData,
    UnderutilizedCache,
    RedundantMetadata,
}

#[derive(Debug, Clone)]
pub struct StorageHotspot {
    pub location: HotspotLocation,
    pub usage_percentage: f64,
    pub access_frequency: usize,
    pub optimization_priority: Priority,
}

#[derive(Debug, Clone)]
pub enum HotspotLocation {
    NodeCluster { node_range: Range<NodeId> },
    PropertyType { property_name: String },
    InheritanceLevel { depth: u32 },
    CacheRegion { cache_type: String },
}

#[derive(Debug, Clone)]
pub struct StorageInefficiency {
    pub inefficiency_type: InefficiencyType,
    pub affected_components: Vec<String>,
    pub waste_amount: usize,
    pub severity: Severity,
    pub recommended_action: String,
}

#[derive(Debug, Clone)]
pub enum InefficiencyType {
    OverAllocation,
    UnderUtilization,
    PoorAlignment,
    RedundantStorage,
    SuboptimalLayout,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_type: OptimizationType,
    pub description: String,
    pub projected_savings: usize,
    pub implementation_effort: ImplementationEffort,
    pub priority: Priority,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    MemoryReorganization,
    CacheOptimization,
    PropertyDeduplication,
    IndexRestructuring,
    AlignmentImprovement,
}

#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct StorageTrends {
    pub growth_rate: f64,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub predicted_usage: Vec<UsagePrediction>,
}

#[derive(Debug, Clone)]
pub struct StorageImpact {
    pub memory_overhead: usize,
    pub performance_impact: f64,
    pub scalability_concern: bool,
}

impl StorageAnalyzer {
    pub fn new() -> Self;
    
    pub fn with_options(
        track_historical: bool,
        enable_fragmentation: bool,
        optimization_threshold: f64
    ) -> Self;
    
    pub fn analyze_storage(&self, hierarchy: &InheritanceHierarchy, metrics: &CompressionMetrics) -> StorageAnalysis;
    
    pub fn analyze_component_breakdown(&self, hierarchy: &InheritanceHierarchy) -> ComponentStorageBreakdown;
    
    pub fn analyze_fragmentation(&self, hierarchy: &InheritanceHierarchy) -> FragmentationAnalysis;
    
    pub fn detect_storage_patterns(&self, hierarchy: &InheritanceHierarchy) -> Vec<StoragePattern>;
    
    pub fn identify_hotspots(&self, hierarchy: &InheritanceHierarchy) -> Vec<StorageHotspot>;
    
    pub fn find_inefficiencies(&self, hierarchy: &InheritanceHierarchy) -> Vec<StorageInefficiency>;
    
    pub fn generate_optimization_recommendations(&self, analysis: &StorageAnalysis) -> Vec<OptimizationRecommendation>;
    
    pub fn calculate_projected_savings(&self, recommendations: &[OptimizationRecommendation]) -> usize;
    
    pub fn track_historical_trends(&self, current_analysis: &StorageAnalysis) -> Option<StorageTrends>;
}
```

## Test Requirements

Must pass comprehensive storage analysis tests:
```rust
#[test]
fn test_component_breakdown_accuracy() {
    let hierarchy = create_large_hierarchy(5000);
    let metrics = calculate_test_metrics(&hierarchy);
    let analyzer = StorageAnalyzer::new();
    
    let breakdown = analyzer.analyze_component_breakdown(&hierarchy);
    
    // All major components should have measurable storage
    assert!(breakdown.node_headers > 0);
    assert!(breakdown.property_names > 0);
    assert!(breakdown.property_values > 0);
    assert!(breakdown.cache_entries > 0);
    
    // Total should match actual memory usage within 1%
    let calculated_total = breakdown.node_headers + breakdown.node_metadata + 
                          breakdown.property_names + breakdown.property_values +
                          breakdown.cache_entries + breakdown.struct_overhead;
    
    let actual_total = hierarchy.calculate_actual_memory_usage();
    let accuracy = (calculated_total as f64 - actual_total as f64).abs() / actual_total as f64;
    assert!(accuracy < 0.01); // Within 1%
}

#[test]
fn test_fragmentation_analysis() {
    let fragmented_hierarchy = create_fragmented_hierarchy();
    let analyzer = StorageAnalyzer::with_options(false, true, 0.1);
    
    let fragmentation = analyzer.analyze_fragmentation(&fragmented_hierarchy);
    
    // Should detect fragmentation in test hierarchy
    assert!(fragmentation.internal_fragmentation > 0.0);
    assert!(fragmentation.fragmented_blocks > 0);
    
    // Impact assessment should be reasonable
    match fragmentation.fragmentation_impact {
        FragmentationImpact::Severe { recommended_defrag } => assert!(recommended_defrag),
        FragmentationImpact::Moderate { estimated_overhead } => assert!(estimated_overhead > 0),
        FragmentationImpact::Minimal => {}, // OK for some test cases
    }
}

#[test]
fn test_storage_pattern_detection() {
    let hierarchy = create_hierarchy_with_patterns();
    let analyzer = StorageAnalyzer::new();
    
    let patterns = analyzer.detect_storage_patterns(&hierarchy);
    
    // Should detect at least some patterns
    assert!(!patterns.is_empty());
    
    // Patterns should have meaningful data
    for pattern in patterns {
        assert!(pattern.frequency > 0);
        assert!(pattern.optimization_potential >= 0.0);
        assert!(pattern.optimization_potential <= 1.0);
    }
}

#[test]
fn test_hotspot_identification() {
    let hierarchy = create_hierarchy_with_hotspots();
    let analyzer = StorageAnalyzer::new();
    
    let hotspots = analyzer.identify_hotspots(&hierarchy);
    
    // Should identify storage hotspots
    assert!(!hotspots.is_empty());
    
    // Hotspots should have valid data
    for hotspot in hotspots {
        assert!(hotspot.usage_percentage > 0.0);
        assert!(hotspot.usage_percentage <= 100.0);
        assert!(hotspot.access_frequency > 0);
    }
}

#[test]
fn test_inefficiency_detection() {
    let inefficient_hierarchy = create_inefficient_hierarchy();
    let analyzer = StorageAnalyzer::new();
    
    let inefficiencies = analyzer.find_inefficiencies(&inefficient_hierarchy);
    
    // Should detect inefficiencies in test hierarchy
    assert!(!inefficiencies.is_empty());
    
    // Each inefficiency should have actionable information
    for inefficiency in inefficiencies {
        assert!(inefficiency.waste_amount > 0);
        assert!(!inefficiency.recommended_action.is_empty());
        assert!(!inefficiency.affected_components.is_empty());
    }
}

#[test]
fn test_optimization_recommendations() {
    let hierarchy = create_suboptimal_hierarchy();
    let analyzer = StorageAnalyzer::new();
    let analysis = analyzer.analyze_storage(&hierarchy, &create_test_metrics(&hierarchy));
    
    let recommendations = analyzer.generate_optimization_recommendations(&analysis);
    
    // Should provide actionable recommendations
    assert!(!recommendations.is_empty());
    
    // Each recommendation should be complete
    for rec in &recommendations {
        assert!(!rec.description.is_empty());
        assert!(rec.projected_savings > 0);
    }
    
    // Total projected savings should be reasonable
    let total_savings = analyzer.calculate_projected_savings(&recommendations);
    assert!(total_savings > 0);
    assert!(total_savings < hierarchy.calculate_actual_memory_usage());
}

#[test]
fn test_analysis_performance() {
    let large_hierarchy = create_large_hierarchy(10000);
    let metrics = calculate_test_metrics(&large_hierarchy);
    let analyzer = StorageAnalyzer::new();
    
    let start = Instant::now();
    let analysis = analyzer.analyze_storage(&large_hierarchy, &metrics);
    let elapsed = start.elapsed();
    
    // Should complete analysis in < 25ms for 10k nodes
    assert!(elapsed < Duration::from_millis(25));
    
    // Analysis should be comprehensive
    assert!(!analysis.component_breakdown.node_headers == 0);
    assert!(!analysis.storage_patterns.is_empty());
    assert!(analysis.allocation_efficiency > 0.0);
}

#[test]
fn test_projected_savings_accuracy() {
    let hierarchy = create_hierarchy_with_known_waste();
    let analyzer = StorageAnalyzer::new();
    let analysis = analyzer.analyze_storage(&hierarchy, &create_test_metrics(&hierarchy));
    
    // Test hierarchy has 1MB of known waste
    let total_waste: usize = analysis.inefficiencies.iter()
        .map(|i| i.waste_amount)
        .sum();
    
    // Should detect most of the waste (within 10%)
    assert!(total_waste > 900_000); // At least 90% of 1MB
    assert!(total_waste < 1_100_000); // At most 110% of 1MB
    
    // Projected savings should be reasonable
    assert!(analysis.projected_savings <= total_waste);
    assert!(analysis.projected_savings > total_waste / 2); // At least 50% recoverable
}
```

## File Location
`src/compression/storage_analyzer.rs`

## Next Micro Phase
After completion, proceed to Micro 5.3: Compression Verifier