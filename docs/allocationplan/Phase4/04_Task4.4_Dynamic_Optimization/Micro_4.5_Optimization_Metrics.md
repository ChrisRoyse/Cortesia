# Micro Phase 4.5: Optimization Metrics

**Estimated Time**: 30 minutes
**Dependencies**: Micro 4.4 Complete (Incremental Optimizer)
**Objective**: Implement comprehensive metrics system to measure, track, and analyze optimization effectiveness and hierarchy health

## Task Description

Create a sophisticated metrics collection and analysis system that can quantify optimization improvements, track performance trends, and provide actionable insights for tuning optimization strategies.

## Deliverables

Create `src/optimization/metrics.rs` with:

1. **OptimizationMetrics struct**: Core metrics collection and analysis engine
2. **Performance tracking**: Monitor optimization execution times and resource usage
3. **Quality assessment**: Measure hierarchy structural quality and improvement
4. **Trend analysis**: Track optimization effectiveness over time
5. **Reporting system**: Generate detailed optimization reports and recommendations

## Success Criteria

- [ ] Captures 100% of optimization operations with detailed metrics
- [ ] Calculates hierarchy quality scores with < 1ms overhead
- [ ] Tracks performance trends with configurable time windows
- [ ] Generates actionable optimization recommendations
- [ ] Memory usage for metrics < 2% of total system memory
- [ ] Provides real-time optimization dashboard data

## Implementation Requirements

```rust
pub struct OptimizationMetrics {
    collection_enabled: bool,
    retention_period: Duration,
    sampling_rate: f32,
    quality_cache: LruCache<NodeId, QualityScore>,
    performance_history: VecDeque<PerformanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyQualityScore {
    pub overall_score: f32,
    pub depth_score: f32,
    pub balance_score: f32,
    pub redundancy_score: f32,
    pub inheritance_efficiency: f32,
    pub cache_locality_score: f32,
    pub memory_efficiency: f32,
    pub access_pattern_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEvent {
    pub timestamp: Instant,
    pub operation_type: OptimizationType,
    pub execution_time: Duration,
    pub nodes_affected: usize,
    pub quality_improvement: f32,
    pub memory_delta: isize,
    pub success: bool,
    pub error_details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub hierarchy_size: usize,
    pub optimization_queue_length: usize,
    pub average_response_time: Duration,
    pub memory_usage: usize,
    pub cpu_utilization: f32,
    pub optimization_rate: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub time_period: (Instant, Instant),
    pub total_optimizations: usize,
    pub successful_optimizations: usize,
    pub average_execution_time: Duration,
    pub total_quality_improvement: f32,
    pub memory_savings: isize,
    pub performance_trends: Vec<TrendAnalysis>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub change_rate: f32,
    pub confidence: f32,
    pub prediction: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub expected_benefit: f32,
    pub estimated_cost: Duration,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    PerformanceTuning,
    StructuralOptimization,
    ResourceManagement,
    ConfigurationChange,
    ArchitecturalImprovement,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

impl OptimizationMetrics {
    pub fn new() -> Self;
    
    pub fn record_optimization_event(&mut self, event: OptimizationEvent);
    
    pub fn calculate_hierarchy_quality(&self, hierarchy: &InheritanceHierarchy) -> HierarchyQualityScore;
    
    pub fn take_performance_snapshot(&mut self, hierarchy: &InheritanceHierarchy, optimizer: &IncrementalOptimizer);
    
    pub fn generate_optimization_report(&self, time_window: Duration) -> OptimizationReport;
    
    pub fn analyze_trends(&self, metric_name: &str, time_window: Duration) -> TrendAnalysis;
    
    pub fn generate_recommendations(&self, hierarchy: &InheritanceHierarchy) -> Vec<OptimizationRecommendation>;
    
    pub fn get_real_time_dashboard_data(&self) -> DashboardData;
    
    pub fn calculate_optimization_roi(&self, operation_type: OptimizationType, time_window: Duration) -> f32;
    
    pub fn export_metrics(&self, format: ExportFormat) -> Result<String, MetricsError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub current_quality_score: HierarchyQualityScore,
    pub optimization_rate: f32,
    pub average_response_time: Duration,
    pub recent_improvements: Vec<OptimizationEvent>,
    pub performance_trend: TrendDirection,
    pub active_recommendations: usize,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemHealth {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
    InfluxDB,
}

#[derive(Debug, Clone)]
pub struct QualityScore {
    score: f32,
    timestamp: Instant,
    components: HashMap<String, f32>,
}
```

## Test Requirements

Must pass optimization metrics tests:
```rust
#[test]
fn test_hierarchy_quality_calculation() {
    let hierarchy = create_test_hierarchy();
    let metrics = OptimizationMetrics::new();
    
    let start = Instant::now();
    let quality = metrics.calculate_hierarchy_quality(&hierarchy);
    let elapsed = start.elapsed();
    
    // Should calculate quickly
    assert!(elapsed < Duration::from_millis(1));
    
    // Quality score should be reasonable
    assert!(quality.overall_score >= 0.0 && quality.overall_score <= 1.0);
    assert!(quality.depth_score >= 0.0 && quality.depth_score <= 1.0);
    assert!(quality.balance_score >= 0.0 && quality.balance_score <= 1.0);
    assert!(quality.redundancy_score >= 0.0 && quality.redundancy_score <= 1.0);
    
    // Components should contribute to overall score
    let component_avg = (quality.depth_score + quality.balance_score + 
                        quality.redundancy_score + quality.inheritance_efficiency) / 4.0;
    let diff = (quality.overall_score - component_avg).abs();
    assert!(diff < 0.3); // Overall should be close to component average
}

#[test]
fn test_optimization_event_recording() {
    let mut metrics = OptimizationMetrics::new();
    
    let event = OptimizationEvent {
        timestamp: Instant::now(),
        operation_type: OptimizationType::LocalRebalancing { subtree_root: NodeId::new() },
        execution_time: Duration::from_millis(15),
        nodes_affected: 10,
        quality_improvement: 0.25,
        memory_delta: -1024, // Saved memory
        success: true,
        error_details: None,
    };
    
    metrics.record_optimization_event(event.clone());
    
    // Generate report to verify event was recorded
    let report = metrics.generate_optimization_report(Duration::from_secs(60));
    assert_eq!(report.total_optimizations, 1);
    assert_eq!(report.successful_optimizations, 1);
    assert_eq!(report.total_quality_improvement, 0.25);
    assert_eq!(report.memory_savings, -1024);
}

#[test]
fn test_performance_snapshot_collection() {
    let hierarchy = create_medium_hierarchy(500);
    let optimizer = IncrementalOptimizer::new();
    let mut metrics = OptimizationMetrics::new();
    
    // Take multiple snapshots
    for _ in 0..5 {
        metrics.take_performance_snapshot(&hierarchy, &optimizer);
        std::thread::sleep(Duration::from_millis(10));
    }
    
    assert_eq!(metrics.performance_history.len(), 5);
    
    // Snapshots should have reasonable data
    for snapshot in &metrics.performance_history {
        assert_eq!(snapshot.hierarchy_size, 500);
        assert!(snapshot.average_response_time > Duration::ZERO);
        assert!(snapshot.memory_usage > 0);
    }
}

#[test]
fn test_trend_analysis() {
    let mut metrics = OptimizationMetrics::new();
    
    // Create artificial trend data (improving performance)
    let base_time = Instant::now() - Duration::from_secs(60);
    for i in 0..10 {
        let snapshot = PerformanceSnapshot {
            timestamp: base_time + Duration::from_secs(i * 6),
            hierarchy_size: 1000,
            optimization_queue_length: 10 - i, // Decreasing queue
            average_response_time: Duration::from_millis(20 - i), // Improving response time
            memory_usage: 1000000 - (i * 10000), // Decreasing memory
            cpu_utilization: 0.8 - (i as f32 * 0.05), // Decreasing CPU
            optimization_rate: 10.0 + (i as f32), // Increasing optimization rate
        };
        metrics.performance_history.push_back(snapshot);
    }
    
    let trend = metrics.analyze_trends("average_response_time", Duration::from_secs(60));
    assert_eq!(trend.trend_direction, TrendDirection::Improving);
    assert!(trend.change_rate < 0.0); // Negative because response time is decreasing (improving)
    assert!(trend.confidence > 0.7); // Should be confident in clear trend
}

#[test]
fn test_recommendation_generation() {
    let hierarchy = create_suboptimal_hierarchy(); // Create hierarchy with known issues
    let metrics = OptimizationMetrics::new();
    
    let recommendations = metrics.generate_recommendations(&hierarchy);
    
    assert!(!recommendations.is_empty());
    
    // Check recommendation quality
    for rec in &recommendations {
        assert!(!rec.description.is_empty());
        assert!(rec.expected_benefit >= 0.0);
        assert!(rec.estimated_cost > Duration::ZERO);
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
    }
    
    // Should prioritize high-impact recommendations
    let high_priority: Vec<_> = recommendations.iter()
        .filter(|r| r.priority >= RecommendationPriority::High)
        .collect();
    assert!(!high_priority.is_empty());
}

#[test]
fn test_real_time_dashboard_data() {
    let hierarchy = create_balanced_hierarchy(200);
    let mut metrics = OptimizationMetrics::new();
    
    // Add some optimization events
    for i in 0..5 {
        let event = OptimizationEvent {
            timestamp: Instant::now(),
            operation_type: OptimizationType::PropertyConsolidation { nodes: vec![NodeId::new()] },
            execution_time: Duration::from_millis(5 + i),
            nodes_affected: 10 + i,
            quality_improvement: 0.1 + (i as f32 * 0.05),
            memory_delta: -(100 * (i as isize + 1)),
            success: true,
            error_details: None,
        };
        metrics.record_optimization_event(event);
    }
    
    let dashboard = metrics.get_real_time_dashboard_data();
    
    assert!(dashboard.current_quality_score.overall_score > 0.0);
    assert!(dashboard.optimization_rate >= 0.0);
    assert!(dashboard.recent_improvements.len() <= 5);
    assert!(matches!(dashboard.system_health, SystemHealth::Good | SystemHealth::Excellent));
}

#[test]
fn test_optimization_roi_calculation() {
    let mut metrics = OptimizationMetrics::new();
    
    // Record several optimization events of the same type
    for i in 0..10 {
        let event = OptimizationEvent {
            timestamp: Instant::now() - Duration::from_secs(i * 6),
            operation_type: OptimizationType::LocalRebalancing { subtree_root: NodeId::new() },
            execution_time: Duration::from_millis(20),
            nodes_affected: 50,
            quality_improvement: 0.3,
            memory_delta: -2048,
            success: true,
            error_details: None,
        };
        metrics.record_optimization_event(event);
    }
    
    let roi = metrics.calculate_optimization_roi(
        OptimizationType::LocalRebalancing { subtree_root: NodeId::new() },
        Duration::from_secs(60)
    );
    
    assert!(roi > 0.0); // Should have positive ROI
    assert!(roi < 100.0); // Should be reasonable value
}

#[test]
fn test_metrics_memory_overhead() {
    let hierarchy = create_large_hierarchy(2000);
    let mut metrics = OptimizationMetrics::new();
    
    let initial_memory = std::mem::size_of_val(&metrics);
    
    // Generate lots of metrics data
    for i in 0..1000 {
        let event = OptimizationEvent {
            timestamp: Instant::now(),
            operation_type: OptimizationType::RedundancyRemoval { node: NodeId::new() },
            execution_time: Duration::from_millis(5),
            nodes_affected: 1,
            quality_improvement: 0.01,
            memory_delta: -100,
            success: true,
            error_details: None,
        };
        metrics.record_optimization_event(event);
        
        if i % 100 == 0 {
            metrics.take_performance_snapshot(&hierarchy, &IncrementalOptimizer::new());
        }
    }
    
    let final_memory = std::mem::size_of_val(&metrics);
    let hierarchy_memory = hierarchy.estimated_memory_usage();
    let overhead_percentage = ((final_memory - initial_memory) as f32 / hierarchy_memory as f32) * 100.0;
    
    assert!(overhead_percentage < 2.0); // < 2% memory overhead
}

#[test]
fn test_metrics_export() {
    let mut metrics = OptimizationMetrics::new();
    
    // Add some data
    let event = OptimizationEvent {
        timestamp: Instant::now(),
        operation_type: OptimizationType::StructuralSimplification { area: vec![NodeId::new()] },
        execution_time: Duration::from_millis(10),
        nodes_affected: 5,
        quality_improvement: 0.15,
        memory_delta: -512,
        success: true,
        error_details: None,
    };
    metrics.record_optimization_event(event);
    
    // Test JSON export
    let json_export = metrics.export_metrics(ExportFormat::Json);
    assert!(json_export.is_ok());
    let json_data = json_export.unwrap();
    assert!(json_data.contains("optimization"));
    assert!(json_data.contains("quality_improvement"));
    
    // Test CSV export
    let csv_export = metrics.export_metrics(ExportFormat::Csv);
    assert!(csv_export.is_ok());
    let csv_data = csv_export.unwrap();
    assert!(csv_data.contains("timestamp"));
    assert!(csv_data.contains("execution_time"));
}

#[test]
fn test_report_generation() {
    let mut metrics = OptimizationMetrics::new();
    
    // Add events spanning different time periods
    let now = Instant::now();
    for i in 0..20 {
        let event = OptimizationEvent {
            timestamp: now - Duration::from_secs(i * 30), // Events every 30 seconds
            operation_type: OptimizationType::PropertyConsolidation { nodes: vec![NodeId::new()] },
            execution_time: Duration::from_millis(10 + (i % 5)),
            nodes_affected: 10,
            quality_improvement: 0.1 + ((i % 3) as f32 * 0.05),
            memory_delta: -100,
            success: i % 10 != 0, // 10% failure rate
            error_details: if i % 10 == 0 { Some("Test error".to_string()) } else { None },
        };
        metrics.record_optimization_event(event);
    }
    
    let report = metrics.generate_optimization_report(Duration::from_secs(300)); // 5 minutes
    
    assert_eq!(report.total_optimizations, 11); // Events in last 5 minutes
    assert_eq!(report.successful_optimizations, 10); // 1 failure
    assert!(report.average_execution_time > Duration::ZERO);
    assert!(report.total_quality_improvement > 0.0);
    assert!(!report.recommendations.is_empty());
}
```

## File Location
`src/optimization/metrics.rs`

## Next Micro Phase
After completion, proceed to Micro 4.6: Optimization Integration Tests