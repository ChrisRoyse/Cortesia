# Task 021: Concurrent Results Analysis Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 008, 017, 018, and 019 (PerformanceBenchmark, EnhancedPerformanceMetrics, ConcurrentBenchmark, and AdvancedPercentileCalculations). The concurrent results analysis provides comprehensive analysis of multi-threaded test results with thread-level performance metrics and bottleneck identification.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement comprehensive concurrent test result analysis with thread-level performance metrics, contention analysis, bottleneck identification, resource utilization per thread, and scalability curve generation for production-grade performance analysis.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. Comprehensive concurrent test result analysis structure
3. Thread-level performance metrics with detailed statistics
4. Contention analysis and bottleneck identification algorithms
5. Resource utilization tracking per thread
6. Scalability curve generation and analysis
7. Performance anomaly detection in concurrent scenarios

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentResultsAnalyzer {
    test_results: Vec<ConcurrentTestResult>,
    thread_analytics: ThreadAnalytics,
    contention_analyzer: ContentionAnalyzer,
    bottleneck_detector: BottleneckDetector,
    scalability_analyzer: ScalabilityAnalyzer,
    anomaly_detector: PerformanceAnomalyDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveConcurrentAnalysis {
    pub summary: ConcurrentTestSummary,
    pub thread_analysis: DetailedThreadAnalysis,
    pub contention_analysis: ContentionAnalysisResult,
    pub bottleneck_analysis: BottleneckAnalysisResult,
    pub resource_analysis: ResourceUtilizationAnalysis,
    pub scalability_analysis: ScalabilityAnalysisResult,
    pub anomaly_analysis: AnomalyAnalysisResult,
    pub recommendations: Vec<PerformanceRecommendation>,
    pub analysis_metadata: AnalysisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentTestSummary {
    pub total_tests: usize,
    pub total_threads: usize,
    pub total_requests: usize,
    pub overall_success_rate: f64,
    pub average_throughput_qps: f64,
    pub peak_throughput_qps: f64,
    pub average_response_time_ms: f64,
    pub response_time_percentiles: PercentileSuite,
    pub test_duration_seconds: f64,
    pub concurrent_users_range: (usize, usize),
    pub performance_score: f64, // 0-100 overall performance rating
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedThreadAnalysis {
    pub thread_performances: Vec<EnhancedThreadPerformance>,
    pub thread_performance_variance: f64,
    pub fastest_thread: ThreadPerformanceHighlight,
    pub slowest_thread: ThreadPerformanceHighlight,
    pub thread_efficiency_distribution: Vec<f64>,
    pub cross_thread_correlation: ThreadCorrelationMatrix,
    pub load_balancing_effectiveness: f64, // 0-100 score
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedThreadPerformance {
    pub thread_id: usize,
    pub requests_completed: usize,
    pub requests_per_second: f64,
    pub latency_statistics: ThreadLatencyStatistics,
    pub error_analysis: ThreadErrorAnalysis,
    pub resource_usage: ThreadResourceUsage,
    pub performance_consistency: f64, // Coefficient of variation
    pub efficiency_score: f64, // Requests per CPU time
    pub contention_score: f64, // How much this thread was affected by contention
    pub anomaly_flags: Vec<ThreadAnomalyFlag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadLatencyStatistics {
    pub mean_latency_ms: f64,
    pub median_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub standard_deviation_ms: f64,
    pub latency_trend: LatencyTrend,
    pub outlier_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadErrorAnalysis {
    pub total_errors: usize,
    pub error_rate: f64,
    pub error_types: HashMap<String, usize>,
    pub error_clustering: Vec<ErrorCluster>,
    pub error_pattern: ErrorPattern,
    pub recovery_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadResourceUsage {
    pub cpu_time_ms: f64,
    pub memory_peak_mb: f64,
    pub memory_average_mb: f64,
    pub io_operations: usize,
    pub network_bytes: u64,
    pub context_switches: usize,
    pub cache_hit_rate: f64,
    pub resource_efficiency: f64, // Performance per resource unit
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyTrend {
    Improving,
    Degrading,
    Stable,
    Volatile,
    BiModal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorPattern {
    Random,
    Clustered,
    Periodic,
    Cascading,
    InitialSpike,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCluster {
    pub start_time_ms: f64,
    pub duration_ms: f64,
    pub error_count: usize,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadAnomalyFlag {
    UnusuallyHighLatency,
    UnusuallyLowThroughput,
    HighErrorRate,
    ResourceStarvation,
    ContentionVictim,
    MemoryLeak,
    DeadlockParticipant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPerformanceHighlight {
    pub thread_id: usize,
    pub performance_metric: f64,
    pub reason: String,
    pub impact_on_overall: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadCorrelationMatrix {
    pub correlations: HashMap<(usize, usize), f64>, // Thread pair -> correlation coefficient
    pub strong_correlations: Vec<ThreadCorrelation>,
    pub anti_correlations: Vec<ThreadCorrelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadCorrelation {
    pub thread_a: usize,
    pub thread_b: usize,
    pub correlation_coefficient: f64,
    pub correlation_type: CorrelationType,
    pub statistical_significance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    PositiveStrong,     // > 0.7
    PositiveModerate,   // 0.3 to 0.7
    NegativeModerate,   // -0.7 to -0.3
    NegativeStrong,     // < -0.7
    NoCorrelation,      // -0.3 to 0.3
}

// Contention Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionAnalyzer {
    lock_contention_metrics: Vec<LockContentionMetric>,
    resource_contention_scores: HashMap<String, f64>,
    temporal_analysis: TemporalContentionAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionAnalysisResult {
    pub overall_contention_score: f64, // 0-100, higher = more contention
    pub contention_hotspots: Vec<ContentionHotspot>,
    pub temporal_patterns: Vec<ContentionPattern>,
    pub affected_threads: Vec<usize>,
    pub contention_impact_on_performance: f64, // Percentage performance loss
    pub mitigation_suggestions: Vec<ContentionMitigation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionHotspot {
    pub resource_name: String,
    pub contention_severity: ContentionSeverity,
    pub affected_thread_count: usize,
    pub average_wait_time_ms: f64,
    pub peak_contention_time: std::time::SystemTime,
    pub contention_frequency: f64, // Contentions per second
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentionSeverity {
    Minimal,
    Low,
    Moderate,
    High,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionPattern {
    pub pattern_type: ContentionPatternType,
    pub frequency: f64,
    pub impact_score: f64,
    pub affected_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentionPatternType {
    BurstContention,    // Sudden spikes
    PeriodicContention, // Regular intervals
    GradualIncrease,    // Slowly building up
    RandomContention,   // No clear pattern
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentionMitigation {
    ReduceConcurrency,
    OptimizeLockGranularity,
    ImplementLockFreeStructures,
    AddConnectionPooling,
    BalanceLoad,
    CacheOptimization,
}

// Bottleneck Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckDetector {
    performance_profiles: Vec<PerformanceProfile>,
    bottleneck_candidates: Vec<BottleneckCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysisResult {
    pub primary_bottleneck: Option<IdentifiedBottleneck>,
    pub secondary_bottlenecks: Vec<IdentifiedBottleneck>,
    pub bottleneck_severity_score: f64, // 0-100
    pub performance_limiting_factor: LimitingFactor,
    pub capacity_estimates: CapacityEstimates,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f64, // 0-100
    pub impact_on_throughput: f64, // Percentage
    pub affected_operations: Vec<String>,
    pub evidence: BottleneckEvidence,
    pub remediation_priority: RemediationPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    DiskIO,
    NetworkIO,
    DatabaseConnection,
    Lock,
    Algorithm,
    ExternalService,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitingFactor {
    ComputeBound,
    IOBound,
    MemoryBound,
    ConcurrencyBound,
    NetworkBound,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEstimates {
    pub current_utilization_percent: f64,
    pub estimated_max_throughput: f64,
    pub headroom_percentage: f64,
    pub scale_limit_users: usize,
}

// Implementation
impl ConcurrentResultsAnalyzer {
    pub fn new() -> Self {
        Self {
            test_results: Vec::new(),
            thread_analytics: ThreadAnalytics::new(),
            contention_analyzer: ContentionAnalyzer::new(),
            bottleneck_detector: BottleneckDetector::new(),
            scalability_analyzer: ScalabilityAnalyzer::new(),
            anomaly_detector: PerformanceAnomalyDetector::new(),
        }
    }
    
    pub fn add_test_result(&mut self, result: ConcurrentTestResult) {
        self.test_results.push(result);
    }
    
    pub fn analyze_results(&mut self) -> Result<ComprehensiveConcurrentAnalysis> {
        if self.test_results.is_empty() {
            return Err(anyhow::anyhow!("No test results to analyze"));
        }
        
        let analysis_start = std::time::Instant::now();
        
        // Generate summary
        let summary = self.generate_summary();
        
        // Analyze threads
        let thread_analysis = self.analyze_threads()?;
        
        // Analyze contention
        let contention_analysis = self.analyze_contention()?;
        
        // Detect bottlenecks
        let bottleneck_analysis = self.detect_bottlenecks()?;
        
        // Analyze resource utilization
        let resource_analysis = self.analyze_resource_utilization()?;
        
        // Analyze scalability
        let scalability_analysis = self.analyze_scalability()?;
        
        // Detect anomalies
        let anomaly_analysis = self.detect_anomalies()?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &thread_analysis,
            &contention_analysis,
            &bottleneck_analysis,
        );
        
        let analysis_duration = analysis_start.elapsed();
        
        Ok(ComprehensiveConcurrentAnalysis {
            summary,
            thread_analysis,
            contention_analysis,
            bottleneck_analysis,
            resource_analysis,
            scalability_analysis,
            anomaly_analysis,
            recommendations,
            analysis_metadata: AnalysisMetadata {
                analysis_duration_ms: analysis_duration.as_millis() as f64,
                analyzer_version: "1.0.0".to_string(),
                analysis_timestamp: std::time::SystemTime::now(),
                total_data_points: self.calculate_total_data_points(),
            },
        })
    }
    
    fn generate_summary(&self) -> ConcurrentTestSummary {
        let total_tests = self.test_results.len();
        let total_requests: usize = self.test_results.iter().map(|r| r.total_requests).sum();
        let total_successful: usize = self.test_results.iter().map(|r| r.successful_requests).sum();
        
        let overall_success_rate = if total_requests > 0 {
            (total_successful as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        let avg_throughput: f64 = self.test_results.iter().map(|r| r.requests_per_second).sum::<f64>() / total_tests as f64;
        let peak_throughput = self.test_results.iter().map(|r| r.requests_per_second).fold(0.0, f64::max);
        
        let avg_response_time: f64 = self.test_results.iter().map(|r| r.average_response_time_ms).sum::<f64>() / total_tests as f64;
        
        let total_threads: usize = self.test_results.iter().map(|r| r.concurrent_users).sum();
        
        let min_users = self.test_results.iter().map(|r| r.concurrent_users).min().unwrap_or(0);
        let max_users = self.test_results.iter().map(|r| r.concurrent_users).max().unwrap_or(0);
        
        // Calculate performance score based on multiple factors
        let success_rate_score = overall_success_rate;
        let throughput_score = (avg_throughput / peak_throughput * 100.0).min(100.0);
        let latency_score = (100.0 - (avg_response_time / 1000.0).min(100.0)).max(0.0);
        let performance_score = (success_rate_score + throughput_score + latency_score) / 3.0;
        
        // Generate basic percentiles from all response times
        let all_response_times: Vec<f64> = self.test_results.iter()
            .flat_map(|r| r.thread_performance.iter().map(|t| t.average_latency_ms))
            .collect();
        
        let response_time_percentiles = if !all_response_times.is_empty() {
            self.calculate_percentiles_from_data(&all_response_times)
        } else {
            PercentileSuite::default()
        };
        
        ConcurrentTestSummary {
            total_tests,
            total_threads,
            total_requests,
            overall_success_rate,
            average_throughput_qps: avg_throughput,
            peak_throughput_qps: peak_throughput,
            average_response_time_ms: avg_response_time,
            response_time_percentiles,
            test_duration_seconds: 0.0, // Would be calculated from actual test durations
            concurrent_users_range: (min_users, max_users),
            performance_score,
        }
    }
    
    fn analyze_threads(&self) -> Result<DetailedThreadAnalysis> {
        let mut all_thread_performances = Vec::new();
        
        for test_result in &self.test_results {
            for thread_perf in &test_result.thread_performance {
                let enhanced_perf = self.enhance_thread_performance(thread_perf, test_result);
                all_thread_performances.push(enhanced_perf);
            }
        }
        
        if all_thread_performances.is_empty() {
            return Err(anyhow::anyhow!("No thread performance data available"));
        }
        
        // Calculate variance
        let latencies: Vec<f64> = all_thread_performances.iter().map(|p| p.latency_statistics.mean_latency_ms).collect();
        let thread_performance_variance = self.calculate_variance(&latencies);
        
        // Find fastest/slowest threads
        let fastest_thread = all_thread_performances.iter()
            .min_by(|a, b| a.latency_statistics.mean_latency_ms.partial_cmp(&b.latency_statistics.mean_latency_ms).unwrap())
            .map(|p| ThreadPerformanceHighlight {
                thread_id: p.thread_id,
                performance_metric: p.latency_statistics.mean_latency_ms,
                reason: "Lowest average latency".to_string(),
                impact_on_overall: 0.0, // Would calculate actual impact
            })
            .unwrap();
        
        let slowest_thread = all_thread_performances.iter()
            .max_by(|a, b| a.latency_statistics.mean_latency_ms.partial_cmp(&b.latency_statistics.mean_latency_ms).unwrap())
            .map(|p| ThreadPerformanceHighlight {
                thread_id: p.thread_id,
                performance_metric: p.latency_statistics.mean_latency_ms,
                reason: "Highest average latency".to_string(),
                impact_on_overall: 0.0,
            })
            .unwrap();
        
        // Calculate efficiency distribution
        let thread_efficiency_distribution: Vec<f64> = all_thread_performances.iter()
            .map(|p| p.efficiency_score)
            .collect();
        
        // Calculate load balancing effectiveness
        let load_balancing_effectiveness = self.calculate_load_balancing_effectiveness(&all_thread_performances);
        
        // Calculate cross-thread correlation
        let cross_thread_correlation = self.calculate_thread_correlations(&all_thread_performances);
        
        Ok(DetailedThreadAnalysis {
            thread_performances: all_thread_performances,
            thread_performance_variance,
            fastest_thread,
            slowest_thread,
            thread_efficiency_distribution,
            cross_thread_correlation,
            load_balancing_effectiveness,
        })
    }
    
    fn enhance_thread_performance(&self, thread_perf: &ThreadPerformance, test_result: &ConcurrentTestResult) -> EnhancedThreadPerformance {
        // Enhanced version with additional calculated metrics
        let requests_per_second = if thread_perf.cpu_time_ms > 0.0 {
            thread_perf.requests_completed as f64 / (thread_perf.cpu_time_ms / 1000.0)
        } else {
            0.0
        };
        
        let performance_consistency = if thread_perf.average_latency_ms > 0.0 {
            1.0 / (1.0 + (thread_perf.max_latency_ms - thread_perf.average_latency_ms) / thread_perf.average_latency_ms)
        } else {
            1.0
        };
        
        let efficiency_score = if thread_perf.cpu_time_ms > 0.0 {
            thread_perf.requests_completed as f64 / thread_perf.cpu_time_ms * 1000.0
        } else {
            0.0
        };
        
        // Detect anomalies
        let mut anomaly_flags = Vec::new();
        if thread_perf.average_latency_ms > 1000.0 {
            anomaly_flags.push(ThreadAnomalyFlag::UnusuallyHighLatency);
        }
        if requests_per_second < 1.0 {
            anomaly_flags.push(ThreadAnomalyFlag::UnusuallyLowThroughput);
        }
        if thread_perf.errors_encountered > thread_perf.requests_completed / 10 {
            anomaly_flags.push(ThreadAnomalyFlag::HighErrorRate);
        }
        
        EnhancedThreadPerformance {
            thread_id: thread_perf.thread_id,
            requests_completed: thread_perf.requests_completed,
            requests_per_second,
            latency_statistics: ThreadLatencyStatistics {
                mean_latency_ms: thread_perf.average_latency_ms,
                median_latency_ms: thread_perf.average_latency_ms, // Simplified
                p95_latency_ms: thread_perf.max_latency_ms * 0.95, // Approximation
                p99_latency_ms: thread_perf.max_latency_ms * 0.99,
                min_latency_ms: thread_perf.average_latency_ms * 0.5, // Approximation
                max_latency_ms: thread_perf.max_latency_ms,
                standard_deviation_ms: thread_perf.max_latency_ms - thread_perf.average_latency_ms,
                latency_trend: LatencyTrend::Stable, // Would need time series data
                outlier_count: 0, // Would need detailed data
            },
            error_analysis: ThreadErrorAnalysis {
                total_errors: thread_perf.errors_encountered,
                error_rate: if thread_perf.requests_completed > 0 {
                    thread_perf.errors_encountered as f64 / thread_perf.requests_completed as f64 * 100.0
                } else {
                    0.0
                },
                error_types: HashMap::new(), // Would need detailed error data
                error_clustering: Vec::new(),
                error_pattern: ErrorPattern::Random,
                recovery_time_ms: 0.0,
            },
            resource_usage: ThreadResourceUsage {
                cpu_time_ms: thread_perf.cpu_time_ms,
                memory_peak_mb: 0.0, // Would need memory tracking
                memory_average_mb: 0.0,
                io_operations: 0,
                network_bytes: 0,
                context_switches: 0,
                cache_hit_rate: 0.95, // Default assumption
                resource_efficiency: efficiency_score,
            },
            performance_consistency,
            efficiency_score,
            contention_score: 0.0, // Would calculate based on wait times
            anomaly_flags,
        }
    }
    
    // Helper methods would continue here...
    fn calculate_variance(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }
    
    fn calculate_load_balancing_effectiveness(&self, performances: &[EnhancedThreadPerformance]) -> f64 {
        if performances.is_empty() {
            return 0.0;
        }
        
        let request_counts: Vec<f64> = performances.iter().map(|p| p.requests_completed as f64).collect();
        let variance = self.calculate_variance(&request_counts);
        let mean = request_counts.iter().sum::<f64>() / request_counts.len() as f64;
        
        // Lower coefficient of variation = better load balancing
        let cv = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
        (1.0 / (1.0 + cv) * 100.0).min(100.0)
    }
    
    // Additional helper methods would be implemented...
}

// Additional structs and implementations would continue...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub analysis_duration_ms: f64,
    pub analyzer_version: String,
    pub analysis_timestamp: std::time::SystemTime,
    pub total_data_points: usize,
}

// Default implementations
impl Default for PercentileSuite {
    fn default() -> Self {
        Self {
            p50: PercentileResult::default(),
            p90: PercentileResult::default(),
            p95: PercentileResult::default(),
            p99: PercentileResult::default(),
            p99_9: PercentileResult::default(),
            p99_99: PercentileResult::default(),
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            standard_deviation: 0.0,
            coefficient_of_variation: 0.0,
            interquartile_range: 0.0,
        }
    }
}

impl Default for PercentileResult {
    fn default() -> Self {
        Self {
            percentile: 0.0,
            value: 0.0,
            method_used: PercentileMethod::Exact,
            confidence_interval: ConfidenceInterval { lower_bound: 0.0, upper_bound: 0.0, confidence_level: 0.95 },
            accuracy_score: 1.0,
            sample_count: 0,
            calculation_time_ms: 0.0,
        }
    }
}
```

## Success Criteria
- Comprehensive concurrent test result analysis compiles without errors
- Thread-level performance metrics provide detailed insights
- Contention analysis accurately identifies resource conflicts
- Bottleneck detection pinpoints performance limiting factors
- Resource utilization analysis tracks per-thread consumption
- Scalability analysis generates meaningful performance curves
- Anomaly detection flags unusual performance patterns
- Analysis results provide actionable performance recommendations

## Time Limit
10 minutes maximum