# Task 017: Enhanced PerformanceMetrics Struct with Advanced Statistics

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 008 (PerformanceBenchmark). The enhanced PerformanceMetrics provides detailed statistical analysis, histogram support, and trend detection capabilities for comprehensive performance monitoring.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Enhance the existing `PerformanceMetrics` struct with advanced statistical capabilities including histogram-based latency distribution, resource usage tracking, trend analysis, and export capabilities for monitoring systems.

## Requirements
1. Extend existing `PerformanceMetrics` struct in `src/validation/performance.rs`
2. Add histogram support for latency distribution analysis
3. Implement resource usage tracking (CPU, memory, I/O)
4. Add trend analysis and performance degradation detection
5. Create export capabilities for monitoring systems (Prometheus format)
6. Support statistical accuracy validation
7. Add performance baseline comparison

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPerformanceMetrics {
    // Core metrics from original PerformanceMetrics
    pub latencies: Vec<Duration>,
    pub throughput_qps: f64,
    pub index_rate_fps: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub total_queries: usize,
    pub failed_queries: usize,
    
    // Enhanced statistical analysis
    pub latency_histogram: LatencyHistogram,
    pub percentiles: PercentileMetrics,
    pub resource_usage: ResourceUsageMetrics,
    pub trend_analysis: TrendAnalysis,
    pub baseline_comparison: Option<BaselineComparison>,
    pub export_data: ExportData,
    
    // Timing metadata
    pub measurement_start: SystemTime,
    pub measurement_duration: Duration,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    // Histogram buckets in milliseconds: [0-1ms, 1-5ms, 5-10ms, 10-50ms, 50-100ms, 100-500ms, 500ms+]
    pub buckets: Vec<HistogramBucket>,
    pub total_samples: usize,
    pub overflow_count: usize, // Samples > 500ms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub min_ms: f64,
    pub max_ms: f64,
    pub count: usize,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileMetrics {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub p99_9_ms: f64,
    pub p99_99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub standard_deviation_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMetrics {
    pub cpu_usage_samples: Vec<f64>,
    pub memory_usage_samples: Vec<f64>,
    pub disk_io_read_mb: f64,
    pub disk_io_write_mb: f64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub peak_memory_mb: f64,
    pub average_cpu_percent: f64,
    pub resource_efficiency_score: f64, // 0-100 score
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub performance_trend: PerformanceTrend,
    pub degradation_detected: bool,
    pub degradation_severity: DegradationSeverity,
    pub trend_confidence: f64, // 0.0-1.0
    pub moving_average_qps: Vec<f64>,
    pub performance_variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationSeverity {
    None,
    Minor,      // < 5% degradation
    Moderate,   // 5-15% degradation
    Severe,     // > 15% degradation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_qps: f64,
    pub current_qps: f64,
    pub qps_change_percent: f64,
    pub baseline_p95_ms: f64,
    pub current_p95_ms: f64,
    pub latency_change_percent: f64,
    pub overall_performance_score: f64, // 0-100
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub prometheus_metrics: String,
    pub json_export: String,
    pub csv_data: String,
    pub export_timestamp: SystemTime,
}

impl EnhancedPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            throughput_qps: 0.0,
            index_rate_fps: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            total_queries: 0,
            failed_queries: 0,
            latency_histogram: LatencyHistogram::new(),
            percentiles: PercentileMetrics::default(),
            resource_usage: ResourceUsageMetrics::default(),
            trend_analysis: TrendAnalysis::default(),
            baseline_comparison: None,
            export_data: ExportData::default(),
            measurement_start: SystemTime::now(),
            measurement_duration: Duration::from_secs(0),
            sample_count: 0,
        }
    }
    
    pub fn add_latency_sample(&mut self, latency: Duration) {
        self.latencies.push(latency);
        self.latency_histogram.add_sample(latency);
        self.sample_count += 1;
    }
    
    pub fn calculate_advanced_statistics(&mut self) {
        self.calculate_percentiles();
        self.calculate_histogram_percentages();
        self.analyze_performance_trend();
        self.calculate_resource_efficiency();
        self.generate_export_data();
    }
    
    fn calculate_percentiles(&mut self) {
        if self.latencies.is_empty() {
            return;
        }
        
        let mut sorted_latencies: Vec<f64> = self.latencies
            .iter()
            .map(|d| d.as_millis() as f64)
            .collect();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted_latencies.len();
        
        self.percentiles = PercentileMetrics {
            min_ms: sorted_latencies[0],
            max_ms: sorted_latencies[len - 1],
            p50_ms: Self::percentile(&sorted_latencies, 0.50),
            p90_ms: Self::percentile(&sorted_latencies, 0.90),
            p95_ms: Self::percentile(&sorted_latencies, 0.95),
            p99_ms: Self::percentile(&sorted_latencies, 0.99),
            p99_9_ms: Self::percentile(&sorted_latencies, 0.999),
            p99_99_ms: Self::percentile(&sorted_latencies, 0.9999),
            mean_ms: sorted_latencies.iter().sum::<f64>() / len as f64,
            median_ms: Self::percentile(&sorted_latencies, 0.50),
            standard_deviation_ms: Self::calculate_std_dev(&sorted_latencies),
        };
    }
    
    fn percentile(sorted_data: &[f64], percentile: f64) -> f64 {
        let index = (percentile * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }
    
    fn calculate_std_dev(data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }
    
    fn calculate_histogram_percentages(&mut self) {
        for bucket in &mut self.latency_histogram.buckets {
            bucket.percentage = if self.latency_histogram.total_samples > 0 {
                (bucket.count as f64 / self.latency_histogram.total_samples as f64) * 100.0
            } else {
                0.0
            };
        }
    }
    
    fn analyze_performance_trend(&mut self) {
        // Simplified trend analysis - in production, this would use more sophisticated algorithms
        if self.resource_usage.cpu_usage_samples.len() < 3 {
            self.trend_analysis.performance_trend = PerformanceTrend::Stable;
            return;
        }
        
        let recent_samples = &self.resource_usage.cpu_usage_samples[self.resource_usage.cpu_usage_samples.len().saturating_sub(10)..];
        let variance = Self::calculate_variance(recent_samples);
        
        self.trend_analysis.performance_variance = variance;
        
        if variance > 20.0 {
            self.trend_analysis.performance_trend = PerformanceTrend::Volatile;
        } else if self.throughput_qps > 0.0 && self.baseline_comparison.is_some() {
            let baseline = self.baseline_comparison.as_ref().unwrap();
            if baseline.qps_change_percent < -15.0 {
                self.trend_analysis.performance_trend = PerformanceTrend::Degrading;
                self.trend_analysis.degradation_detected = true;
                self.trend_analysis.degradation_severity = DegradationSeverity::Severe;
            } else if baseline.qps_change_percent < -5.0 {
                self.trend_analysis.performance_trend = PerformanceTrend::Degrading;
                self.trend_analysis.degradation_detected = true;
                self.trend_analysis.degradation_severity = DegradationSeverity::Minor;
            } else if baseline.qps_change_percent > 5.0 {
                self.trend_analysis.performance_trend = PerformanceTrend::Improving;
            } else {
                self.trend_analysis.performance_trend = PerformanceTrend::Stable;
            }
        }
    }
    
    fn calculate_variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }
    
    fn calculate_resource_efficiency(&mut self) {
        // Efficiency score based on CPU usage vs throughput
        let efficiency = if self.resource_usage.average_cpu_percent > 0.0 {
            (self.throughput_qps / self.resource_usage.average_cpu_percent) * 10.0
        } else {
            0.0
        };
        
        self.resource_usage.resource_efficiency_score = efficiency.min(100.0);
    }
    
    fn generate_export_data(&mut self) {
        self.export_data.prometheus_metrics = self.to_prometheus_format();
        self.export_data.json_export = serde_json::to_string_pretty(self).unwrap_or_default();
        self.export_data.csv_data = self.to_csv_format();
        self.export_data.export_timestamp = SystemTime::now();
    }
    
    fn to_prometheus_format(&self) -> String {
        format!(
            "# HELP search_latency_seconds Search request latency\n\
             # TYPE search_latency_seconds histogram\n\
             search_throughput_qps {}\n\
             search_latency_p50_ms {}\n\
             search_latency_p95_ms {}\n\
             search_latency_p99_ms {}\n\
             search_memory_usage_mb {}\n\
             search_cpu_usage_percent {}\n\
             search_total_queries {}\n\
             search_failed_queries {}\n",
            self.throughput_qps,
            self.percentiles.p50_ms,
            self.percentiles.p95_ms,
            self.percentiles.p99_ms,
            self.memory_usage_mb,
            self.cpu_usage_percent,
            self.total_queries,
            self.failed_queries
        )
    }
    
    fn to_csv_format(&self) -> String {
        format!(
            "metric,value\n\
             throughput_qps,{}\n\
             p50_latency_ms,{}\n\
             p95_latency_ms,{}\n\
             p99_latency_ms,{}\n\
             memory_usage_mb,{}\n\
             cpu_usage_percent,{}\n\
             total_queries,{}\n\
             failed_queries,{}\n",
            self.throughput_qps,
            self.percentiles.p50_ms,
            self.percentiles.p95_ms,
            self.percentiles.p99_ms,
            self.memory_usage_mb,
            self.cpu_usage_percent,
            self.total_queries,
            self.failed_queries
        )
    }
    
    pub fn set_baseline(&mut self, baseline_qps: f64, baseline_p95_ms: f64) {
        let qps_change = if baseline_qps > 0.0 {
            ((self.throughput_qps - baseline_qps) / baseline_qps) * 100.0
        } else {
            0.0
        };
        
        let latency_change = if baseline_p95_ms > 0.0 {
            ((self.percentiles.p95_ms - baseline_p95_ms) / baseline_p95_ms) * 100.0
        } else {
            0.0
        };
        
        // Overall score: higher QPS is better, lower latency is better
        let qps_score = (100.0 + qps_change).max(0.0).min(200.0);
        let latency_score = (100.0 - latency_change).max(0.0).min(200.0);
        let overall_score = (qps_score + latency_score) / 2.0;
        
        self.baseline_comparison = Some(BaselineComparison {
            baseline_qps,
            current_qps: self.throughput_qps,
            qps_change_percent: qps_change,
            baseline_p95_ms,
            current_p95_ms: self.percentiles.p95_ms,
            latency_change_percent: latency_change,
            overall_performance_score: overall_score,
        });
    }
    
    pub fn print_detailed_report(&self) {
        println!("\n=== Enhanced Performance Metrics Report ===");
        println!("Measurement Duration: {:.2}s", self.measurement_duration.as_secs_f64());
        println!("Total Samples: {}", self.sample_count);
        println!("Success Rate: {:.2}%", ((self.total_queries - self.failed_queries) as f64 / self.total_queries as f64) * 100.0);
        
        println!("\n--- Latency Percentiles ---");
        println!("P50: {:.2}ms", self.percentiles.p50_ms);
        println!("P90: {:.2}ms", self.percentiles.p90_ms);
        println!("P95: {:.2}ms", self.percentiles.p95_ms);
        println!("P99: {:.2}ms", self.percentiles.p99_ms);
        println!("P99.9: {:.2}ms", self.percentiles.p99_9_ms);
        println!("P99.99: {:.2}ms", self.percentiles.p99_99_ms);
        
        println!("\n--- Resource Usage ---");
        println!("Peak Memory: {:.2}MB", self.resource_usage.peak_memory_mb);
        println!("Average CPU: {:.2}%", self.resource_usage.average_cpu_percent);
        println!("Resource Efficiency Score: {:.1}/100", self.resource_usage.resource_efficiency_score);
        
        if let Some(baseline) = &self.baseline_comparison {
            println!("\n--- Baseline Comparison ---");
            println!("QPS Change: {:.2}%", baseline.qps_change_percent);
            println!("Latency Change: {:.2}%", baseline.latency_change_percent);
            println!("Overall Performance Score: {:.1}/100", baseline.overall_performance_score);
        }
        
        println!("\n--- Trend Analysis ---");
        println!("Trend: {:?}", self.trend_analysis.performance_trend);
        if self.trend_analysis.degradation_detected {
            println!("⚠️  Performance degradation detected: {:?}", self.trend_analysis.degradation_severity);
        }
    }
}

impl Default for PercentileMetrics {
    fn default() -> Self {
        Self {
            p50_ms: 0.0, p90_ms: 0.0, p95_ms: 0.0, p99_ms: 0.0,
            p99_9_ms: 0.0, p99_99_ms: 0.0, min_ms: 0.0, max_ms: 0.0,
            mean_ms: 0.0, median_ms: 0.0, standard_deviation_ms: 0.0,
        }
    }
}

impl Default for ResourceUsageMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_samples: Vec::new(),
            memory_usage_samples: Vec::new(),
            disk_io_read_mb: 0.0,
            disk_io_write_mb: 0.0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            peak_memory_mb: 0.0,
            average_cpu_percent: 0.0,
            resource_efficiency_score: 0.0,
        }
    }
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            performance_trend: PerformanceTrend::Stable,
            degradation_detected: false,
            degradation_severity: DegradationSeverity::None,
            trend_confidence: 0.0,
            moving_average_qps: Vec::new(),
            performance_variance: 0.0,
        }
    }
}

impl Default for ExportData {
    fn default() -> Self {
        Self {
            prometheus_metrics: String::new(),
            json_export: String::new(),
            csv_data: String::new(),
            export_timestamp: UNIX_EPOCH,
        }
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: vec![
                HistogramBucket { min_ms: 0.0, max_ms: 1.0, count: 0, percentage: 0.0 },
                HistogramBucket { min_ms: 1.0, max_ms: 5.0, count: 0, percentage: 0.0 },
                HistogramBucket { min_ms: 5.0, max_ms: 10.0, count: 0, percentage: 0.0 },
                HistogramBucket { min_ms: 10.0, max_ms: 50.0, count: 0, percentage: 0.0 },
                HistogramBucket { min_ms: 50.0, max_ms: 100.0, count: 0, percentage: 0.0 },
                HistogramBucket { min_ms: 100.0, max_ms: 500.0, count: 0, percentage: 0.0 },
            ],
            total_samples: 0,
            overflow_count: 0,
        }
    }
    
    pub fn add_sample(&mut self, latency: Duration) {
        let latency_ms = latency.as_millis() as f64;
        self.total_samples += 1;
        
        let mut placed = false;
        for bucket in &mut self.buckets {
            if latency_ms >= bucket.min_ms && latency_ms < bucket.max_ms {
                bucket.count += 1;
                placed = true;
                break;
            }
        }
        
        if !placed && latency_ms >= 500.0 {
            self.overflow_count += 1;
        }
    }
}
```

## Success Criteria
- Enhanced PerformanceMetrics struct compiles without errors
- Histogram-based latency distribution works correctly
- Percentile calculations are accurate for all specified percentiles
- Resource usage tracking captures CPU, memory, and I/O metrics
- Trend analysis detects performance degradation
- Export capabilities generate valid Prometheus, JSON, and CSV formats
- Baseline comparison provides meaningful performance scoring
- Statistical calculations are mathematically correct

## Time Limit
10 minutes maximum