# Task 40: Performance Report Generation

## Context
You are implementing Phase 4 of a vector indexing system. Advanced statistical calculations are now available. This task implements comprehensive performance report generation with charts, summaries, and exportable formats for analysis and monitoring.

## Current State
- `src/monitor.rs` exists with `PerformanceMonitor` struct
- Advanced statistical calculations are implemented
- Thread-safe monitoring wrapper is available
- Need comprehensive reporting functionality

## Task Objective
Implement comprehensive performance report generation including text reports, charts, CSV export, and HTML dashboards for performance analysis and monitoring.

## Implementation Requirements

### 1. Add report generation structures
Add these report-related structures to `src/monitor.rs`:
```rust
use std::fmt::Write;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub timestamp: std::time::SystemTime,
    pub basic_stats: PerformanceStats,
    pub advanced_stats: AdvancedPerformanceStats,
    pub summary: String,
    pub recommendations: Vec<String>,
    pub charts: Vec<ChartData>,
}

#[derive(Debug, Clone)]
pub struct ChartData {
    pub chart_type: ChartType,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub data_points: Vec<(f64, f64)>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum ChartType {
    TimeSeries,
    Histogram,
    ScatterPlot,
    BoxPlot,
    PercentileChart,
}

#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub include_charts: bool,
    pub include_predictions: bool,
    pub include_outliers: bool,
    pub confidence_level: f64,
    pub chart_resolution: usize,
    pub format: ReportFormat,
}

#[derive(Debug, Clone)]
pub enum ReportFormat {
    Text,
    Html,
    Csv,
    Json,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            include_charts: true,
            include_predictions: true,
            include_outliers: true,
            confidence_level: 0.95,
            chart_resolution: 100,
            format: ReportFormat::Text,
        }
    }
}
```

### 2. Add report generation methods
```rust
impl PerformanceMonitor {
    pub fn generate_report(&self, config: &ReportConfig) -> PerformanceReport {
        let basic_stats = self.get_stats();
        let advanced_stats = self.get_advanced_stats(config.confidence_level);
        
        let summary = self.generate_summary(&basic_stats, &advanced_stats, config);
        let recommendations = self.generate_recommendations(&basic_stats, &advanced_stats);
        let charts = if config.include_charts {
            self.generate_charts(config)
        } else {
            Vec::new()
        };
        
        PerformanceReport {
            timestamp: std::time::SystemTime::now(),
            basic_stats,
            advanced_stats,
            summary,
            recommendations,
            charts,
        }
    }
    
    fn generate_summary(&self, basic: &PerformanceStats, advanced: &AdvancedPerformanceStats, config: &ReportConfig) -> String {
        let mut summary = String::new();
        
        writeln!(&mut summary, "=== PERFORMANCE MONITORING REPORT ===").unwrap();
        writeln!(&mut summary, "Generated: {:?}", std::time::SystemTime::now()).unwrap();
        writeln!(&mut summary, "Uptime: {:.2} seconds", basic.uptime.as_secs_f64()).unwrap();
        writeln!(&mut summary, "").unwrap();
        
        // Query Performance Summary
        writeln!(&mut summary, "QUERY PERFORMANCE:").unwrap();
        writeln!(&mut summary, "  Total Queries: {}", basic.total_queries).unwrap();
        writeln!(&mut summary, "  Queries/Second: {:.2}", basic.queries_per_second).unwrap();
        writeln!(&mut summary, "  Average Time: {:.2}ms", basic.avg_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Median (P50): {:.2}ms", self.percentile(&self.query_times, 50).as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  P95: {:.2}ms", basic.p95_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  P99: {:.2}ms", basic.p99_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Min: {:.2}ms", basic.min_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Max: {:.2}ms", basic.max_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Std Dev: {:.2}ms", advanced.query_std_dev.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Coefficient of Variation: {:.2}%", advanced.query_cv * 100.0).unwrap();
        writeln!(&mut summary, "  IQR: {:.2}ms", advanced.query_iqr.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Trend Slope: {:.4}ms/operation", advanced.query_trend_slope).unwrap();
        writeln!(&mut summary, "  R-squared: {:.4}", advanced.query_correlation).unwrap();
        
        if config.include_predictions {
            writeln!(&mut summary, "  Next Prediction: {:.2}ms", advanced.query_prediction.as_secs_f64() * 1000.0).unwrap();
        }
        
        if config.include_outliers {
            writeln!(&mut summary, "  Outliers Detected: {}", advanced.query_outliers.len()).unwrap();
        }
        
        writeln!(&mut summary, "").unwrap();
        
        // Index Performance Summary
        writeln!(&mut summary, "INDEX PERFORMANCE:").unwrap();
        writeln!(&mut summary, "  Total Indexes: {}", basic.total_indexes).unwrap();
        writeln!(&mut summary, "  Indexes/Second: {:.2}", basic.indexes_per_second).unwrap();
        writeln!(&mut summary, "  Average Time: {:.2}ms", basic.avg_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Median (P50): {:.2}ms", self.percentile(&self.index_times, 50).as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  P95: {:.2}ms", basic.p95_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  P99: {:.2}ms", basic.p99_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Min: {:.2}ms", basic.min_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Max: {:.2}ms", basic.max_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Std Dev: {:.2}ms", advanced.index_std_dev.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Coefficient of Variation: {:.2}%", advanced.index_cv * 100.0).unwrap();
        writeln!(&mut summary, "  IQR: {:.2}ms", advanced.index_iqr.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut summary, "  Trend Slope: {:.4}ms/operation", advanced.index_trend_slope).unwrap();
        writeln!(&mut summary, "  R-squared: {:.4}", advanced.index_correlation).unwrap();
        
        if config.include_predictions {
            writeln!(&mut summary, "  Next Prediction: {:.2}ms", advanced.index_prediction.as_secs_f64() * 1000.0).unwrap();
        }
        
        if config.include_outliers {
            writeln!(&mut summary, "  Outliers Detected: {}", advanced.index_outliers.len()).unwrap();
        }
        
        summary
    }
    
    fn generate_recommendations(&self, basic: &PerformanceStats, advanced: &AdvancedPerformanceStats) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Query performance recommendations
        if advanced.query_cv > 0.5 {
            recommendations.push("HIGH VARIABILITY: Query times are highly variable (CV > 50%). Consider investigating inconsistent performance patterns.".to_string());
        }
        
        if advanced.query_trend_slope > 0.1 {
            recommendations.push("PERFORMANCE DEGRADATION: Query times are trending upward. Monitor for potential issues.".to_string());
        }
        
        if basic.p99_query_time.as_secs_f64() > basic.avg_query_time.as_secs_f64() * 3.0 {
            recommendations.push("TAIL LATENCY: P99 query time is significantly higher than average. Check for outliers and bottlenecks.".to_string());
        }
        
        if !advanced.query_outliers.is_empty() {
            recommendations.push(format!("OUTLIERS DETECTED: {} query outliers found. Investigate anomalous operations.", advanced.query_outliers.len()));
        }
        
        // Index performance recommendations
        if advanced.index_cv > 0.5 {
            recommendations.push("HIGH VARIABILITY: Index times are highly variable (CV > 50%). Consider investigating inconsistent indexing performance.".to_string());
        }
        
        if advanced.index_trend_slope > 0.1 {
            recommendations.push("INDEXING DEGRADATION: Index times are trending upward. Monitor indexing performance.".to_string());
        }
        
        // Throughput recommendations
        if basic.queries_per_second < 10.0 && basic.total_queries > 100 {
            recommendations.push("LOW THROUGHPUT: Query throughput is low. Consider optimizations or scaling.".to_string());
        }
        
        if basic.indexes_per_second < 5.0 && basic.total_indexes > 50 {
            recommendations.push("LOW INDEXING RATE: Indexing rate is low. Consider batch processing or optimizations.".to_string());
        }
        
        // General recommendations
        if advanced.query_correlation > 0.8 && advanced.query_trend_slope > 0.0 {
            recommendations.push("CONSISTENT DEGRADATION: Strong correlation with positive trend indicates systematic performance degradation.".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("GOOD PERFORMANCE: No significant performance issues detected.".to_string());
        }
        
        recommendations
    }
    
    fn generate_charts(&self, config: &ReportConfig) -> Vec<ChartData> {
        let mut charts = Vec::new();
        
        // Query time series chart
        if !self.query_times.is_empty() {
            let query_data: Vec<(f64, f64)> = self.query_times.iter()
                .enumerate()
                .map(|(i, d)| (i as f64, d.as_secs_f64() * 1000.0))
                .collect();
            
            charts.push(ChartData {
                chart_type: ChartType::TimeSeries,
                title: "Query Time Series".to_string(),
                x_label: "Operation Index".to_string(),
                y_label: "Time (ms)".to_string(),
                data_points: query_data,
                metadata: HashMap::new(),
            });
        }
        
        // Index time series chart
        if !self.index_times.is_empty() {
            let index_data: Vec<(f64, f64)> = self.index_times.iter()
                .enumerate()
                .map(|(i, d)| (i as f64, d.as_secs_f64() * 1000.0))
                .collect();
            
            charts.push(ChartData {
                chart_type: ChartType::TimeSeries,
                title: "Index Time Series".to_string(),
                x_label: "Operation Index".to_string(),
                y_label: "Time (ms)".to_string(),
                data_points: index_data,
                metadata: HashMap::new(),
            });
        }
        
        // Query histogram
        if !self.query_times.is_empty() {
            let histogram_data = self.create_histogram(&self.query_times, config.chart_resolution);
            charts.push(ChartData {
                chart_type: ChartType::Histogram,
                title: "Query Time Distribution".to_string(),
                x_label: "Time (ms)".to_string(),
                y_label: "Frequency".to_string(),
                data_points: histogram_data,
                metadata: HashMap::new(),
            });
        }
        
        // Percentile chart
        if !self.query_times.is_empty() {
            let percentiles = vec![50.0, 75.0, 90.0, 95.0, 99.0, 99.9];
            let percentile_values = self.calculate_percentiles(&self.query_times, &percentiles);
            let percentile_data: Vec<(f64, f64)> = percentiles.iter()
                .zip(percentile_values.iter())
                .map(|(p, v)| (*p, v.as_secs_f64() * 1000.0))
                .collect();
            
            charts.push(ChartData {
                chart_type: ChartType::PercentileChart,
                title: "Query Time Percentiles".to_string(),
                x_label: "Percentile".to_string(),
                y_label: "Time (ms)".to_string(),
                data_points: percentile_data,
                metadata: HashMap::new(),
            });
        }
        
        charts
    }
    
    fn create_histogram(&self, durations: &VecDeque<Duration>, bins: usize) -> Vec<(f64, f64)> {
        if durations.is_empty() {
            return Vec::new();
        }
        
        let min_ms = durations.iter().min().unwrap().as_secs_f64() * 1000.0;
        let max_ms = durations.iter().max().unwrap().as_secs_f64() * 1000.0;
        let bin_width = (max_ms - min_ms) / bins as f64;
        
        if bin_width == 0.0 {
            return vec![(min_ms, durations.len() as f64)];
        }
        
        let mut histogram = vec![0; bins];
        
        for duration in durations {
            let value = duration.as_secs_f64() * 1000.0;
            let bin_index = ((value - min_ms) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }
        
        histogram.iter()
            .enumerate()
            .map(|(i, &count)| (min_ms + i as f64 * bin_width + bin_width / 2.0, count as f64))
            .collect()
    }
}
```

### 3. Add export functionality
```rust
impl PerformanceReport {
    pub fn export_text(&self) -> String {
        format!("{}\n\nRECOMMENDATIONS:\n{}", 
            self.summary,
            self.recommendations.join("\n")
        )
    }
    
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();
        
        // Header
        writeln!(&mut csv, "timestamp,metric,value,unit").unwrap();
        
        // Basic stats
        let timestamp = format!("{:?}", self.timestamp);
        writeln!(&mut csv, "{},total_queries,{},count", timestamp, self.basic_stats.total_queries).unwrap();
        writeln!(&mut csv, "{},total_indexes,{},count", timestamp, self.basic_stats.total_indexes).unwrap();
        writeln!(&mut csv, "{},queries_per_second,{:.2},rate", timestamp, self.basic_stats.queries_per_second).unwrap();
        writeln!(&mut csv, "{},indexes_per_second,{:.2},rate", timestamp, self.basic_stats.indexes_per_second).unwrap();
        writeln!(&mut csv, "{},avg_query_time,{:.2},ms", timestamp, self.basic_stats.avg_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut csv, "{},p95_query_time,{:.2},ms", timestamp, self.basic_stats.p95_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut csv, "{},p99_query_time,{:.2},ms", timestamp, self.basic_stats.p99_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut csv, "{},avg_index_time,{:.2},ms", timestamp, self.basic_stats.avg_index_time.as_secs_f64() * 1000.0).unwrap();
        
        // Advanced stats
        writeln!(&mut csv, "{},query_std_dev,{:.2},ms", timestamp, self.advanced_stats.query_std_dev.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut csv, "{},query_cv,{:.4},ratio", timestamp, self.advanced_stats.query_cv).unwrap();
        writeln!(&mut csv, "{},query_trend_slope,{:.4},ms_per_op", timestamp, self.advanced_stats.query_trend_slope).unwrap();
        writeln!(&mut csv, "{},query_correlation,{:.4},r_squared", timestamp, self.advanced_stats.query_correlation).unwrap();
        
        csv
    }
    
    pub fn export_html(&self) -> String {
        let mut html = String::new();
        
        writeln!(&mut html, "<!DOCTYPE html>").unwrap();
        writeln!(&mut html, "<html><head>").unwrap();
        writeln!(&mut html, "<title>Performance Report</title>").unwrap();
        writeln!(&mut html, "<style>").unwrap();
        writeln!(&mut html, "body {{ font-family: Arial, sans-serif; margin: 20px; }}").unwrap();
        writeln!(&mut html, "table {{ border-collapse: collapse; width: 100%; }}").unwrap();
        writeln!(&mut html, "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}").unwrap();
        writeln!(&mut html, "th {{ background-color: #f2f2f2; }}").unwrap();
        writeln!(&mut html, ".metric {{ font-weight: bold; }}").unwrap();
        writeln!(&mut html, ".recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; }}").unwrap();
        writeln!(&mut html, "</style>").unwrap();
        writeln!(&mut html, "</head><body>").unwrap();
        
        writeln!(&mut html, "<h1>Performance Monitoring Report</h1>").unwrap();
        writeln!(&mut html, "<p>Generated: {:?}</p>", self.timestamp).unwrap();
        
        // Basic metrics table
        writeln!(&mut html, "<h2>Performance Metrics</h2>").unwrap();
        writeln!(&mut html, "<table>").unwrap();
        writeln!(&mut html, "<tr><th>Metric</th><th>Query</th><th>Index</th></tr>").unwrap();
        writeln!(&mut html, "<tr><td class='metric'>Total Operations</td><td>{}</td><td>{}</td></tr>", 
            self.basic_stats.total_queries, self.basic_stats.total_indexes).unwrap();
        writeln!(&mut html, "<tr><td class='metric'>Operations/Second</td><td>{:.2}</td><td>{:.2}</td></tr>", 
            self.basic_stats.queries_per_second, self.basic_stats.indexes_per_second).unwrap();
        writeln!(&mut html, "<tr><td class='metric'>Average Time (ms)</td><td>{:.2}</td><td>{:.2}</td></tr>", 
            self.basic_stats.avg_query_time.as_secs_f64() * 1000.0, 
            self.basic_stats.avg_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut html, "<tr><td class='metric'>P95 Time (ms)</td><td>{:.2}</td><td>{:.2}</td></tr>", 
            self.basic_stats.p95_query_time.as_secs_f64() * 1000.0, 
            self.basic_stats.p95_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut html, "<tr><td class='metric'>P99 Time (ms)</td><td>{:.2}</td><td>{:.2}</td></tr>", 
            self.basic_stats.p99_query_time.as_secs_f64() * 1000.0, 
            self.basic_stats.p99_index_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut html, "</table>").unwrap();
        
        // Recommendations
        writeln!(&mut html, "<h2>Recommendations</h2>").unwrap();
        for recommendation in &self.recommendations {
            writeln!(&mut html, "<div class='recommendation'>{}</div>", recommendation).unwrap();
        }
        
        writeln!(&mut html, "</body></html>").unwrap();
        
        html
    }
    
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        // Note: This requires serde_json dependency
        serde_json::to_string_pretty(self)
    }
}
```

### 4. Add comprehensive reporting tests
```rust
#[cfg(test)]
mod reporting_tests {
    use super::*;
    
    #[test]
    fn test_report_generation() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add sample data
        for i in 1..=20 {
            monitor.record_query_time(Duration::from_millis(i * 5));
            monitor.record_index_time(Duration::from_millis(i * 10));
        }
        
        let config = ReportConfig::default();
        let report = monitor.generate_report(&config);
        
        assert!(!report.summary.is_empty());
        assert!(!report.recommendations.is_empty());
        assert!(!report.charts.is_empty());
        
        // Verify report contains key information
        assert!(report.summary.contains("QUERY PERFORMANCE"));
        assert!(report.summary.contains("INDEX PERFORMANCE"));
        assert!(report.summary.contains("Total Queries: 20"));
        assert!(report.summary.contains("Total Indexes: 20"));
    }
    
    #[test]
    fn test_chart_generation() {
        let mut monitor = PerformanceMonitor::new();
        
        for i in 1..=10 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let config = ReportConfig::default();
        let charts = monitor.generate_charts(&config);
        
        assert!(!charts.is_empty());
        
        // Find time series chart
        let time_series = charts.iter()
            .find(|c| matches!(c.chart_type, ChartType::TimeSeries))
            .expect("Should have time series chart");
        
        assert_eq!(time_series.data_points.len(), 10);
        assert_eq!(time_series.title, "Query Time Series");
    }
    
    #[test]
    fn test_histogram_generation() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add data with known distribution
        for _ in 0..50 {
            monitor.record_query_time(Duration::from_millis(10));
        }
        for _ in 0..30 {
            monitor.record_query_time(Duration::from_millis(20));
        }
        for _ in 0..20 {
            monitor.record_query_time(Duration::from_millis(30));
        }
        
        let histogram = monitor.create_histogram(&monitor.query_times, 10);
        
        assert!(!histogram.is_empty());
        
        // Should have bins covering the range 10-30ms
        let total_count: f64 = histogram.iter().map(|(_, count)| count).sum();
        assert_eq!(total_count, 100.0); // 50 + 30 + 20
    }
    
    #[test]
    fn test_export_formats() {
        let mut monitor = PerformanceMonitor::new();
        
        for i in 1..=5 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let config = ReportConfig::default();
        let report = monitor.generate_report(&config);
        
        // Test text export
        let text_report = report.export_text();
        assert!(text_report.contains("PERFORMANCE MONITORING REPORT"));
        assert!(text_report.contains("RECOMMENDATIONS"));
        
        // Test CSV export
        let csv_report = report.export_csv();
        assert!(csv_report.contains("timestamp,metric,value,unit"));
        assert!(csv_report.contains("total_queries,5,count"));
        
        // Test HTML export
        let html_report = report.export_html();
        assert!(html_report.contains("<!DOCTYPE html>"));
        assert!(html_report.contains("Performance Monitoring Report"));
        assert!(html_report.contains("<table>"));
    }
    
    #[test]
    fn test_recommendations() {
        let mut monitor = PerformanceMonitor::new();
        
        // Create high variability data
        monitor.record_query_time(Duration::from_millis(10));
        monitor.record_query_time(Duration::from_millis(100));
        monitor.record_query_time(Duration::from_millis(15));
        monitor.record_query_time(Duration::from_millis(200));
        
        let basic_stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        let recommendations = monitor.generate_recommendations(&basic_stats, &advanced_stats);
        
        assert!(!recommendations.is_empty());
        
        // Should detect high variability
        let has_variability_warning = recommendations.iter()
            .any(|r| r.contains("HIGH VARIABILITY"));
        assert!(has_variability_warning);
    }
    
    #[test]
    fn test_report_config_options() {
        let mut monitor = PerformanceMonitor::new();
        
        for i in 1..=10 {
            monitor.record_query_time(Duration::from_millis(i * 5));
        }
        
        // Test with charts disabled
        let config_no_charts = ReportConfig {
            include_charts: false,
            ..Default::default()
        };
        let report_no_charts = monitor.generate_report(&config_no_charts);
        assert!(report_no_charts.charts.is_empty());
        
        // Test with predictions disabled
        let config_no_predictions = ReportConfig {
            include_predictions: false,
            ..Default::default()
        };
        let report_no_predictions = monitor.generate_report(&config_no_predictions);
        assert!(!report_no_predictions.summary.contains("Next Prediction"));
        
        // Test different confidence level
        let config_99 = ReportConfig {
            confidence_level: 0.99,
            ..Default::default()
        };
        let report_99 = monitor.generate_report(&config_99);
        assert!(!report_99.summary.is_empty());
    }
}
```

## Success Criteria
- [ ] Report generation creates comprehensive performance reports
- [ ] Chart generation works for time series, histograms, and percentiles
- [ ] Export functionality supports text, CSV, and HTML formats
- [ ] Recommendations engine provides actionable insights
- [ ] Report configuration options work correctly
- [ ] All export formats are properly formatted
- [ ] Charts contain accurate data and proper labels
- [ ] All tests pass with correct functionality
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Comprehensive reporting with multiple visualization types
- Actionable recommendations based on statistical analysis
- Multiple export formats for different use cases
- Configurable report generation for different scenarios
- HTML export includes styled tables and recommendations
- CSV export enables integration with external analysis tools
- Histogram generation for distribution analysis