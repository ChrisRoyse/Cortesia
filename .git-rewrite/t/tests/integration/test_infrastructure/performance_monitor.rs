// Performance Monitor
// Tracks and analyzes performance metrics during integration tests

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

/// Performance monitor for tracking test metrics
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    measurements: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    memory_samples: Arc<Mutex<Vec<MemorySample>>>,
    start_time: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    pub timestamp: Duration,
    pub bytes_used: u64,
    pub allocations: u64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            measurements: Arc::new(Mutex::new(HashMap::new())),
            memory_samples: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    /// Record a duration measurement
    pub fn record_duration(&self, metric_name: &str, duration: Duration) {
        let mut measurements = self.measurements.lock().unwrap();
        measurements
            .entry(metric_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Start a measurement
    pub fn start_measurement(&self, metric_name: &str) -> MeasurementHandle {
        MeasurementHandle {
            metric_name: metric_name.to_string(),
            start_time: Instant::now(),
            monitor: self.clone(),
        }
    }

    /// Record memory usage
    pub fn record_memory(&self, bytes_used: u64, allocations: u64) {
        let mut samples = self.memory_samples.lock().unwrap();
        samples.push(MemorySample {
            timestamp: self.start_time.elapsed(),
            bytes_used,
            allocations,
        });
    }

    /// Get performance summary
    pub fn get_summary(&self) -> PerformanceSummary {
        let measurements = self.measurements.lock().unwrap();
        let mut summary_measurements = HashMap::new();

        for (metric_name, durations) in measurements.iter() {
            if !durations.is_empty() {
                let stats = calculate_duration_stats(durations);
                summary_measurements.insert(metric_name.clone(), stats);
            }
        }

        let memory_samples = self.memory_samples.lock().unwrap();
        let memory_stats = if !memory_samples.is_empty() {
            Some(calculate_memory_stats(&memory_samples))
        } else {
            None
        };

        PerformanceSummary {
            measurements: summary_measurements,
            memory_usage: memory_stats,
        }
    }

    /// Get percentile for a metric
    pub fn get_percentile(&self, metric_name: &str, percentile: f64) -> Option<Duration> {
        let measurements = self.measurements.lock().unwrap();
        
        if let Some(durations) = measurements.get(metric_name) {
            if !durations.is_empty() {
                let mut sorted = durations.clone();
                sorted.sort();
                
                let index = ((percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
                return Some(sorted[index]);
            }
        }
        
        None
    }

    /// Compare two metrics
    pub fn compare_metrics(&self, metric1: &str, metric2: &str) -> Option<MetricComparison> {
        let measurements = self.measurements.lock().unwrap();
        
        let durations1 = measurements.get(metric1)?;
        let durations2 = measurements.get(metric2)?;
        
        if durations1.is_empty() || durations2.is_empty() {
            return None;
        }
        
        let stats1 = calculate_duration_stats(durations1);
        let stats2 = calculate_duration_stats(durations2);
        
        Some(MetricComparison {
            metric1: metric1.to_string(),
            metric2: metric2.to_string(),
            speedup: stats2.avg.as_secs_f64() / stats1.avg.as_secs_f64(),
            avg_difference: if stats1.avg > stats2.avg {
                stats1.avg - stats2.avg
            } else {
                stats2.avg - stats1.avg
            },
        })
    }
}

/// Handle for an ongoing measurement
pub struct MeasurementHandle {
    metric_name: String,
    start_time: Instant,
    monitor: PerformanceMonitor,
}

impl MeasurementHandle {
    /// Complete the measurement
    pub fn complete(self) -> Duration {
        let duration = self.start_time.elapsed();
        self.monitor.record_duration(&self.metric_name, duration);
        duration
    }
}

impl Drop for MeasurementHandle {
    fn drop(&mut self) {
        // Auto-complete if not explicitly completed
        let duration = self.start_time.elapsed();
        self.monitor.record_duration(&self.metric_name, duration);
    }
}

/// Metric comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric1: String,
    pub metric2: String,
    pub speedup: f64,
    pub avg_difference: Duration,
}

/// Calculate duration statistics
fn calculate_duration_stats(durations: &[Duration]) -> DurationStats {
    let mut sorted = durations.to_vec();
    sorted.sort();
    
    let count = sorted.len();
    let total: Duration = sorted.iter().sum();
    let avg = total / count as u32;
    
    let min = sorted[0];
    let max = sorted[count - 1];
    
    let p50_idx = count / 2;
    let p95_idx = (count as f64 * 0.95) as usize;
    let p99_idx = (count as f64 * 0.99) as usize;
    
    DurationStats {
        count,
        total,
        min,
        max,
        avg,
        p50: sorted[p50_idx],
        p95: sorted[p95_idx.min(count - 1)],
        p99: sorted[p99_idx.min(count - 1)],
    }
}

/// Calculate memory statistics
fn calculate_memory_stats(samples: &[MemorySample]) -> MemoryStats {
    if samples.is_empty() {
        return MemoryStats {
            initial: 0,
            peak: 0,
            final_value: 0,
            allocations: 0,
        };
    }
    
    let initial = samples[0].bytes_used;
    let final_value = samples[samples.len() - 1].bytes_used;
    let peak = samples.iter().map(|s| s.bytes_used).max().unwrap_or(0);
    let allocations = samples.iter().map(|s| s.allocations).sum();
    
    MemoryStats {
        initial,
        peak,
        final_value,
        allocations,
    }
}

// Re-export types from parent module
pub use super::{DurationStats, MemoryStats, PerformanceSummary};

/// Advanced performance analysis
pub struct PerformanceAnalyzer {
    monitors: Vec<PerformanceMonitor>,
}

impl PerformanceAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Add a monitor for analysis
    pub fn add_monitor(&mut self, monitor: PerformanceMonitor) {
        self.monitors.push(monitor);
    }

    /// Analyze scaling behavior
    pub fn analyze_scaling(&self, metric_name: &str) -> Option<ScalingAnalysis> {
        if self.monitors.len() < 2 {
            return None;
        }
        
        let mut data_points = Vec::new();
        
        for (i, monitor) in self.monitors.iter().enumerate() {
            if let Some(p50) = monitor.get_percentile(metric_name, 50.0) {
                data_points.push((i as f64, p50.as_secs_f64()));
            }
        }
        
        if data_points.len() < 2 {
            return None;
        }
        
        // Simple linear regression
        let n = data_points.len() as f64;
        let sum_x: f64 = data_points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data_points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data_points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = data_points.iter().map(|(x, _)| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot: f64 = data_points.iter()
            .map(|(_, y)| (y - y_mean).powi(2))
            .sum();
        let ss_res: f64 = data_points.iter()
            .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
            .sum();
        let r_squared = 1.0 - (ss_res / ss_tot);
        
        Some(ScalingAnalysis {
            metric_name: metric_name.to_string(),
            data_points,
            slope,
            intercept,
            r_squared,
            complexity: if slope.abs() < 0.01 {
                "O(1)".to_string()
            } else if (slope - 1.0).abs() < 0.1 {
                "O(n)".to_string()
            } else if (slope - 2.0).abs() < 0.1 {
                "O(n²)".to_string()
            } else {
                format!("O(n^{:.2})", slope)
            },
        })
    }

    /// Generate comparative report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut summaries = Vec::new();
        let mut scaling_analyses = HashMap::new();
        
        for monitor in &self.monitors {
            summaries.push(monitor.get_summary());
        }
        
        // Analyze scaling for common metrics
        if let Some(first_summary) = summaries.first() {
            for metric_name in first_summary.measurements.keys() {
                if let Some(analysis) = self.analyze_scaling(metric_name) {
                    scaling_analyses.insert(metric_name.clone(), analysis);
                }
            }
        }
        
        PerformanceReport {
            summaries,
            scaling_analyses,
            timestamp: Instant::now(),
        }
    }
}

/// Scaling analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub metric_name: String,
    pub data_points: Vec<(f64, f64)>,
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub complexity: String,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub summaries: Vec<PerformanceSummary>,
    pub scaling_analyses: HashMap<String, ScalingAnalysis>,
    pub timestamp: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        // Record some measurements
        monitor.record_duration("test_metric", Duration::from_millis(100));
        monitor.record_duration("test_metric", Duration::from_millis(150));
        monitor.record_duration("test_metric", Duration::from_millis(200));
        
        let summary = monitor.get_summary();
        let stats = &summary.measurements["test_metric"];
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, Duration::from_millis(100));
        assert_eq!(stats.max, Duration::from_millis(200));
        assert_eq!(stats.avg, Duration::from_millis(150));
    }

    #[test]
    fn test_measurement_handle() {
        let monitor = PerformanceMonitor::new();
        
        {
            let handle = monitor.start_measurement("auto_complete");
            thread::sleep(Duration::from_millis(10));
            // Handle drops here, auto-completing the measurement
        }
        
        let summary = monitor.get_summary();
        assert!(summary.measurements.contains_key("auto_complete"));
    }

    #[test]
    fn test_percentiles() {
        let monitor = PerformanceMonitor::new();
        
        for i in 0..100 {
            monitor.record_duration("percentile_test", Duration::from_millis(i));
        }
        
        let p50 = monitor.get_percentile("percentile_test", 50.0).unwrap();
        assert_eq!(p50, Duration::from_millis(50));
        
        let p95 = monitor.get_percentile("percentile_test", 95.0).unwrap();
        assert_eq!(p95, Duration::from_millis(95));
    }

    #[test]
    fn test_metric_comparison() {
        let monitor = PerformanceMonitor::new();
        
        monitor.record_duration("slow_metric", Duration::from_millis(200));
        monitor.record_duration("fast_metric", Duration::from_millis(100));
        
        let comparison = monitor.compare_metrics("fast_metric", "slow_metric").unwrap();
        assert!((comparison.speedup - 2.0).abs() < 0.01);
        assert_eq!(comparison.avg_difference, Duration::from_millis(100));
    }
}