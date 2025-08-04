# Task 39: Advanced Statistical Calculations

## Context
You are implementing Phase 4 of a vector indexing system. The basic performance monitoring is implemented with time recording methods. Now you need to enhance the statistical calculations with advanced metrics including percentiles, regression analysis, and trend detection.

## Current State
- `src/monitor.rs` exists with `PerformanceMonitor` struct
- Basic time recording methods are implemented
- Thread-safe monitoring wrapper is available
- Need advanced statistical calculations for better performance insights

## Task Objective
Implement advanced statistical calculations including more accurate percentiles, trend analysis, regression calculations, and statistical confidence intervals for performance data.

## Implementation Requirements

### 1. Add advanced percentile calculations
Add these enhanced statistical methods to `PerformanceMonitor`:
```rust
impl PerformanceMonitor {
    /// Calculate multiple percentiles efficiently
    pub fn calculate_percentiles(&self, durations: &VecDeque<Duration>, percentiles: &[f64]) -> Vec<Duration> {
        if durations.is_empty() {
            return vec![Duration::from_millis(0); percentiles.len()];
        }
        
        let mut sorted: Vec<_> = durations.iter().cloned().collect();
        sorted.sort();
        
        percentiles.iter().map(|&p| {
            let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
            sorted[index.min(sorted.len() - 1)]
        }).collect()
    }
    
    /// Calculate standard deviation
    pub fn standard_deviation(&self, durations: &VecDeque<Duration>) -> Duration {
        if durations.len() < 2 {
            return Duration::from_millis(0);
        }
        
        let avg_ms = self.average(durations).as_secs_f64() * 1000.0;
        let variance: f64 = durations.iter()
            .map(|d| {
                let diff = d.as_secs_f64() * 1000.0 - avg_ms;
                diff * diff
            })
            .sum::<f64>() / (durations.len() - 1) as f64;
        
        Duration::from_millis(variance.sqrt() as u64)
    }
    
    /// Calculate coefficient of variation (CV)
    pub fn coefficient_of_variation(&self, durations: &VecDeque<Duration>) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        
        let avg = self.average(durations).as_secs_f64() * 1000.0;
        let std_dev = self.standard_deviation(durations).as_secs_f64() * 1000.0;
        
        if avg == 0.0 {
            0.0
        } else {
            std_dev / avg
        }
    }
    
    /// Calculate interquartile range (IQR)
    pub fn interquartile_range(&self, durations: &VecDeque<Duration>) -> Duration {
        if durations.len() < 4 {
            return Duration::from_millis(0);
        }
        
        let percentiles = self.calculate_percentiles(durations, &[25.0, 75.0]);
        percentiles[1] - percentiles[0]
    }
    
    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, durations: &VecDeque<Duration>) -> Vec<Duration> {
        if durations.len() < 4 {
            return Vec::new();
        }
        
        let percentiles = self.calculate_percentiles(durations, &[25.0, 75.0]);
        let q1 = percentiles[0];
        let q3 = percentiles[1];
        let iqr = q3 - q1;
        
        let lower_bound = q1.saturating_sub(iqr + iqr / 2); // 1.5 * IQR
        let upper_bound = q3 + iqr + iqr / 2;
        
        durations.iter()
            .filter(|&&d| d < lower_bound || d > upper_bound)
            .cloned()
            .collect()
    }
}
```

### 2. Add trend analysis methods
```rust
impl PerformanceMonitor {
    /// Calculate linear regression slope for trend detection
    pub fn calculate_trend_slope(&self, durations: &VecDeque<Duration>) -> f64 {
        if durations.len() < 2 {
            return 0.0;
        }
        
        let n = durations.len() as f64;
        let x_mean = (n - 1.0) / 2.0; // Time indices
        let y_values: Vec<f64> = durations.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        let y_mean = y_values.iter().sum::<f64>() / n;
        
        let numerator: f64 = (0..durations.len())
            .map(|i| (i as f64 - x_mean) * (y_values[i] - y_mean))
            .sum();
        
        let denominator: f64 = (0..durations.len())
            .map(|i| {
                let diff = i as f64 - x_mean;
                diff * diff
            })
            .sum();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Calculate correlation coefficient (R-squared)
    pub fn calculate_correlation(&self, durations: &VecDeque<Duration>) -> f64 {
        if durations.len() < 2 {
            return 0.0;
        }
        
        let n = durations.len() as f64;
        let x_values: Vec<f64> = (0..durations.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = durations.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        
        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;
        
        let numerator: f64 = x_values.iter().zip(y_values.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        let x_var: f64 = x_values.iter()
            .map(|x| (x - x_mean).powi(2))
            .sum();
        
        let y_var: f64 = y_values.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();
        
        if x_var == 0.0 || y_var == 0.0 {
            0.0
        } else {
            let correlation = numerator / (x_var * y_var).sqrt();
            correlation * correlation // R-squared
        }
    }
    
    /// Predict next value using linear regression
    pub fn predict_next_value(&self, durations: &VecDeque<Duration>) -> Duration {
        if durations.len() < 2 {
            return self.average(durations);
        }
        
        let slope = self.calculate_trend_slope(durations);
        let n = durations.len() as f64;
        let y_values: Vec<f64> = durations.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        let y_mean = y_values.iter().sum::<f64>() / n;
        let x_mean = (n - 1.0) / 2.0;
        
        let intercept = y_mean - slope * x_mean;
        let predicted_ms = slope * n + intercept;
        
        Duration::from_millis(predicted_ms.max(0.0) as u64)
    }
}
```

### 3. Add confidence interval calculations
```rust
impl PerformanceMonitor {
    /// Calculate confidence interval for mean
    pub fn confidence_interval(&self, durations: &VecDeque<Duration>, confidence_level: f64) -> (Duration, Duration) {
        if durations.len() < 2 {
            let avg = self.average(durations);
            return (avg, avg);
        }
        
        let mean = self.average(durations).as_secs_f64() * 1000.0;
        let std_dev = self.standard_deviation(durations).as_secs_f64() * 1000.0;
        let n = durations.len() as f64;
        
        // Use t-distribution for small samples, normal for large samples
        let t_value = if n < 30.0 {
            self.t_value(confidence_level, (n - 1.0) as i32)
        } else {
            self.z_value(confidence_level)
        };
        
        let margin_of_error = t_value * std_dev / n.sqrt();
        let lower = (mean - margin_of_error).max(0.0);
        let upper = mean + margin_of_error;
        
        (Duration::from_millis(lower as u64), Duration::from_millis(upper as u64))
    }
    
    /// Approximate t-value for confidence intervals
    fn t_value(&self, confidence_level: f64, degrees_of_freedom: i32) -> f64 {
        // Simplified t-table approximation for common confidence levels
        let alpha = 1.0 - confidence_level;
        match (confidence_level * 100.0) as i32 {
            95 => match degrees_of_freedom {
                1 => 12.706,
                2 => 4.303,
                3 => 3.182,
                4 => 2.776,
                5 => 2.571,
                10 => 2.228,
                20 => 2.086,
                30 => 2.042,
                _ => 1.96, // Approximate for large df
            },
            99 => match degrees_of_freedom {
                1 => 63.657,
                2 => 9.925,
                3 => 5.841,
                4 => 4.604,
                5 => 4.032,
                10 => 3.169,
                20 => 2.845,
                30 => 2.750,
                _ => 2.576,
            },
            _ => 1.96, // Default to 95% normal approximation
        }
    }
    
    /// Z-value for normal distribution
    fn z_value(&self, confidence_level: f64) -> f64 {
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96, // Default to 95%
        }
    }
}
```

### 4. Enhanced performance statistics
Update the `PerformanceStats` struct:
```rust
#[derive(Debug, Clone)]
pub struct AdvancedPerformanceStats {
    pub basic_stats: PerformanceStats,
    
    // Query statistics
    pub query_std_dev: Duration,
    pub query_cv: f64,
    pub query_iqr: Duration,
    pub query_outliers: Vec<Duration>,
    pub query_trend_slope: f64,
    pub query_correlation: f64,
    pub query_confidence_interval: (Duration, Duration),
    pub query_prediction: Duration,
    
    // Index statistics
    pub index_std_dev: Duration,
    pub index_cv: f64,
    pub index_iqr: Duration,
    pub index_outliers: Vec<Duration>,
    pub index_trend_slope: f64,
    pub index_correlation: f64,
    pub index_confidence_interval: (Duration, Duration),
    pub index_prediction: Duration,
}
```

### 5. Add method to get advanced statistics
```rust
impl PerformanceMonitor {
    pub fn get_advanced_stats(&self, confidence_level: f64) -> AdvancedPerformanceStats {
        AdvancedPerformanceStats {
            basic_stats: self.get_stats(),
            
            // Query advanced statistics
            query_std_dev: self.standard_deviation(&self.query_times),
            query_cv: self.coefficient_of_variation(&self.query_times),
            query_iqr: self.interquartile_range(&self.query_times),
            query_outliers: self.detect_outliers(&self.query_times),
            query_trend_slope: self.calculate_trend_slope(&self.query_times),
            query_correlation: self.calculate_correlation(&self.query_times),
            query_confidence_interval: self.confidence_interval(&self.query_times, confidence_level),
            query_prediction: self.predict_next_value(&self.query_times),
            
            // Index advanced statistics
            index_std_dev: self.standard_deviation(&self.index_times),
            index_cv: self.coefficient_of_variation(&self.index_times),
            index_iqr: self.interquartile_range(&self.index_times),
            index_outliers: self.detect_outliers(&self.index_times),
            index_trend_slope: self.calculate_trend_slope(&self.index_times),
            index_correlation: self.calculate_correlation(&self.index_times),
            index_confidence_interval: self.confidence_interval(&self.index_times, confidence_level),
            index_prediction: self.predict_next_value(&self.index_times),
        }
    }
}
```

### 6. Add comprehensive statistical tests
```rust
#[cfg(test)]
mod advanced_stats_tests {
    use super::*;
    
    #[test]
    fn test_percentile_calculations() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add known data: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ms
        for i in 1..=10 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let percentiles = monitor.calculate_percentiles(&monitor.query_times, &[25.0, 50.0, 75.0, 95.0]);
        
        assert_eq!(percentiles[0], Duration::from_millis(30)); // 25th percentile
        assert_eq!(percentiles[1], Duration::from_millis(60)); // 50th percentile (median)
        assert_eq!(percentiles[2], Duration::from_millis(80)); // 75th percentile
        assert_eq!(percentiles[3], Duration::from_millis(100)); // 95th percentile
    }
    
    #[test]
    fn test_standard_deviation() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add data with known standard deviation
        monitor.record_query_time(Duration::from_millis(10));
        monitor.record_query_time(Duration::from_millis(20));
        monitor.record_query_time(Duration::from_millis(30));
        
        let std_dev = monitor.standard_deviation(&monitor.query_times);
        
        // Standard deviation of [10, 20, 30] is 10ms
        assert_eq!(std_dev, Duration::from_millis(10));
    }
    
    #[test]
    fn test_coefficient_of_variation() {
        let mut monitor = PerformanceMonitor::new();
        
        monitor.record_query_time(Duration::from_millis(10));
        monitor.record_query_time(Duration::from_millis(20));
        monitor.record_query_time(Duration::from_millis(30));
        
        let cv = monitor.coefficient_of_variation(&monitor.query_times);
        
        // CV = std_dev / mean = 10 / 20 = 0.5
        assert!((cv - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_outlier_detection() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add normal data
        for i in 1..=10 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        // Add outliers
        monitor.record_query_time(Duration::from_millis(1000)); // Extreme outlier
        
        let outliers = monitor.detect_outliers(&monitor.query_times);
        
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&Duration::from_millis(1000)));
    }
    
    #[test]
    fn test_trend_analysis() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add increasing trend: 10, 20, 30, 40, 50 ms
        for i in 1..=5 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let slope = monitor.calculate_trend_slope(&monitor.query_times);
        
        // Should detect positive trend (slope should be positive)
        assert!(slope > 0.0);
        
        let correlation = monitor.calculate_correlation(&monitor.query_times);
        
        // Perfect linear correlation should be close to 1.0
        assert!(correlation > 0.9);
    }
    
    #[test]
    fn test_prediction() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add linear trend
        for i in 1..=5 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let prediction = monitor.predict_next_value(&monitor.query_times);
        
        // Next value should be around 60ms
        assert!(prediction >= Duration::from_millis(55));
        assert!(prediction <= Duration::from_millis(65));
    }
    
    #[test]
    fn test_confidence_intervals() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add consistent data
        for _ in 0..10 {
            monitor.record_query_time(Duration::from_millis(50));
        }
        
        let (lower, upper) = monitor.confidence_interval(&monitor.query_times, 0.95);
        
        // With consistent data, confidence interval should be narrow
        assert!(upper.saturating_sub(lower) < Duration::from_millis(5));
        
        // Mean should be within the interval
        let mean = monitor.average(&monitor.query_times);
        assert!(mean >= lower && mean <= upper);
    }
    
    #[test]
    fn test_advanced_stats_integration() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add sample data
        for i in 1..=20 {
            monitor.record_query_time(Duration::from_millis(i * 5));
            monitor.record_index_time(Duration::from_millis(i * 10));
        }
        
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Verify all fields are populated
        assert!(advanced_stats.query_std_dev > Duration::from_millis(0));
        assert!(advanced_stats.query_cv > 0.0);
        assert!(advanced_stats.query_iqr > Duration::from_millis(0));
        assert!(advanced_stats.query_trend_slope != 0.0);
        assert!(advanced_stats.query_correlation > 0.0);
        
        assert!(advanced_stats.index_std_dev > Duration::from_millis(0));
        assert!(advanced_stats.index_cv > 0.0);
        assert!(advanced_stats.index_iqr > Duration::from_millis(0));
    }
}
```

## Success Criteria
- [ ] Advanced percentile calculations work accurately
- [ ] Standard deviation and CV calculations are correct
- [ ] Outlier detection using IQR method functions properly
- [ ] Trend analysis with regression calculations works
- [ ] Confidence interval calculations are accurate
- [ ] Prediction functionality provides reasonable forecasts
- [ ] Advanced statistics integration works seamlessly
- [ ] All tests pass with accurate results
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Uses linear regression for trend analysis and prediction
- IQR method for robust outlier detection
- T-distribution for small samples, normal distribution for large samples
- Coefficient of variation helps assess relative variability
- Confidence intervals provide uncertainty quantification
- All calculations handle edge cases gracefully