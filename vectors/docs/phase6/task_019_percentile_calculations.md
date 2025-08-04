# Task 019: Advanced Percentile Calculations

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 008, 017, and 018 (PerformanceBenchmark, EnhancedPerformanceMetrics, and ConcurrentBenchmark). The advanced percentile calculation provides high-precision statistical analysis with interpolation, streaming estimation, and histogram-based calculations.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement advanced percentile calculation methods with interpolation support for P90, P95, P99, P99.9, P99.99, streaming percentile estimation for large datasets, histogram-based percentile calculation, and statistical accuracy validation.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. Advanced percentile calculation with linear interpolation
3. Support for P90, P95, P99, P99.9, P99.99 with high precision
4. Streaming percentile estimation for memory-efficient large datasets
5. Histogram-based percentile calculation for performance
6. Statistical accuracy validation and confidence intervals
7. Windows-compatible implementation with proper error handling

## Expected Code Structure to Add
```rust
use std::collections::VecDeque;
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPercentileCalculator {
    data_points: Vec<f64>,
    streaming_calculator: StreamingPercentileCalculator,
    histogram_calculator: HistogramPercentileCalculator,
    accuracy_validator: StatisticalAccuracyValidator,
    calculation_method: PercentileMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PercentileMethod {
    Exact,           // Sort all data - most accurate but memory intensive
    Streaming,       // P² algorithm - memory efficient for large datasets
    Histogram,       // Histogram-based - fast but slightly less accurate
    Hybrid,          // Automatically choose based on data size
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileResult {
    pub percentile: f64,           // Requested percentile (0.0-1.0)
    pub value: f64,               // Calculated percentile value
    pub method_used: PercentileMethod,
    pub confidence_interval: ConfidenceInterval,
    pub accuracy_score: f64,      // 0.0-1.0, higher is more accurate
    pub sample_count: usize,
    pub calculation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,    // e.g., 0.95 for 95% confidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileSuite {
    pub p50: PercentileResult,
    pub p90: PercentileResult,
    pub p95: PercentileResult,
    pub p99: PercentileResult,
    pub p99_9: PercentileResult,
    pub p99_99: PercentileResult,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub standard_deviation: f64,
    pub coefficient_of_variation: f64,
    pub interquartile_range: f64,
}

impl AdvancedPercentileCalculator {
    pub fn new(calculation_method: PercentileMethod) -> Self {
        Self {
            data_points: Vec::new(),
            streaming_calculator: StreamingPercentileCalculator::new(),
            histogram_calculator: HistogramPercentileCalculator::new(1000), // 1000 buckets
            accuracy_validator: StatisticalAccuracyValidator::new(),
            calculation_method,
        }
    }
    
    pub fn add_data_point(&mut self, value: f64) {
        // Validate input
        if !value.is_finite() {
            return; // Skip invalid values
        }
        
        match self.calculation_method {
            PercentileMethod::Exact => {
                self.data_points.push(value);
            }
            PercentileMethod::Streaming => {
                self.streaming_calculator.add_observation(value);
            }
            PercentileMethod::Histogram => {
                self.histogram_calculator.add_observation(value);
            }
            PercentileMethod::Hybrid => {
                self.data_points.push(value);
                self.streaming_calculator.add_observation(value);
                self.histogram_calculator.add_observation(value);
            }
        }
    }
    
    pub fn add_batch(&mut self, values: &[f64]) {
        for &value in values {
            self.add_data_point(value);
        }
    }
    
    pub fn calculate_percentile(&self, percentile: f64) -> Result<PercentileResult> {
        if percentile < 0.0 || percentile > 1.0 {
            return Err(anyhow::anyhow!("Percentile must be between 0.0 and 1.0"));
        }
        
        let start_time = std::time::Instant::now();
        
        let (value, method_used, sample_count) = match &self.calculation_method {
            PercentileMethod::Exact => {
                (self.calculate_exact_percentile(percentile)?, PercentileMethod::Exact, self.data_points.len())
            }
            PercentileMethod::Streaming => {
                (self.streaming_calculator.get_percentile(percentile)?, PercentileMethod::Streaming, self.streaming_calculator.sample_count())
            }
            PercentileMethod::Histogram => {
                (self.histogram_calculator.get_percentile(percentile)?, PercentileMethod::Histogram, self.histogram_calculator.sample_count())
            }
            PercentileMethod::Hybrid => {
                let method = self.choose_optimal_method();
                match method {
                    PercentileMethod::Exact => (self.calculate_exact_percentile(percentile)?, method, self.data_points.len()),
                    PercentileMethod::Streaming => (self.streaming_calculator.get_percentile(percentile)?, method, self.streaming_calculator.sample_count()),
                    PercentileMethod::Histogram => (self.histogram_calculator.get_percentile(percentile)?, method, self.histogram_calculator.sample_count()),
                    _ => unreachable!(),
                }
            }
        };
        
        let calculation_time = start_time.elapsed().as_millis() as f64;
        
        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(percentile, value, sample_count);
        
        // Calculate accuracy score
        let accuracy_score = self.accuracy_validator.calculate_accuracy_score(
            percentile, 
            value, 
            sample_count, 
            &method_used
        );
        
        Ok(PercentileResult {
            percentile,
            value,
            method_used,
            confidence_interval,
            accuracy_score,
            sample_count,
            calculation_time_ms: calculation_time,
        })
    }
    
    pub fn calculate_percentile_suite(&self) -> Result<PercentileSuite> {
        let percentiles = [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999];
        let mut results = Vec::new();
        
        for &p in &percentiles {
            results.push(self.calculate_percentile(p)?);
        }
        
        // Calculate basic statistics
        let data = match &self.calculation_method {
            PercentileMethod::Exact | PercentileMethod::Hybrid => &self.data_points,
            _ => return Err(anyhow::anyhow!("Basic statistics require exact data points")),
        };
        
        if data.is_empty() {
            return Err(anyhow::anyhow!("No data points available for calculation"));
        }
        
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        
        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = results[0].value; // P50
        
        // Standard deviation
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        let standard_deviation = variance.sqrt();
        
        // Coefficient of variation
        let coefficient_of_variation = if mean != 0.0 {
            standard_deviation / mean.abs()
        } else {
            0.0
        };
        
        // Interquartile range (P75 - P25)
        let p25 = self.calculate_exact_percentile(0.25)?;
        let p75 = self.calculate_exact_percentile(0.75)?;
        let interquartile_range = p75 - p25;
        
        Ok(PercentileSuite {
            p50: results[0].clone(),
            p90: results[1].clone(),
            p95: results[2].clone(),
            p99: results[3].clone(),
            p99_9: results[4].clone(),
            p99_99: results[5].clone(),
            min,
            max,
            mean,
            median,
            standard_deviation,
            coefficient_of_variation,
            interquartile_range,
        })
    }
    
    fn calculate_exact_percentile(&self, percentile: f64) -> Result<f64> {
        if self.data_points.is_empty() {
            return Err(anyhow::anyhow!("No data points available"));
        }
        
        let mut sorted_data = self.data_points.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        
        // Use linear interpolation for precise percentile calculation
        let index = percentile * (sorted_data.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        
        if lower_index == upper_index {
            Ok(sorted_data[lower_index])
        } else {
            let lower_value = sorted_data[lower_index];
            let upper_value = sorted_data[upper_index];
            let weight = index - lower_index as f64;
            
            Ok(lower_value + weight * (upper_value - lower_value))
        }
    }
    
    fn choose_optimal_method(&self) -> PercentileMethod {
        let data_size = self.data_points.len();
        
        if data_size < 1000 {
            PercentileMethod::Exact
        } else if data_size < 100000 {
            PercentileMethod::Histogram
        } else {
            PercentileMethod::Streaming
        }
    }
    
    fn calculate_confidence_interval(&self, percentile: f64, value: f64, sample_count: usize) -> ConfidenceInterval {
        if sample_count < 30 {
            // For small samples, use wide confidence intervals
            return ConfidenceInterval {
                lower_bound: value * 0.8,
                upper_bound: value * 1.2,
                confidence_level: 0.95,
            };
        }
        
        // For large samples, use normal approximation
        let z_score = 1.96; // 95% confidence interval
        let standard_error = (percentile * (1.0 - percentile) / sample_count as f64).sqrt();
        let margin_of_error = z_score * standard_error * value;
        
        ConfidenceInterval {
            lower_bound: (value - margin_of_error).max(0.0),
            upper_bound: value + margin_of_error,
            confidence_level: 0.95,
        }
    }
}

// Streaming Percentile Calculator using P² Algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingPercentileCalculator {
    markers: Vec<f64>,           // 5 markers for P² algorithm
    positions: Vec<f64>,         // Current positions of markers
    desired_positions: Vec<f64>, // Desired positions of markers
    sample_count: usize,
    target_percentile: f64,
}

impl StreamingPercentileCalculator {
    pub fn new() -> Self {
        Self {
            markers: vec![0.0; 5],
            positions: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            desired_positions: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            sample_count: 0,
            target_percentile: 0.95, // Default to P95
        }
    }
    
    pub fn add_observation(&mut self, value: f64) {
        self.sample_count += 1;
        
        if self.sample_count <= 5 {
            // Initialize first 5 observations
            self.markers[self.sample_count - 1] = value;
            if self.sample_count == 5 {
                self.markers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            }
            return;
        }
        
        // Find cell k
        let mut k = 0;
        for i in 0..4 {
            if value < self.markers[i + 1] {
                k = i;
                break;
            }
            k = i + 1;
        }
        
        // Increment positions of markers k+1 through 5
        for i in (k + 1)..5 {
            self.positions[i] += 1.0;
        }
        
        // Update desired positions
        self.desired_positions[1] = 1.0 + 2.0 * self.target_percentile * (self.sample_count - 1) as f64;
        self.desired_positions[2] = 1.0 + self.target_percentile * (self.sample_count - 1) as f64;
        self.desired_positions[3] = 1.0 + (1.0 + self.target_percentile) * (self.sample_count - 1) as f64;
        self.desired_positions[4] = self.sample_count as f64;
        
        // Adjust marker positions
        for i in 1..4 {
            let d = self.desired_positions[i] - self.positions[i];
            if (d >= 1.0 && self.positions[i + 1] - self.positions[i] > 1.0) ||
               (d <= -1.0 && self.positions[i - 1] - self.positions[i] < -1.0) {
                
                let d_sign = if d > 0.0 { 1.0 } else { -1.0 };
                
                // Try parabolic formula
                let parabolic = self.markers[i] + d_sign / (self.positions[i + 1] - self.positions[i - 1]) *
                    ((self.positions[i] - self.positions[i - 1] + d_sign) * (self.markers[i + 1] - self.markers[i]) / 
                     (self.positions[i + 1] - self.positions[i]) +
                     (self.positions[i + 1] - self.positions[i] - d_sign) * (self.markers[i] - self.markers[i - 1]) /
                     (self.positions[i] - self.positions[i - 1]));
                
                // Use parabolic if it's between adjacent markers, otherwise use linear
                if self.markers[i - 1] < parabolic && parabolic < self.markers[i + 1] {
                    self.markers[i] = parabolic;
                } else {
                    // Linear formula
                    self.markers[i] = self.markers[i] + d_sign * 
                        (if d_sign > 0.0 { self.markers[i + 1] - self.markers[i] } else { self.markers[i] - self.markers[i - 1] }) /
                        (if d_sign > 0.0 { self.positions[i + 1] - self.positions[i] } else { self.positions[i] - self.positions[i - 1] });
                }
                
                self.positions[i] += d_sign;
            }
        }
    }
    
    pub fn get_percentile(&self, percentile: f64) -> Result<f64> {
        if self.sample_count < 5 {
            return Err(anyhow::anyhow!("Not enough samples for streaming percentile calculation"));
        }
        
        // For simplicity, return the middle marker (could be enhanced to interpolate for any percentile)
        Ok(self.markers[2])
    }
    
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

// Histogram-based Percentile Calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramPercentileCalculator {
    buckets: Vec<usize>,
    bucket_boundaries: Vec<f64>,
    min_value: f64,
    max_value: f64,
    total_count: usize,
    auto_resize: bool,
}

impl HistogramPercentileCalculator {
    pub fn new(bucket_count: usize) -> Self {
        Self {
            buckets: vec![0; bucket_count],
            bucket_boundaries: Vec::new(),
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            total_count: 0,
            auto_resize: true,
        }
    }
    
    pub fn add_observation(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        
        if self.total_count == 0 {
            self.min_value = value;
            self.max_value = value;
            self.initialize_buckets();
        } else {
            if value < self.min_value {
                self.min_value = value;
                if self.auto_resize {
                    self.resize_buckets();
                }
            }
            if value > self.max_value {
                self.max_value = value;
                if self.auto_resize {
                    self.resize_buckets();
                }
            }
        }
        
        let bucket_index = self.get_bucket_index(value);
        if bucket_index < self.buckets.len() {
            self.buckets[bucket_index] += 1;
            self.total_count += 1;
        }
    }
    
    pub fn get_percentile(&self, percentile: f64) -> Result<f64> {
        if self.total_count == 0 {
            return Err(anyhow::anyhow!("No observations in histogram"));
        }
        
        let target_count = (percentile * self.total_count as f64) as usize;
        let mut cumulative_count = 0;
        
        for (i, &bucket_count) in self.buckets.iter().enumerate() {
            cumulative_count += bucket_count;
            if cumulative_count >= target_count {
                // Linear interpolation within bucket
                let bucket_start = self.bucket_boundaries[i];
                let bucket_end = self.bucket_boundaries[i + 1];
                let bucket_position = if bucket_count > 0 {
                    (target_count - (cumulative_count - bucket_count)) as f64 / bucket_count as f64
                } else {
                    0.5
                };
                
                return Ok(bucket_start + bucket_position * (bucket_end - bucket_start));
            }
        }
        
        Ok(self.max_value)
    }
    
    pub fn sample_count(&self) -> usize {
        self.total_count
    }
    
    fn initialize_buckets(&mut self) {
        let bucket_count = self.buckets.len();
        let range = self.max_value - self.min_value;
        let bucket_width = if range > 0.0 { range / bucket_count as f64 } else { 1.0 };
        
        self.bucket_boundaries.clear();
        for i in 0..=bucket_count {
            self.bucket_boundaries.push(self.min_value + i as f64 * bucket_width);
        }
    }
    
    fn resize_buckets(&mut self) {
        // Simplified resize - in production, would preserve existing data
        self.initialize_buckets();
    }
    
    fn get_bucket_index(&self, value: f64) -> usize {
        if self.bucket_boundaries.len() <= 1 {
            return 0;
        }
        
        let range = self.max_value - self.min_value;
        if range <= 0.0 {
            return 0;
        }
        
        let normalized = (value - self.min_value) / range;
        let index = (normalized * (self.buckets.len() - 1) as f64) as usize;
        index.min(self.buckets.len() - 1)
    }
}

// Statistical Accuracy Validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAccuracyValidator;

impl StatisticalAccuracyValidator {
    pub fn new() -> Self {
        Self
    }
    
    pub fn calculate_accuracy_score(
        &self,
        _percentile: f64,
        _value: f64,
        sample_count: usize,
        method: &PercentileMethod,
    ) -> f64 {
        // Base accuracy on method and sample size
        let method_accuracy = match method {
            PercentileMethod::Exact => 1.0,
            PercentileMethod::Histogram => 0.95,
            PercentileMethod::Streaming => 0.90,
            PercentileMethod::Hybrid => 0.98,
        };
        
        // Adjust for sample size
        let sample_factor = if sample_count < 30 {
            0.7
        } else if sample_count < 100 {
            0.85
        } else if sample_count < 1000 {
            0.95
        } else {
            1.0
        };
        
        method_accuracy * sample_factor
    }
}
```

## Success Criteria
- Advanced percentile calculations work correctly for all specified percentiles
- Linear interpolation provides precise results between data points
- Streaming percentile estimation handles large datasets efficiently
- Histogram-based calculation provides fast approximations
- Statistical accuracy validation provides meaningful confidence scores
- Confidence intervals are calculated correctly
- Memory usage is optimized for different calculation methods
- All methods handle edge cases (empty data, single values, etc.) gracefully

## Time Limit
10 minutes maximum