/*!
Phase 5.4: Advanced Metrics Collection System
Comprehensive metrics collection, aggregation, and storage for LLMKG performance monitoring
*/

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram {
        count: u64,
        sum: f64,
        buckets: Vec<(f64, u64)>, // (upper_bound, count)
    },
    Timer {
        count: u64,
        sum_duration_ms: f64,
        min_ms: f64,
        max_ms: f64,
        percentiles: HashMap<String, f64>, // "p50", "p90", "p95", "p99"
    },
    Summary {
        count: u64,
        sum: f64,
        quantiles: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    pub name: String,
    pub value: MetricValue,
    pub labels: HashMap<String, String>,
    pub timestamp: u64,
    pub help: Option<String>,
}

pub struct Counter {
    value: Arc<RwLock<u64>>,
    name: String,
    labels: HashMap<String, String>,
}

impl Counter {
    pub fn new(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            value: Arc::new(RwLock::new(0)),
            name,
            labels,
        }
    }
    
    pub fn increment(&self) {
        self.add(1);
    }
    
    pub fn add(&self, value: u64) {
        let mut counter = self.value.write().unwrap();
        *counter += value;
    }
    
    pub fn get(&self) -> u64 {
        *self.value.read().unwrap()
    }
    
    pub fn sample(&self) -> MetricSample {
        MetricSample {
            name: self.name.clone(),
            value: MetricValue::Counter(self.get()),
            labels: self.labels.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            help: None,
        }
    }
}

pub struct Gauge {
    value: Arc<RwLock<f64>>,
    name: String,
    labels: HashMap<String, String>,
}

impl Gauge {
    pub fn new(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            value: Arc::new(RwLock::new(0.0)),
            name,
            labels,
        }
    }
    
    pub fn set(&self, value: f64) {
        let mut gauge = self.value.write().unwrap();
        *gauge = value;
    }
    
    pub fn add(&self, value: f64) {
        let mut gauge = self.value.write().unwrap();
        *gauge += value;
    }
    
    pub fn subtract(&self, value: f64) {
        let mut gauge = self.value.write().unwrap();
        *gauge -= value;
    }
    
    pub fn get(&self) -> f64 {
        *self.value.read().unwrap()
    }
    
    pub fn sample(&self) -> MetricSample {
        MetricSample {
            name: self.name.clone(),
            value: MetricValue::Gauge(self.get()),
            labels: self.labels.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            help: None,
        }
    }
}

pub struct Histogram {
    buckets: Vec<f64>,
    bucket_counts: Arc<RwLock<Vec<u64>>>,
    count: Arc<RwLock<u64>>,
    sum: Arc<RwLock<f64>>,
    name: String,
    labels: HashMap<String, String>,
}

impl Histogram {
    pub fn new(name: String, labels: HashMap<String, String>, buckets: Vec<f64>) -> Self {
        let bucket_count = buckets.len();
        Self {
            buckets,
            bucket_counts: Arc::new(RwLock::new(vec![0; bucket_count])),
            count: Arc::new(RwLock::new(0)),
            sum: Arc::new(RwLock::new(0.0)),
            name,
            labels,
        }
    }
    
    pub fn with_default_buckets(name: String, labels: HashMap<String, String>) -> Self {
        let buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, f64::INFINITY
        ];
        Self::new(name, labels, buckets)
    }
    
    pub fn observe(&self, value: f64) {
        {
            let mut count = self.count.write().unwrap();
            *count += 1;
        }
        
        {
            let mut sum = self.sum.write().unwrap();
            *sum += value;
        }
        
        {
            let mut bucket_counts = self.bucket_counts.write().unwrap();
            for (i, &bucket_upper_bound) in self.buckets.iter().enumerate() {
                if value <= bucket_upper_bound {
                    bucket_counts[i] += 1;
                }
            }
        }
    }
    
    pub fn sample(&self) -> MetricSample {
        let count = *self.count.read().unwrap();
        let sum = *self.sum.read().unwrap();
        let bucket_counts = self.bucket_counts.read().unwrap();
        
        let buckets: Vec<(f64, u64)> = self.buckets.iter()
            .zip(bucket_counts.iter())
            .map(|(&upper_bound, &count)| (upper_bound, count))
            .collect();
        
        MetricSample {
            name: self.name.clone(),
            value: MetricValue::Histogram { count, sum, buckets },
            labels: self.labels.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            help: None,
        }
    }
}

pub struct Timer {
    histogram: Histogram,
    samples: Arc<Mutex<Vec<f64>>>,
    max_samples: usize,
}

impl Timer {
    pub fn new(name: String, labels: HashMap<String, String>) -> Self {
        let histogram = Histogram::with_default_buckets(name.clone(), labels.clone());
        Self {
            histogram,
            samples: Arc::new(Mutex::new(Vec::new())),
            max_samples: 10000, // Keep last 10k samples for percentile calculation
        }
    }
    
    pub fn time<F, R>(&self, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        self.observe_duration(duration);
        result
    }
    
    pub fn observe_duration(&self, duration: Duration) {
        let duration_ms = duration.as_secs_f64() * 1000.0;
        self.histogram.observe(duration_ms);
        
        // Keep samples for percentile calculation
        {
            let mut samples = self.samples.lock().unwrap();
            samples.push(duration_ms);
            
            // Keep only the most recent samples
            if samples.len() > self.max_samples {
                samples.remove(0);
            }
        }
    }
    
    pub fn calculate_percentiles(&self) -> HashMap<String, f64> {
        let mut samples = self.samples.lock().unwrap();
        if samples.is_empty() {
            return HashMap::new();
        }
        
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut percentiles = HashMap::new();
        
        let percentile_values = [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)];
        
        for (name, percentile) in percentile_values {
            let index = ((samples.len() as f64 - 1.0) * percentile) as usize;
            percentiles.insert(name.to_string(), samples[index]);
        }
        
        percentiles
    }
    
    pub fn sample(&self) -> MetricSample {
        let histogram_sample = self.histogram.sample();
        let percentiles = self.calculate_percentiles();
        
        if let MetricValue::Histogram { count, sum, .. } = histogram_sample.value {
            let samples = self.samples.lock().unwrap();
            let min_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let max_ms = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            
            MetricSample {
                name: self.histogram.name.clone(),
                value: MetricValue::Timer {
                    count,
                    sum_duration_ms: sum,
                    min_ms: if min_ms.is_finite() { min_ms } else { 0.0 },
                    max_ms: if max_ms.is_finite() { max_ms } else { 0.0 },
                    percentiles,
                },
                labels: self.histogram.labels.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                help: None,
            }
        } else {
            histogram_sample
        }
    }
}

pub struct MetricRegistry {
    counters: Arc<RwLock<HashMap<String, Arc<Counter>>>>,
    gauges: Arc<RwLock<HashMap<String, Arc<Gauge>>>>,
    histograms: Arc<RwLock<HashMap<String, Arc<Histogram>>>>,
    timers: Arc<RwLock<HashMap<String, Arc<Timer>>>>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    fn metric_key(name: &str, labels: &HashMap<String, String>) -> String {
        let mut key = name.to_string();
        if !labels.is_empty() {
            let mut label_pairs: Vec<_> = labels.iter().collect();
            label_pairs.sort_by_key(|(k, _)| *k);
            
            key.push('{');
            for (i, (k, v)) in label_pairs.iter().enumerate() {
                if i > 0 {
                    key.push(',');
                }
                key.push_str(&format!("{}=\"{}\"", k, v));
            }
            key.push('}');
        }
        key
    }
    
    pub fn counter(&self, name: &str, labels: HashMap<String, String>) -> Arc<Counter> {
        let key = Self::metric_key(name, &labels);
        let mut counters = self.counters.write().unwrap();
        
        if let Some(counter) = counters.get(&key) {
            counter.clone()
        } else {
            let counter = Arc::new(Counter::new(name.to_string(), labels));
            counters.insert(key, counter.clone());
            counter
        }
    }
    
    pub fn gauge(&self, name: &str, labels: HashMap<String, String>) -> Arc<Gauge> {
        let key = Self::metric_key(name, &labels);
        let mut gauges = self.gauges.write().unwrap();
        
        if let Some(gauge) = gauges.get(&key) {
            gauge.clone()
        } else {
            let gauge = Arc::new(Gauge::new(name.to_string(), labels));
            gauges.insert(key, gauge.clone());
            gauge
        }
    }
    
    pub fn histogram(&self, name: &str, labels: HashMap<String, String>, buckets: Option<Vec<f64>>) -> Arc<Histogram> {
        let key = Self::metric_key(name, &labels);
        let mut histograms = self.histograms.write().unwrap();
        
        if let Some(histogram) = histograms.get(&key) {
            histogram.clone()
        } else {
            let histogram = if let Some(buckets) = buckets {
                Arc::new(Histogram::new(name.to_string(), labels, buckets))
            } else {
                Arc::new(Histogram::with_default_buckets(name.to_string(), labels))
            };
            histograms.insert(key, histogram.clone());
            histogram
        }
    }
    
    pub fn timer(&self, name: &str, labels: HashMap<String, String>) -> Arc<Timer> {
        let key = Self::metric_key(name, &labels);
        let mut timers = self.timers.write().unwrap();
        
        if let Some(timer) = timers.get(&key) {
            timer.clone()
        } else {
            let timer = Arc::new(Timer::new(name.to_string(), labels));
            timers.insert(key, timer.clone());
            timer
        }
    }
    
    pub fn collect_all_samples(&self) -> Vec<MetricSample> {
        let mut samples = Vec::new();
        
        // Collect counter samples
        {
            let counters = self.counters.read().unwrap();
            for counter in counters.values() {
                samples.push(counter.sample());
            }
        }
        
        // Collect gauge samples
        {
            let gauges = self.gauges.read().unwrap();
            for gauge in gauges.values() {
                samples.push(gauge.sample());
            }
        }
        
        // Collect histogram samples
        {
            let histograms = self.histograms.read().unwrap();
            for histogram in histograms.values() {
                samples.push(histogram.sample());
            }
        }
        
        // Collect timer samples
        {
            let timers = self.timers.read().unwrap();
            for timer in timers.values() {
                samples.push(timer.sample());
            }
        }
        
        samples
    }
    
    pub fn clear(&self) {
        self.counters.write().unwrap().clear();
        self.gauges.write().unwrap().clear();
        self.histograms.write().unwrap().clear();
        self.timers.write().unwrap().clear();
    }
    
    pub fn metrics_count(&self) -> usize {
        let counters_count = self.counters.read().unwrap().len();
        let gauges_count = self.gauges.read().unwrap().len();
        let histograms_count = self.histograms.read().unwrap().len();
        let timers_count = self.timers.read().unwrap().len();
        
        counters_count + gauges_count + histograms_count + timers_count
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter".to_string(), HashMap::new());
        
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        let sample = counter.sample();
        assert_eq!(sample.name, "test_counter");
        if let MetricValue::Counter(value) = sample.value {
            assert_eq!(value, 6);
        } else {
            panic!("Expected Counter metric value");
        }
    }
    
    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge".to_string(), HashMap::new());
        
        assert_eq!(gauge.get(), 0.0);
        
        gauge.set(10.5);
        assert_eq!(gauge.get(), 10.5);
        
        gauge.add(2.5);
        assert_eq!(gauge.get(), 13.0);
        
        gauge.subtract(3.0);
        assert_eq!(gauge.get(), 10.0);
    }
    
    #[test]
    fn test_histogram() {
        let buckets = vec![1.0, 5.0, 10.0, f64::INFINITY];
        let histogram = Histogram::new("test_histogram".to_string(), HashMap::new(), buckets);
        
        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(15.0);
        
        let sample = histogram.sample();
        if let MetricValue::Histogram { count, sum, buckets } = sample.value {
            assert_eq!(count, 4);
            assert_eq!(sum, 25.5);
            
            // Check bucket counts
            assert_eq!(buckets[0], (1.0, 1));   // 0.5 <= 1.0
            assert_eq!(buckets[1], (5.0, 2));   // 0.5, 3.0 <= 5.0
            assert_eq!(buckets[2], (10.0, 3));  // 0.5, 3.0, 7.0 <= 10.0
            assert_eq!(buckets[3], (f64::INFINITY, 4)); // All values <= infinity
        } else {
            panic!("Expected Histogram metric value");
        }
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new("test_timer".to_string(), HashMap::new());
        
        // Simulate some operations with different durations
        timer.observe_duration(Duration::from_millis(10));
        timer.observe_duration(Duration::from_millis(20));
        timer.observe_duration(Duration::from_millis(30));
        timer.observe_duration(Duration::from_millis(100));
        
        let sample = timer.sample();
        if let MetricValue::Timer { count, sum_duration_ms, min_ms, max_ms, percentiles } = sample.value {
            assert_eq!(count, 4);
            assert_eq!(sum_duration_ms, 160.0);
            assert_eq!(min_ms, 10.0);
            assert_eq!(max_ms, 100.0);
            
            // Check that percentiles are calculated
            assert!(percentiles.contains_key("p50"));
            assert!(percentiles.contains_key("p90"));
        } else {
            panic!("Expected Timer metric value");
        }
    }
    
    #[test]
    fn test_metric_registry() {
        let registry = MetricRegistry::new();
        
        // Test counter registration and retrieval
        let counter1 = registry.counter("requests", HashMap::new());
        let counter2 = registry.counter("requests", HashMap::new());
        
        // Should return the same instance
        assert!(Arc::ptr_eq(&counter1, &counter2));
        
        // Test with different labels
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        let counter3 = registry.counter("requests", labels);
        
        // Should be different instances
        assert!(!Arc::ptr_eq(&counter1, &counter3));
        
        counter1.increment();
        counter3.add(5);
        
        let samples = registry.collect_all_samples();
        assert_eq!(samples.len(), 2);
        
        assert_eq!(registry.metrics_count(), 2);
    }
    
    #[test]
    fn test_timer_operation() {
        let timer = Timer::new("operation_time".to_string(), HashMap::new());
        
        let result = timer.time(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        
        let sample = timer.sample();
        if let MetricValue::Timer { count, sum_duration_ms, .. } = sample.value {
            assert_eq!(count, 1);
            assert!(sum_duration_ms >= 10.0); // Should be at least 10ms
        } else {
            panic!("Expected Timer metric value");
        }
    }
}