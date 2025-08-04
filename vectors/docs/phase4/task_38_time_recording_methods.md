# Task 38: Implement Time Recording Methods

## Context
You are implementing Phase 4 of a vector indexing system. The basic performance monitor structure was created in the previous task. Now you need to enhance the time recording functionality with timing utilities and automatic measurement helpers.

## Current State
- `src/monitor.rs` exists with `PerformanceMonitor` struct
- Basic time recording methods are implemented
- Statistical calculations are working

## Task Objective
Enhance the time recording functionality with timing utilities, automatic measurement, and integration helpers for easy performance tracking.

## Implementation Requirements

### 1. Add timing utility functions
Add these timing utilities to the `PerformanceMonitor` implementation:
```rust
impl PerformanceMonitor {
    // ... existing methods ...
    
    /// Time a query operation and record the duration
    pub fn time_query<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record_query_time(duration);
        result
    }
    
    /// Time an indexing operation and record the duration
    pub fn time_index<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record_index_time(duration);
        result
    }
    
    /// Time a generic operation and return both result and duration
    pub fn time_operation<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Create a timer that records when dropped
    pub fn start_query_timer(&mut self) -> QueryTimer {
        QueryTimer::new(self)
    }
    
    /// Create a timer that records when dropped
    pub fn start_index_timer(&mut self) -> IndexTimer {
        IndexTimer::new(self)
    }
}
```

### 2. Add RAII timer structs
Add these timer structs before the `PerformanceMonitor` implementation:
```rust
use std::sync::{Arc, Mutex};

pub struct QueryTimer {
    start_time: Instant,
    monitor: *mut PerformanceMonitor,
}

pub struct IndexTimer {
    start_time: Instant,
    monitor: *mut PerformanceMonitor,
}

impl QueryTimer {
    fn new(monitor: &mut PerformanceMonitor) -> Self {
        Self {
            start_time: Instant::now(),
            monitor: monitor as *mut PerformanceMonitor,
        }
    }
    
    /// Stop the timer early and record the time
    pub fn stop(self) {
        // Timer will be automatically recorded when dropped
        drop(self);
    }
}

impl Drop for QueryTimer {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        unsafe {
            (*self.monitor).record_query_time(duration);
        }
    }
}

impl IndexTimer {
    fn new(monitor: &mut PerformanceMonitor) -> Self {
        Self {
            start_time: Instant::now(),
            monitor: monitor as *mut PerformanceMonitor,
        }
    }
    
    /// Stop the timer early and record the time
    pub fn stop(self) {
        drop(self);
    }
}

impl Drop for IndexTimer {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        unsafe {
            (*self.monitor).record_index_time(duration);
        }
    }
}
```

### 3. Add thread-safe monitoring wrapper
Add this thread-safe wrapper:
```rust
#[derive(Clone)]
pub struct SharedPerformanceMonitor {
    inner: Arc<Mutex<PerformanceMonitor>>,
}

impl SharedPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PerformanceMonitor::new())),
        }
    }
    
    pub fn with_capacity(max_samples: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PerformanceMonitor::with_capacity(max_samples))),
        }
    }
    
    pub fn record_query_time(&self, duration: Duration) {
        if let Ok(mut monitor) = self.inner.lock() {
            monitor.record_query_time(duration);
        }
    }
    
    pub fn record_index_time(&self, duration: Duration) {
        if let Ok(mut monitor) = self.inner.lock() {
            monitor.record_index_time(duration);
        }
    }
    
    pub fn time_query<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record_query_time(duration);
        result
    }
    
    pub fn time_index<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record_index_time(duration);
        result
    }
    
    pub fn get_stats(&self) -> Option<PerformanceStats> {
        self.inner.lock().ok().map(|monitor| monitor.get_stats())
    }
    
    pub fn reset(&self) {
        if let Ok(mut monitor) = self.inner.lock() {
            monitor.reset();
        }
    }
}
```

### 4. Add performance monitoring macros
Add these convenience macros at the top of the file:
```rust
/// Macro to time a query operation
#[macro_export]
macro_rules! time_query {
    ($monitor:expr, $operation:expr) => {
        $monitor.time_query(|| $operation)
    };
}

/// Macro to time an indexing operation
#[macro_export]
macro_rules! time_index {
    ($monitor:expr, $operation:expr) => {
        $monitor.time_index(|| $operation)
    };
}

/// Macro to create a scoped timer
#[macro_export]
macro_rules! scoped_timer {
    ($monitor:expr, query) => {
        let _timer = $monitor.start_query_timer();
    };
    ($monitor:expr, index) => {
        let _timer = $monitor.start_index_timer();
    };
}
```

### 5. Add batch recording methods
Add these methods for batch operations:
```rust
impl PerformanceMonitor {
    /// Record multiple query times at once
    pub fn record_query_times(&mut self, durations: Vec<Duration>) {
        for duration in durations {
            self.record_query_time(duration);
        }
    }
    
    /// Record multiple index times at once
    pub fn record_index_times(&mut self, durations: Vec<Duration>) {
        for duration in durations {
            self.record_index_time(duration);
        }
    }
    
    /// Get recent query times for analysis
    pub fn get_recent_query_times(&self, count: usize) -> Vec<Duration> {
        self.query_times.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Get recent index times for analysis
    pub fn get_recent_index_times(&self, count: usize) -> Vec<Duration> {
        self.index_times.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Check if performance is degrading
    pub fn is_performance_degrading(&self, window_size: usize, threshold: f64) -> bool {
        if self.query_times.len() < window_size * 2 {
            return false; // Not enough data
        }
        
        let recent_times = self.get_recent_query_times(window_size);
        let older_times: Vec<_> = self.query_times.iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .cloned()
            .collect();
        
        if older_times.len() < window_size {
            return false;
        }
        
        let recent_avg = recent_times.iter().sum::<Duration>().as_secs_f64() / recent_times.len() as f64;
        let older_avg = older_times.iter().sum::<Duration>().as_secs_f64() / older_times.len() as f64;
        
        if older_avg == 0.0 {
            return false;
        }
        
        let degradation_ratio = recent_avg / older_avg;
        degradation_ratio > (1.0 + threshold)
    }
}
```

### 6. Add comprehensive timing tests
Add these tests to the test module:
```rust
#[test]
fn test_timing_utilities() {
    let mut monitor = PerformanceMonitor::new();
    
    // Test time_query
    let result = monitor.time_query(|| {
        std::thread::sleep(Duration::from_millis(10));
        "query result"
    });
    
    assert_eq!(result, "query result");
    assert_eq!(monitor.get_stats().total_queries, 1);
    assert!(monitor.get_stats().avg_query_time >= Duration::from_millis(10));
}

#[test]
fn test_timer_drop_recording() {
    let mut monitor = PerformanceMonitor::new();
    
    {
        let _timer = monitor.start_query_timer();
        std::thread::sleep(Duration::from_millis(5));
        // Timer will record when dropped here
    }
    
    let stats = monitor.get_stats();
    assert_eq!(stats.total_queries, 1);
    assert!(stats.avg_query_time >= Duration::from_millis(5));
}

#[test]
fn test_shared_monitor() {
    let shared_monitor = SharedPerformanceMonitor::new();
    
    // Test thread safety
    let monitor_clone = shared_monitor.clone();
    let handle = std::thread::spawn(move || {
        monitor_clone.record_query_time(Duration::from_millis(10));
        monitor_clone.record_index_time(Duration::from_millis(20));
    });
    
    shared_monitor.record_query_time(Duration::from_millis(15));
    handle.join().unwrap();
    
    if let Some(stats) = shared_monitor.get_stats() {
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.total_indexes, 1);
    }
}

#[test]
fn test_batch_recording() {
    let mut monitor = PerformanceMonitor::new();
    
    let query_times = vec![
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(30),
    ];
    
    monitor.record_query_times(query_times);
    
    let stats = monitor.get_stats();
    assert_eq!(stats.total_queries, 3);
    assert_eq!(stats.avg_query_time, Duration::from_millis(20));
}

#[test]
fn test_recent_times_retrieval() {
    let mut monitor = PerformanceMonitor::new();
    
    // Add some times
    for i in 1..=5 {
        monitor.record_query_time(Duration::from_millis(i * 10));
    }
    
    let recent_3 = monitor.get_recent_query_times(3);
    assert_eq!(recent_3.len(), 3);
    
    // Should be in reverse order (most recent first)
    assert_eq!(recent_3[0], Duration::from_millis(50));
    assert_eq!(recent_3[1], Duration::from_millis(40));
    assert_eq!(recent_3[2], Duration::from_millis(30));
}

#[test]
fn test_performance_degradation_detection() {
    let mut monitor = PerformanceMonitor::new();
    
    // Add some "good" times
    for _ in 0..10 {
        monitor.record_query_time(Duration::from_millis(10));
    }
    
    // Add some "bad" times
    for _ in 0..10 {
        monitor.record_query_time(Duration::from_millis(30));
    }
    
    // Should detect degradation (30ms vs 10ms = 200% increase > 50% threshold)
    assert!(monitor.is_performance_degrading(10, 0.5));
    
    // Should not detect degradation with higher threshold
    assert!(!monitor.is_performance_degrading(10, 3.0));
}

#[test]
fn test_time_operation_utility() {
    let (result, duration) = PerformanceMonitor::time_operation(|| {
        std::thread::sleep(Duration::from_millis(5));
        "operation result"
    });
    
    assert_eq!(result, "operation result");
    assert!(duration >= Duration::from_millis(5));
}
```

## Success Criteria
- [ ] Timing utility functions work correctly
- [ ] RAII timers record automatically on drop
- [ ] Thread-safe wrapper enables concurrent access
- [ ] Batch recording methods work efficiently
- [ ] Performance degradation detection functions correctly
- [ ] Recent times retrieval works as expected
- [ ] All tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- RAII timers provide automatic timing without manual start/stop calls
- Thread-safe wrapper enables monitoring in multi-threaded scenarios
- Performance degradation detection helps identify performance regressions
- Macros provide convenient timing syntax
- Batch operations improve efficiency for bulk measurements