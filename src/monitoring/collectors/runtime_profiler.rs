/*!
LLMKG Runtime Execution Profiler
Real-time function call tracing and execution monitoring
*/

use crate::monitoring::metrics::MetricRegistry;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub function_name: String,
    pub module_path: String,
    pub thread_id: u64,
    pub start_time_millis: u64,
    pub end_time_millis: Option<u64>,
    pub duration: Option<Duration>,
    pub parameters: Vec<ParameterValue>,
    pub return_value: Option<String>,
    pub memory_usage: u64,
    pub cpu_usage: f32,
    pub call_stack: Vec<String>,
    pub children: Vec<ExecutionTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterValue {
    pub name: String,
    pub value_type: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub active_functions: HashMap<String, u64>,
    pub function_call_count: HashMap<String, u64>,
    pub function_execution_times: HashMap<String, ExecutionStats>,
    pub memory_allocations: HashMap<String, u64>,
    pub thread_activity: HashMap<u64, ThreadInfo>,
    pub hot_paths: Vec<HotPath>,
    pub performance_bottlenecks: Vec<PerformanceBottleneck>,
    pub execution_timeline: VecDeque<ExecutionEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub total_calls: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub p50_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    pub thread_id: u64,
    pub thread_name: String,
    pub active_functions: Vec<String>,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub start_time_millis: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    pub path: Vec<String>,
    pub call_count: u64,
    pub total_duration: Duration,
    pub frequency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub function_name: String,
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub description: String,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    SlowExecution,
    HighMemoryUsage,
    FrequentCalls,
    LongBlockingOperation,
    DeepRecursion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    pub timestamp_millis: u64,
    pub event_type: ExecutionEventType,
    pub function_name: String,
    pub duration: Option<Duration>,
    pub thread_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEventType {
    FunctionStart,
    FunctionEnd,
    MemoryAllocation,
    MemoryDeallocation,
    ThreadSpawn,
    ThreadExit,
}

pub struct RuntimeProfiler {
    metrics: Arc<RwLock<RuntimeMetrics>>,
    active_traces: Arc<Mutex<HashMap<String, ExecutionTrace>>>,
    event_sender: broadcast::Sender<ExecutionEvent>,
    max_timeline_events: usize,
    profiling_enabled: Arc<RwLock<bool>>,
}

impl Default for RuntimeProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeProfiler {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            metrics: Arc::new(RwLock::new(RuntimeMetrics {
                active_functions: HashMap::new(),
                function_call_count: HashMap::new(),
                function_execution_times: HashMap::new(),
                memory_allocations: HashMap::new(),
                thread_activity: HashMap::new(),
                hot_paths: Vec::new(),
                performance_bottlenecks: Vec::new(),
                execution_timeline: VecDeque::new(),
            })),
            active_traces: Arc::new(Mutex::new(HashMap::new())),
            event_sender,
            max_timeline_events: 10000,
            profiling_enabled: Arc::new(RwLock::new(true)),
        }
    }

    pub fn start_function_trace(&self, function_name: String, module_path: String, parameters: Vec<ParameterValue>) -> String {
        if !*self.profiling_enabled.read().unwrap() {
            return String::new();
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let trace_id = format!("{function_name}_{now}");
        let thread_id = self.get_current_thread_id();
        
        let trace = ExecutionTrace {
            function_name: function_name.clone(),
            module_path,
            thread_id,
            start_time_millis: now,
            end_time_millis: None,
            duration: None,
            parameters,
            return_value: None,
            memory_usage: self.get_current_memory_usage(),
            cpu_usage: 0.0,
            call_stack: self.get_current_call_stack(),
            children: Vec::new(),
        };

        // Store active trace
        {
            let mut active_traces = self.active_traces.lock().unwrap();
            active_traces.insert(trace_id.clone(), trace);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            *metrics.active_functions.entry(function_name.clone()).or_insert(0) += 1;
            *metrics.function_call_count.entry(function_name.clone()).or_insert(0) += 1;
            
            // Add to timeline
            let event = ExecutionEvent {
                timestamp_millis: now,
                event_type: ExecutionEventType::FunctionStart,
                function_name: function_name.clone(),
                duration: None,
                thread_id,
            };
            
            metrics.execution_timeline.push_back(event.clone());
            if metrics.execution_timeline.len() > self.max_timeline_events {
                metrics.execution_timeline.pop_front();
            }
        }

        // Send event
        let _ = self.event_sender.send(ExecutionEvent {
            timestamp_millis: now,
            event_type: ExecutionEventType::FunctionStart,
            function_name,
            duration: None,
            thread_id,
        });

        trace_id
    }

    pub fn end_function_trace(&self, trace_id: String, return_value: Option<String>) {
        if !*self.profiling_enabled.read().unwrap() {
            return;
        }

        let mut completed_trace = None;
        
        // Remove from active traces and calculate duration
        {
            let mut active_traces = self.active_traces.lock().unwrap();
            if let Some(mut trace) = active_traces.remove(&trace_id) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                trace.end_time_millis = Some(now);
                trace.duration = Some(Duration::from_millis(now - trace.start_time_millis));
                trace.return_value = return_value;
                trace.memory_usage = self.get_current_memory_usage();
                completed_trace = Some(trace);
            }
        }

        if let Some(trace) = completed_trace {
            let duration = trace.duration.unwrap();
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().unwrap();
                
                // Update active function count
                if let Some(count) = metrics.active_functions.get_mut(&trace.function_name) {
                    *count = count.saturating_sub(1);
                }
                
                // Update execution statistics
                let stats = metrics.function_execution_times
                    .entry(trace.function_name.clone())
                    .or_insert_with(|| ExecutionStats {
                        total_calls: 0,
                        total_duration: Duration::new(0, 0),
                        avg_duration: Duration::new(0, 0),
                        min_duration: duration,
                        max_duration: duration,
                        p50_duration: duration,
                        p95_duration: duration,
                        p99_duration: duration,
                    });
                
                stats.total_calls += 1;
                stats.total_duration += duration;
                stats.avg_duration = stats.total_duration / stats.total_calls as u32;
                stats.min_duration = stats.min_duration.min(duration);
                stats.max_duration = stats.max_duration.max(duration);
                
                // Add to timeline
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let event = ExecutionEvent {
                    timestamp_millis: now,
                    event_type: ExecutionEventType::FunctionEnd,
                    function_name: trace.function_name.clone(),
                    duration: Some(duration),
                    thread_id: trace.thread_id,
                };
                
                metrics.execution_timeline.push_back(event.clone());
                if metrics.execution_timeline.len() > self.max_timeline_events {
                    metrics.execution_timeline.pop_front();
                }
            }

            // Send event
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            let _ = self.event_sender.send(ExecutionEvent {
                timestamp_millis: now,
                event_type: ExecutionEventType::FunctionEnd,
                function_name: trace.function_name,
                duration: Some(duration),
                thread_id: trace.thread_id,
            });

            // Analyze for performance bottlenecks
            self.analyze_performance_bottlenecks();
        }
    }

    pub fn record_memory_allocation(&self, function_name: String, size: u64) {
        if !*self.profiling_enabled.read().unwrap() {
            return;
        }

        let mut metrics = self.metrics.write().unwrap();
        *metrics.memory_allocations.entry(function_name.clone()).or_insert(0) += size;
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let event = ExecutionEvent {
            timestamp_millis: now,
            event_type: ExecutionEventType::MemoryAllocation,
            function_name,
            duration: None,
            thread_id: self.get_current_thread_id(),
        };
        
        metrics.execution_timeline.push_back(event.clone());
        if metrics.execution_timeline.len() > self.max_timeline_events {
            metrics.execution_timeline.pop_front();
        }
        
        let _ = self.event_sender.send(event);
    }

    pub fn get_metrics(&self) -> RuntimeMetrics {
        self.metrics.read().unwrap().clone()
    }

    pub fn get_active_traces(&self) -> Vec<ExecutionTrace> {
        self.active_traces.lock().unwrap().values().cloned().collect()
    }

    pub fn subscribe_to_events(&self) -> broadcast::Receiver<ExecutionEvent> {
        self.event_sender.subscribe()
    }

    pub fn enable_profiling(&self, enabled: bool) {
        *self.profiling_enabled.write().unwrap() = enabled;
    }

    pub fn clear_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.active_functions.clear();
        metrics.function_call_count.clear();
        metrics.function_execution_times.clear();
        metrics.memory_allocations.clear();
        metrics.thread_activity.clear();
        metrics.hot_paths.clear();
        metrics.performance_bottlenecks.clear();
        metrics.execution_timeline.clear();
    }

    fn get_current_thread_id(&self) -> u64 {
        // Simplified thread ID - use a hash of the thread ID for stable conversion
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }

    fn get_current_memory_usage(&self) -> u64 {
        // Simplified memory usage - in real implementation would use proper memory tracking
        0
    }

    fn get_current_call_stack(&self) -> Vec<String> {
        // Simplified call stack - in real implementation would capture actual stack trace
        Vec::new()
    }

    fn analyze_performance_bottlenecks(&self) {
        let metrics = self.metrics.read().unwrap();
        let mut bottlenecks = Vec::new();
        
        // Analyze slow functions
        for (function_name, stats) in &metrics.function_execution_times {
            if stats.avg_duration > Duration::from_millis(100) {
                bottlenecks.push(PerformanceBottleneck {
                    function_name: function_name.clone(),
                    bottleneck_type: BottleneckType::SlowExecution,
                    severity: stats.avg_duration.as_millis() as f32 / 1000.0,
                    description: format!("Function {} has high average execution time: {:?}", function_name, stats.avg_duration),
                    suggested_fix: "Consider optimizing algorithm or caching results".to_string(),
                });
            }
            
            if stats.total_calls > 1000 {
                bottlenecks.push(PerformanceBottleneck {
                    function_name: function_name.clone(),
                    bottleneck_type: BottleneckType::FrequentCalls,
                    severity: (stats.total_calls as f32 / 1000.0).min(10.0),
                    description: format!("Function {} is called very frequently: {} times", function_name, stats.total_calls),
                    suggested_fix: "Consider caching or reducing call frequency".to_string(),
                });
            }
        }
        
        // Analyze memory usage
        for (function_name, memory_usage) in &metrics.memory_allocations {
            if *memory_usage > 1024 * 1024 * 10 { // 10MB
                bottlenecks.push(PerformanceBottleneck {
                    function_name: function_name.clone(),
                    bottleneck_type: BottleneckType::HighMemoryUsage,
                    severity: (*memory_usage as f32 / (1024.0 * 1024.0)).min(10.0),
                    description: format!("Function {function_name} uses high memory: {memory_usage} bytes"),
                    suggested_fix: "Consider memory optimization or streaming processing".to_string(),
                });
            }
        }
        
        drop(metrics);
        
        // Update bottlenecks
        if !bottlenecks.is_empty() {
            let mut metrics = self.metrics.write().unwrap();
            metrics.performance_bottlenecks = bottlenecks;
        }
    }

    pub fn analyze_hot_paths(&self) {
        // TODO: Implement hot path analysis based on execution timeline
        let mut hot_paths = Vec::new();
        
        // Analyze common execution patterns
        let metrics = self.metrics.read().unwrap();
        let timeline = &metrics.execution_timeline;
        
        // Simple hot path detection - consecutive function calls
        let mut current_path = Vec::new();
        let mut path_counts: HashMap<Vec<String>, u64> = HashMap::new();
        
        for event in timeline {
            match event.event_type {
                ExecutionEventType::FunctionStart => {
                    current_path.push(event.function_name.clone());
                    if current_path.len() >= 3 {
                        *path_counts.entry(current_path.clone()).or_insert(0) += 1;
                    }
                }
                ExecutionEventType::FunctionEnd => {
                    if !current_path.is_empty() {
                        current_path.pop();
                    }
                }
                _ => {}
            }
        }
        
        let timeline_len = timeline.len();
        drop(metrics);
        
        // Convert to hot paths
        for (path, count) in path_counts {
            if count > 10 {
                hot_paths.push(HotPath {
                    path: path.clone(),
                    call_count: count,
                    total_duration: Duration::from_millis(count * 10), // Simplified
                    frequency: count as f32 / timeline_len as f32,
                });
            }
        }
        
        // Update hot paths
        if !hot_paths.is_empty() {
            let mut metrics = self.metrics.write().unwrap();
            metrics.hot_paths = hot_paths;
        }
    }
}

impl super::MetricsCollector for RuntimeProfiler {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        let metrics = self.get_metrics();
        
        // Register runtime metrics
        let active_functions_gauge = registry.gauge("runtime_active_functions", HashMap::new());
        active_functions_gauge.set(metrics.active_functions.values().sum::<u64>() as f64);
        
        let total_function_calls_gauge = registry.gauge("runtime_total_function_calls", HashMap::new());
        total_function_calls_gauge.set(metrics.function_call_count.values().sum::<u64>() as f64);
        
        let avg_execution_time_gauge = registry.gauge("runtime_avg_execution_time_ms", HashMap::new());
        let avg_time = if !metrics.function_execution_times.is_empty() {
            metrics.function_execution_times.values()
                .map(|s| s.avg_duration.as_millis() as f64)
                .sum::<f64>() / metrics.function_execution_times.len() as f64
        } else {
            0.0
        };
        avg_execution_time_gauge.set(avg_time);
        
        let memory_allocations_gauge = registry.gauge("runtime_memory_allocations_bytes", HashMap::new());
        memory_allocations_gauge.set(metrics.memory_allocations.values().sum::<u64>() as f64);
        
        let bottlenecks_gauge = registry.gauge("runtime_performance_bottlenecks", HashMap::new());
        bottlenecks_gauge.set(metrics.performance_bottlenecks.len() as f64);
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "runtime_profiler"
    }
    
    fn is_enabled(&self, config: &super::MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"runtime_profiler".to_string())
    }
}

// Macro for easy function tracing
#[macro_export]
macro_rules! trace_function {
    // Pattern for no parameters
    ($profiler:expr, $func_name:expr) => {{
        let params = vec![];
        
        let trace_id = $profiler.start_function_trace(
            $func_name.to_string(),
            module_path!().to_string(),
            params
        );
        
        struct TraceGuard<'a> {
            profiler: &'a $crate::monitoring::collectors::runtime_profiler::RuntimeProfiler,
            trace_id: String,
        }
        
        impl<'a> Drop for TraceGuard<'a> {
            fn drop(&mut self) {
                self.profiler.end_function_trace(self.trace_id.clone(), None);
            }
        }
        
        TraceGuard {
            profiler: $profiler,
            trace_id,
        }
    }};
    // Pattern for parameters
    ($profiler:expr, $func_name:expr, $($param:expr),+) => {{
        let params = vec![
            $($crate::monitoring::collectors::runtime_profiler::ParameterValue {
                name: stringify!($param).to_string(),
                value_type: std::any::type_name_of_val(&$param).to_string(),
                value: format!("{:?}", $param),
            }),*
        ];
        
        let trace_id = $profiler.start_function_trace(
            $func_name.to_string(),
            module_path!().to_string(),
            params
        );
        
        struct TraceGuard<'a> {
            profiler: &'a $crate::monitoring::collectors::runtime_profiler::RuntimeProfiler,
            trace_id: String,
        }
        
        impl<'a> Drop for TraceGuard<'a> {
            fn drop(&mut self) {
                self.profiler.end_function_trace(self.trace_id.clone(), None);
            }
        }
        
        TraceGuard {
            profiler: $profiler,
            trace_id,
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_runtime_profiler() {
        let profiler = RuntimeProfiler::new();
        
        // Start function trace
        let trace_id = profiler.start_function_trace(
            "test_function".to_string(),
            "test_module".to_string(),
            vec![]
        );
        
        // Simulate function execution
        thread::sleep(Duration::from_millis(10));
        
        // End function trace
        profiler.end_function_trace(trace_id, Some("test_return".to_string()));
        
        // Check metrics
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.function_call_count.get("test_function"), Some(&1));
        assert!(metrics.function_execution_times.contains_key("test_function"));
    }

    #[test]
    fn test_memory_allocation_tracking() {
        let profiler = RuntimeProfiler::new();
        
        profiler.record_memory_allocation("test_function".to_string(), 1024);
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.memory_allocations.get("test_function"), Some(&1024));
    }
}