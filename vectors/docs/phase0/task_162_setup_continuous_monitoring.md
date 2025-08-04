# Micro-Task 162: Setup Continuous Monitoring

## Objective
Implement continuous performance monitoring infrastructure for ongoing performance tracking.

## Prerequisites
- Task 161 completed (Baseline report generated)

## Time Estimate
9 minutes

## Instructions
1. Create monitoring configuration `monitoring_config.toml`:
   ```toml
   [monitoring]
   enabled = true
   interval_seconds = 60
   metrics_retention_hours = 24
   alert_threshold_percent = 120  # 20% degradation triggers alert
   
   [metrics]
   allocation_latency = true
   memory_usage = true
   search_performance = true
   throughput = true
   error_rate = true
   
   [alerting]
   enabled = true
   console_output = true
   log_file = "performance_alerts.log"
   
   [baseline_targets]
   allocation_ms = 5.0
   memory_mb = 1024
   search_ms = 100
   throughput_qps = 1000
   ```
2. Create monitoring service `monitoring_service.rs`:
   ```rust
   use std::collections::HashMap;
   use std::sync::{Arc, Mutex};
   use std::time::{Duration, Instant};
   use std::thread;
   
   pub struct PerformanceMonitor {
       metrics: Arc<Mutex<HashMap<String, f64>>>,
       baselines: HashMap<String, f64>,
       running: Arc<Mutex<bool>>,
   }
   
   impl PerformanceMonitor {
       pub fn new() -> Self {
           let mut baselines = HashMap::new();
           baselines.insert("allocation_ms".to_string(), 5.0);
           baselines.insert("memory_mb".to_string(), 1024.0);
           baselines.insert("search_ms".to_string(), 100.0);
           
           Self {
               metrics: Arc::new(Mutex::new(HashMap::new())),
               baselines,
               running: Arc::new(Mutex::new(false)),
           }
       }
       
       pub fn start(&self) {
           *self.running.lock().unwrap() = true;
           
           let metrics = Arc::clone(&self.metrics);
           let running = Arc::clone(&self.running);
           
           thread::spawn(move || {
               while *running.lock().unwrap() {
                   Self::collect_metrics(&metrics);
                   thread::sleep(Duration::from_secs(60));
               }
           });
       }
       
       pub fn stop(&self) {
           *self.running.lock().unwrap() = false;
       }
       
       pub fn record_metric(&self, name: &str, value: f64) {
           let mut metrics = self.metrics.lock().unwrap();
           metrics.insert(name.to_string(), value);
           
           if let Some(&baseline) = self.baselines.get(name) {
               if value > baseline * 1.2 {
                   println!("⚠ ALERT: {} = {:.3} exceeds baseline {:.3} by {:.1}%", 
                           name, value, baseline, ((value / baseline) - 1.0) * 100.0);
               }
           }
       }
       
       pub fn get_metrics(&self) -> HashMap<String, f64> {
           self.metrics.lock().unwrap().clone()
       }
       
       fn collect_metrics(metrics: &Arc<Mutex<HashMap<String, f64>>>) {
           let cpu_usage = Self::get_cpu_usage();
           let memory_usage = Self::get_memory_usage();
           
           let mut m = metrics.lock().unwrap();
           m.insert("cpu_percent".to_string(), cpu_usage);
           m.insert("memory_mb".to_string(), memory_usage);
       }
       
       fn get_cpu_usage() -> f64 {
           // Simplified CPU usage - would use system APIs in production
           50.0
       }
       
       fn get_memory_usage() -> f64 {
           // Simplified memory usage - would use system APIs in production
           256.0
       }
   }
   
   fn main() {
       println!("Starting performance monitoring...");
       
       let monitor = PerformanceMonitor::new();
       monitor.start();
       
       // Simulate some metrics
       for i in 0..10 {
           monitor.record_metric("allocation_ms", 2.0 + (i as f64 * 0.5));
           monitor.record_metric("search_ms", 50.0 + (i as f64 * 10.0));
           thread::sleep(Duration::from_secs(5));
       }
       
       monitor.stop();
       
       let metrics = monitor.get_metrics();
       println!("Final metrics: {:?}", metrics);
   }
   ```
3. Run: `cargo run --release --bin monitoring_service`
4. Commit: `git add monitoring_config.toml src/bin/monitoring_service.rs && git commit -m "Setup continuous performance monitoring infrastructure"`

## Success Criteria
- [ ] Monitoring configuration created
- [ ] Monitoring service implemented
- [ ] Alert system functional
- [ ] Baseline comparison working
- [ ] Monitoring infrastructure committed

## Next Task
task_163_implement_regression_detection.md