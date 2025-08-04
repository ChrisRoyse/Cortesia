# Micro-Task 164: Create Performance Dashboard

## Objective
Create a simple text-based performance dashboard for monitoring key metrics.

## Prerequisites
- Task 163 completed (Regression detection implemented)

## Time Estimate
9 minutes

## Instructions
1. Create performance dashboard `performance_dashboard.rs`:
   ```rust
   use std::collections::HashMap;
   use std::time::{Duration, Instant};
   use std::thread;
   use std::io::{self, Write};
   
   struct PerformanceDashboard {
       metrics: HashMap<String, MetricData>,
       start_time: Instant,
   }
   
   struct MetricData {
       current: f64,
       baseline: f64,
       min: f64,
       max: f64,
       avg: f64,
       samples: usize,
   }
   
   impl PerformanceDashboard {
       fn new() -> Self {
           Self {
               metrics: HashMap::new(),
               start_time: Instant::now(),
           }
       }
       
       fn update_metric(&mut self, name: &str, value: f64, baseline: f64) {
           let entry = self.metrics.entry(name.to_string()).or_insert(MetricData {
               current: value,
               baseline,
               min: value,
               max: value,
               avg: value,
               samples: 0,
           });
           
           entry.current = value;
           entry.min = entry.min.min(value);
           entry.max = entry.max.max(value);
           entry.avg = (entry.avg * entry.samples as f64 + value) / (entry.samples + 1) as f64;
           entry.samples += 1;
       }
       
       fn display(&self) {
           // Clear screen (Windows)
           print!("\x1B[2J\x1B[1;1H");
           io::stdout().flush().unwrap();
           
           println!("═══════════════════════════════════════════════════════════════");
           println!("              VECTOR SEARCH PERFORMANCE DASHBOARD");
           println!("═══════════════════════════════════════════════════════════════");
           println!("Uptime: {:.1}s\n", self.start_time.elapsed().as_secs_f64());
           
           println!("{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}", 
                   "Metric", "Current", "Baseline", "Min", "Max", "Avg", "Status");
           println!("────────────────────────────────────────────────────────────────");
           
           for (name, data) in &self.metrics {
               let status = if data.current <= data.baseline * 1.1 {
                   "✅ OK"
               } else if data.current <= data.baseline * 1.5 {
                   "⚠ WARN"
               } else {
                   "❌ CRIT"
               };
               
               println!("{:20} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2} {:>6}", 
                       name, data.current, data.baseline, data.min, data.max, data.avg, status);
           }
           
           println!("\n═══════════════════════════════════════════════════════════════");
           println!("Legend: ✅ Within 110% of baseline  ⚠ 110-150% of baseline  ❌ >150% of baseline");
           println!("Press Ctrl+C to exit");
       }
   }
   
   fn simulate_metrics(dashboard: &mut PerformanceDashboard) {
       use rand::Rng;
       let mut rng = rand::thread_rng();
       
       // Simulate allocation latency
       let base_alloc = 2.0;
       let alloc_variance = rng.gen_range(-0.5..1.5);
       dashboard.update_metric("Allocation (ms)", base_alloc + alloc_variance, 5.0);
       
       // Simulate search latency
       let base_search = 45.0;
       let search_variance = rng.gen_range(-10.0..30.0);
       dashboard.update_metric("Search (ms)", base_search + search_variance, 100.0);
       
       // Simulate memory usage
       let base_memory = 256.0;
       let memory_variance = rng.gen_range(-50.0..200.0);
       dashboard.update_metric("Memory (MB)", base_memory + memory_variance, 1024.0);
       
       // Simulate throughput
       let base_throughput = 850.0;
       let throughput_variance = rng.gen_range(-100.0..150.0);
       dashboard.update_metric("Throughput (qps)", base_throughput + throughput_variance, 1000.0);
       
       // Simulate CPU usage
       let base_cpu = 35.0;
       let cpu_variance = rng.gen_range(-10.0..25.0);
       dashboard.update_metric("CPU (%)", base_cpu + cpu_variance, 80.0);
   }
   
   fn main() {
       println!("Starting performance dashboard...");
       
       let mut dashboard = PerformanceDashboard::new();
       
       loop {
           simulate_metrics(&mut dashboard);
           dashboard.display();
           thread::sleep(Duration::from_secs(2));
       }
   }
   ```
2. Add dependency to Cargo.toml: `rand = "0.8"`
3. Run: `cargo run --release --bin performance_dashboard`
4. Commit: `git add src/bin/performance_dashboard.rs && git commit -m "Create real-time performance dashboard"`

## Success Criteria
- [ ] Performance dashboard created
- [ ] Real-time metrics display working
- [ ] Status indicators functional
- [ ] Dashboard UI clear and informative
- [ ] Dashboard committed

## Next Task
task_165_benchmark_load_scenarios.md