# Task 19e: Implement System Monitoring

**Estimated Time**: 5 minutes  
**Dependencies**: 19d  
**Stage**: Performance Monitoring  

## Objective
Implement system resource monitoring using the sysinfo crate.

## Implementation Steps

1. Create `src/monitoring/system.rs`:
```rust
use sysinfo::{System, SystemExt, CpuExt, ProcessExt};
use std::time::Duration;
use tokio::time::interval;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub memory_used_mb: f64,
    pub memory_total_mb: f64,
    pub memory_usage_percent: f64,
    pub process_memory_mb: f64,
    pub process_cpu_usage: f32,
    pub uptime_seconds: u64,
    pub load_average: Option<(f64, f64, f64)>, // 1min, 5min, 15min
}

pub struct SystemMonitor {
    system: Arc<RwLock<System>>,
    current_metrics: Arc<RwLock<SystemMetrics>>,
    collection_interval: Duration,
}

impl SystemMonitor {
    pub fn new(collection_interval_ms: u64) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let initial_metrics = SystemMetrics {
            cpu_usage_percent: 0.0,
            memory_used_mb: 0.0,
            memory_total_mb: 0.0,
            memory_usage_percent: 0.0,
            process_memory_mb: 0.0,
            process_cpu_usage: 0.0,
            uptime_seconds: 0,
            load_average: None,
        };
        
        Self {
            system: Arc::new(RwLock::new(system)),
            current_metrics: Arc::new(RwLock::new(initial_metrics)),
            collection_interval: Duration::from_millis(collection_interval_ms),
        }
    }
    
    pub async fn start_monitoring(&self) {
        let system = self.system.clone();
        let metrics = self.current_metrics.clone();
        let interval_duration = self.collection_interval;
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let new_metrics = {
                    let mut sys = system.write().await;
                    
                    // Refresh system information
                    sys.refresh_cpu();
                    sys.refresh_memory();
                    sys.refresh_processes();
                    
                    let current_pid = sysinfo::get_current_pid().ok();
                    let process_info = current_pid
                        .and_then(|pid| sys.process(pid))
                        .map(|process| {
                            (
                                process.memory() as f64 / 1024.0 / 1024.0, // Convert to MB
                                process.cpu_usage()
                            )
                        })
                        .unwrap_or((0.0, 0.0));
                    
                    // Calculate CPU usage (average across all cores)
                    let cpu_usage = sys.cpus().iter()
                        .map(|cpu| cpu.cpu_usage())
                        .sum::<f32>() / sys.cpus().len() as f32;
                    
                    // Memory information
                    let memory_total_mb = sys.total_memory() as f64 / 1024.0 / 1024.0;
                    let memory_used_mb = sys.used_memory() as f64 / 1024.0 / 1024.0;
                    let memory_usage_percent = (memory_used_mb / memory_total_mb) * 100.0;
                    
                    // Load average (Unix-like systems only)
                    let load_average = sys.load_average();
                    let load_avg = if load_average.one > 0.0 {
                        Some((load_average.one, load_average.five, load_average.fifteen))
                    } else {
                        None
                    };
                    
                    SystemMetrics {
                        cpu_usage_percent: cpu_usage,
                        memory_used_mb,
                        memory_total_mb,
                        memory_usage_percent,
                        process_memory_mb: process_info.0,
                        process_cpu_usage: process_info.1,
                        uptime_seconds: sys.uptime(),
                        load_average: load_avg,
                    }
                };
                
                // Update current metrics
                {
                    let mut current = metrics.write().await;
                    *current = new_metrics;
                }
            }
        });
    }
    
    pub async fn get_current_metrics(&self) -> SystemMetrics {
        self.current_metrics.read().await.clone()
    }
    
    pub async fn get_memory_usage_percent(&self) -> f64 {
        self.current_metrics.read().await.memory_usage_percent
    }
    
    pub async fn get_cpu_usage_percent(&self) -> f32 {
        self.current_metrics.read().await.cpu_usage_percent
    }
    
    pub async fn get_process_memory_mb(&self) -> f64 {
        self.current_metrics.read().await.process_memory_mb
    }
    
    pub async fn is_system_healthy(&self) -> bool {
        let metrics = self.current_metrics.read().await;
        
        // Define health criteria
        metrics.memory_usage_percent < 85.0 && 
        metrics.cpu_usage_percent < 80.0 &&
        metrics.process_memory_mb < 2048.0 // Less than 2GB for this process
    }
    
    pub async fn get_system_summary(&self) -> String {
        let metrics = self.current_metrics.read().await;
        
        format!(
            "System: CPU {:.1}%, Memory {:.1}% ({:.0}MB/{:.0}MB), Process: {:.0}MB, Uptime: {}s",
            metrics.cpu_usage_percent,
            metrics.memory_usage_percent,
            metrics.memory_used_mb,
            metrics.memory_total_mb,
            metrics.process_memory_mb,
            metrics.uptime_seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_system_monitor_creation() {
        let monitor = SystemMonitor::new(1000);
        let metrics = monitor.get_current_metrics().await;
        
        // Should have valid initial metrics
        assert!(metrics.memory_total_mb > 0.0);
    }
    
    #[tokio::test]
    async fn test_system_monitoring() {
        let monitor = SystemMonitor::new(100); // 100ms interval
        monitor.start_monitoring().await;
        
        // Wait for a few collection cycles
        sleep(Duration::from_millis(300)).await;
        
        let metrics = monitor.get_current_metrics().await;
        
        // Should have updated metrics
        assert!(metrics.memory_total_mb > 0.0);
        assert!(metrics.cpu_usage_percent >= 0.0);
        
        let summary = monitor.get_system_summary().await;
        assert!(summary.contains("System:"));
        assert!(summary.contains("CPU"));
        assert!(summary.contains("Memory"));
    }
}
```

## Acceptance Criteria
- [ ] System monitoring implemented
- [ ] CPU and memory metrics collected
- [ ] Process-specific metrics tracked
- [ ] Health status calculation

## Success Metrics
- System metrics update regularly
- Accurate resource usage reporting
- Low overhead monitoring

## Next Task
19f_integrate_monitoring_with_core.md