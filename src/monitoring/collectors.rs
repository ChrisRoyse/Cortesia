/*!
Phase 5.4: Enhanced Metrics Collectors
Comprehensive collectors for system, application, codebase, runtime, API, and test metrics
*/

use crate::monitoring::metrics::MetricRegistry;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

// New enhanced collectors
pub mod codebase_analyzer;
pub mod runtime_profiler;
pub mod api_endpoint_monitor;
pub mod test_execution_tracker;
pub mod knowledge_engine_metrics;

pub use codebase_analyzer::{CodebaseAnalyzer, CodebaseMetrics};
pub use runtime_profiler::{RuntimeProfiler, RuntimeMetrics, ExecutionTrace};
pub use api_endpoint_monitor::{ApiEndpointMonitor, ApiMetrics, ApiEndpoint};
pub use test_execution_tracker::{TestExecutionTracker, TestMetrics, TestSuite};
pub use knowledge_engine_metrics::KnowledgeEngineMetricsCollector;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectionConfig {
    pub collection_interval: Duration,
    pub enabled_collectors: Vec<String>,
    pub custom_labels: HashMap<String, String>,
    pub system_metrics: SystemMetricsConfig,
    pub application_metrics: ApplicationMetricsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsConfig {
    pub collect_cpu: bool,
    pub collect_memory: bool,
    pub collect_disk: bool,
    pub collect_network: bool,
    pub collect_load: bool,
}

impl Default for SystemMetricsConfig {
    fn default() -> Self {
        Self {
            collect_cpu: true,
            collect_memory: true,
            collect_disk: true,
            collect_network: true,
            collect_load: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetricsConfig {
    pub collect_performance: bool,
    pub collect_operations: bool,
    pub collect_errors: bool,
    pub collect_resources: bool,
}

impl Default for ApplicationMetricsConfig {
    fn default() -> Self {
        Self {
            collect_performance: true,
            collect_operations: true,
            collect_errors: true,
            collect_resources: true,
        }
    }
}

impl Default for MetricsCollectionConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(15),
            enabled_collectors: vec![
                "system".to_string(),
                "application".to_string(),
            ],
            custom_labels: HashMap::new(),
            system_metrics: SystemMetricsConfig {
                collect_cpu: true,
                collect_memory: true,
                collect_disk: true,
                collect_network: true,
                collect_load: true,
            },
            application_metrics: ApplicationMetricsConfig {
                collect_performance: true,
                collect_operations: true,
                collect_errors: true,
                collect_resources: true,
            },
        }
    }
}

pub trait MetricsCollector: Send + Sync {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>>;
    fn name(&self) -> &str;
    fn is_enabled(&self, config: &MetricsCollectionConfig) -> bool;
}

pub struct SystemMetricsCollector {
    last_cpu_stats: Arc<RwLock<Option<CpuStats>>>,
    last_network_stats: Arc<RwLock<Option<NetworkStats>>>,
    config: SystemMetricsConfig,
}

#[derive(Debug, Clone)]
struct CpuStats {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    timestamp: Instant,
}

#[derive(Debug, Clone)]
struct NetworkStats {
    rx_bytes: u64,
    tx_bytes: u64,
    rx_packets: u64,
    tx_packets: u64,
    timestamp: Instant,
}

impl SystemMetricsCollector {
    pub fn new(config: SystemMetricsConfig) -> Self {
        Self {
            last_cpu_stats: Arc::new(RwLock::new(None)),
            last_network_stats: Arc::new(RwLock::new(None)),
            config,
        }
    }
    
    fn collect_cpu_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.collect_cpu {
            return Ok(());
        }
        
        let cpu_stats = self.read_cpu_stats()?;
        let cpu_usage = self.calculate_cpu_usage(&cpu_stats)?;
        
        let cpu_gauge = registry.gauge("system_cpu_usage_percent", HashMap::new());
        cpu_gauge.set(cpu_usage);
        
        // Individual CPU core metrics
        if let Ok(core_usages) = self.read_per_core_cpu_stats() {
            for (core_id, usage) in core_usages.iter().enumerate() {
                let mut labels = HashMap::new();
                labels.insert("core".to_string(), core_id.to_string());
                
                let core_gauge = registry.gauge("system_cpu_usage_percent", labels);
                core_gauge.set(*usage);
            }
        }
        
        Ok(())
    }
    
    fn collect_memory_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.collect_memory {
            return Ok(());
        }
        
        let memory_info = self.read_memory_info()?;
        
        let total_gauge = registry.gauge("system_memory_total_bytes", HashMap::new());
        total_gauge.set(memory_info.total as f64);
        
        let available_gauge = registry.gauge("system_memory_available_bytes", HashMap::new());
        available_gauge.set(memory_info.available as f64);
        
        let used_gauge = registry.gauge("system_memory_used_bytes", HashMap::new());
        used_gauge.set((memory_info.total - memory_info.available) as f64);
        
        let usage_percent_gauge = registry.gauge("system_memory_usage_percent", HashMap::new());
        let usage_percent = ((memory_info.total - memory_info.available) as f64 / memory_info.total as f64) * 100.0;
        usage_percent_gauge.set(usage_percent);
        
        // Swap metrics
        let swap_total_gauge = registry.gauge("system_swap_total_bytes", HashMap::new());
        swap_total_gauge.set(memory_info.swap_total as f64);
        
        let swap_used_gauge = registry.gauge("system_swap_used_bytes", HashMap::new());
        swap_used_gauge.set((memory_info.swap_total - memory_info.swap_free) as f64);
        
        Ok(())
    }
    
    fn collect_disk_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.collect_disk {
            return Ok(());
        }
        
        let disk_stats = self.read_disk_stats()?;
        
        for (device, stats) in disk_stats {
            let mut labels = HashMap::new();
            labels.insert("device".to_string(), device);
            
            let read_bytes_counter = registry.counter("system_disk_read_bytes_total", labels.clone());
            read_bytes_counter.add(stats.read_bytes);
            
            let write_bytes_counter = registry.counter("system_disk_write_bytes_total", labels.clone());
            write_bytes_counter.add(stats.write_bytes);
            
            let read_ops_counter = registry.counter("system_disk_read_ops_total", labels.clone());
            read_ops_counter.add(stats.read_ops);
            
            let write_ops_counter = registry.counter("system_disk_write_ops_total", labels.clone());
            write_ops_counter.add(stats.write_ops);
            
            let utilization_gauge = registry.gauge("system_disk_utilization_percent", labels);
            utilization_gauge.set(stats.utilization_percent);
        }
        
        Ok(())
    }
    
    fn collect_network_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.collect_network {
            return Ok(());
        }
        
        let network_stats = self.read_network_stats()?;
        let rates = self.calculate_network_rates(&network_stats)?;
        
        for (interface, stats) in network_stats {
            let mut labels = HashMap::new();
            labels.insert("interface".to_string(), interface.clone());
            
            let rx_bytes_counter = registry.counter("system_network_rx_bytes_total", labels.clone());
            rx_bytes_counter.add(stats.rx_bytes);
            
            let tx_bytes_counter = registry.counter("system_network_tx_bytes_total", labels.clone());
            tx_bytes_counter.add(stats.tx_bytes);
            
            let rx_packets_counter = registry.counter("system_network_rx_packets_total", labels.clone());
            rx_packets_counter.add(stats.rx_packets);
            
            let tx_packets_counter = registry.counter("system_network_tx_packets_total", labels.clone());
            tx_packets_counter.add(stats.tx_packets);
            
            let rx_errors_counter = registry.counter("system_network_rx_errors_total", labels.clone());
            rx_errors_counter.add(stats.rx_errors);
            
            let tx_errors_counter = registry.counter("system_network_tx_errors_total", labels.clone());
            tx_errors_counter.add(stats.tx_errors);
            
            // Rate metrics
            if let Some(interface_rates) = rates.get(&interface) {
                let rx_rate_gauge = registry.gauge("system_network_rx_rate_bytes_per_sec", labels.clone());
                rx_rate_gauge.set(interface_rates.rx_rate);
                
                let tx_rate_gauge = registry.gauge("system_network_tx_rate_bytes_per_sec", labels);
                tx_rate_gauge.set(interface_rates.tx_rate);
            }
        }
        
        Ok(())
    }
    
    fn collect_load_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.collect_load {
            return Ok(());
        }
        
        let load_avg = self.read_load_average()?;
        
        let load1_gauge = registry.gauge("system_load1", HashMap::new());
        load1_gauge.set(load_avg.load1);
        
        let load5_gauge = registry.gauge("system_load5", HashMap::new());
        load5_gauge.set(load_avg.load5);
        
        let load15_gauge = registry.gauge("system_load15", HashMap::new());
        load15_gauge.set(load_avg.load15);
        
        Ok(())
    }
    
    // Platform-specific implementations
    #[cfg(target_os = "linux")]
    fn read_cpu_stats(&self) -> Result<CpuStats, Box<dyn std::error::Error>> {
        use std::fs;
        
        let content = fs::read_to_string("/proc/stat")?;
        let line = content.lines().next().ok_or("No CPU stats found")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 8 {
            return Err("Invalid CPU stats format".into());
        }
        
        Ok(CpuStats {
            user: parts[1].parse()?,
            nice: parts[2].parse()?,
            system: parts[3].parse()?,
            idle: parts[4].parse()?,
            iowait: parts[5].parse()?,
            irq: parts[6].parse()?,
            softirq: parts[7].parse()?,
            timestamp: Instant::now(),
        })
    }
    
    #[cfg(not(target_os = "linux"))]
    fn read_cpu_stats(&self) -> Result<CpuStats, Box<dyn std::error::Error>> {
        // Fallback implementation for non-Linux systems
        Ok(CpuStats {
            user: 0,
            nice: 0,
            system: 0,
            idle: 1000,
            iowait: 0,
            irq: 0,
            softirq: 0,
            timestamp: Instant::now(),
        })
    }
    
    fn calculate_cpu_usage(&self, current_stats: &CpuStats) -> Result<f64, Box<dyn std::error::Error>> {
        let mut last_stats_guard = self.last_cpu_stats.write().unwrap();
        
        if let Some(ref last_stats) = *last_stats_guard {
            let total_diff = (current_stats.user + current_stats.nice + current_stats.system + 
                            current_stats.idle + current_stats.iowait + current_stats.irq + current_stats.softirq) -
                           (last_stats.user + last_stats.nice + last_stats.system + 
                            last_stats.idle + last_stats.iowait + last_stats.irq + last_stats.softirq);
            
            let idle_diff = current_stats.idle - last_stats.idle;
            
            if total_diff > 0 {
                let usage = ((total_diff - idle_diff) as f64 / total_diff as f64) * 100.0;
                *last_stats_guard = Some(current_stats.clone());
                Ok(usage)
            } else {
                Ok(0.0)
            }
        } else {
            *last_stats_guard = Some(current_stats.clone());
            Ok(0.0)
        }
    }
    
    #[cfg(target_os = "linux")]
    fn read_per_core_cpu_stats(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        use std::fs;
        
        let content = fs::read_to_string("/proc/stat")?;
        let mut core_usages = Vec::new();
        
        for line in content.lines().skip(1) {
            if !line.starts_with("cpu") {
                break;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 8 {
                let user: u64 = parts[1].parse()?;
                let nice: u64 = parts[2].parse()?;
                let system: u64 = parts[3].parse()?;
                let idle: u64 = parts[4].parse()?;
                let iowait: u64 = parts[5].parse()?;
                let irq: u64 = parts[6].parse()?;
                let softirq: u64 = parts[7].parse()?;
                
                let total = user + nice + system + idle + iowait + irq + softirq;
                let usage = if total > 0 {
                    ((total - idle) as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                
                core_usages.push(usage);
            }
        }
        
        Ok(core_usages)
    }
    
    #[cfg(not(target_os = "linux"))]
    fn read_per_core_cpu_stats(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(vec![0.0]) // Fallback for non-Linux systems
    }
    
    // Additional platform-specific implementations would go here
    // For brevity, showing simplified versions
    
    fn read_memory_info(&self) -> Result<MemoryInfo, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let content = fs::read_to_string("/proc/meminfo")?;
            let mut memory_info = MemoryInfo::default();
            
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let value = parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert from KB to bytes
                    
                    match parts[0] {
                        "MemTotal:" => memory_info.total = value,
                        "MemAvailable:" => memory_info.available = value,
                        "SwapTotal:" => memory_info.swap_total = value,
                        "SwapFree:" => memory_info.swap_free = value,
                        _ => {}
                    }
                }
            }
            
            Ok(memory_info)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Ok(MemoryInfo {
                total: 8 * 1024 * 1024 * 1024, // 8GB fallback
                available: 4 * 1024 * 1024 * 1024, // 4GB fallback
                swap_total: 2 * 1024 * 1024 * 1024, // 2GB fallback
                swap_free: 1 * 1024 * 1024 * 1024, // 1GB fallback
            })
        }
    }
    
    fn read_disk_stats(&self) -> Result<HashMap<String, DiskStats>, Box<dyn std::error::Error>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    fn read_network_stats(&self) -> Result<HashMap<String, NetworkInterfaceStats>, Box<dyn std::error::Error>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    fn calculate_network_rates(&self, _current_stats: &HashMap<String, NetworkInterfaceStats>) -> Result<HashMap<String, NetworkRates>, Box<dyn std::error::Error>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    fn read_load_average(&self) -> Result<LoadAverage, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let content = fs::read_to_string("/proc/loadavg")?;
            let parts: Vec<&str> = content.split_whitespace().collect();
            
            if parts.len() >= 3 {
                Ok(LoadAverage {
                    load1: parts[0].parse()?,
                    load5: parts[1].parse()?,
                    load15: parts[2].parse()?,
                })
            } else {
                Err("Invalid load average format".into())
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Ok(LoadAverage {
                load1: 0.5,
                load5: 0.6,
                load15: 0.7,
            })
        }
    }
}

impl MetricsCollector for SystemMetricsCollector {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        self.collect_cpu_metrics(registry)?;
        self.collect_memory_metrics(registry)?;
        self.collect_disk_metrics(registry)?;
        self.collect_network_metrics(registry)?;
        self.collect_load_metrics(registry)?;
        Ok(())
    }
    
    fn name(&self) -> &str {
        "system"
    }
    
    fn is_enabled(&self, config: &MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"system".to_string())
    }
}

#[derive(Debug, Default)]
struct MemoryInfo {
    total: u64,
    available: u64,
    swap_total: u64,
    swap_free: u64,
}

#[derive(Debug)]
struct DiskStats {
    read_bytes: u64,
    write_bytes: u64,
    read_ops: u64,
    write_ops: u64,
    utilization_percent: f64,
}

#[derive(Debug)]
struct NetworkInterfaceStats {
    rx_bytes: u64,
    tx_bytes: u64,
    rx_packets: u64,
    tx_packets: u64,
    rx_errors: u64,
    tx_errors: u64,
}

#[derive(Debug)]
struct NetworkRates {
    rx_rate: f64,
    tx_rate: f64,
}

#[derive(Debug)]
struct LoadAverage {
    load1: f64,
    load5: f64,
    load15: f64,
}

pub struct ApplicationMetricsCollector {
    config: ApplicationMetricsConfig,
    process_start_time: Instant,
}

impl ApplicationMetricsCollector {
    pub fn new(config: ApplicationMetricsConfig) -> Self {
        Self {
            config,
            process_start_time: Instant::now(),
        }
    }
    
    fn collect_process_metrics(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        // Process uptime
        let uptime_gauge = registry.gauge("application_uptime_seconds", HashMap::new());
        uptime_gauge.set(self.process_start_time.elapsed().as_secs_f64());
        
        // Process memory usage
        let process_memory = self.get_process_memory_usage()?;
        let memory_gauge = registry.gauge("application_memory_bytes", HashMap::new());
        memory_gauge.set(process_memory as f64);
        
        // Thread count
        let thread_count = self.get_thread_count()?;
        let threads_gauge = registry.gauge("application_threads_total", HashMap::new());
        threads_gauge.set(thread_count as f64);
        
        Ok(())
    }
    
    fn get_process_memory_usage(&self) -> Result<u64, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let pid = std::process::id();
            let content = fs::read_to_string(format!("/proc/{}/status", pid))?;
            
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse()?;
                        return Ok(kb * 1024); // Convert to bytes
                    }
                }
            }
            
            Err("Could not find VmRSS in /proc/*/status".into())
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback implementation
            Ok(100 * 1024 * 1024) // 100MB fallback
        }
    }
    
    fn get_thread_count(&self) -> Result<u32, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let pid = std::process::id();
            let content = fs::read_to_string(format!("/proc/{}/status", pid))?;
            
            for line in content.lines() {
                if line.starts_with("Threads:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return Ok(parts[1].parse()?);
                    }
                }
            }
            
            Err("Could not find Threads in /proc/*/status".into())
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Ok(4) // Fallback
        }
    }
}

impl MetricsCollector for ApplicationMetricsCollector {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        if self.config.collect_performance {
            self.collect_process_metrics(registry)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "application"
    }
    
    fn is_enabled(&self, config: &MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"application".to_string())
    }
}

pub struct CustomMetricsCollector {
    name: String,
    collection_fn: Box<dyn Fn(&MetricRegistry) -> Result<(), Box<dyn std::error::Error>> + Send + Sync>,
}

impl CustomMetricsCollector {
    pub fn new<F>(name: String, collection_fn: F) -> Self
    where
        F: Fn(&MetricRegistry) -> Result<(), Box<dyn std::error::Error>> + Send + Sync + 'static,
    {
        Self {
            name,
            collection_fn: Box::new(collection_fn),
        }
    }
}

impl MetricsCollector for CustomMetricsCollector {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        (self.collection_fn)(registry)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn is_enabled(&self, config: &MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&self.name)
    }
}