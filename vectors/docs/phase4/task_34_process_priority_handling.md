# Task 34: Implement Process Priority Handling

## Context
You are implementing Phase 4 of a vector indexing system. File system optimizations have been implemented. Now you need to create comprehensive process priority handling that manages system resources efficiently, prevents system lockup during indexing, and provides adaptive priority management based on system load.

## Current State
- `src/windows.rs` has file system optimizations with performance monitoring
- Filesystem-specific optimizations are working efficiently
- Need process priority management for optimal system resource usage
- Must balance indexing performance with system responsiveness

## Task Objective
Implement comprehensive process priority handling with dynamic priority adjustment, system load monitoring, resource throttling, and adaptive scheduling for optimal indexing performance without impacting system usability.

## Implementation Requirements

### 1. Add process priority management system
Add this priority management system to `src/windows.rs`:
```rust
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessPriority {
    Idle,           // Lowest priority - only when system is idle
    BelowNormal,    // Below normal priority
    Normal,         // Normal priority
    AboveNormal,    // Above normal priority  
    High,           // High priority (use carefully)
    Realtime,       // Realtime priority (dangerous - avoid)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadPriority {
    Idle,
    Lowest,
    BelowNormal,
    Normal,
    AboveNormal,
    Highest,
    TimeCritical,
}

#[derive(Debug, Clone)]
pub struct SystemLoadMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub active_processes: u32,
    pub system_responsive: bool,
    pub measurement_time: Instant,
}

#[derive(Debug, Clone)]
pub struct PrioritySettings {
    pub base_process_priority: ProcessPriority,
    pub indexing_thread_priority: ThreadPriority,
    pub io_thread_priority: ThreadPriority,
    pub adaptive_priority: bool,
    pub max_cpu_usage_threshold: f64,
    pub max_memory_usage_threshold: f64,
    pub priority_adjustment_interval: Duration,
    pub throttle_on_high_load: bool,
}

impl Default for PrioritySettings {
    fn default() -> Self {
        Self {
            base_process_priority: ProcessPriority::BelowNormal,
            indexing_thread_priority: ThreadPriority::BelowNormal,
            io_thread_priority: ThreadPriority::Normal,
            adaptive_priority: true,
            max_cpu_usage_threshold: 80.0,
            max_memory_usage_threshold: 85.0,
            priority_adjustment_interval: Duration::from_secs(5),
            throttle_on_high_load: true,
        }
    }
}

pub struct ProcessPriorityManager {
    settings: PrioritySettings,
    current_priority: ProcessPriority,
    system_monitor: Arc<Mutex<SystemLoadMonitor>>,
    adjustment_thread: Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
    priority_change_count: AtomicU64,
}

impl ProcessPriorityManager {
    pub fn new(settings: PrioritySettings) -> Result<Self> {
        let system_monitor = Arc::new(Mutex::new(SystemLoadMonitor::new()?));
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        
        let mut manager = Self {
            current_priority: settings.base_process_priority,
            settings,
            system_monitor,
            adjustment_thread: None,
            shutdown_signal,
            priority_change_count: AtomicU64::new(0),
        };
        
        // Set initial process priority
        manager.set_process_priority(manager.settings.base_process_priority)?;
        
        // Start adaptive priority adjustment if enabled
        if manager.settings.adaptive_priority {
            manager.start_adaptive_priority_thread()?;
        }
        
        Ok(manager)
    }
    
    pub fn set_process_priority(&mut self, priority: ProcessPriority) -> Result<()> {
        #[cfg(windows)]
        {
            self.set_windows_process_priority(priority)?;
        }
        #[cfg(not(windows))]
        {
            self.set_unix_process_priority(priority)?;
        }
        
        self.current_priority = priority;
        self.priority_change_count.fetch_add(1, Ordering::Relaxed);
        
        println!("Process priority changed to: {:?}", priority);
        Ok(())
    }
    
    #[cfg(windows)]
    fn set_windows_process_priority(&self, priority: ProcessPriority) -> Result<()> {
        use std::process;
        
        // This would use Windows API to set process priority
        // For now, we'll simulate the behavior
        let priority_class = match priority {
            ProcessPriority::Idle => "IDLE_PRIORITY_CLASS",
            ProcessPriority::BelowNormal => "BELOW_NORMAL_PRIORITY_CLASS",
            ProcessPriority::Normal => "NORMAL_PRIORITY_CLASS",
            ProcessPriority::AboveNormal => "ABOVE_NORMAL_PRIORITY_CLASS",
            ProcessPriority::High => "HIGH_PRIORITY_CLASS",
            ProcessPriority::Realtime => "REALTIME_PRIORITY_CLASS",
        };
        
        println!("Setting Windows process priority to: {}", priority_class);
        
        // In a real implementation, this would call:
        // SetPriorityClass(GetCurrentProcess(), priority_class_value);
        
        Ok(())
    }
    
    #[cfg(not(windows))]
    fn set_unix_process_priority(&self, priority: ProcessPriority) -> Result<()> {
        // Unix nice values: -20 (highest) to 19 (lowest)
        let nice_value = match priority {
            ProcessPriority::Idle => 19,
            ProcessPriority::BelowNormal => 10,
            ProcessPriority::Normal => 0,
            ProcessPriority::AboveNormal => -5,
            ProcessPriority::High => -10,
            ProcessPriority::Realtime => -20,
        };
        
        println!("Setting Unix process nice value to: {}", nice_value);
        
        // In a real implementation, this would call:
        // unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice_value) };
        
        Ok(())
    }
    
    pub fn set_thread_priority(&self, priority: ThreadPriority) -> Result<()> {
        #[cfg(windows)]
        {
            self.set_windows_thread_priority(priority)?;
        }
        #[cfg(not(windows))]
        {
            self.set_unix_thread_priority(priority)?;
        }
        
        Ok(())
    }
    
    #[cfg(windows)]
    fn set_windows_thread_priority(&self, priority: ThreadPriority) -> Result<()> {
        let thread_priority = match priority {
            ThreadPriority::Idle => "THREAD_PRIORITY_IDLE",
            ThreadPriority::Lowest => "THREAD_PRIORITY_LOWEST",
            ThreadPriority::BelowNormal => "THREAD_PRIORITY_BELOW_NORMAL",
            ThreadPriority::Normal => "THREAD_PRIORITY_NORMAL",
            ThreadPriority::AboveNormal => "THREAD_PRIORITY_ABOVE_NORMAL",
            ThreadPriority::Highest => "THREAD_PRIORITY_HIGHEST",
            ThreadPriority::TimeCritical => "THREAD_PRIORITY_TIME_CRITICAL",
        };
        
        println!("Setting Windows thread priority to: {}", thread_priority);
        
        // In a real implementation:
        // SetThreadPriority(GetCurrentThread(), thread_priority_value);
        
        Ok(())
    }
    
    #[cfg(not(windows))]
    fn set_unix_thread_priority(&self, priority: ThreadPriority) -> Result<()> {
        // Unix thread priorities are typically handled through scheduling policies
        println!("Setting Unix thread priority to: {:?}", priority);
        
        // In a real implementation, this would use pthread_setschedparam
        
        Ok(())
    }
    
    fn start_adaptive_priority_thread(&mut self) -> Result<()> {
        let system_monitor = Arc::clone(&self.system_monitor);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let settings = self.settings.clone();
        let adjustment_interval = self.settings.priority_adjustment_interval;
        
        let handle = thread::spawn(move || {
            let mut last_adjustment = Instant::now();
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
                
                if last_adjustment.elapsed() >= adjustment_interval {
                    if let Ok(mut monitor) = system_monitor.lock() {
                        if let Ok(metrics) = monitor.get_current_metrics() {
                            Self::adjust_priority_based_on_load(&metrics, &settings);
                        }
                    }
                    last_adjustment = Instant::now();
                }
            }
        });
        
        self.adjustment_thread = Some(handle);
        Ok(())
    }
    
    fn adjust_priority_based_on_load(metrics: &SystemLoadMetrics, settings: &PrioritySettings) {
        let should_throttle = metrics.cpu_usage_percent > settings.max_cpu_usage_threshold
            || metrics.memory_usage_percent > settings.max_memory_usage_threshold;
        
        if should_throttle && settings.throttle_on_high_load {
            // System is under high load - reduce priority
            println!("High system load detected - reducing indexing priority");
            println!("  CPU: {:.1}%, Memory: {:.1}%", 
                    metrics.cpu_usage_percent, metrics.memory_usage_percent);
            
            // In a real implementation, we would adjust the actual process priority here
        } else if metrics.cpu_usage_percent < 50.0 && metrics.memory_usage_percent < 60.0 {
            // System has plenty of resources - can increase priority
            println!("Low system load - can increase indexing priority");
        }
    }
    
    pub fn get_current_priority(&self) -> ProcessPriority {
        self.current_priority
    }
    
    pub fn get_system_metrics(&self) -> Result<SystemLoadMetrics> {
        let monitor = self.system_monitor.lock().unwrap();
        monitor.get_current_metrics()
    }
    
    pub fn get_priority_change_count(&self) -> u64 {
        self.priority_change_count.load(Ordering::Relaxed)
    }
    
    pub fn pause_indexing(&self, duration: Duration) -> Result<()> {
        println!("Pausing indexing for {:?} to reduce system load", duration);
        
        // Set to idle priority temporarily
        // In a real implementation, we would:
        // 1. Set process/thread priorities to idle
        // 2. Pause indexing operations
        // 3. Wait for the specified duration
        // 4. Resume with previous priorities
        
        thread::sleep(duration);
        Ok(())
    }
    
    pub fn emergency_throttle(&mut self) -> Result<()> {
        println!("Emergency throttle activated - setting minimum priority");
        self.set_process_priority(ProcessPriority::Idle)?;
        
        // Pause all non-essential operations
        self.pause_indexing(Duration::from_secs(10))?;
        
        Ok(())
    }
}

impl Drop for ProcessPriorityManager {
    fn drop(&mut self) {
        // Signal shutdown and wait for adjustment thread
        self.shutdown_signal.store(true, Ordering::Relaxed);
        
        if let Some(handle) = self.adjustment_thread.take() {
            let _ = handle.join();
        }
        
        // Reset to normal priority on shutdown
        let _ = self.set_process_priority(ProcessPriority::Normal);
    }
}

pub struct SystemLoadMonitor {
    last_measurement: Option<SystemLoadMetrics>,
    measurement_history: Vec<SystemLoadMetrics>,
    max_history_size: usize,
}

impl SystemLoadMonitor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            last_measurement: None,
            measurement_history: Vec::new(),
            max_history_size: 60, // Keep last 60 measurements
        })
    }
    
    pub fn get_current_metrics(&mut self) -> Result<SystemLoadMetrics> {
        let metrics = self.measure_system_load()?;
        
        // Update history
        self.measurement_history.push(metrics.clone());
        if self.measurement_history.len() > self.max_history_size {
            self.measurement_history.remove(0);
        }
        
        self.last_measurement = Some(metrics.clone());
        Ok(metrics)
    }
    
    fn measure_system_load(&self) -> Result<SystemLoadMetrics> {
        // This would use platform-specific APIs to get actual system metrics
        // For now, we'll simulate realistic values
        
        #[cfg(windows)]
        {
            self.measure_windows_system_load()
        }
        #[cfg(not(windows))]
        {
            self.measure_unix_system_load()
        }
    }
    
    #[cfg(windows)]
    fn measure_windows_system_load(&self) -> Result<SystemLoadMetrics> {
        use std::process::Command;
        
        // Simulate getting system metrics
        // In a real implementation, this would use WMI or Performance Counters
        let mut cpu_usage = 0.0;
        let mut memory_usage = 0.0;
        
        // Try to get CPU usage from wmic (simplified)
        if let Ok(output) = Command::new("wmic")
            .args(&["cpu", "get", "loadpercentage", "/value"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().find(|l| l.starts_with("LoadPercentage=")) {
                if let Some(value_str) = line.split('=').nth(1) {
                    cpu_usage = value_str.trim().parse().unwrap_or(0.0);
                }
            }
        }
        
        // Simulate other metrics
        memory_usage = 45.0 + (rand::random::<f64>() * 30.0); // 45-75%
        let disk_usage = 20.0 + (rand::random::<f64>() * 40.0); // 20-60%
        
        Ok(SystemLoadMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_percent: memory_usage,
            disk_usage_percent: disk_usage,
            active_processes: 150 + (rand::random::<u32>() % 50),
            system_responsive: cpu_usage < 90.0 && memory_usage < 95.0,
            measurement_time: Instant::now(),
        })
    }
    
    #[cfg(not(windows))]
    fn measure_unix_system_load(&self) -> Result<SystemLoadMetrics> {
        use std::process::Command;
        
        let mut cpu_usage = 0.0;
        let mut memory_usage = 0.0;
        
        // Try to get load average
        if let Ok(output) = Command::new("uptime").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse load average - simplified implementation
            if let Some(load_part) = output_str.split("load average:").nth(1) {
                if let Some(first_load) = load_part.split(',').next() {
                    let load = first_load.trim().parse::<f64>().unwrap_or(0.0);
                    cpu_usage = (load * 100.0).min(100.0); // Rough conversion
                }
            }
        }
        
        // Try to get memory usage from free command
        if let Ok(output) = Command::new("free").args(&["-m"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.starts_with("Mem:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        let total: f64 = parts[1].parse().unwrap_or(1.0);
                        let used: f64 = parts[2].parse().unwrap_or(0.0);
                        memory_usage = (used / total) * 100.0;
                    }
                    break;
                }
            }
        }
        
        Ok(SystemLoadMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_percent: memory_usage,
            disk_usage_percent: 30.0, // Simplified
            active_processes: 200,     // Simplified
            system_responsive: cpu_usage < 90.0 && memory_usage < 95.0,
            measurement_time: Instant::now(),
        })
    }
    
    pub fn get_average_metrics(&self, duration: Duration) -> Option<SystemLoadMetrics> {
        let cutoff_time = Instant::now() - duration;
        let recent_metrics: Vec<&SystemLoadMetrics> = self.measurement_history
            .iter()
            .filter(|m| m.measurement_time > cutoff_time)
            .collect();
        
        if recent_metrics.is_empty() {
            return None;
        }
        
        let count = recent_metrics.len() as f64;
        let avg_cpu = recent_metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / count;
        let avg_memory = recent_metrics.iter().map(|m| m.memory_usage_percent).sum::<f64>() / count;
        let avg_disk = recent_metrics.iter().map(|m| m.disk_usage_percent).sum::<f64>() / count;
        
        Some(SystemLoadMetrics {
            cpu_usage_percent: avg_cpu,
            memory_usage_percent: avg_memory,
            disk_usage_percent: avg_disk,
            active_processes: recent_metrics.last()?.active_processes,
            system_responsive: avg_cpu < 80.0 && avg_memory < 85.0,
            measurement_time: Instant::now(),
        })
    }
}

// Add random number generation for simulation
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    pub fn random<T>() -> T 
    where 
        T: From<u64>,
    {
        let mut hasher = DefaultHasher::new();
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().hash(&mut hasher);
        T::from(hasher.finish())
    }
}
```

### 2. Add integration with indexing system
Add these integration methods:
```rust
// Integration with WindowsPathHandler
impl WindowsPathHandler {
    pub fn with_priority_manager(self, settings: PrioritySettings) -> Result<IndexingSystemWithPriority> {
        let priority_manager = ProcessPriorityManager::new(settings)?;
        
        Ok(IndexingSystemWithPriority {
            path_handler: self,
            priority_manager: Arc::new(Mutex::new(priority_manager)),
        })
    }
}

pub struct IndexingSystemWithPriority {
    path_handler: WindowsPathHandler,
    priority_manager: Arc<Mutex<ProcessPriorityManager>>,
}

impl IndexingSystemWithPriority {
    pub fn index_directory_with_priority_management(&self, path: &Path) -> Result<IndexingResults> {
        let mut results = IndexingResults::new();
        let start_time = Instant::now();
        
        // Set appropriate thread priority for indexing
        {
            let manager = self.priority_manager.lock().unwrap();
            manager.set_thread_priority(ThreadPriority::BelowNormal)?;
        }
        
        // Check system load before starting intensive work
        let initial_metrics = {
            let manager = self.priority_manager.lock().unwrap();
            manager.get_system_metrics()?
        };
        
        if initial_metrics.cpu_usage_percent > 85.0 {
            println!("High CPU usage detected ({:.1}%) - throttling indexing", 
                    initial_metrics.cpu_usage_percent);
            
            let mut manager = self.priority_manager.lock().unwrap();
            manager.pause_indexing(Duration::from_secs(5))?;
        }
        
        // Perform indexing with periodic priority checks
        self.index_with_adaptive_priority(path, &mut results)?;
        
        results.total_time = start_time.elapsed();
        Ok(results)
    }
    
    fn index_with_adaptive_priority(&self, path: &Path, results: &mut IndexingResults) -> Result<()> {
        let mut files_processed = 0;
        let check_interval = 50; // Check system load every 50 files
        
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            
            if entry_path.is_file() {
                // Process file
                self.path_handler.validate_windows_path(&entry_path)?;
                results.files_processed += 1;
                files_processed += 1;
                
                // Periodically check system load
                if files_processed % check_interval == 0 {
                    let metrics = {
                        let manager = self.priority_manager.lock().unwrap();
                        manager.get_system_metrics()?
                    };
                    
                    // Adapt behavior based on system load
                    if metrics.cpu_usage_percent > 90.0 || metrics.memory_usage_percent > 90.0 {
                        println!("High system load - emergency throttle");
                        let mut manager = self.priority_manager.lock().unwrap();
                        manager.emergency_throttle()?;
                        
                        results.throttle_events += 1;
                    } else if metrics.cpu_usage_percent > 80.0 {
                        // Mild throttling - small pause
                        thread::sleep(Duration::from_millis(10));
                        results.mild_throttle_events += 1;
                    }
                }
            } else if entry_path.is_dir() {
                // Recursively process subdirectory
                self.index_with_adaptive_priority(&entry_path, results)?;
            }
        }
        
        Ok(())
    }
    
    pub fn get_priority_statistics(&self) -> Result<PriorityStatistics> {
        let manager = self.priority_manager.lock().unwrap();
        let current_metrics = manager.get_system_metrics()?;
        
        Ok(PriorityStatistics {
            current_priority: manager.get_current_priority(),
            priority_changes: manager.get_priority_change_count(),
            current_cpu_usage: current_metrics.cpu_usage_percent,
            current_memory_usage: current_metrics.memory_usage_percent,
            system_responsive: current_metrics.system_responsive,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IndexingResults {
    pub files_processed: u64,
    pub directories_processed: u64,
    pub total_time: Duration,
    pub throttle_events: u64,
    pub mild_throttle_events: u64,
    pub errors: Vec<String>,
}

impl IndexingResults {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            directories_processed: 0,
            total_time: Duration::new(0, 0),
            throttle_events: 0,
            mild_throttle_events: 0,
            errors: Vec::new(),
        }
    }
    
    pub fn files_per_second(&self) -> f64 {
        if self.total_time.as_secs() > 0 {
            self.files_processed as f64 / self.total_time.as_secs() as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct PriorityStatistics {
    pub current_priority: ProcessPriority,
    pub priority_changes: u64,
    pub current_cpu_usage: f64,
    pub current_memory_usage: f64,
    pub system_responsive: bool,
}
```

### 3. Add comprehensive tests
Add these test modules:
```rust
#[cfg(test)]
mod priority_management_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_priority_manager_creation() -> Result<()> {
        let settings = PrioritySettings::default();
        let manager = ProcessPriorityManager::new(settings)?;
        
        assert_eq!(manager.get_current_priority(), ProcessPriority::BelowNormal);
        assert_eq!(manager.get_priority_change_count(), 1); // Initial set
        
        Ok(())
    }
    
    #[test]
    fn test_priority_changes() -> Result<()> {
        let settings = PrioritySettings {
            adaptive_priority: false, // Disable adaptive for testing
            ..Default::default()
        };
        let mut manager = ProcessPriorityManager::new(settings)?;
        
        // Test setting different priorities
        manager.set_process_priority(ProcessPriority::Normal)?;
        assert_eq!(manager.get_current_priority(), ProcessPriority::Normal);
        
        manager.set_process_priority(ProcessPriority::High)?;
        assert_eq!(manager.get_current_priority(), ProcessPriority::High);
        
        manager.set_process_priority(ProcessPriority::Idle)?;
        assert_eq!(manager.get_current_priority(), ProcessPriority::Idle);
        
        // Should have made 4 priority changes (initial + 3 manual)
        assert_eq!(manager.get_priority_change_count(), 4);
        
        Ok(())
    }
    
    #[test]
    fn test_system_load_monitoring() -> Result<()> {
        let mut monitor = SystemLoadMonitor::new()?;
        
        let metrics = monitor.get_current_metrics()?;
        
        println!("System metrics:");
        println!("  CPU: {:.1}%", metrics.cpu_usage_percent);
        println!("  Memory: {:.1}%", metrics.memory_usage_percent);
        println!("  Disk: {:.1}%", metrics.disk_usage_percent);
        println!("  Processes: {}", metrics.active_processes);
        println!("  Responsive: {}", metrics.system_responsive);
        
        // Basic sanity checks
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.cpu_usage_percent <= 100.0);
        assert!(metrics.memory_usage_percent >= 0.0);
        assert!(metrics.memory_usage_percent <= 100.0);
        
        Ok(())
    }
    
    #[test]
    fn test_adaptive_priority_adjustment() -> Result<()> {
        let settings = PrioritySettings {
            adaptive_priority: true,
            priority_adjustment_interval: Duration::from_millis(100),
            max_cpu_usage_threshold: 50.0, // Low threshold for testing
            ..Default::default()
        };
        
        let manager = ProcessPriorityManager::new(settings)?;
        
        // Wait for a few adjustment cycles
        thread::sleep(Duration::from_millis(300));
        
        // Check that system metrics are being collected
        let metrics = manager.get_system_metrics()?;
        assert!(metrics.measurement_time.elapsed() < Duration::from_secs(1));
        
        Ok(())
    }
    
    #[test]
    fn test_emergency_throttle() -> Result<()> {
        let settings = PrioritySettings::default();
        let mut manager = ProcessPriorityManager::new(settings)?;
        
        // Set to normal priority first
        manager.set_process_priority(ProcessPriority::Normal)?;
        assert_eq!(manager.get_current_priority(), ProcessPriority::Normal);
        
        // Trigger emergency throttle
        let start = Instant::now();
        manager.emergency_throttle()?;
        let duration = start.elapsed();
        
        // Should have changed to idle priority
        assert_eq!(manager.get_current_priority(), ProcessPriority::Idle);
        
        // Should have taken at least 10 seconds due to pause
        assert!(duration >= Duration::from_secs(10));
        
        Ok(())
    }
    
    #[test]
    fn test_thread_priority_setting() -> Result<()> {
        let settings = PrioritySettings::default();
        let manager = ProcessPriorityManager::new(settings)?;
        
        // Test setting various thread priorities
        manager.set_thread_priority(ThreadPriority::Lowest)?;
        manager.set_thread_priority(ThreadPriority::Normal)?;
        manager.set_thread_priority(ThreadPriority::AboveNormal)?;
        
        // Should complete without errors
        Ok(())
    }
    
    #[test]
    fn test_indexing_with_priority_management() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create test files
        for i in 0..10 {
            std::fs::write(temp_dir.path().join(format!("file_{}.txt", i)), 
                          format!("Content {}", i))?;
        }
        
        let handler = WindowsPathHandler::new();
        let settings = PrioritySettings {
            adaptive_priority: false, // Disable for predictable testing
            ..Default::default()
        };
        
        let indexing_system = handler.with_priority_manager(settings)?;
        let results = indexing_system.index_directory_with_priority_management(temp_dir.path())?;
        
        println!("Indexing results:");
        println!("  Files processed: {}", results.files_processed);
        println!("  Total time: {:?}", results.total_time);
        println!("  Files per second: {:.2}", results.files_per_second());
        println!("  Throttle events: {}", results.throttle_events);
        
        assert_eq!(results.files_processed, 10);
        assert!(results.total_time > Duration::new(0, 0));
        
        Ok(())
    }
    
    #[test]
    fn test_priority_statistics() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = WindowsPathHandler::new();
        let settings = PrioritySettings::default();
        
        let indexing_system = handler.with_priority_manager(settings)?;
        let stats = indexing_system.get_priority_statistics()?;
        
        println!("Priority statistics:");
        println!("  Current priority: {:?}", stats.current_priority);
        println!("  Priority changes: {}", stats.priority_changes);
        println!("  CPU usage: {:.1}%", stats.current_cpu_usage);
        println!("  Memory usage: {:.1}%", stats.current_memory_usage);
        println!("  System responsive: {}", stats.system_responsive);
        
        assert!(stats.priority_changes >= 1); // At least initial setting
        
        Ok(())
    }
    
    #[test]
    fn test_system_load_history() -> Result<()> {
        let mut monitor = SystemLoadMonitor::new()?;
        
        // Collect several measurements
        for _ in 0..5 {
            monitor.get_current_metrics()?;
            thread::sleep(Duration::from_millis(10));
        }
        
        // Get average over recent measurements
        let avg_metrics = monitor.get_average_metrics(Duration::from_secs(1));
        assert!(avg_metrics.is_some());
        
        let avg = avg_metrics.unwrap();
        println!("Average metrics over 1 second:");
        println!("  CPU: {:.1}%", avg.cpu_usage_percent);
        println!("  Memory: {:.1}%", avg.memory_usage_percent);
        
        Ok(())
    }
    
    #[test]
    fn test_priority_manager_cleanup() -> Result<()> {
        let settings = PrioritySettings {
            adaptive_priority: true,
            priority_adjustment_interval: Duration::from_millis(50),
            ..Default::default()
        };
        
        {
            let manager = ProcessPriorityManager::new(settings)?;
            thread::sleep(Duration::from_millis(100));
            // Manager goes out of scope here and should clean up
        }
        
        // Should complete without hanging or errors
        println!("Priority manager cleanup completed successfully");
        
        Ok(())
    }
    
    #[test]
    #[ignore] // Expensive test - run with --ignored
    fn test_high_load_simulation() -> Result<()> {
        let settings = PrioritySettings {
            max_cpu_usage_threshold: 30.0, // Very low threshold
            max_memory_usage_threshold: 30.0,
            throttle_on_high_load: true,
            ..Default::default()
        };
        
        let temp_dir = TempDir::new()?;
        
        // Create many files to trigger throttling
        for i in 0..1000 {
            std::fs::write(temp_dir.path().join(format!("file_{}.txt", i)), 
                          format!("Content {}", i))?;
        }
        
        let handler = WindowsPathHandler::new();
        let indexing_system = handler.with_priority_manager(settings)?;
        
        let start = Instant::now();
        let results = indexing_system.index_directory_with_priority_management(temp_dir.path())?;
        let duration = start.elapsed();
        
        println!("High load simulation results:");
        println!("  Files processed: {}", results.files_processed);
        println!("  Duration: {:?}", duration);
        println!("  Throttle events: {}", results.throttle_events);
        println!("  Mild throttle events: {}", results.mild_throttle_events);
        
        // Should eventually process all files, possibly with throttling
        assert_eq!(results.files_processed, 1000);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Comprehensive process and thread priority management
- [ ] Real-time system load monitoring with historical tracking
- [ ] Adaptive priority adjustment based on system load
- [ ] Emergency throttling for high load conditions
- [ ] Integration with indexing system for optimal performance
- [ ] Cross-platform priority handling (Windows and Unix)
- [ ] Thread-safe priority management with proper cleanup
- [ ] Performance statistics and monitoring
- [ ] Graceful degradation under high system load
- [ ] All tests pass with realistic system load simulation
- [ ] No system lockups or unresponsive behavior
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Process priorities should be used carefully to avoid system unresponsiveness
- Realtime priority can cause system lockups and should generally be avoided
- Adaptive priority adjustment helps balance performance with system usability
- System load monitoring provides early warning of resource constraints
- Thread priorities are more granular than process priorities
- Emergency throttling prevents the indexer from overwhelming the system
- Cleanup on shutdown ensures system resources are properly restored
- Cross-platform compatibility requires different APIs on Windows vs Unix systems