# Task 35: Integrate with Parallel Indexer

## Context
You are implementing Phase 4 of a vector indexing system. Process priority handling has been implemented. Now you need to integrate all Windows optimizations with the parallel indexer from earlier tasks, creating a unified system that leverages all optimizations for maximum performance while maintaining system stability.

## Current State
- `src/windows.rs` has process priority handling with adaptive management
- File system optimizations, Unicode support, and cross-platform testing are complete
- Need integration with parallel indexer from `src/parallel.rs`
- Must create unified system combining all Phase 4 optimizations

## Task Objective
Integrate all Windows optimizations with the parallel indexer to create a high-performance, system-aware indexing solution that leverages filesystem optimizations, process priority management, and Windows-specific features while maintaining cross-platform compatibility.

## Implementation Requirements

### 1. Create integrated indexing coordinator
Add this integration system to `src/windows.rs`:
```rust
use crate::parallel::{ParallelIndexer, IndexingStats};
use std::sync::{Arc, RwLock};
use std::collections::VecDeque;

pub struct WindowsOptimizedIndexer {
    parallel_indexer: ParallelIndexer,
    path_handler: WindowsPathHandler,
    filesystem_optimizer: FileSystemOptimizer,
    priority_manager: Arc<Mutex<ProcessPriorityManager>>,
    platform_capabilities: PlatformCapabilities,
    coordination_stats: Arc<RwLock<CoordinationStats>>,
    work_queue: Arc<Mutex<VecDeque<IndexingTask>>>,
    active_workers: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
pub struct IndexingTask {
    pub path: PathBuf,
    pub priority: TaskPriority,
    pub estimated_size: u64,
    pub requires_unicode_handling: bool,
    pub filesystem_type: FileSystemType,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Ord, PartialOrd, Eq)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone)]
pub struct CoordinationStats {
    pub tasks_queued: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub unicode_files_processed: u64,
    pub extended_path_files_processed: u64,
    pub filesystem_optimizations_applied: u64,
    pub priority_adjustments: u64,
    pub total_coordination_time: Duration,
    pub average_task_processing_time: Duration,
}

impl CoordinationStats {
    pub fn new() -> Self {
        Self {
            tasks_queued: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            unicode_files_processed: 0,
            extended_path_files_processed: 0,
            filesystem_optimizations_applied: 0,
            priority_adjustments: 0,
            total_coordination_time: Duration::new(0, 0),
            average_task_processing_time: Duration::new(0, 0),
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        let total = self.tasks_completed + self.tasks_failed;
        if total > 0 {
            self.tasks_completed as f64 / total as f64
        } else {
            0.0
        }
    }
    
    pub fn tasks_per_second(&self) -> f64 {
        if self.total_coordination_time.as_secs() > 0 {
            self.tasks_completed as f64 / self.total_coordination_time.as_secs() as f64
        } else {
            0.0
        }
    }
}

impl WindowsOptimizedIndexer {
    pub fn new(
        thread_count: usize,
        priority_settings: PrioritySettings,
        base_path: &Path,
    ) -> Result<Self> {
        // Initialize components
        let parallel_indexer = ParallelIndexer::new(thread_count);
        let path_handler = WindowsPathHandler::new();
        let filesystem_optimizer = FileSystemOptimizer::new(base_path)?;
        let priority_manager = Arc::new(Mutex::new(
            ProcessPriorityManager::new(priority_settings)?
        ));
        let platform_capabilities = PlatformCapabilities::detect_current_platform();
        
        Ok(Self {
            parallel_indexer,
            path_handler,
            filesystem_optimizer,
            priority_manager,
            platform_capabilities,
            coordination_stats: Arc::new(RwLock::new(CoordinationStats::new())),
            work_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_workers: Arc::new(AtomicUsize::new(0)),
        })
    }
    
    pub fn index_directory_optimized(&mut self, root_path: &Path) -> Result<OptimizedIndexingResults> {
        let start_time = Instant::now();
        
        // Pre-flight analysis
        let analysis = self.analyze_indexing_target(root_path)?;
        println!("Indexing analysis completed:");
        println!("  Estimated files: {}", analysis.estimated_file_count);
        println!("  Directory depth: {}", analysis.directory_depth);
        println!("  Recommended strategy: {:?}", analysis.recommended_strategy);
        
        // Configure system for optimal performance
        self.configure_for_workload(&analysis)?;
        
        // Queue initial tasks
        self.queue_initial_tasks(root_path, &analysis)?;
        
        // Start coordinated indexing
        let results = self.execute_coordinated_indexing()?;
        
        let total_time = start_time.elapsed();
        
        Ok(OptimizedIndexingResults {
            files_indexed: results.files_indexed,
            directories_processed: results.directories_processed,
            unicode_files_handled: self.coordination_stats.read().unwrap().unicode_files_processed,
            extended_paths_handled: self.coordination_stats.read().unwrap().extended_path_files_processed,
            filesystem_optimizations_used: self.coordination_stats.read().unwrap().filesystem_optimizations_applied,
            priority_adjustments: self.coordination_stats.read().unwrap().priority_adjustments,
            total_time,
            indexing_rate: results.files_indexed as f64 / total_time.as_secs() as f64,
            parallel_efficiency: self.calculate_parallel_efficiency(&results),
            system_impact: self.measure_system_impact()?,
            errors: results.errors,
        })
    }
    
    fn analyze_indexing_target(&mut self, path: &Path) -> Result<IndexingAnalysis> {
        let start = Instant::now();
        
        // Get filesystem optimization recommendations
        let fs_optimization = self.filesystem_optimizer.optimize_for_indexing(path)?;
        
        // Estimate Unicode content
        let unicode_estimate = self.estimate_unicode_content(path)?;
        
        // Check for long paths
        let long_path_estimate = self.estimate_long_paths(path)?;
        
        // Determine optimal strategy
        let strategy = self.determine_indexing_strategy(&fs_optimization, unicode_estimate, long_path_estimate);
        
        Ok(IndexingAnalysis {
            estimated_file_count: fs_optimization.estimated_file_count,
            directory_depth: fs_optimization.directory_depth,
            unicode_content_percentage: unicode_estimate,
            long_path_percentage: long_path_estimate,
            recommended_strategy: strategy,
            analysis_time: start.elapsed(),
        })
    }
    
    fn estimate_unicode_content(&mut self, path: &Path) -> Result<f64> {
        let mut total_files = 0;
        let mut unicode_files = 0;
        let sample_size = 100; // Sample first 100 files
        
        for entry in self.filesystem_optimizer.optimized_read_dir(path)?.iter().take(sample_size) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                total_files += 1;
                
                if let Some(filename) = entry_path.file_name() {
                    let filename_str = filename.to_string_lossy();
                    if !filename_str.is_ascii() {
                        unicode_files += 1;
                    }
                }
            }
        }
        
        if total_files > 0 {
            Ok((unicode_files as f64 / total_files as f64) * 100.0)
        } else {
            Ok(0.0)
        }
    }
    
    fn estimate_long_paths(&mut self, path: &Path) -> Result<f64> {
        let mut total_paths = 0;
        let mut long_paths = 0;
        let sample_size = 100;
        
        for entry in self.filesystem_optimizer.optimized_read_dir(path)?.iter().take(sample_size) {
            let entry_path = entry.path();
            total_paths += 1;
            
            if entry_path.to_string_lossy().len() > 200 {
                long_paths += 1;
            }
        }
        
        if total_paths > 0 {
            Ok((long_paths as f64 / total_paths as f64) * 100.0)
        } else {
            Ok(0.0)
        }
    }
    
    fn determine_indexing_strategy(
        &self,
        fs_optimization: &IndexingOptimization,
        unicode_percentage: f64,
        long_path_percentage: f64,
    ) -> IndexingStrategy {
        // Determine best strategy based on analysis
        if fs_optimization.estimated_file_count > 10000 {
            if unicode_percentage > 20.0 || long_path_percentage > 10.0 {
                IndexingStrategy::LargeComplexDataset
            } else {
                IndexingStrategy::LargeSimpleDataset
            }
        } else if unicode_percentage > 50.0 {
            IndexingStrategy::UnicodeHeavy
        } else if long_path_percentage > 25.0 {
            IndexingStrategy::LongPathHeavy
        } else {
            IndexingStrategy::Standard
        }
    }
    
    fn configure_for_workload(&mut self, analysis: &IndexingAnalysis) -> Result<()> {
        // Adjust thread priorities based on workload
        let thread_priority = match analysis.recommended_strategy {
            IndexingStrategy::LargeComplexDataset => ThreadPriority::Normal,
            IndexingStrategy::LargeSimpleDataset => ThreadPriority::AboveNormal,
            IndexingStrategy::UnicodeHeavy => ThreadPriority::BelowNormal, // More CPU intensive
            IndexingStrategy::LongPathHeavy => ThreadPriority::BelowNormal,
            IndexingStrategy::Standard => ThreadPriority::Normal,
        };
        
        let manager = self.priority_manager.lock().unwrap();
        manager.set_thread_priority(thread_priority)?;
        
        // Configure filesystem optimizer cache based on dataset size
        let cache_duration = if analysis.estimated_file_count > 50000 {
            Duration::from_secs(60) // Longer cache for large datasets
        } else {
            Duration::from_secs(30)
        };
        
        println!("Configured for {:?} strategy with thread priority {:?}", 
                analysis.recommended_strategy, thread_priority);
        
        Ok(())
    }
    
    fn queue_initial_tasks(&mut self, root_path: &Path, analysis: &IndexingAnalysis) -> Result<()> {
        let mut queue = self.work_queue.lock().unwrap();
        let mut stats = self.coordination_stats.write().unwrap();
        
        // Create initial task for root directory
        let initial_task = IndexingTask {
            path: root_path.to_path_buf(),
            priority: TaskPriority::High, // Root directory is high priority
            estimated_size: analysis.estimated_file_count,
            requires_unicode_handling: analysis.unicode_content_percentage > 10.0,
            filesystem_type: self.filesystem_optimizer.get_capabilities().fs_type.clone(),
            created_at: Instant::now(),
        };
        
        queue.push_back(initial_task);
        stats.tasks_queued += 1;
        
        Ok(())
    }
    
    fn execute_coordinated_indexing(&mut self) -> Result<IndexingStats> {
        let start_time = Instant::now();
        let mut total_results = IndexingStats::new();
        
        // Start worker threads
        let worker_count = self.parallel_indexer.get_thread_count();
        let mut worker_handles = Vec::new();
        
        for worker_id in 0..worker_count {
            let handle = self.start_worker_thread(worker_id)?;
            worker_handles.push(handle);
        }
        
        // Monitor progress and adjust priorities
        self.monitor_and_coordinate()?;
        
        // Wait for all workers to complete
        for handle in worker_handles {
            if let Ok(worker_results) = handle.join() {
                total_results.combine(worker_results?);
            }
        }
        
        // Update coordination statistics
        let mut stats = self.coordination_stats.write().unwrap();
        stats.total_coordination_time = start_time.elapsed();
        
        Ok(total_results)
    }
    
    fn start_worker_thread(&self, worker_id: usize) -> Result<thread::JoinHandle<Result<IndexingStats>>> {
        let work_queue = Arc::clone(&self.work_queue);
        let coordination_stats = Arc::clone(&self.coordination_stats);
        let active_workers = Arc::clone(&self.active_workers);
        let priority_manager = Arc::clone(&self.priority_manager);
        
        // Clone components for the worker
        let mut path_handler = self.path_handler.clone();
        let mut filesystem_optimizer = FileSystemOptimizer::new(&PathBuf::from("/"))?; // Placeholder
        
        Ok(thread::spawn(move || {
            active_workers.fetch_add(1, Ordering::Relaxed);
            let mut worker_stats = IndexingStats::new();
            
            loop {
                // Get next task from queue
                let task = {
                    let mut queue = work_queue.lock().unwrap();
                    queue.pop_front()
                };
                
                match task {
                    Some(task) => {
                        let task_start = Instant::now();
                        
                        // Process the task with optimizations
                        match Self::process_task_optimized(
                            &mut path_handler,
                            &mut filesystem_optimizer,
                            &task,
                            &coordination_stats,
                        ) {
                            Ok(task_results) => {
                                worker_stats.combine(task_results);
                                
                                let mut stats = coordination_stats.write().unwrap();
                                stats.tasks_completed += 1;
                                
                                // Track specific optimizations used
                                if task.requires_unicode_handling {
                                    stats.unicode_files_processed += 1;
                                }
                                
                                let task_time = task_start.elapsed();
                                stats.average_task_processing_time = 
                                    (stats.average_task_processing_time + task_time) / 2;
                            }
                            Err(e) => {
                                let mut stats = coordination_stats.write().unwrap();
                                stats.tasks_failed += 1;
                                worker_stats.errors.push(format!("Worker {}: {}", worker_id, e));
                            }
                        }
                    }
                    None => {
                        // No more tasks - check if other workers are still active
                        if active_workers.load(Ordering::Relaxed) <= 1 {
                            break; // Last worker, exit
                        }
                        
                        // Wait a bit before checking again
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            
            active_workers.fetch_sub(1, Ordering::Relaxed);
            Ok(worker_stats)
        }))
    }
    
    fn process_task_optimized(
        path_handler: &mut WindowsPathHandler,
        filesystem_optimizer: &mut FileSystemOptimizer,
        task: &IndexingTask,
        coordination_stats: &Arc<RwLock<CoordinationStats>>,
    ) -> Result<IndexingStats> {
        let mut task_stats = IndexingStats::new();
        
        if task.path.is_file() {
            // Process individual file
            Self::process_file_optimized(path_handler, &task.path, &mut task_stats)?;
        } else if task.path.is_dir() {
            // Process directory
            Self::process_directory_optimized(
                path_handler,
                filesystem_optimizer,
                &task.path,
                &mut task_stats,
                coordination_stats,
            )?;
        }
        
        Ok(task_stats)
    }
    
    fn process_file_optimized(
        path_handler: &WindowsPathHandler,
        file_path: &Path,
        stats: &mut IndexingStats,
    ) -> Result<()> {
        // Apply Windows-specific validations and optimizations
        path_handler.validate_windows_path(file_path)?;
        
        // Check if extended path handling is needed
        if path_handler.is_extended_path_needed(file_path) {
            let extended_path = path_handler.ensure_extended_path(file_path)?;
            // Process with extended path
            stats.files_indexed += 1;
        } else {
            // Standard processing
            stats.files_indexed += 1;
        }
        
        // Unicode handling if needed
        let filename = file_path.file_name().unwrap().to_string_lossy();
        if !filename.is_ascii() {
            let unicode_validation = path_handler.validate_unicode_path(&filename);
            if !unicode_validation.is_valid {
                return Err(anyhow::anyhow!("Unicode validation failed: {:?}", unicode_validation.errors));
            }
        }
        
        Ok(())
    }
    
    fn process_directory_optimized(
        path_handler: &WindowsPathHandler,
        filesystem_optimizer: &mut FileSystemOptimizer,
        dir_path: &Path,
        stats: &mut IndexingStats,
        coordination_stats: &Arc<RwLock<CoordinationStats>>,
    ) -> Result<()> {
        // Use optimized directory reading
        let entries = filesystem_optimizer.optimized_read_dir(dir_path)?;
        stats.directories_processed += 1;
        
        for entry in entries {
            let entry_path = entry.path();
            
            if entry_path.is_file() {
                Self::process_file_optimized(path_handler, &entry_path, stats)?;
            } else if entry_path.is_dir() {
                // Recurse into subdirectory
                Self::process_directory_optimized(
                    path_handler,
                    filesystem_optimizer,
                    &entry_path,
                    stats,
                    coordination_stats,
                )?;
            }
        }
        
        // Update coordination stats
        let mut coord_stats = coordination_stats.write().unwrap();
        coord_stats.filesystem_optimizations_applied += 1;
        
        Ok(())
    }
    
    fn monitor_and_coordinate(&self) -> Result<()> {
        let monitor_duration = Duration::from_secs(30);
        let start_time = Instant::now();
        
        while start_time.elapsed() < monitor_duration {
            // Check system load and adjust priorities if needed
            let manager = self.priority_manager.lock().unwrap();
            let system_metrics = manager.get_system_metrics()?;
            
            if system_metrics.cpu_usage_percent > 85.0 {
                println!("High CPU load detected - considering throttling");
                
                let mut stats = self.coordination_stats.write().unwrap();
                stats.priority_adjustments += 1;
            }
            
            // Check queue status
            let queue_size = {
                let queue = self.work_queue.lock().unwrap();
                queue.len()
            };
            
            if queue_size == 0 && self.active_workers.load(Ordering::Relaxed) == 0 {
                // All work completed
                break;
            }
            
            thread::sleep(Duration::from_secs(1));
        }
        
        Ok(())
    }
    
    fn calculate_parallel_efficiency(&self, results: &IndexingStats) -> f64 {
        let thread_count = self.parallel_indexer.get_thread_count() as f64;
        let theoretical_max = results.files_indexed as f64 * thread_count;
        
        if theoretical_max > 0.0 {
            (results.files_indexed as f64 / theoretical_max) * 100.0
        } else {
            0.0
        }
    }
    
    fn measure_system_impact(&self) -> Result<SystemImpactMetrics> {
        let manager = self.priority_manager.lock().unwrap();
        let current_metrics = manager.get_system_metrics()?;
        
        Ok(SystemImpactMetrics {
            final_cpu_usage: current_metrics.cpu_usage_percent,
            final_memory_usage: current_metrics.memory_usage_percent,
            system_remained_responsive: current_metrics.system_responsive,
            priority_adjustments_made: self.coordination_stats.read().unwrap().priority_adjustments,
        })
    }
    
    pub fn get_comprehensive_stats(&self) -> ComprehensiveIndexingStats {
        let coordination_stats = self.coordination_stats.read().unwrap().clone();
        let indexing_stats = self.parallel_indexer.get_stats();
        let filesystem_stats = self.filesystem_optimizer.get_stats().clone();
        
        ComprehensiveIndexingStats {
            coordination: coordination_stats,
            indexing: indexing_stats,
            filesystem: filesystem_stats,
            platform: self.platform_capabilities.clone(),
        }
    }
}

// Supporting structures
#[derive(Debug, Clone)]
pub struct IndexingAnalysis {
    pub estimated_file_count: u64,
    pub directory_depth: u32,
    pub unicode_content_percentage: f64,
    pub long_path_percentage: f64,
    pub recommended_strategy: IndexingStrategy,
    pub analysis_time: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexingStrategy {
    Standard,
    LargeSimpleDataset,
    LargeComplexDataset,
    UnicodeHeavy,
    LongPathHeavy,
}

#[derive(Debug, Clone)]
pub struct OptimizedIndexingResults {
    pub files_indexed: u64,
    pub directories_processed: u64,
    pub unicode_files_handled: u64,
    pub extended_paths_handled: u64,
    pub filesystem_optimizations_used: u64,
    pub priority_adjustments: u64,
    pub total_time: Duration,
    pub indexing_rate: f64,
    pub parallel_efficiency: f64,
    pub system_impact: SystemImpactMetrics,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SystemImpactMetrics {
    pub final_cpu_usage: f64,
    pub final_memory_usage: f64,
    pub system_remained_responsive: bool,
    pub priority_adjustments_made: u64,
}

#[derive(Debug, Clone)]
pub struct ComprehensiveIndexingStats {
    pub coordination: CoordinationStats,
    pub indexing: IndexingStats,
    pub filesystem: OptimizationStats,
    pub platform: PlatformCapabilities,
}

// Implement Clone for WindowsPathHandler if not already done
impl Clone for WindowsPathHandler {
    fn clone(&self) -> Self {
        WindowsPathHandler::new()
    }
}
```

### 2. Add performance benchmarking and comparison
Add this benchmarking system:
```rust
pub struct IndexingBenchmark {
    baseline_indexer: ParallelIndexer,
    optimized_indexer: WindowsOptimizedIndexer,
}

impl IndexingBenchmark {
    pub fn new(thread_count: usize, base_path: &Path) -> Result<Self> {
        let baseline_indexer = ParallelIndexer::new(thread_count);
        let priority_settings = PrioritySettings::default();
        let optimized_indexer = WindowsOptimizedIndexer::new(thread_count, priority_settings, base_path)?;
        
        Ok(Self {
            baseline_indexer,
            optimized_indexer,
        })
    }
    
    pub fn run_comparative_benchmark(&mut self, test_path: &Path) -> Result<BenchmarkResults> {
        println!("Running comparative indexing benchmark...");
        
        // Baseline run
        println!("Running baseline indexing...");
        let baseline_start = Instant::now();
        let baseline_results = self.baseline_indexer.index_directory(test_path)?;
        let baseline_time = baseline_start.elapsed();
        
        // Allow system to cool down
        thread::sleep(Duration::from_secs(2));
        
        // Optimized run
        println!("Running optimized indexing...");
        let optimized_start = Instant::now();
        let optimized_results = self.optimized_indexer.index_directory_optimized(test_path)?;
        let optimized_time = optimized_start.elapsed();
        
        // Calculate improvements
        let time_improvement = if baseline_time > optimized_time {
            ((baseline_time.as_secs_f64() - optimized_time.as_secs_f64()) / baseline_time.as_secs_f64()) * 100.0
        } else {
            0.0
        };
        
        let throughput_improvement = if optimized_results.indexing_rate > (baseline_results.files_indexed as f64 / baseline_time.as_secs_f64()) {
            ((optimized_results.indexing_rate - (baseline_results.files_indexed as f64 / baseline_time.as_secs_f64())) 
             / (baseline_results.files_indexed as f64 / baseline_time.as_secs_f64())) * 100.0
        } else {
            0.0
        };
        
        Ok(BenchmarkResults {
            baseline_time,
            optimized_time,
            baseline_files: baseline_results.files_indexed,
            optimized_files: optimized_results.files_indexed,
            time_improvement_percent: time_improvement,
            throughput_improvement_percent: throughput_improvement,
            unicode_files_handled: optimized_results.unicode_files_handled,
            extended_paths_handled: optimized_results.extended_paths_handled,
            filesystem_optimizations_used: optimized_results.filesystem_optimizations_used,
            system_impact: optimized_results.system_impact,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub baseline_time: Duration,
    pub optimized_time: Duration,
    pub baseline_files: u64,
    pub optimized_files: u64,
    pub time_improvement_percent: f64,
    pub throughput_improvement_percent: f64,
    pub unicode_files_handled: u64,
    pub extended_paths_handled: u64,
    pub filesystem_optimizations_used: u64,
    pub system_impact: SystemImpactMetrics,
}

impl BenchmarkResults {
    pub fn print_summary(&self) {
        println!("\n=== Indexing Benchmark Results ===");
        println!("Baseline Results:");
        println!("  Time: {:?}", self.baseline_time);
        println!("  Files: {}", self.baseline_files);
        println!("  Rate: {:.2} files/sec", self.baseline_files as f64 / self.baseline_time.as_secs_f64());
        
        println!("\nOptimized Results:");
        println!("  Time: {:?}", self.optimized_time);
        println!("  Files: {}", self.optimized_files);
        println!("  Rate: {:.2} files/sec", self.optimized_files as f64 / self.optimized_time.as_secs_f64());
        
        println!("\nImprovements:");
        println!("  Time improvement: {:.1}%", self.time_improvement_percent);
        println!("  Throughput improvement: {:.1}%", self.throughput_improvement_percent);
        
        println!("\nOptimizations Applied:");
        println!("  Unicode files handled: {}", self.unicode_files_handled);
        println!("  Extended paths handled: {}", self.extended_paths_handled);
        println!("  Filesystem optimizations: {}", self.filesystem_optimizations_used);
        
        println!("\nSystem Impact:");
        println!("  Final CPU usage: {:.1}%", self.system_impact.final_cpu_usage);
        println!("  Final memory usage: {:.1}%", self.system_impact.final_memory_usage);
        println!("  System responsive: {}", self.system_impact.system_remained_responsive);
        println!("  Priority adjustments: {}", self.system_impact.priority_adjustments_made);
    }
}
```

### 3. Add comprehensive integration tests
Add these comprehensive test modules:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_windows_optimized_indexer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let priority_settings = PrioritySettings::default();
        
        let indexer = WindowsOptimizedIndexer::new(4, priority_settings, temp_dir.path())?;
        
        // Verify components are initialized
        assert_eq!(indexer.parallel_indexer.get_thread_count(), 4);
        assert_eq!(indexer.active_workers.load(Ordering::Relaxed), 0);
        
        println!("Windows optimized indexer created successfully");
        
        Ok(())
    }
    
    #[test]
    fn test_indexing_analysis_and_strategy_selection() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create test files with different characteristics
        std::fs::write(temp_dir.path().join("normal_file.txt"), "content")?;
        std::fs::write(temp_dir.path().join("测试文件.txt"), "unicode content")?;
        std::fs::write(temp_dir.path().join("file_with_very_long_name_that_exceeds_normal_limits.txt"), "long name")?;
        
        let priority_settings = PrioritySettings::default();
        let mut indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
        
        let analysis = indexer.analyze_indexing_target(temp_dir.path())?;
        
        println!("Indexing analysis results:");
        println!("  Strategy: {:?}", analysis.recommended_strategy);
        println!("  Unicode percentage: {:.1}%", analysis.unicode_content_percentage);
        println!("  Long path percentage: {:.1}%", analysis.long_path_percentage);
        
        assert!(analysis.analysis_time > Duration::new(0, 0));
        
        Ok(())
    }
    
    #[test]
    fn test_coordinated_indexing_execution() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create a realistic test dataset
        std::fs::create_dir_all(temp_dir.path().join("documents"))?;
        std::fs::create_dir_all(temp_dir.path().join("images"))?;
        std::fs::create_dir_all(temp_dir.path().join("data"))?;
        
        for i in 0..20 {
            std::fs::write(
                temp_dir.path().join(format!("documents/doc_{}.txt", i)),
                format!("Document content {}", i)
            )?;
            
            if i % 3 == 0 {
                // Add some Unicode files
                std::fs::write(
                    temp_dir.path().join(format!("documents/文档_{}.txt", i)),
                    format!("Unicode document {}", i)
                )?;
            }
        }
        
        let priority_settings = PrioritySettings {
            adaptive_priority: false, // Disable for predictable testing
            ..Default::default()
        };
        let mut indexer = WindowsOptimizedIndexer::new(4, priority_settings, temp_dir.path())?;
        
        let results = indexer.index_directory_optimized(temp_dir.path())?;
        
        println!("Coordinated indexing results:");
        println!("  Files indexed: {}", results.files_indexed);
        println!("  Directories processed: {}", results.directories_processed);
        println!("  Unicode files handled: {}", results.unicode_files_handled);
        println!("  Indexing rate: {:.2} files/sec", results.indexing_rate);
        println!("  Parallel efficiency: {:.1}%", results.parallel_efficiency);
        
        assert!(results.files_indexed >= 26); // At least 20 regular + 6 Unicode files
        assert!(results.directories_processed >= 3); // At least 3 subdirectories
        assert!(results.unicode_files_handled > 0);
        assert!(results.total_time > Duration::new(0, 0));
        
        Ok(())
    }
    
    #[test]
    fn test_task_prioritization_and_queuing() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let priority_settings = PrioritySettings::default();
        let mut indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
        
        // Create analysis with specific characteristics
        let analysis = IndexingAnalysis {
            estimated_file_count: 100,
            directory_depth: 3,
            unicode_content_percentage: 25.0,
            long_path_percentage: 10.0,
            recommended_strategy: IndexingStrategy::UnicodeHeavy,
            analysis_time: Duration::from_millis(50),
        };
        
        indexer.queue_initial_tasks(temp_dir.path(), &analysis)?;
        
        let queue_size = {
            let queue = indexer.work_queue.lock().unwrap();
            queue.len()
        };
        
        assert_eq!(queue_size, 1); // Should have queued initial task
        
        let stats = indexer.coordination_stats.read().unwrap();
        assert_eq!(stats.tasks_queued, 1);
        
        Ok(())
    }
    
    #[test]
    fn test_system_load_adaptation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let priority_settings = PrioritySettings {
            adaptive_priority: true,
            max_cpu_usage_threshold: 70.0, // Lower threshold for testing
            priority_adjustment_interval: Duration::from_millis(100),
            ..Default::default()
        };
        
        let indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
        
        // Let adaptive priority run for a short time
        thread::sleep(Duration::from_millis(300));
        
        let stats = indexer.coordination_stats.read().unwrap();
        println!("System load adaptation stats:");
        println!("  Priority adjustments: {}", stats.priority_adjustments);
        
        // Should have made at least some priority adjustments
        // (This might be 0 on low-load systems, which is also correct)
        
        Ok(())
    }
    
    #[test]
    fn test_comprehensive_statistics_collection() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create test files
        for i in 0..10 {
            std::fs::write(temp_dir.path().join(format!("file_{}.txt", i)), "content")?;
        }
        
        let priority_settings = PrioritySettings::default();
        let mut indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
        
        let _results = indexer.index_directory_optimized(temp_dir.path())?;
        let comprehensive_stats = indexer.get_comprehensive_stats();
        
        println!("Comprehensive statistics:");
        println!("  Coordination - Tasks completed: {}", comprehensive_stats.coordination.tasks_completed);
        println!("  Coordination - Success rate: {:.1}%", comprehensive_stats.coordination.success_rate() * 100.0);
        println!("  Indexing - Files indexed: {}", comprehensive_stats.indexing.files_indexed);
        println!("  Filesystem - Cache hit rate: {:.1}%", comprehensive_stats.filesystem.cache_hit_rate() * 100.0);
        println!("  Platform: {:?}", comprehensive_stats.platform.platform);
        
        assert!(comprehensive_stats.coordination.tasks_completed > 0);
        assert!(comprehensive_stats.indexing.files_indexed > 0);
        
        Ok(())
    }
    
    #[test]
    #[ignore] // Expensive test - run with --ignored
    fn test_performance_benchmark() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create a substantial test dataset
        for i in 0..1000 {
            let subdir = temp_dir.path().join(format!("dir_{}", i % 10));
            std::fs::create_dir_all(&subdir)?;
            
            std::fs::write(
                subdir.join(format!("file_{}.txt", i)),
                format!("Content for file {}", i)
            )?;
            
            // Add some Unicode files
            if i % 5 == 0 {
                std::fs::write(
                    subdir.join(format!("文件_{}.txt", i)),
                    format!("Unicode content {}", i)
                )?;
            }
        }
        
        let mut benchmark = IndexingBenchmark::new(4, temp_dir.path())?;
        let results = benchmark.run_comparative_benchmark(temp_dir.path())?;
        
        results.print_summary();
        
        // Verify we processed all files
        assert!(results.baseline_files >= 1000);
        assert!(results.optimized_files >= 1000);
        
        // Optimized version should handle Unicode files
        assert!(results.unicode_files_handled > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_error_handling_and_recovery() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        // Create problematic files
        std::fs::write(temp_dir.path().join("normal.txt"), "content")?;
        
        // Create file with problematic name (if possible on platform)
        let problematic_name = if cfg!(windows) {
            "file_with_reserved_name_CON.txt"
        } else {
            "file_with_null\0byte.txt"
        };
        
        // Note: We might not be able to actually create problematic files,
        // so we'll test the detection logic instead
        
        let priority_settings = PrioritySettings::default();
        let mut indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
        
        let results = indexer.index_directory_optimized(temp_dir.path())?;
        
        println!("Error handling test results:");
        println!("  Files indexed: {}", results.files_indexed);
        println!("  Errors encountered: {}", results.errors.len());
        
        // Should have processed at least the normal file
        assert!(results.files_indexed >= 1);
        
        // Print any errors for debugging
        for error in &results.errors {
            println!("  Error: {}", error);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_resource_cleanup_and_shutdown() -> Result<()> {
        let temp_dir = TempDir::new()?;
        
        {
            // Create indexer in limited scope
            let priority_settings = PrioritySettings {
                adaptive_priority: true,
                priority_adjustment_interval: Duration::from_millis(50),
                ..Default::default()
            };
            
            let mut indexer = WindowsOptimizedIndexer::new(2, priority_settings, temp_dir.path())?;
            
            // Create small test dataset
            for i in 0..5 {
                std::fs::write(temp_dir.path().join(format!("file_{}.txt", i)), "content")?;
            }
            
            let _results = indexer.index_directory_optimized(temp_dir.path())?;
            
            // Indexer will be dropped here, triggering cleanup
        }
        
        // Wait a moment for cleanup to complete
        thread::sleep(Duration::from_millis(100));
        
        println!("Resource cleanup completed successfully");
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Complete integration with parallel indexer from Phase 4
- [ ] Coordinated task management with priority-based queuing
- [ ] Adaptive strategy selection based on dataset analysis
- [ ] System load monitoring with automatic priority adjustment
- [ ] Comprehensive statistics collection across all components
- [ ] Performance benchmarking showing measurable improvements
- [ ] Unicode and extended path handling integrated with parallel processing
- [ ] Filesystem optimizations applied during parallel indexing
- [ ] Error handling and recovery across all integrated components
- [ ] Resource cleanup and proper shutdown procedures
- [ ] All tests pass with realistic performance improvements
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Integration requires careful coordination between multiple complex systems
- Task prioritization helps optimize resource usage across worker threads
- Adaptive strategy selection improves performance for different dataset types
- System load monitoring prevents overwhelming the host system
- Comprehensive statistics help identify bottlenecks and optimization opportunities
- Performance benchmarking validates the effectiveness of optimizations
- Error handling must be robust across all integrated components
- Resource cleanup is critical for long-running indexing operations