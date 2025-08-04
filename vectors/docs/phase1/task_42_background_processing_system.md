# Task 42: Background Processing System

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 41 completed
**Input Files:**
- C:/code/LLMKG/vectors/tantivy_search/src/connection_pool.rs
- C:/code/LLMKG/vectors/tantivy_search/src/search.rs
- C:/code/LLMKG/vectors/tantivy_search/Cargo.toml

## Complete Context (For AI with ZERO Knowledge)

You are implementing a **background processing system** for the Tantivy-based search system. This system handles maintenance tasks like index optimization, connection pool cleanup, cache eviction, and health monitoring without blocking search operations.

**What is Background Processing?** A system that performs long-running or periodic tasks in separate threads/tasks while the main application continues serving requests. Essential for production systems that need continuous maintenance.

**System Context:** After task 41, we have connection pooling for efficient resource management. This task adds a background task scheduler that keeps the system healthy and optimized.

**This Task:** Creates a BackgroundProcessor that manages scheduled tasks, monitors system health, and performs maintenance operations asynchronously.

## Exact Steps (6 minutes implementation)

### Step 1: Add Background Processing Dependencies (1 minute)
Edit `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml`, add to `[dependencies]` section:
```toml
# Background task processing
tokio-cron-scheduler = "0.9.4"
tracing = "0.1.40"
tracing-subscriber = "0.3.18"
```

### Step 2: Create Background Processor Module (2.5 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/background_processor.rs`:
```rust
use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info, warn};

use crate::connection_pool::ConnectionPool;
use crate::search::SearchEngine;

#[derive(Debug, Clone)]
pub enum BackgroundTask {
    CleanupConnections,
    OptimizeIndex,
    HealthCheck,
    CacheEviction,
    MetricsCollection,
}

#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub cleanup_interval: Duration,
    pub optimization_interval: Duration,
    pub health_check_interval: Duration,
    pub metrics_interval: Duration,
    pub max_concurrent_tasks: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            optimization_interval: Duration::from_secs(3600), // 1 hour
            health_check_interval: Duration::from_secs(60), // 1 minute
            metrics_interval: Duration::from_secs(30), // 30 seconds
            max_concurrent_tasks: 5,
        }
    }
}

#[derive(Debug)]
pub struct TaskStats {
    pub task_type: BackgroundTask,
    pub last_run: Option<Instant>,
    pub run_count: u64,
    pub error_count: u64,
    pub avg_duration: Duration,
}

pub struct BackgroundProcessor {
    config: ProcessorConfig,
    scheduler: Arc<RwLock<Option<JobScheduler>>>,
    task_handles: Arc<RwLock<Vec<JoinHandle<()>>>>,
    task_stats: Arc<RwLock<Vec<TaskStats>>>,
    task_sender: mpsc::UnboundedSender<BackgroundTask>,
    task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<BackgroundTask>>>>,
    is_running: Arc<RwLock<bool>>,
}
```

### Step 3: Implement Background Processor Core (1.5 minutes)
Continue in `src/background_processor.rs`:
```rust
impl BackgroundProcessor {
    pub fn new(config: ProcessorConfig) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        
        Self {
            config,
            scheduler: Arc::new(RwLock::new(None)),
            task_handles: Arc::new(RwLock::new(Vec::new())),
            task_stats: Arc::new(RwLock::new(Vec::new())),
            task_sender,
            task_receiver: Arc::new(RwLock::new(Some(task_receiver))),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&self, search_engine: Arc<SearchEngine>) -> Result<()> {
        if *self.is_running.read().await {
            return Err(anyhow!("Background processor already running"));
        }

        *self.is_running.write().await = true;
        
        // Initialize task stats
        let mut stats = self.task_stats.write().await;
        stats.extend(vec![
            TaskStats {
                task_type: BackgroundTask::CleanupConnections,
                last_run: None,
                run_count: 0,
                error_count: 0,
                avg_duration: Duration::ZERO,
            },
            TaskStats {
                task_type: BackgroundTask::OptimizeIndex,
                last_run: None,
                run_count: 0,
                error_count: 0,
                avg_duration: Duration::ZERO,
            },
            TaskStats {
                task_type: BackgroundTask::HealthCheck,
                last_run: None,
                run_count: 0,
                error_count: 0,
                avg_duration: Duration::ZERO,
            },
        ]);
        drop(stats);

        // Start scheduler
        let scheduler = JobScheduler::new().await?;
        
        // Schedule cleanup task
        let cleanup_job = self.create_cleanup_job(Arc::clone(&search_engine))?;
        scheduler.add(cleanup_job).await?;
        
        // Schedule health check
        let health_job = self.create_health_job(Arc::clone(&search_engine))?;
        scheduler.add(health_job).await?;
        
        scheduler.start().await?;
        *self.scheduler.write().await = Some(scheduler);

        // Start task processor
        self.start_task_processor(search_engine).await;
        
        info!("Background processor started successfully");
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        *self.is_running.write().await = false;
        
        if let Some(scheduler) = self.scheduler.write().await.take() {
            scheduler.shutdown().await?;
        }

        // Cancel all running tasks
        let mut handles = self.task_handles.write().await;
        for handle in handles.drain(..) {
            handle.abort();
        }

        info!("Background processor stopped");
        Ok(())
    }

    pub async fn get_stats(&self) -> Vec<TaskStats> {
        self.task_stats.read().await.clone()
    }

    pub fn schedule_task(&self, task: BackgroundTask) -> Result<()> {
        self.task_sender.send(task).map_err(|e| anyhow!("Failed to schedule task: {}", e))
    }
}
```

### Step 4: Implement Task Execution Methods (1 minute)
Continue in `src/background_processor.rs`:
```rust
impl BackgroundProcessor {
    fn create_cleanup_job(&self, search_engine: Arc<SearchEngine>) -> Result<Job> {
        let task_sender = self.task_sender.clone();
        
        Job::new_async("0 */5 * * * *", move |_uuid, _l| {
            let sender = task_sender.clone();
            Box::pin(async move {
                if let Err(e) = sender.send(BackgroundTask::CleanupConnections) {
                    error!("Failed to schedule cleanup task: {}", e);
                }
            })
        })
    }

    fn create_health_job(&self, search_engine: Arc<SearchEngine>) -> Result<Job> {
        let task_sender = self.task_sender.clone();
        
        Job::new_async("0 * * * * *", move |_uuid, _l| {
            let sender = task_sender.clone();
            Box::pin(async move {
                if let Err(e) = sender.send(BackgroundTask::HealthCheck) {
                    error!("Failed to schedule health check: {}", e);
                }
            })
        })
    }

    async fn start_task_processor(&self, search_engine: Arc<SearchEngine>) {
        let mut receiver = self.task_receiver.write().await.take()
            .expect("Task receiver should be available");
        
        let stats = Arc::clone(&self.task_stats);
        let is_running = Arc::clone(&self.is_running);
        
        let handle = tokio::spawn(async move {
            while *is_running.read().await {
                match receiver.recv().await {
                    Some(task) => {
                        let start_time = Instant::now();
                        let result = Self::execute_task(task.clone(), Arc::clone(&search_engine)).await;
                        let duration = start_time.elapsed();
                        
                        Self::update_task_stats(&stats, task, result.is_ok(), duration).await;
                        
                        if let Err(e) = result {
                            error!("Background task failed: {:?} - {}", task, e);
                        }
                    }
                    None => break,
                }
            }
        });
        
        self.task_handles.write().await.push(handle);
    }

    async fn execute_task(task: BackgroundTask, search_engine: Arc<SearchEngine>) -> Result<()> {
        match task {
            BackgroundTask::CleanupConnections => {
                info!("Executing connection cleanup");
                // Connection pool cleanup would be called here
                tokio::time::sleep(Duration::from_millis(100)).await; // Simulate work
                Ok(())
            }
            BackgroundTask::HealthCheck => {
                info!("Executing health check");
                // Health check logic would be implemented here
                tokio::time::sleep(Duration::from_millis(50)).await; // Simulate work
                Ok(())
            }
            BackgroundTask::OptimizeIndex => {
                info!("Executing index optimization");
                tokio::time::sleep(Duration::from_millis(200)).await; // Simulate work
                Ok(())
            }
            _ => Ok(()),
        }
    }

    async fn update_task_stats(
        stats: &Arc<RwLock<Vec<TaskStats>>>,
        task: BackgroundTask,
        success: bool,
        duration: Duration,
    ) {
        let mut stats = stats.write().await;
        if let Some(stat) = stats.iter_mut().find(|s| matches!(s.task_type, task)) {
            stat.last_run = Some(Instant::now());
            stat.run_count += 1;
            if !success {
                stat.error_count += 1;
            }
            // Update average duration (simple moving average)
            let total_duration = stat.avg_duration * (stat.run_count - 1) as u32 + duration;
            stat.avg_duration = total_duration / stat.run_count as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchEngine;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_background_processor_lifecycle() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let search_engine = Arc::new(SearchEngine::new(temp_dir.path())?);
        
        let config = ProcessorConfig::default();
        let processor = BackgroundProcessor::new(config);
        
        // Test start
        processor.start(Arc::clone(&search_engine)).await?;
        
        // Schedule a task
        processor.schedule_task(BackgroundTask::HealthCheck)?;
        
        // Wait for task execution
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Check stats
        let stats = processor.get_stats().await;
        assert!(!stats.is_empty());
        
        // Test stop
        processor.stop().await?;
        
        Ok(())
    }
}
```

## Verification Steps (2 minutes)
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test background_processor
cargo test test_background_processor_lifecycle
```

## Success Validation Checklist
- [ ] File exists: `src/background_processor.rs` with BackgroundProcessor and TaskStats
- [ ] Dependencies added: `tokio-cron-scheduler`, `tracing`, `tracing-subscriber`
- [ ] Background processor can start/stop successfully
- [ ] Tasks can be scheduled and executed asynchronously
- [ ] Task statistics are tracked correctly (run count, errors, duration)
- [ ] Command `cargo check` completes without errors
- [ ] Test `test_background_processor_lifecycle` passes
- [ ] Cron scheduler integrates properly with task execution

## Files Created For Next Task
1. **C:/code/LLMKG/vectors/tantivy_search/src/background_processor.rs** - Background task system with scheduling
2. **Enhanced system** - Now supports continuous maintenance operations

**Next Task (Task 43)** will implement comprehensive error types for robust error handling across all system components.