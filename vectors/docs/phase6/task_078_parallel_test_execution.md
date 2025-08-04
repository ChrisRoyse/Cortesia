# Task 078: Create Parallel Test Execution Engine

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The Parallel Test Execution Engine runs validation tests concurrently to maximize throughput while maintaining result accuracy and resource management.

## Project Structure
```
src/
  validation/
    parallel.rs        <- Create this file
  lib.rs
```

## Task Description
Create the `ParallelExecutor` that manages concurrent test execution, load balancing, and result collection for the validation system.

## Requirements
1. Create `src/validation/parallel.rs`
2. Implement thread-safe parallel test execution
3. Add load balancing and resource management
4. Implement result aggregation and progress tracking
5. Handle error recovery and timeout management

## Expected Code Structure
```rust
use anyhow::{Result, Context};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tokio::{task::JoinHandle, time::{Duration, timeout, Instant}};
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};

use crate::validation::{
    ground_truth::{GroundTruthCase, QueryType},
    correctness::{CorrectnessValidator, ValidationResult},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub max_concurrent_tests: usize,
    pub test_timeout_seconds: u64,
    pub retry_attempts: usize,
    pub batch_size: usize,
    pub resource_monitoring: bool,
    pub progress_reporting_interval: Duration,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tests: num_cpus::get().min(8), // Cap at 8 to avoid overwhelming system
            test_timeout_seconds: 30,
            retry_attempts: 2,
            batch_size: 10,
            resource_monitoring: true,
            progress_reporting_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestJob {
    pub id: String,
    pub test_case: GroundTruthCase,
    pub priority: TestPriority,
    pub retry_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestPriority {
    High = 3,
    Normal = 2,
    Low = 1,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub job_id: String,
    pub validation_result: Option<ValidationResult>,
    pub execution_time: Duration,
    pub retry_count: usize,
    pub error: Option<String>,
    pub worker_id: usize,
}

#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub average_execution_time: Duration,
    pub peak_concurrent_jobs: usize,
    pub total_retries: usize,
    pub throughput_per_second: f64,
}

pub struct ParallelExecutor {
    config: ParallelConfig,
    job_queue: Arc<Mutex<VecDeque<TestJob>>>,
    results: Arc<Mutex<Vec<ExecutionResult>>>,
    stats: Arc<Mutex<ExecutionStats>>,
    active_workers: Arc<Mutex<usize>>,
    validator: Arc<CorrectnessValidator>,
}

impl ParallelExecutor {
    pub async fn new(config: ParallelConfig, validator: CorrectnessValidator) -> Result<Self> {
        let stats = ExecutionStats {
            total_jobs: 0,
            completed_jobs: 0,
            failed_jobs: 0,
            average_execution_time: Duration::from_millis(0),
            peak_concurrent_jobs: 0,
            total_retries: 0,
            throughput_per_second: 0.0,
        };
        
        Ok(Self {
            config,
            job_queue: Arc::new(Mutex::new(VecDeque::new())),
            results: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(stats)),
            active_workers: Arc::new(Mutex::new(0)),
            validator: Arc::new(validator),
        })
    }
    
    pub fn add_test_cases(&self, test_cases: Vec<GroundTruthCase>) -> Result<()> {
        let mut queue = self.job_queue.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        for (i, test_case) in test_cases.into_iter().enumerate() {
            let priority = Self::determine_priority(&test_case);
            let job = TestJob {
                id: format!("test_{:06}", i),
                test_case,
                priority,
                retry_count: 0,
            };
            queue.push_back(job);
        }
        
        // Sort by priority (highest first)
        let mut sorted_jobs: Vec<_> = queue.drain(..).collect();
        sorted_jobs.sort_by(|a, b| b.priority.cmp(&a.priority));
        queue.extend(sorted_jobs);
        
        stats.total_jobs = queue.len();
        
        info!("Added {} test cases to parallel execution queue", stats.total_jobs);
        
        Ok(())
    }
    
    fn determine_priority(test_case: &GroundTruthCase) -> TestPriority {
        match test_case.query_type {
            QueryType::SpecialCharacters => TestPriority::High,
            QueryType::BooleanAnd | QueryType::BooleanOr | QueryType::BooleanNot => TestPriority::High,
            QueryType::Proximity | QueryType::Phrase => TestPriority::Normal,
            QueryType::Wildcard | QueryType::Regex => TestPriority::Normal,
            QueryType::Vector | QueryType::Hybrid => TestPriority::Low,
        }
    }
    
    pub async fn execute_all(&self) -> Result<Vec<ExecutionResult>> {
        let start_time = Instant::now();
        info!("Starting parallel test execution with {} workers", self.config.max_concurrent_tests);
        
        // Start progress reporter
        let progress_handle = self.start_progress_reporter();
        
        // Start resource monitor if enabled
        let resource_handle = if self.config.resource_monitoring {
            Some(self.start_resource_monitor())
        } else {
            None
        };
        
        // Start worker tasks
        let mut worker_handles = Vec::new();
        for worker_id in 0..self.config.max_concurrent_tests {
            let handle = self.spawn_worker(worker_id).await;
            worker_handles.push(handle);
        }
        
        // Wait for all workers to complete
        for handle in worker_handles {
            if let Err(e) = handle.await {
                error!("Worker task failed: {}", e);
            }
        }
        
        // Stop background tasks
        progress_handle.abort();
        if let Some(handle) = resource_handle {
            handle.abort();
        }
        
        // Calculate final statistics
        let total_duration = start_time.elapsed();
        let mut stats = self.stats.lock().unwrap();
        stats.throughput_per_second = stats.completed_jobs as f64 / total_duration.as_secs_f64();
        
        let results = self.results.lock().unwrap().clone();
        
        info!(
            "Parallel execution complete: {}/{} jobs completed in {:.2}s ({:.1} jobs/sec)",
            stats.completed_jobs,
            stats.total_jobs,
            total_duration.as_secs_f64(),
            stats.throughput_per_second
        );
        
        Ok(results)
    }
    
    async fn spawn_worker(&self, worker_id: usize) -> JoinHandle<()> {
        let job_queue = Arc::clone(&self.job_queue);
        let results = Arc::clone(&self.results);
        let stats = Arc::clone(&self.stats);
        let active_workers = Arc::clone(&self.active_workers);
        let validator = Arc::clone(&self.validator);
        let config = self.config.clone();
        
        tokio::spawn(async move {
            debug!("Worker {} started", worker_id);
            
            loop {
                // Get next job from queue
                let job = {
                    let mut queue = job_queue.lock().unwrap();
                    queue.pop_front()
                };
                
                let job = match job {
                    Some(job) => job,
                    None => {
                        debug!("Worker {} finished - no more jobs", worker_id);
                        break;
                    }
                };
                
                // Update active worker count
                {
                    let mut active = active_workers.lock().unwrap();
                    *active += 1;
                    let mut stats = stats.lock().unwrap();
                    stats.peak_concurrent_jobs = stats.peak_concurrent_jobs.max(*active);
                }
                
                // Execute job with timeout
                let execution_start = Instant::now();
                let result = Self::execute_job_with_timeout(&validator, &job, &config).await;
                let execution_time = execution_start.elapsed();
                
                // Process result
                match result {
                    Ok(validation_result) => {
                        // Job succeeded
                        let exec_result = ExecutionResult {
                            job_id: job.id.clone(),
                            validation_result: Some(validation_result),
                            execution_time,
                            retry_count: job.retry_count,
                            error: None,
                            worker_id,
                        };
                        
                        results.lock().unwrap().push(exec_result);
                        
                        let mut stats = stats.lock().unwrap();
                        stats.completed_jobs += 1;
                        stats.average_execution_time = 
                            (stats.average_execution_time * (stats.completed_jobs - 1) as u32 + execution_time) / stats.completed_jobs as u32;
                    }
                    Err(e) => {
                        // Job failed - decide whether to retry
                        if job.retry_count < config.retry_attempts {
                            warn!("Job {} failed (attempt {}), retrying: {}", job.id, job.retry_count + 1, e);
                            
                            let retry_job = TestJob {
                                retry_count: job.retry_count + 1,
                                ..job
                            };
                            
                            job_queue.lock().unwrap().push_back(retry_job);
                            stats.lock().unwrap().total_retries += 1;
                        } else {
                            error!("Job {} failed after {} attempts: {}", job.id, job.retry_count + 1, e);
                            
                            let exec_result = ExecutionResult {
                                job_id: job.id.clone(),
                                validation_result: None,
                                execution_time,
                                retry_count: job.retry_count,
                                error: Some(e.to_string()),
                                worker_id,
                            };
                            
                            results.lock().unwrap().push(exec_result);
                            
                            let mut stats = stats.lock().unwrap();
                            stats.failed_jobs += 1;
                            stats.completed_jobs += 1; // Count failed jobs as completed
                        }
                    }
                }
                
                // Update active worker count
                {
                    let mut active = active_workers.lock().unwrap();
                    *active -= 1;
                }
            }
            
            debug!("Worker {} terminated", worker_id);
        })
    }
    
    async fn execute_job_with_timeout(
        validator: &CorrectnessValidator,
        job: &TestJob,
        config: &ParallelConfig,
    ) -> Result<ValidationResult> {
        let timeout_duration = Duration::from_secs(config.test_timeout_seconds);
        
        match timeout(timeout_duration, validator.validate(&job.test_case)).await {
            Ok(result) => result,
            Err(_) => {
                anyhow::bail!("Test case '{}' timed out after {}s", job.test_case.query, config.test_timeout_seconds)
            }
        }
    }
    
    fn start_progress_reporter(&self) -> JoinHandle<()> {
        let stats = Arc::clone(&self.stats);
        let active_workers = Arc::clone(&self.active_workers);
        let interval = self.config.progress_reporting_interval;
        
        tokio::spawn(async move {
            let mut last_completed = 0;
            
            loop {
                tokio::time::sleep(interval).await;
                
                let (completed, total, active, throughput) = {
                    let stats = stats.lock().unwrap();
                    let active = *active_workers.lock().unwrap();
                    (stats.completed_jobs, stats.total_jobs, active, stats.throughput_per_second)
                };
                
                let jobs_since_last = completed - last_completed;
                last_completed = completed;
                
                if total > 0 {
                    let progress_pct = (completed as f64 / total as f64) * 100.0;
                    info!(
                        "Progress: {}/{} ({:.1}%) completed, {} active workers, +{} jobs in last {}s",
                        completed,
                        total,
                        progress_pct,
                        active,
                        jobs_since_last,
                        interval.as_secs()
                    );
                }
                
                if completed >= total {
                    break;
                }
            }
        })
    }
    
    fn start_resource_monitor(&self) -> JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                
                // Monitor system resources
                // This would integrate with system monitoring libraries
                if let Ok(memory_usage) = Self::get_memory_usage() {
                    if memory_usage > 0.8 { // 80% memory usage
                        warn!("High memory usage detected: {:.1}%", memory_usage * 100.0);
                    }
                }
                
                if let Ok(cpu_usage) = Self::get_cpu_usage() {
                    if cpu_usage > 0.9 { // 90% CPU usage
                        warn!("High CPU usage detected: {:.1}%", cpu_usage * 100.0);
                    }
                }
            }
        })
    }
    
    fn get_memory_usage() -> Result<f64> {
        // Placeholder - would use system monitoring library
        Ok(0.5) // 50% as example
    }
    
    fn get_cpu_usage() -> Result<f64> {
        // Placeholder - would use system monitoring library
        Ok(0.6) // 60% as example
    }
    
    pub fn get_stats(&self) -> ExecutionStats {
        self.stats.lock().unwrap().clone()
    }
    
    pub fn get_failed_jobs(&self) -> Vec<ExecutionResult> {
        self.results
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.validation_result.is_none())
            .cloned()
            .collect()
    }
    
    pub fn get_successful_jobs(&self) -> Vec<ExecutionResult> {
        self.results
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.validation_result.is_some())
            .cloned()
            .collect()
    }
}

// Utility functions for batch processing
impl ParallelExecutor {
    pub async fn execute_batch(&self, batch: Vec<GroundTruthCase>) -> Result<Vec<ExecutionResult>> {
        // Clear previous results
        {
            let mut results = self.results.lock().unwrap();
            results.clear();
            let mut stats = self.stats.lock().unwrap();
            *stats = ExecutionStats {
                total_jobs: 0,
                completed_jobs: 0,
                failed_jobs: 0,
                average_execution_time: Duration::from_millis(0),
                peak_concurrent_jobs: 0,
                total_retries: 0,
                throughput_per_second: 0.0,
            };
        }
        
        self.add_test_cases(batch)?;
        self.execute_all().await
    }
    
    pub fn queue_status(&self) -> (usize, usize, usize) {
        let queue = self.job_queue.lock().unwrap();
        let results = self.results.lock().unwrap();
        let stats = self.stats.lock().unwrap();
        
        (queue.len(), results.len(), stats.total_jobs)
    }
}
```

## Integration Example
```rust
// Example usage in validation runner
use crate::validation::parallel::{ParallelExecutor, ParallelConfig};

pub async fn run_parallel_validation(
    test_cases: Vec<GroundTruthCase>,
    validator: CorrectnessValidator,
) -> Result<Vec<ExecutionResult>> {
    let config = ParallelConfig {
        max_concurrent_tests: 4,
        test_timeout_seconds: 30,
        retry_attempts: 2,
        batch_size: 20,
        resource_monitoring: true,
        progress_reporting_interval: Duration::from_secs(5),
    };
    
    let executor = ParallelExecutor::new(config, validator).await?;
    executor.execute_batch(test_cases).await
}
```

## Success Criteria
- ParallelExecutor manages concurrent test execution efficiently
- Load balancing distributes work evenly across workers
- Progress tracking provides real-time visibility
- Resource monitoring prevents system overload
- Error recovery and retry logic work correctly
- Results are collected and aggregated properly

## Time Limit
15 minutes maximum