# Task 018: Concurrent Benchmark Implementation

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 008 and 017 (PerformanceBenchmark and EnhancedPerformanceMetrics). The concurrent benchmark simulates realistic multi-user load patterns with proper thread management and resource contention testing.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement concurrent user simulation with realistic load patterns, thread pool management, deadlock detection, and scalability testing with increasing concurrent loads. This provides critical insights into system behavior under real-world concurrent access patterns.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. Implement concurrent user simulation with realistic load patterns
3. Thread pool management and resource contention testing
4. Deadlock detection and performance isolation
5. Scalability testing with increasing concurrent loads
6. Connection pooling and resource sharing simulation
7. Windows-compatible async patterns

## Expected Code Structure to Add
```rust
use tokio::sync::{Semaphore, RwLock};
use tokio::time::{sleep, timeout, Duration};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use rayon::prelude::*;
use futures::future::join_all;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentBenchmark {
    search_system: Arc<UnifiedSearchSystem>,
    max_concurrent_users: usize,
    ramp_up_duration: Duration,
    connection_pool_size: usize,
    deadlock_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentTestResult {
    pub concurrent_users: usize,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub timeout_requests: usize,
    pub deadlock_detected: bool,
    pub average_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub min_response_time_ms: f64,
    pub requests_per_second: f64,
    pub thread_performance: Vec<ThreadPerformance>,
    pub resource_contention_score: f64, // 0-100, higher = more contention
    pub scalability_efficiency: f64, // How well performance scales with users
    pub connection_pool_usage: ConnectionPoolStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPerformance {
    pub thread_id: usize,
    pub requests_completed: usize,
    pub average_latency_ms: f64,
    pub max_latency_ms: f64,
    pub errors_encountered: usize,
    pub cpu_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    pub pool_size: usize,
    pub peak_usage: usize,
    pub average_usage: f64,
    pub wait_time_ms: f64,
    pub connection_timeouts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityReport {
    pub test_results: Vec<ConcurrentTestResult>,
    pub optimal_concurrent_users: usize,
    pub scalability_breaking_point: usize,
    pub linear_scaling_range: (usize, usize),
    pub efficiency_curve: Vec<(usize, f64)>, // (users, efficiency_percent)
    pub recommendation: ScalabilityRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalabilityRecommendation {
    IncreaseResources,
    OptimizeCode,
    AcceptablePerformance,
    ReduceLoad,
}

impl ConcurrentBenchmark {
    pub async fn new<P: AsRef<Path>>(
        text_index_path: P,
        vector_db_path: &str,
        max_concurrent_users: usize,
    ) -> Result<Self> {
        let search_system = Arc::new(
            UnifiedSearchSystem::new(text_index_path.as_ref(), vector_db_path)
                .await
                .context("Failed to initialize search system for concurrent testing")?
        );
        
        Ok(Self {
            search_system,
            max_concurrent_users,
            ramp_up_duration: Duration::from_secs(30),
            connection_pool_size: max_concurrent_users * 2, // 2x for safety
            deadlock_timeout: Duration::from_secs(30),
        })
    }
    
    pub async fn run_concurrent_test(
        &self,
        queries: Vec<String>,
        target_users: usize,
        test_duration: Duration,
    ) -> Result<ConcurrentTestResult> {
        println!("Starting concurrent test with {} users for {:?}...", target_users, test_duration);
        
        if queries.is_empty() {
            return Err(anyhow::anyhow!("No queries provided for concurrent test"));
        }
        
        let semaphore = Arc::new(Semaphore::new(target_users));
        let connection_pool = Arc::new(ConnectionPool::new(self.connection_pool_size));
        let deadlock_detector = Arc::new(DeadlockDetector::new(self.deadlock_timeout));
        
        // Shared counters for thread-safe statistics
        let total_requests = Arc::new(AtomicUsize::new(0));
        let successful_requests = Arc::new(AtomicUsize::new(0));
        let failed_requests = Arc::new(AtomicUsize::new(0));
        let timeout_requests = Arc::new(AtomicUsize::new(0));
        let test_start = std::time::Instant::now();
        
        // Response time tracking
        let response_times = Arc::new(RwLock::new(Vec::new()));
        let thread_performances = Arc::new(RwLock::new(Vec::new()));
        
        // Create concurrent user tasks
        let mut user_tasks = Vec::new();
        
        for user_id in 0..target_users {
            let queries_clone = queries.clone();
            let semaphore_clone = Arc::clone(&semaphore);
            let connection_pool_clone = Arc::clone(&connection_pool);
            let deadlock_detector_clone = Arc::clone(&deadlock_detector);
            let search_system_clone = Arc::clone(&self.search_system);
            let total_requests_clone = Arc::clone(&total_requests);
            let successful_requests_clone = Arc::clone(&successful_requests);
            let failed_requests_clone = Arc::clone(&failed_requests);
            let timeout_requests_clone = Arc::clone(&timeout_requests);
            let response_times_clone = Arc::clone(&response_times);
            
            let user_task = tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                
                let mut user_performance = ThreadPerformance {
                    thread_id: user_id,
                    requests_completed: 0,
                    average_latency_ms: 0.0,
                    max_latency_ms: 0.0,
                    errors_encountered: 0,
                    cpu_time_ms: 0.0,
                };
                
                let mut user_response_times = Vec::new();
                let user_start = std::time::Instant::now();
                
                while user_start.elapsed() < test_duration {
                    // Select random query
                    let query_index = user_id % queries_clone.len();
                    let query = &queries_clone[query_index];
                    
                    // Acquire connection from pool
                    let _connection = match connection_pool_clone.acquire().await {
                        Ok(conn) => conn,
                        Err(_) => {
                            timeout_requests_clone.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    
                    let request_start = std::time::Instant::now();
                    total_requests_clone.fetch_add(1, Ordering::Relaxed);
                    
                    // Execute search with deadlock detection
                    let search_result = timeout(
                        deadlock_detector_clone.timeout_duration(),
                        search_system_clone.search_hybrid(query, SearchMode::Hybrid)
                    ).await;
                    
                    let request_duration = request_start.elapsed();
                    let latency_ms = request_duration.as_millis() as f64;
                    
                    match search_result {
                        Ok(Ok(_results)) => {
                            successful_requests_clone.fetch_add(1, Ordering::Relaxed);
                            user_response_times.push(latency_ms);
                            user_performance.requests_completed += 1;
                            
                            if latency_ms > user_performance.max_latency_ms {
                                user_performance.max_latency_ms = latency_ms;
                            }
                        }
                        Ok(Err(_)) => {
                            failed_requests_clone.fetch_add(1, Ordering::Relaxed);
                            user_performance.errors_encountered += 1;
                        }
                        Err(_) => {
                            timeout_requests_clone.fetch_add(1, Ordering::Relaxed);
                            deadlock_detector_clone.report_timeout().await;
                            user_performance.errors_encountered += 1;
                        }
                    }
                    
                    // Add response time to global collection
                    {
                        let mut times = response_times_clone.write().await;
                        times.push(latency_ms);
                    }
                    
                    // Small delay to simulate realistic user behavior
                    sleep(Duration::from_millis(50 + (user_id % 100) as u64)).await;
                }
                
                // Calculate user performance statistics
                if !user_response_times.is_empty() {
                    user_performance.average_latency_ms = 
                        user_response_times.iter().sum::<f64>() / user_response_times.len() as f64;
                }
                user_performance.cpu_time_ms = user_start.elapsed().as_millis() as f64;
                
                user_performance
            });
            
            user_tasks.push(user_task);
        }
        
        // Wait for all user tasks to complete
        let thread_results = join_all(user_tasks).await;
        let mut thread_performances = Vec::new();
        
        for result in thread_results {
            match result {
                Ok(perf) => thread_performances.push(perf),
                Err(e) => println!("Thread task failed: {}", e),
            }
        }
        
        let test_duration_actual = test_start.elapsed();
        let total_req = total_requests.load(Ordering::Relaxed);
        let successful_req = successful_requests.load(Ordering::Relaxed);
        let failed_req = failed_requests.load(Ordering::Relaxed);
        let timeout_req = timeout_requests.load(Ordering::Relaxed);
        
        // Calculate statistics
        let response_times_vec = response_times.read().await.clone();
        let (avg_response, max_response, min_response) = if !response_times_vec.is_empty() {
            let sum: f64 = response_times_vec.iter().sum();
            let avg = sum / response_times_vec.len() as f64;
            let max = response_times_vec.iter().cloned().fold(0.0, f64::max);
            let min = response_times_vec.iter().cloned().fold(f64::MAX, f64::min);
            (avg, max, min)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        let requests_per_second = total_req as f64 / test_duration_actual.as_secs_f64();
        let resource_contention_score = self.calculate_resource_contention(&thread_performances);
        let scalability_efficiency = self.calculate_scalability_efficiency(
            target_users, 
            requests_per_second, 
            avg_response
        );
        
        let connection_pool_stats = connection_pool.get_stats().await;
        let deadlock_detected = deadlock_detector.deadlock_detected().await;
        
        Ok(ConcurrentTestResult {
            concurrent_users: target_users,
            total_requests: total_req,
            successful_requests: successful_req,
            failed_requests: failed_req,
            timeout_requests: timeout_req,
            deadlock_detected,
            average_response_time_ms: avg_response,
            max_response_time_ms: max_response,
            min_response_time_ms: min_response,
            requests_per_second,
            thread_performance: thread_performances,
            resource_contention_score,
            scalability_efficiency,
            connection_pool_usage: connection_pool_stats,
        })
    }
    
    pub async fn run_scalability_test(
        &self,
        queries: Vec<String>,
        max_users: usize,
        step_size: usize,
        test_duration_per_step: Duration,
    ) -> Result<ScalabilityReport> {
        println!("Running scalability test up to {} users with step size {}...", max_users, step_size);
        
        let mut test_results = Vec::new();
        let mut current_users = step_size;
        
        while current_users <= max_users {
            println!("Testing with {} concurrent users...", current_users);
            
            let result = self.run_concurrent_test(
                queries.clone(),
                current_users,
                test_duration_per_step,
            ).await?;
            
            test_results.push(result);
            
            // Short break between tests to allow system recovery
            sleep(Duration::from_secs(5)).await;
            
            current_users += step_size;
        }
        
        let analysis = self.analyze_scalability_results(&test_results);
        
        Ok(ScalabilityReport {
            test_results,
            optimal_concurrent_users: analysis.optimal_users,
            scalability_breaking_point: analysis.breaking_point,
            linear_scaling_range: analysis.linear_range,
            efficiency_curve: analysis.efficiency_curve,
            recommendation: analysis.recommendation,
        })
    }
    
    fn calculate_resource_contention(&self, performances: &[ThreadPerformance]) -> f64 {
        if performances.is_empty() {
            return 0.0;
        }
        
        // Calculate variance in thread performance as an indicator of contention
        let avg_latency: f64 = performances.iter()
            .map(|p| p.average_latency_ms)
            .sum::<f64>() / performances.len() as f64;
        
        let variance: f64 = performances.iter()
            .map(|p| (p.average_latency_ms - avg_latency).powi(2))
            .sum::<f64>() / performances.len() as f64;
        
        // Normalize variance to 0-100 scale (higher = more contention)
        (variance.sqrt() / avg_latency * 100.0).min(100.0)
    }
    
    fn calculate_scalability_efficiency(&self, users: usize, qps: f64, latency: f64) -> f64 {
        // Ideal scaling would be linear QPS increase with constant latency
        // Efficiency = actual_qps / (ideal_qps_per_user * users) * (ideal_latency / actual_latency)
        let ideal_qps_per_user = 10.0; // Baseline assumption
        let ideal_latency = 50.0; // Baseline assumption in ms
        
        let qps_efficiency = (qps / (ideal_qps_per_user * users as f64)).min(1.0);
        let latency_efficiency = (ideal_latency / latency.max(1.0)).min(1.0);
        
        ((qps_efficiency + latency_efficiency) / 2.0 * 100.0).min(100.0)
    }
    
    fn analyze_scalability_results(&self, results: &[ConcurrentTestResult]) -> ScalabilityAnalysis {
        let mut optimal_users = 1;
        let mut breaking_point = results.len();
        let mut linear_start = 0;
        let mut linear_end = 0;
        let mut efficiency_curve = Vec::new();
        
        let mut best_efficiency = 0.0;
        let mut consecutive_degradation = 0;
        
        for (i, result) in results.iter().enumerate() {
            efficiency_curve.push((result.concurrent_users, result.scalability_efficiency));
            
            if result.scalability_efficiency > best_efficiency {
                best_efficiency = result.scalability_efficiency;
                optimal_users = result.concurrent_users;
            }
            
            // Detect linear scaling region
            if i > 0 {
                let prev_result = &results[i - 1];
                let efficiency_change = result.scalability_efficiency - prev_result.scalability_efficiency;
                
                if efficiency_change.abs() < 5.0 { // Stable efficiency
                    if linear_start == 0 {
                        linear_start = prev_result.concurrent_users;
                    }
                    linear_end = result.concurrent_users;
                } else if efficiency_change < -10.0 { // Significant degradation
                    consecutive_degradation += 1;
                    if consecutive_degradation >= 2 && breaking_point == results.len() {
                        breaking_point = result.concurrent_users;
                    }
                } else {
                    consecutive_degradation = 0;
                }
            }
        }
        
        let recommendation = if best_efficiency < 30.0 {
            ScalabilityRecommendation::OptimizeCode
        } else if breaking_point < results.len() / 2 {
            ScalabilityRecommendation::IncreaseResources
        } else if best_efficiency > 70.0 {
            ScalabilityRecommendation::AcceptablePerformance
        } else {
            ScalabilityRecommendation::ReduceLoad
        };
        
        ScalabilityAnalysis {
            optimal_users,
            breaking_point,
            linear_range: (linear_start, linear_end),
            efficiency_curve,
            recommendation,
        }
    }
}

struct ScalabilityAnalysis {
    optimal_users: usize,
    breaking_point: usize,
    linear_range: (usize, usize),
    efficiency_curve: Vec<(usize, f64)>,
    recommendation: ScalabilityRecommendation,
}

// Connection Pool Implementation
struct ConnectionPool {
    available: Arc<Semaphore>,
    pool_size: usize,
    usage_stats: Arc<RwLock<ConnectionPoolStats>>,
}

impl ConnectionPool {
    fn new(size: usize) -> Self {
        Self {
            available: Arc::new(Semaphore::new(size)),
            pool_size: size,
            usage_stats: Arc::new(RwLock::new(ConnectionPoolStats {
                pool_size: size,
                peak_usage: 0,
                average_usage: 0.0,
                wait_time_ms: 0.0,
                connection_timeouts: 0,
            })),
        }
    }
    
    async fn acquire(&self) -> Result<ConnectionHandle> {
        let acquire_start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(5), self.available.acquire()).await {
            Ok(Ok(permit)) => {
                let wait_time = acquire_start.elapsed().as_millis() as f64;
                let current_usage = self.pool_size - self.available.available_permits();
                
                // Update stats
                {
                    let mut stats = self.usage_stats.write().await;
                    if current_usage > stats.peak_usage {
                        stats.peak_usage = current_usage;
                    }
                    stats.wait_time_ms = (stats.wait_time_ms + wait_time) / 2.0; // Moving average
                }
                
                Ok(ConnectionHandle { _permit: permit })
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("Failed to acquire connection: {}", e)),
            Err(_) => {
                {
                    let mut stats = self.usage_stats.write().await;
                    stats.connection_timeouts += 1;
                }
                Err(anyhow::anyhow!("Connection pool timeout"))
            }
        }
    }
    
    async fn get_stats(&self) -> ConnectionPoolStats {
        self.usage_stats.read().await.clone()
    }
}

struct ConnectionHandle {
    _permit: tokio::sync::SemaphorePermit<'static>,
}

// Deadlock Detection
struct DeadlockDetector {
    timeout_duration: Duration,
    timeout_count: Arc<AtomicUsize>,
    deadlock_detected: Arc<AtomicBool>,
}

impl DeadlockDetector {
    fn new(timeout_duration: Duration) -> Self {
        Self {
            timeout_duration,
            timeout_count: Arc::new(AtomicUsize::new(0)),
            deadlock_detected: Arc::new(AtomicBool::new(false)),
        }
    }
    
    fn timeout_duration(&self) -> Duration {
        self.timeout_duration
    }
    
    async fn report_timeout(&self) {
        let count = self.timeout_count.fetch_add(1, Ordering::Relaxed);
        
        // Consider deadlock if we have multiple consecutive timeouts
        if count > 5 {
            self.deadlock_detected.store(true, Ordering::Relaxed);
        }
    }
    
    async fn deadlock_detected(&self) -> bool {
        self.deadlock_detected.load(Ordering::Relaxed)
    }
}
```

## Success Criteria
- Concurrent benchmark runs with specified user count
- Thread pool management handles resource contention properly
- Deadlock detection identifies hanging operations
- Scalability testing shows performance curves accurately
- Connection pooling prevents resource exhaustion
- Windows-compatible async patterns work correctly
- Resource contention scoring reflects actual system behavior
- Scalability recommendations are meaningful and actionable

## Time Limit
10 minutes maximum