//! Model Loading Performance Benchmarks
//!
//! Measures the performance of model loading, caching, eviction, and resource management
//! for different sized models and concurrent scenarios.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use llmkg::enhanced_knowledge_storage::{
    model_management::{ModelResourceManager, ModelResourceConfig},
    types::*,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create a test resource manager with specified configuration
fn create_manager(config: ModelResourceConfig) -> ModelResourceManager {
    ModelResourceManager::new(config)
}

/// Create default test configuration
fn default_test_config() -> ModelResourceConfig {
    ModelResourceConfig {
        max_memory_usage: 4_000_000_000, // 4GB for benchmarks
        max_concurrent_models: 5,
        idle_timeout: Duration::from_secs(300),
        min_memory_threshold: 100_000_000,
    }
}

/// Benchmark loading a small 135M parameter model
pub fn benchmark_model_loading_small(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = default_test_config();
    
    c.bench_function("load_small_model_135m", |b| {
        b.iter(|| {
            rt.block_on(async {
                let manager = create_manager(config.clone());
                let task = ProcessingTask::new(ComplexityLevel::Low, "benchmark task");
                let result = manager.process_with_optimal_model(task).await;
                black_box(result.unwrap());
            })
        });
    });
}

/// Benchmark loading a medium 360M parameter model
pub fn benchmark_model_loading_medium(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = default_test_config();
    
    c.bench_function("load_medium_model_360m", |b| {
        b.iter(|| {
            rt.block_on(async {
                let manager = create_manager(config.clone());
                let task = ProcessingTask::new(ComplexityLevel::Medium, "benchmark task");
                let result = manager.process_with_optimal_model(task).await;
                black_box(result.unwrap());
            })
        });
    });
}

/// Benchmark concurrent model loading with multiple tasks
pub fn benchmark_concurrent_model_loading(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = default_test_config();
    
    let mut group = c.benchmark_group("concurrent_model_loading");
    
    for &num_concurrent in &[2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_tasks", num_concurrent),
            &num_concurrent,
            |b, &num_concurrent| {
                b.to_async(&rt).iter(|| async {
                    let manager = Arc::new(create_manager(config.clone()));
                    let mut handles = Vec::new();
                    
                    for i in 0..num_concurrent {
                        let manager_clone = manager.clone();
                        let complexity = match i % 3 {
                            0 => ComplexityLevel::Low,
                            1 => ComplexityLevel::Medium,
                            _ => ComplexityLevel::High,
                        };
                        
                        let handle = tokio::spawn(async move {
                            let task = ProcessingTask::new(complexity, &format!("task_{}", i));
                            manager_clone.process_with_optimal_model(task).await
                        });
                        handles.push(handle);
                    }
                    
                    let mut results = Vec::new();
                    for handle in handles {
                        let result = handle.await.unwrap().unwrap();
                        results.push(result);
                    }
                    
                    black_box(results);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark LRU eviction performance under memory pressure
pub fn benchmark_lru_eviction_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create a memory-constrained configuration to force evictions
    let config = ModelResourceConfig {
        max_memory_usage: 1_500_000_000, // 1.5GB - forces eviction
        max_concurrent_models: 2,
        idle_timeout: Duration::from_secs(1),
        min_memory_threshold: 100_000_000,
    };
    
    c.bench_function("lru_eviction_under_pressure", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = create_manager(config.clone());
            
            // Load multiple models to trigger eviction
            let tasks = vec![
                ProcessingTask::new(ComplexityLevel::Low, "task 1"),
                ProcessingTask::new(ComplexityLevel::Medium, "task 2"),
                ProcessingTask::new(ComplexityLevel::High, "task 3"),
                ProcessingTask::new(ComplexityLevel::Low, "task 4"), // Should trigger eviction
            ];
            
            let mut results = Vec::new();
            for task in tasks {
                let result = manager.process_with_optimal_model(task).await.unwrap();
                results.push(result);
            }
            
            black_box(results);
        });
    });
}

/// Benchmark model cache operations (hit/miss scenarios)
pub fn benchmark_model_cache_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = default_test_config();
    
    let mut group = c.benchmark_group("model_cache_operations");
    
    // Benchmark cache hits (same complexity tasks)
    group.bench_function("cache_hit_scenario", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = create_manager(config.clone());
            
            // First task loads the model
            let task1 = ProcessingTask::new(ComplexityLevel::Low, "first task");
            let result1 = manager.process_with_optimal_model(task1).await.unwrap();
            
            // Second task should hit cache
            let task2 = ProcessingTask::new(ComplexityLevel::Low, "second task");
            let result2 = manager.process_with_optimal_model(task2).await.unwrap();
            
            black_box((result1, result2));
        });
    });
    
    // Benchmark cache misses (different complexity tasks)
    group.bench_function("cache_miss_scenario", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = create_manager(config.clone());
            
            // Tasks with different complexities to force different models
            let tasks = vec![
                ProcessingTask::new(ComplexityLevel::Low, "low task"),
                ProcessingTask::new(ComplexityLevel::Medium, "medium task"),
                ProcessingTask::new(ComplexityLevel::High, "high task"),
            ];
            
            let mut results = Vec::new();
            for task in tasks {
                let result = manager.process_with_optimal_model(task).await.unwrap();
                results.push(result);
            }
            
            black_box(results);
        });
    });
    
    group.finish();
}

/// Benchmark resource monitoring overhead
pub fn benchmark_resource_monitoring(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = default_test_config();
    
    c.bench_function("resource_monitoring_overhead", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = create_manager(config.clone());
            
            // Process a task and measure monitoring overhead
            let task = ProcessingTask::new(ComplexityLevel::Medium, "monitoring test");
            let _result = manager.process_with_optimal_model(task).await.unwrap();
            
            // Get stats multiple times to measure monitoring overhead
            let stats1 = manager.get_stats().await;
            let stats2 = manager.get_stats().await;
            let memory_usage = manager.current_memory_usage().await;
            let _evicted = manager.cleanup_idle_models().await;
            
            black_box((stats1, stats2, memory_usage));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_benchmark_setup() {
        let config = default_test_config();
        let manager = create_manager(config);
        
        let task = ProcessingTask::new(ComplexityLevel::Low, "test task");
        let result = manager.process_with_optimal_model(task).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(!result.output.is_empty());
    }
    
    #[tokio::test]
    async fn test_concurrent_benchmark_setup() {
        let config = default_test_config();
        let manager = Arc::new(create_manager(config));
        
        let mut handles = Vec::new();
        for i in 0..3 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let task = ProcessingTask::new(ComplexityLevel::Low, &format!("concurrent_test_{}", i));
                manager_clone.process_with_optimal_model(task).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            results.push(result);
        }
        
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.success);
        }
    }
    
    #[tokio::test]
    async fn test_memory_pressure_scenario() {
        let config = ModelResourceConfig {
            max_memory_usage: 1_000_000_000, // 1GB limit
            max_concurrent_models: 2,
            idle_timeout: Duration::from_millis(100),
            min_memory_threshold: 100_000_000,
        };
        
        let manager = create_manager(config);
        
        // Process tasks that should trigger memory management
        let tasks = vec![
            ProcessingTask::new(ComplexityLevel::Low, "pressure_test_1"),
            ProcessingTask::new(ComplexityLevel::Medium, "pressure_test_2"),
            ProcessingTask::new(ComplexityLevel::High, "pressure_test_3"),
        ];
        
        let mut results = Vec::new();
        for task in tasks {
            let result = manager.process_with_optimal_model(task).await;
            if let Ok(result) = result {
                results.push(result);
            }
        }
        
        // Should have processed at least some tasks successfully
        assert!(!results.is_empty());
        
        // Memory usage should be within limits
        let memory_usage = manager.current_memory_usage().await;
        assert!(memory_usage <= 1_000_000_000);
    }
}