//! Unit Testing Framework for LLMKG
//! 
//! This module provides comprehensive unit testing capabilities for all LLMKG components,
//! ensuring 100% code coverage with deterministic, predictable outcomes.

pub mod core;
pub mod storage;
pub mod embedding;
pub mod query;
pub mod federation;
pub mod mcp;
pub mod wasm;
pub mod test_utils;

// Individual test modules with complete implementations
pub mod storage_tests;
pub mod embedding_tests;
pub mod query_tests;
pub mod federation_tests;
pub mod mcp_tests;
pub mod wasm_tests;

// Re-export test infrastructure for unit tests
pub use crate::infrastructure::*;

// Unit test specific constants and utilities
pub const UNIT_TEST_TIMEOUT_MS: u64 = 5000; // 5 seconds max per unit test
pub const MAX_MEMORY_PER_TEST: u64 = 100 * 1024 * 1024; // 100MB max per test

// Test seeds for deterministic behavior
pub const ENTITY_TEST_SEED: u64 = 0x1234567890ABCDEF;
pub const GRAPH_TEST_SEED: u64 = 0x2345678901BCDEF0;
pub const CSR_TEST_SEED: u64 = 0x3456789012CDEF01;
pub const BLOOM_TEST_SEED: u64 = 0x456789023DEF012;
pub const PQ_TEST_SEED: u64 = 0x56789034EF0123;
pub const SIMD_TEST_SEED: u64 = 0x6789045F01234;
pub const RAG_TEST_SEED: u64 = 0x789056012345;

// Expected memory usage constants (in bytes)
pub const EXPECTED_EMPTY_ENTITY_SIZE: u64 = 64;
pub const EXPECTED_EMPTY_GRAPH_SIZE: u64 = 128;
pub const EXPECTED_ENTITY_SIZE_UPPER_BOUND: u64 = 200;
pub const ATTRIBUTE_OVERHEAD: usize = 16;

// Performance expectations
pub const EXPECTED_PQ_AVERAGE_ERROR: f32 = 0.15;
pub const EXPECTED_PQ_MAX_ERROR: f32 = 0.5;
pub const EXPECTED_MIN_COMPRESSION_RATIO: f64 = 15.0;
pub const EXPECTED_FINAL_TRAINING_LOSS: f32 = 0.01;
pub const EXPECTED_LOSS_VARIANCE: f32 = 0.001;

// Additional test seeds for specific scenarios
pub const CSR_OPERATIONS_SEED: u64 = 0x789A0B1C2D3E4F5;
pub const BLOOM_SERIALIZE_SEED: u64 = 0x89AB1C2D3E4F567;
pub const FALSE_POSITIVE_SEED: u64 = 0x9ABC2D3E4F56789;
pub const PQ_TRAINING_SEED: u64 = 0xABCD3E4F567890A;
pub const PQ_MEMORY_SEED: u64 = 0xBCDE4F567890AB;
pub const SIMD_BATCH_SEED: u64 = 0xCDEF567890ABCD;
pub const SIMD_ALIGNMENT_SEED: u64 = 0xDEF0678901BCDE;
pub const SIMD_VECTOR_OPS_SEED: u64 = 0xEF01789012CDEF;
pub const ACCESS_PATTERN_SEED: u64 = 0xF0123890123DEF0;

/// Unit test configuration
#[derive(Debug, Clone)]
pub struct UnitTestConfig {
    pub timeout_ms: u64,
    pub max_memory_bytes: u64,
    pub enable_coverage: bool,
    pub enable_performance_tracking: bool,
    pub fail_on_memory_leak: bool,
    pub deterministic_mode: bool,
}

impl Default for UnitTestConfig {
    fn default() -> Self {
        Self {
            timeout_ms: UNIT_TEST_TIMEOUT_MS,
            max_memory_bytes: MAX_MEMORY_PER_TEST,
            enable_coverage: true,
            enable_performance_tracking: true,
            fail_on_memory_leak: true,
            deterministic_mode: true,
        }
    }
}

/// Unit test result with detailed metrics
#[derive(Debug, Clone)]
pub struct UnitTestResult {
    pub name: String,
    pub passed: bool,
    pub duration_ms: u64,
    pub memory_usage_bytes: u64,
    pub coverage_percentage: f64,
    pub error_message: Option<String>,
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Unit test runner with isolated execution
pub struct UnitTestRunner {
    config: UnitTestConfig,
    environment: TestEnvironment,
    monitor: PerformanceMonitor,
}

impl UnitTestRunner {
    pub fn new(config: UnitTestConfig) -> anyhow::Result<Self> {
        let resource_limits = ResourceLimits {
            max_memory: config.max_memory_bytes,
            max_cpu_percent: 50.0,
            timeout: std::time::Duration::from_millis(config.timeout_ms),
        };
        
        let environment = TestEnvironment::new(resource_limits)?;
        let monitor = PerformanceMonitor::new()?;
        
        Ok(Self {
            config,
            environment,
            monitor,
        })
    }

    /// Execute a single unit test with full isolation
    pub async fn run_test<F, Fut>(&mut self, test_name: &str, test_fn: F) -> UnitTestResult
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
    {
        let start_time = std::time::Instant::now();
        let initial_memory = self.monitor.current_memory_usage();
        
        // Set up isolated environment
        let _isolation_guard = self.environment.enter().await;
        
        // Execute test with timeout
        let result = match tokio::time::timeout(
            std::time::Duration::from_millis(self.config.timeout_ms),
            test_fn()
        ).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow::anyhow!("Test timed out after {}ms", self.config.timeout_ms)),
        };
        
        let duration = start_time.elapsed();
        let final_memory = self.monitor.current_memory_usage();
        let memory_used = final_memory.saturating_sub(initial_memory);
        
        // Check for memory leaks
        let memory_leak = if self.config.fail_on_memory_leak {
            memory_used > self.config.max_memory_bytes / 10 // Allow 10% overhead
        } else {
            false
        };
        
        let passed = result.is_ok() && !memory_leak;
        let error_message = if !passed {
            Some(match result {
                Err(e) => format!("Test failed: {}", e),
                Ok(()) if memory_leak => format!("Memory leak detected: {} bytes", memory_used),
                _ => "Unknown error".to_string(),
            })
        } else {
            None
        };
        
        UnitTestResult {
            name: test_name.to_string(),
            passed,
            duration_ms: duration.as_millis() as u64,
            memory_usage_bytes: memory_used,
            coverage_percentage: 0.0, // Will be populated by coverage analysis
            error_message,
            performance_metrics: self.monitor.get_current_metrics().ok(),
        }
    }
}

/// Common test utilities for all unit tests
pub mod test_utils {
    use super::*;
    use crate::core::{Entity, EntityKey, KnowledgeGraph};
    use crate::embedding::store::EmbeddingStore;
    use std::collections::HashMap;

    /// Create a test entity with deterministic properties
    pub fn create_test_entity(id: &str, name: &str) -> Entity {
        let key = EntityKey::from_hash(id);
        Entity::new(key, name.to_string())
    }

    /// Create a test graph with specified number of entities and relationships
    pub fn create_test_graph(entity_count: usize, relationship_count: usize) -> KnowledgeGraph {
        let mut rng = DeterministicRng::new(GRAPH_TEST_SEED);
        let mut graph = KnowledgeGraph::new();
        
        // Add entities
        for i in 0..entity_count {
            let entity = create_test_entity(
                &format!("test_entity_{}", i),
                &format!("Test Entity {}", i)
            );
            graph.add_entity(entity).unwrap();
        }
        
        // Add relationships
        for _ in 0..relationship_count {
            let source_idx = rng.gen_range(0..entity_count);
            let target_idx = rng.gen_range(0..entity_count);
            
            if source_idx != target_idx {
                let source_key = EntityKey::from_hash(&format!("test_entity_{}", source_idx));
                let target_key = EntityKey::from_hash(&format!("test_entity_{}", target_idx));
                
                let relationship = crate::core::Relationship::new(
                    "test_relationship".to_string(),
                    rng.gen_range(0.1..1.0),
                    crate::core::RelationshipType::Directed
                );
                
                let _ = graph.add_relationship(source_key, target_key, relationship);
            }
        }
        
        graph
    }

    /// Verify two vectors are approximately equal
    pub fn assert_vectors_equal(v1: &[f32], v2: &[f32], tolerance: f32) {
        assert_eq!(v1.len(), v2.len(), "Vector lengths differ");
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert!((a - b).abs() < tolerance, 
                   "Vectors differ at index {}: {} vs {} (tolerance: {})", 
                   i, a, b, tolerance);
        }
    }

    /// Calculate expected memory usage for an entity
    pub fn calculate_expected_entity_memory(entity: &Entity) -> u64 {
        let mut size = EXPECTED_EMPTY_ENTITY_SIZE;
        size += entity.name().len() as u64;
        
        for (key, value) in entity.attributes() {
            size += key.len() as u64 + value.len() as u64 + ATTRIBUTE_OVERHEAD as u64;
        }
        
        size
    }

    /// Measure execution time of a function
    pub fn measure_execution_time<F, R>(f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Create test vectors with specific properties
    pub fn create_test_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = DeterministicRng::new(seed);
        (0..count)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    /// Verify matrix properties for CSR testing
    pub fn verify_csr_properties<T>(
        row_offsets: &[usize], 
        column_indices: &[usize], 
        values: &[T],
        num_rows: usize
    ) -> bool {
        // Check row offsets length
        if row_offsets.len() != num_rows + 1 {
            return false;
        }
        
        // Check monotonic increase
        for i in 1..row_offsets.len() {
            if row_offsets[i] < row_offsets[i-1] {
                return false;
            }
        }
        
        // Check consistent lengths
        let nnz = row_offsets[num_rows];
        column_indices.len() == nnz && values.len() == nnz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unit_test_runner_creation() {
        let config = UnitTestConfig::default();
        let runner = UnitTestRunner::new(config);
        assert!(runner.is_ok());
    }

    #[tokio::test]
    async fn test_deterministic_rng_reproducibility() {
        let mut rng1 = DeterministicRng::new(ENTITY_TEST_SEED);
        let mut rng2 = DeterministicRng::new(ENTITY_TEST_SEED);
        
        for _ in 0..100 {
            assert_eq!(rng1.gen::<u64>(), rng2.gen::<u64>());
        }
    }

    #[test]
    fn test_test_utils_entity_creation() {
        let entity = test_utils::create_test_entity("test", "Test Entity");
        assert_eq!(entity.name(), "Test Entity");
        assert_eq!(entity.attributes().len(), 0);
    }

    #[test]
    fn test_test_utils_graph_creation() {
        let graph = test_utils::create_test_graph(10, 15);
        assert_eq!(graph.entity_count(), 10);
        assert!(graph.relationship_count() <= 15); // Some may be duplicates
    }

    #[test]
    fn test_vector_equality_assertion() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.001, 2.001, 3.001];
        
        test_utils::assert_vectors_equal(&v1, &v2, 0.01);
        
        // This should panic
        let result = std::panic::catch_unwind(|| {
            test_utils::assert_vectors_equal(&v1, &v2, 0.0001);
        });
        assert!(result.is_err());
    }
}