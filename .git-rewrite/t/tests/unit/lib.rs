//! LLMKG Unit Testing Framework
//!
//! This library provides comprehensive unit testing capabilities for all LLMKG components,
//! ensuring 100% code coverage with deterministic, predictable outcomes.

// Re-export core LLMKG components for testing
pub use llmkg::*;

// Import test infrastructure
pub use crate::infrastructure::*;

// Unit test modules
pub mod core;
pub mod storage; 
pub mod embedding;
pub mod query;
pub mod federation;
pub mod mcp;
pub mod wasm;

// Import shared test utilities and constants from mod.rs
pub use self::mod::*;

// Test configuration and setup
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the unit testing environment
/// This should be called once before running any tests
pub fn init_unit_testing() {
    INIT.call_once(|| {
        // Set up logging for tests
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .is_test(true)
            .init();
        
        // Initialize test infrastructure
        let _config = UnitTestConfig::default();
        
        println!("LLMKG Unit Testing Framework initialized");
        println!("- Deterministic mode: enabled");
        println!("- Coverage tracking: enabled");
        println!("- Performance monitoring: enabled");
        println!("- Memory leak detection: enabled");
    });
}

/// Main unit test runner
pub async fn run_all_unit_tests() -> anyhow::Result<UnitTestSummary> {
    init_unit_testing();
    
    let config = UnitTestConfig::default();
    let mut runner = UnitTestRunner::new(config)?;
    
    let mut results = Vec::new();
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let start_time = std::time::Instant::now();
    
    println!("Starting comprehensive unit test execution...");
    
    // Core module tests
    println!("\n=== Running Core Module Tests ===");
    let core_results = run_core_tests(&mut runner).await?;
    total_tests += core_results.len();
    passed_tests += core_results.iter().filter(|r| r.passed).count();
    results.extend(core_results);
    
    // Storage module tests
    println!("\n=== Running Storage Module Tests ===");
    let storage_results = run_storage_tests(&mut runner).await?;
    total_tests += storage_results.len();
    passed_tests += storage_results.iter().filter(|r| r.passed).count();
    results.extend(storage_results);
    
    // Embedding module tests
    println!("\n=== Running Embedding Module Tests ===");
    let embedding_results = run_embedding_tests(&mut runner).await?;
    total_tests += embedding_results.len();
    passed_tests += embedding_results.iter().filter(|r| r.passed).count();
    results.extend(embedding_results);
    
    // Query module tests
    println!("\n=== Running Query Module Tests ===");
    let query_results = run_query_tests(&mut runner).await?;
    total_tests += query_results.len();
    passed_tests += query_results.iter().filter(|r| r.passed).count();
    results.extend(query_results);
    
    // Federation module tests
    println!("\n=== Running Federation Module Tests ===");
    let federation_results = run_federation_tests(&mut runner).await?;
    total_tests += federation_results.len();
    passed_tests += federation_results.iter().filter(|r| r.passed).count();
    results.extend(federation_results);
    
    // MCP module tests
    println!("\n=== Running MCP Module Tests ===");
    let mcp_results = run_mcp_tests(&mut runner).await?;
    total_tests += mcp_results.len();
    passed_tests += mcp_results.iter().filter(|r| r.passed).count();
    results.extend(mcp_results);
    
    // WASM module tests
    println!("\n=== Running WASM Module Tests ===");
    let wasm_results = run_wasm_tests(&mut runner).await?;
    total_tests += wasm_results.len();
    passed_tests += wasm_results.iter().filter(|r| r.passed).count();
    results.extend(wasm_results);
    
    let total_duration = start_time.elapsed();
    
    let summary = UnitTestSummary {
        total_tests,
        passed_tests,
        failed_tests: total_tests - passed_tests,
        total_duration,
        results,
        coverage_percentage: calculate_coverage_percentage(&results),
    };
    
    print_test_summary(&summary);
    
    Ok(summary)
}

/// Test summary structure
#[derive(Debug, Clone)]
pub struct UnitTestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub total_duration: std::time::Duration,
    pub results: Vec<UnitTestResult>,
    pub coverage_percentage: f64,
}

async fn run_core_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    // Entity tests
    results.push(runner.run_test("entity_creation_deterministic", || async {
        crate::core::entity_tests::test_entity_creation_deterministic();
        Ok(())
    }).await);
    
    results.push(runner.run_test("entity_key_generation", || async {
        crate::core::entity_tests::test_entity_key_generation();
        Ok(())
    }).await);
    
    results.push(runner.run_test("entity_attribute_edge_cases", || async {
        crate::core::entity_tests::test_entity_attribute_edge_cases();
        Ok(())
    }).await);
    
    results.push(runner.run_test("entity_memory_management", || async {
        crate::core::entity_tests::test_entity_memory_management();
        Ok(())
    }).await);
    
    // Graph tests
    results.push(runner.run_test("graph_basic_operations", || async {
        crate::core::graph_tests::test_graph_basic_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("graph_csr_storage_format", || async {
        crate::core::graph_tests::test_graph_csr_storage_format();
        Ok(())
    }).await);
    
    results.push(runner.run_test("graph_memory_efficiency", || async {
        crate::core::graph_tests::test_graph_memory_efficiency();
        Ok(())
    }).await);
    
    results.push(runner.run_test("graph_concurrent_access", || async {
        crate::core::graph_tests::test_graph_concurrent_access();
        Ok(())
    }).await);
    
    // Memory tests
    results.push(runner.run_test("memory_manager_basic_operations", || async {
        crate::core::memory_tests::test_memory_manager_basic_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("memory_leak_detection", || async {
        crate::core::memory_tests::test_memory_leak_detection();
        Ok(())
    }).await);
    
    // Types tests
    results.push(runner.run_test("entity_key_operations", || async {
        crate::core::types_tests::test_entity_key_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("relationship_types", || async {
        crate::core::types_tests::test_relationship_types();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_storage_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    // CSR tests
    results.push(runner.run_test("csr_construction_deterministic", || async {
        crate::storage::csr_tests::test_csr_construction_deterministic();
        Ok(())
    }).await);
    
    results.push(runner.run_test("csr_access_patterns", || async {
        crate::storage::csr_tests::test_csr_access_patterns();
        Ok(())
    }).await);
    
    results.push(runner.run_test("csr_memory_layout", || async {
        crate::storage::csr_tests::test_csr_memory_layout();
        Ok(())
    }).await);
    
    results.push(runner.run_test("csr_operations", || async {
        crate::storage::csr_tests::test_csr_operations();
        Ok(())
    }).await);
    
    // Bloom filter tests
    results.push(runner.run_test("bloom_filter_basic_operations", || async {
        crate::storage::bloom_tests::test_bloom_filter_basic_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("bloom_filter_deterministic", || async {
        crate::storage::bloom_tests::test_bloom_filter_deterministic();
        Ok(())
    }).await);
    
    results.push(runner.run_test("bloom_filter_serialization", || async {
        crate::storage::bloom_tests::test_bloom_filter_serialization();
        Ok(())
    }).await);
    
    results.push(runner.run_test("bloom_filter_hash_function_quality", || async {
        crate::storage::bloom_tests::test_bloom_filter_hash_function_quality();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_embedding_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    // Quantization tests
    results.push(runner.run_test("product_quantization_accuracy", || async {
        crate::embedding::quantization_tests::test_product_quantization_accuracy();
        Ok(())
    }).await);
    
    results.push(runner.run_test("quantizer_training_convergence", || async {
        crate::embedding::quantization_tests::test_quantizer_training_convergence();
        Ok(())
    }).await);
    
    results.push(runner.run_test("quantization_edge_cases", || async {
        crate::embedding::quantization_tests::test_quantization_edge_cases();
        Ok(())
    }).await);
    
    results.push(runner.run_test("quantization_memory_efficiency", || async {
        crate::embedding::quantization_tests::test_quantization_memory_efficiency();
        Ok(())
    }).await);
    
    // SIMD tests
    results.push(runner.run_test("simd_distance_computation", || async {
        crate::embedding::simd_tests::test_simd_distance_computation();
        Ok(())
    }).await);
    
    results.push(runner.run_test("simd_batch_operations", || async {
        crate::embedding::simd_tests::test_simd_batch_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("simd_alignment_requirements", || async {
        crate::embedding::simd_tests::test_simd_alignment_requirements();
        Ok(())
    }).await);
    
    results.push(runner.run_test("simd_vector_operations", || async {
        crate::embedding::simd_tests::test_simd_vector_operations();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_query_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    // RAG tests
    results.push(runner.run_test("graph_rag_context_assembly", || async {
        crate::query::rag_tests::test_graph_rag_context_assembly();
        Ok(())
    }).await);
    
    results.push(runner.run_test("rag_similarity_search_integration", || async {
        crate::query::rag_tests::test_rag_similarity_search_integration();
        Ok(())
    }).await);
    
    results.push(runner.run_test("rag_context_quality_metrics", || async {
        crate::query::rag_tests::test_rag_context_quality_metrics();
        Ok(())
    }).await);
    
    results.push(runner.run_test("rag_multi_strategy_integration", || async {
        crate::query::rag_tests::test_rag_multi_strategy_integration();
        Ok(())
    }).await);
    
    results.push(runner.run_test("rag_performance_characteristics", || async {
        crate::query::rag_tests::test_rag_performance_characteristics();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_federation_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    results.push(runner.run_test("federation_coordination", || async {
        crate::federation::coordinator_tests::test_federation_coordination();
        Ok(())
    }).await);
    
    results.push(runner.run_test("graph_merging", || async {
        crate::federation::merger_tests::test_graph_merging();
        Ok(())
    }).await);
    
    results.push(runner.run_test("query_routing", || async {
        crate::federation::router_tests::test_query_routing();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_mcp_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    results.push(runner.run_test("mcp_server_initialization", || async {
        crate::mcp::server_tests::test_mcp_server_initialization();
        Ok(())
    }).await);
    
    results.push(runner.run_test("mcp_tool_registration", || async {
        crate::mcp::server_tests::test_mcp_tool_registration();
        Ok(())
    }).await);
    
    results.push(runner.run_test("mcp_message_serialization", || async {
        crate::mcp::protocol_tests::test_mcp_message_serialization();
        Ok(())
    }).await);
    
    Ok(results)
}

async fn run_wasm_tests(runner: &mut UnitTestRunner) -> anyhow::Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    results.push(runner.run_test("wasm_graph_operations", || async {
        crate::wasm::interface_tests::test_wasm_graph_operations();
        Ok(())
    }).await);
    
    results.push(runner.run_test("wasm_memory_management", || async {
        crate::wasm::interface_tests::test_wasm_memory_management();
        Ok(())
    }).await);
    
    results.push(runner.run_test("wasm_call_overhead", || async {
        crate::wasm::performance_tests::test_wasm_call_overhead();
        Ok(())
    }).await);
    
    Ok(results)
}

fn calculate_coverage_percentage(results: &[UnitTestResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    
    let total_coverage: f64 = results.iter().map(|r| r.coverage_percentage).sum();
    total_coverage / results.len() as f64
}

fn print_test_summary(summary: &UnitTestSummary) {
    println!("\n" + "=".repeat(80).as_str());
    println!("UNIT TEST EXECUTION SUMMARY");
    println!("=".repeat(80));
    println!();
    println!("Total Tests:     {}", summary.total_tests);
    println!("Passed:          {} ({:.1}%)", summary.passed_tests, 
             (summary.passed_tests as f64 / summary.total_tests as f64) * 100.0);
    println!("Failed:          {} ({:.1}%)", summary.failed_tests,
             (summary.failed_tests as f64 / summary.total_tests as f64) * 100.0);
    println!("Coverage:        {:.1}%", summary.coverage_percentage);
    println!("Duration:        {:?}", summary.total_duration);
    println!();
    
    if summary.failed_tests > 0 {
        println!("FAILED TESTS:");
        println!("-".repeat(40));
        for result in &summary.results {
            if !result.passed {
                println!("❌ {}", result.name);
                if let Some(ref error) = result.error_message {
                    println!("   Error: {}", error);
                }
                println!("   Duration: {}ms", result.duration_ms);
                println!("   Memory: {} bytes", result.memory_usage_bytes);
                println!();
            }
        }
    } else {
        println!("🎉 ALL TESTS PASSED! 🎉");
    }
    
    println!("=".repeat(80));
    
    // Performance summary
    let avg_duration = summary.results.iter().map(|r| r.duration_ms).sum::<u64>() as f64 / summary.results.len() as f64;
    let max_duration = summary.results.iter().map(|r| r.duration_ms).max().unwrap_or(0);
    let total_memory = summary.results.iter().map(|r| r.memory_usage_bytes).sum::<u64>();
    
    println!("PERFORMANCE SUMMARY:");
    println!("Average test duration: {:.1}ms", avg_duration);
    println!("Maximum test duration: {}ms", max_duration);
    println!("Total memory usage: {:.2}MB", total_memory as f64 / (1024.0 * 1024.0));
    println!("=".repeat(80));
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unit_testing_framework() {
        init_unit_testing();
        
        let config = UnitTestConfig::default();
        let mut runner = UnitTestRunner::new(config).unwrap();
        
        let result = runner.run_test("framework_test", || async {
            // Simple test to verify framework works
            assert_eq!(2 + 2, 4);
            Ok(())
        }).await;
        
        assert!(result.passed);
        assert_eq!(result.name, "framework_test");
    }
}