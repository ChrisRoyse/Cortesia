//! LLMKG Test Runner
//! 
//! Main entry point for running LLMKG unit tests

use llmkg_tests::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 LLMKG Unit Testing Framework");
    println!("================================");
    
    // Initialize testing environment
    init_unit_testing();
    
    // Run comprehensive unit tests
    let summary = run_all_unit_tests().await?;
    
    // Print final summary
    println!("\n🎯 Test Execution Complete!");
    println!("Total Tests: {}", summary.total_tests);
    println!("Passed: {} ({:.1}%)", 
             summary.passed_tests, 
             (summary.passed_tests as f64 / summary.total_tests as f64) * 100.0);
    println!("Failed: {}", summary.failed_tests);
    println!("Coverage: {:.1}%", summary.coverage_percentage);
    println!("Duration: {:?}", summary.total_duration);
    
    if summary.failed_tests == 0 {
        println!("\n✅ All tests passed!");
        Ok(())
    } else {
        println!("\n❌ Some tests failed!");
        std::process::exit(1);
    }
}

/// Initialize the unit testing environment
fn init_unit_testing() {
    // Set up logging for tests
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .init();
    
    println!("✅ LLMKG Unit Testing Framework initialized");
    println!("- Deterministic mode: enabled");
    println!("- Coverage tracking: enabled");
    println!("- Performance monitoring: enabled");
    println!("- Memory leak detection: enabled");
}

/// Run all unit tests
async fn run_all_unit_tests() -> Result<UnitTestSummary> {
    let config = UnitTestConfig::default();
    let mut runner = UnitTestRunner::new(config)?;
    
    let mut results = Vec::new();
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let start_time = std::time::Instant::now();
    
    println!("\nStarting comprehensive unit test execution...");
    
    // Run basic tests
    println!("\n=== Running Basic Framework Tests ===");
    let basic_results = run_basic_tests(&mut runner).await?;
    total_tests += basic_results.len();
    passed_tests += basic_results.iter().filter(|r| r.passed).count();
    results.extend(basic_results);

    // Run comprehensive layer tests
    println!("\n=== Running Storage Layer Tests ===");
    let storage_results = llmkg_tests::unit::storage_tests::run_storage_tests().await?;
    total_tests += storage_results.len();
    passed_tests += storage_results.iter().filter(|r| r.passed).count();
    results.extend(storage_results);

    println!("\n=== Running Embedding Layer Tests ===");
    let embedding_results = llmkg_tests::unit::embedding_tests::run_embedding_tests().await?;
    total_tests += embedding_results.len();
    passed_tests += embedding_results.iter().filter(|r| r.passed).count();
    results.extend(embedding_results);

    println!("\n=== Running Query Engine Tests ===");
    let query_results = llmkg_tests::unit::query_tests::run_query_tests().await?;
    total_tests += query_results.len();
    passed_tests += query_results.iter().filter(|r| r.passed).count();
    results.extend(query_results);

    println!("\n=== Running Federation Layer Tests ===");
    let federation_results = llmkg_tests::unit::federation_tests::run_federation_tests().await?;
    total_tests += federation_results.len();
    passed_tests += federation_results.iter().filter(|r| r.passed).count();
    results.extend(federation_results);

    println!("\n=== Running MCP Integration Tests ===");
    let mcp_results = llmkg_tests::unit::mcp_tests::run_mcp_tests().await?;
    total_tests += mcp_results.len();
    passed_tests += mcp_results.iter().filter(|r| r.passed).count();
    results.extend(mcp_results);

    println!("\n=== Running WASM Runtime Tests ===");
    let wasm_results = llmkg_tests::unit::wasm_tests::run_wasm_tests().await?;
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

/// Run basic framework tests
async fn run_basic_tests(runner: &mut UnitTestRunner) -> Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();
    
    // Entity tests
    results.push(runner.run_test("entity_creation", || async {
        let key = EntityKey::from_hash("test");
        let entity = Entity::new(key, "Test Entity".to_string());
        assert_eq!(entity.key(), key);
        assert_eq!(entity.name(), "Test Entity");
        Ok(())
    }).await);
    
    results.push(runner.run_test("entity_attributes", || async {
        let mut entity = create_test_entity("test", "Test");
        entity.add_attribute("key", "value");
        assert_eq!(entity.get_attribute("key"), Some("value"));
        Ok(())
    }).await);
    
    // Graph tests
    results.push(runner.run_test("graph_creation", || async {
        let graph = KnowledgeGraph::new();
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        Ok(())
    }).await);
    
    results.push(runner.run_test("graph_operations", || async {
        let mut graph = KnowledgeGraph::new();
        let entity = create_test_entity("test", "Test Entity");
        let key = entity.key();
        
        graph.add_entity(entity)?;
        assert_eq!(graph.entity_count(), 1);
        assert!(graph.contains_entity(key));
        Ok(())
    }).await);
    
    // Deterministic RNG tests
    results.push(runner.run_test("deterministic_rng", || async {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(12345);
        
        for _ in 0..10 {
            assert_eq!(rng1.gen::<u64>(), rng2.gen::<u64>());
        }
        Ok(())
    }).await);
    
    // Test utilities
    results.push(runner.run_test("test_utilities", || async {
        let graph = create_test_graph(5, 8);
        assert_eq!(graph.entity_count(), 5);
        assert!(graph.relationship_count() <= 8);
        
        let vectors = create_test_vectors(3, 4, 12345);
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0].len(), 4);
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