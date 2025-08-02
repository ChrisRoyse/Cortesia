# Micro Task 40: Integration Tests

**Priority**: CRITICAL  
**Estimated Time**: 50 minutes  
**Dependencies**: Tasks 01-39 completed  
**Skills Required**: System testing, performance validation

## Objective

Implement comprehensive integration tests that validate the entire Phase 7 system working together as a cohesive query processing engine.

## Context

Integration tests ensure that all components work together correctly, performance targets are met, and the system handles real-world scenarios robustly. These tests validate the complete user journey from query input to explained results.

## Specifications

### Test Coverage Requirements

1. **End-to-End Scenarios**
   - Simple query processing (filter, definition)
   - Complex multi-hop queries (relationship, hierarchy)
   - Compound query decomposition and merging
   - Error handling and recovery

2. **Performance Validation**
   - Response time requirements (< 50ms)
   - Concurrent query handling (> 100/s)
   - Memory usage limits (< 50MB)
   - Activation accuracy (> 95%)

3. **Quality Assurance**
   - Intent recognition accuracy (> 90%)
   - Explanation quality (> 85%)
   - Result relevance verification
   - System stability under load

## Implementation Guide

### Step 1: Test Infrastructure
```rust
// File: tests/integration/query_system_integration.rs

use std::sync::Arc;
use tokio::time::{Duration, Instant};
use LLMKG::query::*;
use LLMKG::core::*;

#[derive(Debug)]
pub struct IntegrationTestSuite {
    query_processor: Arc<QueryProcessor>,
    test_graph: Arc<TestGraph>,
    performance_monitor: Arc<TestPerformanceMonitor>,
    result_validator: ResultValidator,
}

impl IntegrationTestSuite {
    pub async fn new() -> Self {
        // Create test environment
        let test_graph = Arc::new(TestGraph::create_comprehensive_test_graph().await);
        let query_processor = Arc::new(
            QueryProcessor::new_with_graph(test_graph.clone()).await
        );
        
        Self {
            query_processor,
            test_graph,
            performance_monitor: Arc::new(TestPerformanceMonitor::new()),
            result_validator: ResultValidator::new(),
        }
    }
    
    pub async fn run_all_tests(&self) -> IntegrationTestResults {
        let mut results = IntegrationTestResults::new();
        
        // Run test suites
        results.end_to_end = self.run_end_to_end_tests().await;
        results.performance = self.run_performance_tests().await;
        results.quality = self.run_quality_tests().await;
        results.stress = self.run_stress_tests().await;
        results.edge_cases = self.run_edge_case_tests().await;
        
        results
    }
}

#[derive(Debug)]
pub struct IntegrationTestResults {
    pub end_to_end: TestSuiteResult,
    pub performance: TestSuiteResult,
    pub quality: TestSuiteResult,
    pub stress: TestSuiteResult,
    pub edge_cases: TestSuiteResult,
    pub overall_success: bool,
}
```

### Step 2: End-to-End Test Scenarios
```rust
impl IntegrationTestSuite {
    async fn run_end_to_end_tests(&self) -> TestSuiteResult {
        let mut suite = TestSuiteResult::new("End-to-End Tests");
        
        // Test 1: Simple Filter Query
        suite.add_test(self.test_simple_filter_query().await);
        
        // Test 2: Relationship Query
        suite.add_test(self.test_relationship_query().await);
        
        // Test 3: Hierarchy Navigation
        suite.add_test(self.test_hierarchy_query().await);
        
        // Test 4: Complex Comparison
        suite.add_test(self.test_comparison_query().await);
        
        // Test 5: Compound Query Processing
        suite.add_test(self.test_compound_query().await);
        
        suite
    }
    
    async fn test_simple_filter_query(&self) -> TestResult {
        let query = "What animals can fly?";
        let start = Instant::now();
        
        match self.query_processor.process_query(query, &QueryContext::default()).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                
                // Validate result structure
                assert!(!result.results.is_empty(), "No results returned");
                assert!(!result.explanations.is_empty(), "No explanations generated");
                
                // Validate intent recognition
                assert!(matches!(result.intent.intent_type, QueryIntent::Filter { .. }), 
                       "Wrong intent classification");
                assert!(result.intent.confidence > 0.8, "Low intent confidence");
                
                // Validate performance
                assert!(elapsed < Duration::from_millis(50), 
                       "Query took too long: {:?}", elapsed);
                
                // Validate result relevance
                for result_item in &result.results {
                    assert!(result_item.relevance_score > 0.5, "Low relevance score");
                }
                
                TestResult::success("Simple filter query", elapsed)
            }
            Err(e) => TestResult::failure("Simple filter query", format!("Query failed: {}", e))
        }
    }
    
    async fn test_relationship_query(&self) -> TestResult {
        let query = "How are dogs related to wolves?";
        let start = Instant::now();
        
        match self.query_processor.process_query(query, &QueryContext::default()).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                
                // Validate relationship intent
                if let QueryIntent::Relationship { entity1, entity2, .. } = &result.intent.intent_type {
                    assert!(entity1.to_lowercase().contains("dog") || 
                           entity2.to_lowercase().contains("dog"), "Dog entity not found");
                    assert!(entity1.to_lowercase().contains("wolf") || 
                           entity2.to_lowercase().contains("wolf"), "Wolf entity not found");
                } else {
                    return TestResult::failure("Relationship query", "Wrong intent type".to_string());
                }
                
                // Validate activation patterns
                assert!(result.activation_trace.len() > 1, "No activation spreading occurred");
                
                // Validate explanations mention both entities
                let explanation_text = result.explanations.iter()
                    .map(|e| &e.human_readable)
                    .collect::<Vec<_>>()
                    .join(" ");
                
                assert!(explanation_text.to_lowercase().contains("dog"), 
                       "Explanation missing dog reference");
                assert!(explanation_text.to_lowercase().contains("wolf"), 
                       "Explanation missing wolf reference");
                
                TestResult::success("Relationship query", elapsed)
            }
            Err(e) => TestResult::failure("Relationship query", format!("Query failed: {}", e))
        }
    }
    
    async fn test_compound_query(&self) -> TestResult {
        let query = "Compare the digestive systems of carnivorous and herbivorous mammals";
        let start = Instant::now();
        
        match self.query_processor.process_query(query, &QueryContext::default()).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                
                // Should be classified as comparison
                assert!(matches!(result.intent.intent_type, QueryIntent::Comparison { .. }), 
                       "Wrong intent classification for compound query");
                
                // Should have sub-queries or complex processing
                assert!(result.intent.sub_queries.len() > 0 || 
                       result.processing_time > Duration::from_millis(20), 
                       "Compound query not properly processed");
                
                // Should have comprehensive results
                assert!(result.results.len() >= 2, "Insufficient results for comparison");
                
                // Performance should still be acceptable
                assert!(elapsed < Duration::from_millis(100), 
                       "Compound query took too long: {:?}", elapsed);
                
                TestResult::success("Compound query", elapsed)
            }
            Err(e) => TestResult::failure("Compound query", format!("Query failed: {}", e))
        }
    }
}
```

### Step 3: Performance Integration Tests
```rust
impl IntegrationTestSuite {
    async fn run_performance_tests(&self) -> TestSuiteResult {
        let mut suite = TestSuiteResult::new("Performance Tests");
        
        suite.add_test(self.test_response_time_requirements().await);
        suite.add_test(self.test_concurrent_query_performance().await);
        suite.add_test(self.test_memory_usage_limits().await);
        suite.add_test(self.test_activation_performance().await);
        
        suite
    }
    
    async fn test_concurrent_query_performance(&self) -> TestResult {
        let concurrent_queries = 50;
        let queries = vec![
            "What animals live in water?",
            "How are birds related to reptiles?",
            "Find large predators",
            "What is photosynthesis?",
            "Compare lions and tigers",
        ];
        
        let start = Instant::now();
        
        // Launch concurrent queries
        let handles: Vec<_> = (0..concurrent_queries).map(|i| {
            let query = queries[i % queries.len()];
            let processor = self.query_processor.clone();
            tokio::spawn(async move {
                processor.process_query(query, &QueryContext::default()).await
            })
        }).collect();
        
        // Wait for all to complete
        let results = futures::future::try_join_all(handles).await;
        let elapsed = start.elapsed();
        
        match results {
            Ok(query_results) => {
                // Validate all succeeded
                let success_count = query_results.iter()
                    .filter(|r| r.is_ok())
                    .count();
                
                assert_eq!(success_count, concurrent_queries, 
                          "Not all concurrent queries succeeded");
                
                // Calculate throughput
                let throughput = concurrent_queries as f64 / elapsed.as_secs_f64();
                assert!(throughput >= 100.0, 
                       "Throughput {} below requirement of 100 queries/second", throughput);
                
                TestResult::success("Concurrent performance", elapsed)
            }
            Err(e) => TestResult::failure("Concurrent performance", 
                                        format!("Concurrent queries failed: {}", e))
        }
    }
    
    async fn test_memory_usage_limits(&self) -> TestResult {
        let initial_memory = self.measure_memory_usage().await;
        
        // Process many queries to stress memory
        for i in 0..100 {
            let query = format!("Find entities related to test_{}", i);
            let _ = self.query_processor.process_query(&query, &QueryContext::default()).await;
        }
        
        let final_memory = self.measure_memory_usage().await;
        let memory_increase = final_memory - initial_memory;
        
        // Memory increase should be reasonable (< 50MB)
        if memory_increase < 50 * 1024 * 1024 {
            TestResult::success("Memory usage", Duration::from_secs(0))
        } else {
            TestResult::failure("Memory usage", 
                              format!("Memory increased by {} bytes", memory_increase))
        }
    }
}
```

### Step 4: Quality Integration Tests
```rust
impl IntegrationTestSuite {
    async fn run_quality_tests(&self) -> TestSuiteResult {
        let mut suite = TestSuiteResult::new("Quality Tests");
        
        suite.add_test(self.test_intent_recognition_accuracy().await);
        suite.add_test(self.test_explanation_quality().await);
        suite.add_test(self.test_result_relevance().await);
        suite.add_test(self.test_activation_accuracy().await);
        
        suite
    }
    
    async fn test_intent_recognition_accuracy(&self) -> TestResult {
        let test_cases = vec![
            ("What animals can fly?", QueryIntent::Filter { .. }),
            ("How are dogs related to wolves?", QueryIntent::Relationship { .. }),
            ("Show me the mammal hierarchy", QueryIntent::Hierarchy { .. }),
            ("What's the difference between cats and dogs?", QueryIntent::Comparison { .. }),
            ("Why do birds migrate?", QueryIntent::Causal { .. }),
            ("What is photosynthesis?", QueryIntent::Definition { .. }),
        ];
        
        let mut correct = 0;
        let mut total_confidence = 0.0;
        
        for (query, expected_intent) in &test_cases {
            match self.query_processor.process_query(query, &QueryContext::default()).await {
                Ok(result) => {
                    if std::mem::discriminant(&result.intent.intent_type) == 
                       std::mem::discriminant(expected_intent) {
                        correct += 1;
                    }
                    total_confidence += result.intent.confidence;
                }
                Err(_) => {
                    // Count as incorrect
                }
            }
        }
        
        let accuracy = correct as f64 / test_cases.len() as f64;
        let avg_confidence = total_confidence / test_cases.len() as f32;
        
        if accuracy >= 0.9 && avg_confidence >= 0.8 {
            TestResult::success("Intent recognition accuracy", Duration::from_secs(0))
        } else {
            TestResult::failure("Intent recognition accuracy", 
                              format!("Accuracy: {:.2}, Avg Confidence: {:.2}", accuracy, avg_confidence))
        }
    }
    
    async fn test_explanation_quality(&self) -> TestResult {
        let test_queries = vec![
            "What animals can fly?",
            "How are penguins related to birds?",
            "Why don't penguins fly like other birds?",
        ];
        
        let mut quality_scores = Vec::new();
        
        for query in &test_queries {
            match self.query_processor.process_query(query, &QueryContext::default()).await {
                Ok(result) => {
                    for explanation in &result.explanations {
                        let quality = self.result_validator.assess_explanation_quality(explanation);
                        quality_scores.push(quality);
                    }
                }
                Err(_) => {
                    quality_scores.push(0.0); // Failed explanations get 0 score
                }
            }
        }
        
        let avg_quality: f64 = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        
        if avg_quality >= 0.85 {
            TestResult::success("Explanation quality", Duration::from_secs(0))
        } else {
            TestResult::failure("Explanation quality", 
                              format!("Average quality: {:.2}", avg_quality))
        }
    }
}
```

### Step 5: Stress and Edge Case Tests
```rust
impl IntegrationTestSuite {
    async fn run_stress_tests(&self) -> TestSuiteResult {
        let mut suite = TestSuiteResult::new("Stress Tests");
        
        suite.add_test(self.test_sustained_load().await);
        suite.add_test(self.test_memory_pressure().await);
        suite.add_test(self.test_large_result_sets().await);
        
        suite
    }
    
    async fn test_sustained_load(&self) -> TestResult {
        let duration = Duration::from_secs(30);
        let start = Instant::now();
        let mut query_count = 0;
        let mut error_count = 0;
        
        while start.elapsed() < duration {
            let query = format!("Find test entities {}", query_count);
            
            match self.query_processor.process_query(&query, &QueryContext::default()).await {
                Ok(_) => query_count += 1,
                Err(_) => error_count += 1,
            }
            
            // Small delay to prevent overwhelming
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        let error_rate = error_count as f64 / (query_count + error_count) as f64;
        
        if error_rate < 0.05 && query_count > 100 {
            TestResult::success("Sustained load", duration)
        } else {
            TestResult::failure("Sustained load", 
                              format!("Error rate: {:.2}, Queries: {}", error_rate, query_count))
        }
    }
    
    async fn run_edge_case_tests(&self) -> TestSuiteResult {
        let mut suite = TestSuiteResult::new("Edge Case Tests");
        
        suite.add_test(self.test_empty_query().await);
        suite.add_test(self.test_very_long_query().await);
        suite.add_test(self.test_nonsensical_query().await);
        suite.add_test(self.test_special_characters().await);
        
        suite
    }
    
    async fn test_empty_query(&self) -> TestResult {
        match self.query_processor.process_query("", &QueryContext::default()).await {
            Ok(_) => TestResult::failure("Empty query", "Should have failed".to_string()),
            Err(_) => TestResult::success("Empty query", Duration::from_secs(0)),
        }
    }
    
    async fn test_nonsensical_query(&self) -> TestResult {
        let query = "Purple elephant mathematics quantum rainbow seventeen";
        
        match self.query_processor.process_query(query, &QueryContext::default()).await {
            Ok(result) => {
                // Should classify as unknown or handle gracefully
                assert!(matches!(result.intent.intent_type, QueryIntent::Unknown) ||
                       result.intent.confidence < 0.5, "Should handle nonsensical queries");
                
                TestResult::success("Nonsensical query", Duration::from_secs(0))
            }
            Err(_) => {
                // Acceptable to fail on nonsensical queries
                TestResult::success("Nonsensical query", Duration::from_secs(0))
            }
        }
    }
}
```

## File Locations

- `tests/integration/query_system_integration.rs` - Main test suite
- `tests/integration/performance_tests.rs` - Performance validation
- `tests/integration/quality_tests.rs` - Quality assessment
- `tests/integration/stress_tests.rs` - Load testing
- `tests/integration/test_utils.rs` - Test utilities

## Success Criteria

- [ ] All end-to-end scenarios pass
- [ ] Performance targets met in integration
- [ ] Quality metrics achieved
- [ ] System stable under stress
- [ ] Edge cases handled gracefully
- [ ] No memory leaks detected
- [ ] Error recovery functional

## Test Execution

```bash
# Run all integration tests
cargo test --test query_system_integration

# Run specific test suites
cargo test --test query_system_integration end_to_end
cargo test --test query_system_integration performance
cargo test --test query_system_integration quality

# Run with performance monitoring
cargo test --test query_system_integration --features performance-monitoring
```

## Quality Gates

- [ ] 95% test pass rate
- [ ] All performance benchmarks met
- [ ] Quality metrics above thresholds
- [ ] No critical failures under load
- [ ] Proper cleanup after tests

## Next Task

Upon completion, proceed to **41_benchmarking.md**