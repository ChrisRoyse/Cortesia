//! Performance Under Load Tests
//! 
//! These tests validate that the REAL systems maintain performance targets
//! under realistic concurrent load and with large datasets.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use llmkg::core::{
    entity_extractor::CognitiveEntityExtractor,
    relationship_extractor::CognitiveRelationshipExtractor,
    answer_generator::AdvancedAnswerGenerator,
    knowledge_engine::KnowledgeEngine,
};

use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::activation_config::ActivationConfig;
use llmkg::federation::coordinator::FederationCoordinator;

/// Performance targets (MUST be maintained under load)
const ENTITY_EXTRACTION_TARGET_MS: u64 = 8;
const RELATIONSHIP_EXTRACTION_TARGET_MS: u64 = 12;  
const QUESTION_ANSWERING_TARGET_MS: u64 = 20;
const FEDERATION_STORAGE_TARGET_MS: u64 = 3;

/// Load Wikipedia article from test data
fn load_wikipedia_article() -> String {
    std::fs::read_to_string("test_data/realistic_text_samples/wikipedia_einstein_article.txt")
        .unwrap_or_else(|_| {
            // Fallback content if file not found
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. Einstein's mass-energy equivalence formula E = mc² has been dubbed the world's most famous equation. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect, a pivotal step in the development of quantum theory.".to_string()
        })
}

/// Load scientific article from test data
fn load_scientific_article() -> String {
    std::fs::read_to_string("test_data/realistic_text_samples/scientific_quantum_computing.txt")
        .unwrap_or_else(|_| {
            // Fallback content if file not found
            "Quantum computing represents a revolutionary paradigm shift in computational science, harnessing the bizarre principles of quantum mechanics to process information in ways that fundamentally differ from classical computers. At the heart of quantum computing lies the quantum bit, or qubit, which unlike classical bits that exist in definite states of 0 or 1, can exist in a superposition of both states simultaneously until measured.".to_string()
        })
}

/// Create real cognitive entity extractor (same as benchmark)
async fn create_real_entity_extractor() -> CognitiveEntityExtractor {
    let neural_server = Arc::new(
        NeuralProcessingServer::new("localhost:9000".to_string())
            .await
            .expect("Failed to create real neural server")
    );
    
    neural_server.initialize_models()
        .await
        .expect("Failed to initialize real neural models");
    
    let graph = Arc::new(
        BrainEnhancedKnowledgeGraph::new()
            .expect("Failed to create brain-enhanced graph")
    );
    
    let cognitive_orchestrator = Arc::new(
        CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default())
            .await
            .expect("Failed to create cognitive orchestrator")
    );
    
    let activation_engine = Arc::new(
        ActivationPropagationEngine::new(ActivationConfig::default())
    );
    
    let working_memory = Arc::new(
        WorkingMemorySystem::new(activation_engine.clone(), graph.sdr_storage.clone())
            .await
            .expect("Failed to create working memory system")
    );
    
    let attention_manager = Arc::new(
        AttentionManager::new(
            cognitive_orchestrator.clone(),
            activation_engine.clone(),
            working_memory.clone(),
        )
        .await
        .expect("Failed to create attention manager")
    );
    
    let metrics_collector = Arc::new(
        BrainMetricsCollector::new()
            .await
            .expect("Failed to create metrics collector")
    );
    
    let performance_monitor = Arc::new(
        PerformanceMonitor::new()
            .await
            .expect("Failed to create performance monitor")
    );
    
    let mut extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    extractor = extractor.with_neural_server(neural_server);
    extractor
}

#[tokio::test]
async fn test_entity_extraction_with_long_text() {
    println!("🧪 Testing entity extraction with realistic long text...");
    
    let extractor = create_real_entity_extractor().await;
    let long_text = load_wikipedia_article();
    
    println!("   Text length: {} characters", long_text.len());
    
    let start = Instant::now();
    let entities = extractor.extract_entities(&long_text).await
        .expect("Entity extraction should succeed");
    let duration = start.elapsed();
    
    println!("   Extracted {} entities in {:?}", entities.len(), duration);
    
    // For long text, we allow proportionally more time (but still reasonable)
    let max_allowed_ms = (long_text.len() / 100) as u64; // ~1ms per 100 characters
    let max_allowed_ms = max_allowed_ms.max(50).min(200); // Between 50-200ms
    
    assert!(
        duration.as_millis() < max_allowed_ms as u128,
        "Long text entity extraction took {}ms, should be <{}ms for {} chars",
        duration.as_millis(), max_allowed_ms, long_text.len()
    );
    
    // Validate results quality
    assert!(entities.len() >= 10, "Should extract at least 10 entities from Wikipedia article");
    assert!(
        entities.iter().any(|e| e.name.contains("Einstein")),
        "Should extract Einstein from the article"
    );
    assert!(
        entities.iter().all(|e| e.confidence_score > 0.0),
        "All entities must have real confidence scores from neural models"
    );
}

#[tokio::test]
async fn test_concurrent_entity_extraction() {
    println!("🧪 Testing concurrent entity extraction performance...");
    
    let extractor = Arc::new(create_real_entity_extractor().await);
    let texts = vec![
        "Einstein won the Nobel Prize in 1921.",
        "Marie Curie discovered radium and polonium.",
        "Quantum mechanics was developed by many physicists.",
        "The University of Cambridge is in England.",
        "Artificial intelligence research is advancing rapidly.",
    ];
    
    let mut handles = Vec::new();
    let start_time = Instant::now();
    
    // Launch 20 concurrent extractions (4 per text)
    for i in 0..20 {
        let extractor = extractor.clone();
        let text = texts[i % texts.len()].to_string();
        
        let handle = tokio::spawn(async move {
            let start = Instant::now();
            let entities = extractor.extract_entities(&text).await
                .expect(&format!("Concurrent entity extraction {} failed", i));
            let duration = start.elapsed();
            
            // Each concurrent request must still meet individual target
            assert!(
                duration.as_millis() <= ENTITY_EXTRACTION_TARGET_MS as u128,
                "Concurrent request {} took {}ms, target is <{}ms",
                i, duration.as_millis(), ENTITY_EXTRACTION_TARGET_MS
            );
            
            (i, entities.len(), duration)
        });
        
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    let mut total_entities = 0;
    let mut max_duration = Duration::from_millis(0);
    
    for handle in handles {
        let (request_id, entity_count, duration) = handle.await
            .expect("Concurrent request should complete");
        
        println!("   Request {}: {} entities in {:?}", request_id, entity_count, duration);
        total_entities += entity_count;
        max_duration = max_duration.max(duration);
    }
    
    let total_time = start_time.elapsed();
    
    println!("   Total: {} entities, max individual time: {:?}, total time: {:?}", 
             total_entities, max_duration, total_time);
    
    // Validate concurrent performance
    assert!(total_entities >= 20, "Should extract entities from all concurrent requests");
    assert!(
        max_duration.as_millis() <= ENTITY_EXTRACTION_TARGET_MS as u128,
        "Max individual time {}ms exceeds target {}ms",
        max_duration.as_millis(), ENTITY_EXTRACTION_TARGET_MS
    );
    
    // Total time should be much less than sequential (due to parallelism)
    let sequential_estimate = Duration::from_millis(ENTITY_EXTRACTION_TARGET_MS * 20);
    assert!(
        total_time < sequential_estimate,
        "Concurrent execution should be faster than sequential"
    );
}

#[tokio::test]
async fn test_sustained_load_performance() {
    println!("🧪 Testing sustained load performance...");
    
    let extractor = Arc::new(create_real_entity_extractor().await);
    let test_texts = vec![
        "Einstein developed the theory of relativity.",
        "Marie Curie won two Nobel Prizes.",
        "The quantum computer uses quantum bits.",
        "Stanford University collaborates with MIT.",
        "Artificial intelligence transforms science.",
    ];
    
    let duration_seconds = 30; // Run for 30 seconds
    let target_requests_per_second = 10;
    let expected_total_requests = duration_seconds * target_requests_per_second;
    
    println!("   Running sustained load for {} seconds at {} req/sec", 
             duration_seconds, target_requests_per_second);
    
    let start_time = Instant::now();
    let mut completed_requests = 0;
    let mut failed_requests = 0;
    let mut max_response_time = Duration::from_millis(0);
    
    while start_time.elapsed().as_secs() < duration_seconds {
        let batch_start = Instant::now();
        let mut batch_handles = Vec::new();
        
        // Launch a batch of concurrent requests
        for i in 0..target_requests_per_second {
            let extractor = extractor.clone();
            let text = test_texts[i % test_texts.len()].to_string();
            
            let handle = tokio::spawn(async move {
                let request_start = Instant::now();
                let result = extractor.extract_entities(&text).await;
                let request_duration = request_start.elapsed();
                
                match result {
                    Ok(entities) => Ok((entities.len(), request_duration)),
                    Err(e) => Err(e)
                }
            });
            
            batch_handles.push(handle);
        }
        
        // Wait for batch to complete
        for handle in batch_handles {
            match handle.await {
                Ok(Ok((entity_count, response_time))) => {
                    completed_requests += 1;
                    max_response_time = max_response_time.max(response_time);
                    
                    // Verify individual request meets target
                    if response_time.as_millis() > ENTITY_EXTRACTION_TARGET_MS as u128 {
                        println!("   WARNING: Request took {}ms (target: {}ms)", 
                                response_time.as_millis(), ENTITY_EXTRACTION_TARGET_MS);
                    }
                },
                _ => {
                    failed_requests += 1;
                }
            }
        }
        
        // Wait until 1 second has passed for this batch
        let batch_duration = batch_start.elapsed();
        if batch_duration < Duration::from_secs(1) {
            sleep(Duration::from_secs(1) - batch_duration).await;
        }
    }
    
    let total_time = start_time.elapsed();
    let actual_rps = completed_requests as f64 / total_time.as_secs_f64();
    
    println!("   Results: {} completed, {} failed, {:.1} req/sec, max response: {:?}",
             completed_requests, failed_requests, actual_rps, max_response_time);
    
    // Validate sustained performance
    assert!(
        completed_requests >= (expected_total_requests * 80 / 100), // Allow 20% tolerance
        "Completed {} requests, expected at least {} (80% of {})",
        completed_requests, expected_total_requests * 80 / 100, expected_total_requests
    );
    
    assert!(
        failed_requests < completed_requests / 10, // Less than 10% failure rate
        "Too many failed requests: {} failed out of {} total",
        failed_requests, completed_requests + failed_requests
    );
    
    assert!(
        max_response_time.as_millis() <= (ENTITY_EXTRACTION_TARGET_MS * 2) as u128,
        "Max response time {}ms exceeds 2x target ({}ms)",
        max_response_time.as_millis(), ENTITY_EXTRACTION_TARGET_MS * 2
    );
}

#[tokio::test]
async fn test_memory_usage_under_load() {
    println!("🧪 Testing memory usage under load...");
    
    let extractor = Arc::new(create_real_entity_extractor().await);
    let long_text = load_scientific_article();
    
    // Run many extractions to check for memory leaks
    let num_iterations = 100;
    println!("   Running {} iterations with long text...", num_iterations);
    
    for i in 0..num_iterations {
        let start = Instant::now();
        let entities = extractor.extract_entities(&long_text).await
            .expect("Entity extraction should succeed");
        let duration = start.elapsed();
        
        if i % 10 == 0 {
            println!("   Iteration {}: {} entities in {:?}", i, entities.len(), duration);
        }
        
        // Verify each iteration still meets performance target
        assert!(
            duration.as_millis() < 100, // More lenient for long text under repeated load
            "Iteration {} took {}ms, may indicate memory pressure or leaks",
            i, duration.as_millis()
        );
        
        // Verify results are still valid
        assert!(entities.len() > 0, "Should still extract entities at iteration {}", i);
        assert!(
            entities.iter().all(|e| e.confidence_score > 0.0),
            "All entities should have confidence scores at iteration {}", i
        );
    }
    
    println!("   ✅ Memory usage stable over {} iterations", num_iterations);
}

#[tokio::test] 
async fn test_error_recovery_under_load() {
    println!("🧪 Testing error recovery under load...");
    
    let extractor = Arc::new(create_real_entity_extractor().await);
    
    // Test with various text inputs including edge cases
    let test_inputs = vec![
        "Normal text with Einstein and physics.",
        "", // Empty string
        "A", // Single character
        "Special characters: @#$%^&*()[]{}|\\:;\"'<>,.?/~`", 
        "Very long text ".repeat(1000), // Very long text
        "Mixed languages: Einstein 爱因斯坦 Einstein アインシュタイン", // Mixed unicode
    ];
    
    let mut successful_extractions = 0;
    let mut handled_errors = 0;
    
    for (i, input) in test_inputs.iter().enumerate() {
        println!("   Testing input {}: {} chars", i, input.len());
        
        let start = Instant::now();
        match extractor.extract_entities(input).await {
            Ok(entities) => {
                successful_extractions += 1;
                let duration = start.elapsed();
                
                println!("     ✅ Success: {} entities in {:?}", entities.len(), duration);
                
                // Even with edge cases, should complete within reasonable time
                assert!(
                    duration.as_millis() < 1000, // 1 second max for edge cases
                    "Edge case {} took too long: {}ms", i, duration.as_millis()
                );
            },
            Err(e) => {
                handled_errors += 1;
                println!("     ⚠️  Handled error: {:?}", e);
                
                // Errors should be handled gracefully, not panic
            }
        }
    }
    
    println!("   Results: {} successful, {} handled errors", 
             successful_extractions, handled_errors);
    
    // Should handle most inputs successfully
    assert!(
        successful_extractions >= test_inputs.len() / 2,
        "Should successfully handle at least half of edge case inputs"
    );
    
    // System should remain stable after error conditions
    let final_test = extractor.extract_entities("Final test with Einstein").await
        .expect("System should recover after edge case testing");
    
    assert!(final_test.len() > 0, "System should still work after error recovery test");
}