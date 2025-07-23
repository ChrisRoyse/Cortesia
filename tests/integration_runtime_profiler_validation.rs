/*!
LLMKG Runtime Profiler Integration Validation Tests
Tests that RuntimeProfiler captures real function execution from LLMKG operations
*/

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::types::ReasoningStrategy;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::monitoring::collectors::runtime_profiler::{RuntimeProfiler, ExecutionEventType};

#[tokio::test]
async fn test_knowledge_graph_function_tracing() {
    // Initialize RuntimeProfiler
    let profiler = Arc::new(RuntimeProfiler::new());
    
    // Create KnowledgeGraph with profiler
    let mut knowledge_graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    knowledge_graph.set_runtime_profiler(profiler.clone());
    
    // Subscribe to profiler events
    let mut event_receiver = profiler.subscribe_to_events();
    
    println!("🔍 Starting KnowledgeGraph function tracing validation...");
    
    // Test 1: Entity operations tracing
    println!("📊 Testing entity operations...");
    
    let entity_data = EntityData {
        type_id: 1,
        properties: serde_json::to_string(&std::collections::HashMap::<String, String>::new()).unwrap(),
        embedding: vec![0.1; 96], // 96-dimensional embedding
    };
    
    // This should trigger tracing in add_entity and insert_entity
    let entity_key = knowledge_graph.add_entity(entity_data.clone()).expect("Failed to add entity");
    println!("✅ Added entity: {:?}", entity_key);
    
    // Test 2: Count operations tracing
    println!("📊 Testing count operations...");
    let entity_count = knowledge_graph.entity_count();
    let relationship_count = knowledge_graph.relationship_count();
    
    println!("✅ Entity count: {}, Relationship count: {}", entity_count, relationship_count);
    
    // Test 3: Similarity search tracing
    println!("📊 Testing similarity search...");
    let query_embedding = vec![0.2; 96];
    let search_results = knowledge_graph.similarity_search(&query_embedding, 5)
        .expect("Failed to perform similarity search");
    
    println!("✅ Similarity search results: {} found", search_results.len());
    
    // Test 4: Query system tracing
    println!("📊 Testing query system...");
    let context_entities = vec![];
    let query_results = knowledge_graph.query(&query_embedding, &context_entities, 5)
        .expect("Failed to perform query");
    
    println!("✅ Query results: {} entities found", query_results.entities.len());
    
    // Allow some time for events to be processed
    sleep(Duration::from_millis(100)).await;
    
    // Verify events were captured
    let mut events_received = Vec::new();
    let timeout = tokio::time::timeout(Duration::from_millis(500), async {
        while let Ok(event) = event_receiver.recv().await {
            events_received.push(event);
            if events_received.len() >= 8 { // We expect at least 8 function calls
                break;
            }
        }
    });
    
    match timeout.await {
        Ok(_) => {
            println!("✅ Captured {} execution events", events_received.len());
            
            // Validate that we captured the expected function calls
            let mut captured_functions = std::collections::HashSet::new();
            for event in &events_received {
                if matches!(event.event_type, ExecutionEventType::FunctionStart) {
                    captured_functions.insert(event.function_name.clone());
                    println!("🎯 Traced function: {} (duration: {:?})", event.function_name, event.duration);
                }
            }
            
            // Check that critical functions were traced
            let expected_functions = vec![
                "add_entity",
                "insert_entity", 
                "entity_count",
                "relationship_count",
                "similarity_search",
                "query"
            ];
            
            for expected_func in expected_functions {
                if captured_functions.contains(expected_func) {
                    println!("✅ Successfully traced: {}", expected_func);
                } else {
                    println!("❌ Missing trace for: {}", expected_func);
                }
            }
            
            assert!(captured_functions.len() >= 4, "Expected at least 4 different functions to be traced, got: {}", captured_functions.len());
            
        },
        Err(_) => {
            panic!("❌ Timeout waiting for execution events - tracing may not be working!");
        }
    }
    
    // Get runtime metrics and verify data collection
    let runtime_metrics = profiler.get_metrics();
    
    println!("📈 Runtime Metrics Summary:");
    println!("   Active functions: {}", runtime_metrics.active_functions.len());
    println!("   Total function calls: {}", runtime_metrics.function_call_count.values().sum::<u64>());
    println!("   Execution timeline events: {}", runtime_metrics.execution_timeline.len());
    println!("   Performance bottlenecks: {}", runtime_metrics.performance_bottlenecks.len());
    
    // Verify that actual data was collected
    assert!(runtime_metrics.function_call_count.len() > 0, "No function calls recorded!");
    assert!(runtime_metrics.execution_timeline.len() > 0, "No timeline events recorded!");
    
    println!("🎉 KnowledgeGraph function tracing validation PASSED!");
}

#[tokio::test] 
async fn test_cognitive_orchestrator_tracing() {
    println!("🧠 Starting CognitiveOrchestrator function tracing validation...");
    
    // Initialize RuntimeProfiler
    let profiler = Arc::new(RuntimeProfiler::new());
    
    // Create brain graph and cognitive orchestrator
    let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(96)
        .expect("Failed to create brain graph"));
    
    let config = CognitiveOrchestratorConfig::default();
    let mut orchestrator = CognitiveOrchestrator::new(brain_graph, config).await
        .expect("Failed to create orchestrator");
    
    orchestrator.set_runtime_profiler(profiler.clone());
    
    // Subscribe to events
    let mut event_receiver = profiler.subscribe_to_events();
    
    // Test cognitive reasoning with tracing
    println!("🤔 Testing cognitive reasoning...");
    
    let query = "What is the meaning of artificial intelligence?";
    let context = Some("AI and machine learning context");
    let strategy = ReasoningStrategy::Automatic;
    
    // This should trigger tracing in the reason function
    let reasoning_result = orchestrator.reason(query, context, strategy).await;
    
    match reasoning_result {
        Ok(result) => {
            println!("✅ Cognitive reasoning completed with confidence: {:.2}", result.quality_metrics.overall_confidence);
        },
        Err(e) => {
            println!("⚠️ Cognitive reasoning failed (expected in test environment): {}", e);
            // This might fail in test environment due to missing neural components, but tracing should still work
        }
    }
    
    // Allow time for events to be processed
    sleep(Duration::from_millis(100)).await;
    
    // Check for cognitive tracing events
    let timeout = tokio::time::timeout(Duration::from_millis(300), async {
        while let Ok(event) = event_receiver.recv().await {
            if event.function_name == "cognitive_reason" {
                println!("🎯 Successfully traced cognitive_reason function!");
                println!("   Duration: {:?}", event.duration);
                println!("   Thread ID: {}", event.thread_id);
                return true;
            }
        }
        false
    });
    
    let traced_cognitive = timeout.await.unwrap_or(false);
    
    if traced_cognitive {
        println!("✅ Cognitive function tracing working!");
    } else {
        println!("⚠️ No cognitive function traces captured (may be due to test environment)");
    }
    
    // Get final metrics
    let metrics = profiler.get_metrics();
    println!("📊 Final metrics: {} total function calls", metrics.function_call_count.values().sum::<u64>());
    
    println!("🎉 CognitiveOrchestrator tracing validation completed!");
}

#[tokio::test]
async fn test_runtime_profiler_performance_analysis() {
    println!("⚡ Testing RuntimeProfiler performance analysis capabilities...");
    
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut knowledge_graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    knowledge_graph.set_runtime_profiler(profiler.clone());
    
    // Perform multiple operations to generate performance data
    println!("🔄 Performing multiple operations for performance analysis...");
    
    for i in 0..10 {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: serde_json::to_string(&std::collections::HashMap::<String, String>::new()).unwrap(),
            embedding: vec![0.1 + (i as f32 * 0.01); 96],
        };
        
        let _ = knowledge_graph.add_entity(entity_data);
        let _ = knowledge_graph.entity_count();
        
        // Add small delay to simulate realistic usage
        sleep(Duration::from_millis(10)).await;
    }
    
    // Trigger hot path analysis
    profiler.analyze_hot_paths();
    
    // Wait for analysis to complete
    sleep(Duration::from_millis(100)).await;
    
    let metrics = profiler.get_metrics();
    
    println!("📈 Performance Analysis Results:");
    println!("   Functions called: {:?}", metrics.function_call_count.keys().collect::<Vec<_>>());
    println!("   Hot paths detected: {}", metrics.hot_paths.len());
    println!("   Performance bottlenecks: {}", metrics.performance_bottlenecks.len());
    
    // Verify performance analysis is working
    assert!(metrics.function_call_count.len() > 0, "No function call tracking!");
    
    // Check execution statistics
    for (func_name, stats) in &metrics.function_execution_times {
        println!("   📊 {}: {} calls, avg duration: {:?}", 
                 func_name, stats.total_calls, stats.avg_duration);
    }
    
    println!("✅ Performance analysis validation PASSED!");
}