/*!
Simple RuntimeProfiler Integration Test  
Tests that RuntimeProfiler captures real function execution from basic LLMKG operations
*/

use std::sync::Arc;

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::monitoring::collectors::runtime_profiler::RuntimeProfiler;

#[test]
fn test_simple_knowledge_graph_tracing() {
    println!("🔍 Testing basic KnowledgeGraph function tracing...");
    
    // Initialize RuntimeProfiler
    let profiler = Arc::new(RuntimeProfiler::new());
    
    // Create KnowledgeGraph with profiler
    let mut knowledge_graph = KnowledgeGraph::new_with_dimension(96)
        .expect("Failed to create graph");
    knowledge_graph.set_runtime_profiler(profiler.clone());
    
    println!("📊 Testing basic operations...");
    
    // Test basic count operations (should be traced)
    let initial_count = knowledge_graph.entity_count();
    let rel_count = knowledge_graph.relationship_count();
    
    println!("✅ Initial counts - Entities: {}, Relationships: {}", initial_count, rel_count);
    
    // Add an entity (should be traced) 
    let entity_data = EntityData {
        type_id: 1,
        properties: "test_entity".to_string(),
        embedding: vec![0.1; 96],
    };
    
    let entity_result = knowledge_graph.add_entity(entity_data.clone());
    match entity_result {
        Ok(entity_key) => {
            println!("✅ Added entity: {:?}", entity_key);
        },
        Err(e) => {
            println!("⚠️ Failed to add entity: {}", e);
        }
    }
    
    // Check counts again (should be traced)
    let final_count = knowledge_graph.entity_count();
    println!("✅ Final entity count: {}", final_count);
    
    // Get runtime metrics to validate tracing occurred
    let metrics = profiler.get_metrics();
    
    println!("📈 Tracing Results:");
    println!("   Function calls tracked: {}", metrics.function_call_count.len());
    println!("   Total calls: {}", metrics.function_call_count.values().sum::<u64>());
    println!("   Timeline events: {}", metrics.execution_timeline.len());
    
    // Print detailed function call information
    for (function_name, call_count) in &metrics.function_call_count {
        println!("   📊 {}: {} calls", function_name, call_count);
        
        if let Some(stats) = metrics.function_execution_times.get(function_name) {
            println!("      └─ Avg duration: {:?}", stats.avg_duration);
        }
    }
    
    // Verify that some functions were traced
    assert!(metrics.function_call_count.len() > 0, "No functions were traced!");
    assert!(metrics.execution_timeline.len() > 0, "No timeline events recorded!");
    
    // Check for expected functions
    let expected_functions = ["entity_count", "relationship_count"];
    let mut found_functions = 0;
    
    for expected in &expected_functions {
        if metrics.function_call_count.contains_key(*expected) {
            println!("✅ Successfully traced: {}", expected);
            found_functions += 1;
        } else {
            println!("⚠️ Missing trace for: {}", expected);
        }
    }
    
    assert!(found_functions > 0, "None of the expected functions were traced!");
    
    println!("🎉 Basic tracing validation PASSED!");
}

#[test]
fn test_runtime_profiler_metrics_collection() {
    println!("📊 Testing RuntimeProfiler metrics collection...");
    
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    graph.set_runtime_profiler(profiler.clone());
    
    // Perform multiple operations
    for i in 0..5 {
        let _ = graph.entity_count();
        let _ = graph.relationship_count();
    }
    
    let metrics = profiler.get_metrics();
    
    println!("📈 Collected Metrics:");
    println!("   Total function types: {}", metrics.function_call_count.len());
    
    for (func, count) in &metrics.function_call_count {
        println!("   {} called {} times", func, count);
    }
    
    // Verify metrics are reasonable
    assert!(metrics.function_call_count.values().sum::<u64>() >= 10, 
            "Expected at least 10 total function calls");
    
    println!("✅ Metrics collection validation PASSED!");
}