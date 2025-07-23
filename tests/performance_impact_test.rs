/*!
Performance Impact Assessment for Runtime Profiler
Tests the performance overhead of function tracing on LLMKG operations
*/

use std::sync::Arc;
use std::time::{Duration, Instant};

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::monitoring::collectors::runtime_profiler::RuntimeProfiler;

#[test]
fn test_performance_impact_with_tracing() {
    println!("⚡ Assessing performance impact of runtime profiler tracing...");
    
    // Test parameters
    let num_operations = 1000;
    let embedding_dim = 96;
    
    // Test 1: Performance WITHOUT tracing
    println!("📊 Testing performance WITHOUT tracing...");
    let graph_without_tracing = KnowledgeGraph::new_with_dimension(embedding_dim)
        .expect("Failed to create graph");
    
    let start_without = Instant::now();
    for i in 0..num_operations {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: format!("test_entity_{}", i),
            embedding: vec![0.1; embedding_dim],
        };
        
        let _ = graph_without_tracing.add_entity(entity_data);
        let _ = graph_without_tracing.entity_count();
        
        if i % 100 == 0 {
            let _ = graph_without_tracing.relationship_count();
        }
    }
    let duration_without = start_without.elapsed();
    
    println!("✅ Without tracing: {} operations in {:?}", num_operations, duration_without);
    println!("   Average per operation: {:?}", duration_without / num_operations);
    
    // Test 2: Performance WITH tracing
    println!("📊 Testing performance WITH tracing...");
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut graph_with_tracing = KnowledgeGraph::new_with_dimension(embedding_dim)
        .expect("Failed to create graph");
    graph_with_tracing.set_runtime_profiler(profiler.clone());
    
    let start_with = Instant::now();
    for i in 0..num_operations {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: format!("traced_entity_{}", i),
            embedding: vec![0.1; embedding_dim],
        };
        
        let _ = graph_with_tracing.add_entity(entity_data);
        let _ = graph_with_tracing.entity_count();
        
        if i % 100 == 0 {
            let _ = graph_with_tracing.relationship_count();
        }
    }
    let duration_with = start_with.elapsed();
    
    println!("✅ With tracing: {} operations in {:?}", num_operations, duration_with);
    println!("   Average per operation: {:?}", duration_with / num_operations);
    
    // Calculate overhead
    let overhead = duration_with.as_nanos() as f64 - duration_without.as_nanos() as f64;
    let overhead_percentage = (overhead / duration_without.as_nanos() as f64) * 100.0;
    
    println!("📈 Performance Impact Analysis:");
    println!("   Overhead: {:?}", Duration::from_nanos(overhead as u64));
    println!("   Overhead percentage: {:.2}%", overhead_percentage);
    
    // Get tracing statistics
    let metrics = profiler.get_metrics();
    println!("   Functions traced: {}", metrics.function_call_count.len());
    println!("   Total traces: {}", metrics.function_call_count.values().sum::<u64>());
    println!("   Timeline events: {}", metrics.execution_timeline.len());
    
    // Performance thresholds
    let max_acceptable_overhead = 50.0; // 50% overhead is reasonable for debugging/monitoring
    
    if overhead_percentage <= max_acceptable_overhead {
        println!("🎉 PASSED: Tracing overhead is acceptable ({:.2}% <= {:.2}%)", 
                 overhead_percentage, max_acceptable_overhead);
    } else {
        println!("⚠️ WARNING: Tracing overhead is high ({:.2}% > {:.2}%)", 
                 overhead_percentage, max_acceptable_overhead);
        println!("   Consider optimizing tracing or making it optional for production");
    }
    
    // Verify that tracing actually captured data
    assert!(metrics.function_call_count.len() > 0, "No functions were traced!");
    assert!(metrics.execution_timeline.len() > 0, "No timeline events captured!");
    
    // Check for expected function traces
    let expected_functions = ["add_entity", "entity_count", "insert_entity"];
    let mut traced_functions = 0;
    
    for expected in &expected_functions {
        if metrics.function_call_count.contains_key(*expected) {
            let count = metrics.function_call_count[*expected];
            println!("   ✅ {} traced {} times", expected, count);
            traced_functions += 1;
        }
    }
    
    assert!(traced_functions >= 2, "Expected at least 2 core functions to be traced");
    
    println!("🏁 Performance impact assessment completed successfully!");
}

#[test]
fn test_tracing_disable_functionality() {
    println!("🔧 Testing tracing enable/disable functionality...");
    
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    graph.set_runtime_profiler(profiler.clone());
    
    // Test with tracing enabled
    println!("📊 Testing with tracing enabled...");
    profiler.enable_profiling(true);
    
    for i in 0..10 {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: format!("enabled_entity_{}", i),
            embedding: vec![0.1; 96],
        };
        let _ = graph.add_entity(entity_data);
    }
    
    let metrics_enabled = profiler.get_metrics();
    let calls_when_enabled = metrics_enabled.function_call_count.values().sum::<u64>();
    
    println!("   Function calls traced (enabled): {}", calls_when_enabled);
    
    // Clear metrics and disable tracing
    profiler.clear_metrics();
    profiler.enable_profiling(false);
    
    println!("📊 Testing with tracing disabled...");
    
    for i in 0..10 {
        let entity_data = EntityData {
            type_id: (i + 11) as u16,
            properties: format!("disabled_entity_{}", i),
            embedding: vec![0.2; 96],
        };
        let _ = graph.add_entity(entity_data);
    }
    
    let metrics_disabled = profiler.get_metrics();
    let calls_when_disabled = metrics_disabled.function_call_count.values().sum::<u64>();
    
    println!("   Function calls traced (disabled): {}", calls_when_disabled);
    
    // Verify that disabling works
    assert!(calls_when_enabled > 0, "Tracing should capture calls when enabled");
    assert_eq!(calls_when_disabled, 0, "Tracing should not capture calls when disabled");
    
    // Re-enable and test again
    profiler.enable_profiling(true);
    let _ = graph.entity_count();
    
    let metrics_re_enabled = profiler.get_metrics();
    let calls_after_re_enable = metrics_re_enabled.function_call_count.values().sum::<u64>();
    
    println!("   Function calls traced (re-enabled): {}", calls_after_re_enable);
    assert!(calls_after_re_enable > 0, "Tracing should work again after re-enabling");
    
    println!("✅ Tracing enable/disable functionality works correctly!");
    println!("🎉 All tracing control tests PASSED!");
}

#[test]
fn test_memory_usage_tracking() {
    println!("💾 Testing memory usage tracking in runtime profiler...");
    
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    graph.set_runtime_profiler(profiler.clone());
    
    // Perform operations that should trigger memory tracking
    for i in 0..20 {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: format!("memory_test_entity_{}", i),
            embedding: vec![0.1; 96],
        };
        
        let _ = graph.add_entity(entity_data);
        
        // Simulate memory allocation tracking
        profiler.record_memory_allocation(
            "test_allocation".to_string(), 
            1024 * (i + 1) as u64
        );
    }
    
    let metrics = profiler.get_metrics();
    
    println!("📊 Memory tracking results:");
    println!("   Memory allocations tracked: {}", metrics.memory_allocations.len());
    
    for (function_name, allocation_size) in &metrics.memory_allocations {
        println!("   {}: {} bytes", function_name, allocation_size);
    }
    
    // Verify memory tracking
    assert!(metrics.memory_allocations.len() > 0, "No memory allocations tracked");
    
    if let Some(test_allocation) = metrics.memory_allocations.get("test_allocation") {
        println!("   ✅ Test allocations total: {} bytes", test_allocation);
        assert!(*test_allocation > 0, "Test allocation size should be positive");
    }
    
    // Check for performance bottleneck detection
    println!("🔍 Performance bottleneck analysis:");
    println!("   Bottlenecks detected: {}", metrics.performance_bottlenecks.len());
    
    for bottleneck in &metrics.performance_bottlenecks {
        println!("   ⚠️ {}: {} (severity: {:.2})", 
                 bottleneck.function_name, 
                 bottleneck.description, 
                 bottleneck.severity);
    }
    
    println!("✅ Memory usage tracking test completed!");
}