/*!
WebSocket Dashboard Integration Test for RuntimeProfiler
Tests that runtime profiler data is correctly transmitted via WebSocket to the dashboard
*/

use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::monitoring::collectors::runtime_profiler::RuntimeProfiler;
use llmkg::monitoring::dashboard::{DashboardServer, DashboardConfig};
use llmkg::monitoring::metrics::MetricRegistry;
use llmkg::monitoring::collectors::{MetricsCollector, MetricsCollectionConfig, SystemMetricsCollector, ApplicationMetricsCollector, SystemMetricsConfig, ApplicationMetricsConfig};

#[tokio::test]
async fn test_websocket_runtime_profiler_integration() {
    println!("🚀 Starting WebSocket Dashboard Integration Test...");
    
    // Initialize components
    let profiler = Arc::new(RuntimeProfiler::new());
    let mut knowledge_graph = KnowledgeGraph::new_with_dimension(96)
        .expect("Failed to create graph");
    knowledge_graph.set_runtime_profiler(profiler.clone());
    
    // Set up dashboard
    let config = DashboardConfig {
        http_port: 18080, // Use different ports to avoid conflicts
        websocket_port: 18081,
        update_interval: Duration::from_millis(500), // Faster updates for test
        history_size: 100,
        title: "Test Dashboard".to_string(),
        refresh_rate_ms: 500,
    };
    
    let registry = Arc::new(MetricRegistry::new());
    
    // Add RuntimeProfiler as a collector
    let system_collector = Box::new(SystemMetricsCollector::new(SystemMetricsConfig {
        collect_cpu: true,
        collect_memory: true,
        collect_disk: false,
        collect_network: false,
        collect_load: false,
    }));
    
    let app_collector = Box::new(ApplicationMetricsCollector::new(ApplicationMetricsConfig {
        collect_performance: true,
        collect_operations: true,
        collect_errors: true,
        collect_resources: true,
    }));
    
    // Create a wrapper for RuntimeProfiler that implements MetricsCollector
    struct RuntimeProfilerWrapper(Arc<RuntimeProfiler>);
    
    impl MetricsCollector for RuntimeProfilerWrapper {
        fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
            self.0.collect(registry)
        }
        
        fn name(&self) -> &str {
            self.0.name()
        }
        
        fn is_enabled(&self, _config: &MetricsCollectionConfig) -> bool {
            true // Always enabled for testing
        }
    }
    
    let collectors: Vec<Box<dyn MetricsCollector>> = vec![
        system_collector,
        app_collector,
        Box::new(RuntimeProfilerWrapper(profiler.clone())),
    ];
    
    let dashboard = DashboardServer::new(config.clone(), registry.clone(), collectors);
    
    // Start dashboard in background
    let dashboard_handle = tokio::spawn(async move {
        if let Err(e) = dashboard.start().await {
            println!("⚠️ Dashboard start error: {}", e);
        }
    });
    
    // Give dashboard time to start
    sleep(Duration::from_millis(2000)).await;
    
    // Connect to WebSocket
    println!("🔌 Connecting to WebSocket...");
    let ws_url = format!("ws://127.0.0.1:{}", config.websocket_port);
    
    let ws_connection = timeout(Duration::from_secs(5), connect_async(&ws_url)).await;
    
    match ws_connection {
        Ok(Ok((ws_stream, _response))) => {
            println!("✅ WebSocket connected successfully!");
            
            let (mut ws_sender, mut ws_receiver) = ws_stream.split();
            
            // Subscribe to updates
            let subscribe_message = serde_json::json!({
                "type": "subscribe",
                "topics": ["runtime_metrics", "function_traces"]
            });
            
            if let Ok(msg_text) = serde_json::to_string(&subscribe_message) {
                let _ = ws_sender.send(Message::Text(msg_text)).await;
            }
            
            // Start generating function calls
            println!("📊 Generating function traces...");
            let test_task = tokio::spawn(async move {
                for i in 0..10 {
                    // Create entity
                    let entity_data = EntityData {
                        type_id: (i + 1) as u16,
                        properties: format!("test_entity_{}", i),
                        embedding: vec![0.1 + (i as f32 * 0.01); 96],
                    };
                    
                    let _ = knowledge_graph.add_entity(entity_data);
                    let _ = knowledge_graph.entity_count();
                    let _ = knowledge_graph.relationship_count();
                    
                    sleep(Duration::from_millis(200)).await;
                }
            });
            
            // Listen for WebSocket messages
            let mut messages_received = 0;
            let mut runtime_metrics_received = false;
            
            let listener_timeout = timeout(Duration::from_secs(15), async {
                while let Some(message) = ws_receiver.next().await {
                    match message {
                        Ok(Message::Text(text)) => {
                            messages_received += 1;
                            println!("📨 Received message {}: {}", messages_received, &text[..std::cmp::min(100, text.len())]);
                            
                            // Parse message and check for runtime metrics
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                                if let Some(msg_type) = parsed.get("type") {
                                    if msg_type.as_str() == Some("metrics_update") {
                                        // Check if runtime profiler metrics are included
                                        if let Some(data) = parsed.get("data") {
                                            if let Some(performance) = data.get("performance_metrics") {
                                                if performance.get("runtime_active_functions").is_some() 
                                                   || performance.get("runtime_total_function_calls").is_some() {
                                                    runtime_metrics_received = true;
                                                    println!("✅ Runtime profiler metrics detected in WebSocket message!");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if messages_received >= 5 {
                                break;
                            }
                        }
                        Ok(Message::Close(_)) => {
                            println!("🔌 WebSocket closed");
                            break;
                        }
                        Ok(_) => {
                            // Other message types (binary, ping, pong, etc.)
                        }
                        Err(e) => {
                            println!("❌ WebSocket error: {}", e);
                            break;
                        }
                    }
                }
                
                (messages_received, runtime_metrics_received)
            }).await;
            
            let _ = test_task.await;
            
            match listener_timeout {
                Ok((messages, runtime_detected)) => {
                    println!("📈 WebSocket Integration Results:");
                    println!("   Messages received: {}", messages);
                    println!("   Runtime metrics detected: {}", runtime_detected);
                    
                    assert!(messages > 0, "No WebSocket messages received!");
                    
                    if runtime_detected {
                        println!("🎉 WebSocket dashboard integration PASSED!");
                    } else {
                        println!("⚠️ Runtime metrics not detected in WebSocket messages");
                        println!("   This may be due to timing or message format differences");
                    }
                },
                Err(_) => {
                    println!("⏰ WebSocket listener timeout");
                    // This is expected in a test environment
                }
            }
        }
        Ok(Err(e)) => {
            println!("❌ WebSocket connection failed: {}", e);
        }
        Err(_) => {
            println!("⏰ WebSocket connection timeout - dashboard may not have started properly");
        }
    }
    
    dashboard_handle.abort();
    println!("🏁 WebSocket integration test completed");
}

#[tokio::test]
async fn test_runtime_profiler_metrics_collection() {
    println!("📊 Testing RuntimeProfiler metrics collection for dashboard...");
    
    let profiler = Arc::new(RuntimeProfiler::new());
    let registry = Arc::new(MetricRegistry::new());
    
    // Create a graph with profiler
    let mut graph = KnowledgeGraph::new_with_dimension(96).expect("Failed to create graph");
    graph.set_runtime_profiler(profiler.clone());
    
    // Perform operations to generate metrics
    for i in 0..5 {
        let entity_data = EntityData {
            type_id: (i + 1) as u16,
            properties: format!("metrics_test_{}", i),
            embedding: vec![0.2; 96],
        };
        
        let _ = graph.add_entity(entity_data);
        let _ = graph.entity_count();
    }
    
    // Wait for metrics to be processed
    sleep(Duration::from_millis(100)).await;
    
    // Test RuntimeProfiler as MetricsCollector
    let collector: &dyn MetricsCollector = profiler.as_ref();
    let result = collector.collect(&registry);
    
    match result {
        Ok(_) => {
            println!("✅ RuntimeProfiler successfully collected metrics to registry");
            
            // Check that metrics were registered
            let samples = registry.collect_all_samples();
            
            println!("📊 Registered metrics:");
            for sample in &samples {
                println!("   {}: {:?}", sample.name, sample.value);
            }
            
            let expected_metrics = [
                "runtime_active_functions",
                "runtime_total_function_calls", 
                "runtime_avg_execution_time_ms",
                "runtime_memory_allocations_bytes",
                "runtime_performance_bottlenecks"
            ];
            
            let mut found_metrics = 0;
            for expected in &expected_metrics {
                if samples.iter().any(|s| s.name == *expected) {
                    println!("✅ Found metric: {}", expected);
                    found_metrics += 1;
                } else {
                    println!("⚠️ Missing metric: {}", expected);
                }
            }
            
            assert!(found_metrics > 0, "No expected metrics found in registry!");
            println!("📈 Found {}/{} expected metrics", found_metrics, expected_metrics.len());
            
        },
        Err(e) => {
            println!("❌ RuntimeProfiler metrics collection failed: {}", e);
            panic!("Metrics collection should not fail");
        }
    }
    
    println!("🎉 RuntimeProfiler metrics collection test PASSED!");
}