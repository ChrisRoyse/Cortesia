use llmkg::monitoring::dashboard::{DashboardConfig, DashboardServer};
use llmkg::monitoring::metrics::MetricRegistry;
use llmkg::monitoring::collectors::{
    SystemMetricsCollector, ApplicationMetricsCollector, SystemMetricsConfig, ApplicationMetricsConfig,
    CodebaseAnalyzer, RuntimeProfiler, ApiEndpointMonitor, TestExecutionTracker
};
use llmkg::monitoring::BrainMetricsCollector;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing LLMKG Brain-Enhanced Dashboard Server...");
    
    // Initialize the brain-enhanced knowledge graph
    let brain_graph = Arc::new(RwLock::new(
        BrainEnhancedKnowledgeGraph::new(384)?
    ));
    
    // Populate with some initial data for demonstration
    let entity_keys = {
        let graph = brain_graph.write().await;
        let mut keys = Vec::new();
        let mut rng = rand::rng();
        
        // Add some entities
        for i in 0..20 {
            let entity_data = llmkg::core::types::EntityData {
                type_id: 1,
                properties: format!("{{\"name\": \"Entity {i}\", \"index\": {i}}}"),
                embedding: vec![rng.random::<f32>(); 384],
            };
            
            match graph.core_graph.add_entity(entity_data) {
                Ok(entity_key) => {
                    keys.push(entity_key);
                    // Set random activation levels
                    graph.set_entity_activation(entity_key, rng.random::<f32>()).await;
                },
                Err(e) => eprintln!("Warning: Failed to add entity {i}: {e}"),
            }
        }
        
        // Add some relationships
        for i in 0..15 {
            if keys.len() > 1 {
                let source_idx = i % keys.len();
                let target_idx = (i + 1) % keys.len();
                let source = keys[source_idx];
                let target = keys[target_idx];
                
                if let Err(e) = graph.core_graph.add_relationship(source, target, rng.random::<f32>()) {
                    eprintln!("Warning: Failed to add relationship: {e}");
                }
                
                // Set synaptic weights
                graph.set_synaptic_weight(source, target, rng.random::<f32>()).await;
            }
        }
        
        keys
    };
    
    // Initialize the metric registry
    let registry = Arc::new(MetricRegistry::new());
    
    // Create collector configs
    let system_config = SystemMetricsConfig {
        collect_cpu: true,
        collect_memory: true,
        collect_disk: true,
        collect_network: true,
        collect_load: true,
    };
    
    let app_config = ApplicationMetricsConfig {
        collect_performance: true,
        collect_operations: true,
        collect_errors: true,
        collect_resources: true,
    };
    
    // Initialize enhanced collectors
    let current_dir = std::env::current_dir().unwrap();
    let codebase_analyzer = CodebaseAnalyzer::new(current_dir.clone());
    let runtime_profiler = RuntimeProfiler::new();
    let api_monitor = ApiEndpointMonitor::new();
    let test_tracker = TestExecutionTracker::new(current_dir);
    
    // Analyze codebase and discover endpoints
    println!("Analyzing codebase...");
    if let Err(e) = codebase_analyzer.analyze_codebase().await {
        eprintln!("Warning: Failed to analyze codebase: {e}");
    }
    
    println!("Discovering API endpoints...");
    if let Err(e) = api_monitor.discover_endpoints() {
        eprintln!("Warning: Failed to discover API endpoints: {e}");
    }
    
    println!("Discovering test suites...");
    if let Err(e) = test_tracker.discover_test_suites().await {
        eprintln!("Warning: Failed to discover test suites: {e}");
    }

    // Create collectors including brain metrics and enhanced collectors
    let collectors: Vec<Box<dyn llmkg::monitoring::collectors::MetricsCollector>> = vec![
        Box::new(SystemMetricsCollector::new(system_config)),
        Box::new(ApplicationMetricsCollector::new(app_config)),
        Box::new(BrainMetricsCollector::new(brain_graph.clone())),
        Box::new(codebase_analyzer),
        Box::new(runtime_profiler),
        Box::new(api_monitor),
        Box::new(test_tracker),
    ];
    
    // Configure the dashboard
    let config = DashboardConfig {
        http_port: 8082,
        websocket_port: 8083,
        title: "LLMKG Brain-Enhanced Performance Dashboard".to_string(),
        refresh_rate_ms: 1000,
        ..Default::default()
    };
    
    println!("Starting LLMKG Brain-Enhanced Dashboard Server...");
    println!("HTTP Dashboard: http://localhost:{}", config.http_port);
    println!("WebSocket: ws://localhost:{}", config.websocket_port);
    println!("React Dashboard: http://localhost:3001");
    
    // Create and start the server
    let server = DashboardServer::new(config, registry, collectors);
    server.start().await?;
    
    // Simulate ongoing LLMKG operations while server is running
    let brain_graph_for_simulation = brain_graph.clone();
    let entity_keys_clone = entity_keys.clone();
    let mut simulation_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        let mut operation_count = 0;
        let mut rng = rand::rngs::StdRng::from_rng(&mut rand::rng());
        
        loop {
            interval.tick().await;
            operation_count += 1;
            
            // Simulate real LLMKG operations
            {
                let graph = brain_graph_for_simulation.write().await;
                
                // Add new entities periodically (simulating incoming data)
                if operation_count % 3 == 0 {
                    let entity_data = llmkg::core::types::EntityData {
                        type_id: 2,
                        properties: format!("{{\"name\": \"Dynamic Entity {}\", \"timestamp\": {}}}", 
                            operation_count, chrono::Utc::now().timestamp()),
                        embedding: vec![rng.random::<f32>(); 384],
                    };
                    
                    if let Ok(entity_key) = graph.core_graph.add_entity(entity_data) {
                        // Set initial activation
                        graph.set_entity_activation(entity_key, rng.random::<f32>()).await;
                        println!("🧠 Added new entity: {entity_key:?}");
                    }
                }
                
                // Update existing entity activations (simulating brain activity)
                let activations = graph.get_all_activations().await;
                let mut keys: Vec<_> = activations.keys().cloned().collect();
                keys.extend(entity_keys_clone.iter().cloned());
                
                if !keys.is_empty() {
                    for _ in 0..3 {  // Update 3 random entities
                        if let Some(&key) = keys.choose(&mut rng) {
                            let new_activation = rng.random::<f32>();
                            graph.set_entity_activation(key, new_activation).await;
                        }
                    }
                    println!("🔄 Updated {} entity activations", keys.len().min(3));
                }
                
                // Occasionally add new relationships (simulating learning)
                if operation_count % 5 == 0 && keys.len() >= 2 {
                    let source = keys.choose(&mut rng).unwrap();
                    let target = keys.choose(&mut rng).unwrap();
                    
                    if source != target {
                        let weight = rng.random::<f32>();
                        let _ = graph.core_graph.add_relationship(*source, *target, weight);
                        graph.set_synaptic_weight(*source, *target, weight).await;
                        println!("🔗 Added relationship: {source:?} -> {target:?} (weight: {weight:.3})");
                    }
                }
            }
            
            println!("📊 Operation cycle {} completed - {} total entities", 
                operation_count, entity_keys_clone.len() + operation_count / 3);
        }
    });
    
    // Keep the server running
    println!("\nServer is running with live LLMKG operations simulation.");
    println!("Watch the dashboard for real-time updates of actual LLMKG activity!");
    println!("Press Ctrl+C to stop.");
    
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("Shutting down server...");
            simulation_handle.abort();
            server.stop();
        }
        _ = &mut simulation_handle => {
            println!("Simulation ended unexpectedly");
        }
    }
    
    Ok(())
}