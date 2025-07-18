/*!
Phase 5.4: Performance Dashboard Demo
Demonstrates the real-time performance monitoring dashboard with LLMKG integration
*/

use llmkg::monitoring::{
    MetricRegistry, DashboardServer, DashboardConfig, 
    SystemMetricsCollector, ApplicationMetricsCollector, CustomMetricsCollector,
    MetricsCollectionConfig, SystemMetricsConfig, ApplicationMetricsConfig,
    PrometheusExporter, JsonExporter, MultiExporter, ExportConfig,
    PrometheusConfig, JsonExportConfig
};
use llmkg::{EntityGraph, ProductQuantizer, StringInterner, PersistentMMapStorage, Entity, EntityKey};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::thread;
use tokio::time::{sleep, interval};
use tempfile::TempDir;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LLMKG Performance Dashboard Demo");
    println!("====================================");
    
    // Create metrics registry
    let registry = Arc::new(MetricRegistry::new());
    
    // Setup dashboard configuration
    let dashboard_config = DashboardConfig {
        http_port: 8080,
        websocket_port: 8081,
        update_interval: Duration::from_secs(2),
        history_size: 100,
        title: "LLMKG Performance Dashboard - Demo".to_string(),
        refresh_rate_ms: 1000,
    };
    
    // Create metrics collectors
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
    
    let collectors: Vec<Box<dyn llmkg::monitoring::MetricsCollector>> = vec![
        Box::new(SystemMetricsCollector::new(system_config)),
        Box::new(ApplicationMetricsCollector::new(app_config)),
        Box::new(create_llmkg_metrics_collector(registry.clone())),
    ];
    
    // Setup metrics exporters
    setup_metrics_exporters(registry.clone()).await;
    
    // Create and start dashboard server
    let dashboard_server = DashboardServer::new(
        dashboard_config,
        registry.clone(),
        collectors,
    );
    
    // Start the dashboard in the background
    tokio::spawn(async move {
        if let Err(e) = dashboard_server.start().await {
            eprintln!("Dashboard server error: {}", e);
        }
    });
    
    // Wait a moment for the server to start
    sleep(Duration::from_secs(2)).await;
    
    println!("\n📊 Dashboard started successfully!");
    println!("   🌐 Web Dashboard: http://localhost:8080");
    println!("   🔌 WebSocket API: ws://localhost:8081");
    println!("   📈 Metrics API:   http://localhost:8080/api/metrics");
    println!("   📋 History API:   http://localhost:8080/api/history");
    
    // Start LLMKG workload simulation
    start_llmkg_workload_simulation(registry.clone()).await;
    
    println!("\n🔄 Simulating LLMKG workload...");
    println!("   - Adding entities to knowledge graph");
    println!("   - Performing similarity searches");
    println!("   - Monitoring performance metrics");
    println!("\n💡 Open http://localhost:8080 in your browser to view the dashboard");
    println!("   Press Ctrl+C to stop the demo");
    
    // Keep the demo running
    let mut interval = interval(Duration::from_secs(10));
    loop {
        interval.tick().await;
        print_metrics_summary(registry.clone()).await;
    }
}

async fn setup_metrics_exporters(registry: Arc<MetricRegistry>) {
    // Setup Prometheus exporter (commented out for demo)
    /*
    let prometheus_config = PrometheusConfig {
        push_gateway_url: "http://localhost:9091".to_string(),
        job_name: "llmkg_demo".to_string(),
        instance: "demo_instance".to_string(),
        basic_auth: None,
        extra_labels: {
            let mut labels = HashMap::new();
            labels.insert("service".to_string(), "llmkg".to_string());
            labels.insert("environment".to_string(), "demo".to_string());
            labels
        },
    };
    */
    
    // Setup JSON exporter
    let json_config = JsonExportConfig {
        output_file: "demo_metrics.json".to_string(),
        pretty_print: true,
        append_mode: true,
        max_file_size_mb: 10,
        rotation_count: 3,
    };
    
    let export_config = ExportConfig {
        enabled: true,
        export_interval: Duration::from_secs(30),
        batch_size: 100,
        timeout: Duration::from_secs(5),
        retry_attempts: 2,
        retry_delay: Duration::from_secs(1),
    };
    
    let exporters: Vec<Box<dyn llmkg::monitoring::MetricsExporter>> = vec![
        Box::new(JsonExporter::new(json_config, export_config.clone())),
        // Box::new(PrometheusExporter::new(prometheus_config, export_config.clone())),
    ];
    
    let multi_exporter = MultiExporter::new(exporters, export_config);
    
    // Start background export
    tokio::spawn(async move {
        multi_exporter.start_background_export(registry).await;
    });
    
    println!("📤 Metrics exporters configured:");
    println!("   - JSON file export: demo_metrics.json");
    // println!("   - Prometheus push gateway: http://localhost:9091");
}

fn create_llmkg_metrics_collector(registry: Arc<MetricRegistry>) -> CustomMetricsCollector {
    CustomMetricsCollector::new(
        "llmkg".to_string(),
        move |registry: &MetricRegistry| {
            // Entity count metric
            let entity_count_gauge = registry.gauge("llmkg_entities_total", HashMap::new());
            entity_count_gauge.set(1000.0); // Demo value
            
            // Query performance metrics
            let query_latency_timer = registry.timer("llmkg_query_latency_seconds", HashMap::new());
            query_latency_timer.observe_duration(Duration::from_millis(50));
            
            // Index size metric
            let index_size_gauge = registry.gauge("llmkg_index_size_bytes", HashMap::new());
            index_size_gauge.set(1024.0 * 1024.0 * 50.0); // 50MB demo value
            
            // Cache metrics
            let cache_hit_rate_gauge = registry.gauge("llmkg_cache_hit_rate", HashMap::new());
            cache_hit_rate_gauge.set(0.85); // 85% hit rate
            
            // Operations counter
            let ops_counter = registry.counter("llmkg_operations_total", HashMap::new());
            ops_counter.increment();
            
            Ok(())
        }
    )
}

async fn start_llmkg_workload_simulation(registry: Arc<MetricRegistry>) {
    tokio::spawn(async move {
        let temp_dir = TempDir::new().unwrap();
        let graph = EntityGraph::new();
        let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
        let mut interner = StringInterner::new();
        let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
        
        let mut rng = StdRng::seed_from_u64(42);
        let mut entity_count = 0;
        
        // Simulation metrics
        let entity_counter = registry.counter("demo_entities_added_total", HashMap::new());
        let query_timer = registry.timer("demo_query_duration_ms", HashMap::new());
        let memory_gauge = registry.gauge("demo_memory_usage_mb", HashMap::new());
        let error_counter = registry.counter("demo_errors_total", HashMap::new());
        
        let mut interval = interval(Duration::from_millis(500));
        
        loop {
            interval.tick().await;
            
            // Simulate adding entities
            if entity_count < 10000 {
                let batch_size = rng.gen_range(1..=10);
                for i in 0..batch_size {
                    let entity_id = format!("entity_{}_{}", entity_count, i);
                    let content = format!("Content for entity {} with various metadata", entity_id);
                    
                    // Generate random embedding
                    let embedding: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    
                    let mut metadata = HashMap::new();
                    metadata.insert("category".to_string(), format!("cat_{}", rng.gen_range(0..10)));
                    metadata.insert("priority".to_string(), rng.gen_range(1..=5).to_string());
                    
                    let key = EntityKey::from_hash(&entity_id);
                    let content_id = interner.insert(&content);
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding,
                        metadata,
                    };
                    
                    graph.add_entity(entity);
                    entity_counter.increment();
                    entity_count += 1;
                }
            }
            
            // Simulate query operations
            if entity_count > 10 {
                let query_embedding: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect();
                
                let query_result = query_timer.time(|| {
                    graph.find_similar_entities(&query_embedding, 10)
                });
                
                // Simulate occasional errors
                if rng.gen_bool(0.02) { // 2% error rate
                    error_counter.increment();
                }
                
                // Update memory usage (simulated)
                let memory_usage = 50.0 + (entity_count as f64 * 0.01) + rng.gen_range(-5.0..5.0);
                memory_gauge.set(memory_usage);
            }
            
            // Simulate varying load
            if rng.gen_bool(0.1) {
                sleep(Duration::from_millis(rng.gen_range(100..500))).await;
            }
        }
    });
}

async fn print_metrics_summary(registry: Arc<MetricRegistry>) {
    let samples = registry.collect_all_samples();
    let metrics_count = samples.len();
    
    println!("\n📊 Metrics Summary ({})", chrono::Utc::now().format("%H:%M:%S"));
    println!("   📈 Active metrics: {}", metrics_count);
    
    // Find some key metrics to display
    for sample in samples.iter().take(5) {
        match &sample.value {
            llmkg::monitoring::MetricValue::Counter(value) => {
                println!("   🔢 {}: {}", sample.name, value);
            }
            llmkg::monitoring::MetricValue::Gauge(value) => {
                println!("   📏 {}: {:.2}", sample.name, value);
            }
            llmkg::monitoring::MetricValue::Timer { count, sum_duration_ms, percentiles, .. } => {
                if let Some(p95) = percentiles.get("p95") {
                    println!("   ⏱️  {} (P95): {:.2}ms ({} samples)", sample.name, p95, count);
                }
            }
            _ => {}
        }
    }
    
    println!("   🌐 Dashboard: http://localhost:8080");
}

// Helper function to demonstrate dashboard features
#[allow(dead_code)]
async fn demonstrate_dashboard_features() {
    println!("\n🎯 Dashboard Features:");
    println!("   📊 Real-time metrics visualization");
    println!("   📈 Interactive charts with Chart.js");
    println!("   🔌 WebSocket live updates");
    println!("   🚨 Alert monitoring");
    println!("   📱 Responsive design");
    println!("   🎨 Modern UI with gradients and animations");
    println!("   📋 Metrics history and trends");
    println!("   🔍 System resource monitoring");
    println!("   ⚡ Low-latency performance tracking");
    println!("   📤 Multiple export formats (JSON, Prometheus)");
}