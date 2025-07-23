use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::monitoring::metrics::MetricRegistry;
use llmkg::monitoring::collectors::MetricsCollector;
use llmkg::monitoring::BrainMetricsCollector;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Brain Metrics Collection");
    println!("===================================\n");
    
    // Create brain graph
    let brain_graph = Arc::new(RwLock::new(
        BrainEnhancedKnowledgeGraph::new(384)?
    ));
    
    println!("1. Creating test entities and relationships...");
    
    // Populate with test data
    {
        let graph = brain_graph.write().await;
        let mut entity_keys = Vec::new();
        
        // Add 20 entities
        for i in 0..20 {
            let entity_data = EntityData {
                type_id: 1,
                properties: format!(r#"{{"name": "Entity {}", "type": "test"}}"#, i),
                embedding: vec![0.05 * i as f32; 384],
            };
            
            if let Ok(key) = graph.core_graph.add_entity(entity_data) {
                entity_keys.push(key);
                // Set activation levels
                let activation = (i as f32 / 20.0) * 0.8 + 0.1;
                graph.set_entity_activation(key, activation).await;
                println!("   Added entity {} with activation {:.2}", i, activation);
            }
        }
        
        // Add 30 relationships
        for i in 0..30 {
            let source_idx = i % entity_keys.len();
            let target_idx = (i + 5) % entity_keys.len();
            
            if source_idx != target_idx {
                let source = entity_keys[source_idx];
                let target = entity_keys[target_idx];
                let weight = 0.3 + (i as f32 / 30.0) * 0.6;
                
                if graph.core_graph.add_relationship(source, target, weight).is_ok() {
                    graph.set_synaptic_weight(source, target, weight).await;
                    println!("   Added relationship {} -> {} (weight: {:.2})", source_idx, target_idx, weight);
                }
            }
        }
        
        println!("\n   Total entities: {}", graph.entity_count());
        println!("   Total relationships: {}", graph.relationship_count());
    }
    
    println!("\n2. Creating metrics collector and registry...");
    let registry = Arc::new(MetricRegistry::new());
    let collector = BrainMetricsCollector::new(brain_graph.clone());
    
    println!("\n3. Collecting brain metrics...");
    collector.collect(&registry)?;
    
    println!("\n4. Reading collected metrics:");
    println!("   ================================");
    
    let samples = registry.collect_all_samples();
    let mut brain_metrics = Vec::new();
    
    for sample in samples {
        if sample.name.starts_with("brain_") {
            brain_metrics.push(sample);
        }
    }
    
    // Sort and display metrics
    brain_metrics.sort_by(|a, b| a.name.cmp(&b.name));
    
    for metric in &brain_metrics {
        let value_str = match &metric.value {
            llmkg::monitoring::metrics::MetricValue::Counter(v) => format!("{}", v),
            llmkg::monitoring::metrics::MetricValue::Gauge(v) => format!("{:.3}", v),
            llmkg::monitoring::metrics::MetricValue::Timer { count, sum_duration_ms, .. } => format!("count: {}, avg: {:.3}ms", count, sum_duration_ms / (*count as f64).max(1.0)),
            llmkg::monitoring::metrics::MetricValue::Histogram { count, sum, .. } => format!("count: {}, sum: {:.3}", count, sum),
            llmkg::monitoring::metrics::MetricValue::Summary { count, sum, .. } => format!("count: {}, sum: {:.3}", count, sum),
        };
        
        println!("   {:<35} = {}", metric.name, value_str);
    }
    
    if brain_metrics.is_empty() {
        println!("   ❌ No brain metrics found!");
    } else {
        println!("\n   ✅ Found {} brain metrics", brain_metrics.len());
    }
    
    // Test dynamic updates
    println!("\n5. Testing dynamic updates...");
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    
    {
        let graph = brain_graph.write().await;
        
        // Update some activations
        let activations = graph.get_all_activations().await;
        let keys: Vec<_> = activations.keys().cloned().collect();
        
        for (i, key) in keys.iter().take(5).enumerate() {
            let new_activation = 0.9 - (i as f32 * 0.1);
            graph.set_entity_activation(*key, new_activation).await;
            println!("   Updated entity activation to {:.2}", new_activation);
        }
    }
    
    // Collect again
    collector.collect(&registry)?;
    
    println!("\n6. Verifying metrics changed:");
    println!("   ================================");
    
    let new_samples = registry.collect_all_samples();
    for sample in new_samples {
        if sample.name == "brain_avg_activation" || sample.name == "brain_max_activation" {
            let value_str = match &sample.value {
                llmkg::monitoring::metrics::MetricValue::Gauge(v) => format!("{:.3}", v),
                _ => "N/A".to_string(),
            };
            println!("   {:<35} = {}", sample.name, value_str);
        }
    }
    
    println!("\n✅ Brain metrics collection test completed!");
    
    Ok(())
}