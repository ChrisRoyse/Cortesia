use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::monitoring::metrics::MetricRegistry;
use llmkg::monitoring::collectors::MetricsCollector;
use llmkg::monitoring::BrainMetricsCollector;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_brain_metrics_collector_produces_real_data() {
    // Create a brain graph with test data
    let brain_graph = Arc::new(RwLock::new(
        BrainEnhancedKnowledgeGraph::new(384).expect("Failed to create brain graph")
    ));
    
    // Add some entities and relationships
    {
        let graph = brain_graph.write().await;
        
        // Add 10 entities
        let mut entity_keys = Vec::new();
        for i in 0..10 {
            let entity_data = EntityData {
                type_id: 1,
                properties: format!(r#"{{"name": "Test Entity {}", "value": {}}}"#, i, i * 10),
                embedding: vec![0.1 * i as f32; 384],
            };
            
            if let Ok(key) = graph.core_graph.add_entity(entity_data) {
                entity_keys.push(key);
                // Set varying activation levels
                graph.set_entity_activation(key, 0.1 * i as f32).await;
            }
        }
        
        // Add some relationships
        for i in 0..5 {
            if let (Some(&source), Some(&target)) = (entity_keys.get(i), entity_keys.get(i + 1)) {
                let weight = 0.5 + 0.1 * i as f32;
                let _ = graph.core_graph.add_relationship(source, target, weight);
                graph.set_synaptic_weight(source, target, weight).await;
            }
        }
    }
    
    // Create metrics collector and registry
    let registry = Arc::new(MetricRegistry::new());
    let collector = BrainMetricsCollector::new(brain_graph.clone());
    
    // Collect metrics
    collector.collect(&registry).expect("Failed to collect metrics");
    
    // Verify metrics were collected
    let samples = registry.collect_all_samples();
    
    println!("\n🧠 Brain Metrics Collected:");
    println!("==========================");
    
    let mut brain_metrics_found = false;
    let mut expected_metrics = vec![
        "brain_entity_count",
        "brain_relationship_count",
        "brain_avg_activation",
        "brain_max_activation",
        "brain_graph_density",
        "brain_clustering_coefficient",
        "brain_concept_coherence",
        "brain_learning_efficiency",
        "brain_total_activation",
        "brain_active_entities",
    ];
    
    for sample in &samples {
        if sample.name.starts_with("brain_") {
            brain_metrics_found = true;
            println!("  {}: {:?}", sample.name, sample.value);
            
            // Remove found metrics from expected list
            expected_metrics.retain(|&m| m != sample.name);
            
            // Verify the metric has a reasonable value
            match sample.name.as_str() {
                "brain_entity_count" => {
                    if let llmkg::monitoring::metrics::MetricValue::Gauge(value) = sample.value {
                        assert_eq!(value as u32, 10, "Expected 10 entities");
                    }
                }
                "brain_relationship_count" => {
                    if let llmkg::monitoring::metrics::MetricValue::Gauge(value) = sample.value {
                        assert_eq!(value as u32, 5, "Expected 5 relationships");
                    }
                }
                "brain_active_entities" => {
                    if let llmkg::monitoring::metrics::MetricValue::Gauge(value) = sample.value {
                        assert!(value > 0.0, "Expected at least some active entities");
                    }
                }
                _ => {}
            }
        }
    }
    
    assert!(brain_metrics_found, "No brain metrics were found in the registry");
    assert!(expected_metrics.is_empty(), "Missing expected metrics: {:?}", expected_metrics);
    
    println!("\n✅ All brain metrics are being collected correctly!");
    
    // Test updating metrics
    {
        let graph = brain_graph.write().await;
        
        // Update some activations
        let activations = graph.get_all_activations().await;
        for (entity, _) in activations.iter().take(3) {
            graph.set_entity_activation(*entity, 0.9).await;
        }
    }
    
    // Collect metrics again
    collector.collect(&registry).expect("Failed to collect metrics second time");
    
    // Verify updated metrics
    let updated_samples = registry.collect_all_samples();
    
    println!("\n🔄 Updated Brain Metrics:");
    println!("========================");
    
    for sample in &updated_samples {
        if sample.name == "brain_avg_activation" || sample.name == "brain_max_activation" {
            println!("  {}: {:?}", sample.name, sample.value);
            
            if let llmkg::monitoring::metrics::MetricValue::Gauge(value) = sample.value {
                assert!(value > 0.0, "Expected positive activation value");
            }
        }
    }
}

#[tokio::test]
async fn test_brain_metrics_with_empty_graph() {
    // Create an empty brain graph
    let brain_graph = Arc::new(RwLock::new(
        BrainEnhancedKnowledgeGraph::new(384).expect("Failed to create brain graph")
    ));
    
    let registry = Arc::new(MetricRegistry::new());
    let collector = BrainMetricsCollector::new(brain_graph);
    
    // Should not panic with empty graph
    collector.collect(&registry).expect("Failed to collect metrics from empty graph");
    
    let samples = registry.collect_all_samples();
    
    // Verify basic metrics exist even with empty graph
    let entity_count = samples.iter()
        .find(|s| s.name == "brain_entity_count")
        .expect("brain_entity_count metric should exist");
    
    if let llmkg::monitoring::metrics::MetricValue::Gauge(value) = entity_count.value {
        assert_eq!(value as u32, 0, "Expected 0 entities in empty graph");
    }
}