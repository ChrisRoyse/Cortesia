use super::collectors::{MetricsCollector, MetricsCollectionConfig};
use super::metrics::MetricRegistry;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

pub struct BrainMetricsCollector {
    brain_graph: Arc<RwLock<BrainEnhancedKnowledgeGraph>>,
}

impl BrainMetricsCollector {
    pub fn new(brain_graph: Arc<RwLock<BrainEnhancedKnowledgeGraph>>) -> Self {
        Self { brain_graph }
    }
}

impl MetricsCollector for BrainMetricsCollector {
    fn name(&self) -> &str {
        "brain_metrics"
    }

    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        // Use tokio to run async code in sync context
        let runtime = tokio::runtime::Handle::current();
        let brain_graph_clone = self.brain_graph.clone();
        
        runtime.block_on(async {
            let brain_graph = brain_graph_clone.read().await;
            
            // Collect entity and relationship counts
            let entity_gauge = registry.gauge("brain_entity_count", HashMap::new());
            entity_gauge.set(brain_graph.entity_count() as f64);
            
            let relationship_gauge = registry.gauge("brain_relationship_count", HashMap::new());
            relationship_gauge.set(brain_graph.relationship_count() as f64);
            
            // Collect brain statistics
            if let Ok(stats) = brain_graph.get_brain_statistics().await {
                let avg_activation_gauge = registry.gauge("brain_avg_activation", HashMap::new());
                avg_activation_gauge.set(stats.avg_activation as f64);
                
                let max_activation_gauge = registry.gauge("brain_max_activation", HashMap::new());
                max_activation_gauge.set(stats.max_activation as f64);
                
                let graph_density_gauge = registry.gauge("brain_graph_density", HashMap::new());
                graph_density_gauge.set(stats.graph_density as f64);
                
                let clustering_gauge = registry.gauge("brain_clustering_coefficient", HashMap::new());
                clustering_gauge.set(stats.clustering_coefficient as f64);
                
                let coherence_gauge = registry.gauge("brain_concept_coherence", HashMap::new());
                coherence_gauge.set(stats.concept_coherence as f64);
                
                let learning_gauge = registry.gauge("brain_learning_efficiency", HashMap::new());
                learning_gauge.set(stats.learning_efficiency as f64);
            }
            
            // Collect activation levels
            let activations = brain_graph.get_all_activations().await;
            if !activations.is_empty() {
                let total_activation: f32 = activations.values().sum();
                
                let total_activation_gauge = registry.gauge("brain_total_activation", HashMap::new());
                total_activation_gauge.set(total_activation as f64);
                
                let active_entities_gauge = registry.gauge("brain_active_entities", HashMap::new());
                active_entities_gauge.set(activations.len() as f64);
                
                // Record activation distribution
                let mut very_low = 0;
                let mut low = 0;
                let mut medium = 0;
                let mut high = 0;
                let mut very_high = 0;
                
                for &activation in activations.values() {
                    match activation {
                        a if a < 0.2 => very_low += 1,
                        a if a < 0.4 => low += 1,
                        a if a < 0.6 => medium += 1,
                        a if a < 0.8 => high += 1,
                        _ => very_high += 1,
                    }
                }
                
                let very_low_gauge = registry.gauge("brain_activation_very_low", HashMap::new());
                very_low_gauge.set(very_low as f64);
                
                let low_gauge = registry.gauge("brain_activation_low", HashMap::new());
                low_gauge.set(low as f64);
                
                let medium_gauge = registry.gauge("brain_activation_medium", HashMap::new());
                medium_gauge.set(medium as f64);
                
                let high_gauge = registry.gauge("brain_activation_high", HashMap::new());
                high_gauge.set(high as f64);
                
                let very_high_gauge = registry.gauge("brain_activation_very_high", HashMap::new());
                very_high_gauge.set(very_high as f64);
            }
            
            Ok(())
        })
    }
    
    fn is_enabled(&self, config: &MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"brain".to_string()) ||
        config.enabled_collectors.contains(&self.name().to_string())
    }
}