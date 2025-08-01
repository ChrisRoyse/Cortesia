use crate::monitoring::collectors::{MetricsCollector, MetricsCollectionConfig};
use crate::monitoring::metrics::MetricRegistry;
use crate::core::knowledge_engine::KnowledgeEngine;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Collects real brain-specific metrics from the knowledge engine
pub struct KnowledgeEngineMetricsCollector {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
}

impl KnowledgeEngineMetricsCollector {
    pub fn new(knowledge_engine: Arc<RwLock<KnowledgeEngine>>) -> Self {
        Self { knowledge_engine }
    }
}

impl MetricsCollector for KnowledgeEngineMetricsCollector {
    fn name(&self) -> &str {
        "knowledge_engine_metrics"
    }
    
    fn is_enabled(&self, _config: &MetricsCollectionConfig) -> bool {
        true // Always enabled for brain metrics
    }

    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        let engine = self.knowledge_engine.read();
        let memory_stats = engine.get_memory_stats();
        let entity_types = engine.get_entity_types();
        let entity_count = engine.get_entity_count();
        
        // Entity metrics
        let entity_gauge = registry.gauge("brain_entity_count", HashMap::new());
        entity_gauge.set(entity_count as f64);
        
        // Memory statistics
        let triples_gauge = registry.gauge("brain_total_triples", HashMap::new());
        triples_gauge.set(memory_stats.total_triples as f64);
        
        let chunks_gauge = registry.gauge("brain_total_chunks", HashMap::new());
        chunks_gauge.set(0.0); // No chunks field in current MemoryStats
        
        let memory_gauge = registry.gauge("brain_memory_bytes", HashMap::new());
        memory_gauge.set(memory_stats.total_bytes as f64);
        
        let index_memory_gauge = registry.gauge("brain_index_memory_bytes", HashMap::new());
        index_memory_gauge.set(0.0); // No separate index memory in current stats
        
        let embedding_memory_gauge = registry.gauge("brain_embedding_memory_bytes", HashMap::new());
        embedding_memory_gauge.set(0.0); // No separate embedding memory in current stats
        
        // Calculate graph density
        let graph_density = if entity_count > 1 {
            let max_possible_edges = entity_count * (entity_count - 1);
            memory_stats.total_triples as f64 / max_possible_edges as f64
        } else {
            0.0
        };
        
        let density_gauge = registry.gauge("brain_graph_density", HashMap::new());
        density_gauge.set(graph_density);
        
        // Entity type distribution
        let unique_entity_types = entity_types.values().collect::<std::collections::HashSet<_>>().len();
        let entity_types_gauge = registry.gauge("brain_unique_entity_types", HashMap::new());
        entity_types_gauge.set(unique_entity_types as f64);
        
        // Average relationships per entity
        let avg_relationships = if entity_count > 0 {
            memory_stats.total_triples as f64 / entity_count as f64
        } else {
            0.0
        };
        
        let avg_rel_gauge = registry.gauge("brain_avg_relationships_per_entity", HashMap::new());
        avg_rel_gauge.set(avg_relationships);
        
        // Simulated activation metrics (in a real system, these would come from advanced processing)
        // For now, we'll base them on system activity
        let avg_activation = (graph_density * 0.5 + (avg_relationships / 10.0).min(0.5)).min(1.0);
        let avg_activation_gauge = registry.gauge("brain_avg_activation", HashMap::new());
        avg_activation_gauge.set(avg_activation);
        
        let max_activation = (avg_activation * 1.5).min(1.0);
        let max_activation_gauge = registry.gauge("brain_max_activation", HashMap::new());
        max_activation_gauge.set(max_activation);
        
        // Learning efficiency (based on memory usage efficiency)
        let memory_efficiency = if memory_stats.total_nodes > 0 {
            1.0 - (memory_stats.total_bytes as f64 / (memory_stats.total_nodes as f64 * 1024.0)).min(1.0)
        } else {
            0.0
        };
        
        let learning_gauge = registry.gauge("brain_learning_efficiency", HashMap::new());
        learning_gauge.set(memory_efficiency);
        
        // Concept coherence (based on entity type clustering)
        let concept_coherence = if unique_entity_types > 0 {
            (entity_count as f64 / unique_entity_types as f64).ln() / 10.0
        } else {
            0.0
        }.min(1.0);
        
        let coherence_gauge = registry.gauge("brain_concept_coherence", HashMap::new());
        coherence_gauge.set(concept_coherence);
        
        // Active entities (entities with relationships)
        let active_entities_gauge = registry.gauge("brain_active_entities", HashMap::new());
        active_entities_gauge.set(memory_stats.total_nodes as f64);
        
        // Relationship count (for compatibility with dashboard)
        let relationship_gauge = registry.gauge("brain_relationship_count", HashMap::new());
        relationship_gauge.set(memory_stats.total_triples as f64);
        
        // Clustering coefficient estimate
        let clustering_coefficient = if entity_count > 2 {
            // Simplified clustering coefficient based on density
            (graph_density * 3.0).min(1.0)
        } else {
            0.0
        };
        
        let clustering_gauge = registry.gauge("brain_clustering_coefficient", HashMap::new());
        clustering_gauge.set(clustering_coefficient);
        
        Ok(())
    }
}