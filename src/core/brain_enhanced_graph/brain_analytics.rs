//! Analytics and statistics for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::collections::{HashMap, HashSet};

impl BrainEnhancedKnowledgeGraph {
    /// Get comprehensive brain statistics
    pub async fn get_brain_statistics(&self) -> Result<BrainStatistics> {
        let entity_count = self.entity_count();
        let relationship_count = self.relationship_count();
        
        if entity_count == 0 {
            return Ok(BrainStatistics::new());
        }
        
        // Calculate activation statistics
        let activations = self.entity_activations.read().await;
        let activation_values: Vec<f32> = activations.values().cloned().collect();
        
        let (avg_activation, max_activation, min_activation) = if activation_values.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let total: f32 = activation_values.iter().sum();
            let avg = total / activation_values.len() as f32;
            let max = activation_values.iter().cloned().fold(0.0, f32::max);
            let min = activation_values.iter().cloned().fold(1.0, f32::min);
            (avg, max, min)
        };
        
        // Calculate graph density
        let max_possible_relationships = entity_count * (entity_count - 1);
        let graph_density = if max_possible_relationships > 0 {
            relationship_count as f32 / max_possible_relationships as f32
        } else {
            0.0
        };
        
        // Calculate clustering coefficient
        let clustering_coefficient = self.calculate_average_clustering_coefficient().await;
        
        // Calculate average path length
        let average_path_length = self.calculate_average_path_length().await;
        
        // Calculate betweenness centrality
        let betweenness_centrality = self.calculate_betweenness_centrality().await;
        
        // Calculate activation distribution
        let activation_distribution = self.calculate_activation_distribution(&activation_values);
        
        // Calculate concept coherence
        let concept_coherence = self.calculate_average_concept_coherence().await;
        
        // Calculate learning efficiency
        let learning_stats = self.learning_stats.read().await;
        let learning_efficiency = learning_stats.learning_efficiency;
        
        Ok(BrainStatistics {
            entity_count,
            relationship_count,
            avg_activation,
            max_activation,
            min_activation,
            graph_density,
            clustering_coefficient,
            average_path_length,
            betweenness_centrality,
            activation_distribution,
            concept_coherence,
            learning_efficiency,
        })
    }

    /// Calculate average clustering coefficient
    async fn calculate_average_clustering_coefficient(&self) -> f32 {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut total_clustering = 0.0;
        let mut entity_count = 0;
        
        for entity_key in entity_keys {
            let clustering = self.calculate_clustering_coefficient(entity_key).await;
            total_clustering += clustering;
            entity_count += 1;
        }
        
        if entity_count > 0 {
            total_clustering / entity_count as f32
        } else {
            0.0
        }
    }

    /// Calculate average path length
    async fn calculate_average_path_length(&self) -> f32 {
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        if entity_keys.len() < 2 {
            return 0.0;
        }
        
        let mut total_path_length = 0.0;
        let mut path_count = 0;
        
        // Sample a subset of entity pairs to avoid O(n²) complexity
        let sample_size = (entity_keys.len() * 100).min(10000);
        
        for i in 0..sample_size {
            let source_idx = i % entity_keys.len();
            let target_idx = (i + 1) % entity_keys.len();
            
            if source_idx != target_idx {
                if let Some(path) = self.core_graph.find_path(entity_keys[source_idx], entity_keys[target_idx]) {
                    total_path_length += (path.len() - 1) as f32;
                    path_count += 1;
                }
            }
        }
        
        if path_count > 0 {
            total_path_length / path_count as f32
        } else {
            0.0
        }
    }

    /// Calculate betweenness centrality for all entities
    async fn calculate_betweenness_centrality(&self) -> HashMap<EntityKey, f32> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut centrality = HashMap::new();
        
        if entity_keys.len() < 3 {
            return centrality;
        }
        
        // Initialize centrality scores
        for &entity_key in &entity_keys {
            centrality.insert(entity_key, 0.0);
        }
        
        // Calculate betweenness centrality (simplified version)
        let sample_pairs = 1000; // Limit for performance
        let mut pair_count = 0;
        
        for i in 0..entity_keys.len() {
            for j in i + 1..entity_keys.len() {
                if pair_count >= sample_pairs {
                    break;
                }
                
                let source = entity_keys[i];
                let target = entity_keys[j];
                
                // Find shortest path
                if let Some(path) = self.core_graph.find_path(source, target) {
                    // Each entity in the path (except source and target) gets centrality score
                    for k in 1..path.len() - 1 {
                        let intermediary = path[k];
                        *centrality.get_mut(&intermediary).unwrap() += 1.0;
                    }
                }
                
                pair_count += 1;
            }
            
            if pair_count >= sample_pairs {
                break;
            }
        }
        
        // Normalize centrality scores
        let max_possible_paths = (entity_keys.len() - 1) * (entity_keys.len() - 2) / 2;
        for centrality_score in centrality.values_mut() {
            *centrality_score /= max_possible_paths as f32;
        }
        
        centrality
    }

    /// Calculate activation distribution
    fn calculate_activation_distribution(&self, activations: &[f32]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for &activation in activations {
            let bucket = if activation < 0.2 {
                "very_low"
            } else if activation < 0.4 {
                "low"
            } else if activation < 0.6 {
                "medium"
            } else if activation < 0.8 {
                "high"
            } else {
                "very_high"
            };
            
            *distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Calculate average concept coherence
    async fn calculate_average_concept_coherence(&self) -> f32 {
        let concepts = self.concept_structures.read().await;
        
        if concepts.is_empty() {
            return 0.0;
        }
        
        let total_coherence: f32 = concepts.values().map(|c| c.coherence_score).sum();
        total_coherence / concepts.len() as f32
    }

    /// Get critical entities (highest centrality)
    pub async fn get_critical_entities(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let centrality = self.calculate_betweenness_centrality().await;
        
        let mut critical_entities: Vec<(EntityKey, f32)> = centrality
            .into_iter()
            .collect();
        
        critical_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        critical_entities.truncate(k);
        
        critical_entities
    }

    /// Get most activated entities
    pub async fn get_most_activated_entities(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let activations = self.entity_activations.read().await;
        
        let mut activated_entities: Vec<(EntityKey, f32)> = activations
            .iter()
            .map(|(&key, &activation)| (key, activation))
            .collect();
        
        activated_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        activated_entities.truncate(k);
        
        activated_entities
    }

    /// Get least activated entities
    pub async fn get_least_activated_entities(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let activations = self.entity_activations.read().await;
        
        let mut activated_entities: Vec<(EntityKey, f32)> = activations
            .iter()
            .map(|(&key, &activation)| (key, activation))
            .collect();
        
        activated_entities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        activated_entities.truncate(k);
        
        activated_entities
    }

    /// Get connection usage frequency
    pub async fn get_connection_usage_frequency(&self) -> HashMap<(EntityKey, EntityKey), u32> {
        let mut usage_frequency = HashMap::new();
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        // This is a simplified implementation
        // In a real scenario, you'd track actual usage over time
        for entity_key in entity_keys {
            let neighbors = self.get_neighbors(entity_key).await;
            for (neighbor_key, weight) in neighbors {
                // Simulate usage frequency based on synaptic weight
                let usage = (weight * 100.0) as u32;
                usage_frequency.insert((entity_key, neighbor_key), usage);
            }
        }
        
        usage_frequency
    }

    /// Get concept usage statistics
    pub async fn get_concept_usage_statistics(&self) -> HashMap<String, ConceptUsageStats> {
        let concepts = self.concept_structures.read().await;
        let mut usage_stats = HashMap::new();
        
        for (concept_name, concept_structure) in concepts.iter() {
            let total_entities = concept_structure.total_entities();
            let avg_activation = concept_structure.concept_activation;
            let coherence = concept_structure.coherence_score;
            
            // Calculate complexity score
            let complexity = (total_entities as f32 * coherence).sqrt();
            
            let stats = ConceptUsageStats {
                total_entities,
                avg_activation,
                coherence,
                complexity,
                last_accessed: std::time::Instant::now(), // Placeholder
            };
            
            usage_stats.insert(concept_name.clone(), stats);
        }
        
        usage_stats
    }

    /// Analyze graph patterns
    pub async fn analyze_graph_patterns(&self) -> GraphPatternAnalysis {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut pattern_analysis = GraphPatternAnalysis::new();
        
        // Analyze degree distribution
        let mut degree_distribution = HashMap::new();
        for entity_key in &entity_keys {
            let degree = self.get_neighbors(*entity_key).await.len();
            *degree_distribution.entry(degree).or_insert(0) += 1;
        }
        pattern_analysis.degree_distribution = degree_distribution;
        
        // Find hub entities (high degree)
        let mut hub_entities = Vec::new();
        for entity_key in &entity_keys {
            let degree = self.get_neighbors(*entity_key).await.len();
            if degree > 10 { // Threshold for hub
                hub_entities.push((*entity_key, degree));
            }
        }
        hub_entities.sort_by(|a, b| b.1.cmp(&a.1));
        pattern_analysis.hub_entities = hub_entities;
        
        // Find isolated entities
        let mut isolated_entities = Vec::new();
        for entity_key in &entity_keys {
            let degree = self.get_neighbors(*entity_key).await.len();
            if degree == 0 {
                isolated_entities.push(*entity_key);
            }
        }
        pattern_analysis.isolated_entities = isolated_entities;
        
        // Analyze activation patterns
        let activations = self.entity_activations.read().await;
        let mut activation_clusters = HashMap::new();
        
        for (&entity_key, &activation) in activations.iter() {
            let cluster = if activation < 0.3 {
                "low_activation"
            } else if activation < 0.7 {
                "medium_activation"
            } else {
                "high_activation"
            };
            
            activation_clusters.entry(cluster.to_string()).or_insert_with(Vec::new).push(entity_key);
        }
        pattern_analysis.activation_clusters = activation_clusters;
        
        pattern_analysis
    }

    /// Calculate graph efficiency
    pub async fn calculate_graph_efficiency(&self) -> f32 {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let n = entity_keys.len();
        
        if n < 2 {
            return 0.0;
        }
        
        let mut total_efficiency = 0.0;
        let mut pair_count = 0;
        
        // Sample pairs to avoid O(n²) complexity
        let sample_size = (n * 100).min(10000);
        
        for i in 0..sample_size {
            let source_idx = i % n;
            let target_idx = (i + 1) % n;
            
            if source_idx != target_idx {
                if let Some(path) = self.core_graph.find_path(entity_keys[source_idx], entity_keys[target_idx]) {
                    let path_length = path.len() - 1;
                    if path_length > 0 {
                        total_efficiency += 1.0 / path_length as f32;
                    }
                }
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            total_efficiency / pair_count as f32
        } else {
            0.0
        }
    }

    /// Generate analytics report
    pub async fn generate_analytics_report(&self) -> Result<String> {
        let stats = self.get_brain_statistics().await?;
        let critical_entities = self.get_critical_entities(5).await;
        let most_activated = self.get_most_activated_entities(5).await;
        let pattern_analysis = self.analyze_graph_patterns().await;
        let efficiency = self.calculate_graph_efficiency().await;
        
        let mut report = String::new();
        
        report.push_str("Brain-Enhanced Knowledge Graph Analytics Report\n");
        report.push_str("=============================================\n\n");
        
        // Basic statistics
        report.push_str(&format!("Entities: {}\n", stats.entity_count));
        report.push_str(&format!("Relationships: {}\n", stats.relationship_count));
        report.push_str(&format!("Graph Density: {:.4}\n", stats.graph_density));
        report.push_str(&format!("Average Activation: {:.4}\n", stats.avg_activation));
        report.push_str(&format!("Clustering Coefficient: {:.4}\n", stats.clustering_coefficient));
        report.push_str(&format!("Average Path Length: {:.4}\n", stats.average_path_length));
        report.push_str(&format!("Graph Efficiency: {:.4}\n", efficiency));
        report.push_str(&format!("Concept Coherence: {:.4}\n", stats.concept_coherence));
        report.push_str(&format!("Learning Efficiency: {:.4}\n", stats.learning_efficiency));
        
        // Critical entities
        report.push_str("\nCritical Entities (by betweenness centrality):\n");
        for (entity_key, centrality) in critical_entities {
            report.push_str(&format!("  {:?}: {:.4}\n", entity_key, centrality));
        }
        
        // Most activated entities
        report.push_str("\nMost Activated Entities:\n");
        for (entity_key, activation) in most_activated {
            report.push_str(&format!("  {:?}: {:.4}\n", entity_key, activation));
        }
        
        // Pattern analysis
        report.push_str(&format!("\nHub Entities: {}\n", pattern_analysis.hub_entities.len()));
        report.push_str(&format!("Isolated Entities: {}\n", pattern_analysis.isolated_entities.len()));
        
        // Activation distribution
        report.push_str("\nActivation Distribution:\n");
        for (level, count) in &stats.activation_distribution {
            report.push_str(&format!("  {}: {}\n", level, count));
        }
        
        // Health assessment
        let health_score = stats.graph_health_score();
        report.push_str(&format!("\nOverall Health Score: {:.4}\n", health_score));
        
        if health_score > 0.7 {
            report.push_str("Status: Healthy\n");
        } else if health_score > 0.5 {
            report.push_str("Status: Moderate - Some optimization needed\n");
        } else {
            report.push_str("Status: Poor - Significant optimization needed\n");
        }
        
        Ok(report)
    }
}

/// Concept usage statistics
#[derive(Debug, Clone)]
pub struct ConceptUsageStats {
    pub total_entities: usize,
    pub avg_activation: f32,
    pub coherence: f32,
    pub complexity: f32,
    pub last_accessed: std::time::Instant,
}

/// Graph pattern analysis
#[derive(Debug, Clone)]
pub struct GraphPatternAnalysis {
    pub degree_distribution: HashMap<usize, usize>,
    pub hub_entities: Vec<(EntityKey, usize)>,
    pub isolated_entities: Vec<EntityKey>,
    pub activation_clusters: HashMap<String, Vec<EntityKey>>,
}

impl GraphPatternAnalysis {
    pub fn new() -> Self {
        Self {
            degree_distribution: HashMap::new(),
            hub_entities: Vec::new(),
            isolated_entities: Vec::new(),
            activation_clusters: HashMap::new(),
        }
    }
    
    /// Get total number of patterns detected
    pub fn total_patterns(&self) -> usize {
        self.hub_entities.len() + self.isolated_entities.len() + self.activation_clusters.len()
    }
    
    /// Check if graph has scale-free properties
    pub fn is_scale_free(&self) -> bool {
        // Simplified check: presence of hub entities suggests scale-free structure
        self.hub_entities.len() > 0 && self.hub_entities.len() < self.degree_distribution.len() / 10
    }
    
    /// Get most common degree
    pub fn most_common_degree(&self) -> Option<usize> {
        self.degree_distribution
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(degree, _)| *degree)
    }
}