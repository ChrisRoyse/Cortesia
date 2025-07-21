//! Analytics and statistics for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::collections::HashMap;

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
        
        let max_degree = betweenness_centrality.len(); // Calculate before moving
        let average_degree = if entity_count > 0 { relationship_count as f32 / entity_count as f32 } else { 0.0 };
        
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
            max_degree,
            average_degree,
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
    async fn _calculate_betweenness_centrality_analytics(&self) -> HashMap<EntityKey, f32> {
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
    fn _calculate_activation_distribution_analytics(&self, activations: &[f32]) -> HashMap<String, usize> {
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
            let neighbors = self.get_neighbors(entity_key);
            for neighbor_key in neighbors {
                // Simulate usage frequency with default weight since get_neighbors doesn't return weights
                let usage = 50u32; // Default frequency
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
            let degree = self.get_neighbors(*entity_key).len();
            *degree_distribution.entry(degree).or_insert(0) += 1;
        }
        pattern_analysis.degree_distribution = degree_distribution;
        
        // Find hub entities (high degree)
        let mut hub_entities = Vec::new();
        for entity_key in &entity_keys {
            let degree = self.get_neighbors(*entity_key).len();
            if degree > 10 { // Threshold for hub
                hub_entities.push((*entity_key, degree));
            }
        }
        hub_entities.sort_by(|a, b| b.1.cmp(&a.1));
        pattern_analysis.hub_entities = hub_entities;
        
        // Find isolated entities
        let mut isolated_entities = Vec::new();
        for entity_key in &entity_keys {
            let degree = self.get_neighbors(*entity_key).len();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityData;
    use std::collections::HashMap;
    use tokio;

    // Helper function to create a test graph with entities and relationships
    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Add test entities
        let entity1 = graph.core_graph.add_entity(EntityData::new(1, "entity1".to_string(), vec![0.0; 64])).unwrap();
        let entity2 = graph.core_graph.add_entity(EntityData::new(1, "entity2".to_string(), vec![0.0; 64])).unwrap();
        let entity3 = graph.core_graph.add_entity(EntityData::new(1, "entity3".to_string(), vec![0.0; 64])).unwrap();
        let entity4 = graph.core_graph.add_entity(EntityData::new(1, "entity4".to_string(), vec![0.0; 64])).unwrap();
        
        // Add relationships to create a connected graph
        let _ = graph.core_graph.add_relationship(entity1, entity2, 0.5);
        let _ = graph.core_graph.add_relationship(entity2, entity3, 0.5);
        let _ = graph.core_graph.add_relationship(entity3, entity4, 0.5);
        let _ = graph.core_graph.add_relationship(entity1, entity3, 0.5); // Creates triangles
        
        // Set up some activations
        {
            let mut activations = graph.entity_activations.write().await;
            activations.insert(entity1, 0.8);
            activations.insert(entity2, 0.6);
            activations.insert(entity3, 0.4);
            activations.insert(entity4, 0.2);
        }
        
        // Set up concept structures for testing
        {
            let mut concepts = graph.concept_structures.write().await;
            let mut concept = ConceptStructure::new();
            concept.add_input(entity1);
            concept.add_output(entity2);
            concept.concept_activation = 0.7;
            concept.coherence_score = 0.8;
            concepts.insert("test_concept".to_string(), concept);
        }
        
        graph
    }

    // Helper function to create an empty test graph
    async fn create_empty_graph() -> BrainEnhancedKnowledgeGraph {
        BrainEnhancedKnowledgeGraph::new_for_test().unwrap()
    }

    #[tokio::test]
    async fn test_get_brain_statistics_with_entities() {
        let graph = create_test_graph().await;
        let stats = graph.get_brain_statistics().await.unwrap();
        
        // Verify basic counts
        assert_eq!(stats.entity_count, 4);
        assert_eq!(stats.relationship_count, 4);
        
        // Verify activation statistics
        assert!((stats.avg_activation - 0.5).abs() < 0.1); // (0.8+0.6+0.4+0.2)/4 = 0.5
        assert_eq!(stats.max_activation, 0.8);
        assert_eq!(stats.min_activation, 0.2);
        
        // Verify graph density is calculated
        assert!(stats.graph_density > 0.0);
        assert!(stats.graph_density <= 1.0);
        
        // Verify clustering coefficient is calculated
        assert!(stats.clustering_coefficient >= 0.0);
        assert!(stats.clustering_coefficient <= 1.0);
        
        // Verify concept coherence from test concept
        assert!((stats.concept_coherence - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_get_brain_statistics_empty_graph() {
        let graph = create_empty_graph().await;
        let stats = graph.get_brain_statistics().await.unwrap();
        
        // Verify empty graph statistics
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.relationship_count, 0);
        assert_eq!(stats.avg_activation, 0.0);
        assert_eq!(stats.max_activation, 0.0);
        assert_eq!(stats.min_activation, 0.0);
        assert_eq!(stats.graph_density, 0.0);
        assert_eq!(stats.clustering_coefficient, 0.0);
        assert_eq!(stats.average_path_length, 0.0);
        assert_eq!(stats.concept_coherence, 0.0);
        assert!(stats.betweenness_centrality.is_empty());
        assert!(stats.activation_distribution.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_average_clustering_coefficient() {
        let graph = create_test_graph().await;
        let clustering = graph.calculate_average_clustering_coefficient().await;
        
        // Should be a valid clustering coefficient
        assert!(clustering >= 0.0);
        assert!(clustering <= 1.0);
    }

    #[tokio::test]
    async fn test_calculate_average_clustering_coefficient_empty_graph() {
        let graph = create_empty_graph().await;
        let clustering = graph.calculate_average_clustering_coefficient().await;
        
        // Empty graph should have 0 clustering
        assert_eq!(clustering, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_average_clustering_coefficient_single_entity() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        let entity = graph.core_graph.add_entity(EntityData::new(1, "single".to_string(), vec![0.0; 64])).unwrap();
        
        let clustering = graph.calculate_average_clustering_coefficient().await;
        
        // Single entity should have 0 clustering
        assert_eq!(clustering, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_average_path_length() {
        let graph = create_test_graph().await;
        let avg_path = graph.calculate_average_path_length().await;
        
        // Should be a reasonable path length for connected graph
        assert!(avg_path >= 0.0);
        assert!(avg_path <= 10.0); // Reasonable upper bound for test graph
    }

    #[tokio::test]
    async fn test_calculate_average_path_length_empty_graph() {
        let graph = create_empty_graph().await;
        let avg_path = graph.calculate_average_path_length().await;
        
        // Empty graph should have 0 path length
        assert_eq!(avg_path, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_average_path_length_single_entity() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        let entity = graph.core_graph.add_entity(EntityData::new(1, "single".to_string(), vec![0.0; 64])).unwrap();
        
        let avg_path = graph.calculate_average_path_length().await;
        
        // Single entity should have 0 path length
        assert_eq!(avg_path, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_activation_distribution() {
        let graph = create_test_graph().await;
        let activations = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let distribution = graph.calculate_activation_distribution(&activations);
        
        // Should distribute into correct buckets
        assert_eq!(distribution.len(), 5); // Should have 5 different buckets
        assert_eq!(distribution.get("0.0-0.1").unwrap_or(&0), &0); // 0.1 goes to 0.1-0.2
        assert_eq!(distribution.get("0.1-0.2").unwrap_or(&0), &1); // 0.1
        assert_eq!(distribution.get("0.2-0.3").unwrap_or(&0), &0); // 0.3 goes to 0.3-0.4
        assert_eq!(distribution.get("0.3-0.4").unwrap_or(&0), &1); // 0.3
        assert_eq!(distribution.get("0.4-0.5").unwrap_or(&0), &0); // 0.5 goes to 0.5-0.6
    }

    #[tokio::test]
    async fn test_calculate_activation_distribution_empty() {
        let graph = create_empty_graph().await;
        let activations = vec![];
        let distribution = graph.calculate_activation_distribution(&activations);
        
        // Empty distribution
        assert!(distribution.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_activation_distribution_edge_values() {
        let graph = create_test_graph().await;
        let activations = vec![0.0, 0.1, 0.5, 1.0];
        let distribution = graph.calculate_activation_distribution(&activations);
        
        // Test boundary values
        assert_eq!(distribution.get("0.0-0.1").unwrap_or(&0), &1); // 0.0
        assert_eq!(distribution.get("0.1-0.2").unwrap_or(&0), &1); // 0.1
        assert_eq!(distribution.get("0.5-0.6").unwrap_or(&0), &1); // 0.5
        assert_eq!(distribution.get("0.9-1.0").unwrap_or(&0), &1); // 1.0
    }

    #[tokio::test]
    async fn test_calculate_average_concept_coherence() {
        let graph = create_test_graph().await;
        let coherence = graph.calculate_average_concept_coherence().await;
        
        // Should match our test concept coherence (0.8)
        assert!((coherence - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_calculate_average_concept_coherence_empty() {
        let graph = create_empty_graph().await;
        let coherence = graph.calculate_average_concept_coherence().await;
        
        // No concepts should return 0
        assert_eq!(coherence, 0.0);
    }

    #[tokio::test]
    async fn test_calculate_average_concept_coherence_multiple_concepts() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Add multiple concepts with different coherence scores
        {
            let mut concepts = graph.concept_structures.write().await;
            
            let mut concept1 = ConceptStructure::new();
            concept1.coherence_score = 0.6;
            concepts.insert("concept1".to_string(), concept1);
            
            let mut concept2 = ConceptStructure::new();
            concept2.coherence_score = 0.8;
            concepts.insert("concept2".to_string(), concept2);
            
            let mut concept3 = ConceptStructure::new();
            concept3.coherence_score = 1.0;
            concepts.insert("concept3".to_string(), concept3);
        }
        
        let coherence = graph.calculate_average_concept_coherence().await;
        
        // Should be average: (0.6 + 0.8 + 1.0) / 3 = 0.8
        assert!((coherence - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_analyze_graph_patterns() {
        let graph = create_test_graph().await;
        let patterns = graph.analyze_graph_patterns().await;
        
        // Verify degree distribution analysis
        assert!(!patterns.degree_distribution.is_empty());
        
        // Verify pattern counts
        assert!(patterns.total_patterns() > 0);
        
        // Verify activation clusters are identified
        assert!(!patterns.activation_clusters.is_empty());
        
        // Should have entities in different activation clusters
        let low_cluster = patterns.activation_clusters.get("low_activation");
        let medium_cluster = patterns.activation_clusters.get("medium_activation");
        let high_cluster = patterns.activation_clusters.get("high_activation");
        
        // Based on our test data: entity4(0.2) = low, entity3(0.4) = medium, entity2(0.6) = medium, entity1(0.8) = high
        assert!(low_cluster.is_some());
        assert!(medium_cluster.is_some());
        assert!(high_cluster.is_some());
    }

    #[tokio::test]
    async fn test_analyze_graph_patterns_empty_graph() {
        let graph = create_empty_graph().await;
        let patterns = graph.analyze_graph_patterns().await;
        
        // Empty graph should have empty patterns
        assert!(patterns.degree_distribution.is_empty());
        assert!(patterns.hub_entities.is_empty());
        assert!(patterns.isolated_entities.is_empty());
        assert!(patterns.activation_clusters.is_empty());
        assert_eq!(patterns.total_patterns(), 0);
    }

    #[tokio::test]
    async fn test_analyze_graph_patterns_isolated_entities() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Add isolated entities (no relationships)
        let entity1 = graph.core_graph.add_entity(EntityData::new(1, "isolated1".to_string(), vec![0.0; 64])).unwrap();
        let entity2 = graph.core_graph.add_entity(EntityData::new(1, "isolated2".to_string(), vec![0.0; 64])).unwrap();
        
        let patterns = graph.analyze_graph_patterns().await;
        
        // Should identify isolated entities
        assert_eq!(patterns.isolated_entities.len(), 2);
        assert!(patterns.isolated_entities.contains(&entity1));
        assert!(patterns.isolated_entities.contains(&entity2));
    }

    #[tokio::test]
    async fn test_analyze_graph_patterns_hub_detection() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Create a hub entity connected to many others
        let hub = graph.core_graph.add_entity(EntityData::new(1, "hub".to_string(), vec![0.0; 64])).unwrap();
        
        // Connect hub to 12 entities (above hub threshold of 10)
        for i in 0..12 {
            let entity = graph.core_graph.add_entity(EntityData::new(1, format!("spoke_{}", i), vec![])).unwrap();
            let _ = graph.core_graph.add_relationship(hub, entity, 0.5);
        }
        
        let patterns = graph.analyze_graph_patterns().await;
        
        // Should identify hub entity
        assert!(!patterns.hub_entities.is_empty());
        assert_eq!(patterns.hub_entities[0].0, hub);
        assert_eq!(patterns.hub_entities[0].1, 12); // Should have degree 12
    }

    #[tokio::test]
    async fn test_graph_pattern_analysis_scale_free_detection() {
        let mut patterns = GraphPatternAnalysis::new();
        
        // Add hub entities (small number with high degree)
        patterns.hub_entities.push((EntityKey::from_raw_parts(1, 0), 20));
        patterns.hub_entities.push((EntityKey::from_raw_parts(2, 0), 15));
        
        // Add degree distribution simulating scale-free network
        patterns.degree_distribution.insert(1, 50);  // Many low-degree nodes
        patterns.degree_distribution.insert(2, 25);
        patterns.degree_distribution.insert(3, 10);
        patterns.degree_distribution.insert(15, 1); // Few high-degree nodes
        patterns.degree_distribution.insert(20, 1);
        
        // Should detect as scale-free
        assert!(patterns.is_scale_free());
    }

    #[tokio::test]
    async fn test_graph_pattern_analysis_most_common_degree() {
        let mut patterns = GraphPatternAnalysis::new();
        
        patterns.degree_distribution.insert(1, 10);
        patterns.degree_distribution.insert(2, 25); // Most common
        patterns.degree_distribution.insert(3, 5);
        
        assert_eq!(patterns.most_common_degree(), Some(2));
    }

    #[tokio::test]
    async fn test_statistics_mathematical_validation() {
        let graph = create_test_graph().await;
        let stats = graph.get_brain_statistics().await.unwrap();
        
        // Mathematical constraints validation
        assert!(stats.graph_density >= 0.0 && stats.graph_density <= 1.0);
        assert!(stats.clustering_coefficient >= 0.0 && stats.clustering_coefficient <= 1.0);
        assert!(stats.avg_activation >= stats.min_activation);
        assert!(stats.max_activation >= stats.avg_activation);
        assert!(stats.concept_coherence >= 0.0 && stats.concept_coherence <= 1.0);
        assert!(stats.learning_efficiency >= 0.0);
        
        // Graph density formula validation: actual_edges / max_possible_edges
        let n = stats.entity_count;
        if n > 1 {
            let max_possible = n * (n - 1);
            let expected_density = stats.relationship_count as f32 / max_possible as f32;
            assert!((stats.graph_density - expected_density).abs() < 0.001);
        }
    }

    #[tokio::test]
    async fn test_activation_statistics_edge_cases() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Test with single entity
        let entity = graph.core_graph.add_entity(EntityData::new(1, "single".to_string(), vec![0.0; 64])).unwrap();
        
        {
            let mut activations = graph.entity_activations.write().await;
            activations.insert(entity, 0.5);
        }
        
        let stats = graph.get_brain_statistics().await.unwrap();
        
        // Single entity statistics
        assert_eq!(stats.entity_count, 1);
        assert_eq!(stats.avg_activation, 0.5);
        assert_eq!(stats.max_activation, 0.5);
        assert_eq!(stats.min_activation, 0.5);
    }

    #[tokio::test]
    async fn test_clustering_coefficient_triangle_detection() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Create a perfect triangle (each node connected to every other)
        let a = graph.core_graph.add_entity(EntityData::new(1, "a".to_string(), vec![0.0; 64])).unwrap();
        let b = graph.core_graph.add_entity(EntityData::new(1, "b".to_string(), vec![0.0; 64])).unwrap();
        let c = graph.core_graph.add_entity(EntityData::new(1, "c".to_string(), vec![0.0; 64])).unwrap();
        
        // Connect all pairs
        let _ = graph.core_graph.add_relationship(a, b, 0.5);
        let _ = graph.core_graph.add_relationship(b, c, 0.5);
        let _ = graph.core_graph.add_relationship(c, a, 0.5);
        
        let clustering = graph.calculate_average_clustering_coefficient().await;
        
        // Perfect triangle should have high clustering coefficient
        assert!(clustering > 0.8); // Should be close to 1.0 for perfect triangle
    }

    #[tokio::test]
    async fn test_path_length_calculation_accuracy() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Create a linear chain: A -> B -> C -> D
        let a = graph.core_graph.add_entity(EntityData::new(1, "a".to_string(), vec![0.0; 64])).unwrap();
        let b = graph.core_graph.add_entity(EntityData::new(1, "b".to_string(), vec![0.0; 64])).unwrap();
        let c = graph.core_graph.add_entity(EntityData::new(1, "c".to_string(), vec![0.0; 64])).unwrap();
        let d = graph.core_graph.add_entity(EntityData::new(1, "d".to_string(), vec![0.0; 64])).unwrap();
        
        let _ = graph.core_graph.add_relationship(a, b, 0.5);
        let _ = graph.core_graph.add_relationship(b, c, 0.5);
        let _ = graph.core_graph.add_relationship(c, d, 0.5);
        
        let avg_path = graph.calculate_average_path_length().await;
        
        // In a linear chain of 4 nodes, average path length should be around 2
        // (paths: A-B=1, A-C=2, A-D=3, B-C=1, B-D=2, C-D=1, average=(1+2+3+1+2+1)/6=1.67)
        assert!(avg_path >= 1.0 && avg_path <= 3.0);
    }

    #[tokio::test]
    async fn test_betweenness_centrality_calculation() {
        let graph = create_test_graph().await;
        let centrality = graph.calculate_betweenness_centrality().await;
        
        // Should have centrality scores for all entities
        assert_eq!(centrality.len(), 4);
        
        // All centrality scores should be valid (0.0 to 1.0)
        for &score in centrality.values() {
            assert!(score >= 0.0);
            assert!(score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_concept_structure_validation() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
        
        // Test concept structure with complex configuration
        {
            let mut concepts = graph.concept_structures.write().await;
            
            let mut complex_concept = ConceptStructure::new();
            complex_concept.add_input(EntityKey::from_raw_parts(1, 0));
            complex_concept.add_input(EntityKey::from_raw_parts(2, 0));
            complex_concept.add_output(EntityKey::from_raw_parts(3, 0));
            complex_concept.add_gate(EntityKey::from_raw_parts(4, 0));
            complex_concept.concept_activation = 0.75;
            complex_concept.coherence_score = 0.9;
            
            concepts.insert("complex_concept".to_string(), complex_concept);
        }
        
        let coherence = graph.calculate_average_concept_coherence().await;
        assert!((coherence - 0.9).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_performance_with_large_activation_set() {
        let graph = create_empty_graph().await;
        
        // Test with large activation dataset
        let large_activations: Vec<f32> = (0..10000).map(|i| (i % 100) as f32 / 100.0).collect();
        let distribution = graph.calculate_activation_distribution(&large_activations);
        
        // Should handle large datasets efficiently
        assert_eq!(distribution.len(), 10); // Should have all 10 buckets
        
        // Verify total count matches input
        let total_count: usize = distribution.values().sum();
        assert_eq!(total_count, 10000);
    }
}