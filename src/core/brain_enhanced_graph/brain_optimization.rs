//! Optimization operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::error::Result;

impl BrainEnhancedKnowledgeGraph {
    /// Assess graph health
    pub async fn assess_graph_health(&self) -> Result<GraphHealthMetrics> {
        let entity_count = self.entity_count();
        let relationship_count = self.relationship_count();
        
        if entity_count == 0 {
            return Ok(GraphHealthMetrics {
                connectivity_score: 0.0,
                activation_balance: 0.0,
                learning_stability: 0.0,
                concept_coherence: 0.0,
                overall_health: 0.0,
            });
        }
        
        // Calculate connectivity score
        let connectivity_score = self.calculate_connectivity_score(entity_count, relationship_count).await;
        
        // Calculate activation balance
        let activation_balance = self.calculate_activation_balance().await;
        
        // Calculate learning stability
        let learning_stability = self.calculate_learning_stability().await;
        
        // Calculate concept coherence
        let concept_coherence = self.calculate_overall_concept_coherence().await;
        
        // Calculate overall health
        let overall_health = (connectivity_score + activation_balance + learning_stability + concept_coherence) / 4.0;
        
        Ok(GraphHealthMetrics {
            connectivity_score,
            activation_balance,
            learning_stability,
            concept_coherence,
            overall_health,
        })
    }

    /// Optimize graph structure
    pub async fn optimize_graph_structure(&self) -> Result<OptimizationResult> {
        let mut optimization_result = OptimizationResult::new();
        
        // 1. Prune weak relationships
        let weak_threshold = 0.1;
        let pruned_relationships = self.prune_weak_relationships(weak_threshold).await?;
        optimization_result.pruned_relationships = pruned_relationships;
        
        // 2. Strengthen frequently co-activated relationships
        let strengthened_relationships = self.strengthen_coactivated_relationships().await?;
        optimization_result.strengthened_relationships = strengthened_relationships;
        
        // 3. Create new learned relationships
        let new_relationships = self.create_new_learned_relationships().await?;
        optimization_result.new_relationships = new_relationships;
        
        // 4. Optimize concept structures
        let optimized_concepts = self.optimize_concept_structures().await?;
        optimization_result.optimized_concepts = optimized_concepts;
        
        Ok(optimization_result)
    }

    /// Calculate connectivity score
    pub(crate) async fn calculate_connectivity_score(&self, entity_count: usize, relationship_count: usize) -> f32 {
        if entity_count <= 1 {
            return 0.0;
        }
        
        let max_possible_relationships = entity_count * (entity_count - 1);
        let density = relationship_count as f32 / max_possible_relationships as f32;
        
        // Optimal density is around 0.1-0.3 for most graphs
        if density < 0.1 {
            density * 10.0 // Scale up sparse graphs
        } else if density > 0.3 {
            (1.0 - density) * 1.43 // Scale down dense graphs (1/0.7)
        } else {
            1.0 // Optimal range
        }
    }

    /// Calculate activation balance
    pub(crate) async fn calculate_activation_balance(&self) -> f32 {
        let activations = self.entity_activations.read().await;
        
        if activations.is_empty() {
            return 0.0;
        }
        
        let total_activation: f32 = activations.values().sum();
        let avg_activation = total_activation / activations.len() as f32;
        
        // Balance is better when average is around 0.5
        1.0 - (avg_activation - 0.5).abs() * 2.0
    }

    /// Calculate learning stability
    pub(crate) async fn calculate_learning_stability(&self) -> f32 {
        let stats = self.learning_stats.read().await;
        stats.learning_efficiency.clamp(0.0, 1.0)
    }

    /// Calculate overall concept coherence
    pub(crate) async fn calculate_overall_concept_coherence(&self) -> f32 {
        let concepts = self.concept_structures.read().await;
        
        if concepts.is_empty() {
            return 0.0;
        }
        
        let total_coherence: f32 = concepts.values().map(|c| c.coherence_score).sum();
        total_coherence / concepts.len() as f32
    }

    /// Strengthen co-activated relationships
    pub(crate) async fn strengthen_coactivated_relationships(&self) -> Result<usize> {
        let mut strengthened_count = 0;
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        for entity_key in entity_keys {
            let neighbors = self.get_neighbors_with_weights(entity_key).await;
            let entity_activation = self.get_entity_activation(entity_key).await;
            
            for (neighbor_key, _) in neighbors {
                let neighbor_activation = self.get_entity_activation(neighbor_key).await;
                
                // If both entities are highly activated, strengthen their connection
                if entity_activation > 0.7 && neighbor_activation > 0.7 {
                    self.strengthen_relationship(entity_key, neighbor_key).await?;
                    strengthened_count += 1;
                }
            }
        }
        
        Ok(strengthened_count)
    }

    /// Create new learned relationships
    pub(crate) async fn create_new_learned_relationships(&self) -> Result<usize> {
        let mut new_count = 0;
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        for i in 0..entity_keys.len() {
            for j in i + 1..entity_keys.len() {
                let entity1 = entity_keys[i];
                let entity2 = entity_keys[j];
                
                // Skip if relationship already exists
                if self.has_relationship(entity1, entity2).await ||
                   self.has_relationship(entity2, entity1).await {
                    continue;
                }
                
                // Create relationship if entities are co-activated
                let activation1 = self.get_entity_activation(entity1).await;
                let activation2 = self.get_entity_activation(entity2).await;
                
                if activation1 > 0.8 && activation2 > 0.8 {
                    self.create_learned_relationship(entity1, entity2).await?;
                    new_count += 1;
                }
            }
        }
        
        Ok(new_count)
    }

    /// Optimize concept structures
    pub(crate) async fn optimize_concept_structures(&self) -> Result<usize> {
        let concept_names = self.get_concept_names().await;
        let mut optimized_count = 0;
        
        for concept_name in concept_names {
            if let Some(mut concept) = self.get_concept_structure(&concept_name).await {
                // Recalculate activation and coherence
                let old_coherence = concept.coherence_score;
                concept.concept_activation = self.calculate_concept_activation(&concept).await;
                concept.coherence_score = self.calculate_concept_coherence(&concept).await;
                
                // Only update if coherence improved
                if concept.coherence_score > old_coherence {
                    self.store_concept_structure(concept_name, concept).await;
                    optimized_count += 1;
                }
            }
        }
        
        Ok(optimized_count)
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub pruned_relationships: usize,
    pub strengthened_relationships: usize,
    pub new_relationships: usize,
    pub optimized_concepts: usize,
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationResult {
    pub fn new() -> Self {
        Self {
            pruned_relationships: 0,
            strengthened_relationships: 0,
            new_relationships: 0,
            optimized_concepts: 0,
        }
    }
    
    pub fn total_changes(&self) -> usize {
        self.pruned_relationships + self.strengthened_relationships + self.new_relationships + self.optimized_concepts
    }
}