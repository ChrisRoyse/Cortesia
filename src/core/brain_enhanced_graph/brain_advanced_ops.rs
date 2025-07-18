//! Advanced operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::core::sdr_storage::{SDRQuery, SDR};
use crate::error::Result;
use std::collections::{HashMap, HashSet};

impl BrainEnhancedKnowledgeGraph {
    /// Create concept structure from related entities
    pub async fn create_concept_structure(&self, concept_name: String, entity_keys: Vec<EntityKey>) -> Result<ConceptStructure> {
        let mut concept_structure = ConceptStructure::new();
        
        // Analyze entities to determine their roles
        let mut input_entities = Vec::new();
        let mut output_entities = Vec::new();
        let mut gate_entities = Vec::new();
        
        for entity_key in entity_keys {
            let entity_role = self.determine_entity_role(entity_key).await;
            
            match entity_role {
                EntityRole::Input => input_entities.push(entity_key),
                EntityRole::Output => output_entities.push(entity_key),
                EntityRole::Gate => gate_entities.push(entity_key),
                EntityRole::Processing => {
                    // Processing entities can be either inputs or outputs based on connectivity
                    let incoming_count = self.get_parent_entities(entity_key).await.len();
                    let outgoing_count = self.get_child_entities(entity_key).await.len();
                    
                    if incoming_count > outgoing_count {
                        output_entities.push(entity_key);
                    } else {
                        input_entities.push(entity_key);
                    }
                }
            }
        }
        
        // Set concept structure
        concept_structure.input_entities = input_entities;
        concept_structure.output_entities = output_entities;
        concept_structure.gate_entities = gate_entities;
        
        // Calculate concept activation and coherence
        concept_structure.concept_activation = self.calculate_concept_activation(&concept_structure).await;
        concept_structure.coherence_score = self.calculate_concept_coherence(&concept_structure).await;
        
        // Store concept structure
        self.store_concept_structure(concept_name, concept_structure.clone()).await;
        
        Ok(concept_structure)
    }

    /// Find similar concepts using SDR
    pub async fn find_similar_concepts(&self, concept_name: &str, k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(concept_structure) = self.get_concept_structure(concept_name).await {
            // Calculate concept embedding
            // Use simple placeholder embedding for now
            let concept_embedding = vec![0.0; 384];
            
            // Create SDR query
            // Create SDR from concept embedding
            let sdr_config = crate::core::sdr_storage::SDRConfig::default();
            let query_sdr = SDR::from_dense_vector(&concept_embedding, &sdr_config);
            
            let sdr_query = SDRQuery {
                query_sdr,
                top_k: k,
                min_overlap: 0.5,
            };
            
            // Search for similar concepts
            let sdr_results = self.sdr_storage.find_similar_patterns(&sdr_query.query_sdr, sdr_query.top_k).await?;
            
            // Convert results
            let mut similar_concepts = Vec::new();
            for (pattern_id, similarity_score) in sdr_results {
                // Check if this ID corresponds to a concept
                if let Some(other_concept) = self.get_concept_structure(&pattern_id).await {
                    let similarity = self.calculate_concept_similarity(&concept_structure, &other_concept).await;
                    similar_concepts.push((pattern_id, similarity));
                }
            }
            
            // Sort by similarity
            similar_concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            similar_concepts.truncate(k);
            
            Ok(similar_concepts)
        } else {
            Err(crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept_name)))
        }
    }

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

    /// Create inheritance relationship
    pub async fn create_inheritance(&self, parent: EntityKey, child: EntityKey, inheritance_strength: f32) -> Result<()> {
        // Create inheritance relationship
        let inheritance_relationship = crate::core::types::Relationship {
            from: parent,
            to: child,
            rel_type: 1, // Inheritance relationship type
            weight: inheritance_strength,
        };
        
        self.insert_brain_relationship(inheritance_relationship).await?;
        
        // Copy partial activation from parent to child
        let parent_activation = self.get_entity_activation(parent).await;
        let child_activation = self.get_entity_activation(child).await;
        
        let inherited_activation = parent_activation * inheritance_strength * 0.3;
        let new_child_activation = (child_activation + inherited_activation).clamp(0.0, 1.0);
        
        self.set_entity_activation(child, new_child_activation).await;
        
        Ok(())
    }

    /// Perform concept merging
    pub async fn merge_concepts(&self, concept1_name: &str, concept2_name: &str, merged_name: String) -> Result<ConceptStructure> {
        let concept1 = self.get_concept_structure(concept1_name).await
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept1_name)))?;
        
        let concept2 = self.get_concept_structure(concept2_name).await
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept2_name)))?;
        
        // Merge concepts
        let mut merged_concept = ConceptStructure::new();
        
        // Combine input entities
        let mut all_inputs = concept1.input_entities.clone();
        all_inputs.extend(concept2.input_entities);
        merged_concept.input_entities = all_inputs.into_iter().collect::<HashSet<_>>().into_iter().collect();
        
        // Combine output entities
        let mut all_outputs = concept1.output_entities.clone();
        all_outputs.extend(concept2.output_entities);
        merged_concept.output_entities = all_outputs.into_iter().collect::<HashSet<_>>().into_iter().collect();
        
        // Combine gate entities
        let mut all_gates = concept1.gate_entities.clone();
        all_gates.extend(concept2.gate_entities);
        merged_concept.gate_entities = all_gates.into_iter().collect::<HashSet<_>>().into_iter().collect();
        
        // Calculate merged activation and coherence
        merged_concept.concept_activation = (concept1.concept_activation + concept2.concept_activation) / 2.0;
        merged_concept.coherence_score = self.calculate_concept_coherence(&merged_concept).await;
        
        // Store merged concept
        self.store_concept_structure(merged_name, merged_concept.clone()).await;
        
        // Remove original concepts
        self.remove_concept_structure(concept1_name).await;
        self.remove_concept_structure(concept2_name).await;
        
        Ok(merged_concept)
    }

    /// Perform concept splitting
    pub async fn split_concept(&self, concept_name: &str, split_criteria: SplitCriteria) -> Result<(ConceptStructure, ConceptStructure)> {
        let original_concept = self.get_concept_structure(concept_name).await
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept_name)))?;
        
        let (concept1, concept2) = match split_criteria {
            SplitCriteria::ByConnectivity => self.split_by_connectivity(&original_concept).await,
            SplitCriteria::ByActivation => self.split_by_activation(&original_concept).await,
            SplitCriteria::ByType => self.split_by_type(&original_concept).await,
        };
        
        // Store split concepts
        let concept1_name = format!("{}_part1", concept_name);
        let concept2_name = format!("{}_part2", concept_name);
        
        self.store_concept_structure(concept1_name, concept1.clone()).await;
        self.store_concept_structure(concept2_name, concept2.clone()).await;
        
        // Remove original concept
        self.remove_concept_structure(concept_name).await;
        
        Ok((concept1, concept2))
    }

    /// Determine entity role in concept
    async fn determine_entity_role(&self, entity_key: EntityKey) -> EntityRole {
        if let Some(data) = self.core_graph.get_entity_data(entity_key) {
            // Check entity type from properties string
            if data.properties.contains("input") {
                return EntityRole::Input;
            } else if data.properties.contains("output") {
                return EntityRole::Output;
            } else if data.properties.contains("logic_gate") {
                return EntityRole::Gate;
            }
        }
        
        // Determine role based on connectivity
        let incoming_count = self.get_parent_entities(entity_key).await.len();
        let outgoing_count = self.get_child_entities(entity_key).await.len();
        
        if incoming_count == 0 && outgoing_count > 0 {
            EntityRole::Input
        } else if incoming_count > 0 && outgoing_count == 0 {
            EntityRole::Output
        } else if incoming_count > 1 && outgoing_count > 1 {
            EntityRole::Gate
        } else {
            EntityRole::Processing
        }
    }

    /// Calculate concept activation
    async fn calculate_concept_activation(&self, concept: &ConceptStructure) -> f32 {
        let mut total_activation = 0.0;
        let mut entity_count = 0;
        
        for entity_key in concept.get_all_entities() {
            let activation = self.get_entity_activation(entity_key).await;
            total_activation += activation;
            entity_count += 1;
        }
        
        if entity_count > 0 {
            total_activation / entity_count as f32
        } else {
            0.0
        }
    }

    /// Calculate concept coherence
    async fn calculate_concept_coherence(&self, concept: &ConceptStructure) -> f32 {
        let entities = concept.get_all_entities();
        
        if entities.len() < 2 {
            return 1.0;
        }
        
        let mut total_similarity = 0.0;
        let mut pair_count = 0;
        
        // Calculate average similarity between all pairs
        for i in 0..entities.len() {
            for j in i + 1..entities.len() {
                if let (Some(data1), Some(data2)) = (
                    self.core_graph.get_entity_data(entities[i]),
                    self.core_graph.get_entity_data(entities[j])
                ) {
                    let similarity = self.calculate_embedding_similarity(&data1.embedding, &data2.embedding);
                    total_similarity += similarity;
                    pair_count += 1;
                }
            }
        }
        
        if pair_count > 0 {
            total_similarity / pair_count as f32
        } else {
            0.0
        }
    }

    /// Calculate similarity between two embeddings
    fn calculate_embedding_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Calculate concept similarity
    async fn calculate_concept_similarity(&self, concept1: &ConceptStructure, concept2: &ConceptStructure) -> f32 {
        // Use simple placeholder embeddings for now
        let embedding1 = vec![0.0; 384];
        let embedding2 = vec![0.0; 384];
        
        self.calculate_embedding_similarity(&embedding1, &embedding2)
    }

    /// Calculate connectivity score
    async fn calculate_connectivity_score(&self, entity_count: usize, relationship_count: usize) -> f32 {
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
    async fn calculate_activation_balance(&self) -> f32 {
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
    async fn calculate_learning_stability(&self) -> f32 {
        let stats = self.learning_stats.read().await;
        stats.learning_efficiency.clamp(0.0, 1.0)
    }

    /// Calculate overall concept coherence
    async fn calculate_overall_concept_coherence(&self) -> f32 {
        let concepts = self.concept_structures.read().await;
        
        if concepts.is_empty() {
            return 0.0;
        }
        
        let total_coherence: f32 = concepts.values().map(|c| c.coherence_score).sum();
        total_coherence / concepts.len() as f32
    }

    /// Strengthen co-activated relationships
    async fn strengthen_coactivated_relationships(&self) -> Result<usize> {
        let mut strengthened_count = 0;
        let entity_keys = self.core_graph.get_all_entity_keys();
        
        for entity_key in entity_keys {
            let neighbors = BrainEnhancedKnowledgeGraph::get_neighbors(self, entity_key).await;
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
    async fn create_new_learned_relationships(&self) -> Result<usize> {
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
    async fn optimize_concept_structures(&self) -> Result<usize> {
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

    /// Split concept by connectivity
    async fn split_by_connectivity(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
        // Simple split: divide entities by connectivity patterns
        let entities = concept.get_all_entities();
        let mid_point = entities.len() / 2;
        
        let mut concept1 = ConceptStructure::new();
        let mut concept2 = ConceptStructure::new();
        
        for (i, entity) in entities.iter().enumerate() {
            if i < mid_point {
                concept1.input_entities.push(*entity);
            } else {
                concept2.input_entities.push(*entity);
            }
        }
        
        (concept1, concept2)
    }

    /// Split concept by activation
    async fn split_by_activation(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
        let entities = concept.get_all_entities();
        let mut entity_activations = Vec::new();
        
        for entity in entities {
            let activation = self.get_entity_activation(entity).await;
            entity_activations.push((entity, activation));
        }
        
        // Sort by activation
        entity_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid_point = entity_activations.len() / 2;
        let mut concept1 = ConceptStructure::new();
        let mut concept2 = ConceptStructure::new();
        
        for (i, (entity, _)) in entity_activations.iter().enumerate() {
            if i < mid_point {
                concept1.input_entities.push(*entity);
            } else {
                concept2.input_entities.push(*entity);
            }
        }
        
        (concept1, concept2)
    }

    /// Split concept by type
    async fn split_by_type(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
        let mut concept1 = ConceptStructure::new();
        let mut concept2 = ConceptStructure::new();
        
        // Put inputs and gates in concept1, outputs in concept2
        concept1.input_entities = concept.input_entities.clone();
        concept1.gate_entities = concept.gate_entities.clone();
        concept2.output_entities = concept.output_entities.clone();
        
        (concept1, concept2)
    }
}

/// Entity role in concept
#[derive(Debug, Clone, PartialEq)]
pub enum EntityRole {
    Input,
    Output,
    Gate,
    Processing,
}

/// Split criteria for concept splitting
#[derive(Debug, Clone)]
pub enum SplitCriteria {
    ByConnectivity,
    ByActivation,
    ByType,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub pruned_relationships: usize,
    pub strengthened_relationships: usize,
    pub new_relationships: usize,
    pub optimized_concepts: usize,
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