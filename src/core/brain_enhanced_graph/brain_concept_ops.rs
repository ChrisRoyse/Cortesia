//! Concept operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::core::sdr_types::{SDRQuery, SDR};
use crate::error::Result;
use std::collections::HashSet;

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
            let sdr_config = crate::core::sdr_types::SDRConfig::default();
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
            for (pattern_id, _similarity_score) in sdr_results {
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

    /// Calculate concept activation
    pub(crate) async fn calculate_concept_activation(&self, concept: &ConceptStructure) -> f32 {
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
    pub(crate) async fn calculate_concept_coherence(&self, concept: &ConceptStructure) -> f32 {
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

    /// Calculate concept similarity
    pub(crate) async fn calculate_concept_similarity(&self, _concept1: &ConceptStructure, _concept2: &ConceptStructure) -> f32 {
        // Use simple placeholder embeddings for now
        let embedding1 = vec![0.0; 384];
        let embedding2 = vec![0.0; 384];
        
        self.calculate_embedding_similarity(&embedding1, &embedding2)
    }

    /// Split concept by connectivity
    pub(crate) async fn split_by_connectivity(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
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
    pub(crate) async fn split_by_activation(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
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
    pub(crate) async fn split_by_type(&self, concept: &ConceptStructure) -> (ConceptStructure, ConceptStructure) {
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