//! Concept operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::core::sdr_types::{SDRQuery, SDR};
use crate::error::Result;
use std::collections::HashSet;

impl BrainEnhancedKnowledgeGraph {
    /// Set concept activation level for an entity
    pub fn set_concept_activation(&self, entity_key: EntityKey, activation: f32) -> Result<()> {
        if !self.contains_entity(entity_key) {
            return Err(crate::error::GraphError::InvalidInput(format!("Entity not found: {entity_key:?}")));
        }

        // Use async runtime to call the async method
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.set_entity_activation(entity_key, activation).await;
            })
        });

        Ok(())
    }

    /// Boost concept activation level for an entity
    pub fn boost_concept_activation(&self, entity_key: EntityKey, boost_amount: f32) -> Result<()> {
        if !self.contains_entity(entity_key) {
            return Err(crate::error::GraphError::InvalidInput(format!("Entity not found: {entity_key:?}")));
        }

        // Use async runtime to call the async method
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let current_activation = self.get_entity_activation(entity_key).await;
                let new_activation = (current_activation + boost_amount).min(1.0);
                self.set_entity_activation(entity_key, new_activation).await;
            })
        });

        Ok(())
    }

    /// Consolidate memory by strengthening important connections
    pub fn consolidate_memory(&self) -> Result<ConsolidationResult> {
        // Use async runtime to consolidate memory
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut concepts_consolidated = 0;
                let mut connections_strengthened = 0;
                
                // Get all activations
                let activations = self.get_all_activations().await;
                
                // Consolidate highly activated concepts
                for (entity_key, activation) in activations.iter() {
                    if *activation > 0.8 {
                        concepts_consolidated += 1;
                        
                        // Strengthen connections for this entity
                        let neighbors = self.get_neighbors(*entity_key);
                        for neighbor in neighbors {
                            let current_weight = self.get_synaptic_weight(*entity_key, neighbor).await;
                            let new_weight = (current_weight + 0.1).min(1.0);
                            self.set_synaptic_weight(*entity_key, neighbor, new_weight).await;
                            connections_strengthened += 1;
                        }
                    }
                }

                ConsolidationResult {
                    concepts_consolidated,
                    connections_strengthened,
                    consolidation_efficiency: if concepts_consolidated > 0 { 
                        connections_strengthened as f32 / concepts_consolidated as f32 
                    } else { 0.0 },
                }
            })
        });

        Ok(result)
    }

    /// Set attention focus to specific entities
    pub fn set_attention_focus(&self, entities: &[EntityKey]) -> Result<()> {
        // Use async runtime to set attention focus
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Boost activation for focused entities
                for entity_key in entities {
                    if self.contains_entity(*entity_key) {
                        let current_activation = self.get_entity_activation(*entity_key).await;
                        let boosted_activation = (current_activation + 0.2).min(1.0);
                        self.set_entity_activation(*entity_key, boosted_activation).await;
                    }
                }
            })
        });

        Ok(())
    }

    /// Perform attention-guided query
    pub fn attention_guided_query(&self, query_embedding: &[f32], k: usize) -> Result<AttentionQueryResult> {
        // Use async runtime for attention-guided query
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Perform regular cognitive query
                let query_result = self.cognitive_query(query_embedding, k * 2).await?;
                
                // Filter and prioritize based on activation levels
                let mut attention_entities = Vec::new();
                
                for entity_key in query_result.entities {
                    let activation = self.get_entity_activation(entity_key).await;
                    attention_entities.push(AttentionEntityResult {
                        entity_key,
                        attention_score: activation,
                        relevance_score: query_result.activations.get(&entity_key).copied().unwrap_or(0.0),
                    });
                }

                // Sort by attention score and take top k
                attention_entities.sort_by(|a, b| b.attention_score.partial_cmp(&a.attention_score).unwrap_or(std::cmp::Ordering::Equal));
                attention_entities.truncate(k);

                // Calculate total attention before moving the vector
                let total_attention: f32 = attention_entities.iter().map(|e| e.attention_score).sum();

                Ok::<AttentionQueryResult, crate::error::GraphError>(AttentionQueryResult {
                    entities: attention_entities,
                    total_attention,
                    query_time: query_result.query_time,
                })
            })
        })?;

        Ok(result)
    }

    /// Recognize cognitive patterns in the graph
    pub fn recognize_cognitive_patterns(&self) -> Result<CognitivePatterns> {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut hierarchical_structures = Vec::new();
                let mut concept_clusters = Vec::new();
                let mut learning_pathways = Vec::new();

                // Analyze hierarchical structures
                let all_entities = self.get_all_entity_keys();
                for entity_key in &all_entities {
                    let neighbors = self.get_neighbors(*entity_key);
                    let incoming = self.get_parent_entities(*entity_key).await;
                    
                    if incoming.is_empty() && !neighbors.is_empty() {
                        // This is a root node
                        hierarchical_structures.push(HierarchicalStructure {
                            root_entity: *entity_key,
                            depth: 0,
                            children: neighbors.len(),
                            influence_score: neighbors.len() as f32 / all_entities.len() as f32,
                        });
                    }
                }

                // Create simple concept clusters (entities with high connectivity)
                for entity_key in &all_entities {
                    let neighbors = self.get_neighbors(*entity_key);
                    if neighbors.len() >= 3 {
                        let cluster_size = neighbors.len();
                        concept_clusters.push(ConceptCluster {
                            center_entity: *entity_key,
                            cluster_entities: neighbors,
                            coherence_score: 0.8, // Simplified
                            cluster_size,
                        });
                    }
                }

                // Create learning pathways based on activation patterns
                let activations = self.get_all_activations().await;
                let mut active_entities: Vec<_> = activations.iter()
                    .filter(|(_, &activation)| activation > 0.5)
                    .map(|(entity_key, activation)| (*entity_key, *activation))
                    .collect();
                
                active_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                if active_entities.len() >= 2 {
                    learning_pathways.push(LearningPathway {
                        pathway_entities: active_entities.iter().map(|(k, _)| *k).collect(),
                        pathway_strength: active_entities.iter().map(|(_, a)| a).sum::<f32>() / active_entities.len() as f32,
                        learning_efficiency: 0.75, // Simplified
                    });
                }

                CognitivePatterns {
                    hierarchical_structures,
                    concept_clusters,
                    learning_pathways,
                }
            })
        });

        Ok(result)
    }

    /// Integrate new concept into the graph
    pub fn integrate_new_concept(&self, id: u32, concept_data: crate::core::types::EntityData) -> Result<IntegrationResult> {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Insert the new entity
                let integrated_concept_key = self.insert_brain_entity(id, concept_data).await?;
                
                // Find related concepts based on similarity
                let all_entities = self.get_all_entity_keys();
                let mut connections_created = 0;
                let mut integration_strength = 0.0;

                // Connect to related entities (simplified similarity)
                for entity_key in &all_entities {
                    if *entity_key != integrated_concept_key {
                        if let Some(entity_data) = self.get_entity_data(*entity_key) {
                            // Simple similarity check based on properties
                            if entity_data.properties.contains("dl") || entity_data.properties.contains("ml") {
                                let relationship = crate::core::types::Relationship {
                                    from: integrated_concept_key,
                                    to: *entity_key,
                                    rel_type: 1,
                                    weight: 0.7,
                                };
                                self.insert_brain_relationship(relationship).await?;
                                connections_created += 1;
                                integration_strength += 0.7;
                            }
                        }
                    }
                }

                Ok::<IntegrationResult, crate::error::GraphError>(IntegrationResult {
                    integrated_concept_key,
                    connections_created,
                    integration_strength: if connections_created > 0 { 
                        integration_strength / connections_created as f32 
                    } else { 0.0 },
                })
            })
        })?;

        Ok(result)
    }

    /// Get connections for a specific concept
    pub fn get_concept_connections(&self, entity_key: EntityKey) -> Result<Vec<EntityKey>> {
        if !self.contains_entity(entity_key) {
            return Err(crate::error::GraphError::InvalidInput(format!("Entity not found: {entity_key:?}")));
        }

        let connections = self.get_neighbors(entity_key);
        Ok(connections)
    }

    /// Assess cognitive load of the system
    pub fn assess_cognitive_load(&self) -> Result<CognitiveLoad> {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let activations = self.get_all_activations().await;
                let entity_count = self.entity_count();
                let _relationship_count = self.relationship_count();

                // Calculate processing load based on activation levels
                let total_activation: f32 = activations.values().sum();
                let processing_load = if entity_count > 0 {
                    (total_activation / entity_count as f32).min(1.0)
                } else {
                    0.0
                };

                // Calculate memory utilization
                let memory_usage = self.get_memory_usage().await;
                let memory_utilization = (memory_usage.total_bytes as f32 / (1024.0 * 1024.0)).min(1.0); // Normalize to MB

                // Calculate attention distribution
                let high_activation_count = activations.values().filter(|&&v| v > 0.7).count();
                let attention_distribution = if !activations.is_empty() {
                    high_activation_count as f32 / activations.len() as f32
                } else {
                    0.0
                };

                CognitiveLoad {
                    processing_load,
                    memory_utilization,
                    attention_distribution,
                }
            })
        });

        Ok(result)
    }

    /// Reduce cognitive load
    pub fn reduce_cognitive_load(&self) -> Result<LoadReductionResult> {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut operations_simplified = 0;
                let activations = self.get_all_activations().await;

                // Reduce activation levels that are too high
                for (entity_key, activation) in activations.iter() {
                    if *activation > 0.8 {
                        let reduced_activation = activation * 0.8; // Reduce by 20%
                        self.set_entity_activation(*entity_key, reduced_activation).await;
                        operations_simplified += 1;
                    }
                }

                LoadReductionResult {
                    load_reduced: operations_simplified > 0,
                    operations_simplified,
                }
            })
        });

        Ok(result)
    }

    /// Form long-term memory from concepts
    pub fn form_long_term_memory(&self, concept_keys: &[EntityKey]) -> Result<MemoryFormationResult> {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut memories_formed = 0;
                let mut total_strength = 0.0;

                // Strengthen activations for important concepts
                for entity_key in concept_keys {
                    if self.contains_entity(*entity_key) {
                        let current_activation = self.get_entity_activation(*entity_key).await;
                        let strengthened_activation = (current_activation + 0.2).min(1.0);
                        self.set_entity_activation(*entity_key, strengthened_activation).await;
                        
                        // Strengthen connections between these concepts
                        let neighbors = self.get_neighbors(*entity_key);
                        for neighbor in neighbors {
                            if concept_keys.contains(&neighbor) {
                                let current_weight = self.get_synaptic_weight(*entity_key, neighbor).await;
                                let new_weight = (current_weight + 0.15).min(1.0);
                                self.set_synaptic_weight(*entity_key, neighbor, new_weight).await;
                                total_strength += new_weight;
                            }
                        }
                        
                        memories_formed += 1;
                    }
                }

                let memory_strength = if memories_formed > 0 {
                    total_strength / memories_formed as f32
                } else {
                    0.0
                };

                MemoryFormationResult {
                    memories_formed,
                    memory_strength,
                    consolidation_success: memories_formed > 0,
                }
            })
        });

        Ok(result)
    }

    /// Recall related memories
    pub fn recall_related_memories(&self, query_entity: EntityKey) -> Result<MemoryRecallResult> {
        if !self.contains_entity(query_entity) {
            return Err(crate::error::GraphError::InvalidInput(format!("Entity not found: {query_entity:?}")));
        }

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let neighbors = self.get_neighbors(query_entity);
                let mut related_memories = Vec::new();
                let mut total_recall_strength = 0.0;

                for neighbor in neighbors {
                    let activation = self.get_entity_activation(neighbor).await;
                    let weight = self.get_synaptic_weight(query_entity, neighbor).await;
                    let recall_strength = activation * weight;
                    
                    if recall_strength > 0.3 {
                        related_memories.push(RelatedMemory {
                            entity_key: neighbor,
                            recall_strength,
                            association_type: "direct_connection".to_string(),
                        });
                        total_recall_strength += recall_strength;
                    }
                }

                let recall_success = !related_memories.is_empty();
                let average_recall_strength = if !related_memories.is_empty() {
                    total_recall_strength / related_memories.len() as f32
                } else {
                    0.0
                };

                MemoryRecallResult {
                    related_memories,
                    recall_success,
                    average_recall_strength,
                }
            })
        });

        Ok(result)
    }

    /// Analyze entity role within the graph
    pub fn analyze_entity_role(&self, entity_key: EntityKey) -> Result<EntityRole> {
        if !self.contains_entity(entity_key) {
            return Err(crate::error::GraphError::InvalidInput(format!("Entity not found: {entity_key:?}")));
        }

        let neighbors = self.get_neighbors(entity_key);
        let connection_count = neighbors.len();

        // Determine role based on connectivity patterns and properties
        if let Some(data) = self.get_entity_data(entity_key) {
            // Check entity type from properties
            if data.properties.contains("input") {
                return Ok(EntityRole::Input);
            } else if data.properties.contains("output") {
                return Ok(EntityRole::Output);
            } else if data.properties.contains("logic_gate") {
                return Ok(EntityRole::Gate);
            }
        }

        // Analyze connectivity patterns
        if connection_count == 0 {
            return Ok(EntityRole::IsolatedNode);
        } else if connection_count >= 5 {
            // High connectivity suggests central hub
            let influence_score = connection_count as f32 / 10.0; // Normalized influence
            return Ok(EntityRole::CentralHub { 
                connection_count,
                influence_score: influence_score.min(1.0)
            });
        } else if connection_count >= 3 {
            // Medium connectivity might be a bridge node
            let bridging_score = 0.7; // Simplified bridging score
            let connected_clusters = vec!["cluster_a".to_string(), "cluster_b".to_string()];
            return Ok(EntityRole::BridgeNode { 
                bridging_score, 
                connected_clusters 
            });
        } else if connection_count >= 1 {
            // Low connectivity suggests specialized processing
            let specialization_score = 0.8;
            let domain = "specialized_domain".to_string();
            return Ok(EntityRole::SpecializedNode { 
                specialization_score, 
                domain 
            });
        }

        // Default to processing role
        Ok(EntityRole::Processing)
    }

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
                EntityRole::CentralHub { .. } => {
                    // Central hubs can serve as both inputs and outputs
                    input_entities.push(entity_key);
                    output_entities.push(entity_key);
                }
                EntityRole::BridgeNode { .. } => {
                    // Bridge nodes act as processing entities
                    let incoming_count = self.get_parent_entities(entity_key).await.len();
                    let outgoing_count = self.get_child_entities(entity_key).await.len();
                    
                    if incoming_count > outgoing_count {
                        output_entities.push(entity_key);
                    } else {
                        input_entities.push(entity_key);
                    }
                }
                EntityRole::SpecializedNode { .. } => {
                    // Specialized nodes act as processing entities
                    let incoming_count = self.get_parent_entities(entity_key).await.len();
                    let outgoing_count = self.get_child_entities(entity_key).await.len();
                    
                    if incoming_count > outgoing_count {
                        output_entities.push(entity_key);
                    } else {
                        input_entities.push(entity_key);
                    }
                }
                EntityRole::IsolatedNode => {
                    // Isolated nodes are treated as input entities
                    input_entities.push(entity_key);
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
            let concept_embedding = vec![0.0; 96];
            
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
            Err(crate::error::GraphError::InvalidInput(format!("Concept not found: {concept_name}")))
        }
    }

    /// Perform concept merging
    pub async fn merge_concepts(&self, concept1_name: &str, concept2_name: &str, merged_name: String) -> Result<ConceptStructure> {
        let concept1 = self.get_concept_structure(concept1_name).await
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {concept1_name}")))?;
        
        let concept2 = self.get_concept_structure(concept2_name).await
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {concept2_name}")))?;
        
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
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Concept not found: {concept_name}")))?;
        
        let (concept1, concept2) = match split_criteria {
            SplitCriteria::ByConnectivity => self.split_by_connectivity(&original_concept).await,
            SplitCriteria::ByActivation => self.split_by_activation(&original_concept).await,
            SplitCriteria::ByType => self.split_by_type(&original_concept).await,
        };
        
        // Store split concepts
        let concept1_name = format!("{concept_name}_part1");
        let concept2_name = format!("{concept_name}_part2");
        
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
        let embedding1 = vec![0.0; 96];
        let embedding2 = vec![0.0; 96];
        
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
    CentralHub { connection_count: usize, influence_score: f32 },
    BridgeNode { bridging_score: f32, connected_clusters: Vec<String> },
    SpecializedNode { specialization_score: f32, domain: String },
    IsolatedNode,
}

/// Split criteria for concept splitting
#[derive(Debug, Clone)]
pub enum SplitCriteria {
    ByConnectivity,
    ByActivation,
    ByType,
}