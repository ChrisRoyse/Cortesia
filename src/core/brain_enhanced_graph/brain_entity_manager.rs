//! Entity management for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::{EntityKey, EntityData, Relationship};
use crate::error::Result;
use std::collections::HashMap;
use std::time::Instant;

impl BrainEnhancedKnowledgeGraph {
    /// Insert brain entity with neural activation
    pub async fn insert_brain_entity(&self, id: u32, data: EntityData) -> Result<EntityKey> {
        let _start_time = Instant::now();
        
        // Insert into core graph
        let entity_key = self.core_graph.insert_entity(id, data.clone())?;
        
        // Set initial activation based on importance
        let initial_activation = self.calculate_initial_activation(&data);
        self.set_entity_activation(entity_key, initial_activation).await;
        
        // Store in SDR if enabled
        if self.config.enable_concept_formation {
            // Store the embedding as SDR
            self.sdr_storage.store_dense_vector(
                entity_key,
                &data.embedding,
                format!("entity_{}", id)
            ).await?;
        }
        
        // Update statistics
        self.update_learning_stats(|stats| {
            stats.entity_count += 1;
            stats.avg_activation = (stats.avg_activation * (stats.entity_count - 1) as f32 + initial_activation) / stats.entity_count as f32;
            stats.max_activation = stats.max_activation.max(initial_activation);
            stats.min_activation = if stats.entity_count == 1 { initial_activation } else { stats.min_activation.min(initial_activation) };
        }).await;
        
        // Trigger concept formation if enabled
        if self.config.enable_concept_formation {
            self.trigger_concept_formation(entity_key).await?;
        }
        
        Ok(entity_key)
    }

    /// Insert logic gate entity
    pub async fn insert_logic_gate(&self, id: u32, gate_type: &str, inputs: Vec<EntityKey>, outputs: Vec<EntityKey>) -> Result<EntityKey> {
        // Create gate properties
        let mut properties = HashMap::new();
        properties.insert("type".to_string(), "logic_gate".to_string());
        properties.insert("gate_type".to_string(), gate_type.to_string());
        properties.insert("input_count".to_string(), inputs.len().to_string());
        properties.insert("output_count".to_string(), outputs.len().to_string());
        
        // Generate gate embedding based on type and connections
        let gate_embedding = self.generate_gate_embedding(gate_type, &inputs, &outputs)?;
        
        let entity_data = EntityData {
            type_id: 0, // Logic gate type
            embedding: gate_embedding,
            properties: serde_json::to_string(&properties).unwrap_or_default(),
        };
        
        // Insert as brain entity
        let gate_key = self.insert_brain_entity(id, entity_data).await?;
        
        // Create relationships to inputs and outputs
        // Note: We need to track entity IDs separately since EntityKey is opaque
        // For now, we'll use the gate's ID as a reference point
        for input_key in inputs.iter() {
            let relationship = Relationship {
                from: *input_key,
                to: gate_key,
                rel_type: 1, // Gate input relationship
                weight: 0.8, // Input connections are strong
            };
            self.insert_brain_relationship(relationship).await?;
        }
        
        for output_key in outputs.iter() {
            let relationship = Relationship {
                from: gate_key,
                to: *output_key,
                rel_type: 2, // Gate output relationship
                weight: 0.9, // Output connections are stronger
            };
            self.insert_brain_relationship(relationship).await?;
        }
        
        Ok(gate_key)
    }

    /// Insert brain relationship with synaptic weight
    pub async fn insert_brain_relationship(&self, relationship: Relationship) -> Result<()> {
        // Insert into core graph
        self.core_graph.insert_relationship(relationship.clone())?;
        
        // Set synaptic weight
        self.set_synaptic_weight(relationship.from, relationship.to, relationship.weight).await;
        
        // Update statistics
        self.update_learning_stats(|stats| {
            stats.relationship_count += 1;
        }).await;
        
        // Trigger Hebbian learning if enabled
        if self.config.enable_hebbian_learning {
            self.apply_hebbian_learning(relationship.from, relationship.to).await?;
        }
        
        Ok(())
    }

    /// Create traditional entity (backward compatibility)
    pub async fn create_traditional_entity(&self, id: u32, properties: HashMap<String, String>) -> Result<EntityKey> {
        // Generate embedding from properties
        let embedding = self.generate_property_embedding(&properties)?;
        
        let entity_data = EntityData {
            type_id: 1, // Traditional entity type
            embedding,
            properties: serde_json::to_string(&properties).unwrap_or_default(),
        };
        
        self.insert_brain_entity(id, entity_data).await
    }

    /// Get entity with activation
    pub async fn get_entity(&self, entity_key: EntityKey) -> Option<(EntityData, f32)> {
        if let Some(data) = self.core_graph.get_entity_data(entity_key) {
            let activation = self.get_entity_activation(entity_key).await;
            Some((data, activation))
        } else {
            None
        }
    }

    /// Get all entities with activations
    pub async fn get_all_entities(&self) -> Vec<(EntityKey, EntityData, f32)> {
        let entity_keys = self.core_graph.get_all_entity_keys();
        let mut entities = Vec::new();
        
        for key in entity_keys {
            if let Some(data) = self.core_graph.get_entity_data(key) {
                let activation = self.get_entity_activation(key).await;
                entities.push((key, data, activation));
            }
        }
        
        entities
    }

    /// Update entity activation
    pub async fn update_entity_activation(&self, entity_key: EntityKey, activation: f32) -> Result<()> {
        if !self.contains_entity(entity_key) {
            return Err(crate::error::GraphError::EntityKeyNotFound { key: entity_key });
        }
        
        let old_activation = self.get_entity_activation(entity_key).await;
        self.set_entity_activation(entity_key, activation).await;
        
        // Update statistics
        self.update_learning_stats(|stats| {
            let entity_count = stats.entity_count as f32;
            stats.avg_activation = (stats.avg_activation * entity_count - old_activation + activation) / entity_count;
            stats.max_activation = stats.max_activation.max(activation);
            // Note: min_activation would need full recalculation for accuracy
        }).await;
        
        Ok(())
    }

    /// Remove entity and its brain-specific data
    pub async fn remove_brain_entity(&self, entity_key: EntityKey) -> Result<bool> {
        if !self.contains_entity(entity_key) {
            return Ok(false);
        }
        
        // Remove from core graph
        let removed = self.core_graph.remove_entity(entity_key)?;
        
        if removed {
            // Remove activation
            let mut activations = self.entity_activations.write().await;
            activations.remove(&entity_key);
            
            // Remove synaptic weights
            let mut weights = self.synaptic_weights.write().await;
            weights.retain(|&(source, target), _| source != entity_key && target != entity_key);
            
            // Remove from concept structures
            let mut concepts = self.concept_structures.write().await;
            for structure in concepts.values_mut() {
                structure.input_entities.retain(|&key| key != entity_key);
                structure.output_entities.retain(|&key| key != entity_key);
                structure.gate_entities.retain(|&key| key != entity_key);
            }
            
            // Update statistics
            self.update_learning_stats(|stats| {
                stats.entity_count = stats.entity_count.saturating_sub(1);
            }).await;
        }
        
        Ok(removed)
    }

    /// Batch insert entities
    pub async fn batch_insert_entities(&self, entities: Vec<(u32, EntityData)>) -> Result<Vec<EntityKey>> {
        let mut entity_keys = Vec::new();
        
        for (id, data) in entities {
            let key = self.insert_brain_entity(id, data).await?;
            entity_keys.push(key);
        }
        
        Ok(entity_keys)
    }

    /// Get entities by type
    pub async fn get_entities_by_type(&self, entity_type: &str) -> Vec<(EntityKey, EntityData, f32)> {
        let all_entities = self.get_all_entities().await;
        
        all_entities
            .into_iter()
            .filter(|(_, data, _)| {
                data.properties.contains(entity_type)
            })
            .collect()
    }

    /// Get highly activated entities
    pub async fn get_highly_activated_entities(&self, threshold: f32) -> Vec<(EntityKey, f32)> {
        let activations = self.entity_activations.read().await;
        
        activations
            .iter()
            .filter(|(_, &activation)| activation >= threshold)
            .map(|(&key, &activation)| (key, activation))
            .collect()
    }

    /// Calculate initial activation for entity
    fn calculate_initial_activation(&self, data: &EntityData) -> f32 {
        // Base activation from embedding magnitude
        let embedding_magnitude: f32 = data.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut activation = (embedding_magnitude / data.embedding.len() as f32).clamp(0.0, 1.0);
        
        // Boost activation based on properties
        if let Some(importance) = data.properties.get("importance") {
            if let Ok(importance_value) = importance.parse::<f32>() {
                activation = (activation + importance_value).clamp(0.0, 1.0);
            }
        }
        
        // Apply configuration
        if activation > self.config.activation_threshold {
            activation
        } else {
            activation * 0.5 // Reduce sub-threshold activations
        }
    }

    /// Generate gate embedding
    fn generate_gate_embedding(&self, gate_type: &str, inputs: &[EntityKey], outputs: &[EntityKey]) -> Result<Vec<f32>> {
        let embedding_dim = self.embedding_dimension();
        let mut embedding = vec![0.0; embedding_dim];
        
        // Base embedding from gate type
        let type_hash = self.hash_string(gate_type);
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((type_hash as u64).wrapping_mul(i as u64 + 1) % 1000) as f32 / 1000.0;
        }
        
        // Incorporate input embeddings
        for input_key in inputs {
            if let Some(input_data) = self.core_graph.get_entity_data(*input_key) {
                for (i, (gate_val, input_val)) in embedding.iter_mut().zip(input_data.embedding.iter()).enumerate() {
                    *gate_val = (*gate_val + input_val * 0.3).clamp(0.0, 1.0);
                }
            }
        }
        
        // Incorporate output embeddings
        for output_key in outputs {
            if let Some(output_data) = self.core_graph.get_entity_data(*output_key) {
                for (gate_val, output_val) in embedding.iter_mut().zip(output_data.embedding.iter()) {
                    *gate_val = (*gate_val + output_val * 0.2).clamp(0.0, 1.0);
                }
            }
        }
        
        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in embedding.iter_mut() {
                *val /= magnitude;
            }
        }
        
        Ok(embedding)
    }

    /// Generate embedding from properties
    fn generate_property_embedding(&self, properties: &HashMap<String, String>) -> Result<Vec<f32>> {
        let embedding_dim = self.embedding_dimension();
        let mut embedding = vec![0.0; embedding_dim];
        
        for (key, value) in properties {
            let combined = format!("{}:{}", key, value);
            let hash = self.hash_string(&combined);
            
            for (i, val) in embedding.iter_mut().enumerate() {
                *val = (*val + ((hash as u64).wrapping_mul(i as u64 + 1) % 1000) as f32 / 1000.0).clamp(0.0, 1.0);
            }
        }
        
        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in embedding.iter_mut() {
                *val /= magnitude;
            }
        }
        
        Ok(embedding)
    }

    /// Hash string to u32
    fn hash_string(&self, s: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Trigger concept formation for entity
    async fn trigger_concept_formation(&self, entity_key: EntityKey) -> Result<()> {
        if !self.config.enable_concept_formation {
            return Ok(());
        }
        
        // Find similar entities
        if let Some(entity_data) = self.core_graph.get_entity_data(entity_key) {
            let similar_entities = self.core_graph.similarity_search(&entity_data.embedding, 5)?;
            
            // If we have enough similar entities, create/update concept
            if similar_entities.len() >= 3 {
                let concept_name = format!("concept_{:?}", entity_key);
                let mut concept_structure = ConceptStructure::new();
                
                // Add similar entities to concept
                for (similar_key, similarity) in similar_entities {
                    if similarity > 0.7 {
                        concept_structure.add_input(similar_key);
                    }
                }
                
                concept_structure.add_output(entity_key);
                concept_structure.concept_activation = self.get_entity_activation(entity_key).await;
                concept_structure.coherence_score = 0.8; // Could be calculated more sophisticatedly
                
                self.store_concept_structure(concept_name, concept_structure).await;
            }
        }
        
        Ok(())
    }

    /// Apply Hebbian learning between entities
    async fn apply_hebbian_learning(&self, source: EntityKey, target: EntityKey) -> Result<()> {
        if !self.config.enable_hebbian_learning {
            return Ok(());
        }
        
        let source_activation = self.get_entity_activation(source).await;
        let target_activation = self.get_entity_activation(target).await;
        
        // Hebbian rule: neurons that fire together, wire together
        let current_weight = self.get_synaptic_weight(source, target).await;
        let learning_signal = source_activation * target_activation * self.config.learning_rate;
        let new_weight = (current_weight + learning_signal).clamp(0.0, 1.0);
        
        self.set_synaptic_weight(source, target, new_weight).await;
        
        // Update learning efficiency
        self.update_learning_stats(|stats| {
            stats.learning_efficiency = (stats.learning_efficiency * 0.95 + learning_signal * 0.05).clamp(0.0, 1.0);
        }).await;
        
        Ok(())
    }

    /// Get entity statistics
    pub async fn get_entity_statistics(&self) -> EntityStatistics {
        let all_entities = self.get_all_entities().await;
        let total_entities = all_entities.len();
        
        if total_entities == 0 {
            return EntityStatistics::default();
        }
        
        let total_activation: f32 = all_entities.iter().map(|(_, _, activation)| activation).sum();
        let avg_activation = total_activation / total_entities as f32;
        
        let max_activation = all_entities.iter().map(|(_, _, activation)| *activation).fold(0.0, f32::max);
        let min_activation = all_entities.iter().map(|(_, _, activation)| *activation).fold(1.0, f32::min);
        
        // Count entities by type
        let mut type_counts = HashMap::new();
        for (_, data, _) in &all_entities {
            // For now, use a simple heuristic to extract type from properties string
            let entity_type = if data.properties.contains("input") { "input" }
                           else if data.properties.contains("output") { "output" }
                           else if data.properties.contains("gate") { "gate" }
                           else { "unknown" };
            *type_counts.entry(entity_type.to_string()).or_insert(0) += 1;
        }
        
        EntityStatistics {
            total_entities,
            avg_activation,
            max_activation,
            min_activation,
            type_distribution: type_counts,
        }
    }
}

/// Entity statistics
#[derive(Debug, Clone)]
pub struct EntityStatistics {
    pub total_entities: usize,
    pub avg_activation: f32,
    pub max_activation: f32,
    pub min_activation: f32,
    pub type_distribution: HashMap<String, usize>,
}

impl Default for EntityStatistics {
    fn default() -> Self {
        Self {
            total_entities: 0,
            avg_activation: 0.0,
            max_activation: 0.0,
            min_activation: 0.0,
            type_distribution: HashMap::new(),
        }
    }
}

impl EntityStatistics {
    /// Get activation range
    pub fn activation_range(&self) -> f32 {
        self.max_activation - self.min_activation
    }
    
    /// Get most common entity type
    pub fn most_common_type(&self) -> Option<(&String, usize)> {
        self.type_distribution.iter().max_by_key(|(_, count)| *count)
    }
    
    /// Check if activations are well-distributed
    pub fn is_well_distributed(&self) -> bool {
        self.activation_range() > 0.3 && self.avg_activation > 0.2 && self.avg_activation < 0.8
    }
}