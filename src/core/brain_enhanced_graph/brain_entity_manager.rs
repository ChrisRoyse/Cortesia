//! Entity management for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::{EntityKey, EntityData, Relationship};
use crate::error::Result;
use std::collections::HashMap;
use std::time::Instant;

impl BrainEnhancedKnowledgeGraph {
    /// Batch update activations for multiple entities
    pub async fn batch_update_activations(&self, activation_updates: &[(EntityKey, f32)]) {
        for (entity_key, activation) in activation_updates {
            self.set_entity_activation(*entity_key, *activation).await;
        }
    }

    /// Count entities above activation threshold
    pub async fn count_entities_above_threshold(&self, threshold: f32) -> usize {
        let activations = self.get_all_activations().await;
        activations.values().filter(|&&activation| activation > threshold).count()
    }

    /// Get entities in activation range
    pub async fn get_entities_in_activation_range(&self, min_activation: f32, max_activation: f32) -> Vec<(EntityKey, f32)> {
        let activations = self.get_all_activations().await;
        activations
            .into_iter()
            .filter(|(_, activation)| *activation >= min_activation && *activation <= max_activation)
            .collect()
    }

    /// Get top k entities by activation
    pub async fn get_top_k_entities(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let activations = self.get_all_activations().await;
        let mut entities: Vec<_> = activations.into_iter().collect();
        entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entities.truncate(k);
        entities
    }

    /// Get concept statistics
    pub async fn get_concept_statistics(&self) -> ConceptStatistics {
        let activations = self.get_all_activations().await;
        let entity_count = self.entity_count();
        let relationship_count = self.relationship_count();
        
        let total_activation: f32 = activations.values().sum();
        let average_activation = if entity_count > 0 {
            total_activation / entity_count as f32
        } else {
            0.0
        };

        let max_activation = activations.values().cloned().fold(0.0f32, f32::max);
        let min_activation = activations.values().cloned().fold(1.0f32, f32::min);
        
        let active_concepts = activations.values().filter(|&&v| v > 0.1).count();
        
        ConceptStatistics {
            total_concepts: entity_count,
            active_concepts,
            average_activation,
            max_activation,
            min_activation,
            connectivity_density: if entity_count > 0 {
                relationship_count as f32 / entity_count as f32
            } else {
                0.0
            },
        }
    }

    /// Insert brain entity with activation
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
                format!("entity_{id}")
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


    /// Add entity with string parameters (test compatibility method)
    pub async fn add_entity_with_id(&self, id: &str, description: &str) -> Result<EntityKey> {
        // Generate a numeric ID from the string identifier
        let numeric_id = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            (hasher.finish() as u32) % 1000000 + 1 // Ensure non-zero
        };
        
        // Create entity data with the description as properties
        let entity_data = EntityData {
            type_id: 1, // Default type
            embedding: vec![0.1; self.embedding_dimension()],
            properties: description.to_string(),
        };
        
        self.insert_brain_entity(numeric_id, entity_data).await
    }
    
    /// Add entity with EntityData (original method)
    pub async fn add_entity_data(&self, data: EntityData) -> Result<EntityKey> {
        // Generate a unique ID for the entity
        let id = {
            let core_graph = &self.core_graph;
            (core_graph.entity_count() as u32) + 1
        };
        self.insert_brain_entity(id, data).await
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

    /// Add entity with EntityData directly (used by most cognitive tests)
    pub async fn add_entity(&self, data: EntityData) -> Result<EntityKey> {
        self.add_entity_data(data).await
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


    /// Get entity by description (searches properties for matching description)
    pub async fn get_entity_by_description(&self, description: &str) -> Result<Option<crate::core::types::Entity>> {
        let all_entities = self.get_all_entities().await;
        
        for (key, data, activation) in all_entities {
            // First check if the properties string directly matches the description
            if data.properties == description {
                // Get the entity ID from the reverse mapping
                let id = self.get_entity_id_for_key(key).unwrap_or(0);
                return Ok(Some(crate::core::types::Entity {
                    id,
                    key,
                    data,
                    activation,
                }));
            }
            
            // Also try parsing as JSON and check for matches
            if let Ok(props) = serde_json::from_str::<serde_json::Value>(&data.properties) {
                // Check if any property value matches the description
                if let Some(obj) = props.as_object() {
                    for (_, value) in obj {
                        if let Some(value_str) = value.as_str() {
                            if value_str == description {
                                // Get the entity ID from the reverse mapping
                                let id = self.get_entity_id_for_key(key).unwrap_or(0);
                                return Ok(Some(crate::core::types::Entity {
                                    id,
                                    key,
                                    data,
                                    activation,
                                }));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(None)
    }

    /// Get entity ID for a given EntityKey
    fn get_entity_id_for_key(&self, entity_key: EntityKey) -> Option<u32> {
        // This is a simplified implementation - in practice you'd want a reverse mapping
        // For now, we'll use a simple approach based on the entity key's internal representation
        use slotmap::{Key, KeyData};
        let key_data: KeyData = entity_key.data();
        Some(key_data.as_ffi() as u32)
    }

    /// Calculate initial activation for entity
    fn calculate_initial_activation(&self, data: &EntityData) -> f32 {
        // Base activation from embedding magnitude
        let embedding_magnitude: f32 = data.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut activation = (embedding_magnitude / data.embedding.len() as f32).clamp(0.0, 1.0);
        
        // Boost activation based on properties
        if let Ok(props) = serde_json::from_str::<serde_json::Value>(&data.properties) {
            if let Some(importance) = props.get("importance") {
                if let Some(importance_value) = importance.as_f64() {
                    activation = (activation + importance_value as f32).clamp(0.0, 1.0);
                }
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
                for (_i, (gate_val, input_val)) in embedding.iter_mut().zip(input_data.embedding.iter()).enumerate() {
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
            let combined = format!("{key}:{value}");
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
        
        // Hebbian rule: entities that activate together, strengthen together
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
        self.type_distribution.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, &count)| (type_name, count))
    }
    
    /// Check if activations are well-distributed
    pub fn is_well_distributed(&self) -> bool {
        self.activation_range() > 0.3 && self.avg_activation > 0.2 && self.avg_activation < 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use std::collections::HashMap;
    use tokio;

    /// Helper function to create a test brain graph
    async fn create_test_brain_graph() -> BrainEnhancedKnowledgeGraph {
        BrainEnhancedKnowledgeGraph::new_for_test().unwrap()
    }

    /// Helper function to create test entity data
    fn create_test_entity_data(type_id: u16, properties: Option<HashMap<String, String>>) -> EntityData {
        let mut embedding = vec![0.0; 96]; // 96-dimensional embedding
        embedding[0] = 0.1;
        embedding[1] = 0.2;
        embedding[2] = 0.3;
        embedding[3] = 0.4;
        embedding[4] = 0.5;
        let props = if let Some(p) = properties {
            serde_json::to_string(&p).unwrap_or_default()
        } else {
            "{}".to_string()
        };
        
        EntityData {
            type_id,
            embedding,
            properties: props,
        }
    }

    /// Helper function to create entity data with no properties
    fn create_empty_entity_data() -> EntityData {
        let mut embedding = vec![0.0; 96]; // 96-dimensional embedding
        embedding[0] = 0.1;
        embedding[1] = 0.2;
        embedding[2] = 0.3;
        
        EntityData {
            type_id: 1,
            embedding,
            properties: String::new(),
        }
    }

    /// Helper function to create entity data with importance property
    fn create_entity_with_importance(importance: f64) -> EntityData {
        let mut props = HashMap::new();
        props.insert("importance".to_string(), importance.to_string());
        let properties_json = serde_json::json!({"importance": importance}).to_string();
        
        let mut embedding = vec![0.0; 96]; // 96-dimensional embedding
        embedding[0] = 0.5;
        embedding[1] = 0.5;
        embedding[2] = 0.5;
        
        EntityData {
            type_id: 1,
            embedding,
            properties: properties_json,
        }
    }

    mod insert_brain_entity_tests {
        use super::*;

        #[tokio::test]
        async fn test_insert_brain_entity_basic() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            
            let result = graph.insert_brain_entity(1, entity_data.clone()).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            
            // Verify entity was inserted
            assert!(graph.contains_entity(entity_key));
            
            // Verify activation was set
            let activation = graph.get_entity_activation(entity_key).await;
            assert!(activation > 0.0);
            assert!(activation <= 1.0);
        }

        #[tokio::test]
        async fn test_insert_brain_entity_with_properties() {
            let graph = create_test_brain_graph().await;
            let mut properties = HashMap::new();
            properties.insert("name".to_string(), "test_entity".to_string());
            properties.insert("type".to_string(), "concept".to_string());
            
            let entity_data = create_test_entity_data(2, Some(properties));
            
            let result = graph.insert_brain_entity(2, entity_data).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            
            // Verify entity exists and has data
            if let Some(stored_data) = graph.get_entity_data(entity_key) {
                assert_eq!(stored_data.type_id, 2);
                assert!(!stored_data.properties.is_empty());
            } else {
                panic!("Entity data should exist");
            }
        }

        #[tokio::test]
        async fn test_insert_brain_entity_with_no_properties() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_empty_entity_data();
            
            let result = graph.insert_brain_entity(3, entity_data).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            
            // Verify entity was inserted even with empty properties
            assert!(graph.contains_entity(entity_key));
            
            // Verify entity data
            if let Some(stored_data) = graph.get_entity_data(entity_key) {
                assert_eq!(stored_data.type_id, 1);
                assert!(stored_data.properties.is_empty());
            } else {
                panic!("Entity data should exist");
            }
        }

        #[tokio::test]
        async fn test_insert_brain_entity_with_importance() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_entity_with_importance(0.8);
            
            let result = graph.insert_brain_entity(4, entity_data).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            
            // Verify higher activation due to importance
            let activation = graph.get_entity_activation(entity_key).await;
            assert!(activation > 0.5); // Should be boosted by importance
        }

        #[tokio::test]
        async fn test_insert_brain_entity_updates_statistics() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            
            // Get initial stats
            let initial_stats = graph.get_learning_stats().await;
            let initial_count = initial_stats.entity_count;
            
            // Insert entity
            let result = graph.insert_brain_entity(5, entity_data).await;
            assert!(result.is_ok());
            
            // Verify stats were updated
            let updated_stats = graph.get_learning_stats().await;
            assert_eq!(updated_stats.entity_count, initial_count + 1);
            assert!(updated_stats.avg_activation >= 0.0);
            assert!(updated_stats.max_activation >= updated_stats.avg_activation);
        }

        #[tokio::test]
        async fn test_insert_multiple_brain_entities() {
            let graph = create_test_brain_graph().await;
            
            let entity1 = create_test_entity_data(1, None);
            let entity2 = create_test_entity_data(2, None);
            
            let key1 = graph.insert_brain_entity(6, entity1).await.unwrap();
            let key2 = graph.insert_brain_entity(7, entity2).await.unwrap();
            
            // Verify both entities exist and are different
            assert!(graph.contains_entity(key1));
            assert!(graph.contains_entity(key2));
            assert_ne!(key1, key2);
            
            // Verify activations are set
            assert!(graph.get_entity_activation(key1).await > 0.0);
            assert!(graph.get_entity_activation(key2).await > 0.0);
        }
    }

    mod get_entity_tests {
        use super::*;

        #[tokio::test]
        async fn test_get_entity_exists() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(10, entity_data.clone()).await.unwrap();
            
            let result = graph.get_entity(entity_key).await;
            assert!(result.is_some());
            
            let (retrieved_data, activation) = result.unwrap();
            assert_eq!(retrieved_data.type_id, entity_data.type_id);
            assert_eq!(retrieved_data.embedding, entity_data.embedding);
            assert!(activation >= 0.0 && activation <= 1.0);
        }

        #[tokio::test]
        async fn test_get_entity_nonexistent() {
            let graph = create_test_brain_graph().await;
            
            // Use a default EntityKey that doesn't exist
            let fake_key = EntityKey::default();
            let result = graph.get_entity(fake_key).await;
            assert!(result.is_none());
        }

        #[tokio::test]
        async fn test_get_entity_with_activation() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(11, entity_data).await.unwrap();
            
            // Set a specific activation
            graph.set_entity_activation(entity_key, 0.75).await;
            
            let result = graph.get_entity(entity_key).await;
            assert!(result.is_some());
            
            let (_, activation) = result.unwrap();
            assert!((activation - 0.75).abs() < f32::EPSILON);
        }
    }

    mod insert_logic_gate_tests {
        use super::*;

        #[tokio::test]
        async fn test_insert_logic_gate_and() {
            let graph = create_test_brain_graph().await;
            
            // Create input entities
            let input1_data = create_test_entity_data(1, None);
            let input2_data = create_test_entity_data(1, None);
            let output_data = create_test_entity_data(1, None);
            
            let input1_key = graph.insert_brain_entity(20, input1_data).await.unwrap();
            let input2_key = graph.insert_brain_entity(21, input2_data).await.unwrap();
            let output_key = graph.insert_brain_entity(22, output_data).await.unwrap();
            
            // Insert AND gate
            let gate_result = graph.insert_logic_gate(
                23, 
                "AND", 
                vec![input1_key, input2_key], 
                vec![output_key]
            ).await;
            
            assert!(gate_result.is_ok());
            let gate_key = gate_result.unwrap();
            
            // Verify gate entity exists
            assert!(graph.contains_entity(gate_key));
            
            // Verify gate data
            if let Some(gate_data) = graph.get_entity_data(gate_key) {
                assert_eq!(gate_data.type_id, 0); // Logic gate type
                assert!(gate_data.properties.contains("logic_gate"));
                assert!(gate_data.properties.contains("AND"));
            }
        }

        #[tokio::test]
        async fn test_insert_logic_gate_or() {
            let graph = create_test_brain_graph().await;
            
            let input_data = create_test_entity_data(1, None);
            let output_data = create_test_entity_data(1, None);
            
            let input_key = graph.insert_brain_entity(24, input_data).await.unwrap();
            let output_key = graph.insert_brain_entity(25, output_data).await.unwrap();
            
            let gate_result = graph.insert_logic_gate(
                26, 
                "OR", 
                vec![input_key], 
                vec![output_key]
            ).await;
            
            assert!(gate_result.is_ok());
            let gate_key = gate_result.unwrap();
            
            // Verify gate properties
            if let Some(gate_data) = graph.get_entity_data(gate_key) {
                assert!(gate_data.properties.contains("OR"));
                assert!(gate_data.properties.contains("input_count"));
                assert!(gate_data.properties.contains("output_count"));
            }
        }

        #[tokio::test]
        async fn test_insert_logic_gate_not() {
            let graph = create_test_brain_graph().await;
            
            let input_data = create_test_entity_data(1, None);
            let output_data = create_test_entity_data(1, None);
            
            let input_key = graph.insert_brain_entity(27, input_data).await.unwrap();
            let output_key = graph.insert_brain_entity(28, output_data).await.unwrap();
            
            let gate_result = graph.insert_logic_gate(
                29, 
                "NOT", 
                vec![input_key], 
                vec![output_key]
            ).await;
            
            assert!(gate_result.is_ok());
            let gate_key = gate_result.unwrap();
            
            // Verify activation is set
            let activation = graph.get_entity_activation(gate_key).await;
            assert!(activation > 0.0);
        }

        #[tokio::test]
        async fn test_insert_logic_gate_no_inputs() {
            let graph = create_test_brain_graph().await;
            
            let output_data = create_test_entity_data(1, None);
            let output_key = graph.insert_brain_entity(30, output_data).await.unwrap();
            
            let gate_result = graph.insert_logic_gate(
                31, 
                "CONSTANT", 
                vec![], // No inputs
                vec![output_key]
            ).await;
            
            assert!(gate_result.is_ok());
            let gate_key = gate_result.unwrap();
            
            // Verify gate was created even with no inputs
            assert!(graph.contains_entity(gate_key));
        }

        #[tokio::test]
        async fn test_insert_logic_gate_no_outputs() {
            let graph = create_test_brain_graph().await;
            
            let input_data = create_test_entity_data(1, None);
            let input_key = graph.insert_brain_entity(32, input_data).await.unwrap();
            
            let gate_result = graph.insert_logic_gate(
                33, 
                "SINK", 
                vec![input_key], 
                vec![] // No outputs
            ).await;
            
            assert!(gate_result.is_ok());
            let gate_key = gate_result.unwrap();
            
            // Verify gate was created even with no outputs
            assert!(graph.contains_entity(gate_key));
        }
    }

    mod update_entity_activation_tests {
        use super::*;

        #[tokio::test]
        async fn test_update_entity_activation_valid() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(40, entity_data).await.unwrap();
            
            // Update activation
            let result = graph.update_entity_activation(entity_key, 0.8).await;
            assert!(result.is_ok());
            
            // Verify activation was updated
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 0.8).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_update_entity_activation_nonexistent() {
            let graph = create_test_brain_graph().await;
            
            let fake_key = EntityKey::default();
            let result = graph.update_entity_activation(fake_key, 0.5).await;
            
            assert!(result.is_err());
            match result.unwrap_err() {
                crate::error::GraphError::EntityKeyNotFound { key } => {
                    assert_eq!(key, fake_key);
                }
                _ => panic!("Expected EntityKeyNotFound error"),
            }
        }

        #[tokio::test]
        async fn test_update_entity_activation_invalid_levels() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(41, entity_data).await.unwrap();
            
            // Test negative activation (should be clamped)
            let result = graph.update_entity_activation(entity_key, -0.5).await;
            assert!(result.is_ok());
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 0.0).abs() < f32::EPSILON);
            
            // Test activation > 1.0 (should be clamped)
            let result = graph.update_entity_activation(entity_key, 1.5).await;
            assert!(result.is_ok());
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 1.0).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_update_entity_activation_updates_stats() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(42, entity_data).await.unwrap();
            
            // Get initial stats
            let initial_stats = graph.get_learning_stats().await;
            
            // Update activation
            let result = graph.update_entity_activation(entity_key, 0.9).await;
            assert!(result.is_ok());
            
            // Verify stats may have been updated
            let updated_stats = graph.get_learning_stats().await;
            // Note: The exact behavior depends on how statistics are calculated
            assert!(updated_stats.max_activation >= 0.9);
        }

        #[tokio::test]
        async fn test_update_entity_activation_boundary_values() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(43, entity_data).await.unwrap();
            
            // Test 0.0
            let result = graph.update_entity_activation(entity_key, 0.0).await;
            assert!(result.is_ok());
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 0.0).abs() < f32::EPSILON);
            
            // Test 1.0
            let result = graph.update_entity_activation(entity_key, 1.0).await;
            assert!(result.is_ok());
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 1.0).abs() < f32::EPSILON);
        }
    }

    mod remove_brain_entity_tests {
        use super::*;

        #[tokio::test]
        async fn test_remove_brain_entity_exists() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(50, entity_data).await.unwrap();
            
            // Verify entity exists
            assert!(graph.contains_entity(entity_key));
            
            // Remove entity
            let result = graph.remove_brain_entity(entity_key).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), true);
            
            // Verify entity no longer exists
            assert!(!graph.contains_entity(entity_key));
        }

        #[tokio::test]
        async fn test_remove_brain_entity_nonexistent() {
            let graph = create_test_brain_graph().await;
            
            let fake_key = EntityKey::default();
            let result = graph.remove_brain_entity(fake_key).await;
            
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), false);
        }

        #[tokio::test]
        async fn test_remove_brain_entity_clears_activation() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(51, entity_data).await.unwrap();
            
            // Set activation
            graph.set_entity_activation(entity_key, 0.8).await;
            
            // Verify activation exists
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 0.8).abs() < f32::EPSILON);
            
            // Remove entity
            let result = graph.remove_brain_entity(entity_key).await;
            assert!(result.is_ok());
            
            // Verify activation is cleared (should return default 0.0)
            let activation = graph.get_entity_activation(entity_key).await;
            assert!((activation - 0.0).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_remove_brain_entity_clears_synaptic_weights() {
            let graph = create_test_brain_graph().await;
            
            let entity1_data = create_test_entity_data(1, None);
            let entity2_data = create_test_entity_data(1, None);
            
            let entity1_key = graph.insert_brain_entity(52, entity1_data).await.unwrap();
            let entity2_key = graph.insert_brain_entity(53, entity2_data).await.unwrap();
            
            // Set synaptic weight
            graph.set_synaptic_weight(entity1_key, entity2_key, 0.7).await;
            
            // Verify weight exists
            let weight = graph.get_synaptic_weight(entity1_key, entity2_key).await;
            assert!((weight - 0.7).abs() < f32::EPSILON);
            
            // Remove first entity
            let result = graph.remove_brain_entity(entity1_key).await;
            assert!(result.is_ok());
            
            // Verify synaptic weight is cleared
            let weight = graph.get_synaptic_weight(entity1_key, entity2_key).await;
            assert!((weight - 0.0).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_remove_brain_entity_updates_concept_structures() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(54, entity_data).await.unwrap();
            
            // Create a concept structure with this entity
            let mut concept = ConceptStructure::new();
            concept.add_input(entity_key);
            concept.add_output(entity_key);
            graph.store_concept_structure("test_concept".to_string(), concept).await;
            
            // Verify entity is in concept
            let stored_concept = graph.get_concept_structure("test_concept").await.unwrap();
            assert!(stored_concept.input_entities.contains(&entity_key));
            assert!(stored_concept.output_entities.contains(&entity_key));
            
            // Remove entity
            let result = graph.remove_brain_entity(entity_key).await;
            assert!(result.is_ok());
            
            // Verify entity is removed from concept structures
            let updated_concept = graph.get_concept_structure("test_concept").await.unwrap();
            assert!(!updated_concept.input_entities.contains(&entity_key));
            assert!(!updated_concept.output_entities.contains(&entity_key));
        }

        #[tokio::test]
        async fn test_remove_brain_entity_updates_statistics() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(55, entity_data).await.unwrap();
            
            // Get initial count
            let initial_stats = graph.get_learning_stats().await;
            let initial_count = initial_stats.entity_count;
            
            // Remove entity
            let result = graph.remove_brain_entity(entity_key).await;
            assert!(result.is_ok());
            
            // Verify count decreased
            let updated_stats = graph.get_learning_stats().await;
            assert_eq!(updated_stats.entity_count, initial_count.saturating_sub(1));
        }
    }

    mod private_method_tests {
        use super::*;

        #[tokio::test]
        async fn test_calculate_initial_activation_basic() {
            let graph = create_test_brain_graph().await;
            let entity_data = create_test_entity_data(1, None);
            
            let activation = graph.calculate_initial_activation(&entity_data);
            assert!(activation >= 0.0);
            assert!(activation <= 1.0);
        }

        #[tokio::test]
        async fn test_calculate_initial_activation_with_importance() {
            let graph = create_test_brain_graph().await;
            
            // Create entity with importance
            let high_importance_data = create_entity_with_importance(0.8);
            let low_importance_data = create_entity_with_importance(0.1);
            
            let high_activation = graph.calculate_initial_activation(&high_importance_data);
            let low_activation = graph.calculate_initial_activation(&low_importance_data);
            
            // High importance should result in higher activation
            assert!(high_activation > low_activation);
        }

        #[tokio::test]
        async fn test_calculate_initial_activation_empty_embedding() {
            let graph = create_test_brain_graph().await;
            let mut entity_data = create_test_entity_data(1, None);
            entity_data.embedding = vec![];
            
            let activation = graph.calculate_initial_activation(&entity_data);
            // Should handle empty embedding gracefully
            assert!(activation >= 0.0);
            assert!(activation <= 1.0);
        }

        #[tokio::test]
        async fn test_generate_gate_embedding_basic() {
            let graph = create_test_brain_graph().await;
            
            let input1_data = create_test_entity_data(1, None);
            let input2_data = create_test_entity_data(1, None);
            let output_data = create_test_entity_data(1, None);
            
            let input1_key = graph.insert_brain_entity(60, input1_data).await.unwrap();
            let input2_key = graph.insert_brain_entity(61, input2_data).await.unwrap();
            let output_key = graph.insert_brain_entity(62, output_data).await.unwrap();
            
            let result = graph.generate_gate_embedding(
                "AND", 
                &[input1_key, input2_key], 
                &[output_key]
            );
            
            assert!(result.is_ok());
            let embedding = result.unwrap();
            assert!(!embedding.is_empty());
            assert_eq!(embedding.len(), graph.embedding_dimension());
            
            // Verify normalized (magnitude should be close to 1.0)
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((magnitude - 1.0).abs() < 0.1);
        }

        #[tokio::test]
        async fn test_generate_gate_embedding_different_types() {
            let graph = create_test_brain_graph().await;
            
            let input_data = create_test_entity_data(1, None);
            let output_data = create_test_entity_data(1, None);
            
            let input_key = graph.insert_brain_entity(63, input_data).await.unwrap();
            let output_key = graph.insert_brain_entity(64, output_data).await.unwrap();
            
            let and_embedding = graph.generate_gate_embedding("AND", &[input_key], &[output_key]).unwrap();
            let or_embedding = graph.generate_gate_embedding("OR", &[input_key], &[output_key]).unwrap();
            let not_embedding = graph.generate_gate_embedding("NOT", &[input_key], &[output_key]).unwrap();
            
            // Different gate types should produce different embeddings
            assert_ne!(and_embedding, or_embedding);
            assert_ne!(and_embedding, not_embedding);
            assert_ne!(or_embedding, not_embedding);
        }

        #[tokio::test]
        async fn test_generate_property_embedding() {
            let graph = create_test_brain_graph().await;
            
            let mut properties = HashMap::new();
            properties.insert("name".to_string(), "test".to_string());
            properties.insert("type".to_string(), "concept".to_string());
            
            let result = graph.generate_property_embedding(&properties);
            assert!(result.is_ok());
            
            let embedding = result.unwrap();
            assert!(!embedding.is_empty());
            assert_eq!(embedding.len(), graph.embedding_dimension());
            
            // Verify normalized
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((magnitude - 1.0).abs() < 0.1);
        }

        #[tokio::test]
        async fn test_generate_property_embedding_empty() {
            let graph = create_test_brain_graph().await;
            let empty_properties = HashMap::new();
            
            let result = graph.generate_property_embedding(&empty_properties);
            assert!(result.is_ok());
            
            let embedding = result.unwrap();
            // Should return zero vector for empty properties
            assert!(embedding.iter().all(|&x| x == 0.0));
        }

        #[tokio::test]
        async fn test_hash_string_consistency() {
            let graph = create_test_brain_graph().await;
            
            let hash1 = graph.hash_string("test");
            let hash2 = graph.hash_string("test");
            let hash3 = graph.hash_string("different");
            
            // Same string should produce same hash
            assert_eq!(hash1, hash2);
            // Different strings should produce different hashes
            assert_ne!(hash1, hash3);
        }

        #[tokio::test]
        async fn test_trigger_concept_formation() {
            let graph = create_test_brain_graph().await;
            
            // Need to enable concept formation
            let mut config = graph.config.clone();
            config.enable_concept_formation = true;
            // Note: We can't modify the config in this immutable reference context,
            // but we can test that the method doesn't panic
            
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(70, entity_data).await.unwrap();
            
            let result = graph.trigger_concept_formation(entity_key).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_apply_hebbian_learning() {
            let graph = create_test_brain_graph().await;
            
            let entity1_data = create_test_entity_data(1, None);
            let entity2_data = create_test_entity_data(1, None);
            
            let entity1_key = graph.insert_brain_entity(71, entity1_data).await.unwrap();
            let entity2_key = graph.insert_brain_entity(72, entity2_data).await.unwrap();
            
            // Set initial activations
            graph.set_entity_activation(entity1_key, 0.8).await;
            graph.set_entity_activation(entity2_key, 0.7).await;
            
            let result = graph.apply_hebbian_learning(entity1_key, entity2_key).await;
            assert!(result.is_ok());
            
            // Verify synaptic weight was updated
            let weight = graph.get_synaptic_weight(entity1_key, entity2_key).await;
            assert!(weight > 0.0);
        }
    }

    mod entity_lifecycle_tests {
        use super::*;

        #[tokio::test]
        async fn test_complete_entity_lifecycle() {
            let graph = create_test_brain_graph().await;
            
            // 1. Create entity
            let entity_data = create_test_entity_data(1, None);
            let entity_key = graph.insert_brain_entity(80, entity_data.clone()).await.unwrap();
            
            // 2. Verify entity exists
            assert!(graph.contains_entity(entity_key));
            
            // 3. Retrieve entity data
            let retrieved = graph.get_entity(entity_key).await;
            assert!(retrieved.is_some());
            let (retrieved_data, initial_activation) = retrieved.unwrap();
            assert_eq!(retrieved_data.type_id, entity_data.type_id);
            
            // 4. Update activation
            let new_activation = 0.85;
            let update_result = graph.update_entity_activation(entity_key, new_activation).await;
            assert!(update_result.is_ok());
            
            // 5. Verify activation updated
            let current_activation = graph.get_entity_activation(entity_key).await;
            assert!((current_activation - new_activation).abs() < f32::EPSILON);
            
            // 6. Remove entity
            let remove_result = graph.remove_brain_entity(entity_key).await;
            assert!(remove_result.is_ok());
            assert_eq!(remove_result.unwrap(), true);
            
            // 7. Verify entity no longer exists
            assert!(!graph.contains_entity(entity_key));
            let final_get = graph.get_entity(entity_key).await;
            assert!(final_get.is_none());
        }

        #[tokio::test]
        async fn test_batch_entity_operations() {
            let graph = create_test_brain_graph().await;
            
            // Create multiple entities
            let entities = vec![
                (81, create_test_entity_data(1, None)),
                (82, create_test_entity_data(2, None)),
                (83, create_test_entity_data(3, None)),
            ];
            
            let keys = graph.batch_insert_entities(entities).await.unwrap();
            assert_eq!(keys.len(), 3);
            
            // Verify all entities exist
            for key in &keys {
                assert!(graph.contains_entity(*key));
            }
            
            // Update all activations
            for (i, key) in keys.iter().enumerate() {
                let activation = 0.1 * (i + 1) as f32;
                let result = graph.update_entity_activation(*key, activation).await;
                assert!(result.is_ok());
            }
            
            // Verify activations
            for (i, key) in keys.iter().enumerate() {
                let expected_activation = 0.1 * (i + 1) as f32;
                let actual_activation = graph.get_entity_activation(*key).await;
                assert!((actual_activation - expected_activation).abs() < f32::EPSILON);
            }
            
            // Remove all entities
            for key in &keys {
                let result = graph.remove_brain_entity(*key).await;
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), true);
            }
            
            // Verify all entities removed
            for key in &keys {
                assert!(!graph.contains_entity(*key));
            }
        }
    }

    mod edge_cases_tests {
        use super::*;

        #[tokio::test]
        async fn test_large_embedding_entity() {
            let graph = create_test_brain_graph().await;
            
            // Create entity with large embedding
            let large_embedding: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
            let mut entity_data = create_test_entity_data(1, None);
            entity_data.embedding = large_embedding;
            
            let result = graph.insert_brain_entity(90, entity_data).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            assert!(graph.contains_entity(entity_key));
        }

        #[tokio::test]
        async fn test_zero_embedding_entity() {
            let graph = create_test_brain_graph().await;
            
            // Create entity with zero embedding
            let mut entity_data = create_test_entity_data(1, None);
            entity_data.embedding = vec![0.0; 96];
            
            let result = graph.insert_brain_entity(91, entity_data).await;
            assert!(result.is_ok());
            
            let entity_key = result.unwrap();
            assert!(graph.contains_entity(entity_key));
            
            // Activation should still be calculated
            let activation = graph.get_entity_activation(entity_key).await;
            assert!(activation >= 0.0);
        }

        #[tokio::test]
        async fn test_malformed_properties_json() {
            let graph = create_test_brain_graph().await;
            
            // Create entity with malformed JSON properties
            let mut entity_data = create_test_entity_data(1, None);
            entity_data.properties = "{ invalid json".to_string();
            
            let result = graph.insert_brain_entity(92, entity_data).await;
            assert!(result.is_ok());
            
            // Should handle malformed JSON gracefully
            let entity_key = result.unwrap();
            assert!(graph.contains_entity(entity_key));
        }

        #[tokio::test]
        async fn test_extremely_long_properties() {
            let graph = create_test_brain_graph().await;
            
            // Create entity with very long properties string
            let long_value = "x".repeat(10000);
            let mut properties = HashMap::new();
            properties.insert("long_key".to_string(), long_value);
            
            let entity_data = create_test_entity_data(1, Some(properties));
            let result = graph.insert_brain_entity(93, entity_data).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_concurrent_entity_operations() {
            let graph = std::sync::Arc::new(create_test_brain_graph().await);
            
            // Test concurrent insertions
            let mut handles = vec![];
            
            for i in 100..110 {
                let graph_clone = graph.clone();
                let handle = tokio::spawn(async move {
                    let entity_data = create_test_entity_data(1, None);
                    graph_clone.insert_brain_entity(i, entity_data).await
                });
                handles.push(handle);
            }
            
            // Wait for all insertions to complete
            let mut keys = vec![];
            for handle in handles {
                let result = handle.await.unwrap();
                assert!(result.is_ok());
                keys.push(result.unwrap());
            }
            
            // Verify all entities were inserted
            assert_eq!(keys.len(), 10);
            for key in &keys {
                assert!(graph.contains_entity(*key));
            }
        }
    }

    mod entity_statistics_tests {
        use super::*;

        #[tokio::test]
        async fn test_entity_statistics_empty() {
            let graph = create_test_brain_graph().await;
            
            let stats = graph.get_entity_statistics().await;
            assert_eq!(stats.total_entities, 0);
            assert_eq!(stats.avg_activation, 0.0);
            assert_eq!(stats.max_activation, 0.0);
            assert_eq!(stats.min_activation, 1.0); // Default min value when no entities
        }

        #[tokio::test]
        async fn test_entity_statistics_with_entities() {
            let graph = create_test_brain_graph().await;
            
            // Insert a few entities
            let entity1 = create_test_entity_data(1, None);
            let entity2 = create_test_entity_data(1, None);
            
            let key1 = graph.insert_brain_entity(200, entity1).await.unwrap();
            let key2 = graph.insert_brain_entity(201, entity2).await.unwrap();
            
            // Set specific activations
            graph.set_entity_activation(key1, 0.3).await;
            graph.set_entity_activation(key2, 0.7).await;
            
            let stats = graph.get_entity_statistics().await;
            assert_eq!(stats.total_entities, 2);
            assert!((stats.avg_activation - 0.5).abs() < f32::EPSILON);
            assert!((stats.max_activation - 0.7).abs() < f32::EPSILON);
            assert!((stats.min_activation - 0.3).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_entity_statistics_activation_range() {
            let stats = EntityStatistics {
                total_entities: 2,
                avg_activation: 0.5,
                max_activation: 0.8,
                min_activation: 0.2,
                type_distribution: HashMap::new(),
            };
            
            assert!((stats.activation_range() - 0.6).abs() < f32::EPSILON);
        }

        #[tokio::test]
        async fn test_entity_statistics_is_well_distributed() {
            let well_distributed = EntityStatistics {
                total_entities: 10,
                avg_activation: 0.5,
                max_activation: 0.8,
                min_activation: 0.2,
                type_distribution: HashMap::new(),
            };
            
            let poorly_distributed = EntityStatistics {
                total_entities: 10,
                avg_activation: 0.9,
                max_activation: 1.0,
                min_activation: 0.95,
                type_distribution: HashMap::new(),
            };
            
            assert!(well_distributed.is_well_distributed());
            assert!(!poorly_distributed.is_well_distributed());
        }

        #[tokio::test]
        async fn test_entity_statistics_most_common_type() {
            let mut type_distribution = HashMap::new();
            type_distribution.insert("input".to_string(), 5);
            type_distribution.insert("output".to_string(), 3);
            type_distribution.insert("gate".to_string(), 7);
            
            let stats = EntityStatistics {
                total_entities: 15,
                avg_activation: 0.5,
                max_activation: 1.0,
                min_activation: 0.0,
                type_distribution,
            };
            
            let most_common = stats.most_common_type();
            assert!(most_common.is_some());
            let (type_name, count) = most_common.unwrap();
            assert_eq!(type_name, "gate");
            assert_eq!(count, 7);
        }
    }
}