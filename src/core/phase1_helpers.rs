use std::collections::HashMap;
use crate::core::brain_types::{BrainInspiredEntity, EntityDirection};
use crate::core::phase1_types::EntityInfo;
use crate::neural::canonicalization::EnhancedNeuralCanonicalizer;
use crate::neural::neural_server::NeuralProcessingServer;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::sync::Arc;

pub struct Phase1Helpers {
    canonicalizer: Arc<EnhancedNeuralCanonicalizer>,
    neural_server: Arc<NeuralProcessingServer>,
    brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
}

impl Phase1Helpers {
    pub fn new(
        canonicalizer: Arc<EnhancedNeuralCanonicalizer>,
        neural_server: Arc<NeuralProcessingServer>,
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            canonicalizer,
            neural_server,
            brain_graph,
        }
    }

    pub async fn canonicalize_text_entities(
        &self,
        text: &str,
        context: Option<&str>,
    ) -> Result<HashMap<String, String>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut canonical_map = HashMap::new();
        
        for word in words {
            if word.len() > 2 {
                let canonical = if let Some(ctx) = context {
                    self.canonicalizer.canonicalize_with_context(word, ctx).await?
                } else {
                    self.canonicalizer.base_canonicalizer.canonicalize_entity(word).await?
                };
                canonical_map.insert(word.to_string(), canonical);
            }
        }
        
        Ok(canonical_map)
    }

    pub async fn generate_embedding(&self, concept: &str) -> Result<Vec<f32>> {
        // Create a simple input vector from the concept
        let mut input_vector = vec![0.0; 384];
        for (i, byte) in concept.bytes().enumerate() {
            if i < 384 {
                input_vector[i] = (byte as f32) / 255.0;
            }
        }
        
        let result = self.neural_server.neural_predict(
            "embedding_model",
            input_vector,
        ).await?;
        
        Ok(result.prediction)
    }

    pub async fn get_brain_entity(
        &self,
        entity_key: crate::core::types::EntityKey,
    ) -> Result<Option<BrainInspiredEntity>> {
        // Entities are stored in the core graph
        let entity_data = self.brain_graph.core_graph.get_entity(entity_key);
        
        // Convert to BrainInspiredEntity
        if let Some((_meta, data)) = entity_data {
            Ok(Some(BrainInspiredEntity {
                id: entity_key,
                concept_id: format!("entity_{:?}", entity_key),
                direction: EntityDirection::Input,
                properties: HashMap::new(),
                embedding: data.embedding,
                activation_state: 0.0,
                last_activation: std::time::SystemTime::now(),
                last_update: std::time::SystemTime::now(),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn extract_query_entities(&self, query: &str) -> Result<Vec<String>> {
        // Simple entity extraction - would use NLP in practice
        Ok(query.split_whitespace()
            .filter(|word| word.len() > 2)
            .map(|word| word.to_lowercase())
            .collect())
    }

    pub async fn find_entities_by_concept(
        &self,
        concept: &str,
    ) -> Result<Vec<crate::core::types::EntityKey>> {
        let all_entities = self.brain_graph.get_all_entities().await;
        let mut matching_keys = Vec::new();
        
        for (key, entity_data, _) in &all_entities {
            // Check if entity properties contain the concept
            if entity_data.properties.to_lowercase().contains(&concept.to_lowercase()) {
                matching_keys.push(*key);
            }
        }
        
        Ok(matching_keys)
    }

    pub async fn get_entities_info(
        &self,
        activations: &[(crate::core::types::EntityKey, f32)],
    ) -> Result<Vec<EntityInfo>> {
        let all_entities = self.brain_graph.get_all_entities().await;
        let mut info = Vec::new();
        
        for (key, activation) in activations {
            if let Some((_, _entity_data, _)) = all_entities.iter().find(|(k, _, _)| k == key) {
                info.push(EntityInfo {
                    entity_key: *key,
                    concept_id: format!("entity_{:?}", key),
                    direction: EntityDirection::Input,
                    activation_level: *activation,
                });
            }
        }
        
        Ok(info)
    }
}