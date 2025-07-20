//! Advanced operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_concept_ops::EntityRole;
use crate::core::types::EntityKey;
use crate::error::Result;

impl BrainEnhancedKnowledgeGraph {

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

    /// Calculate similarity between two embeddings
    pub(crate) fn calculate_embedding_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
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

    /// Determine entity role in concept
    pub(crate) async fn determine_entity_role(&self, entity_key: EntityKey) -> EntityRole {
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
}

