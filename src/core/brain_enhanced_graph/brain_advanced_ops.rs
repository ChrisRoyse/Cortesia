//! Advanced operations for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_concept_ops::EntityRole;
use super::brain_graph_types::*;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{EntityData, Relationship};
    use std::collections::HashMap;
    use tokio;

    // Helper function to create test graph
    async fn create_test_graph() -> Result<BrainEnhancedKnowledgeGraph> {
        BrainEnhancedKnowledgeGraph::new_for_test()
    }

    // Helper function to add test entity
    async fn add_test_entity(graph: &BrainEnhancedKnowledgeGraph, name: &str, properties: &str, embedding: Vec<f32>) -> Result<EntityKey> {
        let entity_data = EntityData {
            name: name.to_string(),
            properties: properties.to_string(),
            embedding,
        };
        graph.core_graph.insert_entity(entity_data)
    }

    #[tokio::test]
    async fn test_create_inheritance_basic() {
        let graph = create_test_graph().await.unwrap();
        
        // Add parent and child entities
        let parent_embedding = vec![1.0, 0.0, 0.0, 0.0]; // 4-dimensional for simplicity
        let child_embedding = vec![0.5, 0.0, 0.0, 0.0];
        
        let parent = add_test_entity(&graph, "parent", "type:concept", parent_embedding).await.unwrap();
        let child = add_test_entity(&graph, "child", "type:concept", child_embedding).await.unwrap();
        
        // Set initial activations
        graph.set_entity_activation(parent, 0.8).await;
        graph.set_entity_activation(child, 0.2).await;
        
        // Create inheritance relationship
        let inheritance_strength = 0.7;
        let result = graph.create_inheritance(parent, child, inheritance_strength).await;
        
        assert!(result.is_ok());
        
        // Verify relationship was created
        assert!(graph.has_relationship(parent, child).await);
        
        // Verify child activation was increased due to inheritance
        let child_activation = graph.get_entity_activation(child).await;
        assert!(child_activation > 0.2); // Should be higher than initial
        assert!(child_activation <= 1.0); // Should be clamped
    }

    #[tokio::test]
    async fn test_create_inheritance_with_high_strength() {
        let graph = create_test_graph().await.unwrap();
        
        let parent_embedding = vec![1.0, 1.0, 1.0, 1.0];
        let child_embedding = vec![0.0, 0.0, 0.0, 0.0];
        
        let parent = add_test_entity(&graph, "strong_parent", "type:concept", parent_embedding).await.unwrap();
        let child = add_test_entity(&graph, "weak_child", "type:concept", child_embedding).await.unwrap();
        
        // Set high parent activation
        graph.set_entity_activation(parent, 1.0).await;
        graph.set_entity_activation(child, 0.0).await;
        
        // Create strong inheritance
        let result = graph.create_inheritance(parent, child, 1.0).await;
        assert!(result.is_ok());
        
        // Child should get significant activation boost
        let child_activation = graph.get_entity_activation(child).await;
        assert!(child_activation > 0.2); // Should get 1.0 * 1.0 * 0.3 = 0.3 boost
    }

    #[tokio::test]
    async fn test_create_inheritance_activation_clamping() {
        let graph = create_test_graph().await.unwrap();
        
        let parent_embedding = vec![1.0, 1.0, 1.0, 1.0];
        let child_embedding = vec![0.8, 0.8, 0.8, 0.8];
        
        let parent = add_test_entity(&graph, "active_parent", "type:concept", parent_embedding).await.unwrap();
        let child = add_test_entity(&graph, "active_child", "type:concept", child_embedding).await.unwrap();
        
        // Set both to high activation
        graph.set_entity_activation(parent, 1.0).await;
        graph.set_entity_activation(child, 0.9).await;
        
        let result = graph.create_inheritance(parent, child, 1.0).await;
        assert!(result.is_ok());
        
        // Child activation should be clamped to 1.0
        let child_activation = graph.get_entity_activation(child).await;
        assert_eq!(child_activation, 1.0);
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_identical() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![1.0, 0.0, 0.0, 0.0];
        let embedding2 = vec![1.0, 0.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert!((similarity - 1.0).abs() < 0.001); // Should be 1.0 for identical embeddings
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_orthogonal() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![1.0, 0.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert!((similarity - 0.0).abs() < 0.001); // Should be 0.0 for orthogonal embeddings
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_opposite() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![1.0, 0.0, 0.0, 0.0];
        let embedding2 = vec![-1.0, 0.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert!((similarity - (-1.0)).abs() < 0.001); // Should be -1.0 for opposite embeddings
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_different_lengths() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![1.0, 0.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert_eq!(similarity, 0.0); // Should be 0.0 for different length embeddings
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_zero_norm() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![0.0, 0.0, 0.0, 0.0];
        let embedding2 = vec![1.0, 0.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert_eq!(similarity, 0.0); // Should be 0.0 when one embedding has zero norm
    }

    #[tokio::test]
    async fn test_calculate_embedding_similarity_normalized() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding1 = vec![3.0, 4.0, 0.0, 0.0]; // norm = 5
        let embedding2 = vec![6.0, 8.0, 0.0, 0.0]; // norm = 10, same direction
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert!((similarity - 1.0).abs() < 0.001); // Should be 1.0 for same direction regardless of magnitude
    }

    #[tokio::test]
    async fn test_determine_entity_role_input_from_properties() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let entity = add_test_entity(&graph, "input_entity", "type:input sensor", embedding).await.unwrap();
        
        let role = graph.determine_entity_role(entity).await;
        assert_eq!(role, EntityRole::Input);
    }

    #[tokio::test]
    async fn test_determine_entity_role_output_from_properties() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let entity = add_test_entity(&graph, "output_entity", "type:output actuator", embedding).await.unwrap();
        
        let role = graph.determine_entity_role(entity).await;
        assert_eq!(role, EntityRole::Output);
    }

    #[tokio::test]
    async fn test_determine_entity_role_gate_from_properties() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let entity = add_test_entity(&graph, "gate_entity", "type:logic_gate and_gate", embedding).await.unwrap();
        
        let role = graph.determine_entity_role(entity).await;
        assert_eq!(role, EntityRole::Gate);
    }

    #[tokio::test]
    async fn test_determine_entity_role_input_from_connectivity() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let input_entity = add_test_entity(&graph, "isolated_input", "type:node", embedding.clone()).await.unwrap();
        let output_entity = add_test_entity(&graph, "target", "type:node", embedding).await.unwrap();
        
        // Create outgoing relationship (no incoming)
        let relationship = Relationship {
            from: input_entity,
            to: output_entity,
            rel_type: 0,
            weight: 1.0,
        };
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        let role = graph.determine_entity_role(input_entity).await;
        assert_eq!(role, EntityRole::Input); // No incoming, has outgoing
    }

    #[tokio::test]
    async fn test_determine_entity_role_output_from_connectivity() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let input_entity = add_test_entity(&graph, "source", "type:node", embedding.clone()).await.unwrap();
        let output_entity = add_test_entity(&graph, "isolated_output", "type:node", embedding).await.unwrap();
        
        // Create incoming relationship (no outgoing)
        let relationship = Relationship {
            from: input_entity,
            to: output_entity,
            rel_type: 0,
            weight: 1.0,
        };
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        let role = graph.determine_entity_role(output_entity).await;
        assert_eq!(role, EntityRole::Output); // Has incoming, no outgoing
    }

    #[tokio::test]
    async fn test_determine_entity_role_gate_from_connectivity() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let input1 = add_test_entity(&graph, "input1", "type:node", embedding.clone()).await.unwrap();
        let input2 = add_test_entity(&graph, "input2", "type:node", embedding.clone()).await.unwrap();
        let gate = add_test_entity(&graph, "gate", "type:node", embedding.clone()).await.unwrap();
        let output1 = add_test_entity(&graph, "output1", "type:node", embedding.clone()).await.unwrap();
        let output2 = add_test_entity(&graph, "output2", "type:node", embedding).await.unwrap();
        
        // Create multiple incoming and outgoing relationships
        let relationships = vec![
            Relationship { from: input1, to: gate, rel_type: 0, weight: 1.0 },
            Relationship { from: input2, to: gate, rel_type: 0, weight: 1.0 },
            Relationship { from: gate, to: output1, rel_type: 0, weight: 1.0 },
            Relationship { from: gate, to: output2, rel_type: 0, weight: 1.0 },
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let role = graph.determine_entity_role(gate).await;
        assert_eq!(role, EntityRole::Gate); // Multiple incoming and outgoing
    }

    #[tokio::test]
    async fn test_determine_entity_role_processing_fallback() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let input = add_test_entity(&graph, "input", "type:node", embedding.clone()).await.unwrap();
        let processor = add_test_entity(&graph, "processor", "type:node", embedding.clone()).await.unwrap();
        let output = add_test_entity(&graph, "output", "type:node", embedding).await.unwrap();
        
        // Create single incoming and single outgoing (processing pattern)
        let relationships = vec![
            Relationship { from: input, to: processor, rel_type: 0, weight: 1.0 },
            Relationship { from: processor, to: output, rel_type: 0, weight: 1.0 },
        ];
        
        for rel in relationships {
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let role = graph.determine_entity_role(processor).await;
        assert_eq!(role, EntityRole::Processing); // Single incoming and outgoing
    }

    #[tokio::test]
    async fn test_determine_entity_role_nonexistent_entity() {
        let graph = create_test_graph().await.unwrap();
        
        // Create a dummy entity key that doesn't exist in the graph
        use slotmap::Key;
        let dummy_key = EntityKey::from(slotmap::KeyData::from_ffi(999999));
        
        let role = graph.determine_entity_role(dummy_key).await;
        assert_eq!(role, EntityRole::Input); // Default for isolated entities
    }

    #[tokio::test]
    async fn test_inheritance_with_zero_strength() {
        let graph = create_test_graph().await.unwrap();
        
        let parent_embedding = vec![1.0, 0.0, 0.0, 0.0];
        let child_embedding = vec![0.0, 0.0, 0.0, 0.0];
        
        let parent = add_test_entity(&graph, "parent", "type:concept", parent_embedding).await.unwrap();
        let child = add_test_entity(&graph, "child", "type:concept", child_embedding).await.unwrap();
        
        graph.set_entity_activation(parent, 1.0).await;
        graph.set_entity_activation(child, 0.5).await;
        
        let result = graph.create_inheritance(parent, child, 0.0).await;
        assert!(result.is_ok());
        
        // Child activation should remain unchanged
        let child_activation = graph.get_entity_activation(child).await;
        assert_eq!(child_activation, 0.5);
    }

    #[tokio::test]
    async fn test_inheritance_with_negative_strength() {
        let graph = create_test_graph().await.unwrap();
        
        let parent_embedding = vec![1.0, 0.0, 0.0, 0.0];
        let child_embedding = vec![0.0, 0.0, 0.0, 0.0];
        
        let parent = add_test_entity(&graph, "parent", "type:concept", parent_embedding).await.unwrap();
        let child = add_test_entity(&graph, "child", "type:concept", child_embedding).await.unwrap();
        
        graph.set_entity_activation(parent, 1.0).await;
        graph.set_entity_activation(child, 0.5).await;
        
        let result = graph.create_inheritance(parent, child, -0.5).await;
        assert!(result.is_ok());
        
        // Child activation should potentially be reduced (negative inheritance)
        let child_activation = graph.get_entity_activation(child).await;
        assert!(child_activation >= 0.0); // But still clamped to valid range
    }

    #[tokio::test]
    async fn test_multiple_inheritance_relationships() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let parent1 = add_test_entity(&graph, "parent1", "type:concept", embedding.clone()).await.unwrap();
        let parent2 = add_test_entity(&graph, "parent2", "type:concept", embedding.clone()).await.unwrap();
        let child = add_test_entity(&graph, "child", "type:concept", embedding).await.unwrap();
        
        graph.set_entity_activation(parent1, 0.8).await;
        graph.set_entity_activation(parent2, 0.6).await;
        graph.set_entity_activation(child, 0.1).await;
        
        // Create multiple inheritance relationships
        let result1 = graph.create_inheritance(parent1, child, 0.7).await;
        let result2 = graph.create_inheritance(parent2, child, 0.5).await;
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        // Verify both relationships exist
        assert!(graph.has_relationship(parent1, child).await);
        assert!(graph.has_relationship(parent2, child).await);
        
        // Child should have accumulated activation from both parents
        let child_activation = graph.get_entity_activation(child).await;
        assert!(child_activation > 0.1); // Should be higher than initial
    }

    #[tokio::test]
    async fn test_embedding_similarity_with_various_patterns() {
        let graph = create_test_graph().await.unwrap();
        
        // Test partial similarity
        let embedding1 = vec![1.0, 0.5, 0.0, 0.0];
        let embedding2 = vec![0.5, 1.0, 0.0, 0.0];
        
        let similarity = graph.calculate_embedding_similarity(&embedding1, &embedding2);
        assert!(similarity > 0.0 && similarity < 1.0); // Should be between 0 and 1
        
        // Test with normalized vectors of different magnitudes
        let embedding3 = vec![2.0, 0.0, 0.0, 0.0];
        let embedding4 = vec![4.0, 0.0, 0.0, 0.0];
        
        let similarity2 = graph.calculate_embedding_similarity(&embedding3, &embedding4);
        assert!((similarity2 - 1.0).abs() < 0.001); // Should be 1.0 for same direction
    }

    #[tokio::test]
    async fn test_entity_role_edge_cases() {
        let graph = create_test_graph().await.unwrap();
        
        // Test entity with mixed properties
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let mixed_entity = add_test_entity(&graph, "mixed", "input output logic_gate", embedding).await.unwrap();
        
        let role = graph.determine_entity_role(mixed_entity).await;
        // Should pick the first matching pattern (input in this case)
        assert_eq!(role, EntityRole::Input);
    }

    #[tokio::test]
    async fn test_inheritance_chain() {
        let graph = create_test_graph().await.unwrap();
        
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let grandparent = add_test_entity(&graph, "grandparent", "type:concept", embedding.clone()).await.unwrap();
        let parent = add_test_entity(&graph, "parent", "type:concept", embedding.clone()).await.unwrap();
        let child = add_test_entity(&graph, "child", "type:concept", embedding).await.unwrap();
        
        graph.set_entity_activation(grandparent, 1.0).await;
        graph.set_entity_activation(parent, 0.0).await;
        graph.set_entity_activation(child, 0.0).await;
        
        // Create inheritance chain
        let result1 = graph.create_inheritance(grandparent, parent, 0.8).await;
        let result2 = graph.create_inheritance(parent, child, 0.6).await;
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        // Parent should get activation from grandparent
        let parent_activation = graph.get_entity_activation(parent).await;
        assert!(parent_activation > 0.0);
        
        // Child should get activation from parent
        let child_activation = graph.get_entity_activation(child).await;
        assert!(child_activation > 0.0);
    }
}

