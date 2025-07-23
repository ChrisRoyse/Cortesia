//! Test helper methods for BrainEnhancedKnowledgeGraph
//! 
//! These methods provide a simpler API for tests, matching what the tests expect.
//! They are only compiled in test builds.

#[cfg(test)]
use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
#[cfg(test)]
use crate::core::brain_types::{BrainInspiredEntity, BrainInspiredRelationship, EntityDirection, RelationType};
#[cfg(test)]
use crate::core::types::{EntityKey, EntityData, Relationship};
#[cfg(test)]
use crate::error::Result;


#[cfg(test)]
impl BrainEnhancedKnowledgeGraph {
    /// Create a brain entity with just a concept name and direction
    /// This is a test helper that creates the necessary EntityData internally
    pub async fn create_brain_entity(
        &self,
        concept: String,
        direction: EntityDirection,
    ) -> Result<EntityKey> {
        // Create a BrainInspiredEntity
        let brain_entity = BrainInspiredEntity::new(concept.clone(), direction);
        
        // Convert to EntityData for insertion
        let entity_data = EntityData {
            type_id: match direction {
                EntityDirection::Input => 1,
                EntityDirection::Output => 2,
                EntityDirection::Gate => 3,
                EntityDirection::Hidden => 4,
            },
            properties: serde_json::json!({
                "concept": concept,
                "direction": format!("{:?}", direction),
                "entity_id": format!("{:?}", brain_entity.id),
            }).to_string(),
            embedding: brain_entity.embedding.clone(),
        };
        
        // Use a simple hash of the concept as the ID
        let id = concept.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        
        // Insert using the existing method
        self.insert_brain_entity(id, entity_data).await
    }
    
    /// Create a brain relationship between entities
    /// This is a test helper that creates the necessary Relationship struct
    pub async fn create_brain_relationship(
        &self,
        source: EntityKey,
        target: EntityKey,
        relation_type: RelationType,
        weight: f32,
    ) -> Result<()> {
        // Create BrainInspiredRelationship
        let mut brain_rel = BrainInspiredRelationship::new(source, target, relation_type);
        brain_rel.weight = weight;
        brain_rel.strength = weight;
        
        // Convert to standard Relationship
        let relationship = Relationship {
            from: source,
            to: target,
            rel_type: relation_type as u8,
            weight,
        };
        
        // Insert using the existing method
        self.insert_brain_relationship(relationship).await
    }
}