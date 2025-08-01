pub mod advanced_nlp;

pub use advanced_nlp::{
    AdvancedEntityExtractor, 
    Entity, 
    Relation, 
    NERModel,
    RelationModel,
    EntityLinker,
    CoreferenceResolver,
    RelationExtractor,
    PredicateNormalizer,
    ConfidenceScorer,
};

use crate::error::Result;
use async_trait::async_trait;

/// Trait for entity extraction functionality
#[async_trait]
pub trait EntityExtractor {
    /// Extract entities from the given text
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>>;
    
    /// Extract entities with confidence scores
    async fn extract_entities_with_confidence(&self, text: &str) -> Result<Vec<(Entity, f32)>>;
}

/// Implementation of EntityExtractor for &str (simple regex-based extraction)
#[async_trait]
impl EntityExtractor for &str {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        // Simple regex-based entity extraction for basic cases
        let mut entities = Vec::new();
        
        // Extract simple patterns like proper nouns (capitalized words)
        for word in text.split_whitespace() {
            if word.chars().next().is_some_and(|c| c.is_uppercase()) && word.len() > 2 {
                entities.push(Entity {
                    id: format!("entity_{}", entities.len()),
                    text: word.to_string(),
                    canonical_name: word.to_string(),
                    entity_type: "UNKNOWN".to_string(),
                    start_pos: 0, // Would need proper position tracking
                    end_pos: word.len(),
                    confidence: 0.5,
                    source_model: "regex".to_string(),
                    linked_id: None,
                    properties: std::collections::HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
    
    async fn extract_entities_with_confidence(&self, text: &str) -> Result<Vec<(Entity, f32)>> {
        let entities = self.extract_entities(text).await?;
        Ok(entities.into_iter().map(|e| {
            let confidence = e.confidence;
            (e, confidence)
        }).collect())
    }
}

/// Implementation of EntityExtractor for String
#[async_trait]
impl EntityExtractor for String {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        self.as_str().extract_entities(text).await
    }
    
    async fn extract_entities_with_confidence(&self, text: &str) -> Result<Vec<(Entity, f32)>> {
        self.as_str().extract_entities_with_confidence(text).await
    }
}

/// Implementation for AdvancedEntityExtractor
#[async_trait]
impl EntityExtractor for AdvancedEntityExtractor {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        self.extract_entities(text).await
    }
    
    async fn extract_entities_with_confidence(&self, text: &str) -> Result<Vec<(Entity, f32)>> {
        let entities = self.extract_entities(text).await?;
        Ok(entities.into_iter().map(|e| {
            let confidence = e.confidence;
            (e, confidence)
        }).collect())
    }
}