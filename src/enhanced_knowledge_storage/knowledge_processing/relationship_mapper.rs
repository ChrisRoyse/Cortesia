//! Advanced Relationship Mapper
//! 
//! Extracts complex relationships between entities using instruction-tuned
//! language models, going far beyond simple "is/has" patterns.

use std::sync::Arc;
use std::time::Instant;
use serde_json;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
};

/// Advanced relationship mapper using instruction-tuned language models
pub struct AdvancedRelationshipMapper {
    model_manager: Arc<ModelResourceManager>,
    config: RelationshipExtractionConfig,
}

/// Configuration for relationship extraction
#[derive(Debug, Clone)]
pub struct RelationshipExtractionConfig {
    pub model_id: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub min_confidence: f32,
    pub max_relationships_per_chunk: usize,
    pub enable_temporal_analysis: bool,
    pub enable_causal_analysis: bool,
    pub relationship_strength_threshold: f32,
}

impl Default for RelationshipExtractionConfig {
    fn default() -> Self {
        Self {
            model_id: "smollm2_360m".to_string(),
            max_tokens: 1500,
            temperature: 0.2, // Slightly higher than entity extraction for more creativity
            min_confidence: 0.6,
            max_relationships_per_chunk: 30,
            enable_temporal_analysis: true,
            enable_causal_analysis: true,
            relationship_strength_threshold: 0.5,
        }
    }
}

impl AdvancedRelationshipMapper {
    /// Create new relationship mapper with model manager
    pub fn new(model_manager: Arc<ModelResourceManager>, config: RelationshipExtractionConfig) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Extract complex relationships from text with entities
    pub async fn extract_complex_relationships(
        &self,
        text: &str,
        entities: &[ContextualEntity],
    ) -> KnowledgeProcessingResult2<Vec<ComplexRelationship>> {
        let _start_time = Instant::now();
        
        // Create relationship extraction prompt
        let prompt = self.create_relationship_extraction_prompt(text, entities);
        
        // Create processing task
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        
        // Process with model
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        // Parse the model response
        let relationships = self.parse_relationship_response(&result.output, text, entities)?;
        
        // Filter and enhance relationships
        let filtered_relationships = self.filter_and_enhance_relationships(relationships, text);
        
        Ok(filtered_relationships)
    }
    
    /// Extract relationships from multiple text-entity pairs
    pub async fn extract_relationships_batch(
        &self,
        text_entity_pairs: &[(&str, &[ContextualEntity])],
    ) -> KnowledgeProcessingResult2<Vec<Vec<ComplexRelationship>>> {
        let mut results = Vec::new();
        
        for (text, entities) in text_entity_pairs {
            let relationships = self.extract_complex_relationships(text, entities).await?;
            results.push(relationships);
        }
        
        Ok(results)
    }
    
    /// Create instruction prompt for relationship extraction
    fn create_relationship_extraction_prompt(&self, text: &str, entities: &[ContextualEntity]) -> String {
        let entity_list = entities
            .iter()
            .map(|e| format!("- {} ({})", e.name, e.entity_type.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        
        format!(
            r#"You are an expert relationship extraction system. Given the text and list of entities below, identify all meaningful relationships between the entities.

For each relationship, provide:
1. Source entity (must be from the entity list)
2. Relationship type (be specific: created_by, works_for, located_in, causes, results_in, enables, prevents, before, after, during, similar_to, opposite_of, influences, etc.)
3. Target entity (must be from the entity list)
4. Context (the specific sentence or phrase that establishes this relationship)
5. Confidence (0.0 to 1.0)
6. Supporting evidence (key phrases that support this relationship)
7. Relationship strength (0.0 to 1.0, how strong/important is this connection)

Include these types of relationships:
- Direct relationships (X created Y, A works for B, C is located in D)
- Causal relationships (X causes Y, A results in B, C enables D)
- Temporal relationships (X happened before Y, A occurred during B)
- Hierarchical relationships (X is part of Y, A contains B)
- Semantic relationships (X is similar to Y, A influences B)
- Implicit relationships (X and Y are discussed together, suggesting connection)

Entities found in the text:
{entity_list}

Return results as a JSON array with this exact format:
[
  {{
    "source": "EntityName1",
    "relationship_type": "specific_relationship",
    "target": "EntityName2",
    "context": "sentence establishing the relationship",
    "confidence": 0.85,
    "supporting_evidence": ["phrase1", "phrase2"],
    "relationship_strength": 0.9
  }}
]

Text to analyze:
{text}

JSON Response:"#,
            entity_list = entity_list,
            text = text
        )
    }
    
    /// Parse relationship extraction response from model
    fn parse_relationship_response(
        &self,
        response: &str,
        original_text: &str,
        entities: &[ContextualEntity],
    ) -> KnowledgeProcessingResult2<Vec<ComplexRelationship>> {
        // Find JSON in response
        let json_start = response.find('[').unwrap_or(0);
        let json_end = response.rfind(']').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        // Parse JSON response
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| KnowledgeProcessingError::JsonError(e))?;
        
        let mut relationships = Vec::new();
        
        if let Some(array) = parsed.as_array() {
            for item in array {
                if let Some(relationship) = self.parse_single_relationship(item, original_text, entities)? {
                    relationships.push(relationship);
                }
            }
        }
        
        Ok(relationships)
    }
    
    /// Parse a single relationship from JSON
    fn parse_single_relationship(
        &self,
        item: &serde_json::Value,
        _original_text: &str,
        entities: &[ContextualEntity],
    ) -> KnowledgeProcessingResult2<Option<ComplexRelationship>> {
        let source = item["source"].as_str().unwrap_or("").to_string();
        let relationship_type_str = item["relationship_type"].as_str().unwrap_or("related_to");
        let target = item["target"].as_str().unwrap_or("").to_string();
        let context = item["context"].as_str().unwrap_or("").to_string();
        let confidence = item["confidence"].as_f64().unwrap_or(0.5) as f32;
        let relationship_strength = item["relationship_strength"].as_f64().unwrap_or(0.5) as f32;
        
        // Parse supporting evidence
        let supporting_evidence = if let Some(evidence_array) = item["supporting_evidence"].as_array() {
            evidence_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        // Skip if below minimum confidence or strength, or if entities are missing
        if confidence < self.config.min_confidence 
            || relationship_strength < self.config.relationship_strength_threshold
            || source.is_empty() 
            || target.is_empty() {
            return Ok(None);
        }
        
        // Validate that entities exist in our entity list
        let source_exists = entities.iter().any(|e| e.name == source);
        let target_exists = entities.iter().any(|e| e.name == target);
        
        if !source_exists || !target_exists {
            return Ok(None);
        }
        
        // Extract temporal information if enabled
        let temporal_info = if self.config.enable_temporal_analysis {
            self.extract_temporal_info(&context)
        } else {
            None
        };
        
        let relationship = ComplexRelationship {
            source,
            predicate: RelationshipType::from_string(relationship_type_str),
            target,
            context,
            confidence,
            supporting_evidence,
            relationship_strength,
            temporal_info,
        };
        
        Ok(Some(relationship))
    }
    
    /// Extract temporal information from context
    fn extract_temporal_info(&self, context: &str) -> Option<TemporalInfo> {
        let lower_context = context.to_lowercase();
        
        // Simple temporal pattern matching (could be enhanced with NLP)
        let has_time_markers = lower_context.contains("before") 
            || lower_context.contains("after")
            || lower_context.contains("during")
            || lower_context.contains("in 1")
            || lower_context.contains("in 2")
            || lower_context.contains("since")
            || lower_context.contains("until")
            || lower_context.contains("from")
            || lower_context.contains("to");
        
        if has_time_markers {
            Some(TemporalInfo {
                start_time: None, // Could be extracted with more sophisticated parsing
                end_time: None,
                duration: None,
                frequency: None,
            })
        } else {
            None
        }
    }
    
    /// Filter and enhance extracted relationships
    fn filter_and_enhance_relationships(
        &self,
        mut relationships: Vec<ComplexRelationship>,
        text: &str,
    ) -> Vec<ComplexRelationship> {
        // Remove self-relationships
        relationships.retain(|r| r.source != r.target);
        
        // Remove duplicates (same source, predicate, target)
        relationships.dedup_by(|a, b| {
            a.source == b.source && 
            a.predicate == b.predicate && 
            a.target == b.target
        });
        
        // Sort by confidence * relationship_strength (highest first)
        relationships.sort_by(|a, b| {
            let score_a = a.confidence * a.relationship_strength;
            let score_b = b.confidence * b.relationship_strength;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit to max relationships per chunk
        relationships.truncate(self.config.max_relationships_per_chunk);
        
        // Enhance relationships with additional analysis
        for relationship in &mut relationships {
            self.enhance_relationship(relationship, text);
        }
        
        relationships
    }
    
    /// Enhance relationship with additional analysis
    fn enhance_relationship(&self, relationship: &mut ComplexRelationship, text: &str) {
        // Add more supporting evidence if context appears multiple times
        let context_occurrences = text.matches(&relationship.context).count();
        if context_occurrences > 1 {
            relationship.supporting_evidence.push(
                format!("Context appears {} times in text", context_occurrences)
            );
        }
        
        // Boost confidence for well-supported relationships
        if relationship.supporting_evidence.len() > 2 {
            relationship.confidence = (relationship.confidence * 1.1).min(1.0);
        }
        
        // Analyze relationship direction and add reverse if bidirectional
        if self.is_bidirectional_relationship(&relationship.predicate) {
            // Note: In a full implementation, we might add reverse relationships
            // For now, we just note the bidirectional nature
        }
    }
    
    /// Check if a relationship type is typically bidirectional
    fn is_bidirectional_relationship(&self, predicate: &RelationshipType) -> bool {
        matches!(
            predicate,
            RelationshipType::SimilarTo |
            RelationshipType::RelatedTo |
            RelationshipType::During
        )
    }
    
    /// Validate extracted relationships
    pub fn validate_relationships(
        &self,
        relationships: &[ComplexRelationship],
        entities: &[ContextualEntity],
        original_text: &str,
    ) -> Vec<bool> {
        relationships
            .iter()
            .map(|rel| {
                // Check if source and target entities exist
                let source_exists = entities.iter().any(|e| e.name == rel.source);
                let target_exists = entities.iter().any(|e| e.name == rel.target);
                
                // Check if context appears in text
                let context_exists = rel.context.is_empty() ||
                    original_text.contains(&rel.context) ||
                    original_text.to_lowercase().contains(&rel.context.to_lowercase());
                
                // Check confidence and strength thresholds
                let confidence_ok = rel.confidence >= self.config.min_confidence;
                let strength_ok = rel.relationship_strength >= self.config.relationship_strength_threshold;
                
                source_exists && target_exists && context_exists && confidence_ok && strength_ok
            })
            .collect()
    }
    
    /// Get relationship extraction statistics
    pub fn get_extraction_stats(&self, relationships: &[ComplexRelationship]) -> RelationshipExtractionStats {
        let mut type_counts = std::collections::HashMap::new();
        let mut total_confidence = 0.0;
        let mut total_strength = 0.0;
        
        for rel in relationships {
            *type_counts.entry(rel.predicate.to_string()).or_insert(0) += 1;
            total_confidence += rel.confidence;
            total_strength += rel.relationship_strength;
        }
        
        RelationshipExtractionStats {
            total_relationships: relationships.len(),
            average_confidence: if relationships.is_empty() { 0.0 } else { total_confidence / relationships.len() as f32 },
            average_strength: if relationships.is_empty() { 0.0 } else { total_strength / relationships.len() as f32 },
            type_distribution: type_counts,
            high_confidence_count: relationships.iter().filter(|r| r.confidence > 0.8).count(),
            strong_relationships_count: relationships.iter().filter(|r| r.relationship_strength > 0.8).count(),
            temporal_relationships_count: relationships.iter().filter(|r| r.temporal_info.is_some()).count(),
        }
    }
}

/// Statistics about relationship extraction
#[derive(Debug, Clone)]
pub struct RelationshipExtractionStats {
    pub total_relationships: usize,
    pub average_confidence: f32,
    pub average_strength: f32,
    pub type_distribution: std::collections::HashMap<String, usize>,
    pub high_confidence_count: usize,
    pub strong_relationships_count: usize,
    pub temporal_relationships_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_relationship_mapper_creation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let mapper_config = RelationshipExtractionConfig::default();
        
        let mapper = AdvancedRelationshipMapper::new(model_manager, mapper_config);
        
        assert_eq!(mapper.config.model_id, "smollm2_360m");
        assert!(mapper.config.enable_temporal_analysis);
        assert!(mapper.config.enable_causal_analysis);
    }
    
    #[test]
    fn test_relationship_type_parsing() {
        assert_eq!(RelationshipType::from_string("created_by"), RelationshipType::CreatedBy);
        assert_eq!(RelationshipType::from_string("causes"), RelationshipType::Causes);
        assert_eq!(RelationshipType::from_string("custom_relationship"), 
                   RelationshipType::Custom("custom_relationship".to_string()));
    }
    
    #[test]
    fn test_temporal_info_extraction() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let mapper_config = RelationshipExtractionConfig::default();
        
        let mapper = AdvancedRelationshipMapper::new(model_manager, mapper_config);
        
        let context_with_time = "Einstein developed relativity before 1920";
        let temporal_info = mapper.extract_temporal_info(context_with_time);
        assert!(temporal_info.is_some());
        
        let context_without_time = "Einstein was a physicist";
        let temporal_info = mapper.extract_temporal_info(context_without_time);
        assert!(temporal_info.is_none());
    }
    
    #[test]
    fn test_bidirectional_relationship_detection() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let mapper_config = RelationshipExtractionConfig::default();
        
        let mapper = AdvancedRelationshipMapper::new(model_manager, mapper_config);
        
        assert!(mapper.is_bidirectional_relationship(&RelationshipType::SimilarTo));
        assert!(mapper.is_bidirectional_relationship(&RelationshipType::RelatedTo));
        assert!(!mapper.is_bidirectional_relationship(&RelationshipType::CreatedBy));
    }
    
    #[test]
    fn test_relationship_validation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let mapper_config = RelationshipExtractionConfig::default();
        
        let mapper = AdvancedRelationshipMapper::new(model_manager, mapper_config);
        
        let entities = vec![
            ContextualEntity {
                name: "Einstein".to_string(),
                entity_type: EntityType::Person,
                context: "".to_string(),
                confidence: 0.9,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 0,
            },
            ContextualEntity {
                name: "Relativity".to_string(),
                entity_type: EntityType::Concept,
                context: "".to_string(),
                confidence: 0.8,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 0,
            },
        ];
        
        let relationships = vec![
            ComplexRelationship {
                source: "Einstein".to_string(),
                predicate: RelationshipType::CreatedBy,
                target: "Relativity".to_string(),
                context: "Einstein developed relativity".to_string(),
                confidence: 0.9,
                supporting_evidence: vec!["developed".to_string()],
                relationship_strength: 0.8,
                temporal_info: None,
            },
            ComplexRelationship {
                source: "Newton".to_string(), // Entity doesn't exist
                predicate: RelationshipType::CreatedBy,
                target: "Physics".to_string(), // Entity doesn't exist
                context: "".to_string(),
                confidence: 0.7,
                supporting_evidence: vec![],
                relationship_strength: 0.6,
                temporal_info: None,
            },
        ];
        
        let text = "Einstein developed relativity theory.";
        let validation = mapper.validate_relationships(&relationships, &entities, text);
        assert_eq!(validation, vec![true, false]);
    }
}