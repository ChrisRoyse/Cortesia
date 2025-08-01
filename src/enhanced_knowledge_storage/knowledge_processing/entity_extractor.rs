//! Advanced Entity Extractor
//! 
//! Uses SmolLM instruction-tuned models for sophisticated entity recognition
//! that goes far beyond simple pattern matching.

use std::sync::Arc;
use std::time::Instant;
use serde_json;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
};

/// Advanced entity extractor using instruction-tuned language models
pub struct AdvancedEntityExtractor {
    model_manager: Arc<ModelResourceManager>,
    config: EntityExtractionConfig,
}

/// Configuration for entity extraction
#[derive(Debug, Clone)]
pub struct EntityExtractionConfig {
    pub model_id: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub min_confidence: f32,
    pub max_entities_per_chunk: usize,
    pub enable_context_expansion: bool,
    pub context_window_size: usize,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            model_id: "smollm2_360m".to_string(),
            max_tokens: 1000,
            temperature: 0.1, // Low temperature for consistent extraction
            min_confidence: 0.7,
            max_entities_per_chunk: 20,
            enable_context_expansion: true,
            context_window_size: 200, // characters before/after
        }
    }
}

impl AdvancedEntityExtractor {
    /// Create new entity extractor with model manager
    pub fn new(model_manager: Arc<ModelResourceManager>, config: EntityExtractionConfig) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Extract entities from text with rich context
    pub async fn extract_entities_with_context(&self, text: &str) -> KnowledgeProcessingResult2<Vec<ContextualEntity>> {
        let _start_time = Instant::now();
        
        // Prepare the instruction prompt for entity extraction
        let prompt = self.create_entity_extraction_prompt(text);
        
        // Create processing task
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        
        // Process with model
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        // Parse the model response
        let entities = self.parse_entity_response(&result.output, text)?;
        
        // Filter by confidence and post-process
        let filtered_entities = self.filter_and_enhance_entities(entities, text);
        
        Ok(filtered_entities)
    }
    
    /// Extract entities from multiple chunks efficiently
    pub async fn extract_entities_batch(&self, chunks: &[&str]) -> KnowledgeProcessingResult2<Vec<Vec<ContextualEntity>>> {
        let mut results = Vec::new();
        
        for chunk in chunks {
            let entities = self.extract_entities_with_context(chunk).await?;
            results.push(entities);
        }
        
        Ok(results)
    }
    
    /// Create instruction prompt for entity extraction
    fn create_entity_extraction_prompt(&self, text: &str) -> String {
        format!(
            r#"You are an expert entity extraction system. Extract all important entities from the following text. For each entity, provide:

1. Entity name (the exact text from the document)
2. Entity type (Person, Organization, Location, Concept, Event, Technology, Method, Measurement, TimeExpression, or Other)
3. Context (the sentence or phrase where it appears)
4. Confidence level (0.0 to 1.0, where 1.0 is completely certain)

Be thorough but accurate. Include entities that are:
- People (names, roles, titles)
- Organizations (companies, institutions, groups)
- Locations (places, countries, cities, addresses)
- Concepts (ideas, theories, principles)
- Events (occurrences, meetings, incidents)
- Technologies (software, hardware, systems, tools)
- Methods (approaches, techniques, processes)
- Measurements (quantities, metrics, values)
- Time expressions (dates, periods, durations)

Return results as a JSON array with this exact format:
[
  {{
    "name": "EntityName",
    "type": "EntityType",
    "context": "surrounding sentence or phrase",
    "confidence": 0.95
  }}
]

Text to analyze:
{text}

JSON Response:"#,
            text = text
        )
    }
    
    /// Parse entity extraction response from model
    fn parse_entity_response(&self, response: &str, original_text: &str) -> KnowledgeProcessingResult2<Vec<ContextualEntity>> {
        // Find JSON in response (model might include additional text)
        let json_start = response.find('[').unwrap_or(0);
        let json_end = response.rfind(']').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        // Parse JSON response
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| KnowledgeProcessingError::JsonError(e))?;
        
        let mut entities = Vec::new();
        
        if let Some(array) = parsed.as_array() {
            for item in array {
                if let Some(entity) = self.parse_single_entity(item, original_text)? {
                    entities.push(entity);
                }
            }
        }
        
        Ok(entities)
    }
    
    /// Parse a single entity from JSON
    fn parse_single_entity(&self, item: &serde_json::Value, original_text: &str) -> KnowledgeProcessingResult2<Option<ContextualEntity>> {
        let name = item["name"].as_str().unwrap_or("").to_string();
        let type_str = item["type"].as_str().unwrap_or("Other");
        let context = item["context"].as_str().unwrap_or("").to_string();
        let confidence = item["confidence"].as_f64().unwrap_or(0.5) as f32;
        
        // Skip if below minimum confidence or empty name
        if confidence < self.config.min_confidence || name.is_empty() {
            return Ok(None);
        }
        
        // Find text span if possible
        let span = self.find_text_span(&name, original_text);
        
        let entity = ContextualEntity {
            name,
            entity_type: EntityType::from_string(type_str),
            context,
            confidence,
            span,
            attributes: std::collections::HashMap::new(),
            extracted_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        Ok(Some(entity))
    }
    
    /// Find the text span of an entity in the original text
    fn find_text_span(&self, entity_name: &str, text: &str) -> Option<TextSpan> {
        if let Some(start) = text.find(entity_name) {
            Some(TextSpan {
                start,
                end: start + entity_name.len(),
            })
        } else {
            // Try case-insensitive search
            let lower_text = text.to_lowercase();
            let lower_entity = entity_name.to_lowercase();
            if let Some(start) = lower_text.find(&lower_entity) {
                Some(TextSpan {
                    start,
                    end: start + entity_name.len(),
                })
            } else {
                None
            }
        }
    }
    
    /// Filter and enhance extracted entities
    fn filter_and_enhance_entities(&self, mut entities: Vec<ContextualEntity>, text: &str) -> Vec<ContextualEntity> {
        // Remove duplicates (same name and type)
        entities.dedup_by(|a, b| a.name == b.name && a.entity_type == b.entity_type);
        
        // Sort by confidence (highest first)
        entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to max entities per chunk
        entities.truncate(self.config.max_entities_per_chunk);
        
        // Enhance with additional attributes
        for entity in &mut entities {
            self.enhance_entity_attributes(entity, text);
        }
        
        entities
    }
    
    /// Enhance entity with additional attributes
    fn enhance_entity_attributes(&self, entity: &mut ContextualEntity, text: &str) {
        // Add word count
        entity.attributes.insert(
            "word_count".to_string(),
            entity.name.split_whitespace().count().to_string(),
        );
        
        // Add character count
        entity.attributes.insert(
            "char_count".to_string(),
            entity.name.chars().count().to_string(),
        );
        
        // Add frequency in text
        let frequency = text.matches(&entity.name).count() + 
                       text.to_lowercase().matches(&entity.name.to_lowercase()).count();
        entity.attributes.insert(
            "frequency".to_string(),
            frequency.to_string(),
        );
        
        // Add entity category based on type
        let category = match entity.entity_type {
            EntityType::Person => "Human",
            EntityType::Organization => "Institution",
            EntityType::Location => "Geographic",
            EntityType::Concept => "Abstract",
            EntityType::Event => "Temporal",
            EntityType::Technology => "Technical",
            EntityType::Method => "Procedural",
            EntityType::Measurement => "Quantitative",
            EntityType::TimeExpression => "Temporal",
            EntityType::Other(_) => "Unknown",
        };
        entity.attributes.insert("category".to_string(), category.to_string());
    }
    
    /// Validate extracted entities against original text
    pub fn validate_entities(&self, entities: &[ContextualEntity], original_text: &str) -> Vec<bool> {
        entities
            .iter()
            .map(|entity| {
                // Check if entity name appears in text
                let name_exists = original_text.contains(&entity.name) ||
                    original_text.to_lowercase().contains(&entity.name.to_lowercase());
                
                // Check if context appears in text
                let context_exists = entity.context.is_empty() ||
                    original_text.contains(&entity.context) ||
                    original_text.to_lowercase().contains(&entity.context.to_lowercase());
                
                // Check confidence threshold
                let confidence_ok = entity.confidence >= self.config.min_confidence;
                
                name_exists && context_exists && confidence_ok
            })
            .collect()
    }
    
    /// Get extraction statistics
    pub fn get_extraction_stats(&self, entities: &[ContextualEntity]) -> EntityExtractionStats {
        let mut type_counts = std::collections::HashMap::new();
        let mut total_confidence = 0.0;
        
        for entity in entities {
            *type_counts.entry(entity.entity_type.to_string()).or_insert(0) += 1;
            total_confidence += entity.confidence;
        }
        
        EntityExtractionStats {
            total_entities: entities.len(),
            average_confidence: if entities.is_empty() { 0.0 } else { total_confidence / entities.len() as f32 },
            type_distribution: type_counts,
            high_confidence_count: entities.iter().filter(|e| e.confidence > 0.9).count(),
            medium_confidence_count: entities.iter().filter(|e| e.confidence > 0.7 && e.confidence <= 0.9).count(),
            low_confidence_count: entities.iter().filter(|e| e.confidence <= 0.7).count(),
        }
    }
}

/// Statistics about entity extraction
#[derive(Debug, Clone)]
pub struct EntityExtractionStats {
    pub total_entities: usize,
    pub average_confidence: f32,
    pub type_distribution: std::collections::HashMap<String, usize>,
    pub high_confidence_count: usize,
    pub medium_confidence_count: usize,
    pub low_confidence_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_entity_extractor_creation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let extractor_config = EntityExtractionConfig::default();
        
        let extractor = AdvancedEntityExtractor::new(model_manager, extractor_config);
        
        // Test that extractor is created successfully
        assert_eq!(extractor.config.model_id, "smollm2_360m");
    }
    
    #[tokio::test]
    async fn test_entity_extraction_prompt_creation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let extractor_config = EntityExtractionConfig::default();
        
        let extractor = AdvancedEntityExtractor::new(model_manager, extractor_config);
        
        let text = "Albert Einstein developed the theory of relativity.";
        let prompt = extractor.create_entity_extraction_prompt(text);
        
        assert!(prompt.contains("entity extraction"));
        assert!(prompt.contains(text));
        assert!(prompt.contains("JSON"));
    }
    
    #[test]
    fn test_text_span_finding() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let extractor_config = EntityExtractionConfig::default();
        
        let extractor = AdvancedEntityExtractor::new(model_manager, extractor_config);
        
        let text = "Albert Einstein was a physicist.";
        let span = extractor.find_text_span("Einstein", text);
        
        assert!(span.is_some());
        let span = span.unwrap();
        assert_eq!(span.start, 7);
        assert_eq!(span.end, 15);
    }
    
    #[test]
    fn test_entity_type_parsing() {
        assert_eq!(EntityType::from_string("Person"), EntityType::Person);
        assert_eq!(EntityType::from_string("organization"), EntityType::Organization);
        assert_eq!(EntityType::from_string("unknown"), EntityType::Other("unknown".to_string()));
    }
    
    #[test]
    fn test_entity_validation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let extractor_config = EntityExtractionConfig::default();
        
        let extractor = AdvancedEntityExtractor::new(model_manager, extractor_config);
        
        let text = "Albert Einstein developed relativity theory.";
        let entities = vec![
            ContextualEntity {
                name: "Einstein".to_string(),
                entity_type: EntityType::Person,
                context: "Albert Einstein developed".to_string(),
                confidence: 0.95,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 0,
            },
            ContextualEntity {
                name: "Newton".to_string(), // Not in text
                entity_type: EntityType::Person,
                context: "".to_string(),
                confidence: 0.8,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 0,
            },
        ];
        
        let validation = extractor.validate_entities(&entities, text);
        assert_eq!(validation, vec![true, false]);
    }
}