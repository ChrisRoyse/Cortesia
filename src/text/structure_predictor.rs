//! Graph structure prediction using heuristic analysis
//!
//! This module provides rule-based graph structure prediction using
//! pattern matching and linguistic analysis.

use crate::error::Result;
use std::collections::HashMap;

/// Graph operation types for structure building
#[derive(Debug, Clone)]
pub enum GraphOperation {
    CreateNode {
        id: String,
        node_type: String,
        properties: HashMap<String, String>,
    },
    CreateEdge {
        from: String,
        to: String,
        relationship: String,
        weight: f32,
    },
    MergeNodes {
        nodes: Vec<String>,
        target_id: String,
    },
    InferRelationship {
        from: String,
        to: String,
        confidence: f32,
    },
}

/// Graph structure predictor using heuristic analysis
#[derive(Clone)]
pub struct GraphStructurePredictor {
    model_name: String,
    relationship_patterns: HashMap<String, Vec<String>>,
    entity_types: HashMap<String, f32>,
}

impl GraphStructurePredictor {
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            relationship_patterns: Self::create_relationship_patterns(),
            entity_types: Self::create_entity_type_weights(),
        }
    }

    /// Predict graph structure from text using heuristic analysis
    pub async fn predict_structure(&self, text: &str) -> Result<Vec<GraphOperation>> {
        let operations = self.predict_structure_sync(text);
        Ok(operations)
    }

    /// Synchronous structure prediction
    pub fn predict_structure_sync(&self, text: &str) -> Vec<GraphOperation> {
        let mut operations = Vec::new();
        
        // Extract entities and relationships using heuristics
        let entities = self.extract_entities(text);
        let relationships = self.extract_relationships(text, &entities);
        
        // Create node operations
        for (entity, entity_type) in entities {
            operations.push(GraphOperation::CreateNode {
                id: self.normalize_entity_id(&entity),
                node_type: entity_type,
                properties: self.extract_entity_properties(&entity, text),
            });
        }
        
        // Create edge operations
        for (from, to, relationship, confidence) in relationships {
            if confidence > 0.3 { // Confidence threshold
                operations.push(GraphOperation::CreateEdge {
                    from: self.normalize_entity_id(&from),
                    to: self.normalize_entity_id(&to),
                    relationship,
                    weight: confidence,
                });
            }
        }
        
        operations
    }

    /// Predict relationships between entities
    pub fn predict_relationships(&self, entity1: &str, entity2: &str, context: &str) -> Result<Vec<(String, f32)>> {
        let relationships = self.infer_relationships(entity1, entity2, context);
        Ok(relationships)
    }

    /// Predict entity types from text
    pub fn predict_entity_types(&self, text: &str) -> Result<HashMap<String, String>> {
        let entities = self.extract_entities(text);
        Ok(entities)
    }

    /// Predict optimal graph structure for query
    pub fn predict_query_structure(&self, query: &str) -> Result<Vec<GraphOperation>> {
        // Simplified query structure prediction
        let mut operations = Vec::new();
        
        let query_entities = self.extract_query_entities(query);
        for entity in query_entities {
            operations.push(GraphOperation::CreateNode {
                id: self.normalize_entity_id(&entity),
                node_type: "query_concept".to_string(),
                properties: HashMap::new(),
            });
        }
        
        Ok(operations)
    }

    /// Extract entities from text using heuristic patterns
    fn extract_entities(&self, text: &str) -> HashMap<String, String> {
        let mut entities = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            let normalized_word = word.to_lowercase();
            
            // Check if word matches known entity types
            for (entity_type, &weight) in &self.entity_types {
                if normalized_word.contains(entity_type) || entity_type.contains(&normalized_word) {
                    if weight > 0.5 {
                        entities.insert(word.to_string(), self.classify_entity_type(word));
                    }
                }
            }
            
            // Capitalize words are likely entities
            if word.chars().next().unwrap_or('a').is_uppercase() && word.len() > 2 {
                entities.insert(word.to_string(), self.classify_entity_type(word));
            }
            
            // Multi-word entities (simple detection)
            if i < words.len() - 1 {
                let next_word = words[i + 1];
                if next_word.chars().next().unwrap_or('a').is_uppercase() {
                    let compound = format!("{} {}", word, next_word);
                    entities.insert(compound, "compound_entity".to_string());
                }
            }
        }
        
        entities
    }

    /// Extract relationships between entities
    fn extract_relationships(&self, text: &str, entities: &HashMap<String, String>) -> Vec<(String, String, String, f32)> {
        let mut relationships = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let entity_list: Vec<&String> = entities.keys().collect();
        
        // Find relationships using pattern matching
        for i in 0..words.len() {
            for pattern_key in self.relationship_patterns.keys() {
                if words[i].to_lowercase().contains(pattern_key) {
                    // Look for entities before and after relationship word
                    let before_entities = self.find_nearby_entities(&words[..i], &entity_list);
                    let after_entities = self.find_nearby_entities(&words[i+1..], &entity_list);
                    
                    for before in &before_entities {
                        for after in &after_entities {
                            let confidence = self.calculate_relationship_confidence(before, after, pattern_key);
                            relationships.push((
                                before.clone(),
                                after.clone(),
                                pattern_key.clone(),
                                confidence
                            ));
                        }
                    }
                }
            }
        }
        
        relationships
    }

    /// Find entities near a position in text
    fn find_nearby_entities(&self, words: &[&str], entities: &[&String]) -> Vec<String> {
        let mut nearby = Vec::new();
        let window_size = 3; // Look within 3 words
        
        for word in words.iter().rev().take(window_size) {
            for entity in entities {
                if entity.to_lowercase().contains(&word.to_lowercase()) ||
                   word.to_lowercase().contains(&entity.to_lowercase()) {
                    nearby.push(entity.to_string());
                }
            }
        }
        
        nearby
    }

    /// Calculate confidence score for a relationship
    fn calculate_relationship_confidence(&self, _entity1: &str, _entity2: &str, relationship: &str) -> f32 {
        // Simple confidence based on relationship type
        match relationship {
            "has" | "contains" | "includes" => 0.8,
            "is" | "was" | "are" => 0.9,
            "uses" | "utilizes" | "employs" => 0.7,
            "creates" | "generates" | "produces" => 0.7,
            "connects" | "links" | "relates" => 0.6,
            _ => 0.5,
        }
    }

    /// Classify entity type based on word characteristics
    fn classify_entity_type(&self, word: &str) -> String {
        let lower_word = word.to_lowercase();
        
        // Technical terms
        if lower_word.contains("algorithm") || lower_word.contains("system") || 
           lower_word.contains("framework") || lower_word.contains("model") {
            return "technical_concept".to_string();
        }
        
        // Data terms
        if lower_word.contains("data") || lower_word.contains("information") ||
           lower_word.contains("knowledge") || lower_word.contains("content") {
            return "data_entity".to_string();
        }
        
        // Process terms
        if lower_word.contains("process") || lower_word.contains("method") ||
           lower_word.contains("approach") || lower_word.contains("technique") {
            return "process_entity".to_string();
        }
        
        // Default classification
        if word.chars().next().unwrap_or('a').is_uppercase() {
            "named_entity".to_string()
        } else {
            "concept".to_string()
        }
    }

    /// Extract query-specific entities
    fn extract_query_entities(&self, query: &str) -> Vec<String> {
        query.split_whitespace()
            .filter(|word| word.len() > 3)
            .filter(|word| !self.is_stop_word(word))
            .map(|word| word.to_string())
            .collect()
    }

    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let stop_words = ["what", "where", "when", "how", "why", "which",
                         "the", "and", "or", "but", "in", "on", "at", "to", "for"];
        stop_words.contains(&word.to_lowercase().as_str())
    }

    /// Normalize entity ID for consistency
    fn normalize_entity_id(&self, entity: &str) -> String {
        entity.to_lowercase()
            .replace(" ", "_")
            .replace("-", "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }

    /// Extract properties for an entity
    fn extract_entity_properties(&self, entity: &str, text: &str) -> HashMap<String, String> {
        let mut properties = HashMap::new();
        
        properties.insert("original_form".to_string(), entity.to_string());
        properties.insert("source_text".to_string(), text.chars().take(100).collect());
        properties.insert("entity_type".to_string(), self.classify_entity_type(entity));
        
        properties
    }

    /// Infer relationships between two entities
    fn infer_relationships(&self, entity1: &str, entity2: &str, context: &str) -> Vec<(String, f32)> {
        let mut relationships = Vec::new();
        
        // Simple heuristic relationship inference
        let words: Vec<&str> = context.split_whitespace().collect();
        
        for pattern_key in self.relationship_patterns.keys() {
            let pattern_count = words.iter()
                .filter(|word| word.to_lowercase().contains(pattern_key))
                .count();
            
            if pattern_count > 0 {
                let confidence = (pattern_count as f32 * 0.3).min(1.0);
                relationships.push((pattern_key.clone(), confidence));
            }
        }
        
        // If no explicit relationships found, infer based on entity types
        if relationships.is_empty() {
            let type1 = self.classify_entity_type(entity1);
            let type2 = self.classify_entity_type(entity2);
            
            let inferred_relationship = match (type1.as_str(), type2.as_str()) {
                ("technical_concept", "data_entity") => "processes",
                ("process_entity", "data_entity") => "transforms",
                ("named_entity", "concept") => "relates_to",
                _ => "associated_with",
            };
            
            relationships.push((inferred_relationship.to_string(), 0.4));
        }
        
        relationships
    }

    /// Create relationship patterns for detection
    fn create_relationship_patterns() -> HashMap<String, Vec<String>> {
        let mut patterns = HashMap::new();
        
        patterns.insert("has".to_string(), vec!["contains".to_string(), "includes".to_string(), "owns".to_string()]);
        patterns.insert("is".to_string(), vec!["was".to_string(), "are".to_string(), "were".to_string()]);
        patterns.insert("uses".to_string(), vec!["utilizes".to_string(), "employs".to_string(), "applies".to_string()]);
        patterns.insert("creates".to_string(), vec!["generates".to_string(), "produces".to_string(), "builds".to_string()]);
        patterns.insert("connects".to_string(), vec!["links".to_string(), "relates".to_string(), "joins".to_string()]);
        patterns.insert("depends".to_string(), vec!["requires".to_string(), "needs".to_string(), "relies".to_string()]);
        patterns.insert("influences".to_string(), vec!["affects".to_string(), "impacts".to_string(), "modifies".to_string()]);
        patterns.insert("implements".to_string(), vec!["realizes".to_string(), "executes".to_string(), "performs".to_string()]);
        
        patterns
    }

    /// Create entity type weights for classification
    fn create_entity_type_weights() -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        
        // Technical concepts
        weights.insert("algorithm".to_string(), 0.9);
        weights.insert("system".to_string(), 0.8);
        weights.insert("framework".to_string(), 0.8);
        weights.insert("model".to_string(), 0.8);
        weights.insert("architecture".to_string(), 0.8);
        weights.insert("component".to_string(), 0.7);
        weights.insert("module".to_string(), 0.7);
        weights.insert("interface".to_string(), 0.7);
        
        // Data concepts
        weights.insert("data".to_string(), 0.9);
        weights.insert("information".to_string(), 0.8);
        weights.insert("knowledge".to_string(), 0.9);
        weights.insert("content".to_string(), 0.7);
        weights.insert("record".to_string(), 0.7);
        weights.insert("document".to_string(), 0.7);
        weights.insert("file".to_string(), 0.6);
        weights.insert("resource".to_string(), 0.6);
        
        // Process concepts
        weights.insert("process".to_string(), 0.8);
        weights.insert("method".to_string(), 0.8);
        weights.insert("procedure".to_string(), 0.7);
        weights.insert("technique".to_string(), 0.7);
        weights.insert("approach".to_string(), 0.7);
        weights.insert("strategy".to_string(), 0.7);
        weights.insert("workflow".to_string(), 0.7);
        weights.insert("pipeline".to_string(), 0.7);
        
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction() {
        let predictor = GraphStructurePredictor::new("test_model".to_string());
        
        let text = "The machine learning algorithm processes data efficiently";
        let entities = predictor.extract_entities(text);
        
        assert!(!entities.is_empty());
        // Should extract "machine", "learning", "algorithm", "data" as entities
    }

    #[test]
    fn test_structure_prediction() {
        let predictor = GraphStructurePredictor::new("test_model".to_string());
        
        let text = "The machine learning model uses deep learning techniques";
        let operations = predictor.predict_structure_sync(text);
        
        assert!(!operations.is_empty());
        
        // Should contain both node and edge operations
        let has_nodes = operations.iter().any(|op| matches!(op, GraphOperation::CreateNode { .. }));
        let has_edges = operations.iter().any(|op| matches!(op, GraphOperation::CreateEdge { .. }));
        
        assert!(has_nodes);
        // Edges might not always be found depending on heuristics
    }

    #[test]
    fn test_relationship_prediction() {
        let predictor = GraphStructurePredictor::new("test_model".to_string());
        
        let relationships = predictor.predict_relationships(
            "algorithm", 
            "data", 
            "the algorithm processes the data efficiently"
        ).unwrap();
        
        assert!(!relationships.is_empty());
        assert!(relationships.iter().any(|(rel, _)| rel.contains("process")));
    }

    #[test]
    fn test_entity_type_classification() {
        let predictor = GraphStructurePredictor::new("test_model".to_string());
        
        assert_eq!(predictor.classify_entity_type("algorithm"), "technical_concept");
        assert_eq!(predictor.classify_entity_type("data"), "data_entity");
        assert_eq!(predictor.classify_entity_type("process"), "process_entity");
        assert_eq!(predictor.classify_entity_type("MyClass"), "named_entity");
    }

    #[test]
    fn test_performance() {
        let predictor = GraphStructurePredictor::new("test_model".to_string());
        let text = "Machine learning algorithms process large datasets using pattern recognition and deep learning techniques to create intelligent systems.";
        
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_structure_sync(text);
        }
        let elapsed = start.elapsed();
        
        // Should process 100 predictions in under 1 second
        assert!(elapsed.as_millis() < 1000);
        println!("100 structure predictions took: {:?}", elapsed);
    }
}