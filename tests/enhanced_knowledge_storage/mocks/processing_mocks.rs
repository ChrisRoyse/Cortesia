//! Knowledge Processing Mocks
//! 
//! Mock implementations for knowledge processing components including
//! text analysis, entity extraction, and relationship detection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock text processor implementation
pub struct MockTextProcessor {
    processing_results: Arc<Mutex<HashMap<String, ProcessingResult>>>,
    call_log: Arc<Mutex<Vec<String>>>,
    processing_delay_ms: u64,
}

impl MockTextProcessor {
    pub fn new() -> Self {
        Self {
            processing_results: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            processing_delay_ms: 0,
        }
    }
    
    pub fn with_delay(delay_ms: u64) -> Self {
        Self {
            processing_results: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            processing_delay_ms: delay_ms,
        }
    }
    
    pub fn process_text(&self, text: &str) -> ProcessingResult {
        self.call_log.lock().unwrap().push(format!("process_text: {} chars", text.len()));
        
        // Simulate processing delay
        if self.processing_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.processing_delay_ms));
        }
        
        // Return mock processing result
        ProcessingResult {
            entities: vec!["entity1".to_string(), "entity2".to_string()],
            relationships: vec![
                Relationship {
                    subject: "entity1".to_string(),
                    predicate: "relates_to".to_string(),
                    object: "entity2".to_string(),
                    confidence: 0.9,
                }
            ],
            themes: vec!["theme1".to_string(), "theme2".to_string()],
            quality_score: 0.85,
            processing_time_ms: self.processing_delay_ms,
        }
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
    
    pub fn clear_call_log(&self) {
        self.call_log.lock().unwrap().clear();
    }
}

/// Mock entity extractor implementation
pub struct MockEntityExtractor {
    extracted_entities: HashMap<String, Vec<Entity>>,
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockEntityExtractor {
    pub fn new() -> Self {
        Self {
            extracted_entities: HashMap::new(),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn extract_entities(&self, text: &str) -> Vec<Entity> {
        self.call_log.lock().unwrap().push(format!("extract_entities: {} chars", text.len()));
        
        // Mock entity extraction
        vec![
            Entity {
                name: "MockEntity1".to_string(),
                entity_type: EntityType::Person,
                confidence: 0.9,
                span: (0, 10),
            },
            Entity {
                name: "MockEntity2".to_string(),
                entity_type: EntityType::Organization,
                confidence: 0.8,
                span: (15, 25),
            },
        ]
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Mock relationship detector implementation
pub struct MockRelationshipDetector {
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockRelationshipDetector {
    pub fn new() -> Self {
        Self {
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn detect_relationships(&self, entities: &[Entity]) -> Vec<Relationship> {
        self.call_log.lock().unwrap().push(format!("detect_relationships: {} entities", entities.len()));
        
        // Mock relationship detection
        if entities.len() >= 2 {
            vec![
                Relationship {
                    subject: entities[0].name.clone(),
                    predicate: "interacts_with".to_string(),
                    object: entities[1].name.clone(),
                    confidence: 0.75,
                }
            ]
        } else {
            vec![]
        }
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Processing result structure for testing
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub entities: Vec<String>,
    pub relationships: Vec<Relationship>,
    pub themes: Vec<String>,
    pub quality_score: f64,
    pub processing_time_ms: u64,
}

/// Entity structure for testing
#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub confidence: f64,
    pub span: (usize, usize),
}

/// Entity types for testing
#[derive(Debug, Clone)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Concept,
    Other,
}

/// Relationship structure for testing
#[derive(Debug, Clone)]
pub struct Relationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Helper functions for creating mock processing components
pub fn create_mock_text_processor() -> MockTextProcessor {
    MockTextProcessor::new()
}

pub fn create_mock_text_processor_with_delay(delay_ms: u64) -> MockTextProcessor {
    MockTextProcessor::with_delay(delay_ms)
}

pub fn create_mock_entity_extractor() -> MockEntityExtractor {
    MockEntityExtractor::new()
}

pub fn create_mock_relationship_detector() -> MockRelationshipDetector {
    MockRelationshipDetector::new()
}

/// Test helper for setting up processing mocks
pub fn setup_processing_mocks() -> (MockTextProcessor, MockEntityExtractor, MockRelationshipDetector) {
    (
        create_mock_text_processor(),
        create_mock_entity_extractor(),
        create_mock_relationship_detector(),
    )
}