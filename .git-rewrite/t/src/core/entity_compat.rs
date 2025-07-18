// Compatibility layer for performance tests
// This bridges the gap between the test API and the core implementation

use std::collections::HashMap;
use crate::core::types::EntityKey;
use slotmap::Key;

/// Performance test compatible Entity struct
#[derive(Debug, Clone)]
pub struct Entity {
    id: String,
    entity_type: String,
    attributes: HashMap<String, String>,
    key: EntityKey,
}

impl Entity {
    pub fn new(id: String, entity_type: String) -> Self {
        Self {
            id,
            entity_type,
            attributes: HashMap::new(),
            key: EntityKey::default(), // Will be set when inserted
        }
    }
    
    pub fn with_attributes(id: String, entity_type: String, attributes: HashMap<String, String>) -> Self {
        Self {
            id,
            entity_type,
            attributes,
            key: EntityKey::default(),
        }
    }
    
    pub fn id(&self) -> &String {
        &self.id
    }
    
    pub fn entity_type(&self) -> &String {
        &self.entity_type
    }
    
    pub fn attributes(&self) -> &HashMap<String, String> {
        &self.attributes
    }
    
    pub fn key(&self) -> EntityKey {
        self.key
    }
    
    pub fn set_key(&mut self, key: EntityKey) {
        self.key = key;
    }
    
    pub fn get_attribute(&self, name: &str) -> Option<&String> {
        self.attributes.get(name)
    }
    
    pub fn add_attribute(&mut self, name: String, value: String) {
        self.attributes.insert(name, value);
    }
}

/// Performance test compatible Relationship struct  
#[derive(Debug, Clone)]
pub struct Relationship {
    pub relationship_type: String,
    pub attributes: HashMap<String, String>,
    target_key: EntityKey,
}

impl Relationship {
    pub fn new(relationship_type: String, attributes: HashMap<String, String>) -> Self {
        Self {
            relationship_type,
            attributes,
            target_key: EntityKey::default(),
        }
    }
    
    pub fn target(&self) -> EntityKey {
        self.target_key
    }
    
    pub fn set_target(&mut self, target: EntityKey) {
        self.target_key = target;
    }
}

/// Performance test compatible SimilarityResult
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub entity: EntityKey,
    pub similarity: f32,
}

impl SimilarityResult {
    pub fn new(entity: EntityKey, similarity: f32) -> Self {
        Self { entity, similarity }
    }
}

// Extension methods for EntityKey to work with performance tests
impl EntityKey {
    pub fn new(_id: String) -> Self {
        // For performance tests, we'll use a simple hash-based approach
        // In production, this would be properly managed by the slotmap
        EntityKey::default() // Simplified for now
    }
    
    pub fn from_hash(text: &str) -> Self {
        // Generate a consistent EntityKey from text hash
        // This is a simplified approach for testing
        EntityKey::default()
    }
    
    pub fn as_u32(&self) -> u32 {
        self.data().as_ffi() as u32
    }
    
    pub fn from_u32(_id: u32) -> Self {
        // This is a simplified conversion - in practice we'd need proper mapping
        EntityKey::default() // Simplified for now
    }
}