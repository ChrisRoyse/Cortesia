// Enhanced entity data structures with string interning support
// Phase 4.3: String interning integration for memory optimization

use crate::core::types::EntityKey;
use crate::storage::string_interner::{StringInterner, InternedString, InternedProperties};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Entity data with interned string properties for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternedEntityData {
    pub type_id: u16,
    pub properties: InternedProperties,
    pub embedding: Vec<f32>,
    
    // Additional metadata using interned strings
    pub category: InternedString,
    pub description: InternedString,
    pub tags: Vec<InternedString>,
}

impl InternedEntityData {
    /// Create new entity data with string interning
    pub fn new(_interner: &StringInterner, type_id: u16, embedding: Vec<f32>) -> Self {
        Self {
            type_id,
            properties: InternedProperties::new(),
            embedding,
            category: InternedString::EMPTY,
            description: InternedString::EMPTY,
            tags: Vec::new(),
        }
    }
    
    /// Create from regular property map
    pub fn from_properties(
        interner: &StringInterner, 
        type_id: u16, 
        properties: &HashMap<String, String>,
        embedding: Vec<f32>
    ) -> Self {
        let mut interned_props = InternedProperties::new();
        for (key, value) in properties {
            interned_props.insert(interner, key, value);
        }
        
        Self {
            type_id,
            properties: interned_props,
            embedding,
            category: InternedString::EMPTY,
            description: InternedString::EMPTY,
            tags: Vec::new(),
        }
    }
    
    /// Add property with automatic interning
    pub fn add_property(&mut self, interner: &StringInterner, key: &str, value: &str) {
        self.properties.insert(interner, key, value);
    }
    
    /// Get property value
    pub fn get_property(&self, interner: &StringInterner, key: &str) -> Option<String> {
        self.properties.get(interner, key)
    }
    
    /// Set category
    pub fn set_category(&mut self, interner: &StringInterner, category: &str) {
        self.category = interner.intern(category);
    }
    
    /// Set description
    pub fn set_description(&mut self, interner: &StringInterner, description: &str) {
        self.description = interner.intern(description);
    }
    
    /// Add tag
    pub fn add_tag(&mut self, interner: &StringInterner, tag: &str) {
        let tag_id = interner.intern(tag);
        if !self.tags.contains(&tag_id) {
            self.tags.push(tag_id);
        }
    }
    
    /// Get all tags as strings
    pub fn get_tags(&self, interner: &StringInterner) -> Vec<String> {
        self.tags.iter()
            .filter_map(|&id| interner.get(id))
            .collect()
    }
    
    /// Calculate memory usage including interned strings overhead
    pub fn memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let embedding_size = self.embedding.len() * std::mem::size_of::<f32>();
        let properties_size = self.properties.memory_usage();
        let tags_size = self.tags.len() * std::mem::size_of::<InternedString>();
        
        base_size + embedding_size + properties_size + tags_size
    }
    
    /// Convert to JSON representation using string interner
    pub fn to_json(&self, interner: &StringInterner) -> Result<String, serde_json::Error> {
        let mut json_obj = serde_json::Map::new();
        
        json_obj.insert("type_id".to_string(), serde_json::Value::Number(self.type_id.into()));
        
        // Properties
        let props_json = self.properties.to_json(interner)?;
        let props_value: serde_json::Value = serde_json::from_str(&props_json)?;
        json_obj.insert("properties".to_string(), props_value);
        
        // Category
        if let Some(category) = interner.get(self.category) {
            json_obj.insert("category".to_string(), serde_json::Value::String(category));
        }
        
        // Description
        if let Some(description) = interner.get(self.description) {
            json_obj.insert("description".to_string(), serde_json::Value::String(description));
        }
        
        // Tags
        let tags: Vec<String> = self.get_tags(interner);
        json_obj.insert("tags".to_string(), serde_json::Value::Array(
            tags.into_iter().map(serde_json::Value::String).collect()
        ));
        
        // Embedding (first 5 values for brevity)
        let embedding_preview: Vec<serde_json::Value> = self.embedding.iter()
            .take(5)
            .map(|&x| serde_json::Value::Number(serde_json::Number::from_f64(x as f64).unwrap_or_else(|| serde_json::Number::from(0))))
            .collect();
        json_obj.insert("embedding_preview".to_string(), serde_json::Value::Array(embedding_preview));
        
        serde_json::to_string(&json_obj)
    }
}

/// Relationship with interned string properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternedRelationship {
    pub from: EntityKey,
    pub to: EntityKey,
    pub relationship_type: InternedString,
    pub weight: f32,
    pub properties: InternedProperties,
}

impl InternedRelationship {
    pub fn new(
        interner: &StringInterner,
        from: EntityKey,
        to: EntityKey,
        rel_type: &str,
        weight: f32
    ) -> Self {
        Self {
            from,
            to,
            relationship_type: interner.intern(rel_type),
            weight,
            properties: InternedProperties::new(),
        }
    }
    
    pub fn add_property(&mut self, interner: &StringInterner, key: &str, value: &str) {
        self.properties.insert(interner, key, value);
    }
    
    pub fn get_relationship_type(&self, interner: &StringInterner) -> Option<String> {
        interner.get(self.relationship_type)
    }
    
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.properties.memory_usage()
    }
}

/// Statistics for interned entity data
#[derive(Debug, Clone)]
pub struct InternedDataStats {
    pub entity_count: usize,
    pub total_memory_bytes: usize,
    pub properties_memory_bytes: usize,
    pub embedding_memory_bytes: usize,
    pub avg_properties_per_entity: f32,
    pub interner_stats: crate::storage::string_interner::InternerStats,
}

impl std::fmt::Display for InternedDataStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Interned Entity Data Stats:\n")?;
        write!(f, "  Entities: {}\n", self.entity_count)?;
        write!(f, "  Total memory: {} KB\n", self.total_memory_bytes / 1024)?;
        write!(f, "  Properties memory: {} KB\n", self.properties_memory_bytes / 1024)?;
        write!(f, "  Embeddings memory: {} KB\n", self.embedding_memory_bytes / 1024)?;
        write!(f, "  Avg properties/entity: {:.1}\n", self.avg_properties_per_entity)?;
        write!(f, "  {}", self.interner_stats)
    }
}

/// Collection of interned entities with shared string interner
#[derive(Debug)]
pub struct InternedEntityCollection {
    pub interner: StringInterner,
    pub entities: HashMap<EntityKey, InternedEntityData>,
    pub relationships: Vec<InternedRelationship>,
}

impl InternedEntityCollection {
    pub fn new() -> Self {
        Self {
            interner: StringInterner::new(),
            entities: HashMap::new(),
            relationships: Vec::new(),
        }
    }
    
    /// Add entity with automatic property interning
    pub fn add_entity(
        &mut self, 
        key: EntityKey, 
        type_id: u16,
        properties: &HashMap<String, String>,
        embedding: Vec<f32>
    ) {
        let entity = InternedEntityData::from_properties(&self.interner, type_id, properties, embedding);
        self.entities.insert(key, entity);
    }
    
    /// Add relationship with automatic string interning
    pub fn add_relationship(
        &mut self,
        from: EntityKey,
        to: EntityKey,
        rel_type: &str,
        weight: f32,
        properties: &HashMap<String, String>
    ) {
        let mut relationship = InternedRelationship::new(&self.interner, from, to, rel_type, weight);
        
        for (key, value) in properties {
            relationship.add_property(&self.interner, key, value);
        }
        
        self.relationships.push(relationship);
    }
    
    /// Get comprehensive statistics
    pub fn stats(&self) -> InternedDataStats {
        let entity_count = self.entities.len();
        let total_properties: usize = self.entities.values().map(|e| e.properties.len()).sum();
        
        let properties_memory: usize = self.entities.values().map(|e| e.properties.memory_usage()).sum();
        let embedding_memory: usize = self.entities.values()
            .map(|e| e.embedding.len() * std::mem::size_of::<f32>())
            .sum();
        let total_memory: usize = self.entities.values().map(|e| e.memory_usage()).sum();
        
        InternedDataStats {
            entity_count,
            total_memory_bytes: total_memory,
            properties_memory_bytes: properties_memory,
            embedding_memory_bytes: embedding_memory,
            avg_properties_per_entity: if entity_count > 0 { 
                total_properties as f32 / entity_count as f32 
            } else { 
                0.0 
            },
            interner_stats: self.interner.stats(),
        }
    }
    
    /// Find entities by property value
    pub fn find_by_property(&self, key: &str, value: &str) -> Vec<EntityKey> {
        self.entities.iter()
            .filter_map(|(entity_key, entity)| {
                if entity.get_property(&self.interner, key).as_deref() == Some(value) {
                    Some(*entity_key)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Find entities by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<EntityKey> {
        let tag_id = self.interner.intern(tag);
        
        self.entities.iter()
            .filter_map(|(entity_key, entity)| {
                if entity.tags.contains(&tag_id) {
                    Some(*entity_key)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get all unique property keys
    pub fn get_all_property_keys(&self) -> Vec<String> {
        let mut keys = std::collections::HashSet::new();
        
        for entity in self.entities.values() {
            for (key, _) in entity.properties.iter(&self.interner) {
                keys.insert(key);
            }
        }
        
        keys.into_iter().collect()
    }
    
    /// Export to JSON for analysis
    pub fn export_sample_json(&self, max_entities: usize) -> Result<String, serde_json::Error> {
        let mut samples = Vec::new();
        
        for (key, entity) in self.entities.iter().take(max_entities) {
            let mut sample = serde_json::Map::new();
            sample.insert("entity_key".to_string(), serde_json::Value::String(format!("{:?}", key)));
            
            let entity_json = entity.to_json(&self.interner)?;
            let entity_value: serde_json::Value = serde_json::from_str(&entity_json)?;
            sample.insert("data".to_string(), entity_value);
            
            samples.push(serde_json::Value::Object(sample));
        }
        
        let mut result = serde_json::Map::new();
        result.insert("entities".to_string(), serde_json::Value::Array(samples));
        result.insert("stats".to_string(), serde_json::Value::String(self.stats().to_string()));
        
        serde_json::to_string_pretty(&result)
    }
}

impl Default for InternedEntityCollection {
    fn default() -> Self {
        Self::new()
    }
}