// Enhanced entity data structures with string interning support

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
        writeln!(f, "Interned Entity Data Stats:")?;
        writeln!(f, "  Entities: {}", self.entity_count)?;
        writeln!(f, "  Total memory: {} KB", self.total_memory_bytes / 1024)?;
        writeln!(f, "  Properties memory: {} KB", self.properties_memory_bytes / 1024)?;
        writeln!(f, "  Embeddings memory: {} KB", self.embedding_memory_bytes / 1024)?;
        writeln!(f, "  Avg properties/entity: {:.1}", self.avg_properties_per_entity)?;
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
            sample.insert("entity_key".to_string(), serde_json::Value::String(format!("{key:?}")));
            
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::string_interner::StringInterner;
    use std::collections::HashMap;

    #[test]
    fn test_interned_entity_data_new() {
        let interner = StringInterner::new();
        let embedding = vec![1.0, 2.0, 3.0];
        let entity = InternedEntityData::new(&interner, 42, embedding.clone());
        
        assert_eq!(entity.type_id, 42);
        assert_eq!(entity.embedding, embedding);
        assert_eq!(entity.category, InternedString::EMPTY);
        assert_eq!(entity.description, InternedString::EMPTY);
        assert!(entity.tags.is_empty());
        assert_eq!(entity.properties.len(), 0);
    }

    #[test]
    fn test_interned_entity_data_from_properties_empty() {
        let interner = StringInterner::new();
        let properties = HashMap::new();
        let embedding = vec![1.0, 2.0];
        
        let entity = InternedEntityData::from_properties(&interner, 10, &properties, embedding.clone());
        
        assert_eq!(entity.type_id, 10);
        assert_eq!(entity.embedding, embedding);
        assert_eq!(entity.properties.len(), 0);
    }

    #[test]
    fn test_interned_entity_data_from_properties_with_data() {
        let interner = StringInterner::new();
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), "John".to_string());
        properties.insert("age".to_string(), "30".to_string());
        
        let embedding = vec![0.1, 0.2, 0.3];
        let entity = InternedEntityData::from_properties(&interner, 5, &properties, embedding.clone());
        
        assert_eq!(entity.type_id, 5);
        assert_eq!(entity.embedding, embedding);
        assert_eq!(entity.properties.len(), 2);
        assert_eq!(entity.get_property(&interner, "name"), Some("John".to_string()));
        assert_eq!(entity.get_property(&interner, "age"), Some("30".to_string()));
    }

    #[test]
    fn test_interned_entity_data_add_property_empty_strings() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        // Test adding empty key and value
        entity.add_property(&interner, "", "");
        assert_eq!(entity.get_property(&interner, ""), Some("".to_string()));
        
        // Test adding empty value with non-empty key
        entity.add_property(&interner, "empty_val", "");
        assert_eq!(entity.get_property(&interner, "empty_val"), Some("".to_string()));
        
        // Test adding empty key with non-empty value
        entity.add_property(&interner, "", "some_value");
        assert_eq!(entity.get_property(&interner, ""), Some("some_value".to_string()));
    }

    #[test]
    fn test_interned_entity_data_add_property_long_strings() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        let long_key = "a".repeat(1000);
        let long_value = "b".repeat(2000);
        
        entity.add_property(&interner, &long_key, &long_value);
        assert_eq!(entity.get_property(&interner, &long_key), Some(long_value));
    }

    #[test]
    fn test_interned_entity_data_add_property_overwrites() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        entity.add_property(&interner, "key", "value1");
        entity.add_property(&interner, "key", "value2");
        
        assert_eq!(entity.get_property(&interner, "key"), Some("value2".to_string()));
        assert_eq!(entity.properties.len(), 1);
    }

    #[test]
    fn test_interned_entity_data_set_category_empty() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        entity.set_category(&interner, "");
        assert_ne!(entity.category, InternedString::EMPTY);
        assert_eq!(interner.get(entity.category), Some("".to_string()));
    }

    #[test]
    fn test_interned_entity_data_set_category_long() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        let long_category = "category_".repeat(500);
        entity.set_category(&interner, &long_category);
        assert_eq!(interner.get(entity.category), Some(long_category));
    }

    #[test]
    fn test_interned_entity_data_set_description_empty() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        entity.set_description(&interner, "");
        assert_ne!(entity.description, InternedString::EMPTY);
        assert_eq!(interner.get(entity.description), Some("".to_string()));
    }

    #[test]
    fn test_interned_entity_data_set_description_long() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        let long_description = "This is a very long description that tests the string interning functionality with large text content. ".repeat(100);
        entity.set_description(&interner, &long_description);
        assert_eq!(interner.get(entity.description), Some(long_description));
    }

    #[test]
    fn test_interned_entity_data_add_tag_empty() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        entity.add_tag(&interner, "");
        assert_eq!(entity.tags.len(), 1);
        assert_eq!(entity.get_tags(&interner), vec!["".to_string()]);
    }

    #[test]
    fn test_interned_entity_data_add_tag_duplicates() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        entity.add_tag(&interner, "tag1");
        entity.add_tag(&interner, "tag1");
        entity.add_tag(&interner, "tag2");
        entity.add_tag(&interner, "tag1");
        
        assert_eq!(entity.tags.len(), 2);
        let tags = entity.get_tags(&interner);
        assert!(tags.contains(&"tag1".to_string()));
        assert!(tags.contains(&"tag2".to_string()));
    }

    #[test]
    fn test_interned_entity_data_add_tag_long_strings() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        let long_tag = "tag_".repeat(1000);
        entity.add_tag(&interner, &long_tag);
        
        assert_eq!(entity.tags.len(), 1);
        assert_eq!(entity.get_tags(&interner), vec![long_tag]);
    }

    #[test]
    fn test_interned_entity_data_memory_usage_calculation() {
        let interner = StringInterner::new();
        let embedding = vec![1.0; 100]; // 100 f32 values
        let mut entity = InternedEntityData::new(&interner, 1, embedding);
        
        let base_memory = entity.memory_usage();
        
        // Add properties and verify memory increases
        entity.add_property(&interner, "key1", "value1");
        entity.add_property(&interner, "key2", "value2");
        let with_props_memory = entity.memory_usage();
        assert!(with_props_memory > base_memory);
        
        // Add tags and verify memory increases
        entity.add_tag(&interner, "tag1");
        entity.add_tag(&interner, "tag2");
        let with_tags_memory = entity.memory_usage();
        assert!(with_tags_memory > with_props_memory);
        
        // Verify embedding size is calculated correctly
        let expected_embedding_size = 100 * std::mem::size_of::<f32>();
        assert!(entity.memory_usage() >= expected_embedding_size);
    }

    #[test]
    fn test_interned_entity_data_to_json_empty() {
        let interner = StringInterner::new();
        let entity = InternedEntityData::new(&interner, 42, vec![1.0, 2.0]);
        
        let json = entity.to_json(&interner).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed["type_id"], 42);
        assert!(parsed["properties"].is_object());
        assert!(!parsed.as_object().unwrap().contains_key("category"));
        assert!(!parsed.as_object().unwrap().contains_key("description"));
        assert_eq!(parsed["tags"], serde_json::Value::Array(vec![]));
        assert_eq!(parsed["embedding_preview"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_interned_entity_data_to_json_with_data() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        entity.add_property(&interner, "name", "test");
        entity.set_category(&interner, "test_category");
        entity.set_description(&interner, "test description");
        entity.add_tag(&interner, "tag1");
        entity.add_tag(&interner, "tag2");
        
        let json = entity.to_json(&interner).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed["category"], "test_category");
        assert_eq!(parsed["description"], "test description");
        assert_eq!(parsed["tags"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["embedding_preview"].as_array().unwrap().len(), 5); // Only first 5
    }

    #[test]
    fn test_interned_relationship_new() {
        let interner = StringInterner::new();
        let from_key = EntityKey::from_raw_parts(1, 0);
        let to_key = EntityKey::from_raw_parts(2, 0);
        let rel = InternedRelationship::new(&interner, from_key, to_key, "RELATES_TO", 0.8);
        
        assert_eq!(rel.from, from_key);
        assert_eq!(rel.to, to_key);
        assert_eq!(rel.weight, 0.8);
        assert_eq!(rel.get_relationship_type(&interner), Some("RELATES_TO".to_string()));
        assert_eq!(rel.properties.len(), 0);
    }

    #[test]
    fn test_interned_relationship_add_property() {
        let interner = StringInterner::new();
        let from_key = EntityKey::from_raw_parts(1, 0);
        let to_key = EntityKey::from_raw_parts(2, 0);
        let mut rel = InternedRelationship::new(&interner, from_key, to_key, "RELATES_TO", 0.5);
        
        rel.add_property(&interner, "strength", "high");
        rel.add_property(&interner, "confidence", "0.95");
        
        assert_eq!(rel.properties.len(), 2);
        assert_eq!(rel.properties.get(&interner, "strength"), Some("high".to_string()));
        assert_eq!(rel.properties.get(&interner, "confidence"), Some("0.95".to_string()));
    }

    #[test]
    fn test_interned_relationship_memory_usage() {
        let interner = StringInterner::new();
        let from_key = EntityKey::from_raw_parts(1, 0);
        let to_key = EntityKey::from_raw_parts(2, 0);
        let mut rel = InternedRelationship::new(&interner, from_key, to_key, "RELATES_TO", 0.5);
        
        let base_memory = rel.memory_usage();
        
        rel.add_property(&interner, "key1", "value1");
        rel.add_property(&interner, "key2", "value2");
        
        let with_props_memory = rel.memory_usage();
        assert!(with_props_memory > base_memory);
    }

    #[test]
    fn test_interned_entity_collection_new() {
        let collection = InternedEntityCollection::new();
        
        assert_eq!(collection.entities.len(), 0);
        assert_eq!(collection.relationships.len(), 0);
        assert_eq!(collection.interner.stats().unique_strings, 0);
    }

    #[test]
    fn test_interned_entity_collection_add_entity_empty_properties() {
        let mut collection = InternedEntityCollection::new();
        let properties = HashMap::new();
        let key = EntityKey::from_raw_parts(1, 0);
        
        collection.add_entity(key, 42, &properties, vec![1.0, 2.0]);
        
        assert_eq!(collection.entities.len(), 1);
        assert!(collection.entities.contains_key(&key));
        
        let entity = collection.entities.get(&key).unwrap();
        assert_eq!(entity.type_id, 42);
        assert_eq!(entity.properties.len(), 0);
    }

    #[test]
    fn test_interned_entity_collection_add_entity_with_properties() {
        let mut collection = InternedEntityCollection::new();
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), "Alice".to_string());
        properties.insert("age".to_string(), "25".to_string());
        
        let key = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key, 10, &properties, vec![0.1, 0.2]);
        
        assert_eq!(collection.entities.len(), 1);
        let entity = collection.entities.get(&key).unwrap();
        assert_eq!(entity.properties.len(), 2);
        assert_eq!(entity.get_property(&collection.interner, "name"), Some("Alice".to_string()));
        assert_eq!(entity.get_property(&collection.interner, "age"), Some("25".to_string()));
    }

    #[test]
    fn test_interned_entity_collection_add_relationship_empty_properties() {
        let mut collection = InternedEntityCollection::new();
        let properties = HashMap::new();
        
        let from_key = EntityKey::from_raw_parts(1, 0);
        let to_key = EntityKey::from_raw_parts(2, 0);
        collection.add_relationship(from_key, to_key, "CONNECTS", 0.7, &properties);
        
        assert_eq!(collection.relationships.len(), 1);
        let rel = &collection.relationships[0];
        assert_eq!(rel.from, from_key);
        assert_eq!(rel.to, to_key);
        assert_eq!(rel.weight, 0.7);
        assert_eq!(rel.get_relationship_type(&collection.interner), Some("CONNECTS".to_string()));
        assert_eq!(rel.properties.len(), 0);
    }

    #[test]
    fn test_interned_entity_collection_add_relationship_with_properties() {
        let mut collection = InternedEntityCollection::new();
        let mut properties = HashMap::new();
        properties.insert("type".to_string(), "strong".to_string());
        properties.insert("created".to_string(), "2023-01-01".to_string());
        
        let from_key = EntityKey::from_raw_parts(1, 0);
        let to_key = EntityKey::from_raw_parts(2, 0);
        collection.add_relationship(from_key, to_key, "LINKS", 0.9, &properties);
        
        assert_eq!(collection.relationships.len(), 1);
        let rel = &collection.relationships[0];
        assert_eq!(rel.properties.len(), 2);
        assert_eq!(rel.properties.get(&collection.interner, "type"), Some("strong".to_string()));
        assert_eq!(rel.properties.get(&collection.interner, "created"), Some("2023-01-01".to_string()));
    }

    #[test]
    fn test_interned_entity_collection_stats_empty() {
        let collection = InternedEntityCollection::new();
        let stats = collection.stats();
        
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.total_memory_bytes, 0);
        assert_eq!(stats.properties_memory_bytes, 0);
        assert_eq!(stats.embedding_memory_bytes, 0);
        assert_eq!(stats.avg_properties_per_entity, 0.0);
    }

    #[test]
    fn test_interned_entity_collection_stats_with_entities() {
        let mut collection = InternedEntityCollection::new();
        
        // Add entities with different numbers of properties
        let mut props1 = HashMap::new();
        props1.insert("name".to_string(), "Entity1".to_string());
        let key1 = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key1, 10, &props1, vec![1.0, 2.0]);
        
        let mut props2 = HashMap::new();
        props2.insert("name".to_string(), "Entity2".to_string());
        props2.insert("type".to_string(), "TypeA".to_string());
        props2.insert("value".to_string(), "100".to_string());
        let key2 = EntityKey::from_raw_parts(2, 0);
        collection.add_entity(key2, 20, &props2, vec![3.0, 4.0, 5.0]);
        
        let stats = collection.stats();
        
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.avg_properties_per_entity, 2.0); // (1 + 3) / 2
        assert!(stats.total_memory_bytes > 0);
        assert!(stats.properties_memory_bytes > 0);
        assert!(stats.embedding_memory_bytes > 0);
        
        // Embedding memory should be at least the size of f32s
        let expected_embedding_memory = (2 + 3) * std::mem::size_of::<f32>();
        assert!(stats.embedding_memory_bytes >= expected_embedding_memory);
    }

    #[test]
    fn test_interned_entity_collection_find_by_property_not_found() {
        let mut collection = InternedEntityCollection::new();
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), "Alice".to_string());
        let key = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key, 10, &properties, vec![]);
        
        let results = collection.find_by_property("name", "Bob");
        assert!(results.is_empty());
        
        let results = collection.find_by_property("age", "25");
        assert!(results.is_empty());
    }

    #[test]
    fn test_interned_entity_collection_find_by_property_found() {
        let mut collection = InternedEntityCollection::new();
        
        let mut props1 = HashMap::new();
        props1.insert("name".to_string(), "Alice".to_string());
        props1.insert("type".to_string(), "Person".to_string());
        let key1 = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key1, 10, &props1, vec![]);
        
        let mut props2 = HashMap::new();
        props2.insert("name".to_string(), "Bob".to_string());
        props2.insert("type".to_string(), "Person".to_string());
        let key2 = EntityKey::from_raw_parts(2, 0);
        collection.add_entity(key2, 10, &props2, vec![]);
        
        let mut props3 = HashMap::new();
        props3.insert("name".to_string(), "Company".to_string());
        props3.insert("type".to_string(), "Organization".to_string());
        let key3 = EntityKey::from_raw_parts(3, 0);
        collection.add_entity(key3, 20, &props3, vec![]);
        
        let person_results = collection.find_by_property("type", "Person");
        assert_eq!(person_results.len(), 2);
        assert!(person_results.contains(&key1));
        assert!(person_results.contains(&key2));
        
        let alice_results = collection.find_by_property("name", "Alice");
        assert_eq!(alice_results.len(), 1);
        assert_eq!(alice_results[0], key1);
    }

    #[test]
    fn test_interned_entity_collection_find_by_tag_not_found() {
        let mut collection = InternedEntityCollection::new();
        let key = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key, 10, &HashMap::new(), vec![]);
        
        let results = collection.find_by_tag("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_interned_entity_collection_find_by_tag_found() {
        let mut collection = InternedEntityCollection::new();
        let key1 = EntityKey::from_raw_parts(1, 0);
        let key2 = EntityKey::from_raw_parts(2, 0);
        let key3 = EntityKey::from_raw_parts(3, 0);
        collection.add_entity(key1, 10, &HashMap::new(), vec![]);
        collection.add_entity(key2, 20, &HashMap::new(), vec![]);
        collection.add_entity(key3, 30, &HashMap::new(), vec![]);
        
        // Add tags to entities
        if let Some(entity1) = collection.entities.get_mut(&key1) {
            entity1.add_tag(&collection.interner, "important");
            entity1.add_tag(&collection.interner, "active");
        }
        
        if let Some(entity2) = collection.entities.get_mut(&key2) {
            entity2.add_tag(&collection.interner, "important");
        }
        
        let important_results = collection.find_by_tag("important");
        assert_eq!(important_results.len(), 2);
        assert!(important_results.contains(&key1));
        assert!(important_results.contains(&key2));
        
        let active_results = collection.find_by_tag("active");
        assert_eq!(active_results.len(), 1);
        assert_eq!(active_results[0], key1);
    }

    #[test]
    fn test_interned_entity_collection_get_all_property_keys_empty() {
        let collection = InternedEntityCollection::new();
        let keys = collection.get_all_property_keys();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_interned_entity_collection_get_all_property_keys() {
        let mut collection = InternedEntityCollection::new();
        
        let mut props1 = HashMap::new();
        props1.insert("name".to_string(), "Alice".to_string());
        props1.insert("age".to_string(), "25".to_string());
        let key1 = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key1, 10, &props1, vec![]);
        
        let mut props2 = HashMap::new();
        props2.insert("name".to_string(), "Bob".to_string());
        props2.insert("city".to_string(), "NYC".to_string());
        props2.insert("country".to_string(), "USA".to_string());
        let key2 = EntityKey::from_raw_parts(2, 0);
        collection.add_entity(key2, 20, &props2, vec![]);
        
        let keys = collection.get_all_property_keys();
        assert_eq!(keys.len(), 4);
        assert!(keys.contains(&"name".to_string()));
        assert!(keys.contains(&"age".to_string()));
        assert!(keys.contains(&"city".to_string()));
        assert!(keys.contains(&"country".to_string()));
    }

    #[test]
    fn test_interned_entity_collection_export_sample_json_empty() {
        let collection = InternedEntityCollection::new();
        let json = collection.export_sample_json(10).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        assert!(parsed["entities"].as_array().unwrap().is_empty());
        assert!(parsed["stats"].as_str().unwrap().contains("Entities: 0"));
    }

    #[test]
    fn test_interned_entity_collection_export_sample_json_with_limit() {
        let mut collection = InternedEntityCollection::new();
        
        // Add 5 entities
        for i in 1..=5 {
            let mut props = HashMap::new();
            props.insert("id".to_string(), i.to_string());
            collection.add_entity(EntityKey::new(i.to_string()), 10, &props, vec![i as f32]);
        }
        
        let json = collection.export_sample_json(3).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        let entities = parsed["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 3); // Limited to 3
        assert!(parsed["stats"].as_str().unwrap().contains("Entities: 5")); // But stats show all 5
    }

    #[test]
    fn test_string_interning_optimization() {
        let mut collection = InternedEntityCollection::new();
        
        // Add multiple entities with the same property values to test string interning
        for i in 1..=10 {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "Person".to_string()); // Same value for all
            props.insert("status".to_string(), "Active".to_string()); // Same value for all
            props.insert("id".to_string(), i.to_string()); // Unique values
            collection.add_entity(EntityKey::new(i.to_string()), 10, &props, vec![]);
        }
        
        let stats = collection.stats();
        let interner_stats = stats.interner_stats;
        
        // We should have fewer unique strings than total property insertions
        // due to string interning of repeated values
        assert!(interner_stats.unique_strings < 30); // 10 entities * 3 props = 30 without interning
        
        // Verify that "Person" and "Active" are properly interned
        assert!(interner_stats.unique_strings >= 12); // At least: "type", "status", "id", "Person", "Active", "1"-"10"
    }

    #[test]
    fn test_memory_optimization_with_large_strings() {
        let mut collection = InternedEntityCollection::new();
        
        let large_description = "A".repeat(10000);
        let large_category = "B".repeat(5000);
        
        // Add multiple entities with the same large strings
        for i in 1..=5 {
            collection.add_entity(EntityKey::new(i.to_string()), 10, &HashMap::new(), vec![]);
            
            if let Some(entity) = collection.entities.get_mut(&EntityKey::new(i.to_string())) {
                entity.set_description(&collection.interner, &large_description);
                entity.set_category(&collection.interner, &large_category);
            }
        }
        
        let stats = collection.stats();
        let interner_stats = stats.interner_stats;
        
        // With string interning, we should have only 2 large strings stored
        // instead of 10 (5 descriptions + 5 categories)
        assert_eq!(interner_stats.unique_strings, 2);
        
        // Total memory should be much less than without interning
        let expected_without_interning = 5 * (large_description.len() + large_category.len());
        assert!(interner_stats.total_memory_bytes < expected_without_interning as u32);
    }

    #[test]
    fn test_data_transformation_and_interning_algorithms() {
        let mut collection = InternedEntityCollection::new();
        
        // Test that string interning works correctly with special characters and Unicode
        let unicode_string = "Hello 世界 🌍 café résumé naïve";
        let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~";
        let whitespace = "  \t\n\r  ";
        
        let mut props = HashMap::new();
        props.insert("unicode".to_string(), unicode_string.to_string());
        props.insert("special".to_string(), special_chars.to_string());
        props.insert("whitespace".to_string(), whitespace.to_string());
        
        let key = EntityKey::from_raw_parts(1, 0);
        collection.add_entity(key, 10, &props, vec![]);
        
        // Verify that complex strings are stored and retrieved correctly
        let entity = collection.entities.get(&key).unwrap();
        assert_eq!(entity.get_property(&collection.interner, "unicode"), Some(unicode_string.to_string()));
        assert_eq!(entity.get_property(&collection.interner, "special"), Some(special_chars.to_string()));
        assert_eq!(entity.get_property(&collection.interner, "whitespace"), Some(whitespace.to_string()));
    }

    #[test]
    fn test_private_data_structures_isolation() {
        let interner = StringInterner::new();
        let mut entity = InternedEntityData::new(&interner, 1, vec![]);
        
        // Test that we can access private fields within the same module
        assert_eq!(entity.type_id, 1);
        assert_eq!(entity.category, InternedString::EMPTY);
        assert_eq!(entity.description, InternedString::EMPTY);
        assert!(entity.tags.is_empty());
        
        // Test that private data can be modified through public methods
        entity.set_category(&interner, "test_category");
        entity.set_description(&interner, "test_description");
        entity.add_tag(&interner, "test_tag");
        
        // Verify changes are reflected in private fields
        assert_ne!(entity.category, InternedString::EMPTY);
        assert_ne!(entity.description, InternedString::EMPTY);
        assert_eq!(entity.tags.len(), 1);
    }
}