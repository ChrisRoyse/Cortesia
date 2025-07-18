use crate::core::types::{EntityKey, EntityMeta, EntityData};
use slotmap::Key;
use crate::error::{GraphError, Result};
use ahash::AHashMap;

pub struct EntityStore {
    entities: AHashMap<EntityKey, EntityMeta>,
    properties: Vec<u8>,
    property_offsets: Vec<u32>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            entities: AHashMap::new(),
            properties: Vec::new(),
            property_offsets: Vec::new(),
        }
    }
    
    pub fn insert(&mut self, key: EntityKey, data: &EntityData) -> Result<EntityMeta> {
        let property_offset = self.properties.len() as u32;
        let property_bytes = data.properties.as_bytes();
        
        self.properties.extend_from_slice(property_bytes);
        self.property_offsets.push(property_offset);
        self.property_offsets.push(self.properties.len() as u32);
        
        let meta = EntityMeta {
            type_id: data.type_id,
            embedding_offset: 0, // Will be set by embedding store
            property_offset,
            degree: 0, // Will be updated as relationships are added
            last_accessed: std::time::Instant::now(),
        };
        
        self.entities.insert(key, meta);
        Ok(meta)
    }
    
    pub fn get(&self, key: EntityKey) -> Option<&EntityMeta> {
        self.entities.get(&key)
    }
    
    pub fn get_mut(&mut self, key: EntityKey) -> Option<&mut EntityMeta> {
        self.entities.get_mut(&key)
    }
    
    pub fn get_properties(&self, meta: &EntityMeta) -> Result<String> {
        let start = meta.property_offset as usize;
        let end = self.property_offsets
            .iter()
            .find(|&&offset| offset > meta.property_offset)
            .map(|&offset| offset as usize)
            .unwrap_or(self.properties.len());
        
        // Handle empty properties case
        if start == end {
            return Ok(String::new());
        }
        
        if start > self.properties.len() || end > self.properties.len() {
            return Err(GraphError::IndexCorruption);
        }
        
        String::from_utf8(self.properties[start..end].to_vec())
            .map_err(|_| GraphError::SerializationError("Invalid UTF-8 in properties".to_string()))
    }
    
    pub fn update_degree(&mut self, key: EntityKey, delta: i16) -> Result<()> {
        let meta = self.entities.get_mut(&key)
            .ok_or(GraphError::EntityNotFound { id: key.data().as_ffi() as u32 })?;
        
        let new_degree = (meta.degree as i16 + delta).max(0) as u16;
        meta.degree = new_degree;
        Ok(())
    }
    
    pub fn count(&self) -> usize {
        self.entities.len()
    }
    
    pub fn memory_usage(&self) -> usize {
        self.entities.capacity() * std::mem::size_of::<(EntityKey, EntityMeta)>() +
        self.properties.capacity() +
        self.property_offsets.capacity() * std::mem::size_of::<u32>()
    }
    
    /// Get the capacity of the store
    pub fn capacity(&self) -> usize {
        self.entities.capacity()
    }
    
    /// Add edge (not applicable - EntityStore stores entities, not edges)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation(
            "EntityStore stores entities, not edges. Use CSRGraph for edge storage.".to_string()
        ))
    }
    
    /// Update an entity
    pub fn update_entity(&mut self, key: EntityKey, data: &EntityData) -> Result<()> {
        // For full update, we would need to handle property storage
        // For now, update metadata only
        if let Some(meta) = self.entities.get_mut(&key) {
            meta.type_id = data.type_id;
            Ok(())
        } else {
            Err(GraphError::EntityKeyNotFound { key })
        }
    }
    
    /// Remove an entity
    pub fn remove(&mut self, key: EntityKey) -> Result<()> {
        if self.entities.remove(&key).is_some() {
            Ok(())
        } else {
            Err(GraphError::EntityKeyNotFound { key })
        }
    }
    
    /// Check if store contains an entity
    pub fn contains_entity(&self, key: EntityKey) -> bool {
        self.entities.contains_key(&key)
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        // Approximate size for serialization
        std::mem::size_of::<usize>() + // entity count
        self.entities.len() * (std::mem::size_of::<EntityKey>() + std::mem::size_of::<EntityMeta>()) +
        self.properties.len() +
        self.property_offsets.len() * std::mem::size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    fn create_test_key() -> EntityKey {
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        sm.insert(EntityData {
            type_id: 1,
            properties: "test".to_string(),
            embedding: vec![0.0; 512],
        })
    }

    #[test]
    fn test_entity_store_creation() {
        let store = EntityStore::new();
        assert_eq!(store.count(), 0);
        let _memory_usage = store.memory_usage();
    }

    #[test]
    fn test_entity_insertion() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = EntityData {
            type_id: 1,
            properties: "name:TestEntity,type:Person".to_string(),
            embedding: vec![0.0; 64], // Test embedding
        };
        
        let meta = store.insert(key, &data).unwrap();
        assert_eq!(meta.type_id, 1);
        assert_eq!(meta.degree, 0);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_entity_retrieval() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = EntityData {
            type_id: 2,
            properties: "test:value".to_string(),
            embedding: vec![0.0; 64],
        };
        
        store.insert(key, &data).unwrap();
        
        let meta = store.get(key).unwrap();
        assert_eq!(meta.type_id, 2);
        
        let properties = store.get_properties(meta).unwrap();
        assert_eq!(properties, "test:value");
    }

    #[test]
    fn test_update_degree() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = EntityData {
            type_id: 1,
            properties: "".to_string(),
            embedding: vec![0.0; 64],
        };
        
        store.insert(key, &data).unwrap();
        
        // Test positive update
        store.update_degree(key, 5).unwrap();
        assert_eq!(store.get(key).unwrap().degree, 5);
        
        // Test negative update
        store.update_degree(key, -2).unwrap();
        assert_eq!(store.get(key).unwrap().degree, 3);
        
        // Test that degree doesn't go below 0
        store.update_degree(key, -10).unwrap();
        assert_eq!(store.get(key).unwrap().degree, 0);
    }

    #[test]
    fn test_multiple_entities() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        let mut keys = Vec::new();
        
        for i in 0..5 {
            let key = sm.insert(EntityData {
                type_id: i,
                properties: format!("entity:{}", i),
                embedding: vec![0.0; 64],
            });
            keys.push(key);
            
            let data = EntityData {
                type_id: i,
                properties: format!("entity:{}", i),
                embedding: vec![0.0; 64],
            };
            
            store.insert(key, &data).unwrap();
        }
        
        assert_eq!(store.count(), 5);
        
        for (i, key) in keys.iter().enumerate() {
            let meta = store.get(*key).unwrap();
            assert_eq!(meta.type_id, i as u16);
            
            let properties = store.get_properties(meta).unwrap();
            assert_eq!(properties, format!("entity:{}", i));
        }
    }

    #[test]
    fn test_nonexistent_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        assert!(store.get(key).is_none());
        assert!(store.update_degree(key, 1).is_err());
    }

    #[test]
    fn test_empty_properties() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = EntityData {
            type_id: 1,
            properties: "".to_string(),
            embedding: vec![0.0; 64],
        };
        
        let meta = store.insert(key, &data).unwrap();
        let properties = store.get_properties(&meta).unwrap();
        assert_eq!(properties, "");
    }

    #[test]
    fn test_unicode_properties() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = EntityData {
            type_id: 1,
            properties: "name:测试实体,type:人物".to_string(),
            embedding: vec![0.0; 64],
        };
        
        let meta = store.insert(key, &data).unwrap();
        let properties = store.get_properties(&meta).unwrap();
        assert_eq!(properties, "name:测试实体,type:人物");
    }
}