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
        
        if start >= self.properties.len() || end > self.properties.len() {
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
}