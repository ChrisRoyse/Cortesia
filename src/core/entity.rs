use crate::core::types::{EntityKey, EntityMeta, EntityData};
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
            .ok_or({
                use slotmap::{Key, KeyData};
                let key_data: KeyData = key.data();
                GraphError::EntityNotFound { id: key_data.as_ffi() as u32 }
            })?;
        
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

    fn create_test_entity_data(type_id: u16, properties: &str) -> EntityData {
        EntityData {
            type_id,
            properties: properties.to_string(),
            embedding: vec![0.0; 64],
        }
    }

    // ===== BASIC FUNCTIONALITY TESTS =====

    #[test]
    fn test_entity_store_creation() {
        let store = EntityStore::new();
        assert_eq!(store.count(), 0);
        assert_eq!(store.capacity(), 0);
        assert!(store.memory_usage() >= 0);
        
        // Test internal structure initialization
        assert_eq!(store.entities.len(), 0);
        assert_eq!(store.properties.len(), 0);
        assert_eq!(store.property_offsets.len(), 0);
    }

    #[test]
    fn test_entity_insertion_basic() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "name:TestEntity,type:Person");
        
        let meta = store.insert(key, &data).unwrap();
        assert_eq!(meta.type_id, 1);
        assert_eq!(meta.degree, 0);
        assert_eq!(meta.embedding_offset, 0);
        assert_eq!(meta.property_offset, 0);
        assert_eq!(store.count(), 1);
        
        // Verify internal state
        assert_eq!(store.properties.len(), data.properties.len());
        assert_eq!(store.property_offsets.len(), 2); // start and end offsets
    }

    #[test]
    fn test_entity_retrieval_basic() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(2, "test:value");
        store.insert(key, &data).unwrap();
        
        let meta = store.get(key).unwrap();
        assert_eq!(meta.type_id, 2);
        
        let properties = store.get_properties(meta).unwrap();
        assert_eq!(properties, "test:value");
    }

    // ===== INSERT METHOD COMPREHENSIVE TESTS =====

    #[test]
    fn test_insert_empty_properties() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "");
        let meta = store.insert(key, &data).unwrap();
        
        assert_eq!(meta.property_offset, 0);
        assert_eq!(store.property_offsets, vec![0, 0]); // start and end are same
        assert_eq!(store.properties.len(), 0);
    }

    #[test]
    fn test_insert_large_properties() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        // Create a large property string (10KB)
        let large_props = "x".repeat(10_000);
        let data = create_test_entity_data(1, &large_props);
        
        let meta = store.insert(key, &data).unwrap();
        assert_eq!(meta.property_offset, 0);
        assert_eq!(store.properties.len(), 10_000);
        assert_eq!(store.property_offsets, vec![0, 10_000]);
        
        let retrieved_props = store.get_properties(&meta).unwrap();
        assert_eq!(retrieved_props.len(), 10_000);
        assert_eq!(retrieved_props, large_props);
    }

    #[test]
    fn test_insert_multiple_entities_property_offsets() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        
        // Insert entities with different property sizes
        let props = vec!["short", "medium_length_prop", "very_long_property_string_here"];
        let mut keys = Vec::new();
        let mut expected_offsets = vec![0];
        let mut cumulative_length = 0;
        
        for (i, prop) in props.iter().enumerate() {
            let key = sm.insert(create_test_entity_data(i as u16, prop));
            keys.push(key);
            
            let data = create_test_entity_data(i as u16, prop);
            let meta = store.insert(key, &data).unwrap();
            
            assert_eq!(meta.property_offset, cumulative_length);
            cumulative_length += prop.len() as u32;
            expected_offsets.push(cumulative_length);
        }
        
        // Verify property offsets are correctly maintained
        assert_eq!(store.property_offsets, expected_offsets);
        
        // Verify each entity can retrieve its properties correctly
        for (i, key) in keys.iter().enumerate() {
            let meta = store.get(*key).unwrap();
            let retrieved_props = store.get_properties(meta).unwrap();
            assert_eq!(retrieved_props, props[i]);
        }
    }

    #[test]
    fn test_insert_unicode_and_special_chars() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let unicode_props = "emoji:🚀,chinese:测试,japanese:テスト,special:\"quotes\",newline:\n,tab:\t";
        let data = create_test_entity_data(1, unicode_props);
        
        let meta = store.insert(key, &data).unwrap();
        let retrieved_props = store.get_properties(&meta).unwrap();
        assert_eq!(retrieved_props, unicode_props);
    }

    // ===== GET PROPERTIES METHOD COMPREHENSIVE TESTS =====

    #[test]
    fn test_get_properties_empty() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "");
        let meta = store.insert(key, &data).unwrap();
        
        let properties = store.get_properties(&meta).unwrap();
        assert_eq!(properties, "");
    }

    #[test]
    fn test_get_properties_boundary_conditions() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        
        // Insert multiple entities to test boundary conditions
        let entities = vec!["", "a", "ab", "abc"];
        let mut keys = Vec::new();
        
        for (i, prop) in entities.iter().enumerate() {
            let key = sm.insert(create_test_entity_data(i as u16, prop));
            keys.push(key);
            let data = create_test_entity_data(i as u16, prop);
            store.insert(key, &data).unwrap();
        }
        
        // Test retrieval of each entity's properties
        for (i, key) in keys.iter().enumerate() {
            let meta = store.get(*key).unwrap();
            let props = store.get_properties(meta).unwrap();
            assert_eq!(props, entities[i]);
        }
    }

    #[test]
    fn test_get_properties_corrupted_offset() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Create a corrupted meta with invalid offset
        let corrupted_meta = EntityMeta {
            type_id: 1,
            embedding_offset: 0,
            property_offset: 1000, // Invalid offset beyond properties length
            degree: 0,
            last_accessed: std::time::Instant::now(),
        };
        
        let result = store.get_properties(&corrupted_meta);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::IndexCorruption));
    }

    #[test]
    fn test_get_properties_with_invalid_utf8() {
        let mut store = EntityStore::new();
        
        // Manually construct invalid UTF-8 in properties
        store.properties = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8 sequence
        store.property_offsets = vec![0, 3];
        
        let meta = EntityMeta {
            type_id: 1,
            embedding_offset: 0,
            property_offset: 0,
            degree: 0,
            last_accessed: std::time::Instant::now(),
        };
        
        let result = store.get_properties(&meta);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::SerializationError(_)));
    }

    // ===== UPDATE DEGREE METHOD COMPREHENSIVE TESTS =====

    #[test]
    fn test_update_degree_basic() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
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
    fn test_update_degree_zero_delta() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Set initial degree
        store.update_degree(key, 5).unwrap();
        let initial_degree = store.get(key).unwrap().degree;
        
        // Update with zero delta
        store.update_degree(key, 0).unwrap();
        assert_eq!(store.get(key).unwrap().degree, initial_degree);
    }

    #[test]
    fn test_update_degree_max_values() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Test maximum positive value for i16
        store.update_degree(key, i16::MAX).unwrap();
        assert_eq!(store.get(key).unwrap().degree, i16::MAX as u16);
        
        // Test adding more (should not overflow but cap at u16::MAX)
        store.update_degree(key, 1).unwrap();
        let final_degree = store.get(key).unwrap().degree;
        assert!(final_degree >= i16::MAX as u16);
    }

    #[test]
    fn test_update_degree_negative_edge_cases() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Start with 0 degree and apply large negative delta
        store.update_degree(key, i16::MIN).unwrap();
        assert_eq!(store.get(key).unwrap().degree, 0);
        
        // Set a small degree and apply large negative delta
        store.update_degree(key, 5).unwrap();
        store.update_degree(key, -1000).unwrap();
        assert_eq!(store.get(key).unwrap().degree, 0);
    }

    #[test]
    fn test_update_degree_nonexistent_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        // Try to update degree for non-existent entity
        let result = store.update_degree(key, 1);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::EntityNotFound { .. }));
    }

    // ===== MEMORY AND STORAGE EFFICIENCY TESTS =====

    #[test]
    fn test_memory_usage_calculation() {
        let mut store = EntityStore::new();
        let initial_memory = store.memory_usage();
        
        // Insert entities and verify memory usage increases
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        for i in 0..10 {
            let key = sm.insert(create_test_entity_data(i, &format!("entity_{}", i)));
            let data = create_test_entity_data(i, &format!("entity_{}", i));
            store.insert(key, &data).unwrap();
        }
        
        let final_memory = store.memory_usage();
        assert!(final_memory > initial_memory);
        
        // Memory should be at least the size of properties and offsets
        let expected_minimum = store.properties.len() + 
                              store.property_offsets.len() * std::mem::size_of::<u32>();
        assert!(final_memory >= expected_minimum);
    }

    #[test]
    fn test_encoded_size_calculation() {
        let mut store = EntityStore::new();
        let initial_size = store.encoded_size();
        
        // Insert an entity
        let key = create_test_key();
        let data = create_test_entity_data(1, "test_properties");
        store.insert(key, &data).unwrap();
        
        let final_size = store.encoded_size();
        assert!(final_size > initial_size);
        
        // Encoded size should include all components
        let expected_minimum = std::mem::size_of::<usize>() + // entity count
                              std::mem::size_of::<EntityKey>() + std::mem::size_of::<EntityMeta>() +
                              data.properties.len() +
                              2 * std::mem::size_of::<u32>(); // property offsets
        assert!(final_size >= expected_minimum);
    }

    #[test]
    fn test_capacity_tracking() {
        let mut store = EntityStore::new();
        assert_eq!(store.capacity(), 0);
        
        // Insert entities to trigger capacity growth
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        for i in 0..100 {
            let key = sm.insert(create_test_entity_data(i, "test"));
            let data = create_test_entity_data(i, "test");
            store.insert(key, &data).unwrap();
        }
        
        // Capacity should be >= count
        assert!(store.capacity() >= store.count());
    }

    // ===== PROPERTY STORAGE SYSTEM VALIDATION =====

    #[test]
    fn test_property_offset_consistency() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        
        let properties = vec!["", "a", "ab", "abc", "abcd"];
        let mut total_length = 0;
        
        for (i, prop) in properties.iter().enumerate() {
            let key = sm.insert(create_test_entity_data(i as u16, prop));
            let data = create_test_entity_data(i as u16, prop);
            let meta = store.insert(key, &data).unwrap();
            
            // Verify property offset is correct
            assert_eq!(meta.property_offset, total_length);
            total_length += prop.len() as u32;
        }
        
        // Verify total properties length
        assert_eq!(store.properties.len(), total_length as usize);
        
        // Verify property_offsets array has correct structure
        assert_eq!(store.property_offsets.len(), properties.len() * 2);
    }

    #[test]
    fn test_concurrent_property_access_simulation() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        let mut keys = Vec::new();
        
        // Insert multiple entities with varying property sizes
        for i in 0..50 {
            let prop = format!("entity_{}_properties_with_data_{}", i, "x".repeat(i % 10));
            let key = sm.insert(create_test_entity_data(i as u16, &prop));
            keys.push((key, prop.clone()));
            
            let data = create_test_entity_data(i as u16, &prop);
            store.insert(key, &data).unwrap();
        }
        
        // Simulate concurrent access by accessing properties in random order
        for (key, expected_prop) in keys.iter().rev() {
            let meta = store.get(*key).unwrap();
            let retrieved_prop = store.get_properties(meta).unwrap();
            assert_eq!(retrieved_prop, *expected_prop);
        }
    }

    // ===== MUTABLE ACCESS TESTS =====

    #[test]
    fn test_get_mut_functionality() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Test mutable access
        {
            let meta_mut = store.get_mut(key).unwrap();
            meta_mut.type_id = 999;
            meta_mut.degree = 42;
        }
        
        // Verify changes persisted
        let meta = store.get(key).unwrap();
        assert_eq!(meta.type_id, 999);
        assert_eq!(meta.degree, 42);
    }

    #[test]
    fn test_get_mut_nonexistent_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        assert!(store.get_mut(key).is_none());
    }

    // ===== CRUD OPERATIONS TESTS =====

    #[test]
    fn test_update_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let initial_data = create_test_entity_data(1, "initial");
        store.insert(key, &initial_data).unwrap();
        
        let updated_data = create_test_entity_data(2, "updated");
        store.update_entity(key, &updated_data).unwrap();
        
        let meta = store.get(key).unwrap();
        assert_eq!(meta.type_id, 2);
    }

    #[test]
    fn test_update_nonexistent_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        let result = store.update_entity(key, &data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::EntityKeyNotFound { .. }));
    }

    #[test]
    fn test_remove_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        assert_eq!(store.count(), 1);
        
        store.remove(key).unwrap();
        assert_eq!(store.count(), 0);
        assert!(store.get(key).is_none());
    }

    #[test]
    fn test_remove_nonexistent_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        let result = store.remove(key);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::EntityKeyNotFound { .. }));
    }

    #[test]
    fn test_contains_entity() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        assert!(!store.contains_entity(key));
        
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        assert!(store.contains_entity(key));
        
        store.remove(key).unwrap();
        assert!(!store.contains_entity(key));
    }

    // ===== ERROR HANDLING AND EDGE CASE TESTS =====

    #[test]
    fn test_add_edge_unsupported_operation() {
        let mut store = EntityStore::new();
        let result = store.add_edge(1, 2, 0.5);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphError::UnsupportedOperation(_)));
    }

    #[test]
    fn test_multiple_operations_sequence() {
        let mut store = EntityStore::new();
        let key = create_test_key();
        
        // Insert
        let data = create_test_entity_data(1, "test");
        store.insert(key, &data).unwrap();
        
        // Update degree multiple times
        store.update_degree(key, 5).unwrap();
        store.update_degree(key, -2).unwrap();
        store.update_degree(key, 10).unwrap();
        
        // Update entity
        let new_data = create_test_entity_data(2, "updated");
        store.update_entity(key, &new_data).unwrap();
        
        // Verify final state
        let meta = store.get(key).unwrap();
        assert_eq!(meta.type_id, 2);
        assert_eq!(meta.degree, 13); // 0 + 5 - 2 + 10
        
        // Get properties
        let props = store.get_properties(meta).unwrap();
        assert_eq!(props, "test"); // Properties don't change in update_entity
    }

    #[test]
    fn test_large_scale_operations() {
        let mut store = EntityStore::new();
        let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
        let entity_count = 1000;
        
        // Insert many entities
        for i in 0..entity_count {
            let key = sm.insert(create_test_entity_data(i, &format!("entity_{}", i)));
            let data = create_test_entity_data(i, &format!("entity_{}", i));
            store.insert(key, &data).unwrap();
        }
        
        assert_eq!(store.count(), entity_count);
        
        // Verify memory efficiency at scale
        let memory_per_entity = store.memory_usage() / entity_count;
        assert!(memory_per_entity > 0);
        
        // Test that encoded size is reasonable
        let encoded_size = store.encoded_size();
        assert!(encoded_size > 0);
        assert!(encoded_size < store.memory_usage() * 2); // Should be somewhat efficient
    }

    // Legacy tests preserved for compatibility
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