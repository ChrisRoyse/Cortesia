// Phase 4.4: Simplified Zero-Copy Serialization Tests
// Basic tests for zero-copy functionality

use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer};

#[test]
fn test_basic_zero_copy_serialization() {
    let mut serializer = ZeroCopySerializer::new();
    
    // Create test entity
    let entity = EntityData {
        type_id: 1,
        properties: "test entity".to_string(),
        embedding: vec![0.1, 0.2, 0.3, 0.4],
    };
    
    serializer.add_entity(&entity, 4).unwrap();
    
    // Finalize
    let data = serializer.finalize().unwrap();
    assert!(!data.is_empty());
    
    // Deserialize with zero-copy
    let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
    
    assert_eq!(deserializer.entity_count(), 1);
    
    let entity = deserializer.get_entity(0).unwrap();
    let type_id = entity.type_id; // Avoid packed field reference
    assert_eq!(type_id, 1);
    
    let properties = deserializer.get_entity_properties(entity);
    assert_eq!(properties, "test entity");
}

#[test]
fn test_zero_copy_performance() {
    let mut serializer = ZeroCopySerializer::new();
    
    // Add multiple entities
    let entity_count = 1000;
    for i in 0..entity_count {
        let entity = EntityData {
            type_id: (i % 10) as u16,
            properties: format!("entity_{}", i),
            embedding: vec![i as f32; 32],
        };
        serializer.add_entity(&entity, 32).unwrap();
    }
    
    let data = serializer.finalize().unwrap();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
    
    // Test iteration performance
    let start = std::time::Instant::now();
    let count = deserializer.iter_entities().count();
    let iteration_time = start.elapsed();
    
    assert_eq!(count, entity_count);
    
    // Should be very fast
    let entities_per_ms = count as f64 / iteration_time.as_millis() as f64;
    assert!(entities_per_ms > 100.0); // Should process at least 100 entities per ms
}

#[test]
fn test_zero_copy_memory_efficiency() {
    let mut serializer = ZeroCopySerializer::new();
    
    // Create entities with repeated data for compression testing
    for i in 0..100 {
        let entity = EntityData {
            type_id: 1, // All same type
            properties: "repeated property string".to_string(), // Same properties
            embedding: vec![0.5; 64], // Same embedding
        };
        serializer.add_entity(&entity, 64).unwrap();
    }
    
    let data = serializer.finalize().unwrap();
    
    // Calculate efficiency
    let raw_size = 100 * (64 * 4 + "repeated property string".len() + 8); // Rough estimate
    let compressed_size = data.len();
    
    // Should achieve some compression
    assert!(compressed_size < raw_size);
    
    let compression_ratio = raw_size as f32 / compressed_size as f32;
    println!("Compression ratio: {:.2}:1", compression_ratio);
    
    // Should be at least somewhat compressed
    assert!(compression_ratio > 1.0);
}