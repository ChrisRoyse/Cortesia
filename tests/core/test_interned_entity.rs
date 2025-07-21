// Integration tests for interned_entity.rs module
// Tests the PUBLIC API only - no access to private methods or fields
// Focuses on memory efficiency validation and string interning workflows

use llmkg::core::interned_entity::{
    InternedEntityData, InternedEntityCollection, InternedRelationship, InternedDataStats
};
use llmkg::core::types::EntityKey;
use llmkg::storage::string_interner::{StringInterner, InternedString, InternedProperties};
use std::collections::HashMap;
use slotmap::SlotMap;

/// Helper to create test entity data with repeated strings for memory testing
fn create_test_properties_with_duplicates(count: usize) -> Vec<HashMap<String, String>> {
    let mut properties_list = Vec::new();
    
    // Common values that will be repeated across entities
    let common_values = vec![
        ("type", "Person"),
        ("status", "Active"),
        ("department", "Engineering"),
        ("location", "San Francisco"),
        ("role", "Software Engineer"),
    ];
    
    for i in 0..count {
        let mut props = HashMap::new();
        
        // Add common properties (these should be interned)
        for (key, value) in &common_values {
            props.insert(key.to_string(), value.to_string());
        }
        
        // Add unique properties
        props.insert("id".to_string(), i.to_string());
        props.insert("name".to_string(), format!("Employee_{}", i));
        props.insert("email".to_string(), format!("employee{}@company.com", i));
        
        properties_list.push(props);
    }
    
    properties_list
}

/// Helper to calculate approximate memory usage for standard string storage
fn calculate_standard_memory_usage(properties_list: &[HashMap<String, String>]) -> usize {
    let mut total_memory = 0;
    
    for props in properties_list {
        for (key, value) in props {
            // String overhead + actual characters
            total_memory += std::mem::size_of::<String>() * 2; // Key and value String structs
            total_memory += key.len() + value.len(); // Actual string data
        }
    }
    
    total_memory
}

#[test]
fn test_memory_efficiency_comparison() {
    // Create test data with many repeated strings
    let entity_count = 1000;
    let properties_list = create_test_properties_with_duplicates(entity_count);
    
    // Calculate memory usage without interning
    let standard_memory = calculate_standard_memory_usage(&properties_list);
    
    // Create interned collection
    let mut interned_collection = InternedEntityCollection::new();
    
    // Add all entities with interning
    let mut slot_map = SlotMap::with_key();
    for (i, props) in properties_list.iter().enumerate() {
        let key = slot_map.insert(i);
        let embedding = vec![0.1 * i as f32; 128]; // 128-dimensional embedding
        interned_collection.add_entity(key, 1, props, embedding);
    }
    
    // Get memory statistics through public API
    let stats = interned_collection.stats();
    
    // Verify memory optimization
    println!("Standard memory usage: {} KB", standard_memory / 1024);
    println!("Interned memory usage: {} KB", stats.total_memory_bytes / 1024);
    println!("Memory saved: {} KB", stats.interner_stats.memory_saved_bytes / 1024);
    println!("Deduplication ratio: {:.2}:1", stats.interner_stats.deduplication_ratio);
    
    // Assert significant memory savings
    assert!(stats.total_memory_bytes < standard_memory);
    assert!(stats.interner_stats.memory_saved_bytes > 0);
    assert!(stats.interner_stats.deduplication_ratio > 1.0);
    
    // Verify correct entity count
    assert_eq!(stats.entity_count, entity_count);
}

#[test]
fn test_complete_entity_lifecycle_with_interning() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Phase 1: Create entity with initial properties
    let entity_key = slot_map.insert(1);
    let mut initial_props = HashMap::new();
    initial_props.insert("name".to_string(), "John Doe".to_string());
    initial_props.insert("department".to_string(), "Engineering".to_string());
    
    let embedding = vec![0.5; 256];
    collection.add_entity(entity_key, 100, &initial_props, embedding.clone());
    
    // Phase 2: Modify entity through collection
    if let Some(entity) = collection.entities.get_mut(&entity_key) {
        entity.add_property(&collection.interner, "title", "Senior Engineer");
        entity.set_category(&collection.interner, "Employee");
        entity.set_description(&collection.interner, "A senior member of the engineering team");
        entity.add_tag(&collection.interner, "leadership");
        entity.add_tag(&collection.interner, "backend");
    }
    
    // Phase 3: Verify all data is correctly stored and retrievable
    let entity = collection.entities.get(&entity_key).unwrap();
    
    assert_eq!(entity.get_property(&collection.interner, "name"), Some("John Doe".to_string()));
    assert_eq!(entity.get_property(&collection.interner, "department"), Some("Engineering".to_string()));
    assert_eq!(entity.get_property(&collection.interner, "title"), Some("Senior Engineer".to_string()));
    
    let tags = entity.get_tags(&collection.interner);
    assert_eq!(tags.len(), 2);
    assert!(tags.contains(&"leadership".to_string()));
    assert!(tags.contains(&"backend".to_string()));
    
    // Phase 4: Export to JSON and verify
    let json_result = entity.to_json(&collection.interner);
    assert!(json_result.is_ok());
    
    let json = json_result.unwrap();
    assert!(json.contains("John Doe"));
    assert!(json.contains("Engineering"));
    assert!(json.contains("Senior Engineer"));
    assert!(json.contains("Employee"));
    assert!(json.contains("leadership"));
}

#[test]
fn test_string_interning_workflows_large_scale() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Test with large strings that benefit from interning
    let large_description = "A".repeat(10000);
    let large_category = "B".repeat(5000);
    
    // Add multiple entities with the same large strings
    let entity_count = 100;
    for i in 0..entity_count {
        let key = slot_map.insert(i);
        collection.add_entity(key, 1, &HashMap::new(), vec![i as f32]);
        
        if let Some(entity) = collection.entities.get_mut(&key) {
            entity.set_description(&collection.interner, &large_description);
            entity.set_category(&collection.interner, &large_category);
        }
    }
    
    // Verify memory efficiency through stats
    let stats = collection.stats();
    
    // With interning, we should only store these large strings once
    let expected_without_interning = entity_count * (large_description.len() + large_category.len());
    let actual_string_memory = stats.interner_stats.total_memory_bytes as usize;
    
    println!("Expected memory without interning: {} KB", expected_without_interning / 1024);
    println!("Actual memory with interning: {} KB", actual_string_memory / 1024);
    
    // Should be significantly less memory used
    assert!(actual_string_memory < expected_without_interning / 10);
    
    // Verify all entities have the correct values
    for entity in collection.entities.values() {
        assert_eq!(
            collection.interner.get(entity.category),
            Some(large_category.clone())
        );
        assert_eq!(
            collection.interner.get(entity.description),
            Some(large_description.clone())
        );
    }
}

#[test]
fn test_memory_efficiency_validation_metrics() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create entities with varying property counts
    for i in 0..50 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        
        // Add properties based on entity index
        for j in 0..=i % 10 {
            props.insert(format!("prop_{}", j), format!("value_{}", j % 5));
        }
        
        collection.add_entity(key, (i % 3) as u16, &props, vec![i as f32; 64]);
    }
    
    // Get comprehensive statistics
    let stats = collection.stats();
    
    // Validate statistics are reasonable
    assert_eq!(stats.entity_count, 50);
    assert!(stats.avg_properties_per_entity > 0.0);
    assert!(stats.total_memory_bytes > 0);
    assert!(stats.properties_memory_bytes > 0);
    assert!(stats.embedding_memory_bytes > 0);
    
    // Verify memory breakdown adds up correctly
    assert!(stats.total_memory_bytes >= stats.properties_memory_bytes + stats.embedding_memory_bytes);
    
    // Test string interner statistics
    assert!(stats.interner_stats.unique_strings > 0);
    assert!(stats.interner_stats.total_references >= stats.interner_stats.unique_strings);
    
    println!("Statistics Report:\n{}", stats);
}

#[test]
fn test_relationship_lifecycle_with_interning() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create entities
    let entity1 = slot_map.insert(1);
    let entity2 = slot_map.insert(2);
    let entity3 = slot_map.insert(3);
    
    collection.add_entity(entity1, 1, &HashMap::new(), vec![1.0]);
    collection.add_entity(entity2, 1, &HashMap::new(), vec![2.0]);
    collection.add_entity(entity3, 1, &HashMap::new(), vec![3.0]);
    
    // Add relationships with properties
    let mut rel_props = HashMap::new();
    rel_props.insert("strength".to_string(), "high".to_string());
    rel_props.insert("confidence".to_string(), "0.95".to_string());
    rel_props.insert("verified".to_string(), "true".to_string());
    
    collection.add_relationship(entity1, entity2, "MANAGES", 0.9, &rel_props);
    collection.add_relationship(entity2, entity3, "COLLABORATES", 0.8, &rel_props);
    collection.add_relationship(entity1, entity3, "MANAGES", 0.7, &rel_props);
    
    // Verify relationships
    assert_eq!(collection.relationships.len(), 3);
    
    // Check that relationship types are interned correctly
    let manages_count = collection.relationships.iter()
        .filter(|r| r.get_relationship_type(&collection.interner) == Some("MANAGES".to_string()))
        .count();
    assert_eq!(manages_count, 2);
    
    // Verify properties are accessible
    for relationship in &collection.relationships {
        let props_memory = relationship.memory_usage();
        assert!(props_memory > 0);
        
        // Verify we can retrieve properties through the interner
        assert_eq!(
            relationship.properties.get(&collection.interner, "strength"),
            Some("high".to_string())
        );
    }
}

#[test]
fn test_entity_search_and_retrieval() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create entities with different properties and tags
    for i in 0..20 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        
        props.insert("id".to_string(), i.to_string());
        props.insert("type".to_string(), if i % 2 == 0 { "Employee" } else { "Contractor" }.to_string());
        props.insert("department".to_string(), if i % 3 == 0 { "Engineering" } else { "Sales" }.to_string());
        
        collection.add_entity(key, 1, &props, vec![i as f32]);
        
        // Add tags based on patterns
        if let Some(entity) = collection.entities.get_mut(&key) {
            if i % 2 == 0 {
                entity.add_tag(&collection.interner, "full-time");
            }
            if i % 3 == 0 {
                entity.add_tag(&collection.interner, "technical");
            }
            if i % 5 == 0 {
                entity.add_tag(&collection.interner, "senior");
            }
        }
    }
    
    // Test find by property
    let employees = collection.find_by_property("type", "Employee");
    assert_eq!(employees.len(), 10);
    
    let engineering = collection.find_by_property("department", "Engineering");
    assert_eq!(engineering.len(), 7); // 0, 3, 6, 9, 12, 15, 18
    
    // Test find by tag
    let full_time = collection.find_by_tag("full-time");
    assert_eq!(full_time.len(), 10);
    
    let technical = collection.find_by_tag("technical");
    assert_eq!(technical.len(), 7);
    
    let senior = collection.find_by_tag("senior");
    assert_eq!(senior.len(), 4); // 0, 5, 10, 15
    
    // Test get all property keys
    let all_keys = collection.get_all_property_keys();
    assert!(all_keys.contains(&"id".to_string()));
    assert!(all_keys.contains(&"type".to_string()));
    assert!(all_keys.contains(&"department".to_string()));
    assert_eq!(all_keys.len(), 3);
}

#[test]
fn test_json_export_functionality() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create a few entities with rich data
    for i in 0..5 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        props.insert("name".to_string(), format!("Entity_{}", i));
        props.insert("value".to_string(), (i * 100).to_string());
        
        collection.add_entity(key, i as u16, &props, vec![i as f32; 10]);
        
        if let Some(entity) = collection.entities.get_mut(&key) {
            entity.set_category(&collection.interner, "TestCategory");
            entity.set_description(&collection.interner, format!("Description for entity {}", i).as_str());
            entity.add_tag(&collection.interner, "test");
            entity.add_tag(&collection.interner, format!("group_{}", i % 2).as_str());
        }
    }
    
    // Export sample JSON
    let json_result = collection.export_sample_json(3);
    assert!(json_result.is_ok());
    
    let json = json_result.unwrap();
    
    // Verify JSON structure
    assert!(json.contains("entities"));
    assert!(json.contains("stats"));
    assert!(json.contains("Entity_0"));
    assert!(json.contains("TestCategory"));
    assert!(json.contains("embedding_preview"));
    
    // Parse and validate
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let entities = parsed["entities"].as_array().unwrap();
    assert_eq!(entities.len(), 3); // Limited to 3 as requested
}

#[test]
fn test_performance_characteristics_large_dataset() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Measure time for adding entities
    let start = std::time::Instant::now();
    
    // Add 10,000 entities
    for i in 0..10_000 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        
        // Mix of common and unique properties
        props.insert("type".to_string(), (i % 10).to_string());
        props.insert("status".to_string(), if i % 2 == 0 { "active" } else { "inactive" }.to_string());
        props.insert("id".to_string(), i.to_string());
        
        collection.add_entity(key, (i % 100) as u16, &props, vec![i as f32; 32]);
    }
    
    let add_duration = start.elapsed();
    
    // Measure search performance
    let search_start = std::time::Instant::now();
    let results = collection.find_by_property("type", "5");
    let search_duration = search_start.elapsed();
    
    // Verify results
    assert_eq!(results.len(), 1000); // Should find 1000 entities with type "5"
    
    // Get final statistics
    let stats = collection.stats();
    
    println!("Performance Metrics:");
    println!("  Added 10,000 entities in: {:?}", add_duration);
    println!("  Search completed in: {:?}", search_duration);
    println!("  Average time per entity: {:?}", add_duration / 10_000);
    println!("  Memory efficiency: {:.2}:1", stats.interner_stats.deduplication_ratio);
    println!("  Total unique strings: {}", stats.interner_stats.unique_strings);
    
    // Performance assertions
    assert!(add_duration.as_secs() < 5); // Should complete in under 5 seconds
    assert!(search_duration.as_millis() < 100); // Search should be fast
    assert!(stats.interner_stats.deduplication_ratio > 2.0); // Good deduplication
}

#[test]
fn test_unicode_and_special_characters_handling() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Test various string types
    let test_strings = vec![
        ("unicode", "Hello 世界 🌍 café résumé naïve"),
        ("emoji", "🚀🎯🔥💡✨"),
        ("special", "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"),
        ("whitespace", "  \t\n\r  spaces  \t  "),
        ("empty", ""),
        ("very_long", &"A".repeat(100_000)),
    ];
    
    let key = slot_map.insert(1);
    let mut props = HashMap::new();
    
    for (name, value) in &test_strings {
        props.insert(name.to_string(), value.to_string());
    }
    
    collection.add_entity(key, 1, &props, vec![1.0]);
    
    // Add as tags as well
    if let Some(entity) = collection.entities.get_mut(&key) {
        for (_, value) in &test_strings {
            entity.add_tag(&collection.interner, value);
        }
    }
    
    // Verify all strings are correctly stored and retrieved
    let entity = collection.entities.get(&key).unwrap();
    
    for (name, expected_value) in &test_strings {
        assert_eq!(
            entity.get_property(&collection.interner, name),
            Some(expected_value.to_string())
        );
    }
    
    // Verify tags
    let tags = entity.get_tags(&collection.interner);
    assert_eq!(tags.len(), test_strings.len());
}

#[test]
fn test_concurrent_modification_safety() {
    use std::sync::Arc;
    use std::thread;
    
    // Note: This test verifies that the public API can be used safely
    // even though the actual implementation may not be thread-safe
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Pre-populate with some data
    for i in 0..100 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        props.insert("id".to_string(), i.to_string());
        collection.add_entity(key, 1, &props, vec![i as f32]);
    }
    
    // Since InternedEntityCollection is not Send/Sync, we can only test
    // single-threaded access patterns that might occur in concurrent scenarios
    
    // Simulate rapid additions and searches
    for i in 100..200 {
        let key = slot_map.insert(i);
        collection.add_entity(key, 2, &HashMap::new(), vec![i as f32]);
        
        // Interleave with searches
        if i % 10 == 0 {
            let results = collection.find_by_property("id", "50");
            assert!(!results.is_empty());
        }
    }
    
    // Verify final state
    assert_eq!(collection.entities.len(), 200);
    let stats = collection.stats();
    assert_eq!(stats.entity_count, 200);
}

#[test]
fn test_memory_usage_calculation_accuracy() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create entity with known sizes
    let key = slot_map.insert(1);
    let embedding_size = 1000;
    let embedding = vec![1.0_f32; embedding_size];
    
    let mut props = HashMap::new();
    props.insert("key1".to_string(), "value1".to_string());
    props.insert("key2".to_string(), "value2".to_string());
    
    collection.add_entity(key, 1, &props, embedding);
    
    // Add more properties
    if let Some(entity) = collection.entities.get_mut(&key) {
        entity.add_property(&collection.interner, "key3", "value3");
        entity.add_tag(&collection.interner, "tag1");
        entity.add_tag(&collection.interner, "tag2");
    }
    
    let entity = collection.entities.get(&key).unwrap();
    let memory_usage = entity.memory_usage();
    
    // Calculate expected minimum memory
    let embedding_memory = embedding_size * std::mem::size_of::<f32>();
    let base_struct_size = std::mem::size_of::<InternedEntityData>();
    
    println!("Memory Usage Breakdown:");
    println!("  Total reported: {} bytes", memory_usage);
    println!("  Embedding size: {} bytes", embedding_memory);
    println!("  Base struct size: {} bytes", base_struct_size);
    
    // Memory usage should at least include embedding
    assert!(memory_usage >= embedding_memory);
    assert!(memory_usage >= base_struct_size);
}

#[test]
fn test_empty_collection_behavior() {
    let collection = InternedEntityCollection::new();
    
    // Test empty collection stats
    let stats = collection.stats();
    assert_eq!(stats.entity_count, 0);
    assert_eq!(stats.total_memory_bytes, 0);
    assert_eq!(stats.avg_properties_per_entity, 0.0);
    
    // Test searches on empty collection
    let results = collection.find_by_property("any", "value");
    assert!(results.is_empty());
    
    let results = collection.find_by_tag("any_tag");
    assert!(results.is_empty());
    
    // Test property keys on empty collection
    let keys = collection.get_all_property_keys();
    assert!(keys.is_empty());
    
    // Test JSON export on empty collection
    let json = collection.export_sample_json(10).unwrap();
    assert!(json.contains("\"entities\":[]"));
}

#[test]
fn test_interning_deduplication_effectiveness() {
    let mut collection = InternedEntityCollection::new();
    let mut slot_map = SlotMap::with_key();
    
    // Create pattern that maximizes string duplication
    let common_values = vec![
        "ProductionServer",
        "DatabaseCluster",
        "LoadBalancer",
        "CacheLayer",
        "MessageQueue",
    ];
    
    // Create 1000 entities with highly duplicated values
    for i in 0..1000 {
        let key = slot_map.insert(i);
        let mut props = HashMap::new();
        
        // Each entity gets multiple properties with values from common_values
        for j in 0..10 {
            let key_name = format!("property_{}", j);
            let value = common_values[j % common_values.len()];
            props.insert(key_name, value.to_string());
        }
        
        collection.add_entity(key, 1, &props, vec![i as f32]);
    }
    
    let stats = collection.stats();
    
    // With 1000 entities * 10 properties = 10,000 property values
    // But only ~15 unique strings (10 property names + 5 values)
    println!("Deduplication test results:");
    println!("  Total property references: {}", stats.interner_stats.total_references);
    println!("  Unique strings: {}", stats.interner_stats.unique_strings);
    println!("  Deduplication ratio: {:.2}:1", stats.interner_stats.deduplication_ratio);
    
    // Should have massive deduplication
    assert!(stats.interner_stats.deduplication_ratio > 100.0);
    assert!(stats.interner_stats.unique_strings < 100); // Should be around 15-20
}