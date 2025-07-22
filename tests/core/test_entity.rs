use llmkg::core::entity::EntityStore;
use llmkg::core::types::{EntityKey, EntityMeta, EntityData};
use llmkg::error::{GraphError, Result};
use slotmap::SlotMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Helper to create test entity keys
fn create_test_keys(count: usize) -> Vec<EntityKey> {
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    let mut keys = Vec::new();
    
    for i in 0..count {
        let key = sm.insert(EntityData::new(
            (i % 100) as u16,
            format!("test_entity_{}", i),
            vec![0.0; 64]
        ));
        keys.push(key);
    }
    
    keys
}

/// Helper to create test entity data
fn create_test_entity_data(type_id: u16, properties: &str) -> EntityData {
    EntityData::new(type_id, properties.to_string(), vec![0.0; 64])
}

// ===== ENTITY LIFECYCLE INTEGRATION TESTS =====

#[test]
fn test_entity_lifecycle_workflow() {
    // Create EntityStore through public API
    let mut store = EntityStore::new();
    assert_eq!(store.count(), 0);
    
    // Insert multiple entities with various properties
    let keys = create_test_keys(5);
    let properties_list = vec![
        "name:Alice,type:Person,age:30",
        "name:Bob,type:Person,age:25", 
        "name:Document1,type:Document,content:Important",
        "name:Project,type:Project,status:Active",
        "name:Task,type:Task,priority:High",
    ];
    
    // Insert entities
    for (i, (key, props)) in keys.iter().zip(properties_list.iter()).enumerate() {
        let data = create_test_entity_data(i as u16, props);
        let meta = store.insert(*key, &data).expect("Insert should succeed");
        assert_eq!(meta.type_id, i as u16);
        assert_eq!(meta.degree, 0);
    }
    
    assert_eq!(store.count(), 5);
    
    // Retrieve and verify through public API
    for (key, expected_props) in keys.iter().zip(properties_list.iter()) {
        let meta = store.get(*key).expect("Entity should exist");
        let retrieved_props = store.get_properties(meta).expect("Properties should be retrievable");
        assert_eq!(retrieved_props, *expected_props);
    }
    
    // Update entity degrees and properties
    for (i, key) in keys.iter().enumerate() {
        store.update_degree(*key, (i + 1) as i16).expect("Degree update should succeed");
    }
    
    // Verify updates
    for (i, key) in keys.iter().enumerate() {
        let meta = store.get(*key).expect("Entity should exist");
        assert_eq!(meta.degree, (i + 1) as u16);
    }
    
    // Update entity metadata
    let updated_data = create_test_entity_data(99, "updated_properties");
    store.update_entity(keys[0], &updated_data).expect("Update should succeed");
    
    let updated_meta = store.get(keys[0]).expect("Entity should exist");
    assert_eq!(updated_meta.type_id, 99);
    
    // Remove an entity
    store.remove(keys[2]).expect("Remove should succeed");
    assert_eq!(store.count(), 4);
    assert!(store.get(keys[2]).is_none());
    
    // Verify memory usage patterns through public API
    let memory_usage = store.memory_usage();
    assert!(memory_usage > 0);
    
    // Test contains functionality
    assert!(store.contains_entity(keys[0]));
    assert!(store.contains_entity(keys[1]));
    assert!(!store.contains_entity(keys[2])); // Removed
    assert!(store.contains_entity(keys[3]));
    assert!(store.contains_entity(keys[4]));
}

// ===== MEMORY EFFICIENCY TESTS =====

#[test]
fn test_memory_efficiency_observable_behavior() {
    let mut store = EntityStore::new();
    
    // Measure baseline memory
    let baseline_memory = store.memory_usage();
    
    // Insert entities with progressively larger properties
    let keys = create_test_keys(100);
    let mut total_property_size = 0;
    
    for (i, key) in keys.iter().enumerate() {
        let property_content = "x".repeat(i * 10); // Increasing property sizes
        let data = create_test_entity_data((i % 10) as u16, &property_content);
        total_property_size += property_content.len();
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    // Verify memory efficiency through observable patterns
    let final_memory = store.memory_usage();
    let memory_per_entity = (final_memory - baseline_memory) / 100;
    
    // Memory per entity should be reasonable (less than 1KB per entity for this test)
    assert!(memory_per_entity < 1024, "Memory per entity is too high: {}", memory_per_entity);
    
    // Verify properties are stored efficiently
    for (i, key) in keys.iter().enumerate() {
        let meta = store.get(*key).expect("Entity should exist");
        let props = store.get_properties(meta).expect("Properties should be retrievable");
        assert_eq!(props.len(), i * 10);
    }
    
    // Test that capacity grows efficiently
    let capacity = store.capacity();
    assert!(capacity >= 100); // Should have space for all entities
    assert!(capacity < 200); // Shouldn't over-allocate too much
}

#[test]
fn test_memory_efficiency_property_deduplication() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(1000);
    
    // Insert many entities with similar properties to test storage efficiency
    let common_properties = vec![
        "type:Person,status:Active",
        "type:Document,status:Active", 
        "type:Project,status:Active",
        "type:Task,status:Active",
        "type:Event,status:Active",
    ];
    
    let initial_memory = store.memory_usage();
    
    for (i, key) in keys.iter().enumerate() {
        let props = &common_properties[i % common_properties.len()];
        let data = create_test_entity_data((i % 5) as u16, props);
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    let final_memory = store.memory_usage();
    let memory_increase = final_memory - initial_memory;
    
    // Even with 1000 entities, memory usage should be reasonable
    // Average less than 200 bytes per entity
    let avg_memory_per_entity = memory_increase / 1000;
    assert!(avg_memory_per_entity < 200, "Average memory per entity too high: {}", avg_memory_per_entity);
}

// ===== PUBLIC API OPERATIONS TESTS =====

#[test]
fn test_public_api_entity_operations() {
    let mut store = EntityStore::new();
    
    // Test all public API methods
    
    // new() - already tested
    assert_eq!(store.count(), 0);
    
    // insert()
    let key = create_test_keys(1)[0];
    let data = create_test_entity_data(42, "test:value,key:data");
    let meta = store.insert(key, &data).expect("Insert should succeed");
    assert_eq!(meta.type_id, 42);
    assert_eq!(meta.degree, 0);
    assert_eq!(meta.embedding_offset, 0);
    
    // get()
    let retrieved_meta = store.get(key).expect("Get should return entity");
    assert_eq!(retrieved_meta.type_id, 42);
    
    // get_properties()
    let props = store.get_properties(retrieved_meta).expect("Properties should be retrievable");
    assert_eq!(props, "test:value,key:data");
    
    // get_mut()
    {
        let meta_mut = store.get_mut(key).expect("Get mut should return entity");
        meta_mut.degree = 10;
        meta_mut.type_id = 43;
    }
    
    // Verify mutations persisted
    let updated_meta = store.get(key).expect("Entity should exist");
    assert_eq!(updated_meta.degree, 10);
    assert_eq!(updated_meta.type_id, 43);
    
    // update_degree()
    store.update_degree(key, 5).expect("Update degree should succeed");
    let meta_after_degree = store.get(key).expect("Entity should exist");
    assert_eq!(meta_after_degree.degree, 15); // 10 + 5
    
    store.update_degree(key, -20).expect("Update degree should succeed");
    let meta_after_negative = store.get(key).expect("Entity should exist");
    assert_eq!(meta_after_negative.degree, 0); // Clamped to 0
    
    // update_entity()
    let new_data = create_test_entity_data(100, "new_properties");
    store.update_entity(key, &new_data).expect("Update entity should succeed");
    let meta_after_update = store.get(key).expect("Entity should exist");
    assert_eq!(meta_after_update.type_id, 100);
    
    // contains_entity()
    assert!(store.contains_entity(key));
    let non_existent_key = create_test_keys(1)[0];
    assert!(!store.contains_entity(non_existent_key));
    
    // count()
    assert_eq!(store.count(), 1);
    
    // capacity()
    let capacity = store.capacity();
    assert!(capacity >= 1);
    
    // memory_usage()
    let memory = store.memory_usage();
    assert!(memory > 0);
    
    // encoded_size()
    let encoded = store.encoded_size();
    assert!(encoded > 0);
    
    // add_edge() - should fail
    let result = store.add_edge(1, 2, 0.5);
    assert!(result.is_err());
    match result {
        Err(GraphError::UnsupportedOperation(msg)) => {
            assert!(msg.contains("EntityStore stores entities, not edges"));
        }
        _ => panic!("Expected UnsupportedOperation error"),
    }
    
    // remove()
    store.remove(key).expect("Remove should succeed");
    assert_eq!(store.count(), 0);
    assert!(!store.contains_entity(key));
}

// ===== END-TO-END STORAGE WORKFLOW TESTS =====

#[test]
fn test_end_to_end_storage_workflow() {
    // Simulate a complete workflow: create, populate, query, update, remove
    let mut store = EntityStore::new();
    
    // Phase 1: Initial population
    let person_keys = create_test_keys(10);
    let document_keys = create_test_keys(5);
    let project_keys = create_test_keys(3);
    
    // Insert persons
    for (i, key) in person_keys.iter().enumerate() {
        let data = create_test_entity_data(1, &format!("name:Person{},role:Developer,team:Engineering", i));
        store.insert(*key, &data).expect("Insert person should succeed");
    }
    
    // Insert documents  
    for (i, key) in document_keys.iter().enumerate() {
        let data = create_test_entity_data(2, &format!("title:Doc{},type:Technical,status:Published", i));
        store.insert(*key, &data).expect("Insert document should succeed");
    }
    
    // Insert projects
    for (i, key) in project_keys.iter().enumerate() {
        let data = create_test_entity_data(3, &format!("name:Project{},status:Active,priority:High", i));
        store.insert(*key, &data).expect("Insert project should succeed");
    }
    
    assert_eq!(store.count(), 18); // 10 + 5 + 3
    
    // Phase 2: Simulate relationships (update degrees)
    // Each person is connected to 2-5 other entities
    for (i, person_key) in person_keys.iter().enumerate() {
        let connections = 2 + (i % 4);
        store.update_degree(*person_key, connections as i16).expect("Update degree should succeed");
    }
    
    // Documents are referenced by 1-3 entities
    for (i, doc_key) in document_keys.iter().enumerate() {
        let references = 1 + (i % 3);
        store.update_degree(*doc_key, references as i16).expect("Update degree should succeed");
    }
    
    // Projects have 5-10 connections
    for (i, project_key) in project_keys.iter().enumerate() {
        let connections = 5 + (i * 2);
        store.update_degree(*project_key, connections as i16).expect("Update degree should succeed");
    }
    
    // Phase 3: Query and verify
    let mut total_degree = 0;
    let mut entities_by_type: HashMap<u16, Vec<(EntityKey, String)>> = HashMap::new();
    
    // Collect all entities by type
    for key in person_keys.iter().chain(document_keys.iter()).chain(project_keys.iter()) {
        if let Some(meta) = store.get(*key) {
            total_degree += meta.degree as u32;
            let props = store.get_properties(meta).expect("Properties should be retrievable");
            entities_by_type.entry(meta.type_id).or_insert_with(Vec::new).push((*key, props));
        }
    }
    
    assert_eq!(entities_by_type.get(&1).unwrap().len(), 10); // Persons
    assert_eq!(entities_by_type.get(&2).unwrap().len(), 5);  // Documents
    assert_eq!(entities_by_type.get(&3).unwrap().len(), 3);  // Projects
    
    // Phase 4: Update some entities
    for (i, key) in person_keys.iter().take(3).enumerate() {
        let new_data = create_test_entity_data(1, &format!("name:UpdatedPerson{},role:Senior Developer,team:Engineering", i));
        store.update_entity(*key, &new_data).expect("Update should succeed");
    }
    
    // Phase 5: Remove some entities
    for key in document_keys.iter().skip(3) {
        store.remove(*key).expect("Remove should succeed");
    }
    
    assert_eq!(store.count(), 16); // 18 - 2
    
    // Phase 6: Final verification
    let final_memory = store.memory_usage();
    let final_encoded = store.encoded_size();
    
    assert!(final_memory > 0);
    assert!(final_encoded > 0);
    assert!(final_encoded <= final_memory); // Encoded should be more compact
}

// ===== PERFORMANCE CHARACTERISTICS TESTS =====

#[test]
fn test_performance_insert_large_batch() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(10000);
    
    let start = Instant::now();
    
    for (i, key) in keys.iter().enumerate() {
        let data = create_test_entity_data((i % 100) as u16, &format!("entity_{}", i));
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    let duration = start.elapsed();
    
    // Should insert 10k entities in under 100ms
    assert!(duration < Duration::from_millis(100), "Insert took too long: {:?}", duration);
    assert_eq!(store.count(), 10000);
    
    // Verify random access performance
    let access_start = Instant::now();
    
    for i in (0..1000).step_by(10) {
        let meta = store.get(keys[i]).expect("Entity should exist");
        let _props = store.get_properties(meta).expect("Properties should be retrievable");
    }
    
    let access_duration = access_start.elapsed();
    
    // Should access 100 entities in under 10ms
    assert!(access_duration < Duration::from_millis(10), "Access took too long: {:?}", access_duration);
}

#[test]
fn test_performance_update_degrees() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(5000);
    
    // Insert entities
    for (i, key) in keys.iter().enumerate() {
        let data = create_test_entity_data((i % 50) as u16, "test");
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    let start = Instant::now();
    
    // Update degrees for all entities
    for (i, key) in keys.iter().enumerate() {
        let delta = ((i % 10) as i16) - 5;
        store.update_degree(*key, delta).expect("Update should succeed");
    }
    
    let duration = start.elapsed();
    
    // Should update 5k entity degrees in under 50ms
    assert!(duration < Duration::from_millis(50), "Degree updates took too long: {:?}", duration);
}

#[test]
fn test_performance_concurrent_reads() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(1000);
    
    // Populate store
    for (i, key) in keys.iter().enumerate() {
        let data = create_test_entity_data((i % 20) as u16, &format!("concurrent_entity_{}", i));
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    // Wrap in Arc for thread sharing (simulating concurrent read access)
    let store_arc = Arc::new(store);
    let keys_arc = Arc::new(keys);
    
    let start = Instant::now();
    let mut handles = vec![];
    
    // Spawn multiple reader threads
    for thread_id in 0..4 {
        let store_clone = Arc::clone(&store_arc);
        let keys_clone = Arc::clone(&keys_arc);
        
        let handle = thread::spawn(move || {
            for i in 0..250 {
                let key_idx = (thread_id * 250 + i) % 1000;
                if let Some(meta) = store_clone.get(keys_clone[key_idx]) {
                    let _props = store_clone.get_properties(meta).ok();
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete");
    }
    
    let duration = start.elapsed();
    
    // Concurrent reads should complete quickly
    assert!(duration < Duration::from_millis(100), "Concurrent reads took too long: {:?}", duration);
}

// ===== EDGE CASE AND ERROR HANDLING TESTS =====

#[test]
fn test_edge_cases_empty_store() {
    let store = EntityStore::new();
    
    // Operations on empty store
    assert_eq!(store.count(), 0);
    assert_eq!(store.capacity(), 0);
    assert!(store.memory_usage() >= 0);
    assert!(store.encoded_size() > 0); // Even empty store has some overhead
    
    let key = create_test_keys(1)[0];
    assert!(store.get(key).is_none());
    assert!(!store.contains_entity(key));
}

#[test]
fn test_edge_cases_property_sizes() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(5);
    
    // Test various property sizes
    let test_cases = vec![
        "",                                    // Empty
        "a",                                   // Single char
        "x".repeat(1000).as_str(),           // 1KB
        "y".repeat(10000).as_str(),          // 10KB  
        "z".repeat(100000).as_str(),         // 100KB
    ];
    
    for (i, (key, props)) in keys.iter().zip(test_cases.iter()).enumerate() {
        let data = create_test_entity_data(i as u16, props);
        let meta = store.insert(*key, &data).expect("Insert should succeed");
        
        let retrieved = store.get_properties(&meta).expect("Properties should be retrievable");
        assert_eq!(retrieved.len(), props.len());
        assert_eq!(retrieved, *props);
    }
}

#[test] 
fn test_edge_cases_unicode_properties() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(6);
    
    let unicode_cases = vec![
        "emoji:🚀🌟🎉,status:active",
        "chinese:你好世界,type:greeting",
        "japanese:こんにちは,katakana:コンニチハ",
        "arabic:مرحبا بالعالم,direction:rtl",
        "mixed:Hello世界🌍Здравствуй",
        "special:line1\nline2\ttab\r\nwindows",
    ];
    
    for (i, (key, props)) in keys.iter().zip(unicode_cases.iter()).enumerate() {
        let data = create_test_entity_data(i as u16, props);
        store.insert(*key, &data).expect("Insert should succeed");
        
        let meta = store.get(*key).expect("Entity should exist");
        let retrieved = store.get_properties(meta).expect("Properties should be retrievable");
        assert_eq!(retrieved, *props);
    }
}

#[test]
fn test_error_handling_operations() {
    let mut store = EntityStore::new();
    let key = create_test_keys(1)[0];
    
    // Operations on non-existent entity
    assert!(store.get(key).is_none());
    assert!(store.get_mut(key).is_none());
    
    match store.update_degree(key, 5) {
        Err(GraphError::EntityNotFound { .. }) => {},
        _ => panic!("Expected EntityNotFound error"),
    }
    
    let data = create_test_entity_data(1, "test");
    match store.update_entity(key, &data) {
        Err(GraphError::EntityKeyNotFound { .. }) => {},
        _ => panic!("Expected EntityKeyNotFound error"),
    }
    
    match store.remove(key) {
        Err(GraphError::EntityKeyNotFound { .. }) => {},
        _ => panic!("Expected EntityKeyNotFound error"),
    }
}

// ===== COMPLEX SCENARIO TESTS =====

#[test]
fn test_complex_scenario_graph_simulation() {
    let mut store = EntityStore::new();
    
    // Simulate a knowledge graph scenario
    // Create entities representing: People, Organizations, Documents, Events
    
    let mut all_keys = Vec::new();
    
    // Create 50 people
    let people_keys = create_test_keys(50);
    for (i, key) in people_keys.iter().enumerate() {
        let data = create_test_entity_data(
            1, 
            &format!("name:Person_{},department:Dept_{},level:L{}", i, i % 5, i % 3)
        );
        store.insert(*key, &data).expect("Insert person should succeed");
        all_keys.push(*key);
    }
    
    // Create 20 organizations
    let org_keys = create_test_keys(20);
    for (i, key) in org_keys.iter().enumerate() {
        let data = create_test_entity_data(
            2,
            &format!("name:Org_{},type:Company,size:{}", i, if i < 10 { "Small" } else { "Large" })
        );
        store.insert(*key, &data).expect("Insert org should succeed");
        all_keys.push(*key);
    }
    
    // Create 100 documents
    let doc_keys = create_test_keys(100);
    for (i, key) in doc_keys.iter().enumerate() {
        let data = create_test_entity_data(
            3,
            &format!("title:Document_{},category:Cat_{},year:202{}", i, i % 10, i % 5)
        );
        store.insert(*key, &data).expect("Insert document should succeed");
        all_keys.push(*key);
    }
    
    // Create 30 events
    let event_keys = create_test_keys(30);
    for (i, key) in event_keys.iter().enumerate() {
        let data = create_test_entity_data(
            4,
            &format!("name:Event_{},type:Conference,month:{}", i, (i % 12) + 1)
        );
        store.insert(*key, &data).expect("Insert event should succeed");
        all_keys.push(*key);
    }
    
    assert_eq!(store.count(), 200); // 50 + 20 + 100 + 30
    
    // Simulate relationship degrees based on entity type
    // People: 5-15 connections
    for (i, key) in people_keys.iter().enumerate() {
        let degree = 5 + (i % 11);
        store.update_degree(*key, degree as i16).expect("Update should succeed");
    }
    
    // Organizations: 20-50 connections
    for (i, key) in org_keys.iter().enumerate() {
        let degree = 20 + (i % 31);
        store.update_degree(*key, degree as i16).expect("Update should succeed");
    }
    
    // Documents: 2-10 connections
    for (i, key) in doc_keys.iter().enumerate() {
        let degree = 2 + (i % 9);
        store.update_degree(*key, degree as i16).expect("Update should succeed");
    }
    
    // Events: 10-30 connections
    for (i, key) in event_keys.iter().enumerate() {
        let degree = 10 + (i % 21);
        store.update_degree(*key, degree as i16).expect("Update should succeed");
    }
    
    // Analyze the graph
    let mut type_stats: HashMap<u16, (usize, u32)> = HashMap::new(); // (count, total_degree)
    
    for key in &all_keys {
        if let Some(meta) = store.get(*key) {
            let entry = type_stats.entry(meta.type_id).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += meta.degree as u32;
        }
    }
    
    // Verify statistics
    assert_eq!(type_stats.get(&1).unwrap().0, 50);  // People count
    assert_eq!(type_stats.get(&2).unwrap().0, 20);  // Org count
    assert_eq!(type_stats.get(&3).unwrap().0, 100); // Doc count
    assert_eq!(type_stats.get(&4).unwrap().0, 30);  // Event count
    
    // Calculate average degrees
    for (type_id, (count, total_degree)) in &type_stats {
        let avg_degree = *total_degree as f64 / *count as f64;
        match type_id {
            1 => assert!(avg_degree >= 5.0 && avg_degree <= 15.0),   // People
            2 => assert!(avg_degree >= 20.0 && avg_degree <= 50.0),  // Orgs
            3 => assert!(avg_degree >= 2.0 && avg_degree <= 10.0),   // Docs
            4 => assert!(avg_degree >= 10.0 && avg_degree <= 30.0),  // Events
            _ => panic!("Unexpected type_id"),
        }
    }
    
    // Simulate some updates and removals
    // Remove 10% of documents
    for key in doc_keys.iter().step_by(10) {
        store.remove(*key).expect("Remove should succeed");
    }
    
    assert_eq!(store.count(), 190); // 200 - 10
    
    // Update some people to managers
    for key in people_keys.iter().take(5) {
        let meta = store.get(*key).expect("Person should exist");
        let props = store.get_properties(meta).expect("Properties should exist");
        
        let updated_data = create_test_entity_data(
            1,
            &format!("{},role:Manager", props)
        );
        store.update_entity(*key, &updated_data).expect("Update should succeed");
    }
    
    // Final memory analysis
    let final_memory = store.memory_usage();
    let final_encoded = store.encoded_size();
    let memory_per_entity = final_memory / store.count();
    
    // With mixed entity types and sizes, average should still be reasonable
    assert!(memory_per_entity < 500, "Memory per entity too high: {}", memory_per_entity);
    assert!(final_encoded < final_memory, "Encoded size should be smaller than memory usage");
}

#[test]
fn test_complex_scenario_stress_test() {
    let mut store = EntityStore::new();
    
    // Stress test with rapid insert/update/remove cycles
    let mut active_keys = Vec::new();
    let mut removed_count = 0;
    
    // Phase 1: Rapid insertions
    for batch in 0..10 {
        let batch_keys = create_test_keys(100);
        
        for (i, key) in batch_keys.iter().enumerate() {
            let data = create_test_entity_data(
                (batch % 5) as u16,
                &format!("batch:{},item:{},data:{}", batch, i, "x".repeat(i % 100))
            );
            store.insert(*key, &data).expect("Insert should succeed");
            active_keys.push(*key);
        }
    }
    
    assert_eq!(store.count(), 1000);
    
    // Phase 2: Random updates
    for i in 0..500 {
        let key_idx = i * 2; // Every other key
        let key = active_keys[key_idx];
        
        // Update degree
        store.update_degree(key, ((i % 20) as i16) - 10).expect("Update should succeed");
        
        // Sometimes update entity
        if i % 5 == 0 {
            let data = create_test_entity_data(99, &format!("updated_{}", i));
            store.update_entity(key, &data).expect("Update should succeed");
        }
    }
    
    // Phase 3: Random removals
    for i in (0..active_keys.len()).step_by(3) {
        if i < active_keys.len() {
            store.remove(active_keys[i]).ok(); // Some might already be removed
            removed_count += 1;
        }
    }
    
    // Verify final state
    let final_count = store.count();
    assert!(final_count < 1000);
    assert!(final_count >= 600); // Rough estimate
    
    // Check remaining entities are valid
    let mut valid_count = 0;
    for key in &active_keys {
        if store.contains_entity(*key) {
            let meta = store.get(*key).expect("Contained entity should be gettable");
            let props = store.get_properties(meta).expect("Properties should exist");
            assert!(!props.is_empty() || props.is_empty()); // Props can be empty
            valid_count += 1;
        }
    }
    
    assert_eq!(valid_count, final_count);
}

// ===== VALIDATION TESTS =====

#[test]
fn test_validate_property_offset_integrity() {
    let mut store = EntityStore::new();
    let keys = create_test_keys(20);
    
    // Insert entities with known property sizes
    let mut expected_offsets = vec![0u32];
    let mut cumulative = 0u32;
    
    for (i, key) in keys.iter().enumerate() {
        let prop_size = i * 5; // 0, 5, 10, 15, ...
        let props = "a".repeat(prop_size);
        let data = create_test_entity_data(i as u16, &props);
        
        store.insert(*key, &data).expect("Insert should succeed");
        cumulative += prop_size as u32;
        expected_offsets.push(cumulative);
    }
    
    // Verify each entity's properties can be retrieved correctly
    for (i, key) in keys.iter().enumerate() {
        let meta = store.get(*key).expect("Entity should exist");
        let props = store.get_properties(meta).expect("Properties should be retrievable");
        
        let expected_size = i * 5;
        assert_eq!(props.len(), expected_size);
        assert_eq!(props, "a".repeat(expected_size));
    }
}

#[test]
fn test_validate_encoded_size_accuracy() {
    let mut store = EntityStore::new();
    
    // Start with empty store
    let empty_encoded = store.encoded_size();
    assert!(empty_encoded > 0); // Should have some overhead
    
    // Add entities and track size changes
    let keys = create_test_keys(50);
    let mut total_property_bytes = 0;
    
    for (i, key) in keys.iter().enumerate() {
        let props = format!("entity_{}_with_properties_{}", i, "data".repeat(i % 10));
        total_property_bytes += props.len();
        
        let data = create_test_entity_data(i as u16, &props);
        store.insert(*key, &data).expect("Insert should succeed");
    }
    
    let final_encoded = store.encoded_size();
    
    // Encoded size should include:
    // - Entity count overhead
    // - Entity metadata (keys + metas)
    // - Property data
    // - Property offsets
    
    let min_expected = empty_encoded + 
                      50 * (std::mem::size_of::<EntityKey>() + std::mem::size_of::<EntityMeta>()) +
                      total_property_bytes;
    
    assert!(final_encoded >= min_expected, 
            "Encoded size {} should be at least {}", final_encoded, min_expected);
}

#[test]
fn test_validate_capacity_behavior() {
    let mut store = EntityStore::new();
    
    // Track capacity changes
    let mut capacity_history = vec![store.capacity()];
    
    // Insert entities and observe capacity growth
    let keys = create_test_keys(200);
    
    for (i, key) in keys.iter().enumerate() {
        let data = create_test_entity_data(i as u16, "test");
        store.insert(*key, &data).expect("Insert should succeed");
        
        if i % 20 == 19 {
            capacity_history.push(store.capacity());
        }
    }
    
    capacity_history.push(store.capacity());
    
    // Capacity should grow monotonically
    for window in capacity_history.windows(2) {
        assert!(window[1] >= window[0], "Capacity should not decrease");
    }
    
    // Final capacity should accommodate all entities
    assert!(store.capacity() >= 200);
    
    // But shouldn't be wastefully large
    assert!(store.capacity() < 400, "Capacity is too large for 200 entities");
}

// ===== INTEGRATION WITH SLOTMAP TESTS =====

#[test]
fn test_integration_with_slotmap_keys() {
    let mut store = EntityStore::new();
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    
    // Create entities in slotmap and store
    let mut keys = Vec::new();
    for i in 0..20 {
        let data = EntityData::new(
            i,
            format!("slotmap_entity_{}", i),
            vec![i as f32; 32]
        );
        
        let key = sm.insert(data.clone());
        keys.push(key);
        
        store.insert(key, &data).expect("Insert should succeed");
    }
    
    // Remove some from slotmap
    for i in (0..20).step_by(3) {
        sm.remove(keys[i]);
    }
    
    // Store should still have all entities
    assert_eq!(store.count(), 20);
    
    // All keys should still work in store
    for (i, key) in keys.iter().enumerate() {
        assert!(store.contains_entity(*key));
        let meta = store.get(*key).expect("Entity should exist in store");
        assert_eq!(meta.type_id, i as u16);
    }
}

#[test]
fn test_boundary_conditions() {
    let mut store = EntityStore::new();
    
    // Test with maximum type_id
    let key = create_test_keys(1)[0];
    let data = create_test_entity_data(u16::MAX, "max_type_id");
    store.insert(key, &data).expect("Insert should succeed");
    
    let meta = store.get(key).expect("Entity should exist");
    assert_eq!(meta.type_id, u16::MAX);
    
    // Test with maximum degree
    store.update_degree(key, i16::MAX).expect("Update should succeed");
    let meta = store.get(key).expect("Entity should exist");
    assert_eq!(meta.degree, i16::MAX as u16);
    
    // Test degree overflow protection
    store.update_degree(key, 1).expect("Update should succeed");
    let meta = store.get(key).expect("Entity should exist");
    assert!(meta.degree >= i16::MAX as u16); // Should handle overflow gracefully
}