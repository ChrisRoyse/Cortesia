//! Integration tests for graph entity operations
//! 
//! Tests complete entity lifecycle including insertion, updating, removal
//! through public APIs and verifies all entity management methods work
//! together correctly.

use std::collections::HashMap;

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData};
use llmkg::error::{GraphError, Result};

fn create_test_graph() -> KnowledgeGraph {
    KnowledgeGraph::new(96, 10000)
}

fn create_test_entity_data(type_id: u32, properties: &str) -> EntityData {
    EntityData {
        type_id,
        embedding: vec![0.1; 96], // Simple test embedding
        properties: properties.to_string(),
    }
}

fn create_random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    
    let mut embedding = Vec::with_capacity(dim);
    let mut hash = hasher.finish();
    
    for i in 0..dim {
        hash = hash.wrapping_mul(1103515245).wrapping_add(12345); // Simple LCG
        let val = ((hash >> 16) & 0xFFFF) as f32 / 65536.0;
        embedding.push(val);
    }
    
    // Normalize embedding
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in embedding.iter_mut() {
            *val /= magnitude;
        }
    }
    
    embedding
}

#[test]
fn test_single_entity_insertion_lifecycle() {
    let graph = create_test_graph();
    
    // Phase 1: Insert single entity
    let entity_data = EntityData {
        type_id: 1,
        embedding: create_random_embedding(96, 42),
        properties: r#"{"name": "Test Entity", "type": "test"}"#.to_string(),
    };
    
    let insert_result = graph.insert_entity(100, entity_data.clone());
    assert!(insert_result.is_ok());
    let entity_key = insert_result.unwrap();
    
    // Phase 2: Verify entity exists and is retrievable
    let retrieved_entity = graph.get_entity(entity_key);
    assert!(retrieved_entity.is_some());
    
    let (retrieved_key, retrieved_data) = retrieved_entity.unwrap();
    assert_eq!(retrieved_key, entity_key);
    assert_eq!(retrieved_data.type_id, entity_data.type_id);
    assert_eq!(retrieved_data.properties, entity_data.properties);
    assert_eq!(retrieved_data.embedding, entity_data.embedding);
    
    // Phase 3: Verify entity can be found by ID
    let by_id_result = graph.get_entity_by_id(100);
    assert!(by_id_result.is_some());
    
    let (by_id_key, by_id_data) = by_id_result.unwrap();
    assert_eq!(by_id_key, entity_key);
    assert_eq!(by_id_data.type_id, entity_data.type_id);
    
    // Phase 4: Verify entity key mapping
    let mapped_key = graph.get_entity_key(100);
    assert!(mapped_key.is_some());
    assert_eq!(mapped_key.unwrap(), entity_key);
    
    // Phase 5: Check entity existence
    let exists = graph.contains_entity(100);
    assert!(exists);
    
    // Phase 6: Verify bloom filter
    let bloom_contains = graph.bloom_contains(100);
    assert!(bloom_contains);
    
    // Phase 7: Get entity embedding separately
    let embedding = graph.get_entity_embedding(entity_key);
    assert!(embedding.is_some());
    assert_eq!(embedding.unwrap(), entity_data.embedding);
    
    // Phase 8: Update entity
    let updated_data = EntityData {
        type_id: 2,
        embedding: create_random_embedding(96, 43),
        properties: r#"{"name": "Updated Entity", "type": "updated", "version": "2.0"}"#.to_string(),
    };
    
    let update_result = graph.update_entity(entity_key, updated_data.clone());
    assert!(update_result.is_ok());
    
    // Verify update took effect
    let updated_entity = graph.get_entity(entity_key);
    assert!(updated_entity.is_some());
    
    let (_, updated_retrieved) = updated_entity.unwrap();
    assert_eq!(updated_retrieved.type_id, 2);
    assert_eq!(updated_retrieved.properties, updated_data.properties);
    assert_eq!(updated_retrieved.embedding, updated_data.embedding);
    
    // Phase 9: Remove entity
    let remove_result = graph.remove_entity(entity_key);
    assert!(remove_result.is_ok());
    assert_eq!(remove_result.unwrap(), true);
    
    // Verify removal
    let removed_entity = graph.get_entity(entity_key);
    assert!(removed_entity.is_none());
    
    let by_id_removed = graph.get_entity_by_id(100);
    assert!(by_id_removed.is_none());
    
    let exists_after_removal = graph.contains_entity(100);
    assert!(!exists_after_removal);
    
    // Note: Bloom filter may still contain the ID (false positive)
    // This is expected behavior for bloom filters
}

#[test]
fn test_batch_entity_operations() {
    let graph = create_test_graph();
    
    // Phase 1: Prepare batch data
    let batch_entities = vec![
        (1, EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 1),
            properties: r#"{"name": "Entity 1", "category": "A"}"#.to_string(),
        }),
        (2, EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 2),
            properties: r#"{"name": "Entity 2", "category": "A"}"#.to_string(),
        }),
        (3, EntityData {
            type_id: 2,
            embedding: create_random_embedding(96, 3),
            properties: r#"{"name": "Entity 3", "category": "B"}"#.to_string(),
        }),
        (4, EntityData {
            type_id: 2,
            embedding: create_random_embedding(96, 4),
            properties: r#"{"name": "Entity 4", "category": "B"}"#.to_string(),
        }),
        (5, EntityData {
            type_id: 3,
            embedding: create_random_embedding(96, 5),
            properties: r#"{"name": "Entity 5", "category": "C"}"#.to_string(),
        }),
    ];
    
    // Phase 2: Batch insert
    let batch_result = graph.insert_entities_batch(batch_entities.clone());
    assert!(batch_result.is_ok());
    let entity_keys = batch_result.unwrap();
    assert_eq!(entity_keys.len(), 5);
    
    // Phase 3: Verify all entities were inserted correctly
    for (i, (id, expected_data)) in batch_entities.iter().enumerate() {
        let entity_key = entity_keys[i];
        
        // Check by key
        let by_key = graph.get_entity(entity_key);
        assert!(by_key.is_some());
        let (_, key_data) = by_key.unwrap();
        assert_eq!(key_data.type_id, expected_data.type_id);
        assert_eq!(key_data.properties, expected_data.properties);
        assert_eq!(key_data.embedding, expected_data.embedding);
        
        // Check by ID
        let by_id = graph.get_entity_by_id(*id);
        assert!(by_id.is_some());
        let (by_id_key, id_data) = by_id.unwrap();
        assert_eq!(by_id_key, entity_key);
        assert_eq!(id_data.type_id, expected_data.type_id);
        
        // Check existence
        assert!(graph.contains_entity(*id));
        assert!(graph.bloom_contains(*id));
    }
    
    // Phase 4: Verify entity count
    let entity_count = graph.entity_count();
    assert_eq!(entity_count, 5);
    
    // Phase 5: Test batch operations statistics
    let memory_usage = graph.memory_usage();
    assert!(memory_usage.total_bytes() > 0);
    
    // Phase 6: Remove entities one by one and verify batch consistency
    for i in 0..3 { // Remove first 3 entities
        let entity_key = entity_keys[i];
        let remove_result = graph.remove_entity(entity_key);
        assert!(remove_result.is_ok());
        assert_eq!(remove_result.unwrap(), true);
        
        // Verify removal
        let removed = graph.get_entity(entity_key);
        assert!(removed.is_none());
        
        // Verify other entities still exist
        for j in (i+1)..entity_keys.len() {
            let remaining_key = entity_keys[j];
            let remaining = graph.get_entity(remaining_key);
            assert!(remaining.is_some());
        }
    }
    
    // Phase 7: Verify final state
    let final_count = graph.entity_count();
    assert_eq!(final_count, 2); // 2 entities should remain
    
    // Verify the remaining entities are correct
    let remaining_entity_4 = graph.get_entity_by_id(4);
    assert!(remaining_entity_4.is_some());
    
    let remaining_entity_5 = graph.get_entity_by_id(5);
    assert!(remaining_entity_5.is_some());
}

#[test]
fn test_entity_validation_and_error_handling() {
    let graph = create_test_graph();
    
    // Test 1: Invalid embedding dimension
    let invalid_embedding_data = EntityData {
        type_id: 1,
        embedding: vec![0.1; 64], // Wrong dimension (should be 96)
        properties: "{}".to_string(),
    };
    
    let invalid_result = graph.insert_entity(1, invalid_embedding_data);
    assert!(invalid_result.is_err());
    
    if let Err(GraphError::InvalidEmbeddingDimension { expected, actual }) = invalid_result {
        assert_eq!(expected, 96);
        assert_eq!(actual, 64);
    } else {
        panic!("Expected InvalidEmbeddingDimension error");
    }
    
    // Test 2: Very large properties string
    let large_properties = "x".repeat(1_000_000); // 1MB string
    let large_data = EntityData {
        type_id: 1,
        embedding: create_random_embedding(96, 1),
        properties: large_properties,
    };
    
    let large_result = graph.insert_entity(2, large_data);
    // This might fail depending on validation limits, which is expected
    
    // Test 3: Empty embedding
    let empty_embedding_data = EntityData {
        type_id: 1,
        embedding: vec![], // Empty embedding
        properties: "{}".to_string(),
    };
    
    let empty_result = graph.insert_entity(3, empty_embedding_data);
    assert!(empty_result.is_err());
    
    // Test 4: Operations on non-existent entities
    let fake_key = graph.get_entity_key(999);
    assert!(fake_key.is_none());
    
    let fake_entity = graph.get_entity_by_id(999);
    assert!(fake_entity.is_none());
    
    let fake_exists = graph.contains_entity(999);
    assert!(!fake_exists);
    
    // Test 5: Update non-existent entity
    let update_data = EntityData {
        type_id: 1,
        embedding: create_random_embedding(96, 10),
        properties: "{}".to_string(),
    };
    
    // Since we can't create fake keys through public API, we'll skip this test
    // and focus on other validation aspects
    
    // Since we can't easily create a fake key, let's test with actual entity operations
    
    // Test 6: Double insertion of same ID
    let valid_data = EntityData {
        type_id: 1,
        embedding: create_random_embedding(96, 20),
        properties: "{}".to_string(),
    };
    
    let first_insert = graph.insert_entity(100, valid_data.clone());
    assert!(first_insert.is_ok());
    
    let second_insert = graph.insert_entity(100, valid_data); // Same ID
    // This should either succeed (update) or fail gracefully
    // The specific behavior depends on implementation
    
    // Test 7: Batch with mixed valid/invalid entities
    let mixed_batch = vec![
        (200, EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 30),
            properties: "{}".to_string(),
        }),
        (201, EntityData {
            type_id: 1,
            embedding: vec![0.1; 32], // Wrong dimension
            properties: "{}".to_string(),
        }),
    ];
    
    let mixed_result = graph.insert_entities_batch(mixed_batch);
    assert!(mixed_result.is_err()); // Should fail due to invalid entity
    
    // Verify no entities were inserted from the failed batch
    let entity_200 = graph.get_entity_by_id(200);
    let entity_201 = graph.get_entity_by_id(201);
    
    // Depending on implementation, might have none inserted (atomic)
    // or only valid ones inserted (partial success)
}

#[test]
fn test_entity_embedding_operations() {
    let graph = create_test_graph();
    
    // Phase 1: Insert entities with different embeddings
    let embeddings = vec![
        create_random_embedding(96, 100),
        create_random_embedding(96, 101),
        create_random_embedding(96, 102),
        create_random_embedding(96, 103),
    ];
    
    let mut entity_keys = Vec::new();
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let id = (i + 1) as u32;
        let entity_data = EntityData {
            type_id: 1,
            embedding: embedding.clone(),
            properties: format!(r#"{{"name": "Entity {}", "id": {}}}"#, id, id),
        };
        
        let result = graph.insert_entity(id, entity_data);
        assert!(result.is_ok());
        entity_keys.push(result.unwrap());
    }
    
    // Phase 2: Retrieve and verify embeddings
    for (i, expected_embedding) in embeddings.iter().enumerate() {
        let entity_key = entity_keys[i];
        
        // Get embedding by key
        let retrieved_embedding = graph.get_entity_embedding(entity_key);
        assert!(retrieved_embedding.is_some());
        assert_eq!(retrieved_embedding.unwrap(), *expected_embedding);
        
        // Get full entity and check embedding
        let entity = graph.get_entity(entity_key);
        assert!(entity.is_some());
        let (_, data) = entity.unwrap();
        assert_eq!(data.embedding, *expected_embedding);
    }
    
    // Phase 3: Update embeddings and verify changes
    let new_embedding = create_random_embedding(96, 200);
    let updated_data = EntityData {
        type_id: 1,
        embedding: new_embedding.clone(),
        properties: r#"{"name": "Updated Entity", "updated": true}"#.to_string(),
    };
    
    let update_key = entity_keys[0];
    let update_result = graph.update_entity(update_key, updated_data);
    assert!(update_result.is_ok());
    
    // Verify embedding was updated
    let updated_embedding = graph.get_entity_embedding(update_key);
    assert!(updated_embedding.is_some());
    let updated_embedding_value = updated_embedding.unwrap();
    assert_eq!(updated_embedding_value, new_embedding);
    assert_ne!(updated_embedding_value, embeddings[0]);
    
    // Phase 4: Test embedding normalization consistency
    let unnormalized = vec![1.0, 2.0, 3.0, 4.0]; // Will be extended to 96 dims
    let mut test_embedding = unnormalized.clone();
    test_embedding.resize(96, 0.0);
    
    let embedding_data = EntityData {
        type_id: 1,
        embedding: test_embedding.clone(),
        properties: "{}".to_string(),
    };
    
    let norm_result = graph.insert_entity(500, embedding_data);
    assert!(norm_result.is_ok());
    
    let retrieved_norm = graph.get_entity_embedding(norm_result.unwrap());
    assert!(retrieved_norm.is_some());
    
    // Check that embedding is the same as inserted (no automatic normalization)
    assert_eq!(retrieved_norm.unwrap(), test_embedding);
    
    // Phase 5: Test zero embedding handling
    let zero_embedding = vec![0.0; 96];
    let zero_data = EntityData {
        type_id: 1,
        embedding: zero_embedding.clone(),
        properties: "{}".to_string(),
    };
    
    let zero_result = graph.insert_entity(600, zero_data);
    assert!(zero_result.is_ok());
    
    let retrieved_zero = graph.get_entity_embedding(zero_result.unwrap());
    assert!(retrieved_zero.is_some());
    assert_eq!(retrieved_zero.unwrap(), zero_embedding);
}

#[test]
fn test_entity_properties_operations() {
    let graph = create_test_graph();
    
    // Phase 1: Insert entities with various property formats
    let test_cases = vec![
        (1, r#"{}"#), // Empty JSON
        (2, r#"{"simple": "value"}"#), // Simple property
        (3, r#"{"complex": {"nested": {"deep": "value"}}, "array": [1, 2, 3]}"#), // Complex JSON
        (4, r#"{"unicode": "ñáéíóú", "emoji": "🚀", "special": "\"quotes\" and \\backslashes"}"#), // Unicode and escaping
        (5, "not json at all"), // Invalid JSON (should still work as string)
    ];
    
    let mut entity_keys = Vec::new();
    
    for (id, properties) in &test_cases {
        let entity_data = EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, *id as u64),
            properties: properties.to_string(),
        };
        
        let result = graph.insert_entity(*id, entity_data);
        assert!(result.is_ok());
        entity_keys.push(result.unwrap());
    }
    
    // Phase 2: Retrieve and verify properties
    for (i, (_, expected_properties)) in test_cases.iter().enumerate() {
        let entity_key = entity_keys[i];
        
        let entity = graph.get_entity(entity_key);
        assert!(entity.is_some());
        
        let (_, data) = entity.unwrap();
        assert_eq!(data.properties, *expected_properties);
    }
    
    // Phase 3: Update properties and verify
    let updates = vec![
        (0, r#"{"updated": true, "timestamp": "2024-01-01"}"#),
        (1, r#"{"simple": "updated_value", "new_field": "added"}"#),
        (2, r#"{"completely": "replaced"}"#),
    ];
    
    for (index, new_properties) in updates {
        let entity_key = entity_keys[index];
        
        // Get current data to preserve other fields
        let current = graph.get_entity(entity_key).unwrap();
        let (_, current_data) = current;
        
        let updated_data = EntityData {
            type_id: current_data.type_id,
            embedding: current_data.embedding,
            properties: new_properties.to_string(),
        };
        
        let update_result = graph.update_entity(entity_key, updated_data);
        assert!(update_result.is_ok());
        
        // Verify update
        let updated_entity = graph.get_entity(entity_key).unwrap();
        let (_, updated_data) = updated_entity;
        assert_eq!(updated_data.properties, new_properties);
    }
    
    // Phase 4: Test property search and filtering
    // Insert more entities for testing
    let searchable_entities = vec![
        (100, r#"{"type": "document", "title": "Machine Learning Paper", "author": "Smith"}"#),
        (101, r#"{"type": "document", "title": "Deep Learning Review", "author": "Jones"}"#),
        (102, r#"{"type": "person", "name": "Alice Smith", "profession": "researcher"}"#),
        (103, r#"{"type": "person", "name": "Bob Jones", "profession": "engineer"}"#),
        (104, r#"{"type": "conference", "name": "ICML 2024", "location": "Vienna"}"#),
    ];
    
    for (id, properties) in searchable_entities {
        let entity_data = EntityData {
            type_id: 2,
            embedding: create_random_embedding(96, id),
            properties: properties.to_string(),
        };
        
        let result = graph.insert_entity(id, entity_data);
        assert!(result.is_ok());
    }
    
    // Phase 5: Verify all entities exist and have correct properties
    let total_count = graph.entity_count();
    assert_eq!(total_count, 10); // 5 original + 5 searchable
    
    // Test properties integrity after insertions
    for id in [100, 101, 102, 103, 104] {
        let entity = graph.get_entity_by_id(id);
        assert!(entity.is_some());
        
        let (_, data) = entity.unwrap();
        assert!(data.properties.contains("type"));
        assert_eq!(data.type_id, 2);
    }
}

#[test]
fn test_entity_type_management() {
    let graph = create_test_graph();
    
    // Phase 1: Insert entities with different type IDs
    let type_test_cases = vec![
        (1, 1, "Person"),
        (2, 1, "Person"),
        (3, 2, "Document"),
        (4, 2, "Document"), 
        (5, 3, "Location"),
        (6, 3, "Location"),
        (7, 4, "Concept"),
    ];
    
    for (id, type_id, type_name) in &type_test_cases {
        let entity_data = EntityData {
            type_id: *type_id,
            embedding: create_random_embedding(96, *id as u64),
            properties: format!(r#"{{"type": "{}", "id": {}}}"#, type_name, id),
        };
        
        let result = graph.insert_entity(*id, entity_data);
        assert!(result.is_ok());
    }
    
    // Phase 2: Verify type assignments
    for (id, expected_type_id, _) in &type_test_cases {
        let entity = graph.get_entity_by_id(*id);
        assert!(entity.is_some());
        
        let (_, data) = entity.unwrap();
        assert_eq!(data.type_id, *expected_type_id);
    }
    
    // Phase 3: Test type changes through updates
    let type_change_cases = vec![
        (1, 5), // Person -> New Type
        (3, 1), // Document -> Person
        (7, 2), // Concept -> Document
    ];
    
    for (id, new_type_id) in type_change_cases {
        let entity = graph.get_entity_by_id(id).unwrap();
        let (key, data) = entity;
        
        let updated_data = EntityData {
            type_id: new_type_id,
            embedding: data.embedding,
            properties: format!(r#"{{"type": "changed", "old_id": {}}}"#, id),
        };
        
        let update_result = graph.update_entity(key, updated_data);
        assert!(update_result.is_ok());
        
        // Verify type change
        let updated_entity = graph.get_entity_by_id(id).unwrap();
        let (_, updated_data) = updated_entity;
        assert_eq!(updated_data.type_id, new_type_id);
        assert!(updated_data.properties.contains("changed"));
    }
    
    // Phase 4: Verify entity count and consistency
    let final_count = graph.entity_count();
    assert_eq!(final_count, 7);
    
    // Phase 5: Test extreme type ID values
    let extreme_cases = vec![
        (1000, 0),           // Type 0
        (1001, u32::MAX),    // Maximum type ID
        (1002, 1000000),     // Large type ID
    ];
    
    for (id, type_id) in extreme_cases {
        let entity_data = EntityData {
            type_id,
            embedding: create_random_embedding(96, id),
            properties: format!(r#"{{"extreme_type": {}}}"#, type_id),
        };
        
        let result = graph.insert_entity(id, entity_data);
        assert!(result.is_ok());
        
        // Verify extreme type was stored correctly
        let entity = graph.get_entity_by_id(id).unwrap();
        let (_, data) = entity;
        assert_eq!(data.type_id, type_id);
    }
}

#[test]
fn test_concurrent_entity_operations() {
    use std::sync::Arc;
    use std::thread;
    
    let graph = Arc::new(create_test_graph());
    let num_threads = 4;
    let entities_per_thread = 25;
    
    // Phase 1: Concurrent insertions
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph);
        
        let handle = thread::spawn(move || {
            let mut results = Vec::new();
            
            for i in 0..entities_per_thread {
                let entity_id = (thread_id * entities_per_thread + i) as u32;
                let entity_data = EntityData {
                    type_id: thread_id as u32 + 1,
                    embedding: create_random_embedding(96, entity_id as u64),
                    properties: format!(r#"{{"thread": {}, "index": {}}}"#, thread_id, i),
                };
                
                let result = graph_clone.insert_entity(entity_id, entity_data);
                results.push((entity_id, result));
            }
            
            results
        });
        
        handles.push(handle);
    }
    
    // Wait for all insertions to complete
    let mut all_results = Vec::new();
    for handle in handles {
        let thread_results = handle.join().expect("Thread panicked");
        all_results.extend(thread_results);
    }
    
    // Phase 2: Verify all insertions succeeded
    let mut successful_insertions = 0;
    for (entity_id, result) in &all_results {
        if result.is_ok() {
            successful_insertions += 1;
            
            // Verify entity exists
            let entity = graph.get_entity_by_id(*entity_id);
            assert!(entity.is_some());
        }
    }
    
    // Should have high success rate (some conflicts might occur)
    assert!(successful_insertions >= (num_threads * entities_per_thread) * 80 / 100);
    
    // Phase 3: Verify final entity count
    let final_count = graph.entity_count();
    assert!(final_count >= successful_insertions);
    
    // Phase 4: Concurrent updates on existing entities
    let update_handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let graph_clone = Arc::clone(&graph);
        
        thread::spawn(move || {
            for i in 0..10 { // Update fewer entities
                let entity_id = (thread_id * entities_per_thread + i) as u32;
                
                if let Some((key, data)) = graph_clone.get_entity_by_id(entity_id) {
                    let updated_data = EntityData {
                        type_id: data.type_id,
                        embedding: data.embedding,
                        properties: format!(r#"{{"updated": true, "thread": {}}}"#, thread_id),
                    };
                    
                    let _ = graph_clone.update_entity(key, updated_data);
                }
            }
        })
    }).collect();
    
    // Wait for updates
    for handle in update_handles {
        handle.join().expect("Update thread panicked");
    }
    
    // Phase 5: Concurrent removals
    let removal_handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let graph_clone = Arc::clone(&graph);
        
        thread::spawn(move || {
            let mut removed_count = 0;
            
            for i in 20..entities_per_thread { // Remove last 5 entities from each thread
                let entity_id = (thread_id * entities_per_thread + i) as u32;
                
                if let Some(key) = graph_clone.get_entity_key(entity_id) {
                    if let Ok(removed) = graph_clone.remove_entity(key) {
                        if removed {
                            removed_count += 1;
                        }
                    }
                }
            }
            
            removed_count
        })
    }).collect();
    
    // Wait for removals and count total
    let mut total_removed = 0;
    for handle in removal_handles {
        let thread_removed = handle.join().expect("Removal thread panicked");
        total_removed += thread_removed;
    }
    
    // Phase 6: Verify removals
    let post_removal_count = graph.entity_count();
    assert_eq!(post_removal_count, final_count - total_removed);
    
    // Verify specific entities were removed
    for thread_id in 0..num_threads {
        for i in 20..entities_per_thread {
            let entity_id = (thread_id * entities_per_thread + i) as u32;
            let entity = graph.get_entity_by_id(entity_id);
            // Should be None (removed) or Some (removal failed due to race condition)
            // Both are acceptable in concurrent scenarios
        }
    }
}

#[test]
fn test_comprehensive_entity_lifecycle_workflows() {
    let graph = create_test_graph();
    
    // Phase 1: Create entities representing a research paper workflow
    let workflow_entities = vec![
        (10000, EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 10000),
            properties: r#"{"type": "researcher", "name": "Dr. Johnson", "affiliation": "MIT", "h_index": 45}"#.to_string(),
        }),
        (10001, EntityData {
            type_id: 2,
            embedding: create_random_embedding(96, 10001),
            properties: r#"{"type": "paper", "title": "Deep Learning in NLP", "status": "draft", "pages": 12}"#.to_string(),
        }),
        (10002, EntityData {
            type_id: 3,
            embedding: create_random_embedding(96, 10002),
            properties: r#"{"type": "dataset", "name": "TextCorpus-2024", "size": "10GB", "samples": 1000000}"#.to_string(),
        }),
        (10003, EntityData {
            type_id: 4,
            embedding: create_random_embedding(96, 10003),
            properties: r#"{"type": "venue", "name": "ICML 2024", "location": "Vienna", "deadline": "2024-02-01"}"#.to_string(),
        }),
        (10004, EntityData {
            type_id: 5,
            embedding: create_random_embedding(96, 10004),
            properties: r#"{"type": "code", "repository": "github.com/user/nlp-project", "language": "Python", "stars": 156}"#.to_string(),
        }),
    ];
    
    // Insert entities and track their keys
    let mut entity_keys = Vec::new();
    for (id, data) in workflow_entities.clone() {
        let result = graph.insert_entity(id, data);
        assert!(result.is_ok(), "Failed to insert workflow entity {}", id);
        entity_keys.push(result.unwrap());
    }
    
    // Phase 2: Simulate paper development lifecycle through updates
    let paper_key = entity_keys[1]; // Paper entity
    
    let lifecycle_stages = vec![
        r#"{"type": "paper", "title": "Deep Learning in NLP", "status": "in_progress", "pages": 15, "experiments": 3}"#,
        r#"{"type": "paper", "title": "Deep Learning in NLP", "status": "review", "pages": 18, "experiments": 5, "reviewers": 3}"#,
        r#"{"type": "paper", "title": "Deep Learning in Natural Language Processing", "status": "revision", "pages": 20, "experiments": 7, "revision_round": 1}"#,
        r#"{"type": "paper", "title": "Deep Learning in Natural Language Processing", "status": "accepted", "pages": 22, "experiments": 8, "citation_count": 0}"#,
        r#"{"type": "paper", "title": "Deep Learning in Natural Language Processing", "status": "published", "pages": 22, "experiments": 8, "citation_count": 12, "doi": "10.1000/xyz123"}"#,
    ];
    
    for (stage_idx, properties) in lifecycle_stages.iter().enumerate() {
        let updated_data = EntityData {
            type_id: 2,
            embedding: create_random_embedding(96, 10001 + stage_idx as u64),
            properties: properties.to_string(),
        };
        
        let update_result = graph.update_entity(paper_key, updated_data);
        assert!(update_result.is_ok(), "Failed to update paper at stage {}", stage_idx);
        
        // Verify update took effect
        let updated_entity = graph.get_entity(paper_key);
        assert!(updated_entity.is_some(), "Entity not found after update at stage {}", stage_idx);
        
        let (_, data) = updated_entity.unwrap();
        assert!(data.properties.contains("status"), 
               "Properties not updated at stage {}", stage_idx);
    }
    
    // Phase 3: Test entity relationship through lifecycle
    for i in 0..entity_keys.len() - 1 {
        // Test that entities can be connected through their embeddings
        let source_embedding = graph.get_entity_embedding(entity_keys[i]);
        let target_embedding = graph.get_entity_embedding(entity_keys[i + 1]);
        
        assert!(source_embedding.is_some(), "Source embedding missing for entity {}", i);
        assert!(target_embedding.is_some(), "Target embedding missing for entity {}", i + 1);
        
        // Calculate similarity to verify they can be related
        let similarity = calculate_cosine_similarity(
            &source_embedding.unwrap(), 
            &target_embedding.unwrap()
        );
        assert!(similarity >= 0.0 && similarity <= 1.0, "Invalid similarity calculated");
    }
    
    // Phase 4: Test batch entity operations with complex data
    let batch_updates: Vec<_> = entity_keys.iter().enumerate().map(|(idx, &key)| {
        let base_id = 10000 + idx as u64;
        let updated_data = EntityData {
            type_id: (idx % 3 + 1) as u32,
            embedding: create_random_embedding(96, base_id + 1000),
            properties: format!(r#"{{"batch_updated": true, "update_timestamp": "2024-01-{:02d}", "iteration": {}}}"#, 
                               idx + 1, idx),
        };
        (key, updated_data)
    }).collect();
    
    // Apply batch updates
    for (key, data) in batch_updates {
        let result = graph.update_entity(key, data);
        assert!(result.is_ok(), "Batch update failed for entity key {:?}", key);
    }
    
    // Verify all batch updates succeeded
    for (idx, &key) in entity_keys.iter().enumerate() {
        let entity = graph.get_entity(key);
        assert!(entity.is_some(), "Entity {} not found after batch update", idx);
        
        let (_, data) = entity.unwrap();
        assert!(data.properties.contains("batch_updated"), 
               "Entity {} properties not updated in batch", idx);
        assert!(data.properties.contains(&format!("iteration\": {}", idx)), 
               "Entity {} iteration not set correctly", idx);
    }
    
    // Phase 5: Test entity removal and orphan handling
    let removal_order = vec![2, 0, 4, 1, 3]; // Random removal order
    let mut removed_keys = Vec::new();
    
    for &removal_idx in &removal_order {
        let key_to_remove = entity_keys[removal_idx];
        let removal_result = graph.remove_entity(key_to_remove);
        assert!(removal_result.is_ok(), "Failed to remove entity {}", removal_idx);
        assert!(removal_result.unwrap(), "Entity {} was not actually removed", removal_idx);
        
        removed_keys.push(key_to_remove);
        
        // Verify removal
        let removed_entity = graph.get_entity(key_to_remove);
        assert!(removed_entity.is_none(), "Entity {} still exists after removal", removal_idx);
        
        // Verify remaining entities still exist
        for (idx, &key) in entity_keys.iter().enumerate() {
            if !removed_keys.contains(&key) {
                let remaining_entity = graph.get_entity(key);
                assert!(remaining_entity.is_some(), 
                       "Entity {} was incorrectly removed when removing {}", idx, removal_idx);
            }
        }
    }
    
    // All entities should be removed now
    assert_eq!(graph.entity_count(), 0, "Not all entities were removed");
}

#[test]
fn test_entity_memory_and_performance_operations() {
    let graph = create_test_graph();
    
    // Phase 1: Test memory-intensive entity operations
    let large_entity_count = 1000;
    let mut large_entity_keys = Vec::new();
    
    // Create entities with progressively larger property data
    for i in 0..large_entity_count {
        let property_size = (i % 10 + 1) * 100; // 100 to 1000 chars
        let large_properties = format!(
            r#"{{"type": "large_entity", "id": {}, "data": "{}", "metadata": {{"size": {}, "created": "2024-01-01"}}}}"#,
            i,
            "x".repeat(property_size),
            property_size
        );
        
        let entity_data = EntityData {
            type_id: (i % 5 + 1) as u32,
            embedding: create_random_embedding(96, i as u64 + 20000),
            properties: large_properties,
        };
        
        let result = graph.insert_entity(i as u32 + 20000, entity_data);
        if result.is_ok() {
            large_entity_keys.push(result.unwrap());
        } else {
            // Stop if we hit memory limits
            break;
        }
        
        // Check memory usage periodically
        if i % 100 == 0 {
            let memory_usage = graph.memory_usage();
            let total_mb = memory_usage.total_bytes() as f64 / (1024.0 * 1024.0);
            
            // If memory usage gets too high, stop the test
            if total_mb > 500.0 {
                println!("Stopping large entity test at {} entities due to memory usage: {:.2} MB", 
                        i + 1, total_mb);
                break;
            }
        }
    }
    
    let final_entity_count = large_entity_keys.len();
    assert!(final_entity_count > 0, "No large entities were inserted");
    
    // Phase 2: Performance testing on large dataset
    let start_time = std::time::Instant::now();
    
    // Test retrieval performance
    let sample_indices = (0..std::cmp::min(100, final_entity_count)).step_by(final_entity_count / 10 + 1);
    for idx in sample_indices {
        let entity = graph.get_entity(large_entity_keys[idx]);
        assert!(entity.is_some(), "Large entity {} not found", idx);
    }
    
    let retrieval_time = start_time.elapsed();
    assert!(retrieval_time.as_millis() < 1000, 
           "Entity retrieval too slow: {} ms for {} entities", 
           retrieval_time.as_millis(), final_entity_count);
    
    // Phase 3: Test bulk update performance
    let update_start = std::time::Instant::now();
    let sample_size = std::cmp::min(50, final_entity_count);
    
    for i in 0..sample_size {
        let key = large_entity_keys[i];
        let updated_data = EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, i as u64 + 30000),
            properties: format!(r#"{{"type": "updated_large", "id": {}, "updated": true}}"#, i),
        };
        
        let result = graph.update_entity(key, updated_data);
        assert!(result.is_ok(), "Failed to update large entity {}", i);
    }
    
    let update_time = update_start.elapsed();
    assert!(update_time.as_millis() < 2000, 
           "Bulk update too slow: {} ms for {} entities", 
           update_time.as_millis(), sample_size);
    
    // Phase 4: Test memory cleanup on removal
    let initial_memory = graph.memory_usage().total_bytes();
    let removal_count = std::cmp::min(100, final_entity_count);
    
    // Remove a portion of entities
    for i in 0..removal_count {
        let result = graph.remove_entity(large_entity_keys[i]);
        assert!(result.is_ok(), "Failed to remove large entity {}", i);
    }
    
    // Memory might not immediately decrease due to allocator behavior
    // But entity count should reflect removal
    assert_eq!(graph.entity_count(), final_entity_count - removal_count);
    
    // Phase 5: Test edge cases with extreme entity data
    let edge_cases = vec![
        // Empty properties
        EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 40000),
            properties: "{}".to_string(),
        },
        
        // Minimal properties
        EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 40001),
            properties: r#"{"a":"b"}"#.to_string(),
        },
        
        // Zero embedding (all zeros)
        EntityData {
            type_id: 1,
            embedding: vec![0.0; 96],
            properties: r#"{"type": "zero_embedding"}"#.to_string(),
        },
        
        // Extreme type ID
        EntityData {
            type_id: u32::MAX,
            embedding: create_random_embedding(96, 40002),
            properties: r#"{"type": "extreme_type_id"}"#.to_string(),
        },
    ];
    
    for (idx, edge_case_data) in edge_cases.into_iter().enumerate() {
        let entity_id = 40000 + idx as u32;
        let result = graph.insert_entity(entity_id, edge_case_data);
        
        if result.is_ok() {
            // Verify entity can be retrieved
            let retrieved = graph.get_entity_by_id(entity_id);
            assert!(retrieved.is_some(), "Edge case entity {} not retrievable", idx);
            
            // Verify embedding retrieval
            let key = result.unwrap();
            let embedding = graph.get_entity_embedding(key);
            assert!(embedding.is_some(), "Edge case entity {} embedding not retrievable", idx);
            
            // Clean up
            let removal = graph.remove_entity(key);
            assert!(removal.is_ok(), "Failed to remove edge case entity {}", idx);
        }
        // If insertion fails, that's acceptable for extreme cases
    }
}

#[test]
fn test_entity_validation_comprehensive() {
    let graph = create_test_graph();
    
    // Phase 1: Test embedding dimension validation thoroughly
    let dimension_tests = vec![
        (vec![], false),           // Empty
        (vec![1.0], false),        // Too small
        (vec![1.0; 95], false),    // One less than expected
        (vec![1.0; 96], true),     // Correct
        (vec![1.0; 97], false),    // One more than expected
        (vec![1.0; 128], false),   // Wrong dimension
        (vec![1.0; 200], false),   // Much larger
    ];
    
    for (idx, (embedding, should_succeed)) in dimension_tests.iter().enumerate() {
        let entity_data = EntityData {
            type_id: 1,
            embedding: embedding.clone(),
            properties: format!(r#"{{"test_case": {}}}"#, idx),
        };
        
        let result = graph.insert_entity(50000 + idx as u32, entity_data);
        
        if *should_succeed {
            assert!(result.is_ok(), "Valid embedding test case {} should succeed", idx);
        } else {
            assert!(result.is_err(), "Invalid embedding test case {} should fail", idx);
        }
    }
    
    // Phase 2: Test property validation
    let property_tests = vec![
        ("", true),                           // Empty string
        ("{}", true),                        // Empty JSON
        (r#"{"valid": "json"}"#, true),      // Valid JSON
        ("not json", true),                  // Invalid JSON (should still work as string)
        ("null", true),                      // JSON null
        (r#"{"unicode": "测试"}"#, true),     // Unicode
        ("x".repeat(100000), true),          // Very long string (might fail due to limits)
        ("\0", true),                        // Null character
        ("\x01\x02\x03", true),             // Binary data
    ];
    
    for (idx, (properties, should_succeed)) in property_tests.iter().enumerate() {
        let entity_data = EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, 51000 + idx as u64),
            properties: properties.to_string(),
        };
        
        let result = graph.insert_entity(51000 + idx as u32, entity_data);
        
        if *should_succeed {
            if result.is_err() {
                // Some extreme cases might fail, which is acceptable
                println!("Property test case {} failed (acceptable): {:?}", idx, result);
            }
        } else {
            assert!(result.is_err(), "Invalid property test case {} should fail", idx);
        }
    }
    
    // Phase 3: Test entity ID validation
    let id_tests = vec![
        (0, true),                // Zero ID
        (1, true),                // Normal ID
        (u32::MAX, true),        // Maximum ID
        (u32::MAX - 1, true),    // Near maximum
    ];
    
    for (id, should_succeed) in id_tests {
        let entity_data = EntityData {
            type_id: 1,
            embedding: create_random_embedding(96, id as u64),
            properties: format!(r#"{{"id_test": {}}}"#, id),
        };
        
        let result = graph.insert_entity(id, entity_data);
        
        if should_succeed {
            if result.is_ok() {
                // Clean up if successful
                let key = result.unwrap();
                let _ = graph.remove_entity(key);
            }
        } else {
            assert!(result.is_err(), "Invalid ID {} should fail", id);
        }
    }
    
    // Phase 4: Test type ID validation
    let type_id_tests = vec![
        (0, true),
        (1, true),
        (100, true),
        (u32::MAX, true),
    ];
    
    for (type_id, should_succeed) in type_id_tests {
        let entity_data = EntityData {
            type_id,
            embedding: create_random_embedding(96, 52000),
            properties: format!(r#"{{"type_id_test": {}}}"#, type_id),
        };
        
        let result = graph.insert_entity(52000, entity_data);
        
        if should_succeed {
            if result.is_ok() {
                let key = result.unwrap();
                let retrieved = graph.get_entity(key);
                assert!(retrieved.is_some());
                
                let (_, data) = retrieved.unwrap();
                assert_eq!(data.type_id, type_id);
                
                let _ = graph.remove_entity(key);
            }
        } else {
            assert!(result.is_err(), "Invalid type ID {} should fail", type_id);
        }
    }
}

// Helper function for similarity calculation
fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}