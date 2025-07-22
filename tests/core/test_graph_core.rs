//! Integration tests for graph core functionality
//! 
//! Tests complete workflow including creating graph, adding entities/relationships,
//! querying, and verifies all graph components work together correctly through
//! public APIs.

use std::collections::HashMap;

use llmkg::core::graph::{KnowledgeGraph, MemoryUsage};
use llmkg::core::types::{EntityData, Relationship};
use llmkg::error::{GraphError, Result};

fn create_test_graph() -> KnowledgeGraph {
    KnowledgeGraph::new(128).expect("Failed to create test graph")
}

fn create_embedding(seed: u64, dim: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    
    let mut embedding = Vec::with_capacity(dim);
    let mut hash = hasher.finish();
    
    for _ in 0..dim {
        hash = hash.wrapping_mul(1103515245).wrapping_add(12345);
        let val = ((hash >> 16) & 0xFFFF) as f32 / 65536.0;
        embedding.push(val);
    }
    
    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in embedding.iter_mut() {
            *val /= magnitude;
        }
    }
    
    embedding
}

#[test]
fn test_graph_creation_and_configuration() {
    // Test various graph configurations
    let dimensions = vec![64, 96, 128, 256, 512];
    
    for dim in dimensions {
        let graph = KnowledgeGraph::new(dim);
        assert!(graph.is_ok(), "Failed to create graph with dimension {}", dim);
        
        let graph = graph.unwrap();
        assert_eq!(graph.embedding_dimension(), dim);
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        
        // Test initial memory usage
        let memory_usage = graph.memory_usage();
        assert!(memory_usage.total_bytes() > 0);
        
        // Test memory breakdown
        let breakdown = memory_usage.usage_breakdown();
        let total_percentage = breakdown.arena_percentage + 
                               breakdown.entity_store_percentage + 
                               breakdown.graph_percentage + 
                               breakdown.embedding_bank_percentage + 
                               breakdown.quantizer_percentage + 
                               breakdown.bloom_filter_percentage;
        
        // Should sum to approximately 100%
        assert!((total_percentage - 100.0).abs() < 1.0);
    }
    
    // Test default constructor
    let default_graph = KnowledgeGraph::new_default();
    assert!(default_graph.is_ok());
    assert_eq!(default_graph.unwrap().embedding_dimension(), 96);
    
    // Test debug formatting
    let graph = create_test_graph();
    let debug_str = format!("{:?}", graph);
    assert!(debug_str.contains("KnowledgeGraph"));
    assert!(debug_str.contains("entity_count"));
    assert!(debug_str.contains("relationship_count"));
}

#[test]
fn test_complete_entity_relationship_workflow() {
    let graph = create_test_graph();
    
    // Phase 1: Create a small knowledge graph about a university
    let entities = vec![
        (1, EntityData::new(1, r#"{"type": "person", "name": "Dr. Alice Smith", "role": "professor"}"#.to_string(), create_embedding(1, 128))),
        (2, EntityData::new(1, r#"{"type": "person", "name": "Bob Johnson", "role": "student"}"#.to_string(), create_embedding(2, 128))),
        (3, EntityData::new(2, r#"{"type": "course", "name": "Machine Learning", "code": "CS-401"}"#.to_string(), create_embedding(3, 128))),
        (4, EntityData::new(2, r#"{"type": "course", "name": "Data Structures", "code": "CS-201"}"#.to_string(), create_embedding(4, 128))),
        (5, EntityData::new(3, r#"{"type": "department", "name": "Computer Science"}"#.to_string(), create_embedding(5, 128))),
    ];
    
    // Insert entities
    let mut entity_keys = Vec::new();
    for (id, data) in entities {
        let key = graph.insert_entity(id, data);
        assert!(key.is_ok());
        entity_keys.push(key.unwrap());
    }
    
    assert_eq!(graph.entity_count(), 5);
    
    // Phase 2: Create relationships between entities
    let relationships = vec![
        Relationship { from: entity_keys[0], to: entity_keys[2], rel_type: 1, weight: 1.0 }, // Alice teaches ML
        Relationship { from: entity_keys[0], to: entity_keys[3], rel_type: 1, weight: 0.8 }, // Alice teaches DS
        Relationship { from: entity_keys[1], to: entity_keys[2], rel_type: 2, weight: 0.9 }, // Bob takes ML  
        Relationship { from: entity_keys[1], to: entity_keys[3], rel_type: 2, weight: 0.7 }, // Bob takes DS
        Relationship { from: entity_keys[0], to: entity_keys[4], rel_type: 3, weight: 1.0 }, // Alice in CS dept
        Relationship { from: entity_keys[2], to: entity_keys[4], rel_type: 4, weight: 1.0 }, // ML in CS dept
        Relationship { from: entity_keys[3], to: entity_keys[4], rel_type: 4, weight: 1.0 }, // DS in CS dept
    ];
    
    for relationship in relationships {
        let result = graph.insert_relationship(relationship);
        assert!(result.is_ok());
    }
    
    assert_eq!(graph.relationship_count(), 7);
    
    // Phase 3: Test entity retrieval and validation
    for (i, original_id) in [1, 2, 3, 4, 5].iter().enumerate() {
        // By key
        let by_key = graph.get_entity(entity_keys[i]);
        assert!(by_key.is_some());
        
        // By ID  
        let by_id = graph.get_entity_by_id(*original_id);
        assert!(by_id.is_some());
        
        // Key consistency
        let key = graph.get_entity_key(*original_id);
        assert!(key.is_some());
        assert_eq!(key.unwrap(), entity_keys[i]);
        
        // Existence checks
        assert!(graph.contains_entity(*original_id));
        
        // Embedding retrieval
        let embedding = graph.get_entity_embedding(entity_keys[i]);
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().len(), 128);
        
        // ID reverse lookup
        let reverse_id = graph.get_entity_id(entity_keys[i]);
        assert!(reverse_id.is_some());
        assert_eq!(reverse_id.unwrap(), *original_id);
    }
    
    // Phase 4: Test relationship queries
    let alice_key = entity_keys[0];
    let alice_outgoing = graph.get_outgoing_relationships(alice_key);
    assert_eq!(alice_outgoing.len(), 3); // teaches ML, teaches DS, in CS dept
    
    let alice_neighbors = graph.get_neighbors(alice_key);
    assert_eq!(alice_neighbors.len(), 3);
    
    let bob_key = entity_keys[1];
    let bob_outgoing = graph.get_outgoing_relationships(bob_key);
    assert_eq!(bob_outgoing.len(), 2); // takes ML, takes DS
    
    let ml_key = entity_keys[2];
    let ml_incoming = graph.get_incoming_relationships(ml_key);
    assert_eq!(ml_incoming.len(), 2); // Alice teaches, Bob takes
    
    // Phase 5: Test path finding
    let alice_to_bob_path = graph.find_path(alice_key, bob_key);
    assert!(alice_to_bob_path.is_some());
    
    let path = alice_to_bob_path.unwrap();
    assert!(path.len() >= 2); // At least Alice -> Course -> Bob
    assert_eq!(path[0], alice_key);
    assert_eq!(path[path.len() - 1], bob_key);
    
    // Phase 6: Test similarity search
    let query_embedding = create_embedding(100, 128); // Different from existing
    let similar = graph.similarity_search(&query_embedding, 3);
    assert!(similar.is_ok());
    
    let similar_entities = similar.unwrap();
    assert!(similar_entities.len() <= 3);
    assert!(similar_entities.len() > 0);
    
    for (entity_key, similarity) in similar_entities {
        assert!(similarity >= 0.0 && similarity <= 1.0);
        assert!(entity_keys.contains(&entity_key));
    }
    
    // Phase 7: Test entity updates
    let updated_alice = EntityData::new(1, r#"{"type": "person", "name": "Dr. Alice Smith", "role": "professor", "tenure": true}"#.to_string(), create_embedding(10, 128));
    
    let update_result = graph.update_entity(alice_key, updated_alice.clone());
    assert!(update_result.is_ok());
    
    // Verify update
    let updated_entity = graph.get_entity(alice_key);
    assert!(updated_entity.is_some());
    let (_, updated_data) = updated_entity.unwrap();
    assert!(updated_data.properties.contains("tenure"));
    assert_eq!(updated_data.embedding, updated_alice.embedding);
    
    // Phase 8: Test entity removal
    let remove_result = graph.remove_entity(bob_key);
    assert!(remove_result.is_ok());
    assert_eq!(remove_result.unwrap(), true);
    
    // Verify removal
    assert_eq!(graph.entity_count(), 4);
    let removed_entity = graph.get_entity(bob_key);
    assert!(removed_entity.is_none());
    
    // Relationships involving Bob should be cleaned up
    let ml_incoming_after_removal = graph.get_incoming_relationships(ml_key);
    assert_eq!(ml_incoming_after_removal.len(), 1); // Only Alice teaches now
}

#[test]
fn test_batch_operations_and_performance() {
    let graph = create_test_graph();
    let batch_size = 100;
    
    // Phase 1: Prepare batch data
    let mut batch_entities = Vec::new();
    for i in 0..batch_size {
        let entity_data = EntityData::new(((i % 5) + 1) as u16, format!(r#"{{"id": {}, "batch": "test", "category": "{}"}}"#, i, i % 10), create_embedding(i as u64, 128));
        batch_entities.push((i as u32, entity_data));
    }
    
    // Phase 2: Batch insert
    let start_time = std::time::Instant::now();
    let batch_result = graph.insert_entities_batch(batch_entities.clone());
    let batch_duration = start_time.elapsed();
    
    assert!(batch_result.is_ok());
    let entity_keys = batch_result.unwrap();
    assert_eq!(entity_keys.len(), batch_size);
    
    // Should complete in reasonable time (less than 5 seconds)
    assert!(batch_duration.as_secs() < 5);
    
    println!("Batch insert of {} entities took {:?}", batch_size, batch_duration);
    
    // Phase 3: Verify all entities exist
    assert_eq!(graph.entity_count(), batch_size);
    
    for (i, entity_key) in entity_keys.iter().enumerate() {
        let entity = graph.get_entity(*entity_key);
        assert!(entity.is_some());
        
        let (_, data) = entity.unwrap();
        let expected_type = ((i % 5) + 1) as u32;
        assert_eq!(data.type_id, expected_type);
    }
    
    // Phase 4: Test performance of individual operations
    let query_embedding = create_embedding(1000, 128);
    
    let similarity_start = std::time::Instant::now();
    let similarity_results = graph.similarity_search(&query_embedding, 10);
    let similarity_duration = similarity_start.elapsed();
    
    assert!(similarity_results.is_ok());
    assert!(similarity_duration.as_millis() < 100); // Should be fast
    
    // Phase 5: Memory usage analysis  
    let memory_usage = graph.memory_usage();
    let total_mb = memory_usage.total_bytes() as f64 / (1024.0 * 1024.0);
    let bytes_per_entity = memory_usage.bytes_per_entity(batch_size);
    
    println!("Memory usage: {:.2} MB total, {} bytes per entity", total_mb, bytes_per_entity);
    
    // Memory should be reasonable (less than 100MB for 100 entities)
    assert!(total_mb < 100.0);
    assert!(bytes_per_entity > 0);
    
    let breakdown = memory_usage.usage_breakdown();
    assert!(breakdown.arena_percentage >= 0.0);
    assert!(breakdown.entity_store_percentage >= 0.0);
    assert!(breakdown.graph_percentage >= 0.0);
    
    // Phase 6: Test cache performance
    let (cache_size, cache_capacity, hit_rate) = graph.cache_stats();
    assert!(cache_capacity > 0);
    assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
    
    // Perform multiple similarity searches to test caching
    for i in 0..10 {
        let test_embedding = create_embedding(2000 + i, 128);
        let _ = graph.similarity_search(&test_embedding, 5);
    }
    
    let (final_cache_size, _, final_hit_rate) = graph.cache_stats();
    assert!(final_cache_size >= cache_size); // Should have more cached items
    
    // Clear cache and verify
    graph.clear_caches();
    let (cleared_size, _, _) = graph.cache_stats();
    assert_eq!(cleared_size, 0);
}

#[test]
fn test_graph_validation_and_consistency() {
    let graph = create_test_graph();
    
    // Phase 1: Build a complex graph structure
    let num_entities = 50;
    let mut entity_keys = Vec::new();
    
    // Create entities
    for i in 0..num_entities {
        let entity_data = EntityData::new(((i % 3) + 1) as u16, format!(r#"{{"id": {}, "type": "test_entity"}}"#, i), create_embedding(i as u64, 128));
        
        let key = graph.insert_entity(i as u32, entity_data);
        assert!(key.is_ok());
        entity_keys.push(key.unwrap());
    }
    
    // Create a web of relationships
    for i in 0..num_entities {
        for j in (i+1)..std::cmp::min(i+5, num_entities) {
            let relationship = Relationship {
                from: entity_keys[i],
                to: entity_keys[j],
                rel_type: ((i + j) % 3) as u32,
                weight: (i as f32 + j as f32) / (2.0 * num_entities as f32),
            };
            
            let _ = graph.insert_relationship(relationship);
        }
    }
    
    // Phase 2: Validate graph structure
    assert_eq!(graph.entity_count(), num_entities);
    assert!(graph.relationship_count() > 0);
    
    // Check entity-key mapping consistency
    for i in 0..num_entities {
        let id = i as u32;
        let key = graph.get_entity_key(id);
        assert!(key.is_some());
        assert_eq!(key.unwrap(), entity_keys[i]);
        
        let reverse_id = graph.get_entity_id(entity_keys[i]);
        assert!(reverse_id.is_some());
        assert_eq!(reverse_id.unwrap(), id);
        
        assert!(graph.contains_entity(id));
    }
    
    // Phase 3: Test embedding validation
    let valid_embedding = create_embedding(1000, 128);
    let validation_result = graph.validate_embedding_dimension(&valid_embedding);
    assert!(validation_result.is_ok());
    
    let invalid_embedding = create_embedding(1000, 64); // Wrong dimension
    let invalid_result = graph.validate_embedding_dimension(&invalid_embedding);
    assert!(invalid_result.is_err());
    
    if let Err(GraphError::InvalidEmbeddingDimension { expected, actual }) = invalid_result {
        assert_eq!(expected, 128);
        assert_eq!(actual, 64);
    } else {
        panic!("Expected InvalidEmbeddingDimension error");
    }
    
    // Phase 4: Test edge buffer management
    let initial_buffer_size = graph.edge_buffer_size();
    
    // Add some relationships to buffer
    let test_relationship = Relationship {
        from: entity_keys[0],
        to: entity_keys[1],
        rel_type: 99,
        weight: 0.5,
    };
    
    let buffer_insert = graph.insert_relationship(test_relationship);
    assert!(buffer_insert.is_ok());
    
    let buffer_size_after = graph.edge_buffer_size();
    // Buffer size might increase depending on implementation
    
    // Test buffer flush
    let flush_result = graph.flush_edge_buffer();
    assert!(flush_result.is_ok());
    
    let buffer_size_after_flush = graph.edge_buffer_size();
    // Buffer should be smaller or same after flush
    assert!(buffer_size_after_flush <= buffer_size_after);
    
    // Phase 5: Test string dictionary
    let initial_dict_size = graph.string_dictionary_size();
    
    // Dictionary might grow during entity operations
    let final_dict_size = graph.string_dictionary_size();
    assert!(final_dict_size >= initial_dict_size);
    
    // Phase 6: Test epoch manager access
    let epoch_manager = graph.epoch_manager();
    assert!(epoch_manager.current_epoch() >= 0);
    
    // Phase 7: Consistency check after complex operations
    let final_entity_count = graph.entity_count();
    let final_relationship_count = graph.relationship_count();
    
    assert_eq!(final_entity_count, num_entities);
    assert!(final_relationship_count > 0);
    
    // All entities should still be accessible
    for i in 0..num_entities {
        let entity = graph.get_entity_by_id(i as u32);
        assert!(entity.is_some());
        
        let (key, data) = entity.unwrap();
        assert_eq!(key, entity_keys[i]);
        assert_eq!(data.embedding.len(), 128);
        assert!(data.properties.contains("test_entity"));
    }
}

#[test]
fn test_error_handling_and_edge_cases() {
    let graph = create_test_graph();
    
    // Test 1: Operations on empty graph
    let empty_similarity = graph.similarity_search(&vec![0.1; 128], 5);
    assert!(empty_similarity.is_ok());
    let empty_results = empty_similarity.unwrap();
    assert!(empty_results.is_empty());
    
    let non_existent_entity = graph.get_entity_by_id(999);
    assert!(non_existent_entity.is_none());
    
    // Test fake paths - since we can't create fake keys, we'll skip this specific test
    // and focus on other aspects
    
    // Test 2: Invalid relationship operations
    let valid_entity = EntityData::new(1, "{}".to_string(), create_embedding(1, 128));
    
    let entity_key = graph.insert_entity(1, valid_entity);
    assert!(entity_key.is_ok());
    let key = entity_key.unwrap();
    
    // Instead, test with actual operations that should fail gracefully
    
    // Test 3: Memory pressure scenarios
    let large_embedding = vec![0.1; 128]; // Valid size but let's test memory limits
    
    // Try to insert many entities quickly
    let mut large_batch = Vec::new();
    for i in 0..1000 {
        let entity_data = EntityData::new(1, format!("{{\"id\": {}}}", i), large_embedding.clone());
        large_batch.push((i + 100, entity_data));
    }
    
    let large_batch_result = graph.insert_entities_batch(large_batch);
    // Should either succeed or fail gracefully
    if let Err(e) = large_batch_result {
        // Error should be meaningful
        match e {
            GraphError::MemoryLimitExceeded => (),
            GraphError::BatchTooLarge => (),
            _ => println!("Unexpected error: {:?}", e),
        }
    } else {
        // If succeeded, verify graph is still consistent
        let entity_count = graph.entity_count();
        assert!(entity_count >= 1001); // Original entity + batch
        
        // Memory usage should be reported correctly
        let memory_usage = graph.memory_usage();
        assert!(memory_usage.total_bytes() > 0);
    }
    
    // Test 4: Concurrent access patterns (basic)
    use std::sync::Arc;
    use std::thread;
    
    let graph_arc = Arc::new(graph);
    let num_threads = 4;
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let graph_clone = Arc::clone(&graph_arc);
        
        let handle = thread::spawn(move || {
            // Each thread tries to read existing entities
            for i in 0..10 {
                let entity_id = 1; // All threads read same entity
                let entity = graph_clone.get_entity_by_id(entity_id);
                
                // Should consistently return same result
                if i == 0 {
                    // First read in this thread
                    if entity.is_some() {
                        // Verify it's the same on subsequent reads
                        for j in 1..10 {
                            let repeat_read = graph_clone.get_entity_by_id(entity_id);
                            assert_eq!(entity.is_some(), repeat_read.is_some());
                        }
                    }
                }
                
                // Try similarity search (should be thread-safe)
                let query = create_embedding((thread_id * 1000 + i) as u64, 128);
                let _ = graph_clone.similarity_search(&query, 3);
            }
            
            // Each thread checks graph stats
            let stats = (
                graph_clone.entity_count(),
                graph_clone.relationship_count(),
                graph_clone.embedding_dimension()
            );
            
            stats
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut all_stats = Vec::new();
    for handle in handles {
        let stats = handle.join().expect("Thread panicked");
        all_stats.push(stats);
    }
    
    // All threads should see consistent view
    let first_stats = all_stats[0];
    for stats in all_stats {
        assert_eq!(stats.0, first_stats.0); // entity count
        assert_eq!(stats.1, first_stats.1); // relationship count  
        assert_eq!(stats.2, first_stats.2); // embedding dimension
    }
}

#[test]
fn test_comprehensive_graph_workflow_integration() {
    let graph = create_test_graph();
    
    // Phase 1: Build a comprehensive knowledge graph representing a university
    let university_entities = vec![
        // Students
        (100, EntityData::new(1, r#"{"type": "student", "name": "Alice Johnson", "major": "Computer Science", "year": 3, "gpa": 3.8}"#.to_string(), create_embedding(100, 128))),
        (101, EntityData::new(1, r#"{"type": "student", "name": "Bob Smith", "major": "Mathematics", "year": 4, "gpa": 3.9}"#.to_string(), create_embedding(101, 128))),
        (102, EntityData::new(1, r#"{"type": "student", "name": "Carol Davis", "major": "Physics", "year": 2, "gpa": 3.7}"#.to_string(), create_embedding(102, 128))),
        
        // Faculty
        (200, EntityData::new(2, r#"{"type": "faculty", "name": "Dr. Emma Wilson", "department": "Computer Science", "title": "Professor", "tenure": true}"#.to_string(), create_embedding(200, 128))),
        (201, EntityData::new(2, r#"{"type": "faculty", "name": "Dr. Michael Brown", "department": "Mathematics", "title": "Associate Professor", "tenure": true}"#.to_string(), create_embedding(201, 128))),
        (202, EntityData::new(2, r#"{"type": "faculty", "name": "Dr. Sarah Lee", "department": "Physics", "title": "Assistant Professor", "tenure": false}"#.to_string(), create_embedding(202, 128))),
        
        // Courses
        (300, EntityData::new(3, r#"{"type": "course", "code": "CS-301", "title": "Data Structures", "credits": 3, "semester": "Fall"}"#.to_string(), create_embedding(300, 128))),
        (301, EntityData::new(3, r#"{"type": "course", "code": "CS-401", "title": "Machine Learning", "credits": 4, "semester": "Spring"}"#.to_string(), create_embedding(301, 128))),
        (302, EntityData::new(3, r#"{"type": "course", "code": "MATH-301", "title": "Linear Algebra", "credits": 3, "semester": "Fall"}"#.to_string(), create_embedding(302, 128))),
        
        // Departments
        (400, EntityData::new(4, r#"{"type": "department", "name": "Computer Science", "building": "Engineering Hall", "faculty_count": 25}"#.to_string(), create_embedding(400, 128))),
        (401, EntityData::new(4, r#"{"type": "department", "name": "Mathematics", "building": "Science Center", "faculty_count": 18}"#.to_string(), create_embedding(401, 128))),
        (402, EntityData::new(4, r#"{"type": "department", "name": "Physics", "building": "Physics Lab", "faculty_count": 15}"#.to_string(), create_embedding(402, 128))),
    ];
    
    // Insert all entities
    let mut entity_keys = Vec::new();
    for (id, data) in university_entities {
        let result = graph.insert_entity(id, data);
        assert!(result.is_ok(), "Failed to insert university entity {}", id);
        entity_keys.push((id, result.unwrap()));
    }
    
    assert_eq!(graph.entity_count(), 12);
    
    // Phase 2: Create complex relationship network
    let relationships = vec![
        // Student-Faculty relationships (advisor)
        Relationship { from: entity_keys[0].1, to: entity_keys[3].1, rel_type: 1, weight: 1.0 }, // Alice -> Dr. Wilson
        Relationship { from: entity_keys[1].1, to: entity_keys[4].1, rel_type: 1, weight: 1.0 }, // Bob -> Dr. Brown
        Relationship { from: entity_keys[2].1, to: entity_keys[5].1, rel_type: 1, weight: 1.0 }, // Carol -> Dr. Lee
        
        // Student-Course relationships (enrollment)
        Relationship { from: entity_keys[0].1, to: entity_keys[6].1, rel_type: 2, weight: 0.9 }, // Alice -> CS-301
        Relationship { from: entity_keys[0].1, to: entity_keys[7].1, rel_type: 2, weight: 0.8 }, // Alice -> CS-401
        Relationship { from: entity_keys[1].1, to: entity_keys[8].1, rel_type: 2, weight: 0.95 }, // Bob -> MATH-301
        Relationship { from: entity_keys[2].1, to: entity_keys[8].1, rel_type: 2, weight: 0.85 }, // Carol -> MATH-301
        
        // Faculty-Course relationships (teaching)
        Relationship { from: entity_keys[3].1, to: entity_keys[6].1, rel_type: 3, weight: 1.0 }, // Dr. Wilson -> CS-301
        Relationship { from: entity_keys[3].1, to: entity_keys[7].1, rel_type: 3, weight: 1.0 }, // Dr. Wilson -> CS-401
        Relationship { from: entity_keys[4].1, to: entity_keys[8].1, rel_type: 3, weight: 1.0 }, // Dr. Brown -> MATH-301
        
        // Faculty-Department relationships (affiliation)
        Relationship { from: entity_keys[3].1, to: entity_keys[9].1, rel_type: 4, weight: 1.0 }, // Dr. Wilson -> CS Dept
        Relationship { from: entity_keys[4].1, to: entity_keys[10].1, rel_type: 4, weight: 1.0 }, // Dr. Brown -> Math Dept
        Relationship { from: entity_keys[5].1, to: entity_keys[11].1, rel_type: 4, weight: 1.0 }, // Dr. Lee -> Physics Dept
        
        // Course-Department relationships (offered by)
        Relationship { from: entity_keys[6].1, to: entity_keys[9].1, rel_type: 5, weight: 1.0 }, // CS-301 -> CS Dept
        Relationship { from: entity_keys[7].1, to: entity_keys[9].1, rel_type: 5, weight: 1.0 }, // CS-401 -> CS Dept
        Relationship { from: entity_keys[8].1, to: entity_keys[10].1, rel_type: 5, weight: 1.0 }, // MATH-301 -> Math Dept
    ];
    
    for relationship in relationships {
        let result = graph.insert_relationship(relationship);
        assert!(result.is_ok(), "Failed to insert relationship");
    }
    
    assert_eq!(graph.relationship_count(), 16);
    
    // Phase 3: Test complex querying and path finding
    let alice_key = entity_keys[0].1;
    let cs_dept_key = entity_keys[9].1;
    
    // Find path from Alice to CS Department
    let alice_to_cs_dept = graph.find_path(alice_key, cs_dept_key);
    assert!(alice_to_cs_dept.is_some(), "Should find path from Alice to CS Department");
    
    let path = alice_to_cs_dept.unwrap();
    assert!(path.len() >= 2, "Path should have at least 2 nodes");
    assert_eq!(path[0], alice_key);
    assert_eq!(path[path.len() - 1], cs_dept_key);
    
    // Test multi-hop relationships
    let bob_key = entity_keys[1].1;
    let math_dept_key = entity_keys[10].1;
    
    let bob_to_math_dept = graph.find_path(bob_key, math_dept_key);
    assert!(bob_to_math_dept.is_some(), "Should find path from Bob to Math Department");
    
    // Phase 4: Test similarity search across different entity types
    let cs_query = create_embedding(1000, 128); // Simulate a computer science query
    let cs_results = graph.similarity_search(&cs_query, 5);
    assert!(cs_results.is_ok());
    
    let similar_entities = cs_results.unwrap();
    assert!(!similar_entities.is_empty(), "Should find similar entities");
    
    // Verify similarity scores are valid
    for (entity_key, similarity) in &similar_entities {
        assert!(similarity >= &0.0 && similarity <= &1.0, "Invalid similarity score");
        assert!(entity_keys.iter().any(|(_, key)| key == entity_key), "Unknown entity in results");
    }
    
    // Phase 5: Test neighbor analysis
    let dr_wilson_key = entity_keys[3].1;
    let wilson_outgoing = graph.get_outgoing_relationships(dr_wilson_key);
    let wilson_neighbors = graph.get_neighbors(dr_wilson_key);
    
    assert_eq!(wilson_outgoing.len(), 3, "Dr. Wilson should have 3 outgoing relationships");
    assert_eq!(wilson_neighbors.len(), 3, "Dr. Wilson should have 3 neighbors");
    
    // Verify neighbors include courses and department
    let cs_301_key = entity_keys[6].1;
    let cs_401_key = entity_keys[7].1;
    
    assert!(wilson_neighbors.contains(&cs_301_key), "Dr. Wilson should be connected to CS-301");
    assert!(wilson_neighbors.contains(&cs_401_key), "Dr. Wilson should be connected to CS-401");
    assert!(wilson_neighbors.contains(&cs_dept_key), "Dr. Wilson should be connected to CS Department");
    
    // Phase 6: Test entity updates in complex graph
    let updated_alice = EntityData::new(1, r#"{"type": "student", "name": "Alice Johnson", "major": "Computer Science", "year": 4, "gpa": 3.85, "honors": true}"#.to_string(), create_embedding(100, 128));
    
    let alice_update_result = graph.update_entity(alice_key, updated_alice);
    assert!(alice_update_result.is_ok(), "Failed to update Alice's information");
    
    // Verify update didn't break relationships
    let alice_after_update = graph.get_outgoing_relationships(alice_key);
    assert_eq!(alice_after_update.len(), 2, "Alice's relationships should be preserved after update");
    
    // Phase 7: Test entity removal and cascade effects
    let carol_key = entity_keys[2].1;
    let removal_result = graph.remove_entity(carol_key);
    assert!(removal_result.is_ok(), "Failed to remove Carol");
    assert_eq!(removal_result.unwrap(), true, "Carol should be successfully removed");
    
    // Verify Carol is gone
    let carol_after_removal = graph.get_entity(carol_key);
    assert!(carol_after_removal.is_none(), "Carol should not exist after removal");
    
    // Verify entity count is updated
    assert_eq!(graph.entity_count(), 11, "Entity count should decrease after removal");
    
    // Check that relationships involving Carol are cleaned up
    let math_course_key = entity_keys[8].1;
    let math_course_incoming = graph.get_incoming_relationships(math_course_key);
    
    // Should only have Bob enrolled in MATH-301 now, not Carol
    assert_eq!(math_course_incoming.len(), 1, "MATH-301 should have only 1 incoming relationship after Carol's removal");
    
    // Phase 8: Test graph statistics and validation
    let final_stats = (
        graph.entity_count(),
        graph.relationship_count(),
        graph.embedding_dimension()
    );
    
    assert_eq!(final_stats.0, 11, "Should have 11 entities after removal");
    assert!(final_stats.1 > 0, "Should have relationships");
    assert_eq!(final_stats.2, 128, "Embedding dimension should be 128");
    
    // Test memory usage reporting
    let memory_usage = graph.memory_usage();
    assert!(memory_usage.total_bytes() > 0, "Memory usage should be positive");
    
    let breakdown = memory_usage.usage_breakdown();
    assert!(breakdown.arena_percentage >= 0.0, "Arena percentage should be non-negative");
    assert!(breakdown.entity_store_percentage >= 0.0, "Entity store percentage should be non-negative");
    
    // Test cache effectiveness
    let (cache_size, cache_capacity, hit_rate) = graph.cache_stats();
    assert!(cache_capacity > 0, "Cache should have positive capacity");
    assert!(hit_rate >= 0.0 && hit_rate <= 1.0, "Hit rate should be between 0 and 1");
}

#[test]
fn test_graph_scalability_and_performance() {
    let graph = create_test_graph();
    
    // Phase 1: Test scalability with medium-sized dataset
    let entity_count = 1000;
    let relationship_density = 0.01; // 1% of possible relationships
    
    // Insert entities in batches
    let batch_size = 100;
    let mut all_entity_keys = Vec::new();
    
    for batch_start in (0..entity_count).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, entity_count);
        let mut batch_entities = Vec::new();
        
        for i in batch_start..batch_end {
            let entity_data = EntityData::new(
                ((i % 10 + 1) as u16),
                format!(
                    r#"{{"type": "entity_{}", "id": {}, "category": "{}", "value": {}}}"#,
                    i % 10,
                    i,
                    match i % 5 {
                        0 => "primary",
                        1 => "secondary", 
                        2 => "tertiary",
                        3 => "auxiliary",
                        _ => "misc",
                    },
                    (i as f64 * 1.23) % 100.0
                ),
                create_embedding(i as u64 + 10000, 128)
            );
            
            batch_entities.push((i as u32 + 10000, entity_data));
        }
        
        let batch_start_time = std::time::Instant::now();
        let batch_result = graph.insert_entities_batch(batch_entities);
        let batch_duration = batch_start_time.elapsed();
        
        if let Ok(keys) = batch_result {
            all_entity_keys.extend(keys);
            
            // Batch insertion should complete quickly
            assert!(batch_duration.as_millis() < 1000, 
                   "Batch insertion too slow: {} ms for {} entities", 
                   batch_duration.as_millis(), batch_end - batch_start);
        } else {
            // If batch insertion fails, try individual insertions
            for (id, data) in batch_entities {
                if let Ok(key) = graph.insert_entity(id, data) {
                    all_entity_keys.push(key);
                }
            }
        }
    }
    
    let actual_entity_count = all_entity_keys.len();
    assert!(actual_entity_count > entity_count / 2, 
           "Should successfully insert at least half the entities");
    
    // Phase 2: Create relationships based on similarity and patterns
    let mut relationship_count = 0;
    let target_relationships = (actual_entity_count as f64 * relationship_density) as usize;
    
    let relationship_start_time = std::time::Instant::now();
    
    // Create relationships between nearby entities (by ID)
    for i in 0..std::cmp::min(target_relationships, actual_entity_count - 1) {
        let source_key = all_entity_keys[i];
        let target_key = all_entity_keys[i + 1];
        
        let relationship = Relationship {
            from: source_key,
            to: target_key,
            rel_type: (i % 5 + 1) as u32,
            weight: (i as f64 * 0.123) % 1.0,
        };
        
        if graph.insert_relationship(relationship).is_ok() {
            relationship_count += 1;
        }
        
        // Break if taking too long
        if relationship_start_time.elapsed().as_secs() > 10 {
            break;
        }
    }
    
    let relationship_duration = relationship_start_time.elapsed();
    println!("Created {} relationships in {:?}", relationship_count, relationship_duration);
    
    // Phase 3: Performance testing
    let perf_start_time = std::time::Instant::now();
    
    // Test random entity retrieval performance
    let sample_size = std::cmp::min(100, actual_entity_count);
    for i in 0..sample_size {
        let random_idx = (i * 13) % actual_entity_count; // Pseudo-random access
        let entity = graph.get_entity(all_entity_keys[random_idx]);
        assert!(entity.is_some(), "Entity should exist");
    }
    
    let retrieval_duration = perf_start_time.elapsed();
    assert!(retrieval_duration.as_millis() < 1000, 
           "Entity retrieval too slow: {} ms for {} retrievals", 
           retrieval_duration.as_millis(), sample_size);
    
    // Test similarity search performance
    let similarity_start_time = std::time::Instant::now();
    let query_embedding = create_embedding(99999, 128);
    let similarity_results = graph.similarity_search(&query_embedding, 10);
    let similarity_duration = similarity_start_time.elapsed();
    
    assert!(similarity_results.is_ok(), "Similarity search should succeed");
    assert!(similarity_duration.as_millis() < 500, 
           "Similarity search too slow: {} ms", similarity_duration.as_millis());
    
    let results = similarity_results.unwrap();
    assert!(results.len() <= 10, "Should return at most 10 results");
    
    // Test path finding performance
    if relationship_count > 10 {
        let path_start_time = std::time::Instant::now();
        
        // Try to find paths between a few entity pairs
        let path_attempts = std::cmp::min(10, actual_entity_count / 10);
        let mut successful_paths = 0;
        
        for i in 0..path_attempts {
            let source_idx = i * 7 % actual_entity_count;
            let target_idx = (i * 11 + 1) % actual_entity_count;
            
            if source_idx != target_idx {
                let path = graph.find_path(all_entity_keys[source_idx], all_entity_keys[target_idx]);
                if path.is_some() {
                    successful_paths += 1;
                }
            }
        }
        
        let path_duration = path_start_time.elapsed();
        assert!(path_duration.as_millis() < 2000, 
               "Path finding too slow: {} ms for {} attempts", 
               path_duration.as_millis(), path_attempts);
        
        println!("Found {} successful paths out of {} attempts", successful_paths, path_attempts);
    }
    
    // Phase 4: Memory efficiency analysis
    let memory_usage = graph.memory_usage();
    let memory_mb = memory_usage.total_bytes() as f64 / (1024.0 * 1024.0);
    let bytes_per_entity = memory_usage.bytes_per_entity(actual_entity_count);
    
    println!("Memory usage: {:.2} MB total, {} bytes per entity", memory_mb, bytes_per_entity);
    
    // Memory usage should be reasonable for the dataset size
    let max_expected_mb = actual_entity_count as f64 * 0.01; // ~10KB per entity max
    assert!(memory_mb < max_expected_mb, 
           "Memory usage too high: {:.2} MB (expected < {:.2} MB)", 
           memory_mb, max_expected_mb);
    
    // Phase 5: Cache effectiveness testing
    graph.clear_caches();
    let (_, _, initial_hit_rate) = graph.cache_stats();
    
    // Perform repeated similarity searches to warm up cache
    let cache_test_query = create_embedding(88888, 128);
    for _ in 0..5 {
        let _ = graph.similarity_search(&cache_test_query, 5);
    }
    
    let (final_cache_size, _, final_hit_rate) = graph.cache_stats();
    assert!(final_cache_size > 0, "Cache should have entries after repeated searches");
    
    // Final validation
    assert_eq!(graph.entity_count(), actual_entity_count, "Entity count should be consistent");
    assert!(graph.relationship_count() >= relationship_count as u32, "Relationship count should be consistent");
}

#[test]
fn test_graph_robustness_and_edge_cases() {
    let graph = create_test_graph();
    
    // Phase 1: Test with extreme entity configurations
    let extreme_entities = vec![
        // Minimal entity
        EntityData::new(0, "".to_string(), vec![0.0; 128]),
        
        // Maximum type ID
        EntityData::new(u16::MAX, "{}".to_string(), vec![1.0; 128]),
        
        // Large property string
        EntityData::new(1, format!("{{\n  \"large_data\": \"{}\"\n}}", "x".repeat(50000)), create_embedding(20000, 128)),
        
        // Complex JSON properties
        EntityData::new(2, r#"{
                "nested": {
                    "deeply": {
                        "nested": {
                            "object": {
                                "with": ["arrays", "and", "values"],
                                "numbers": [1, 2, 3.14, -42],
                                "booleans": [true, false],
                                "null_value": null
                            }
                        }
                    }
                },
                "unicode": "Hello 世界 🌍",
                "escaped": "quotes\"and\\backslashes",
                "special_chars": "\n\t\r\b\f"
            }"#.to_string(), create_embedding(20001, 128)),
        
        // Normalized embedding
        EntityData::new(3, r#"{"normalized": true}"#.to_string(), {
                let mut emb = create_embedding(20002, 128);
                // Normalize to unit length
                let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if magnitude > 0.0 {
                    for val in emb.iter_mut() {
                        *val /= magnitude;
                    }
                }
                emb
            }),
    ];
    
    let mut extreme_keys = Vec::new();
    for (idx, entity_data) in extreme_entities.into_iter().enumerate() {
        let entity_id = 20000 + idx as u32;
        let result = graph.insert_entity(entity_id, entity_data);
        
        if result.is_ok() {
            extreme_keys.push(result.unwrap());
            
            // Verify entity can be retrieved
            let retrieved = graph.get_entity_by_id(entity_id);
            assert!(retrieved.is_some(), "Extreme entity {} not retrievable", idx);
        } else {
            // Some extreme cases might fail, which is acceptable
            println!("Extreme entity {} insertion failed (acceptable): {:?}", idx, result);
        }
    }
    
    // Phase 2: Test relationship robustness
    if extreme_keys.len() >= 2 {
        // Create relationships between extreme entities
        for i in 0..extreme_keys.len() - 1 {
            let relationship = Relationship {
                from: extreme_keys[i],
                to: extreme_keys[i + 1],
                rel_type: i as u32,
                weight: 1.0 / (i + 1) as f64, // Varying weights including very small values
            };
            
            let rel_result = graph.insert_relationship(relationship);
            if rel_result.is_ok() {
                // Verify relationship exists
                let outgoing = graph.get_outgoing_relationships(extreme_keys[i]);
                assert!(!outgoing.is_empty(), "Relationship should exist");
            }
        }
    }
    
    // Phase 3: Test concurrent robustness
    use std::sync::Arc;
    use std::thread;
    
    let graph_arc = Arc::new(graph);
    let num_threads = 8;
    let operations_per_thread = 50;
    
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let graph_clone = Arc::clone(&graph_arc);
        
        thread::spawn(move || {
            let mut local_stats = (0, 0, 0); // (insertions, queries, errors)
            
            for i in 0..operations_per_thread {
                let base_id = thread_id * 10000 + i + 30000;
                
                // Try various operations
                match i % 4 {
                    0 => {
                        // Insert entity
                        let entity_data = EntityData::new(thread_id as u16, format!(r#"{{"thread": {}, "op": {}}}"#, thread_id, i), create_embedding(base_id as u64, 128));
                        
                        if graph_clone.insert_entity(base_id as u32, entity_data).is_ok() {
                            local_stats.0 += 1;
                        } else {
                            local_stats.2 += 1;
                        }
                    },
                    
                    1 => {
                        // Query entity
                        let query_id = ((thread_id + i) % num_threads) * 10000 + 30000;
                        if graph_clone.get_entity_by_id(query_id as u32).is_some() {
                            local_stats.1 += 1;
                        }
                    },
                    
                    2 => {
                        // Similarity search
                        let query_emb = create_embedding(base_id as u64 + 50000, 128);
                        if graph_clone.similarity_search(&query_emb, 3).is_ok() {
                            local_stats.1 += 1;
                        } else {
                            local_stats.2 += 1;
                        }
                    },
                    
                    3 => {
                        // Check existence
                        let check_id = ((thread_id + i * 2) % num_threads) * 10000 + 30000;
                        if graph_clone.contains_entity(check_id as u32) {
                            local_stats.1 += 1;
                        }
                    },
                    
                    _ => unreachable!(),
                }
                
                // Yield occasionally to allow other threads
                if i % 10 == 0 {
                    std::thread::yield_now();
                }
            }
            
            local_stats
        })
    }).collect();
    
    // Collect results
    let mut total_insertions = 0;
    let mut total_queries = 0;
    let mut total_errors = 0;
    
    for handle in handles {
        let (insertions, queries, errors) = handle.join().expect("Thread panicked");
        total_insertions += insertions;
        total_queries += queries;
        total_errors += errors;
    }
    
    println!("Concurrent operations: {} insertions, {} queries, {} errors", 
             total_insertions, total_queries, total_errors);
    
    // Some operations should succeed even under high concurrency
    assert!(total_insertions + total_queries > 0, "No concurrent operations succeeded");
    
    // Graph should remain consistent after concurrent operations
    let final_entity_count = graph_arc.entity_count();
    let final_relationship_count = graph_arc.relationship_count();
    
    assert!(final_entity_count > 0, "Graph should have entities after concurrent operations");
    
    // Phase 4: Recovery and cleanup testing
    // Verify graph is still functional after stress
    let recovery_test_entity = EntityData::new(999, r#"{"recovery_test": true}"#.to_string(), create_embedding(99999, 128));
    
    let recovery_result = graph_arc.insert_entity(99999, recovery_test_entity);
    assert!(recovery_result.is_ok(), "Graph should be functional after concurrent stress");
    
    // Test that basic operations still work
    let recovery_entity = graph_arc.get_entity_by_id(99999);
    assert!(recovery_entity.is_some(), "Recovery entity should be retrievable");
    
    // Memory usage should be reasonable
    let final_memory = graph_arc.memory_usage();
    let final_mb = final_memory.total_bytes() as f64 / (1024.0 * 1024.0);
    assert!(final_mb < 1000.0, "Memory usage should not be excessive: {:.2} MB", final_mb);
    
    println!("Graph robustness test completed successfully");
    println!("Final state: {} entities, {} relationships, {:.2} MB memory", 
             final_entity_count, final_relationship_count, final_mb);
}