//! Graph Structure Unit Tests
//!
//! Comprehensive tests for knowledge graph operations, CSR storage format,
//! memory efficiency, and concurrent access patterns.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::core::{Entity, EntityKey, KnowledgeGraph, Relationship, RelationshipType};
use std::time::Duration;

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn test_graph_basic_operations() {
        let mut graph = KnowledgeGraph::new();
        
        // Test 1: Empty graph properties
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        assert_eq!(graph.memory_usage(), EXPECTED_EMPTY_GRAPH_SIZE);
        
        // Test 2: Add single entity
        let entity_key = EntityKey::from_hash("entity_1");
        let entity = Entity::new(entity_key, "Entity 1".to_string());
        
        let add_result = graph.add_entity(entity.clone());
        assert!(add_result.is_ok());
        assert_eq!(graph.entity_count(), 1);
        assert!(graph.contains_entity(entity_key));
        
        // Test 3: Retrieve entity
        let retrieved = graph.get_entity(entity_key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), &entity);
        
        // Test 4: Add duplicate entity (should fail)
        let duplicate_result = graph.add_entity(entity.clone());
        assert!(duplicate_result.is_err());
        assert_eq!(graph.entity_count(), 1); // Count unchanged
        
        // Test 5: Add second entity
        let entity2_key = EntityKey::from_hash("entity_2");
        let entity2 = Entity::new(entity2_key, "Entity 2".to_string());
        
        graph.add_entity(entity2).unwrap();
        assert_eq!(graph.entity_count(), 2);
        
        // Test 6: Add relationship
        let relationship = Relationship::new("connects".to_string(), 1.0, RelationshipType::Directed);
        let rel_result = graph.add_relationship(entity_key, entity2_key, relationship);
        assert!(rel_result.is_ok());
        assert_eq!(graph.relationship_count(), 1);
        
        // Test 7: Verify relationship existence
        let relationships = graph.get_relationships(entity_key);
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].target(), entity2_key);
        assert_eq!(relationships[0].relationship().name(), "connects");
    }
    
    #[test]
    fn test_graph_csr_storage_format() {
        let mut graph = KnowledgeGraph::new();
        let (entities, relationships) = create_test_graph_data(100, 200);
        
        // Add all entities
        for entity in entities {
            graph.add_entity(entity).unwrap();
        }
        
        // Add all relationships
        for (source, target, rel) in relationships {
            graph.add_relationship(source, target, rel).unwrap();
        }
        
        // Test CSR format properties
        let csr_storage = graph.get_csr_storage();
        
        // Verify row offsets are monotonically increasing
        let offsets = csr_storage.row_offsets();
        for i in 1..offsets.len() {
            assert!(offsets[i] >= offsets[i-1], "CSR offsets not monotonic at index {}", i);
        }
        
        // Verify column indices are valid
        let columns = csr_storage.column_indices();
        let max_entity_id = graph.entity_count() as u32;
        for &col_idx in columns {
            assert!(col_idx < max_entity_id, "Invalid column index: {}", col_idx);
        }
        
        // Verify data integrity
        let total_relationships = offsets[offsets.len() - 1];
        assert_eq!(total_relationships as usize, columns.len());
        assert_eq!(total_relationships as u64, graph.relationship_count());
        
        // Test cache-friendly access patterns
        let access_time = measure_sequential_access_time(&csr_storage);
        let random_access_time = measure_random_access_time(&csr_storage);
        
        // Sequential access should be significantly faster
        assert!(access_time < random_access_time * 0.8, 
                "CSR format not showing cache-friendly behavior");
    }
    
    #[test]
    fn test_graph_memory_efficiency() {
        let mut graph = KnowledgeGraph::new();
        let entity_count = 1000u64;
        let relationship_count = 2000u64;
        
        // Add entities and track memory growth
        let mut memory_measurements = Vec::new();
        
        for i in 0..entity_count {
            let entity_key = EntityKey::from_hash(&format!("entity_{}", i));
            let entity = Entity::new(entity_key, format!("Entity {}", i));
            graph.add_entity(entity).unwrap();
            
            if i % 100 == 0 {
                memory_measurements.push((i, graph.memory_usage()));
            }
        }
        
        // Verify linear memory growth for entities
        for window in memory_measurements.windows(2) {
            let (count1, memory1) = window[0];
            let (count2, memory2) = window[1];
            
            let entity_diff = count2 - count1;
            let memory_diff = memory2 - memory1;
            let memory_per_entity = memory_diff / entity_diff;
            
            // Should be close to expected entity size
            assert!(memory_per_entity <= EXPECTED_ENTITY_SIZE_UPPER_BOUND,
                   "Memory per entity {} exceeds bound {}", 
                   memory_per_entity, EXPECTED_ENTITY_SIZE_UPPER_BOUND);
        }
        
        // Test target: < 70 bytes per entity
        let final_memory = graph.memory_usage();
        let memory_per_entity = final_memory / entity_count;
        assert!(memory_per_entity < 70, 
               "Memory per entity {} exceeds 70 byte target", memory_per_entity);
    }
    
    #[test]
    fn test_graph_concurrent_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let graph = Arc::new(Mutex::new(KnowledgeGraph::new()));
        let entity_count = 100;
        let thread_count = 4;
        
        // Prepare entities for each thread
        let entities_per_thread = entity_count / thread_count;
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let graph_clone = Arc::clone(&graph);
            
            let handle = thread::spawn(move || {
                let start_idx = thread_id * entities_per_thread;
                let end_idx = start_idx + entities_per_thread;
                
                for i in start_idx..end_idx {
                    let entity_key = EntityKey::from_hash(&format!("thread_{}_{}", thread_id, i));
                    let entity = Entity::new(entity_key, format!("Entity {}", i));
                    
                    let mut graph = graph_clone.lock().unwrap();
                    graph.add_entity(entity).unwrap();
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify final state
        let graph = graph.lock().unwrap();
        assert_eq!(graph.entity_count(), entity_count as u64);
        
        // Verify all entities are present and accessible
        for thread_id in 0..thread_count {
            for i in 0..entities_per_thread {
                let entity_key = EntityKey::from_hash(&format!("thread_{}_{}", thread_id, i));
                assert!(graph.contains_entity(entity_key));
            }
        }
    }

    #[test]
    fn test_graph_relationship_types() {
        let mut graph = KnowledgeGraph::new();
        
        // Create test entities
        let entity1 = Entity::new(EntityKey::from_hash("entity1"), "Entity 1".to_string());
        let entity2 = Entity::new(EntityKey::from_hash("entity2"), "Entity 2".to_string());
        let entity3 = Entity::new(EntityKey::from_hash("entity3"), "Entity 3".to_string());
        
        let key1 = entity1.key();
        let key2 = entity2.key();
        let key3 = entity3.key();
        
        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();
        graph.add_entity(entity3).unwrap();
        
        // Test directed relationship
        let directed_rel = Relationship::new("points_to".to_string(), 0.8, RelationshipType::Directed);
        graph.add_relationship(key1, key2, directed_rel).unwrap();
        
        // Verify directed relationship exists in one direction only
        let forward_rels = graph.get_relationships(key1);
        assert_eq!(forward_rels.len(), 1);
        assert_eq!(forward_rels[0].target(), key2);
        
        let backward_rels = graph.get_relationships(key2);
        assert_eq!(backward_rels.len(), 0);
        
        // Test undirected relationship
        let undirected_rel = Relationship::new("connected_to".to_string(), 0.9, RelationshipType::Undirected);
        graph.add_relationship(key2, key3, undirected_rel).unwrap();
        
        // Verify undirected relationship exists in both directions
        let rel_2_to_3 = graph.get_relationships(key2);
        let rel_3_to_2 = graph.get_relationships(key3);
        
        assert!(rel_2_to_3.iter().any(|r| r.target() == key3));
        assert!(rel_3_to_2.iter().any(|r| r.target() == key2));
        
        // Test weighted relationship
        let weighted_rel = Relationship::new("similar_to".to_string(), 0.5, RelationshipType::Weighted);
        graph.add_relationship(key1, key3, weighted_rel).unwrap();
        
        let weighted_rels = graph.get_relationships(key1);
        let similar_rel = weighted_rels.iter().find(|r| r.target() == key3).unwrap();
        assert_eq!(similar_rel.relationship().weight(), 0.5);
        assert_eq!(similar_rel.relationship().relationship_type(), &RelationshipType::Weighted);
    }

    #[test]
    fn test_graph_pathfinding() {
        let mut graph = create_path_test_graph();
        
        let start = EntityKey::from_hash("node_0");
        let end = EntityKey::from_hash("node_4");
        
        // Test shortest path exists
        let path = graph.find_shortest_path(start, end);
        assert!(path.is_some());
        
        let path = path.unwrap();
        assert_eq!(path.len(), 5); // 0 -> 1 -> 2 -> 3 -> 4
        assert_eq!(path[0], start);
        assert_eq!(path[4], end);
        
        // Test path length calculation
        let distance = graph.shortest_path_length(start, end);
        assert_eq!(distance, Some(4));
        
        // Test non-existent path
        let isolated = EntityKey::from_hash("isolated");
        let isolated_entity = Entity::new(isolated, "Isolated".to_string());
        graph.add_entity(isolated_entity).unwrap();
        
        let no_path = graph.find_shortest_path(start, isolated);
        assert!(no_path.is_none());
        
        let no_distance = graph.shortest_path_length(start, isolated);
        assert_eq!(no_distance, None);
    }

    #[test]
    fn test_graph_subgraph_extraction() {
        let graph = create_test_graph(50, 100);
        
        let seed_entity = EntityKey::from_hash("test_entity_0");
        let subgraph = graph.extract_subgraph(seed_entity, 2); // 2-hop neighborhood
        
        // Verify subgraph properties
        assert!(subgraph.entity_count() > 0);
        assert!(subgraph.entity_count() <= graph.entity_count());
        assert!(subgraph.contains_entity(seed_entity));
        
        // Verify all entities in subgraph are within 2 hops
        for entity_key in subgraph.get_all_entities() {
            if *entity_key != seed_entity {
                let distance = graph.shortest_path_length(seed_entity, *entity_key);
                assert!(distance.is_some());
                assert!(distance.unwrap() <= 2);
            }
        }
        
        // Test subgraph independence
        let mut modified_subgraph = subgraph.clone();
        let new_entity = Entity::new(EntityKey::from_hash("new_entity"), "New".to_string());
        modified_subgraph.add_entity(new_entity).unwrap();
        
        assert_ne!(subgraph.entity_count(), modified_subgraph.entity_count());
    }

    #[test]
    fn test_graph_serialization() {
        let original_graph = create_test_graph(20, 30);
        
        // Test JSON serialization
        let json_data = original_graph.to_json().unwrap();
        let json_graph = KnowledgeGraph::from_json(&json_data).unwrap();
        
        assert_eq!(original_graph.entity_count(), json_graph.entity_count());
        assert_eq!(original_graph.relationship_count(), json_graph.relationship_count());
        
        // Test binary serialization
        let binary_data = original_graph.to_binary().unwrap();
        let binary_graph = KnowledgeGraph::from_binary(&binary_data).unwrap();
        
        assert_eq!(original_graph.entity_count(), binary_graph.entity_count());
        assert_eq!(original_graph.relationship_count(), binary_graph.relationship_count());
        
        // Verify binary is more compact
        assert!(binary_data.len() < json_data.len());
        
        // Test roundtrip consistency
        let roundtrip_graph = KnowledgeGraph::from_binary(
            &binary_graph.to_binary().unwrap()
        ).unwrap();
        
        assert_eq!(original_graph.entity_count(), roundtrip_graph.entity_count());
        assert_eq!(original_graph.relationship_count(), roundtrip_graph.relationship_count());
    }

    #[test]
    fn test_graph_performance_scaling() {
        let sizes = vec![10, 50, 100, 500];
        
        for &size in &sizes {
            let (creation_time, graph) = measure_execution_time(|| {
                create_test_graph(size, size * 2)
            });
            
            println!("Graph creation time for {} entities: {:?}", size, creation_time);
            
            // Test entity lookup performance
            let entity_key = EntityKey::from_hash("test_entity_0");
            let (_, lookup_time) = measure_execution_time(|| {
                for _ in 0..1000 {
                    let _ = graph.get_entity(entity_key);
                }
            });
            
            println!("Entity lookup time (1000 operations): {:?}", lookup_time);
            
            // Lookup time should be constant regardless of graph size
            assert!(lookup_time.as_micros() < 10000, "Entity lookup too slow for size {}", size);
            
            // Test relationship traversal performance
            let (_, traversal_time) = measure_execution_time(|| {
                for _ in 0..100 {
                    let _ = graph.get_relationships(entity_key);
                }
            });
            
            println!("Relationship traversal time (100 operations): {:?}", traversal_time);
            assert!(traversal_time.as_micros() < 50000, "Relationship traversal too slow for size {}", size);
            
            // Memory usage should scale linearly
            let memory_per_entity = graph.memory_usage() / size as u64;
            assert!(memory_per_entity < 500, "Memory per entity too high: {} bytes", memory_per_entity);
        }
    }

    #[test]
    fn test_graph_batch_operations() {
        let mut graph = KnowledgeGraph::new();
        let batch_size = 1000;
        
        // Prepare batch entities
        let entities: Vec<Entity> = (0..batch_size)
            .map(|i| Entity::new(
                EntityKey::from_hash(&format!("batch_entity_{}", i)),
                format!("Batch Entity {}", i)
            ))
            .collect();
        
        // Test batch entity addition
        let (_, batch_add_time) = measure_execution_time(|| {
            graph.add_entities_batch(&entities).unwrap();
        });
        
        println!("Batch add time for {} entities: {:?}", batch_size, batch_add_time);
        assert_eq!(graph.entity_count(), batch_size as u64);
        assert!(batch_add_time.as_millis() < 1000, "Batch add too slow");
        
        // Test batch relationship addition
        let mut relationships = Vec::new();
        for i in 0..batch_size/2 {
            let source = EntityKey::from_hash(&format!("batch_entity_{}", i));
            let target = EntityKey::from_hash(&format!("batch_entity_{}", i + 1));
            let rel = Relationship::new("batch_connects".to_string(), 1.0, RelationshipType::Directed);
            relationships.push((source, target, rel));
        }
        
        let (_, batch_rel_time) = measure_execution_time(|| {
            graph.add_relationships_batch(&relationships).unwrap();
        });
        
        println!("Batch relationship add time for {} relationships: {:?}", 
                relationships.len(), batch_rel_time);
        assert_eq!(graph.relationship_count(), relationships.len() as u64);
        assert!(batch_rel_time.as_millis() < 500, "Batch relationship add too slow");
    }

    #[test]
    fn test_graph_error_conditions() {
        let mut graph = KnowledgeGraph::new();
        
        // Test adding relationship with non-existent entities
        let non_existent1 = EntityKey::from_hash("non_existent_1");
        let non_existent2 = EntityKey::from_hash("non_existent_2");
        let rel = Relationship::new("invalid".to_string(), 1.0, RelationshipType::Directed);
        
        let result = graph.add_relationship(non_existent1, non_existent2, rel);
        assert!(result.is_err());
        
        // Test getting entity that doesn't exist
        let non_existent_entity = graph.get_entity(non_existent1);
        assert!(non_existent_entity.is_none());
        
        // Test getting relationships for non-existent entity
        let non_existent_rels = graph.get_relationships(non_existent1);
        assert_eq!(non_existent_rels.len(), 0);
        
        // Test removing non-existent entity
        let remove_result = graph.remove_entity(non_existent1);
        assert!(remove_result.is_err());
        
        // Test invalid relationship weight
        let entity1 = Entity::new(EntityKey::from_hash("entity1"), "Entity 1".to_string());
        let entity2 = Entity::new(EntityKey::from_hash("entity2"), "Entity 2".to_string());
        
        graph.add_entity(entity1.clone()).unwrap();
        graph.add_entity(entity2.clone()).unwrap();
        
        let invalid_rel = Relationship::new("invalid".to_string(), -1.0, RelationshipType::Weighted);
        let invalid_result = graph.add_relationship(entity1.key(), entity2.key(), invalid_rel);
        assert!(invalid_result.is_err());
    }
}

// Helper functions for graph tests
fn create_test_graph_data(entity_count: usize, relationship_count: usize) -> (Vec<Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
    let mut rng = DeterministicRng::new(GRAPH_TEST_SEED);
    
    let entities: Vec<Entity> = (0..entity_count)
        .map(|i| Entity::new(
            EntityKey::from_hash(&format!("test_entity_{}", i)),
            format!("Test Entity {}", i)
        ))
        .collect();
    
    let mut relationships = Vec::new();
    for _ in 0..relationship_count {
        let source_idx = rng.gen_range(0..entity_count);
        let target_idx = rng.gen_range(0..entity_count);
        
        if source_idx != target_idx {
            let source_key = entities[source_idx].key();
            let target_key = entities[target_idx].key();
            let rel = Relationship::new(
                "test_relationship".to_string(),
                rng.gen_range(0.1..1.0),
                RelationshipType::Directed
            );
            relationships.push((source_key, target_key, rel));
        }
    }
    
    (entities, relationships)
}

fn create_path_test_graph() -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();
    
    // Create a linear path: 0 -> 1 -> 2 -> 3 -> 4
    for i in 0..5 {
        let entity = Entity::new(
            EntityKey::from_hash(&format!("node_{}", i)),
            format!("Node {}", i)
        );
        graph.add_entity(entity).unwrap();
    }
    
    for i in 0..4 {
        let source = EntityKey::from_hash(&format!("node_{}", i));
        let target = EntityKey::from_hash(&format!("node_{}", i + 1));
        let rel = Relationship::new("connects".to_string(), 1.0, RelationshipType::Directed);
        graph.add_relationship(source, target, rel).unwrap();
    }
    
    graph
}

fn measure_sequential_access_time(csr: &crate::storage::csr::CompressedSparseRow) -> Duration {
    let start = std::time::Instant::now();
    for row in 0..csr.num_rows() {
        let _row_data = csr.get_row(row);
    }
    start.elapsed()
}

fn measure_random_access_time(csr: &crate::storage::csr::CompressedSparseRow) -> Duration {
    let mut rng = DeterministicRng::new(ACCESS_PATTERN_SEED);
    let random_rows: Vec<usize> = (0..csr.num_rows())
        .map(|_| rng.gen_range(0..csr.num_rows()))
        .collect();
    
    let start = std::time::Instant::now();
    for &row in &random_rows {
        let _row_data = csr.get_row(row);
    }
    start.elapsed()
}