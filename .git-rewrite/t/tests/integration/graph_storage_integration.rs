// Graph-Storage Integration Tests
// Tests integration between graph structures and storage backends

use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};

use crate::test_infrastructure::*;
use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::knowledge_graph::KnowledgeGraph;
use crate::storage::csr::{CsrStorage, CsrMatrix};
use crate::storage::bloom::{BloomFilter, BloomConfig};
use crate::storage::index::{AttributeIndex, IndexType};

#[cfg(test)]
mod graph_storage_integration {
    use super::*;

    #[test]
    fn test_graph_csr_storage_integration() {
        let mut test_env = IntegrationTestEnvironment::new("graph_csr_integration");
        
        // Create test graph with known properties
        let graph_spec = GraphSpec {
            entity_count: 10000,
            relationship_count: 25000,
            topology: TopologyType::ScaleFree { exponent: 2.1 },
            clustering_coefficient: 0.3,
        };
        
        let test_graph = test_env.data_generator.generate_graph(&graph_spec);
        
        // Test 1: Graph construction and CSR conversion
        let mut kg = KnowledgeGraph::new();
        
        let construction_start = Instant::now();
        for entity in test_graph.entities.values() {
            kg.add_entity(entity.clone()).expect("Failed to add entity");
        }
        
        for (source, target, relationship) in test_graph.relationships {
            kg.add_relationship(source, target, relationship)
                .expect("Failed to add relationship");
        }
        let construction_time = construction_start.elapsed();
        
        // Verify graph properties
        assert_eq!(kg.entity_count(), graph_spec.entity_count);
        assert_eq!(kg.relationship_count(), graph_spec.relationship_count);
        
        // Test 2: CSR storage format integrity
        let csr_start = Instant::now();
        let csr_storage = kg.export_to_csr().expect("Failed to export to CSR");
        let csr_conversion_time = csr_start.elapsed();
        
        // Verify CSR format properties
        assert_eq!(csr_storage.num_nodes(), graph_spec.entity_count as usize);
        assert_eq!(csr_storage.num_edges(), graph_spec.relationship_count as usize);
        
        // Test 3: Query performance on CSR format
        let query_entities: Vec<EntityKey> = test_graph.entities.keys()
            .take(100)
            .cloned()
            .collect();
        
        let query_start = Instant::now();
        for &entity_key in &query_entities {
            let neighbors = kg.get_neighbors(entity_key);
            
            // Verify neighbors are reachable and valid
            for neighbor in neighbors {
                assert!(kg.contains_entity(neighbor.target()));
                assert!(neighbor.relationship().weight() > 0.0);
            }
        }
        let query_time = query_start.elapsed();
        
        // Performance validation
        let avg_query_time = query_time / query_entities.len() as u32;
        assert!(avg_query_time < Duration::from_micros(100),
               "Average query time too slow: {:?}", avg_query_time);
        
        println!("Graph construction: {:?}, CSR conversion: {:?}, Average query: {:?}", 
                construction_time, csr_conversion_time, avg_query_time);
        
        test_env.record_performance("graph_construction_time", construction_time);
        test_env.record_performance("csr_conversion_time", csr_conversion_time);
        test_env.record_performance("avg_query_time", avg_query_time);
        
        // Test 4: CSR space efficiency
        let graph_memory = kg.estimate_memory_usage();
        let csr_memory = csr_storage.memory_usage();
        let compression_ratio = graph_memory as f64 / csr_memory as f64;
        
        println!("Memory usage - Graph: {} bytes, CSR: {} bytes, Compression: {:.2}x",
                graph_memory, csr_memory, compression_ratio);
        
        assert!(compression_ratio > 1.5, 
               "CSR compression ratio too low: {:.2}x", compression_ratio);
        
        test_env.record_metric("csr_compression_ratio", compression_ratio);
        
        // Test 5: CSR batch operations
        let batch_start = Instant::now();
        let batch_queries: Vec<_> = query_entities.iter()
            .take(50)
            .cloned()
            .collect();
        
        let batch_results = csr_storage.batch_get_neighbors(&batch_queries);
        let batch_time = batch_start.elapsed();
        
        assert_eq!(batch_results.len(), batch_queries.len());
        
        let batch_speedup = (query_time / 2) as f32 / batch_time.as_secs_f32();
        assert!(batch_speedup > 1.5,
               "Batch processing not efficient enough: {:.2}x speedup", batch_speedup);
        
        test_env.record_metric("batch_speedup", batch_speedup as f64);
    }
    
    #[test]
    fn test_graph_bloom_filter_integration() {
        let mut test_env = IntegrationTestEnvironment::new("graph_bloom_integration");
        
        // Create graph with predictable membership patterns
        let entity_count = 5000u64;
        let relationship_count = 12000u64;
        
        let test_data = test_env.data_generator.generate_membership_test_data(
            entity_count, relationship_count
        );
        
        let mut kg = KnowledgeGraph::new();
        
        // Build graph with bloom filter enabled
        let bloom_config = BloomConfig {
            expected_items: entity_count,
            false_positive_rate: 0.01,
            hash_functions: None, // Use optimal
        };
        
        kg.enable_bloom_filter(bloom_config).unwrap();
        
        for entity in test_data.entities {
            kg.add_entity(entity).unwrap();
        }
        
        for (source, target, rel) in test_data.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        // Test 1: Positive membership (should have no false negatives)
        let positive_test_start = Instant::now();
        let mut false_negatives = 0;
        
        for entity_key in test_data.known_entities.iter().take(1000) {
            let bloom_result = kg.bloom_filter_contains(entity_key);
            let actual_result = kg.contains_entity(*entity_key);
            
            assert_eq!(actual_result, true, "Test entity should exist in graph");
            
            if !bloom_result {
                false_negatives += 1;
            }
        }
        let positive_test_time = positive_test_start.elapsed();
        
        assert_eq!(false_negatives, 0, "Bloom filter should have no false negatives");
        
        // Test 2: Negative membership (track false positives)
        let negative_test_start = Instant::now();
        let mut false_positives = 0;
        let test_count = 10000;
        
        for i in 0..test_count {
            let fake_entity = EntityKey::from_hash(format!("fake_entity_{}", i));
            let bloom_result = kg.bloom_filter_contains(&fake_entity);
            let actual_result = kg.contains_entity(fake_entity);
            
            assert_eq!(actual_result, false, "Fake entity should not exist");
            
            if bloom_result {
                false_positives += 1;
            }
        }
        let negative_test_time = negative_test_start.elapsed();
        
        let false_positive_rate = false_positives as f64 / test_count as f64;
        assert!(false_positive_rate <= 0.015, // Allow 50% margin over target
               "False positive rate too high: {} vs target 0.01", false_positive_rate);
        
        println!("False positive rate: {:.4} ({}/{})", 
                false_positive_rate, false_positives, test_count);
        
        // Test 3: Performance impact of bloom filter
        let entities_to_query: Vec<EntityKey> = (0..1000)
            .map(|i| EntityKey::from_hash(format!("perf_test_{}", i)))
            .collect();
        
        // Time bloom filter checks
        let bloom_start = Instant::now();
        for entity in &entities_to_query {
            let _ = kg.bloom_filter_contains(entity);
        }
        let bloom_time = bloom_start.elapsed();
        
        // Time actual membership checks
        let membership_start = Instant::now();
        for entity in &entities_to_query {
            let _ = kg.contains_entity(*entity);
        }
        let membership_time = membership_start.elapsed();
        
        // Bloom filter should be significantly faster
        let speedup = membership_time.as_nanos() as f64 / bloom_time.as_nanos() as f64;
        assert!(speedup > 5.0, "Bloom filter should provide significant speedup: {}x", speedup);
        
        println!("Bloom filter speedup: {:.2}x", speedup);
        
        test_env.record_metric("false_positive_rate", false_positive_rate);
        test_env.record_metric("bloom_speedup", speedup);
        test_env.record_performance("positive_test_time", positive_test_time);
        test_env.record_performance("negative_test_time", negative_test_time);
        
        // Test 4: Memory efficiency
        let bloom_memory = kg.bloom_filter_memory_usage();
        let bits_per_item = (bloom_memory * 8) as f64 / entity_count as f64;
        
        println!("Bloom filter memory: {} bytes, {:.2} bits per item", 
                bloom_memory, bits_per_item);
        
        // Should be close to theoretical optimal
        let theoretical_bits = -1.44 * 0.01_f64.log2();
        assert!(bits_per_item < theoretical_bits * 1.2,
               "Bloom filter using too much memory: {:.2} bits vs theoretical {:.2}",
               bits_per_item, theoretical_bits);
        
        test_env.record_metric("bloom_bits_per_item", bits_per_item);
    }
    
    #[test]
    fn test_graph_index_integration() {
        let mut test_env = IntegrationTestEnvironment::new("graph_index_integration");
        
        // Create graph with indexable attributes
        let graph_data = test_env.data_generator.generate_attributed_graph(
            2000, 5000, vec!["type", "category", "year", "score"]
        );
        
        let mut kg = KnowledgeGraph::new();
        
        // Build graph with indexing enabled
        kg.enable_attribute_indexing(vec!["type", "category", "year"]).unwrap();
        
        let build_start = Instant::now();
        for entity in graph_data.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        
        for (source, target, rel) in graph_data.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        let build_time = build_start.elapsed();
        
        println!("Graph build time with indexing: {:?}", build_time);
        test_env.record_performance("indexed_build_time", build_time);
        
        // Test 1: Attribute-based queries
        let type_values = vec!["paper", "author", "venue"];
        for type_value in type_values {
            let query_start = Instant::now();
            let entities = kg.find_entities_by_attribute("type", type_value);
            let query_time = query_start.elapsed();
            
            // Verify all results have the correct attribute
            for entity_key in &entities {
                let entity = kg.get_entity(*entity_key).unwrap();
                assert_eq!(entity.get_attribute("type"), Some(type_value));
            }
            
            // Performance should be sub-linear with total entity count
            assert!(query_time < Duration::from_millis(10),
                   "Attribute query too slow: {:?} for {} results", 
                   query_time, entities.len());
            
            println!("Query for type='{}': {} results in {:?}", 
                    type_value, entities.len(), query_time);
        }
        
        // Test 2: Range queries (for numeric attributes)
        let year_ranges = vec![(2000, 2010), (2010, 2020), (2020, 2025)];
        for (start_year, end_year) in year_ranges {
            let range_start = Instant::now();
            let entities = kg.find_entities_by_attribute_range(
                "year", 
                &start_year.to_string(), 
                &end_year.to_string()
            );
            let range_time = range_start.elapsed();
            
            // Verify all results are in the correct range
            for entity_key in &entities {
                let entity = kg.get_entity(*entity_key).unwrap();
                if let Some(year_str) = entity.get_attribute("year") {
                    let year: i32 = year_str.parse().unwrap();
                    assert!(year >= start_year && year <= end_year,
                           "Entity year {} not in range [{}, {}]", year, start_year, end_year);
                }
            }
            
            println!("Range query year in [{}, {}]: {} results in {:?}",
                    start_year, end_year, entities.len(), range_time);
            
            assert!(range_time < Duration::from_millis(15),
                   "Range query too slow: {:?}", range_time);
        }
        
        // Test 3: Compound queries
        let compound_start = Instant::now();
        let compound_results = kg.find_entities_by_multiple_attributes(vec![
            ("type", "paper"),
            ("category", "AI"),
        ]);
        let compound_time = compound_start.elapsed();
        
        // Verify all results match all criteria
        for entity_key in &compound_results {
            let entity = kg.get_entity(*entity_key).unwrap();
            assert_eq!(entity.get_attribute("type"), Some("paper"));
            assert_eq!(entity.get_attribute("category"), Some("AI"));
        }
        
        println!("Compound query: {} results in {:?}", 
                compound_results.len(), compound_time);
        
        assert!(compound_time < Duration::from_millis(20),
               "Compound query too slow: {:?}", compound_time);
        
        // Test 4: Index maintenance during updates
        let test_entities: Vec<_> = graph_data.entities.keys().take(10).cloned().collect();
        
        for &test_entity in &test_entities {
            let original_type = kg.get_entity(test_entity).unwrap()
                .get_attribute("type").unwrap().to_string();
            
            // Update entity attribute
            let update_start = Instant::now();
            kg.update_entity_attribute(test_entity, "type", "updated_type").unwrap();
            let update_time = update_start.elapsed();
            
            // Verify old index entry is removed
            let old_results = kg.find_entities_by_attribute("type", &original_type);
            assert!(!old_results.contains(&test_entity));
            
            // Verify new index entry is added
            let new_results = kg.find_entities_by_attribute("type", "updated_type");
            assert!(new_results.contains(&test_entity));
            
            assert!(update_time < Duration::from_millis(5),
                   "Index update too slow: {:?}", update_time);
        }
        
        // Test 5: Index memory overhead
        let index_memory = kg.index_memory_usage();
        let memory_per_entity = index_memory as f64 / graph_data.entities.len() as f64;
        
        println!("Index memory: {} bytes, {:.2} bytes per entity",
                index_memory, memory_per_entity);
        
        // Index shouldn't use excessive memory
        assert!(memory_per_entity < 100.0,
               "Index memory overhead too high: {:.2} bytes per entity", memory_per_entity);
        
        test_env.record_metric("index_memory_per_entity", memory_per_entity);
        
        // Test 6: Index statistics
        let index_stats = kg.get_index_statistics();
        
        println!("Index statistics:");
        for (attr, stats) in index_stats {
            println!("  {}: {} unique values, {} total entries", 
                    attr, stats.unique_values, stats.total_entries);
            
            assert!(stats.unique_values > 0);
            assert!(stats.total_entries > 0);
            assert!(stats.total_entries >= stats.unique_values);
        }
    }
    
    #[test]
    fn test_graph_storage_concurrent_access() {
        let mut test_env = IntegrationTestEnvironment::new("graph_storage_concurrent");
        
        // Create a medium-sized graph
        let graph_spec = GraphSpec {
            entity_count: 5000,
            relationship_count: 10000,
            topology: TopologyType::Random,
            clustering_coefficient: 0.0,
        };
        
        let test_graph = test_env.data_generator.generate_graph(&graph_spec);
        
        // Build graph with all storage features
        let kg = std::sync::Arc::new(std::sync::RwLock::new(KnowledgeGraph::new()));
        
        {
            let mut kg_write = kg.write().unwrap();
            
            // Enable features
            kg_write.enable_bloom_filter(BloomConfig {
                expected_items: graph_spec.entity_count,
                false_positive_rate: 0.01,
                hash_functions: None,
            }).unwrap();
            
            kg_write.enable_attribute_indexing(vec!["type"]).unwrap();
            
            // Build graph
            for entity in test_graph.entities.values() {
                kg_write.add_entity(entity.clone()).unwrap();
            }
            
            for (source, target, rel) in test_graph.relationships {
                kg_write.add_relationship(source, target, rel).unwrap();
            }
        }
        
        // Test concurrent reads
        let num_readers = 8;
        let queries_per_reader = 1000;
        let entity_keys: Vec<_> = test_graph.entities.keys().cloned().collect();
        
        let mut reader_handles = Vec::new();
        
        for reader_id in 0..num_readers {
            let kg_clone = kg.clone();
            let keys_clone = entity_keys.clone();
            
            let handle = std::thread::spawn(move || {
                let mut query_times = Vec::new();
                let mut bloom_hits = 0;
                let mut index_queries = 0;
                
                for i in 0..queries_per_reader {
                    let query_start = Instant::now();
                    
                    let kg_read = kg_clone.read().unwrap();
                    
                    // Mix of different query types
                    match i % 4 {
                        0 => {
                            // Neighbor query
                            let key = keys_clone[i % keys_clone.len()];
                            let _ = kg_read.get_neighbors(key);
                        }
                        1 => {
                            // Bloom filter query
                            let test_key = if i % 2 == 0 {
                                keys_clone[i % keys_clone.len()]
                            } else {
                                EntityKey::from_hash(format!("nonexistent_{}", i))
                            };
                            if kg_read.bloom_filter_contains(&test_key) {
                                bloom_hits += 1;
                            }
                        }
                        2 => {
                            // Index query
                            let _ = kg_read.find_entities_by_attribute("type", "paper");
                            index_queries += 1;
                        }
                        3 => {
                            // Entity lookup
                            let key = keys_clone[i % keys_clone.len()];
                            let _ = kg_read.get_entity(key);
                        }
                        _ => unreachable!(),
                    }
                    
                    drop(kg_read);
                    let query_time = query_start.elapsed();
                    query_times.push(query_time);
                }
                
                (reader_id, query_times, bloom_hits, index_queries)
            });
            
            reader_handles.push(handle);
        }
        
        // Wait for all readers
        let mut all_times = Vec::new();
        
        for handle in reader_handles {
            let (reader_id, times, bloom_hits, index_queries) = handle.join().unwrap();
            
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let max_time = times.iter().max().unwrap();
            
            println!("Reader {}: avg {:?}, max {:?}, bloom hits: {}, index queries: {}",
                    reader_id, avg_time, max_time, bloom_hits, index_queries);
            
            assert!(avg_time < Duration::from_millis(1),
                   "Reader {} average query time too high: {:?}", reader_id, avg_time);
            
            assert!(max_time < Duration::from_millis(10),
                   "Reader {} max query time too high: {:?}", reader_id, max_time);
            
            all_times.extend(times);
        }
        
        // Overall statistics
        let total_queries = all_times.len();
        let overall_avg = all_times.iter().sum::<Duration>() / total_queries as u32;
        
        println!("Total queries: {}, Overall average: {:?}", total_queries, overall_avg);
        
        test_env.record_performance("concurrent_avg_query", overall_avg);
        test_env.record_metric("concurrent_readers", num_readers as f64);
        test_env.record_metric("total_concurrent_queries", total_queries as f64);
    }
    
    #[test]
    fn test_graph_storage_persistence_integration() {
        let mut test_env = IntegrationTestEnvironment::new("graph_storage_persistence");
        
        // Create test graph
        let graph_spec = GraphSpec {
            entity_count: 1000,
            relationship_count: 2000,
            topology: TopologyType::SmallWorld { rewiring_prob: 0.1 },
            clustering_coefficient: 0.3,
        };
        
        let test_graph = test_env.data_generator.generate_graph(&graph_spec);
        let temp_path = test_env.temp_dir.join("test_graph.llmkg");
        
        // Build and save graph
        let save_time = {
            let mut kg = KnowledgeGraph::new();
            
            for entity in test_graph.entities.values() {
                kg.add_entity(entity.clone()).unwrap();
            }
            
            for (source, target, rel) in &test_graph.relationships {
                kg.add_relationship(*source, *target, rel.clone()).unwrap();
            }
            
            // Enable storage features
            kg.enable_bloom_filter(BloomConfig {
                expected_items: graph_spec.entity_count,
                false_positive_rate: 0.01,
                hash_functions: None,
            }).unwrap();
            
            kg.enable_attribute_indexing(vec!["type"]).unwrap();
            
            let save_start = Instant::now();
            kg.save_to_file(&temp_path).unwrap();
            save_start.elapsed()
        };
        
        println!("Save time: {:?}", save_time);
        
        // Load graph and verify
        let load_time = {
            let load_start = Instant::now();
            let loaded_kg = KnowledgeGraph::load_from_file(&temp_path).unwrap();
            let load_elapsed = load_start.elapsed();
            
            // Verify entity count
            assert_eq!(loaded_kg.entity_count(), graph_spec.entity_count);
            assert_eq!(loaded_kg.relationship_count(), graph_spec.relationship_count);
            
            // Verify random samples
            for entity_key in test_graph.entities.keys().take(100) {
                assert!(loaded_kg.contains_entity(*entity_key));
                
                let original = test_graph.entities.get(entity_key).unwrap();
                let loaded = loaded_kg.get_entity(*entity_key).unwrap();
                
                assert_eq!(original.name(), loaded.name());
                assert_eq!(original.attributes(), loaded.attributes());
            }
            
            // Verify bloom filter works
            for entity_key in test_graph.entities.keys().take(100) {
                assert!(loaded_kg.bloom_filter_contains(entity_key));
            }
            
            // Verify indexes work
            let indexed_results = loaded_kg.find_entities_by_attribute("type", "paper");
            assert!(!indexed_results.is_empty());
            
            load_elapsed
        };
        
        println!("Load time: {:?}", load_time);
        
        // Check file size
        let file_size = std::fs::metadata(&temp_path).unwrap().len();
        let size_per_entity = file_size / graph_spec.entity_count;
        
        println!("File size: {} bytes, {} bytes per entity", file_size, size_per_entity);
        
        assert!(size_per_entity < 1000,
               "Storage size per entity too large: {} bytes", size_per_entity);
        
        test_env.record_performance("save_time", save_time);
        test_env.record_performance("load_time", load_time);
        test_env.record_metric("file_size_bytes", file_size as f64);
        test_env.record_metric("bytes_per_entity", size_per_entity as f64);
    }
}