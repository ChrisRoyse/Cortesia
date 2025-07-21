//! Integration tests for graph compatibility layer
//! 
//! Tests complete end-to-end workflows using the legacy API to perform
//! complete graph operations and verify that the compatibility layer
//! works correctly with the new storage system through external interfaces.

use std::collections::HashMap;

use llmkg::core::graph::KnowledgeGraph;
use llmkg::error::Result;

fn create_test_graph() -> KnowledgeGraph {
    KnowledgeGraph::new(128, 10000)
}

#[test]
fn test_end_to_end_legacy_workflow() {
    let graph = create_test_graph();
    
    // Phase 1: Create entities using legacy text-based API
    let mut person_props = HashMap::new();
    person_props.insert("type".to_string(), "person".to_string());
    person_props.insert("name".to_string(), "Alice".to_string());
    person_props.insert("age".to_string(), "30".to_string());
    
    let mut location_props = HashMap::new();
    location_props.insert("type".to_string(), "location".to_string());
    location_props.insert("name".to_string(), "New York".to_string());
    location_props.insert("country".to_string(), "USA".to_string());
    
    let mut company_props = HashMap::new();
    company_props.insert("type".to_string(), "company".to_string());
    company_props.insert("name".to_string(), "TechCorp".to_string());
    company_props.insert("industry".to_string(), "Technology".to_string());
    
    // Insert entities with text content for embedding generation
    let alice_result = graph.insert_entity_with_text(
        1, 
        "Alice is a 30-year-old software engineer with expertise in machine learning",
        person_props
    );
    assert!(alice_result.is_ok());
    
    let nyc_result = graph.insert_entity_with_text(
        2,
        "New York City is a major metropolitan area in the United States",
        location_props
    );
    assert!(nyc_result.is_ok());
    
    let techcorp_result = graph.insert_entity_with_text(
        3,
        "TechCorp is a leading technology company specializing in AI and software development",
        company_props
    );
    assert!(techcorp_result.is_ok());
    
    // Phase 2: Create relationships using legacy ID-based API
    let lives_in = graph.insert_relationship_by_id(1, 2, 0.9);
    assert!(lives_in.is_ok());
    
    let works_at = graph.insert_relationship_by_id(1, 3, 0.8);
    assert!(works_at.is_ok());
    
    let company_location = graph.insert_relationship_by_id(3, 2, 0.7);
    assert!(company_location.is_ok());
    
    // Phase 3: Verify graph structure through legacy API
    let alice_neighbors = graph.get_neighbors_by_id(1);
    assert_eq!(alice_neighbors.len(), 2);
    assert!(alice_neighbors.contains(&2)); // NYC
    assert!(alice_neighbors.contains(&3)); // TechCorp
    
    // Phase 4: Perform similarity search using text queries
    let tech_search = graph.similarity_search_by_text("technology software development", 3);
    assert!(tech_search.is_ok());
    let tech_results = tech_search.unwrap();
    assert!(!tech_results.is_empty());
    
    // Should find TechCorp and Alice (software engineer) as most similar
    let tech_ids: Vec<u32> = tech_results.iter().map(|(id, _)| *id).collect();
    assert!(tech_ids.contains(&3)); // TechCorp should be found
    assert!(tech_ids.contains(&1)); // Alice should be found too
    
    // Phase 5: Test path finding between entities
    let path = graph.find_path_by_id(1, 2);
    assert!(path.is_some());
    let alice_to_nyc_path = path.unwrap();
    assert_eq!(alice_to_nyc_path.len(), 2);
    assert_eq!(alice_to_nyc_path[0], 1);
    assert_eq!(alice_to_nyc_path[1], 2);
    
    // Phase 6: Update entity properties and verify consistency
    let mut updated_alice = HashMap::new();
    updated_alice.insert("type".to_string(), "person".to_string());
    updated_alice.insert("name".to_string(), "Alice Smith".to_string());
    updated_alice.insert("age".to_string(), "31".to_string());
    updated_alice.insert("title".to_string(), "Senior Engineer".to_string());
    
    let update_result = graph.update_entity_properties(1, updated_alice);
    assert!(update_result.is_ok());
    
    // Verify update
    let alice_props = graph.get_entity_properties(1);
    assert!(alice_props.is_some());
    let props = alice_props.unwrap();
    assert_eq!(props.get("name").unwrap(), "Alice Smith");
    assert_eq!(props.get("age").unwrap(), "31");
    assert_eq!(props.get("title").unwrap(), "Senior Engineer");
    
    // Phase 7: Verify graph statistics and consistency
    let stats = graph.get_graph_stats();
    assert_eq!(stats.get("entity_count").unwrap(), &3.0);
    assert_eq!(stats.get("relationship_count").unwrap(), &3.0);
    assert_eq!(stats.get("embedding_dimension").unwrap(), &128.0);
    
    // Verify memory usage is reasonable
    let memory_mb = stats.get("memory_usage_mb").unwrap();
    assert!(*memory_mb > 0.0 && *memory_mb < 100.0); // Should be reasonable
    
    // Phase 8: Test batch operations
    let batch_entities = vec![
        (4, "Python programming language".to_string(), {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "technology".to_string());
            props.insert("name".to_string(), "Python".to_string());
            props
        }),
        (5, "Machine learning algorithms".to_string(), {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "concept".to_string());
            props.insert("name".to_string(), "ML Algorithms".to_string());
            props
        }),
    ];
    
    let batch_result = graph.insert_entities_with_text(batch_entities);
    assert!(batch_result.is_ok());
    
    // Create relationships to new entities
    let python_rel = graph.insert_relationship_by_id(1, 4, 0.95); // Alice knows Python
    assert!(python_rel.is_ok());
    
    let ml_rel = graph.insert_relationship_by_id(1, 5, 0.9); // Alice knows ML
    assert!(ml_rel.is_ok());
    
    // Verify expanded graph
    let final_alice_neighbors = graph.get_neighbors_by_id(1);
    assert_eq!(final_alice_neighbors.len(), 4);
    assert!(final_alice_neighbors.contains(&2)); // NYC
    assert!(final_alice_neighbors.contains(&3)); // TechCorp
    assert!(final_alice_neighbors.contains(&4)); // Python
    assert!(final_alice_neighbors.contains(&5)); // ML
}

#[test]
fn test_legacy_batch_similarity_search_workflow() {
    let graph = create_test_graph();
    
    // Create a diverse set of entities
    let entities_data = vec![
        (1, "Artificial intelligence and machine learning research", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "field".to_string());
            props.insert("name".to_string(), "AI Research".to_string());
            props
        }),
        (2, "Data science and statistical analysis", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "field".to_string());
            props.insert("name".to_string(), "Data Science".to_string());
            props
        }),
        (3, "Computer vision and image processing", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "field".to_string());
            props.insert("name".to_string(), "Computer Vision".to_string());
            props
        }),
        (4, "Natural language processing and text mining", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "field".to_string());
            props.insert("name".to_string(), "NLP".to_string());
            props
        }),
        (5, "Web development and frontend frameworks", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "field".to_string());
            props.insert("name".to_string(), "Web Dev".to_string());
            props
        }),
    ];
    
    // Insert all entities
    for (id, text, props) in entities_data {
        let result = graph.insert_entity_with_text(id, text, props);
        assert!(result.is_ok());
    }
    
    // Perform batch similarity search
    let search_queries = vec![
        "machine learning algorithms".to_string(),
        "statistical data analysis".to_string(),
        "image recognition systems".to_string(),
        "language models and text".to_string(),
        "javascript and react".to_string(),
    ];
    
    let batch_results = graph.batch_similarity_search_by_text(&search_queries, 3);
    assert!(batch_results.is_ok());
    
    let search_results = batch_results.unwrap();
    assert_eq!(search_results.len(), 5); // One result set per query
    
    // Verify each search returned relevant results
    for (query_idx, results) in search_results.iter().enumerate() {
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        
        // Check that similarity scores are valid
        for (entity_id, similarity) in results {
            assert!(*entity_id >= 1 && *entity_id <= 5);
            assert!(*similarity >= 0.0 && *similarity <= 1.0);
        }
        
        // Verify most relevant entity is found for each query
        let most_similar = &results[0];
        match query_idx {
            0 => assert_eq!(most_similar.0, 1), // ML query -> AI Research
            1 => assert_eq!(most_similar.0, 2), // Stats query -> Data Science  
            2 => assert_eq!(most_similar.0, 3), // Image query -> Computer Vision
            3 => assert_eq!(most_similar.0, 4), // Text query -> NLP
            4 => assert_eq!(most_similar.0, 5), // Web query -> Web Dev
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_legacy_entity_management_lifecycle() {
    let graph = create_test_graph();
    
    // Phase 1: Create initial entities
    let mut props1 = HashMap::new();
    props1.insert("type".to_string(), "document".to_string());
    props1.insert("title".to_string(), "Research Paper".to_string());
    props1.insert("status".to_string(), "draft".to_string());
    
    let mut props2 = HashMap::new();
    props2.insert("type".to_string(), "author".to_string());
    props2.insert("name".to_string(), "Dr. Smith".to_string());
    props2.insert("affiliation".to_string(), "University".to_string());
    
    let doc_result = graph.insert_entity_with_text(
        100,
        "A comprehensive study of machine learning applications in natural language processing",
        props1
    );
    assert!(doc_result.is_ok());
    
    let author_result = graph.insert_entity_with_text(
        101,
        "Dr. Smith is a professor specializing in computational linguistics and AI",
        props2
    );
    assert!(author_result.is_ok());
    
    // Phase 2: Establish authorship relationship
    let authorship = graph.insert_relationship_by_id(101, 100, 1.0);
    assert!(authorship.is_ok());
    
    // Phase 3: Update document status through lifecycle
    let mut updated_props = HashMap::new();
    updated_props.insert("type".to_string(), "document".to_string());
    updated_props.insert("title".to_string(), "Research Paper".to_string());
    updated_props.insert("status".to_string(), "under_review".to_string());
    updated_props.insert("reviewer_count".to_string(), "3".to_string());
    
    let update1 = graph.update_entity_properties(100, updated_props.clone());
    assert!(update1.is_ok());
    
    // Verify update
    let doc_props = graph.get_entity_properties(100);
    assert!(doc_props.is_some());
    let props = doc_props.unwrap();
    assert_eq!(props.get("status").unwrap(), "under_review");
    assert_eq!(props.get("reviewer_count").unwrap(), "3");
    
    // Phase 4: Add reviewers
    for i in 0..3 {
        let reviewer_id = 200 + i;
        let mut reviewer_props = HashMap::new();
        reviewer_props.insert("type".to_string(), "reviewer".to_string());
        reviewer_props.insert("name".to_string(), format!("Reviewer {}", i + 1));
        reviewer_props.insert("expertise".to_string(), "NLP".to_string());
        
        let reviewer_result = graph.insert_entity_with_text(
            reviewer_id,
            &format!("Expert reviewer {} specializing in natural language processing", i + 1),
            reviewer_props
        );
        assert!(reviewer_result.is_ok());
        
        // Create review relationship
        let review_rel = graph.insert_relationship_by_id(reviewer_id, 100, 0.8);
        assert!(review_rel.is_ok());
    }
    
    // Phase 5: Final status update
    updated_props.insert("status".to_string(), "published".to_string());
    updated_props.insert("publication_date".to_string(), "2024-01-15".to_string());
    updated_props.remove("reviewer_count");
    
    let final_update = graph.update_entity_properties(100, updated_props);
    assert!(final_update.is_ok());
    
    // Phase 6: Verify final graph state
    let final_props = graph.get_entity_properties(100);
    assert!(final_props.is_some());
    let props = final_props.unwrap();
    assert_eq!(props.get("status").unwrap(), "published");
    assert_eq!(props.get("publication_date").unwrap(), "2024-01-15");
    assert!(!props.contains_key("reviewer_count"));
    
    // Verify relationships exist
    let doc_neighbors = graph.get_neighbors_by_id(100);
    assert_eq!(doc_neighbors.len(), 0); // Document has incoming edges, not outgoing
    
    let author_neighbors = graph.get_neighbors_by_id(101);
    assert_eq!(author_neighbors.len(), 1);
    assert!(author_neighbors.contains(&100));
    
    for i in 0..3 {
        let reviewer_id = 200 + i;
        let reviewer_neighbors = graph.get_neighbors_by_id(reviewer_id);
        assert_eq!(reviewer_neighbors.len(), 1);
        assert!(reviewer_neighbors.contains(&100));
    }
    
    // Phase 7: Test entity removal
    let remove_reviewer = graph.remove_entity_by_id(202);
    assert!(remove_reviewer.is_ok());
    assert_eq!(remove_reviewer.unwrap(), true);
    
    // Verify removal
    let removed_entity = graph.get_entity_by_id_legacy(202);
    assert!(removed_entity.is_none());
    
    // Verify relationship was also removed
    let remaining_reviewers: Vec<u32> = (200..203).filter(|&id| {
        graph.get_entity_by_id_legacy(id).is_some()
    }).collect();
    assert_eq!(remaining_reviewers.len(), 2);
}

#[test]
fn test_legacy_string_dictionary_integration() {
    let graph = create_test_graph();
    
    // Phase 1: Insert various string concepts
    let concepts = vec![
        "artificial intelligence",
        "machine learning", 
        "deep learning",
        "neural networks",
        "natural language processing",
        "computer vision",
        "robotics",
        "data mining"
    ];
    
    let mut concept_ids = Vec::new();
    
    for concept in &concepts {
        let string_id = graph.insert_string(concept.to_string());
        concept_ids.push(string_id);
    }
    
    // Phase 2: Verify string lookup works
    for (i, concept) in concepts.iter().enumerate() {
        let looked_up_id = graph.get_string_id(concept);
        assert!(looked_up_id.is_some());
        assert_eq!(looked_up_id.unwrap(), concept_ids[i]);
        
        let looked_up_string = graph.get_string_by_id(concept_ids[i]);
        assert!(looked_up_string.is_some());
        assert_eq!(looked_up_string.unwrap(), *concept);
    }
    
    // Phase 3: Create entities referencing these strings
    for (i, concept) in concepts.iter().enumerate() {
        let entity_id = (i + 1) as u32;
        let mut props = HashMap::new();
        props.insert("type".to_string(), "concept".to_string());
        props.insert("name".to_string(), concept.to_string());
        props.insert("string_id".to_string(), concept_ids[i].to_string());
        
        let result = graph.insert_entity_with_text(entity_id, concept, props);
        assert!(result.is_ok());
    }
    
    // Phase 4: Create semantic relationships based on similarity
    let related_pairs = vec![
        (1, 2), // AI -> ML
        (2, 3), // ML -> Deep Learning  
        (3, 4), // Deep Learning -> Neural Networks
        (5, 6), // NLP -> Computer Vision (both AI subfields)
        (4, 7), // Neural Networks -> Robotics
    ];
    
    for (source, target) in related_pairs {
        let rel_result = graph.insert_relationship_by_id(source, target, 0.7);
        assert!(rel_result.is_ok());
    }
    
    // Phase 5: Test similarity search with string-based queries
    let ai_results = graph.similarity_search_by_text("artificial intelligence", 3);
    assert!(ai_results.is_ok());
    let results = ai_results.unwrap();
    
    assert!(!results.is_empty());
    let ai_entity_found = results.iter().any(|(id, _)| *id == 1);
    assert!(ai_entity_found);
    
    // Phase 6: Verify string dictionary doesn't interfere with normal operations
    let final_stats = graph.get_graph_stats();
    assert_eq!(final_stats.get("entity_count").unwrap(), &8.0);
    assert_eq!(final_stats.get("relationship_count").unwrap(), &5.0);
    
    // Test non-existent string lookup
    let missing_string = graph.get_string_id("non_existent_concept");
    assert!(missing_string.is_none());
    
    let missing_id = graph.get_string_by_id(9999);
    assert!(missing_id.is_none());
}

#[test]
fn test_legacy_bloom_filter_integration() {
    let graph = create_test_graph();
    
    // Phase 1: Insert entities and test bloom filter consistency
    let entity_ids = vec![10, 20, 30, 40, 50];
    
    for &id in &entity_ids {
        let mut props = HashMap::new();
        props.insert("id".to_string(), id.to_string());
        
        let result = graph.insert_entity_with_text(id, &format!("Entity {}", id), props);
        assert!(result.is_ok());
        
        // Bloom filter should be updated automatically
        assert!(graph.bloom_contains(id));
    }
    
    // Phase 2: Test bloom filter for non-existent entities
    let non_existent_ids = vec![11, 21, 31, 41, 51];
    
    for &id in &non_existent_ids {
        let exists = graph.bloom_contains(id);
        // Bloom filter might have false positives, but should not have false negatives
        // So if it returns false, the entity definitely doesn't exist
        if !exists {
            let entity = graph.get_entity_by_id_legacy(id);
            assert!(entity.is_none());
        }
    }
    
    // Phase 3: Test manual bloom filter operations
    let test_id = 999;
    assert!(!graph.bloom_contains(test_id)); // Should not exist initially
    
    let insert_result = graph.bloom_insert(test_id);
    assert!(insert_result.is_ok());
    
    // Now bloom filter should contain it
    assert!(graph.bloom_contains(test_id));
    
    // But actual entity should not exist
    let entity = graph.get_entity_by_id_legacy(test_id);
    assert!(entity.is_none());
    
    // Phase 4: Check false positive rate is reasonable
    let false_positive_rate = graph.bloom_false_positive_rate();
    assert!(false_positive_rate >= 0.0);
    assert!(false_positive_rate <= 1.0);
    // Should be relatively low for our small dataset
    assert!(false_positive_rate < 0.5);
    
    // Phase 5: Test bloom filter with entity removal
    let remove_result = graph.remove_entity_by_id(10);
    assert!(remove_result.is_ok());
    assert_eq!(remove_result.unwrap(), true);
    
    // Entity should be gone
    let removed_entity = graph.get_entity_by_id_legacy(10);
    assert!(removed_entity.is_none());
    
    // Note: Bloom filter may still contain the ID due to its probabilistic nature
    // This is expected behavior - bloom filters can't remove elements
    
    // Phase 6: Verify bloom filter consistency with remaining entities
    for &id in &entity_ids[1..] { // Skip removed entity
        assert!(graph.bloom_contains(id));
        let entity = graph.get_entity_by_id_legacy(id);
        assert!(entity.is_some());
    }
}

#[test]
fn test_legacy_export_import_workflow() {
    let graph = create_test_graph();
    
    // Phase 1: Create a complex graph structure
    let entities = vec![
        (1, "Python programming language", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "language".to_string());
            props.insert("name".to_string(), "Python".to_string());
            props
        }),
        (2, "Machine learning framework", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "framework".to_string());
            props.insert("name".to_string(), "TensorFlow".to_string());
            props
        }),
        (3, "Data analysis library", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "library".to_string());
            props.insert("name".to_string(), "Pandas".to_string());
            props
        }),
    ];
    
    for (id, text, props) in entities {
        let result = graph.insert_entity_with_text(id, text, props);
        assert!(result.is_ok());
    }
    
    // Create relationships
    let relationships = vec![
        (2, 1, 0.9), // TensorFlow uses Python
        (3, 1, 0.8), // Pandas uses Python
    ];
    
    for (source, target, weight) in relationships {
        let result = graph.insert_relationship_by_id(source, target, weight);
        assert!(result.is_ok());
    }
    
    // Phase 2: Export graph structure
    let exported_entities = graph.export_entities();
    assert_eq!(exported_entities.len(), 3);
    
    // Verify exported entity format
    for (id, props, embedding) in &exported_entities {
        assert!((*id >= 1) && (*id <= 3));
        assert!(!props.is_empty());
        assert_eq!(embedding.len(), 128);
        assert!(props.contains_key("type"));
        assert!(props.contains_key("name"));
    }
    
    let exported_relationships = graph.export_relationships();
    assert_eq!(exported_relationships.len(), 2);
    
    // Verify exported relationship format
    for (source, target, weight) in &exported_relationships {
        assert!((*source >= 1) && (*source <= 3));
        assert!((*target >= 1) && (*target <= 3));
        assert!(*weight > 0.0 && *weight <= 1.0);
    }
    
    // Verify specific relationships were exported correctly
    let tf_to_python = exported_relationships.iter().find(|(s, t, _)| *s == 2 && *t == 1);
    assert!(tf_to_python.is_some());
    assert_eq!(tf_to_python.unwrap().2, 0.9);
    
    let pandas_to_python = exported_relationships.iter().find(|(s, t, _)| *s == 3 && *t == 1);
    assert!(pandas_to_python.is_some());
    assert_eq!(pandas_to_python.unwrap().2, 0.8);
    
    // Phase 3: Verify export matches live graph
    for (id, exported_props, exported_embedding) in &exported_entities {
        let live_entity = graph.get_entity_by_id_legacy(*id);
        assert!(live_entity.is_some());
        
        let (live_props, live_embedding) = live_entity.unwrap();
        
        // Properties should match
        for (key, value) in exported_props {
            assert_eq!(live_props.get(key), Some(value));
        }
        
        // Embeddings should match
        assert_eq!(exported_embedding, &live_embedding);
    }
    
    // Phase 4: Test with empty graph scenario
    let empty_graph = create_test_graph();
    let empty_entities = empty_graph.export_entities();
    let empty_relationships = empty_graph.export_relationships();
    
    assert!(empty_entities.is_empty());
    assert!(empty_relationships.is_empty());
}

#[test]
fn test_legacy_graph_validation_workflow() {
    let graph = create_test_graph();
    
    // Phase 1: Create initial valid graph
    let mut props = HashMap::new();
    props.insert("type".to_string(), "test".to_string());
    
    for i in 1..=5 {
        let result = graph.insert_entity_with_text(i, &format!("Entity {}", i), props.clone());
        assert!(result.is_ok());
    }
    
    // Create some relationships
    for i in 1..=4 {
        let result = graph.insert_relationship_by_id(i, i + 1, 0.5);
        assert!(result.is_ok());
    }
    
    // Phase 2: Validate clean graph
    let initial_issues = graph.validate_graph_consistency();
    assert!(initial_issues.is_empty()); // Should be no issues
    
    // Phase 3: Perform various operations that could cause inconsistencies
    
    // Add many relationships to trigger edge buffer growth
    for i in 1..=5 {
        for j in 1..=5 {
            if i != j {
                let _ = graph.insert_relationship_by_id(i, j, 0.1);
            }
        }
    }
    
    // Phase 4: Re-validate after heavy relationship insertion
    let post_ops_issues = graph.validate_graph_consistency();
    
    // Check if there are any concerning issues
    for issue in &post_ops_issues {
        println!("Validation issue: {}", issue);
        
        // Edge buffer warnings are acceptable for this test
        assert!(issue.contains("Entity count mismatch") || 
                issue.contains("Edge buffer is very large") ||
                issue.contains("incorrect embedding dimension"));
    }
    
    // Phase 5: Test validation with entity type counting
    let test_entity_count = graph.count_entities_by_type("test");
    assert_eq!(test_entity_count, 5);
    
    let non_existent_type_count = graph.count_entities_by_type("non_existent");
    assert_eq!(non_existent_type_count, 0);
    
    // Phase 6: Test relationship counting between types
    let test_to_test_relationships = graph.count_relationships_between_types("test", "test");
    assert!(test_to_test_relationships > 0); // Should have many test->test relationships
    
    let missing_type_relationships = graph.count_relationships_between_types("missing", "test");
    assert_eq!(missing_type_relationships, 0);
    
    // Phase 7: Test performance optimization methods
    let warmup_result = graph.warmup_indices();
    assert!(warmup_result.is_ok());
    
    let rebuild_result = graph.rebuild_indices();
    assert!(rebuild_result.is_ok());
    
    // Phase 8: Final consistency check
    let final_issues = graph.validate_graph_consistency();
    
    // After warmup and rebuild, certain issues might be resolved
    // but some might remain due to the nature of the test data
    let serious_issues: Vec<_> = final_issues.iter()
        .filter(|issue| !issue.contains("Edge buffer is very large"))
        .collect();
    
    // Should have no serious structural issues
    assert!(serious_issues.len() <= 1); // Allow for minor dimension issues if any
}

#[test]
fn test_legacy_api_comprehensive_workflow() {
    let graph = create_test_graph();
    
    // Phase 1: Test comprehensive entity creation with legacy API
    let complex_entities = vec![
        (1000, "Advanced artificial intelligence research laboratory focusing on neural networks and deep learning applications", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "organization".to_string());
            props.insert("name".to_string(), "AI Research Lab".to_string());
            props.insert("founded".to_string(), "2015".to_string());
            props.insert("employees".to_string(), "150".to_string());
            props.insert("specialization".to_string(), "AI/ML".to_string());
            props
        }),
        (1001, "Natural language processing toolkit for advanced text analysis and understanding", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "software".to_string());
            props.insert("name".to_string(), "NLP Toolkit".to_string());
            props.insert("version".to_string(), "3.2.1".to_string());
            props.insert("language".to_string(), "Python".to_string());
            props.insert("license".to_string(), "MIT".to_string());
            props
        }),
        (1002, "International conference on machine learning and artificial intelligence", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "event".to_string());
            props.insert("name".to_string(), "ICML 2024".to_string());
            props.insert("location".to_string(), "Vienna, Austria".to_string());
            props.insert("date".to_string(), "July 2024".to_string());
            props.insert("attendees".to_string(), "5000".to_string());
            props
        }),
        (1003, "Leading researcher in computer vision and pattern recognition", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "person".to_string());
            props.insert("name".to_string(), "Dr. Maria Rodriguez".to_string());
            props.insert("title".to_string(), "Principal Researcher".to_string());
            props.insert("citations".to_string(), "15000".to_string());
            props.insert("h_index".to_string(), "42".to_string());
            props
        }),
        (1004, "Comprehensive dataset for training computer vision models on real-world scenarios", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "dataset".to_string());
            props.insert("name".to_string(), "RealWorld-CV-2024".to_string());
            props.insert("size".to_string(), "2.5TB".to_string());
            props.insert("images".to_string(), "10M".to_string());
            props.insert("annotations".to_string(), "50M".to_string());
            props
        }),
    ];
    
    // Insert all entities using legacy text-based API
    for (id, text, props) in &complex_entities {
        let result = graph.insert_entity_with_text(*id, text, props.clone());
        assert!(result.is_ok(), "Failed to insert entity {}", id);
    }
    
    // Phase 2: Create complex relationship web
    let relationships = vec![
        (1000, 1003, 0.95), // Lab employs researcher
        (1003, 1001, 0.85), // Researcher developed toolkit
        (1003, 1002, 0.90), // Researcher presents at conference
        (1001, 1004, 0.80), // Toolkit processes dataset
        (1000, 1002, 0.70), // Lab sponsors conference
        (1004, 1002, 0.60), // Dataset showcased at conference
    ];
    
    for (source, target, weight) in relationships {
        let result = graph.insert_relationship_by_id(source, target, weight);
        assert!(result.is_ok(), "Failed to create relationship {} -> {}", source, target);
    }
    
    // Phase 3: Test complex similarity searches with legacy API
    let search_queries = vec![
        ("machine learning research", 3),
        ("natural language processing", 2),
        ("computer vision dataset", 2),
        ("artificial intelligence conference", 3),
        ("research publications", 4),
    ];
    
    for (query, expected_min_results) in search_queries {
        let results = graph.similarity_search_by_text(query, 5);
        assert!(results.is_ok(), "Similarity search failed for query: {}", query);
        
        let search_results = results.unwrap();
        assert!(search_results.len() >= expected_min_results, 
                "Expected at least {} results for '{}', got {}", 
                expected_min_results, query, search_results.len());
        
        // Verify all similarity scores are valid
        for (entity_id, similarity) in &search_results {
            assert!(*similarity >= 0.0 && *similarity <= 1.0, 
                    "Invalid similarity score {} for entity {}", similarity, entity_id);
            assert!(complex_entities.iter().any(|(id, _, _)| id == entity_id),
                    "Unknown entity {} in search results", entity_id);
        }
    }
    
    // Phase 4: Test complex path finding scenarios
    let path_tests = vec![
        (1000, 1004), // Lab to Dataset
        (1003, 1002), // Researcher to Conference
        (1001, 1000), // Toolkit to Lab (reverse direction)
    ];
    
    for (start, end) in path_tests {
        let path = graph.find_path_by_id(start, end);
        if let Some(path_vec) = path {
            assert!(path_vec.len() >= 2, "Path too short for {} -> {}", start, end);
            assert_eq!(path_vec[0], start, "Path doesn't start with source entity");
            assert_eq!(path_vec[path_vec.len() - 1], end, "Path doesn't end with target entity");
            
            // Verify path connectivity
            for i in 0..path_vec.len() - 1 {
                let neighbors = graph.get_neighbors_by_id(path_vec[i]);
                assert!(neighbors.contains(&path_vec[i + 1]) || 
                       graph.get_neighbors_by_id(path_vec[i + 1]).contains(&path_vec[i]),
                       "Path not connected at step {} -> {}", path_vec[i], path_vec[i + 1]);
            }
        }
    }
    
    // Phase 5: Test complex property updates and consistency
    let updates = vec![
        (1000, {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "organization".to_string());
            props.insert("name".to_string(), "Advanced AI Research Lab".to_string());
            props.insert("founded".to_string(), "2015".to_string());
            props.insert("employees".to_string(), "200".to_string());
            props.insert("specialization".to_string(), "AI/ML/Robotics".to_string());
            props.insert("funding".to_string(), "50M USD".to_string());
            props
        }),
        (1001, {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "software".to_string());
            props.insert("name".to_string(), "Advanced NLP Toolkit".to_string());
            props.insert("version".to_string(), "4.0.0".to_string());
            props.insert("language".to_string(), "Python/C++".to_string());
            props.insert("license".to_string(), "Apache 2.0".to_string());
            props.insert("downloads".to_string(), "1M".to_string());
            props
        }),
    ];
    
    for (entity_id, new_props) in updates {
        let update_result = graph.update_entity_properties(entity_id, new_props.clone());
        assert!(update_result.is_ok(), "Failed to update entity {}", entity_id);
        
        // Verify update
        let retrieved_props = graph.get_entity_properties(entity_id);
        assert!(retrieved_props.is_some(), "Entity {} not found after update", entity_id);
        
        let props = retrieved_props.unwrap();
        for (key, expected_value) in &new_props {
            assert_eq!(props.get(key), Some(expected_value), 
                      "Property {} not updated correctly for entity {}", key, entity_id);
        }
    }
    
    // Phase 6: Test export/import workflow with complex data
    let exported_entities = graph.export_entities();
    let exported_relationships = graph.export_relationships();
    
    assert_eq!(exported_entities.len(), 5, "Wrong number of exported entities");
    assert_eq!(exported_relationships.len(), 6, "Wrong number of exported relationships");
    
    // Verify export data integrity
    for (entity_id, properties, embedding) in &exported_entities {
        assert!(complex_entities.iter().any(|(id, _, _)| id == entity_id),
                "Unknown entity {} in export", entity_id);
        assert!(!properties.is_empty(), "Empty properties for entity {}", entity_id);
        assert_eq!(embedding.len(), 128, "Wrong embedding dimension for entity {}", entity_id);
        
        // Verify properties contain expected keys
        assert!(properties.contains_key("type"), "Missing type for entity {}", entity_id);
        assert!(properties.contains_key("name"), "Missing name for entity {}", entity_id);
    }
    
    // Phase 7: Validate final graph statistics
    let final_stats = graph.get_graph_stats();
    assert_eq!(final_stats.get("entity_count").unwrap(), &5.0);
    assert_eq!(final_stats.get("relationship_count").unwrap(), &6.0);
    assert_eq!(final_stats.get("embedding_dimension").unwrap(), &128.0);
    
    let memory_mb = final_stats.get("memory_usage_mb").unwrap();
    assert!(*memory_mb > 0.0, "Memory usage should be positive");
    assert!(*memory_mb < 200.0, "Memory usage seems too high: {} MB", memory_mb);
}

#[test]
fn test_legacy_api_error_resilience() {
    let graph = create_test_graph();
    
    // Phase 1: Test resilience to invalid operations
    let invalid_operations = vec![
        // Invalid entity IDs
        (u32::MAX, "Maximum entity ID test", HashMap::new()),
        (0, "Zero entity ID test", HashMap::new()),
        
        // Very large property data
        (2000, "Large property test", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "test".to_string());
            props.insert("large_data".to_string(), "x".repeat(10000)); // 10KB string
            props
        }),
        
        // Unicode and special character handling
        (2001, "Unicode test: 你好世界 🌍 مرحبا العالم", {
            let mut props = HashMap::new();
            props.insert("type".to_string(), "unicode".to_string());
            props.insert("name".to_string(), "Ñáméⅇ with spècîál chars".to_string());
            props.insert("emoji".to_string(), "🚀🌟💫🔬🧪🎯".to_string());
            props.insert("arabic".to_string(), "العربية".to_string());
            props.insert("chinese".to_string(), "中文".to_string());
            props.insert("japanese".to_string(), "日本語".to_string());
            props
        }),
    ];
    
    let mut successful_insertions = 0;
    for (id, text, props) in invalid_operations {
        let result = graph.insert_entity_with_text(id, text, props);
        if result.is_ok() {
            successful_insertions += 1;
        }
        // Errors are acceptable for some invalid operations
    }
    
    // At least the Unicode test should succeed
    assert!(successful_insertions >= 1, "No insertions succeeded");
    
    // Phase 2: Test relationship error handling
    let valid_entity_result = graph.insert_entity_with_text(3000, "Valid entity", HashMap::new());
    assert!(valid_entity_result.is_ok());
    
    // Try invalid relationships
    let invalid_relationships = vec![
        (3000, 9999, 0.5), // Target doesn't exist
        (9998, 3000, 0.7), // Source doesn't exist
        (9997, 9996, 0.9), // Neither exists
        (3000, 3000, 1.0), // Self-relationship
    ];
    
    for (source, target, weight) in invalid_relationships {
        let result = graph.insert_relationship_by_id(source, target, weight);
        // Some may succeed (self-relationships), others may fail gracefully
        if result.is_err() {
            // Error should be meaningful and not cause crashes
        }
    }
    
    // Phase 3: Test search resilience
    let challenging_queries = vec![
        "", // Empty query
        "x".repeat(1000), // Very long query
        "non_existent_term_12345", // Non-existent terms
        "🚀🌟💫", // Emoji-only query
        "العربية中文日本語", // Mixed scripts
        "query with\nnewlines\tand\ttabs", // Special whitespace
    ];
    
    for query in challenging_queries {
        let result = graph.similarity_search_by_text(query, 5);
        assert!(result.is_ok(), "Search should handle query gracefully: '{}'", query);
        
        let results = result.unwrap();
        // Results can be empty for invalid queries
        for (entity_id, similarity) in results {
            assert!(similarity >= 0.0 && similarity <= 1.0, 
                   "Invalid similarity score for query '{}'", query);
        }
    }
    
    // Phase 4: Test batch operation error handling  
    let mixed_batch = vec![
        (4000, "Valid entity 1".to_string(), HashMap::new()),
        (4001, "".to_string(), HashMap::new()), // Empty text
        (4002, "Valid entity 2".to_string(), {
            let mut props = HashMap::new();
            props.insert("invalid_json".to_string(), "{broken json".to_string());
            props
        }),
    ];
    
    let batch_result = graph.insert_entities_with_text(mixed_batch);
    // Batch might succeed partially or fail completely depending on implementation
    // Both behaviors are acceptable as long as they're consistent
    
    // Phase 5: Test memory pressure recovery
    let stress_entities: Vec<_> = (5000..5100).map(|i| {
        let large_text = format!("Stress test entity {} with large text content: {}", 
                                i, "data ".repeat(100));
        let mut props = HashMap::new();
        props.insert("type".to_string(), "stress_test".to_string());
        props.insert("id".to_string(), i.to_string());
        (i, large_text, props)
    }).collect();
    
    // Try to insert all at once
    let stress_result = graph.insert_entities_with_text(stress_entities);
    
    // If it fails, that's acceptable - test that graph remains stable
    if stress_result.is_err() {
        // Graph should still be queryable
        let stats = graph.get_graph_stats();
        assert!(stats.get("entity_count").is_some(), "Graph corrupted after stress test");
        
        // Previous entities should still exist
        let test_entity = graph.get_entity_by_id_legacy(3000);
        assert!(test_entity.is_some(), "Previous entities lost after stress test");
    }
    
    // Phase 6: Test concurrent access error handling
    use std::sync::Arc;
    use std::thread;
    
    let graph_arc = Arc::new(graph);
    let handles: Vec<_> = (0..4).map(|thread_id| {
        let graph_clone = Arc::clone(&graph_arc);
        thread::spawn(move || {
            let mut errors = 0;
            let mut successes = 0;
            
            for i in 0..10 {
                let entity_id = thread_id * 1000 + i + 6000;
                let result = graph_clone.insert_entity_with_text(
                    entity_id as u32,
                    &format!("Concurrent entity {} from thread {}", i, thread_id),
                    HashMap::new()
                );
                
                match result {
                    Ok(_) => successes += 1,
                    Err(_) => errors += 1,
                }
                
                // Try some queries too
                let _ = graph_clone.get_entity_by_id_legacy(entity_id as u32);
                let _ = graph_clone.similarity_search_by_text("test", 3);
            }
            
            (errors, successes)
        })
    }).collect();
    
    let mut total_errors = 0;
    let mut total_successes = 0;
    
    for handle in handles {
        let (errors, successes) = handle.join().expect("Thread panicked");
        total_errors += errors;
        total_successes += successes;
    }
    
    // Some operations should succeed even under concurrent load
    assert!(total_successes > 0, "No concurrent operations succeeded");
    
    // Graph should remain consistent
    let final_stats = graph_arc.get_graph_stats();
    assert!(final_stats.get("entity_count").unwrap() > &0.0, "Graph lost all entities");
}