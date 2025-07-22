//! Brain graph core integration tests
//! Tests complete workflow including creating graph, adding entities/relationships, and querying
//! Tests all components working together correctly through public APIs

use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainMemoryUsage};
use llmkg::core::types::{EntityKey, EntityData, Relationship, ContextEntity, QueryResult};
use llmkg::error::Result;
use std::collections::HashMap;
use tokio;

/// Helper function to create test entity data with custom properties
fn create_entity_data_with_properties(id: u32, embedding_len: usize, properties: HashMap<String, String>) -> EntityData {
    let embedding = (0..embedding_len).map(|i| (i as f32 * 0.01) % 1.0).collect();
    
    EntityData::new(
        id as u16,
        serde_json::to_string(&properties).unwrap_or_default(),
        embedding
    )
}

#[tokio::test]
async fn test_complete_graph_workflow() -> Result<()> {
    // Test 1: Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Verify initial state
    let initial_stats = brain_graph.get_entity_statistics().await;
    assert_eq!(initial_stats.total_entities, 0);
    
    // Test 2: Create entities representing a simple knowledge network
    let mut person_props = HashMap::new();
    person_props.insert("type".to_string(), "person".to_string());
    person_props.insert("name".to_string(), "Alice".to_string());
    person_props.insert("age".to_string(), "30".to_string());
    
    let mut city_props = HashMap::new();
    city_props.insert("type".to_string(), "city".to_string());
    city_props.insert("name".to_string(), "New York".to_string());
    city_props.insert("population".to_string(), "8000000".to_string());
    
    let mut company_props = HashMap::new();
    company_props.insert("type".to_string(), "company".to_string());
    company_props.insert("name".to_string(), "Tech Corp".to_string());
    company_props.insert("industry".to_string(), "Technology".to_string());
    
    let person_data = create_entity_data_with_properties(1, 96, person_props);
    let city_data = create_entity_data_with_properties(2, 96, city_props);
    let company_data = create_entity_data_with_properties(3, 96, company_props);
    
    let person_key = brain_graph.insert_brain_entity(1, person_data).await?;
    let city_key = brain_graph.insert_brain_entity(2, city_data).await?;
    let company_key = brain_graph.insert_brain_entity(3, company_data).await?;
    
    // Test 3: Add relationships to form a knowledge network
    let lives_in_relationship = Relationship {
        from: person_key,
        to: city_key,
        rel_type: 1, // "lives_in"
        weight: 0.9,
    };
    
    let works_for_relationship = Relationship {
        from: person_key,
        to: company_key,
        rel_type: 2, // "works_for"
        weight: 0.8,
    };
    
    let located_in_relationship = Relationship {
        from: company_key,
        to: city_key,
        rel_type: 3, // "located_in"
        weight: 0.7,
    };
    
    brain_graph.insert_brain_relationship(lives_in_relationship).await?;
    brain_graph.insert_brain_relationship(works_for_relationship).await?;
    brain_graph.insert_brain_relationship(located_in_relationship).await?;
    
    // Test 4: Verify the network structure
    let person_neighbors = brain_graph.get_neighbors_with_weights(person_key).await;
    assert_eq!(person_neighbors.len(), 2);
    
    let city_parents = brain_graph.get_parent_entities(city_key).await;
    assert_eq!(city_parents.len(), 2); // Person and Company both connect to City
    
    // Test 5: Query the network
    let person_embedding = (0..96).map(|i| (i as f32 * 0.01) % 1.0).collect::<Vec<f32>>();
    let query_result = brain_graph.neural_query(&person_embedding, 3).await?;
    
    assert!(!query_result.is_empty());
    assert_eq!(query_result.entity_count(), 3);
    
    // Test 6: Check activation propagation
    brain_graph.set_entity_activation(person_key, 1.0).await;
    
    // Allow activation to propagate
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    let city_activation = brain_graph.get_entity_activation(city_key).await;
    let company_activation = brain_graph.get_entity_activation(company_key).await;
    
    // Connected entities should have some activation
    assert!(city_activation > 0.0);
    assert!(company_activation > 0.0);
    
    // Test 7: Memory usage tracking
    let memory_usage = brain_graph.get_memory_usage().await;
    assert!(memory_usage.total_entities > 0);
    assert!(memory_usage.total_relationships > 0);
    assert!(memory_usage.activation_memory > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_complex_graph_operations() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 1: Create a hierarchical knowledge structure
    let mut entities = Vec::new();
    
    // Root concept
    let mut root_props = HashMap::new();
    root_props.insert("level".to_string(), "0".to_string());
    root_props.insert("name".to_string(), "Root".to_string());
    let root_data = create_entity_data_with_properties(1, 96, root_props);
    let root_key = brain_graph.insert_brain_entity(1, root_data).await?;
    entities.push(root_key);
    
    // Level 1 concepts
    for i in 2..6 {
        let mut level1_props = HashMap::new();
        level1_props.insert("level".to_string(), "1".to_string());
        level1_props.insert("name".to_string(), format!("Level1_{}", i-1));
        let level1_data = create_entity_data_with_properties(i, 96, level1_props);
        let level1_key = brain_graph.insert_brain_entity(i, level1_data).await?;
        entities.push(level1_key);
        
        // Connect to root
        let relationship = Relationship {
            from: root_key,
            to: level1_key,
            rel_type: 1,
            weight: 0.8,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Level 2 concepts
    for i in 6..11 {
        let mut level2_props = HashMap::new();
        level2_props.insert("level".to_string(), "2".to_string());
        level2_props.insert("name".to_string(), format!("Level2_{}", i-5));
        let level2_data = create_entity_data_with_properties(i, 96, level2_props);
        let level2_key = brain_graph.insert_brain_entity(i, level2_data).await?;
        entities.push(level2_key);
        
        // Connect to a level 1 entity
        let parent_index = ((i - 6) % 4) + 1; // Cycle through level 1 entities
        let parent_key = entities[parent_index];
        let relationship = Relationship {
            from: parent_key,
            to: level2_key,
            rel_type: 2,
            weight: 0.7,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Test 2: Perform hierarchical queries
    let root_embedding = (0..96).map(|i| (i as f32 * 0.01) % 1.0).collect::<Vec<f32>>();
    let hierarchical_query = brain_graph.neural_query(&root_embedding, 5).await?;
    
    assert_eq!(hierarchical_query.entity_count(), 5);
    
    // Test 3: Check path finding between levels
    let level2_key = entities[6]; // First level 2 entity
    let alternative_paths = brain_graph.find_alternative_paths(root_key, level2_key, 3).await;
    assert!(!alternative_paths.is_empty());
    
    // Should find path: root -> level1 -> level2
    let direct_path = &alternative_paths[0];
    assert_eq!(direct_path.len(), 3); // root, level1, level2
    assert_eq!(direct_path[0], root_key);
    assert_eq!(direct_path[2], level2_key);
    
    // Test 4: Activation cascading through hierarchy
    brain_graph.set_entity_activation(root_key, 1.0).await;
    
    // Propagate activation
    let propagation_result = brain_graph.propagate_activation_from_entity(root_key, 0.1).await?;
    
    // Check that activation propagated down the hierarchy
    let level1_activations: Vec<f32> = entities[1..5]
        .iter()
        .map(|&key| brain_graph.get_entity_activation(key))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .await;
    
    // All level 1 entities should have some activation
    for activation in level1_activations {
        assert!(activation > 0.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_graph_analytics_workflow() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 1: Build a graph with different connection patterns
    let mut hub_props = HashMap::new();
    hub_props.insert("type".to_string(), "hub".to_string());
    hub_props.insert("name".to_string(), "Central Hub".to_string());
    let hub_data = create_entity_data_with_properties(1, 96, hub_props);
    let hub_key = brain_graph.insert_brain_entity(1, hub_data).await?;
    
    let mut connected_entities = Vec::new();
    
    // Create entities connected to hub (star pattern)
    for i in 2..8 {
        let mut props = HashMap::new();
        props.insert("type".to_string(), "spoke".to_string());
        props.insert("name".to_string(), format!("Spoke_{}", i-1));
        let data = create_entity_data_with_properties(i, 96, props);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        connected_entities.push(key);
        
        // Connect to hub
        let relationship = Relationship {
            from: hub_key,
            to: key,
            rel_type: 1,
            weight: 0.8,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Create chain pattern
    let mut chain_entities = Vec::new();
    for i in 8..13 {
        let mut props = HashMap::new();
        props.insert("type".to_string(), "chain".to_string());
        props.insert("name".to_string(), format!("Chain_{}", i-7));
        let data = create_entity_data_with_properties(i, 96, props);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        chain_entities.push(key);
        
        // Connect in chain
        if i > 8 {
            let prev_key = chain_entities[chain_entities.len() - 2];
            let relationship = Relationship {
                from: prev_key,
                to: key,
                rel_type: 2,
                weight: 0.9,
            };
            brain_graph.insert_brain_relationship(relationship).await?;
        }
    }
    
    // Test 2: Analyze graph structure
    let graph_analysis = brain_graph.analyze_graph_patterns().await;
    
    assert!(graph_analysis.total_entities > 10);
    assert!(graph_analysis.total_relationships > 10);
    
    // Hub should have high connectivity
    let hub_degree = brain_graph.get_entity_degree(hub_key).await;
    assert!(hub_degree >= 6); // Connected to 6 spokes
    
    // Test 3: Find central entities
    let central_entities = brain_graph.find_central_entities(3).await;
    assert!(!central_entities.is_empty());
    
    // Hub should be among the most central
    let hub_is_central = central_entities.iter().any(|&(key, _)| key == hub_key);
    assert!(hub_is_central);
    
    // Test 4: Community detection
    let communities = brain_graph.detect_communities().await?;
    assert!(!communities.is_empty());
    
    // Should find at least 2 communities (hub cluster and chain cluster)
    assert!(communities.len() >= 2);
    
    // Test 5: Performance metrics
    let performance_metrics = brain_graph.get_performance_metrics().await;
    assert!(performance_metrics.average_query_time > 0.0);
    assert!(performance_metrics.cache_hit_rate >= 0.0);
    assert!(performance_metrics.activation_efficiency > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_dynamic_graph_evolution() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 1: Start with simple graph
    let mut base_props = HashMap::new();
    base_props.insert("generation".to_string(), "0".to_string());
    base_props.insert("type".to_string(), "base".to_string());
    let base_data = create_entity_data_with_properties(1, 96, base_props);
    let base_key = brain_graph.insert_brain_entity(1, base_data).await?;
    
    // Test 2: Evolve graph through iterations
    let mut current_generation = vec![base_key];
    let mut all_entities = vec![base_key];
    
    for generation in 1..4 {
        let mut next_generation = Vec::new();
        
        for (i, &parent_key) in current_generation.iter().enumerate() {
            // Create 2 children per parent
            for child_idx in 0..2 {
                let entity_id = generation * 10 + i * 2 + child_idx + 1;
                let mut child_props = HashMap::new();
                child_props.insert("generation".to_string(), generation.to_string());
                child_props.insert("parent_index".to_string(), i.to_string());
                child_props.insert("child_index".to_string(), child_idx.to_string());
                
                let child_data = create_entity_data_with_properties(entity_id, 96, child_props);
                let child_key = brain_graph.insert_brain_entity(entity_id, child_data).await?;
                
                next_generation.push(child_key);
                all_entities.push(child_key);
                
                // Connect child to parent
                let relationship = Relationship {
                    from: parent_key,
                    to: child_key,
                    rel_type: 1,
                    weight: 0.8,
                };
                brain_graph.insert_brain_relationship(relationship).await?;
            }
        }
        
        current_generation = next_generation;
    }
    
    // Test 3: Analyze evolved structure
    let final_stats = brain_graph.get_entity_statistics().await;
    assert!(final_stats.total_entities > 10); // Should have created a tree
    
    // Test 4: Query across generations
    let base_embedding = (0..96).map(|i| (i as f32 * 0.01) % 1.0).collect::<Vec<f32>>();
    let multi_gen_query = brain_graph.neural_query(&base_embedding, 8).await?;
    
    assert!(multi_gen_query.entity_count() > 5);
    
    // Test 5: Check path lengths between generations
    let leaf_entity = current_generation[0]; // First leaf
    let paths_to_root = brain_graph.find_alternative_paths(base_key, leaf_entity, 1).await;
    assert!(!paths_to_root.is_empty());
    assert_eq!(paths_to_root[0].len(), 4); // root -> gen1 -> gen2 -> gen3
    
    // Test 6: Dynamic modification - add cross-generation links
    let gen1_entity = all_entities[1]; // First gen1 entity
    let gen3_entity = current_generation[0]; // First gen3 entity
    
    let cross_gen_relationship = Relationship {
        from: gen1_entity,
        to: gen3_entity,
        rel_type: 3, // Cross-generation link
        weight: 0.5,
    };
    brain_graph.insert_brain_relationship(cross_gen_relationship).await?;
    
    // Verify new path exists
    let updated_paths = brain_graph.find_alternative_paths(base_key, gen3_entity, 3).await;
    assert!(updated_paths.len() >= 2); // Original path + new shorter path
    
    // Find the shorter path
    let shortest_path = updated_paths.iter().min_by_key(|path| path.len()).unwrap();
    assert!(shortest_path.len() < 4); // Should be shorter due to cross-link
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_graph_operations() -> Result<()> {
    // Create brain graph
    let brain_graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new_for_test()?);
    
    // Test 1: Concurrent entity insertions
    let mut insert_handles = Vec::new();
    
    for i in 1..21 {
        let graph_clone = brain_graph.clone();
        let handle = tokio::spawn(async move {
            let mut props = HashMap::new();
            props.insert("thread_id".to_string(), format!("thread_{}", i));
            props.insert("entity_id".to_string(), i.to_string());
            let data = create_entity_data_with_properties(i, 96, props);
            graph_clone.insert_brain_entity(i, data).await
        });
        insert_handles.push(handle);
    }
    
    // Wait for all insertions to complete
    let mut inserted_keys = Vec::new();
    for handle in insert_handles {
        let key = handle.await.unwrap()?;
        inserted_keys.push(key);
    }
    
    assert_eq!(inserted_keys.len(), 20);
    
    // Test 2: Concurrent relationship creation
    let mut relationship_handles = Vec::new();
    
    for i in 0..10 {
        let graph_clone = brain_graph.clone();
        let from_key = inserted_keys[i];
        let to_key = inserted_keys[i + 10];
        
        let handle = tokio::spawn(async move {
            let relationship = Relationship {
                from: from_key,
                to: to_key,
                rel_type: 1,
                weight: 0.7,
            };
            graph_clone.insert_brain_relationship(relationship).await
        });
        relationship_handles.push(handle);
    }
    
    // Wait for all relationships to be created
    for handle in relationship_handles {
        handle.await.unwrap()?;
    }
    
    // Test 3: Concurrent queries
    let mut query_handles = Vec::new();
    
    for i in 0..5 {
        let graph_clone = brain_graph.clone();
        let query_embedding = (0..96).map(|j| ((i * 10 + j) as f32 * 0.01) % 1.0).collect();
        
        let handle = tokio::spawn(async move {
            graph_clone.neural_query(&query_embedding, 5).await
        });
        query_handles.push(handle);
    }
    
    // Wait for all queries to complete
    let mut query_results = Vec::new();
    for handle in query_handles {
        let result = handle.await.unwrap()?;
        query_results.push(result);
    }
    
    assert_eq!(query_results.len(), 5);
    for result in query_results {
        assert!(!result.is_empty());
    }
    
    // Test 4: Final consistency check
    let final_stats = brain_graph.get_entity_statistics().await;
    assert_eq!(final_stats.total_entities, 20);
    
    let relationship_stats = brain_graph.get_relationship_statistics().await;
    assert_eq!(relationship_stats.total_relationships, 10);
    
    Ok(())
}

#[tokio::test]
async fn test_graph_persistence_workflow() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 1: Build initial graph state
    let mut entity_keys = Vec::new();
    
    for i in 1..6 {
        let mut props = HashMap::new();
        props.insert("persistent_id".to_string(), format!("persistent_{}", i));
        props.insert("value".to_string(), (i * 10).to_string());
        let data = create_entity_data_with_properties(i, 96, props);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push(key);
    }
    
    // Add some relationships
    for i in 0..4 {
        let relationship = Relationship {
            from: entity_keys[i],
            to: entity_keys[i + 1],
            rel_type: 1,
            weight: 0.8,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Set specific activations
    for (i, &key) in entity_keys.iter().enumerate() {
        let activation = (i + 1) as f32 * 0.2;
        brain_graph.set_entity_activation(key, activation).await;
    }
    
    // Test 2: Capture graph state
    let initial_stats = brain_graph.get_entity_statistics().await;
    let initial_memory = brain_graph.get_memory_usage().await;
    
    // Test 3: Perform operations that modify state
    brain_graph.set_entity_activation(entity_keys[0], 1.0).await;
    let propagation_result = brain_graph.propagate_activation_from_entity(entity_keys[0], 0.1).await?;
    
    // Test 4: Verify state changes
    let updated_stats = brain_graph.get_entity_statistics().await;
    assert!(updated_stats.average_activation != initial_stats.average_activation);
    
    // Test 5: Reset to baseline
    brain_graph.reset_all_activations().await;
    
    let reset_stats = brain_graph.get_entity_statistics().await;
    assert_eq!(reset_stats.average_activation, 0.0);
    
    // Test 6: Rebuild state
    for (i, &key) in entity_keys.iter().enumerate() {
        let activation = (i + 1) as f32 * 0.2;
        brain_graph.set_entity_activation(key, activation).await;
    }
    
    let restored_stats = brain_graph.get_entity_statistics().await;
    assert!((restored_stats.average_activation - initial_stats.average_activation).abs() < 0.01);
    
    Ok(())
}