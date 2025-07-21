//! Brain relationship manager integration tests
//! Tests complete relationship lifecycle including creation, strengthening, weakening, removal
//! Tests all relationship management methods working together correctly through public APIs

use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, RelationshipStatistics, RelationshipPattern};
use llmkg::core::types::{EntityKey, EntityData, Relationship};
use llmkg::error::Result;
use std::collections::HashMap;
use tokio;

/// Helper function to create test entity data
fn create_test_entity_data(id: u32, embedding_len: usize, entity_type: &str) -> EntityData {
    let embedding = (0..embedding_len).map(|i| ((id * 10 + i as u32) as f32) * 0.001).collect();
    let mut properties = HashMap::new();
    properties.insert("id".to_string(), id.to_string());
    properties.insert("type".to_string(), entity_type.to_string());
    properties.insert("name".to_string(), format!("{}_{}", entity_type, id));
    
    EntityData {
        type_id: id,
        embedding,
        properties: serde_json::to_string(&properties).unwrap_or_default(),
    }
}

#[tokio::test]
async fn test_relationship_lifecycle_complete() -> Result<()> {
    // Test 1: Create brain graph and entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let entity1_data = create_test_entity_data(1, 96, "person");
    let entity2_data = create_test_entity_data(2, 96, "organization");
    let entity3_data = create_test_entity_data(3, 96, "location");
    
    let person_key = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let org_key = brain_graph.insert_brain_entity(2, entity2_data).await?;
    let location_key = brain_graph.insert_brain_entity(3, entity3_data).await?;
    
    // Test 2: Create initial relationships
    let works_for_relationship = Relationship {
        from: person_key,
        to: org_key,
        rel_type: 1, // "works_for"
        weight: 0.8,
    };
    
    let located_in_relationship = Relationship {
        from: org_key,
        to: location_key,
        rel_type: 2, // "located_in"
        weight: 0.9,
    };
    
    let lives_in_relationship = Relationship {
        from: person_key,
        to: location_key,
        rel_type: 3, // "lives_in"
        weight: 0.7,
    };
    
    brain_graph.insert_brain_relationship(works_for_relationship.clone()).await?;
    brain_graph.insert_brain_relationship(located_in_relationship.clone()).await?;
    brain_graph.insert_brain_relationship(lives_in_relationship.clone()).await?;
    
    // Test 3: Verify relationships exist
    assert!(brain_graph.has_relationship(person_key, org_key).await);
    assert!(brain_graph.has_relationship(org_key, location_key).await);
    assert!(brain_graph.has_relationship(person_key, location_key).await);
    
    // Test 4: Check relationship weights
    assert_eq!(brain_graph.get_relationship_weight(person_key, org_key), Some(0.8));
    assert_eq!(brain_graph.get_relationship_weight(org_key, location_key), Some(0.9));
    assert_eq!(brain_graph.get_relationship_weight(person_key, location_key), Some(0.7));
    
    // Test 5: Check synaptic weights
    let synaptic_weight_1 = brain_graph.get_synaptic_weight(person_key, org_key).await;
    let synaptic_weight_2 = brain_graph.get_synaptic_weight(org_key, location_key).await;
    let synaptic_weight_3 = brain_graph.get_synaptic_weight(person_key, location_key).await;
    
    assert!(synaptic_weight_1 > 0.0);
    assert!(synaptic_weight_2 > 0.0);
    assert!(synaptic_weight_3 > 0.0);
    
    // Test 6: Strengthen relationships
    brain_graph.strengthen_relationship(person_key, org_key, 0.1).await?;
    brain_graph.strengthen_relationship(org_key, location_key, 0.05).await?;
    
    let strengthened_weight_1 = brain_graph.get_relationship_weight(person_key, org_key);
    let strengthened_weight_2 = brain_graph.get_relationship_weight(org_key, location_key);
    
    assert!(strengthened_weight_1.unwrap_or(0.0) > 0.8);
    assert!(strengthened_weight_2.unwrap_or(0.0) > 0.9);
    
    // Test 7: Weaken relationships
    brain_graph.weaken_relationship(person_key, location_key, 0.2).await?;
    
    let weakened_weight = brain_graph.get_relationship_weight(person_key, location_key);
    assert!(weakened_weight.unwrap_or(1.0) < 0.7);
    
    // Test 8: Update relationship weights directly
    brain_graph.update_relationship_weight(person_key, org_key, 0.95).await?;
    assert_eq!(brain_graph.get_relationship_weight(person_key, org_key), Some(0.95));
    
    let updated_synaptic_weight = brain_graph.get_synaptic_weight(person_key, org_key).await;
    assert!((updated_synaptic_weight - 0.95).abs() < 0.01);
    
    // Test 9: Remove relationships
    let removed = brain_graph.remove_relationship(person_key, location_key).await?;
    assert!(removed);
    
    assert!(!brain_graph.has_relationship(person_key, location_key).await);
    assert_eq!(brain_graph.get_relationship_weight(person_key, location_key), None);
    
    // Try to remove non-existent relationship
    let not_removed = brain_graph.remove_relationship(person_key, location_key).await?;
    assert!(!not_removed);
    
    Ok(())
}

#[tokio::test]
async fn test_neighbor_relationship_queries() -> Result<()> {
    // Test 1: Create brain graph with complex network
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create hub entity and connected entities
    let hub_data = create_test_entity_data(1, 96, "hub");
    let hub_key = brain_graph.insert_brain_entity(1, hub_data).await?;
    
    let mut connected_entities = Vec::new();
    for i in 2..8 {
        let entity_data = create_test_entity_data(i, 96, "node");
        let entity_key = brain_graph.insert_brain_entity(i, entity_data).await?;
        connected_entities.push(entity_key);
    }
    
    // Test 2: Create bidirectional relationships with different weights
    let mut expected_outgoing = Vec::new();
    let mut expected_incoming = Vec::new();
    
    // Outgoing relationships (hub -> nodes)
    for (i, &node_key) in connected_entities.iter().enumerate() {
        let weight = 0.5 + (i as f32) * 0.1;
        let relationship = Relationship {
            from: hub_key,
            to: node_key,
            rel_type: 1,
            weight,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
        expected_outgoing.push((node_key, weight));
    }
    
    // Incoming relationships (some nodes -> hub)
    for i in 0..3 {
        let weight = 0.3 + (i as f32) * 0.15;
        let relationship = Relationship {
            from: connected_entities[i],
            to: hub_key,
            rel_type: 2,
            weight,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
        expected_incoming.push((connected_entities[i], weight));
    }
    
    // Test 3: Get neighbors with weights
    let neighbors = brain_graph.get_neighbors_with_weights(hub_key).await;
    
    // Should include all connected entities (incoming and outgoing)
    assert!(neighbors.len() >= 6); // At least 6 unique connections
    
    // Should be sorted by synaptic weight (descending)
    for i in 1..neighbors.len() {
        assert!(neighbors[i - 1].1 >= neighbors[i].1);
    }
    
    // Test 4: Get child entities (outgoing connections)
    let children = brain_graph.get_child_entities(hub_key).await;
    assert_eq!(children.len(), 6); // All outgoing connections
    
    // Should be sorted by synaptic weight
    for i in 1..children.len() {
        assert!(children[i - 1].1 >= children[i].1);
    }
    
    // Verify all expected children are present
    let child_keys: Vec<EntityKey> = children.iter().map(|(key, _)| *key).collect();
    for &expected_key in &connected_entities {
        assert!(child_keys.contains(&expected_key));
    }
    
    // Test 5: Get parent entities (incoming connections)
    let parents = brain_graph.get_parent_entities(hub_key).await;
    assert_eq!(parents.len(), 3); // Only 3 incoming connections
    
    // Should be sorted by synaptic weight
    for i in 1..parents.len() {
        assert!(parents[i - 1].1 >= parents[i].1);
    }
    
    // Verify expected parents are present
    let parent_keys: Vec<EntityKey> = parents.iter().map(|(key, _)| *key).collect();
    for &expected_key in expected_incoming.iter().map(|(key, _)| *key) {
        assert!(parent_keys.contains(&expected_key));
    }
    
    // Test 6: Test neighbor queries for leaf nodes
    let leaf_node = connected_entities[0];
    let leaf_neighbors = brain_graph.get_neighbors_with_weights(leaf_node).await;
    
    // Leaf node should have 2 connections to hub (one in each direction)
    assert!(leaf_neighbors.len() >= 1);
    assert!(leaf_neighbors.iter().any(|(key, _)| *key == hub_key));
    
    Ok(())
}

#[tokio::test]
async fn test_alternative_path_finding() -> Result<()> {
    // Test 1: Create brain graph with multiple paths
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create a diamond-shaped network: A -> B,C -> D
    let entity_a_data = create_test_entity_data(1, 96, "start");
    let entity_b_data = create_test_entity_data(2, 96, "middle1");
    let entity_c_data = create_test_entity_data(3, 96, "middle2");
    let entity_d_data = create_test_entity_data(4, 96, "end");
    
    let key_a = brain_graph.insert_brain_entity(1, entity_a_data).await?;
    let key_b = brain_graph.insert_brain_entity(2, entity_b_data).await?;
    let key_c = brain_graph.insert_brain_entity(3, entity_c_data).await?;
    let key_d = brain_graph.insert_brain_entity(4, entity_d_data).await?;
    
    // Create diamond pattern relationships
    let relationships = vec![
        Relationship { from: key_a, to: key_b, rel_type: 1, weight: 0.8 }, // A -> B
        Relationship { from: key_a, to: key_c, rel_type: 1, weight: 0.7 }, // A -> C
        Relationship { from: key_b, to: key_d, rel_type: 1, weight: 0.9 }, // B -> D
        Relationship { from: key_c, to: key_d, rel_type: 1, weight: 0.6 }, // C -> D
    ];
    
    for relationship in relationships {
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Test 2: Find alternative paths from A to D
    let paths = brain_graph.find_alternative_paths(key_a, key_d, 5).await;
    
    // Should find 2 paths: A->B->D and A->C->D
    assert_eq!(paths.len(), 2);
    
    // All paths should start with A and end with D
    for path in &paths {
        assert_eq!(path[0], key_a);
        assert_eq!(path[path.len() - 1], key_d);
        assert_eq!(path.len(), 3); // Should be 3-node paths
    }
    
    // Should find both paths
    let path1 = vec![key_a, key_b, key_d];
    let path2 = vec![key_a, key_c, key_d];
    
    assert!(paths.contains(&path1) || paths.contains(&path2));
    assert!(paths.len() >= 2);
    
    // Test 3: Add more complex network
    let entity_e_data = create_test_entity_data(5, 96, "extra");
    let key_e = brain_graph.insert_brain_entity(5, entity_e_data).await?;
    
    // Add longer path: A -> E -> D
    brain_graph.insert_brain_relationship(Relationship {
        from: key_a, to: key_e, rel_type: 1, weight: 0.5
    }).await?;
    brain_graph.insert_brain_relationship(Relationship {
        from: key_e, to: key_d, rel_type: 1, weight: 0.4
    }).await?;
    
    // Test 4: Find paths with limit
    let limited_paths = brain_graph.find_alternative_paths(key_a, key_d, 2).await;
    assert!(limited_paths.len() <= 2);
    
    let all_paths = brain_graph.find_alternative_paths(key_a, key_d, 10).await;
    assert!(all_paths.len() >= 2); // Should find at least the original 2 paths
    
    // Test 5: Test path finding with no connection
    let isolated_data = create_test_entity_data(6, 96, "isolated");
    let isolated_key = brain_graph.insert_brain_entity(6, isolated_data).await?;
    
    let no_paths = brain_graph.find_alternative_paths(key_a, isolated_key, 5).await;
    assert_eq!(no_paths.len(), 0);
    
    // Test 6: Test direct connection
    brain_graph.insert_brain_relationship(Relationship {
        from: key_a, to: key_d, rel_type: 1, weight: 1.0
    }).await?;
    
    let with_direct = brain_graph.find_alternative_paths(key_a, key_d, 10).await;
    assert!(with_direct.len() >= 3); // Should now include direct path
    
    // Should include a direct path of length 2
    let has_direct_path = with_direct.iter().any(|path| path.len() == 2);
    assert!(has_direct_path);
    
    Ok(())
}

#[tokio::test]
async fn test_relationship_strength_dynamics() -> Result<()> {
    // Test 1: Create brain graph with relationships of varying strengths
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let entity1_data = create_test_entity_data(1, 96, "dynamic1");
    let entity2_data = create_test_entity_data(2, 96, "dynamic2");
    let entity3_data = create_test_entity_data(3, 96, "dynamic3");
    
    let key1 = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let key2 = brain_graph.insert_brain_entity(2, entity2_data).await?;
    let key3 = brain_graph.insert_brain_entity(3, entity3_data).await?;
    
    // Test 2: Create relationships with different initial strengths
    let weak_relationship = Relationship {
        from: key1, to: key2, rel_type: 1, weight: 0.3
    };
    let strong_relationship = Relationship {
        from: key2, to: key3, rel_type: 1, weight: 0.9
    };
    
    brain_graph.insert_brain_relationship(weak_relationship).await?;
    brain_graph.insert_brain_relationship(strong_relationship).await?;
    
    // Test 3: Simulate relationship strengthening through repeated activation
    for _ in 0..5 {
        brain_graph.strengthen_relationship(key1, key2, 0.05).await?;
    }
    
    let strengthened_weight = brain_graph.get_relationship_weight(key1, key2);
    assert!(strengthened_weight.unwrap_or(0.0) > 0.4); // Should be stronger
    
    // Test 4: Simulate relationship decay
    for _ in 0..3 {
        brain_graph.weaken_relationship(key2, key3, 0.1).await?;
    }
    
    let weakened_weight = brain_graph.get_relationship_weight(key2, key3);
    assert!(weakened_weight.unwrap_or(1.0) < 0.9); // Should be weaker
    
    // Test 5: Test relationship strength bounds
    // Try to strengthen beyond maximum
    for _ in 0..20 {
        brain_graph.strengthen_relationship(key1, key2, 0.1).await?;
    }
    
    let max_weight = brain_graph.get_relationship_weight(key1, key2);
    assert!(max_weight.unwrap_or(0.0) <= 1.0); // Should not exceed 1.0
    
    // Try to weaken beyond minimum
    for _ in 0..20 {
        brain_graph.weaken_relationship(key2, key3, 0.1).await?;
    }
    
    let min_weight = brain_graph.get_relationship_weight(key2, key3);
    if min_weight.is_some() {
        assert!(min_weight.unwrap() >= 0.0); // Should not go below 0.0
    }
    
    // Test 6: Test synaptic weight consistency with relationship weight
    let final_rel_weight = brain_graph.get_relationship_weight(key1, key2).unwrap_or(0.0);
    let final_syn_weight = brain_graph.get_synaptic_weight(key1, key2).await;
    
    // Should be approximately equal (allowing for small differences due to processing)
    assert!((final_rel_weight - final_syn_weight).abs() < 0.1);
    
    Ok(())
}

#[tokio::test]
async fn test_relationship_pattern_analysis() -> Result<()> {
    // Test 1: Create brain graph with different relationship patterns
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let mut entities = Vec::new();
    
    // Create entities for different patterns
    for i in 1..11 {
        let entity_data = create_test_entity_data(i, 96, "pattern_node");
        let key = brain_graph.insert_brain_entity(i, entity_data).await?;
        entities.push(key);
    }
    
    // Test 2: Create star pattern (one central node connected to many)
    let center = entities[0];
    for i in 1..6 {
        let relationship = Relationship {
            from: center,
            to: entities[i],
            rel_type: 1,
            weight: 0.8,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Test 3: Create chain pattern
    for i in 6..9 {
        let relationship = Relationship {
            from: entities[i],
            to: entities[i + 1],
            rel_type: 2,
            weight: 0.7,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Test 4: Analyze relationship patterns
    let rel_stats = brain_graph.get_relationship_statistics().await;
    
    assert_eq!(rel_stats.total_relationships, 8); // 5 star + 3 chain
    assert!(rel_stats.average_relationship_weight > 0.0);
    assert!(rel_stats.max_relationship_weight <= 1.0);
    assert!(rel_stats.min_relationship_weight >= 0.0);
    
    // Test 5: Analyze connectivity patterns
    let center_degree = brain_graph.get_entity_degree(center).await;
    assert_eq!(center_degree, 5); // Should have 5 outgoing connections
    
    let leaf_degree = brain_graph.get_entity_degree(entities[1]).await;
    assert_eq!(leaf_degree, 1); // Should have 1 incoming connection
    
    let chain_middle_degree = brain_graph.get_entity_degree(entities[7]).await;
    assert_eq!(chain_middle_degree, 2); // Should have 1 incoming + 1 outgoing
    
    // Test 6: Test relationship type analysis
    let type1_count = brain_graph.count_relationships_by_type(1).await;
    let type2_count = brain_graph.count_relationships_by_type(2).await;
    
    assert_eq!(type1_count, 5); // Star pattern relationships
    assert_eq!(type2_count, 3); // Chain pattern relationships
    
    // Test 7: Test weight distribution analysis
    let weight_distribution = brain_graph.analyze_weight_distribution().await;
    
    assert!(weight_distribution.mean > 0.0);
    assert!(weight_distribution.std_dev >= 0.0);
    assert!(weight_distribution.min <= weight_distribution.max);
    
    Ok(())
}

#[tokio::test]
async fn test_relationship_statistics_tracking() -> Result<()> {
    // Test 1: Create brain graph and verify initial statistics
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let initial_stats = brain_graph.get_relationship_statistics().await;
    assert_eq!(initial_stats.total_relationships, 0);
    assert_eq!(initial_stats.strong_relationships, 0);
    assert_eq!(initial_stats.weak_relationships, 0);
    assert_eq!(initial_stats.average_relationship_weight, 0.0);
    
    // Test 2: Add entities and relationships
    let entity1_data = create_test_entity_data(1, 96, "stats1");
    let entity2_data = create_test_entity_data(2, 96, "stats2");
    let entity3_data = create_test_entity_data(3, 96, "stats3");
    
    let key1 = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let key2 = brain_graph.insert_brain_entity(2, entity2_data).await?;
    let key3 = brain_graph.insert_brain_entity(3, entity3_data).await?;
    
    // Add relationships with different strengths
    let strong_rel = Relationship { from: key1, to: key2, rel_type: 1, weight: 0.9 };
    let medium_rel = Relationship { from: key2, to: key3, rel_type: 1, weight: 0.6 };
    let weak_rel = Relationship { from: key1, to: key3, rel_type: 1, weight: 0.3 };
    
    brain_graph.insert_brain_relationship(strong_rel).await?;
    brain_graph.insert_brain_relationship(medium_rel).await?;
    brain_graph.insert_brain_relationship(weak_rel).await?;
    
    // Test 3: Verify updated statistics
    let updated_stats = brain_graph.get_relationship_statistics().await;
    
    assert_eq!(updated_stats.total_relationships, 3);
    
    let expected_avg = (0.9 + 0.6 + 0.3) / 3.0;
    assert!((updated_stats.average_relationship_weight - expected_avg).abs() < 0.01);
    
    assert_eq!(updated_stats.max_relationship_weight, 0.9);
    assert_eq!(updated_stats.min_relationship_weight, 0.3);
    
    // Test 4: Count strong and weak relationships (assuming threshold = 0.7)
    let strong_count = updated_stats.strong_relationships; // weight >= 0.7
    let weak_count = updated_stats.weak_relationships; // weight < 0.7
    
    assert!(strong_count >= 1); // At least the 0.9 weight relationship
    assert!(weak_count >= 2); // The 0.6 and 0.3 weight relationships
    
    // Test 5: Modify relationship strengths and verify statistics update
    brain_graph.strengthen_relationship(key1, key3, 0.5).await?; // 0.3 -> 0.8
    
    let strengthened_stats = brain_graph.get_relationship_statistics().await;
    assert!(strengthened_stats.average_relationship_weight > updated_stats.average_relationship_weight);
    assert_eq!(strengthened_stats.max_relationship_weight, 0.9); // Should remain same
    assert!(strengthened_stats.min_relationship_weight > 0.3); // Should increase
    
    // Test 6: Remove relationship and verify statistics
    let removed = brain_graph.remove_relationship(key2, key3).await?;
    assert!(removed);
    
    let final_stats = brain_graph.get_relationship_statistics().await;
    assert_eq!(final_stats.total_relationships, 2); // One removed
    
    // Average should change due to removed medium-weight relationship
    let expected_final_avg = (0.9 + 0.8) / 2.0; // Assuming 0.3->0.8 change
    assert!((final_stats.average_relationship_weight - expected_final_avg).abs() < 0.1);
    
    Ok(())
}

#[tokio::test]
async fn test_batch_relationship_operations() -> Result<()> {
    // Test 1: Create brain graph and entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let mut entities = Vec::new();
    for i in 1..11 {
        let entity_data = create_test_entity_data(i, 96, "batch");
        let key = brain_graph.insert_brain_entity(i, entity_data).await?;
        entities.push(key);
    }
    
    // Test 2: Batch insert relationships
    let mut batch_relationships = Vec::new();
    for i in 0..5 {
        let relationship = Relationship {
            from: entities[i],
            to: entities[i + 5],
            rel_type: 1,
            weight: 0.5 + (i as f32) * 0.1,
        };
        batch_relationships.push(relationship);
    }
    
    brain_graph.batch_insert_relationships(&batch_relationships).await?;
    
    // Verify all relationships were created
    for relationship in &batch_relationships {
        assert!(brain_graph.has_relationship(relationship.from, relationship.to).await);
        assert_eq!(
            brain_graph.get_relationship_weight(relationship.from, relationship.to),
            Some(relationship.weight)
        );
    }
    
    // Test 3: Batch update relationship weights
    let weight_updates = entities.iter()
        .take(5)
        .zip(entities.iter().skip(5))
        .map(|(&from, &to)| (from, to, 0.8))
        .collect::<Vec<_>>();
    
    brain_graph.batch_update_relationship_weights(&weight_updates).await?;
    
    // Verify all weights were updated
    for (from, to, expected_weight) in &weight_updates {
        let actual_weight = brain_graph.get_relationship_weight(*from, *to);
        assert_eq!(actual_weight, Some(*expected_weight));
    }
    
    // Test 4: Batch strengthen relationships
    let strengthen_ops = entities.iter()
        .take(3)
        .zip(entities.iter().skip(5))
        .map(|(&from, &to)| (from, to, 0.1))
        .collect::<Vec<_>>();
    
    brain_graph.batch_strengthen_relationships(&strengthen_ops).await?;
    
    // Verify relationships were strengthened
    for (from, to, _) in &strengthen_ops {
        let weight = brain_graph.get_relationship_weight(*from, *to);
        assert!(weight.unwrap_or(0.0) > 0.8); // Should be stronger than 0.8
    }
    
    // Test 5: Batch weaken relationships
    let weaken_ops = entities.iter()
        .skip(3)
        .take(2)
        .zip(entities.iter().skip(8))
        .map(|(&from, &to)| (from, to, 0.2))
        .collect::<Vec<_>>();
    
    brain_graph.batch_weaken_relationships(&weaken_ops).await?;
    
    // Verify relationships were weakened
    for (from, to, _) in &weaken_ops {
        let weight = brain_graph.get_relationship_weight(*from, *to);
        assert!(weight.unwrap_or(1.0) < 0.8); // Should be weaker than original 0.8
    }
    
    // Test 6: Batch remove relationships
    let remove_ops = entities.iter()
        .take(2)
        .zip(entities.iter().skip(5))
        .map(|(&from, &to)| (from, to))
        .collect::<Vec<_>>();
    
    brain_graph.batch_remove_relationships(&remove_ops).await?;
    
    // Verify relationships were removed
    for (from, to) in &remove_ops {
        assert!(!brain_graph.has_relationship(*from, *to).await);
        assert_eq!(brain_graph.get_relationship_weight(*from, *to), None);
    }
    
    // Test 7: Verify final statistics
    let final_stats = brain_graph.get_relationship_statistics().await;
    assert_eq!(final_stats.total_relationships, 3); // Started with 5, removed 2
    
    Ok(())
}

#[tokio::test]
async fn test_relationship_consistency_and_integrity() -> Result<()> {
    // Test 1: Create brain graph with entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let entity1_data = create_test_entity_data(1, 96, "integrity1");
    let entity2_data = create_test_entity_data(2, 96, "integrity2");
    
    let key1 = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let key2 = brain_graph.insert_brain_entity(2, entity2_data).await?;
    
    // Test 2: Test duplicate relationship handling
    let relationship = Relationship {
        from: key1, to: key2, rel_type: 1, weight: 0.7
    };
    
    brain_graph.insert_brain_relationship(relationship.clone()).await?;
    
    // Try to insert the same relationship again
    brain_graph.insert_brain_relationship(relationship.clone()).await?;
    
    // Should still only have one relationship
    let neighbors = brain_graph.get_neighbors_with_weights(key1).await;
    assert_eq!(neighbors.len(), 1);
    
    // Test 3: Test relationship weight consistency
    let core_weight = brain_graph.get_relationship_weight(key1, key2);
    let synaptic_weight = brain_graph.get_synaptic_weight(key1, key2).await;
    
    assert_eq!(core_weight, Some(0.7));
    assert!((synaptic_weight - 0.7).abs() < 0.01);
    
    // Test 4: Test weight update consistency
    brain_graph.update_relationship_weight(key1, key2, 0.9).await?;
    
    let updated_core_weight = brain_graph.get_relationship_weight(key1, key2);
    let updated_synaptic_weight = brain_graph.get_synaptic_weight(key1, key2).await;
    
    assert_eq!(updated_core_weight, Some(0.9));
    assert!((updated_synaptic_weight - 0.9).abs() < 0.01);
    
    // Test 5: Test relationship removal consistency
    brain_graph.remove_relationship(key1, key2).await?;
    
    assert!(!brain_graph.has_relationship(key1, key2).await);
    assert_eq!(brain_graph.get_relationship_weight(key1, key2), None);
    
    let removed_synaptic_weight = brain_graph.get_synaptic_weight(key1, key2).await;
    assert_eq!(removed_synaptic_weight, 0.0); // Should return 0 for removed relationship
    
    // Test 6: Test relationship statistics consistency
    let stats = brain_graph.get_relationship_statistics().await;
    assert_eq!(stats.total_relationships, 0); // Should reflect removal
    
    // Test 7: Test self-relationship handling
    let self_relationship = Relationship {
        from: key1, to: key1, rel_type: 1, weight: 0.5
    };
    
    // Should handle self-relationships without issues
    brain_graph.insert_brain_relationship(self_relationship).await?;
    
    assert!(brain_graph.has_relationship(key1, key1).await);
    assert_eq!(brain_graph.get_relationship_weight(key1, key1), Some(0.5));
    
    let self_neighbors = brain_graph.get_neighbors_with_weights(key1).await;
    assert_eq!(self_neighbors.len(), 1);
    assert_eq!(self_neighbors[0].0, key1);
    
    Ok(())
}