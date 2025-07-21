//! Integration tests for brain_advanced_ops.rs
//! Tests complete learning and reasoning cycles, concept operations, and graph optimization

use llm_knowledge_graph::core::brain_enhanced_graph::brain_graph_core::BrainEnhancedKnowledgeGraph;
use llm_knowledge_graph::core::brain_enhanced_graph::brain_graph_types::*;
use llm_knowledge_graph::core::types::{EntityData, EntityKey};
use std::time::Instant;
use tokio;

/// Helper to create a test knowledge graph with initial data
async fn create_test_brain_graph(num_entities: usize) -> BrainEnhancedKnowledgeGraph {
    let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    
    // Add test entities with varied properties
    for i in 0..num_entities {
        let entity_data = EntityData {
            name: format!("entity_{}", i),
            properties: format!("type:{}, category:{}, importance:{}", 
                if i % 3 == 0 { "input" } else if i % 3 == 1 { "output" } else { "processing" },
                i % 5,
                i as f32 / num_entities as f32
            ),
            embedding: (0..128).map(|j| ((i + j) as f32).sin() / 10.0).collect(),
        };
        graph.core_graph.insert_entity(entity_data).unwrap();
    }
    
    // Add relationships to create interesting graph structure
    for i in 0..num_entities {
        if i + 1 < num_entities {
            graph.add_relationship(i as u32, (i + 1) as u32, 1, 0.8).await.unwrap();
        }
        if i + 2 < num_entities {
            graph.add_relationship(i as u32, (i + 2) as u32, 2, 0.5).await.unwrap();
        }
        if i > 0 && i % 5 == 0 {
            graph.add_relationship(i as u32, (i - 5) as u32, 3, 0.3).await.unwrap();
        }
    }
    
    graph
}

#[tokio::test]
async fn test_complete_learning_and_reasoning_cycle() {
    println!("\n=== Complete Learning and Reasoning Cycle Test ===");
    
    let graph = create_test_brain_graph(100).await;
    
    // Phase 1: Initial activation and learning
    println!("\nPhase 1: Initial Learning");
    
    // Activate some seed entities
    let seed_entities = vec![10u32, 20, 30, 40, 50];
    for &entity_id in &seed_entities {
        graph.set_entity_activation(entity_id, 0.9).await;
    }
    
    // Run learning cycles
    let learning_start = Instant::now();
    for cycle in 0..5 {
        graph.update_learning_cycle().await.unwrap();
        
        // Check learning progress
        let stats = graph.get_brain_statistics().await.unwrap();
        println!("Cycle {}: Avg activation = {:.3}, Learning efficiency = {:.3}", 
            cycle, stats.avg_activation, stats.learning_efficiency);
    }
    let learning_time = learning_start.elapsed();
    println!("Learning phase completed in {:?}", learning_time);
    
    // Phase 2: Pattern detection and concept formation
    println!("\nPhase 2: Pattern Detection and Concept Formation");
    
    // Detect activation patterns
    let patterns = graph.detect_activation_patterns(0.3).await.unwrap();
    println!("Detected {} activation patterns", patterns.len());
    
    // Form concepts from patterns
    let concept_start = Instant::now();
    let mut concepts = Vec::new();
    
    for (i, pattern) in patterns.iter().take(3).enumerate() {
        let concept_id = graph.create_concept_structure(
            format!("concept_{}", i),
            pattern.clone(),
            0.8
        ).await.unwrap();
        concepts.push(concept_id);
        
        println!("Created concept {} with {} entities", i, pattern.len());
    }
    let concept_time = concept_start.elapsed();
    println!("Concept formation completed in {:?}", concept_time);
    
    // Phase 3: Reasoning with concepts
    println!("\nPhase 3: Reasoning with Concepts");
    
    if concepts.len() >= 2 {
        // Test concept similarity
        let similarity = graph.compute_concept_similarity(concepts[0], concepts[1]).await.unwrap();
        println!("Similarity between concept 0 and 1: {:.3}", similarity);
        
        // Create inheritance relationship if similar enough
        if similarity > 0.5 {
            graph.create_inheritance(concepts[0], concepts[1], similarity).await.unwrap();
            println!("Created inheritance relationship with strength {:.3}", similarity);
        }
    }
    
    // Phase 4: Query and retrieve knowledge
    println!("\nPhase 4: Knowledge Retrieval");
    
    // Find similar concepts
    if let Some(first_concept) = concepts.first() {
        let similar = graph.find_similar_concepts(*first_concept, 5).await.unwrap();
        println!("Found {} similar concepts", similar.len());
        
        for (concept_id, similarity) in similar.iter().take(3) {
            println!("  Concept {}: similarity = {:.3}", concept_id, similarity);
        }
    }
    
    // Phase 5: Adaptive optimization
    println!("\nPhase 5: Adaptive Optimization");
    
    let optimization_start = Instant::now();
    
    // Run graph optimization
    graph.optimize_graph_structure(0.1).await.unwrap();
    
    // Prune weak connections
    let pruned = graph.prune_weak_connections(0.05).await.unwrap();
    println!("Pruned {} weak connections", pruned);
    
    let optimization_time = optimization_start.elapsed();
    println!("Optimization completed in {:?}", optimization_time);
    
    // Final statistics
    let final_stats = graph.get_brain_statistics().await.unwrap();
    println!("\nFinal Graph Statistics:");
    println!("  Entities: {}", final_stats.entity_count);
    println!("  Relationships: {}", final_stats.relationship_count);
    println!("  Avg activation: {:.3}", final_stats.avg_activation);
    println!("  Graph density: {:.3}", final_stats.graph_density);
    println!("  Clustering coefficient: {:.3}", final_stats.clustering_coefficient);
    println!("  Concept coherence: {:.3}", final_stats.concept_coherence);
    
    // Verify improvements
    assert!(final_stats.concept_coherence > 0.0, "Should have concept coherence");
    assert!(final_stats.learning_efficiency > 0.5, "Learning should be efficient");
}

#[tokio::test]
async fn test_concept_structure_and_similarity_operations() {
    println!("\n=== Concept Structure and Similarity Test ===");
    
    let graph = create_test_brain_graph(200).await;
    
    // Create multiple overlapping concepts
    let concept_configs = vec![
        ("math_concept", vec![0, 5, 10, 15, 20, 25], 0.9),
        ("science_concept", vec![5, 10, 15, 30, 35, 40], 0.85),
        ("art_concept", vec![50, 55, 60, 65, 70], 0.8),
        ("tech_concept", vec![20, 25, 30, 35, 45], 0.95),
    ];
    
    let mut concept_ids = Vec::new();
    
    // Create concepts
    for (name, entities, strength) in concept_configs {
        let concept_id = graph.create_concept_structure(
            name.to_string(),
            entities.into_iter().map(|e| e as u32).collect(),
            strength
        ).await.unwrap();
        concept_ids.push((name, concept_id));
        println!("Created concept '{}' with ID {}", name, concept_id);
    }
    
    // Test concept similarities
    println!("\nConcept Similarity Matrix:");
    for i in 0..concept_ids.len() {
        for j in i+1..concept_ids.len() {
            let similarity = graph.compute_concept_similarity(
                concept_ids[i].1, 
                concept_ids[j].1
            ).await.unwrap();
            
            println!("  {} <-> {}: {:.3}", 
                concept_ids[i].0, concept_ids[j].0, similarity);
            
            // Math and science should be more similar than math and art
            if concept_ids[i].0 == "math_concept" && concept_ids[j].0 == "science_concept" {
                assert!(similarity > 0.3, "Math and science should have some similarity");
            }
            if concept_ids[i].0 == "math_concept" && concept_ids[j].0 == "art_concept" {
                assert!(similarity < 0.5, "Math and art should have low similarity");
            }
        }
    }
    
    // Test finding similar concepts
    println!("\nFinding Similar Concepts:");
    for (name, concept_id) in &concept_ids {
        let similar = graph.find_similar_concepts(*concept_id, 3).await.unwrap();
        println!("\nSimilar to '{}':", name);
        for (similar_id, score) in similar {
            if similar_id != *concept_id {
                let similar_name = concept_ids.iter()
                    .find(|(_, id)| *id == similar_id)
                    .map(|(name, _)| name)
                    .unwrap_or(&"unknown");
                println!("  {} (score: {:.3})", similar_name, score);
            }
        }
    }
    
    // Test concept merging
    if concept_ids.len() >= 2 {
        let (name1, id1) = &concept_ids[0];
        let (name2, id2) = &concept_ids[1];
        
        println!("\nTesting concept merge: {} + {}", name1, name2);
        
        // Get initial entity counts
        let entities1 = graph.get_concept_entities(*id1).await.unwrap();
        let entities2 = graph.get_concept_entities(*id2).await.unwrap();
        
        println!("Before merge: {} has {} entities, {} has {} entities",
            name1, entities1.len(), name2, entities2.len());
        
        // Create inheritance to simulate concept hierarchy
        graph.create_inheritance(*id1, *id2, 0.7).await.unwrap();
        
        // Verify inheritance effect
        let child_activation = graph.get_entity_activation(*id2).await;
        assert!(child_activation > 0.0, "Child concept should have inherited activation");
    }
}

#[tokio::test]
async fn test_graph_optimization_workflow() {
    println!("\n=== Graph Optimization Workflow Test ===");
    
    // Create a dense graph that needs optimization
    let graph = create_test_brain_graph(500).await;
    
    // Add many random weak connections
    println!("Adding random weak connections...");
    for _ in 0..1000 {
        let from = rand::random::<u32>() % 500;
        let to = rand::random::<u32>() % 500;
        if from != to {
            let _ = graph.add_relationship(from, to, 4, 0.1).await;
        }
    }
    
    // Get initial statistics
    let initial_stats = graph.get_brain_statistics().await.unwrap();
    println!("\nInitial Graph Statistics:");
    println!("  Relationships: {}", initial_stats.relationship_count);
    println!("  Graph density: {:.4}", initial_stats.graph_density);
    println!("  Clustering coefficient: {:.4}", initial_stats.clustering_coefficient);
    
    // Phase 1: Identify communities
    println!("\nPhase 1: Community Detection");
    let community_start = Instant::now();
    
    // Activate different regions to create communities
    for i in 0..5 {
        let center = i * 100;
        for j in 0..20 {
            let entity = center + j;
            if entity < 500 {
                graph.set_entity_activation(entity as u32, 0.8 - (j as f32 * 0.02)).await;
            }
        }
    }
    
    // Run activation spreading
    graph.spread_activation(5).await.unwrap();
    
    let patterns = graph.detect_activation_patterns(0.2).await.unwrap();
    let community_time = community_start.elapsed();
    println!("Detected {} communities in {:?}", patterns.len(), community_time);
    
    // Phase 2: Optimize structure
    println!("\nPhase 2: Structure Optimization");
    let optimization_start = Instant::now();
    
    // Optimize graph structure
    graph.optimize_graph_structure(0.15).await.unwrap();
    
    // Prune very weak connections
    let pruned_count = graph.prune_weak_connections(0.05).await.unwrap();
    println!("Pruned {} weak connections", pruned_count);
    
    // Consolidate similar patterns
    graph.consolidate_similar_patterns(0.7).await.unwrap();
    
    let optimization_time = optimization_start.elapsed();
    println!("Optimization completed in {:?}", optimization_time);
    
    // Phase 3: Verify improvements
    let optimized_stats = graph.get_brain_statistics().await.unwrap();
    println!("\nOptimized Graph Statistics:");
    println!("  Relationships: {} (reduced by {})", 
        optimized_stats.relationship_count,
        initial_stats.relationship_count - optimized_stats.relationship_count);
    println!("  Graph density: {:.4} (change: {:.4})", 
        optimized_stats.graph_density,
        optimized_stats.graph_density - initial_stats.graph_density);
    println!("  Clustering coefficient: {:.4} (change: {:.4})", 
        optimized_stats.clustering_coefficient,
        optimized_stats.clustering_coefficient - initial_stats.clustering_coefficient);
    
    // Verify optimization effectiveness
    assert!(optimized_stats.relationship_count < initial_stats.relationship_count,
        "Should have fewer relationships after pruning");
    assert!(pruned_count > 0, "Should have pruned some connections");
    
    // Test query performance after optimization
    println!("\nTesting Query Performance:");
    let query_start = Instant::now();
    
    for i in 0..100 {
        let entity = (i * 5) as u32;
        let neighbors = graph.get_entity_neighbors(entity, 2).await.unwrap();
        assert!(!neighbors.is_empty() || entity >= 500);
    }
    
    let query_time = query_start.elapsed();
    println!("100 neighborhood queries completed in {:?}", query_time);
    assert!(query_time.as_millis() < 1000, "Queries should be fast after optimization");
}

#[tokio::test]
async fn test_advanced_learning_scenarios() {
    println!("\n=== Advanced Learning Scenarios Test ===");
    
    let graph = create_test_brain_graph(300).await;
    
    // Scenario 1: Incremental learning with feedback
    println!("\nScenario 1: Incremental Learning with Feedback");
    
    // Initial learning phase
    for i in 0..10 {
        // Activate different patterns
        let pattern_start = i * 30;
        for j in 0..10 {
            let entity = pattern_start + j;
            if entity < 300 {
                graph.set_entity_activation(entity as u32, 0.7).await;
            }
        }
        
        // Learn from activation
        graph.update_learning_cycle().await.unwrap();
        
        // Simulate feedback by reinforcing successful patterns
        if i % 2 == 0 {
            for j in 0..5 {
                let entity = pattern_start + j;
                if entity < 300 {
                    let current = graph.get_entity_activation(entity as u32).await;
                    graph.set_entity_activation(entity as u32, (current * 1.2).min(1.0)).await;
                }
            }
        }
    }
    
    let learning_stats = graph.get_brain_statistics().await.unwrap();
    println!("Learning efficiency after feedback: {:.3}", learning_stats.learning_efficiency);
    
    // Scenario 2: Competitive learning between concepts
    println!("\nScenario 2: Competitive Learning");
    
    // Create competing concepts
    let concept_a_entities: Vec<u32> = (0..20).map(|i| i * 3).collect();
    let concept_b_entities: Vec<u32> = (0..20).map(|i| i * 3 + 1).collect();
    
    let concept_a = graph.create_concept_structure(
        "concept_a".to_string(),
        concept_a_entities.clone(),
        0.9
    ).await.unwrap();
    
    let concept_b = graph.create_concept_structure(
        "concept_b".to_string(),
        concept_b_entities.clone(),
        0.9
    ).await.unwrap();
    
    // Run competitive activation
    for round in 0..5 {
        // Activate concept A
        for &entity in &concept_a_entities {
            graph.set_entity_activation(entity, 0.8).await;
        }
        graph.spread_activation(3).await.unwrap();
        
        let a_avg = concept_a_entities.iter()
            .map(|&e| graph.get_entity_activation(e).await)
            .sum::<f32>() / concept_a_entities.len() as f32;
        
        // Activate concept B
        for &entity in &concept_b_entities {
            graph.set_entity_activation(entity, 0.8).await;
        }
        graph.spread_activation(3).await.unwrap();
        
        let b_avg = concept_b_entities.iter()
            .map(|&e| graph.get_entity_activation(e).await)
            .sum::<f32>() / concept_b_entities.len() as f32;
        
        println!("Round {}: Concept A avg = {:.3}, Concept B avg = {:.3}", 
            round, a_avg, b_avg);
    }
    
    // Scenario 3: Hierarchical concept learning
    println!("\nScenario 3: Hierarchical Learning");
    
    // Create hierarchy: general -> specific concepts
    let general_entities: Vec<u32> = (0..50).step_by(5).collect();
    let specific1_entities: Vec<u32> = (0..25).step_by(5).collect();
    let specific2_entities: Vec<u32> = (25..50).step_by(5).collect();
    
    let general_concept = graph.create_concept_structure(
        "general".to_string(),
        general_entities,
        0.7
    ).await.unwrap();
    
    let specific1 = graph.create_concept_structure(
        "specific1".to_string(),
        specific1_entities,
        0.9
    ).await.unwrap();
    
    let specific2 = graph.create_concept_structure(
        "specific2".to_string(),
        specific2_entities,
        0.9
    ).await.unwrap();
    
    // Create inheritance relationships
    graph.create_inheritance(general_concept, specific1, 0.8).await.unwrap();
    graph.create_inheritance(general_concept, specific2, 0.8).await.unwrap();
    
    // Test inheritance propagation
    graph.set_entity_activation(general_concept, 1.0).await;
    graph.spread_activation(3).await.unwrap();
    
    let spec1_activation = graph.get_entity_activation(specific1).await;
    let spec2_activation = graph.get_entity_activation(specific2).await;
    
    println!("After inheritance propagation:");
    println!("  Specific1 activation: {:.3}", spec1_activation);
    println!("  Specific2 activation: {:.3}", spec2_activation);
    
    assert!(spec1_activation > 0.0, "Should inherit activation");
    assert!(spec2_activation > 0.0, "Should inherit activation");
}

#[tokio::test]
async fn test_performance_under_load() {
    println!("\n=== Performance Under Load Test ===");
    
    // Create large graph
    let num_entities = 10000;
    let graph = create_test_brain_graph(num_entities).await;
    
    // Add complex relationship structure
    println!("Building complex relationship structure...");
    let setup_start = Instant::now();
    
    for i in 0..num_entities {
        // Add multiple relationship types
        for j in 1..=5 {
            let target = (i + j * 100) % num_entities;
            let _ = graph.add_relationship(i as u32, target as u32, j as u16, 0.5).await;
        }
    }
    
    let setup_time = setup_start.elapsed();
    println!("Setup completed in {:?}", setup_time);
    
    // Test 1: Concept creation at scale
    println!("\nTest 1: Large-scale concept creation");
    let concept_start = Instant::now();
    let mut concepts = Vec::new();
    
    for i in 0..100 {
        let entities: Vec<u32> = (i*100..(i+1)*100).step_by(10).map(|e| e as u32).collect();
        let concept_id = graph.create_concept_structure(
            format!("large_concept_{}", i),
            entities,
            0.8
        ).await.unwrap();
        concepts.push(concept_id);
    }
    
    let concept_time = concept_start.elapsed();
    println!("Created 100 concepts in {:?}", concept_time);
    assert!(concept_time.as_secs() < 10, "Concept creation should be reasonably fast");
    
    // Test 2: Parallel similarity computations
    println!("\nTest 2: Parallel similarity computations");
    let similarity_start = Instant::now();
    let mut similarity_tasks = Vec::new();
    
    for i in 0..10 {
        for j in i+1..10 {
            let similarity = graph.compute_concept_similarity(concepts[i], concepts[j]).await.unwrap();
            similarity_tasks.push(similarity);
        }
    }
    
    let similarity_time = similarity_start.elapsed();
    println!("Computed {} similarities in {:?}", similarity_tasks.len(), similarity_time);
    
    // Test 3: Activation spreading at scale
    println!("\nTest 3: Large-scale activation spreading");
    
    // Activate multiple seed points
    for i in (0..num_entities).step_by(100) {
        graph.set_entity_activation(i as u32, 0.9).await;
    }
    
    let spread_start = Instant::now();
    graph.spread_activation(10).await.unwrap();
    let spread_time = spread_start.elapsed();
    
    println!("Activation spreading completed in {:?}", spread_time);
    assert!(spread_time.as_secs() < 30, "Spreading should complete in reasonable time");
    
    // Test 4: Pattern detection at scale
    println!("\nTest 4: Pattern detection at scale");
    let pattern_start = Instant::now();
    let patterns = graph.detect_activation_patterns(0.1).await.unwrap();
    let pattern_time = pattern_start.elapsed();
    
    println!("Detected {} patterns in {:?}", patterns.len(), pattern_time);
    
    // Final statistics
    let final_stats = graph.get_brain_statistics().await.unwrap();
    println!("\nFinal Performance Metrics:");
    println!("  Total entities: {}", final_stats.entity_count);
    println!("  Total relationships: {}", final_stats.relationship_count);
    println!("  Graph density: {:.6}", final_stats.graph_density);
    println!("  Average activation: {:.3}", final_stats.avg_activation);
}

#[tokio::test]
async fn test_edge_cases_and_error_recovery() {
    println!("\n=== Edge Cases and Error Recovery Test ===");
    
    let graph = create_test_brain_graph(100).await;
    
    // Test 1: Empty concept creation
    let empty_concept = graph.create_concept_structure(
        "empty_concept".to_string(),
        Vec::new(),
        0.5
    ).await;
    assert!(empty_concept.is_ok(), "Should handle empty concept creation");
    
    // Test 2: Duplicate relationships
    let result1 = graph.add_relationship(1, 2, 1, 0.5).await;
    let result2 = graph.add_relationship(1, 2, 1, 0.6).await;
    assert!(result1.is_ok() && result2.is_ok(), "Should handle duplicate relationships");
    
    // Test 3: Self-referential concepts
    let self_ref_entities = vec![50u32];
    let self_ref = graph.create_concept_structure(
        "self_ref".to_string(),
        self_ref_entities,
        0.8
    ).await.unwrap();
    
    // Try to create inheritance to itself
    let self_inheritance = graph.create_inheritance(self_ref, self_ref, 0.5).await;
    assert!(self_inheritance.is_ok(), "Should handle self-inheritance gracefully");
    
    // Test 4: Extreme activation values
    graph.set_entity_activation(10, 100.0).await;
    let clamped = graph.get_entity_activation(10).await;
    assert_eq!(clamped, 1.0, "Should clamp activation to valid range");
    
    graph.set_entity_activation(11, -100.0).await;
    let clamped_neg = graph.get_entity_activation(11).await;
    assert_eq!(clamped_neg, 0.0, "Should clamp negative activation to 0");
    
    // Test 5: Very high similarity threshold
    let similar = graph.find_similar_concepts(self_ref, 10).await.unwrap();
    assert!(similar.len() <= 10, "Should respect result limit");
    
    // Test 6: Pattern detection with no activation
    // Reset all activations
    for i in 0..100 {
        graph.set_entity_activation(i as u32, 0.0).await;
    }
    
    let no_patterns = graph.detect_activation_patterns(0.5).await.unwrap();
    assert_eq!(no_patterns.len(), 0, "Should find no patterns with zero activation");
    
    println!("All edge cases handled successfully");
}