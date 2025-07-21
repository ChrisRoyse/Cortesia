//! Integration tests for brain_analytics.rs
//! Tests comprehensive graph analysis workflows and statistical accuracy

use llm_knowledge_graph::core::brain_enhanced_graph::brain_graph_core::BrainEnhancedKnowledgeGraph;
use llm_knowledge_graph::core::brain_enhanced_graph::brain_graph_types::*;
use llm_knowledge_graph::core::types::EntityData;
use std::time::Instant;
use std::collections::HashMap;
use tokio;

/// Create a graph with known structure for testing analytics
async fn create_structured_graph(
    nodes: usize,
    edges_per_node: usize,
    clustering: f32
) -> BrainEnhancedKnowledgeGraph {
    let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    
    // Add nodes
    for i in 0..nodes {
        let entity_data = EntityData {
            name: format!("node_{}", i),
            properties: format!("type:analytics_test, index:{}, cluster:{}", i, i / 10),
            embedding: (0..128).map(|j| ((i + j) as f32).sin() / 10.0).collect(),
        };
        graph.core_graph.insert_entity(entity_data).unwrap();
    }
    
    // Create edges with controlled structure
    for i in 0..nodes {
        // Connect to next nodes (ring structure)
        for j in 1..=edges_per_node {
            let target = (i + j) % nodes;
            graph.add_relationship(i as u32, target as u32, 1, 0.8).await.unwrap();
        }
        
        // Add clustering by connecting to nearby nodes
        if clustering > 0.0 {
            let cluster_size = (nodes as f32 * clustering) as usize;
            for j in 0..cluster_size {
                let target = (i + j * 2) % nodes;
                if target != i {
                    graph.add_relationship(i as u32, target as u32, 2, 0.6).await.unwrap();
                }
            }
        }
    }
    
    graph
}

/// Create different graph topologies for testing
async fn create_test_topologies() -> HashMap<String, BrainEnhancedKnowledgeGraph> {
    let mut graphs = HashMap::new();
    
    // Ring topology
    let ring = create_structured_graph(100, 2, 0.0).await;
    graphs.insert("ring".to_string(), ring);
    
    // Small world network
    let small_world = create_structured_graph(100, 4, 0.1).await;
    graphs.insert("small_world".to_string(), small_world);
    
    // Highly clustered
    let clustered = create_structured_graph(100, 3, 0.3).await;
    graphs.insert("clustered".to_string(), clustered);
    
    // Random network
    let random = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    for i in 0..100 {
        let entity_data = EntityData {
            name: format!("random_{}", i),
            properties: "type:random".to_string(),
            embedding: vec![0.5; 128],
        };
        random.core_graph.insert_entity(entity_data).unwrap();
    }
    
    // Add random edges
    for _ in 0..300 {
        let from = rand::random::<u32>() % 100;
        let to = rand::random::<u32>() % 100;
        if from != to {
            let _ = random.add_relationship(from, to, 1, 0.5).await;
        }
    }
    graphs.insert("random".to_string(), random);
    
    graphs
}

#[tokio::test]
async fn test_comprehensive_graph_analysis_workflow() {
    println!("\n=== Comprehensive Graph Analysis Workflow ===");
    
    let graphs = create_test_topologies().await;
    
    for (topology, graph) in graphs {
        println!("\n--- Analyzing {} topology ---", topology);
        
        let analysis_start = Instant::now();
        
        // Get comprehensive statistics
        let stats = graph.get_brain_statistics().await.unwrap();
        
        println!("Basic metrics:");
        println!("  Entities: {}", stats.entity_count);
        println!("  Relationships: {}", stats.relationship_count);
        println!("  Density: {:.4}", stats.graph_density);
        
        println!("\nStructural metrics:");
        println!("  Clustering coefficient: {:.4}", stats.clustering_coefficient);
        println!("  Average path length: {:.4}", stats.average_path_length);
        
        println!("\nActivation metrics:");
        println!("  Average activation: {:.4}", stats.avg_activation);
        println!("  Min activation: {:.4}", stats.min_activation);
        println!("  Max activation: {:.4}", stats.max_activation);
        
        println!("\nCognitive metrics:");
        println!("  Concept coherence: {:.4}", stats.concept_coherence);
        println!("  Learning efficiency: {:.4}", stats.learning_efficiency);
        
        let analysis_time = analysis_start.elapsed();
        println!("\nAnalysis completed in {:?}", analysis_time);
        
        // Verify expected properties for each topology
        match topology.as_str() {
            "ring" => {
                assert!(stats.clustering_coefficient < 0.1, "Ring should have low clustering");
                assert!(stats.average_path_length > 10.0, "Ring should have long paths");
            },
            "small_world" => {
                assert!(stats.clustering_coefficient > 0.1, "Small world should have clustering");
                assert!(stats.average_path_length < 10.0, "Small world should have short paths");
            },
            "clustered" => {
                assert!(stats.clustering_coefficient > 0.2, "Should be highly clustered");
            },
            "random" => {
                assert!(stats.graph_density > 0.02, "Random graph should have reasonable density");
            },
            _ => {}
        }
    }
}

#[tokio::test]
async fn test_statistical_accuracy_large_graphs() {
    println!("\n=== Statistical Accuracy Test on Large Graphs ===");
    
    // Create graphs of different sizes
    let sizes = vec![100, 500, 1000, 2500];
    
    for size in sizes {
        println!("\n--- Testing graph with {} nodes ---", size);
        
        let graph = create_structured_graph(size, 6, 0.15).await;
        
        // Activate nodes with known distribution
        for i in 0..size {
            let activation = (i as f32 / size as f32).powf(2.0); // Quadratic distribution
            graph.set_entity_activation(i as u32, activation).await;
        }
        
        let stats_start = Instant::now();
        let stats = graph.get_brain_statistics().await.unwrap();
        let stats_time = stats_start.elapsed();
        
        println!("Statistics computed in {:?}", stats_time);
        
        // Verify activation distribution
        println!("\nActivation distribution:");
        for (range, count) in &stats.activation_distribution {
            let percentage = (*count as f32 / size as f32) * 100.0;
            println!("  {}: {} nodes ({:.1}%)", range, count, percentage);
        }
        
        // Expected average for quadratic distribution: integral of x^2 from 0 to 1 = 1/3
        let expected_avg = 1.0 / 3.0;
        let actual_avg = stats.avg_activation;
        let error = (actual_avg - expected_avg).abs();
        
        println!("\nActivation average:");
        println!("  Expected: {:.4}", expected_avg);
        println!("  Actual: {:.4}", actual_avg);
        println!("  Error: {:.4}", error);
        
        assert!(error < 0.01, "Average activation should be accurate");
        
        // Verify density calculation
        let expected_edges = size * 6 + (size as f32 * 0.15 * size as f32) as usize;
        let max_edges = size * (size - 1);
        let expected_density = expected_edges as f32 / max_edges as f32;
        let density_error = (stats.graph_density - expected_density).abs();
        
        println!("\nGraph density:");
        println!("  Expected: ~{:.4}", expected_density);
        println!("  Actual: {:.4}", stats.graph_density);
        println!("  Error: {:.4}", density_error);
        
        // Betweenness centrality sanity check
        let max_centrality = stats.betweenness_centrality.values()
            .cloned()
            .fold(0.0f32, f32::max);
        println!("\nMax betweenness centrality: {:.4}", max_centrality);
        assert!(max_centrality > 0.0, "Should have non-zero centrality");
        
        // Performance assertion
        assert!(stats_time.as_secs() < 60, "Statistics should compute in under 60 seconds");
    }
}

#[tokio::test]
async fn test_graph_health_assessment() {
    println!("\n=== Graph Health Assessment Test ===");
    
    // Create graphs with different health characteristics
    
    // Healthy graph - well-connected, balanced
    println!("\n--- Healthy Graph ---");
    let healthy = create_structured_graph(200, 5, 0.2).await;
    
    // Activate with good distribution
    for i in 0..200 {
        let activation = 0.3 + (i as f32 / 400.0);
        healthy.set_entity_activation(i as u32, activation).await;
    }
    
    // Add some concepts
    for i in 0..5 {
        let entities: Vec<u32> = (i*40..(i+1)*40).step_by(2).collect();
        healthy.create_concept_structure(
            format!("healthy_concept_{}", i),
            entities,
            0.8
        ).await.unwrap();
    }
    
    let healthy_stats = healthy.get_brain_statistics().await.unwrap();
    println!("Healthy graph metrics:");
    println!("  Clustering: {:.3}", healthy_stats.clustering_coefficient);
    println!("  Path length: {:.3}", healthy_stats.average_path_length);
    println!("  Concept coherence: {:.3}", healthy_stats.concept_coherence);
    println!("  Learning efficiency: {:.3}", healthy_stats.learning_efficiency);
    
    assert!(healthy_stats.clustering_coefficient > 0.1, "Should have good clustering");
    assert!(healthy_stats.concept_coherence > 0.5, "Should have coherent concepts");
    
    // Fragmented graph - disconnected components
    println!("\n--- Fragmented Graph ---");
    let fragmented = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    
    // Create disconnected islands
    for island in 0..5 {
        let base = island * 40;
        for i in 0..30 {
            let entity_data = EntityData {
                name: format!("island{}_{}", island, i),
                properties: format!("island:{}", island),
                embedding: vec![island as f32 / 5.0; 128],
            };
            fragmented.core_graph.insert_entity(entity_data).unwrap();
            
            // Connect within island only
            if i > 0 {
                fragmented.add_relationship(
                    (base + i) as u32,
                    (base + i - 1) as u32,
                    1,
                    0.9
                ).await.unwrap();
            }
        }
    }
    
    let fragmented_stats = fragmented.get_brain_statistics().await.unwrap();
    println!("Fragmented graph metrics:");
    println!("  Clustering: {:.3}", fragmented_stats.clustering_coefficient);
    println!("  Path length: {:.3}", fragmented_stats.average_path_length);
    println!("  Density: {:.6}", fragmented_stats.graph_density);
    
    assert!(fragmented_stats.graph_density < 0.01, "Should have very low density");
    
    // Overconnected graph - too many connections
    println!("\n--- Overconnected Graph ---");
    let overconnected = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    
    // Add nodes
    for i in 0..50 {
        let entity_data = EntityData {
            name: format!("over_{}", i),
            properties: "type:overconnected".to_string(),
            embedding: vec![0.5; 128],
        };
        overconnected.core_graph.insert_entity(entity_data).unwrap();
    }
    
    // Connect almost everything
    for i in 0..50 {
        for j in 0..50 {
            if i != j && rand::random::<f32>() > 0.3 {
                let _ = overconnected.add_relationship(i, j, 1, 0.5).await;
            }
        }
    }
    
    let over_stats = overconnected.get_brain_statistics().await.unwrap();
    println!("Overconnected graph metrics:");
    println!("  Clustering: {:.3}", over_stats.clustering_coefficient);
    println!("  Path length: {:.3}", over_stats.average_path_length);
    println!("  Density: {:.3}", over_stats.graph_density);
    
    assert!(over_stats.graph_density > 0.5, "Should have very high density");
    assert!(over_stats.average_path_length < 2.0, "Should have very short paths");
}

#[tokio::test]
async fn test_dynamic_analytics_over_time() {
    println!("\n=== Dynamic Analytics Over Time ===");
    
    let graph = create_structured_graph(300, 4, 0.1).await;
    let mut timeline = Vec::new();
    
    // Simulate graph evolution
    for epoch in 0..10 {
        println!("\n--- Epoch {} ---", epoch);
        
        // Modify graph structure
        match epoch {
            0..=2 => {
                // Growth phase - add new connections
                for _ in 0..50 {
                    let from = rand::random::<u32>() % 300;
                    let to = rand::random::<u32>() % 300;
                    if from != to {
                        let _ = graph.add_relationship(from, to, 2, 0.7).await;
                    }
                }
            },
            3..=5 => {
                // Learning phase - create concepts
                for i in 0..3 {
                    let start = i * 100;
                    let entities: Vec<u32> = (start..start+30).step_by(3).collect();
                    graph.create_concept_structure(
                        format!("epoch{}_concept{}", epoch, i),
                        entities,
                        0.85
                    ).await.unwrap();
                }
                
                // Spread activation
                graph.spread_activation(5).await.unwrap();
            },
            6..=8 => {
                // Optimization phase - prune weak connections
                let pruned = graph.prune_weak_connections(0.2).await.unwrap();
                println!("Pruned {} connections", pruned);
                
                // Optimize structure
                graph.optimize_graph_structure(0.1).await.unwrap();
            },
            _ => {
                // Stabilization phase
                graph.update_learning_cycle().await.unwrap();
            }
        }
        
        // Collect analytics
        let stats = graph.get_brain_statistics().await.unwrap();
        
        timeline.push((epoch, stats.clone()));
        
        println!("Metrics at epoch {}:", epoch);
        println!("  Relationships: {}", stats.relationship_count);
        println!("  Clustering: {:.3}", stats.clustering_coefficient);
        println!("  Coherence: {:.3}", stats.concept_coherence);
        println!("  Learning efficiency: {:.3}", stats.learning_efficiency);
    }
    
    // Analyze trends
    println!("\n=== Trend Analysis ===");
    
    let initial_relationships = timeline[0].1.relationship_count;
    let final_relationships = timeline[9].1.relationship_count;
    println!("Relationship change: {} -> {} ({:+})", 
        initial_relationships, final_relationships,
        final_relationships as i32 - initial_relationships as i32);
    
    let max_clustering = timeline.iter()
        .map(|(_, stats)| stats.clustering_coefficient)
        .fold(0.0f32, f32::max);
    let final_clustering = timeline[9].1.clustering_coefficient;
    println!("Peak clustering: {:.3}, Final: {:.3}", max_clustering, final_clustering);
    
    // Verify learning improvement
    let initial_learning = timeline[0].1.learning_efficiency;
    let final_learning = timeline[9].1.learning_efficiency;
    println!("Learning efficiency: {:.3} -> {:.3}", initial_learning, final_learning);
    
    assert!(final_learning >= initial_learning, "Learning should improve or maintain");
}

#[tokio::test]
async fn test_centrality_and_importance_metrics() {
    println!("\n=== Centrality and Importance Metrics Test ===");
    
    // Create hub-and-spoke topology
    let graph = BrainEnhancedKnowledgeGraph::new_for_test().unwrap();
    
    // Create hubs
    let num_hubs = 5;
    let nodes_per_hub = 20;
    
    for hub in 0..num_hubs {
        // Hub node
        let hub_id = hub * (nodes_per_hub + 1);
        let hub_data = EntityData {
            name: format!("hub_{}", hub),
            properties: format!("type:hub, importance:high"),
            embedding: vec![1.0; 128],
        };
        graph.core_graph.insert_entity(hub_data).unwrap();
        
        // Spoke nodes
        for spoke in 1..=nodes_per_hub {
            let spoke_id = hub_id + spoke;
            let spoke_data = EntityData {
                name: format!("spoke_{}_{}", hub, spoke),
                properties: format!("type:spoke, hub:{}", hub),
                embedding: vec![0.5; 128],
            };
            graph.core_graph.insert_entity(spoke_data).unwrap();
            
            // Connect spoke to hub
            graph.add_relationship(spoke_id as u32, hub_id as u32, 1, 0.9).await.unwrap();
        }
    }
    
    // Connect hubs to each other
    for i in 0..num_hubs {
        for j in i+1..num_hubs {
            let hub_i = i * (nodes_per_hub + 1);
            let hub_j = j * (nodes_per_hub + 1);
            graph.add_relationship(hub_i as u32, hub_j as u32, 2, 0.8).await.unwrap();
        }
    }
    
    // Get statistics with centrality metrics
    let stats = graph.get_brain_statistics().await.unwrap();
    
    println!("Graph structure created:");
    println!("  {} hubs with {} spokes each", num_hubs, nodes_per_hub);
    println!("  Total nodes: {}", stats.entity_count);
    println!("  Total edges: {}", stats.relationship_count);
    
    // Analyze betweenness centrality
    println!("\nBetweenness Centrality Analysis:");
    
    let mut centralities: Vec<(u32, f32)> = stats.betweenness_centrality.iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    centralities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Top 10 nodes by betweenness centrality:");
    for (i, (node, centrality)) in centralities.iter().take(10).enumerate() {
        let node_type = if *node % (nodes_per_hub + 1) as u32 == 0 { "hub" } else { "spoke" };
        println!("  {}: Node {} ({}) - {:.4}", i+1, node, node_type, centrality);
    }
    
    // Verify hubs have highest centrality
    let hub_centralities: Vec<f32> = (0..num_hubs)
        .map(|i| {
            let hub_id = (i * (nodes_per_hub + 1)) as u32;
            stats.betweenness_centrality.get(&hub_id).cloned().unwrap_or(0.0)
        })
        .collect();
    
    let avg_hub_centrality = hub_centralities.iter().sum::<f32>() / hub_centralities.len() as f32;
    let avg_overall_centrality = stats.betweenness_centrality.values().sum::<f32>() 
        / stats.betweenness_centrality.len() as f32;
    
    println!("\nCentrality comparison:");
    println!("  Average hub centrality: {:.4}", avg_hub_centrality);
    println!("  Average overall centrality: {:.4}", avg_overall_centrality);
    
    assert!(avg_hub_centrality > avg_overall_centrality * 5.0, 
        "Hubs should have much higher centrality");
}

#[tokio::test]
async fn test_activation_distribution_analysis() {
    println!("\n=== Activation Distribution Analysis ===");
    
    let graph = create_structured_graph(1000, 4, 0.15).await;
    
    // Test different activation patterns
    let patterns = vec![
        ("uniform", Box::new(|_: usize| 0.5) as Box<dyn Fn(usize) -> f32>),
        ("linear", Box::new(|i: usize| i as f32 / 1000.0)),
        ("exponential", Box::new(|i: usize| (-(i as f32) / 200.0).exp())),
        ("bimodal", Box::new(|i: usize| if i % 2 == 0 { 0.2 } else { 0.8 })),
        ("normal-like", Box::new(|i: usize| {
            let x = (i as f32 - 500.0) / 100.0;
            (-x * x / 2.0).exp()
        })),
    ];
    
    for (name, pattern) in patterns {
        println!("\n--- Testing {} distribution ---", name);
        
        // Set activations according to pattern
        for i in 0..1000 {
            graph.set_entity_activation(i as u32, pattern(i)).await;
        }
        
        // Get statistics
        let stats = graph.get_brain_statistics().await.unwrap();
        
        println!("Distribution for {}:", name);
        println!("  Mean: {:.3}", stats.avg_activation);
        println!("  Min: {:.3}", stats.min_activation);
        println!("  Max: {:.3}", stats.max_activation);
        
        println!("  Distribution buckets:");
        let mut total_percentage = 0.0;
        for (range, count) in &stats.activation_distribution {
            let percentage = (*count as f32 / 1000.0) * 100.0;
            total_percentage += percentage;
            println!("    {}: {} ({:.1}%)", range, count, percentage);
        }
        
        // Verify distribution sums to 100%
        assert!((total_percentage - 100.0).abs() < 0.1, 
            "Distribution percentages should sum to ~100%");
        
        // Pattern-specific assertions
        match name {
            "uniform" => {
                assert!((stats.avg_activation - 0.5).abs() < 0.01, 
                    "Uniform should have mean ~0.5");
            },
            "linear" => {
                assert!((stats.avg_activation - 0.5).abs() < 0.01,
                    "Linear should have mean ~0.5");
            },
            "exponential" => {
                assert!(stats.avg_activation < 0.3,
                    "Exponential should have low mean");
            },
            "bimodal" => {
                assert!((stats.avg_activation - 0.5).abs() < 0.01,
                    "Bimodal should have mean ~0.5");
                // Check that middle buckets have fewer entries
                let middle_range = stats.activation_distribution.iter()
                    .find(|(range, _)| range.contains("0.4-0.6"))
                    .map(|(_, count)| *count)
                    .unwrap_or(0);
                assert!(middle_range < 100, "Bimodal should have few middle values");
            },
            _ => {}
        }
    }
}

#[tokio::test]
async fn test_performance_scaling_analytics() {
    println!("\n=== Analytics Performance Scaling Test ===");
    
    let sizes = vec![100, 500, 1000, 2500, 5000];
    let mut results = Vec::new();
    
    for size in sizes {
        println!("\n--- Graph size: {} nodes ---", size);
        
        let graph = create_structured_graph(size, 6, 0.1).await;
        
        // Add activation
        for i in 0..size {
            graph.set_entity_activation(i as u32, rand::random()).await;
        }
        
        // Add some concepts
        for i in 0..10 {
            let concept_size = size / 20;
            let start = i * concept_size;
            let entities: Vec<u32> = (start..start + concept_size).map(|e| e as u32).collect();
            graph.create_concept_structure(
                format!("perf_concept_{}", i),
                entities,
                0.8
            ).await.unwrap();
        }
        
        // Measure analytics computation time
        let start_time = Instant::now();
        let stats = graph.get_brain_statistics().await.unwrap();
        let analytics_time = start_time.elapsed();
        
        println!("Analytics computed in {:?}", analytics_time);
        println!("Metrics computed:");
        println!("  Clustering coefficient: {:.4}", stats.clustering_coefficient);
        println!("  Average path length: {:.4}", stats.average_path_length);
        println!("  Betweenness centrality nodes: {}", stats.betweenness_centrality.len());
        
        results.push((size, analytics_time));
        
        // Performance assertion - should scale sub-quadratically
        let expected_max_ms = (size as f64).powf(1.5) / 10.0;
        assert!(analytics_time.as_millis() < expected_max_ms as u128,
            "Analytics should complete in reasonable time for size {}", size);
    }
    
    // Analyze scaling behavior
    println!("\n=== Scaling Analysis ===");
    println!("Size\tTime(ms)\tTime/Node(μs)");
    for (size, time) in &results {
        let time_per_node = time.as_micros() as f64 / *size as f64;
        println!("{}\t{}\t{:.2}", size, time.as_millis(), time_per_node);
    }
    
    // Verify sub-quadratic scaling
    if results.len() >= 2 {
        let (size1, time1) = results[0];
        let (size2, time2) = results[results.len() - 1];
        
        let size_ratio = size2 as f64 / size1 as f64;
        let time_ratio = time2.as_secs_f64() / time1.as_secs_f64();
        let scaling_exponent = time_ratio.log2() / size_ratio.log2();
        
        println!("\nScaling exponent: {:.2}", scaling_exponent);
        assert!(scaling_exponent < 2.0, "Should scale better than O(n²)");
    }
}