//! Graph RAG Unit Tests
//!
//! Comprehensive tests for Graph RAG context assembly, similarity search integration,
//! context quality metrics, multi-strategy approaches, and performance characteristics.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::query::rag::{GraphRagEngine, RagParameters, RagStrategy, ContextQuality};
use crate::core::{Entity, EntityKey, KnowledgeGraph};
use crate::embedding::store::EmbeddingStore;
use std::time::Duration;

#[cfg(test)]
mod rag_tests {
    use super::*;

    #[test]
    fn test_graph_rag_context_assembly() {
        // Create test graph with known structure
        let (graph, embeddings) = create_academic_paper_graph(100, 200);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Test query with known expected context
        let query_entity = EntityKey::from_hash("paper_central");
        let rag_params = RagParameters {
            max_context_entities: 10,
            max_graph_depth: 2,
            similarity_threshold: 0.7,
            diversity_factor: 0.3,
        };
        
        let context = rag_engine.assemble_context(query_entity, &rag_params);
        
        // Verify context properties
        assert!(context.entities.len() <= rag_params.max_context_entities);
        assert!(!context.entities.is_empty());
        assert!(context.entities.contains(&query_entity));
        
        // Verify all entities are within graph distance limit
        for &entity in &context.entities {
            if entity != query_entity {
                let distance = graph.shortest_path_length(query_entity, entity);
                assert!(distance.is_some(), "Context entity not reachable: {:?}", entity);
                assert!(distance.unwrap() <= rag_params.max_graph_depth,
                       "Context entity too far: distance {}", distance.unwrap());
            }
        }
        
        // Verify relevance scores are reasonable
        assert_eq!(context.relevance_scores.len(), context.entities.len());
        for &score in &context.relevance_scores {
            assert!(score >= 0.0 && score <= 1.0, "Invalid relevance score: {}", score);
        }
        
        // Verify diversity in context
        let diversity_score = calculate_context_diversity(&context, &embeddings);
        assert!(diversity_score >= rag_params.diversity_factor * 0.8,
               "Insufficient context diversity: {}", diversity_score);
    }
    
    #[test]
    fn test_rag_similarity_search_integration() {
        let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(500, 1000);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Test with embedding-based query
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4]; // Known test embedding
        let similarity_results = rag_engine.similarity_search(&query_embedding, 20);
        
        assert_eq!(similarity_results.len(), 20);
        
        // Verify results are sorted by similarity (descending)
        for window in similarity_results.windows(2) {
            assert!(window[0].similarity >= window[1].similarity,
                   "Similarity results not properly sorted");
        }
        
        // Verify all similarities are valid
        for result in &similarity_results {
            assert!(result.similarity >= 0.0 && result.similarity <= 1.0,
                   "Invalid similarity score: {}", result.similarity);
        }
        
        // Test integration with graph context expansion
        let top_similar_entity = similarity_results[0].entity;
        let context = rag_engine.expand_from_similarity_seed(
            top_similar_entity, 
            &RagParameters::default()
        );
        
        // Context should include the similar entity
        assert!(context.entities.contains(&top_similar_entity));
        
        // Context should be expanded beyond just similar entities
        assert!(context.entities.len() > similarity_results.len() / 2);
    }
    
    #[test]
    fn test_rag_context_quality_metrics() {
        let (graph, embeddings) = create_test_knowledge_graph_with_clusters(200, 400, 5);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Query from a known cluster center
        let cluster_center = EntityKey::from_hash("cluster_0_center");
        let context = rag_engine.assemble_context(cluster_center, &RagParameters::default());
        
        // Calculate quality metrics
        let quality_metrics = rag_engine.evaluate_context_quality(&context, cluster_center);
        
        // Test coverage metric
        assert!(quality_metrics.coverage >= 0.0 && quality_metrics.coverage <= 1.0);
        
        // For a cluster center, coverage should be high
        assert!(quality_metrics.coverage >= 0.7,
               "Low coverage for cluster center query: {}", quality_metrics.coverage);
        
        // Test diversity metric
        assert!(quality_metrics.diversity >= 0.0 && quality_metrics.diversity <= 1.0);
        
        // Test relevance metric
        assert!(quality_metrics.relevance >= 0.0 && quality_metrics.relevance <= 1.0);
        
        // For a well-formed cluster, relevance should be high
        assert!(quality_metrics.relevance >= 0.6,
               "Low relevance for cluster query: {}", quality_metrics.relevance);
        
        // Test coherence metric (semantic consistency)
        assert!(quality_metrics.coherence >= 0.0 && quality_metrics.coherence <= 1.0);
        
        // Test novelty metric (information richness)
        assert!(quality_metrics.novelty >= 0.0 && quality_metrics.novelty <= 1.0);
        
        // Verify metric relationships make sense
        // High relevance + high coverage should generally correlate with high coherence
        if quality_metrics.relevance > 0.8 && quality_metrics.coverage > 0.8 {
            assert!(quality_metrics.coherence > 0.5,
                   "High relevance/coverage should yield reasonable coherence");
        }
    }
    
    #[test]
    fn test_rag_multi_strategy_integration() {
        let (graph, embeddings) = create_complex_test_graph(300, 600);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        let query_entity = EntityKey::from_hash("multi_strategy_test");
        
        // Test different retrieval strategies
        let embedding_strategy = RagStrategy::EmbeddingSimilarity {
            similarity_threshold: 0.8,
            max_candidates: 15,
        };
        
        let graph_strategy = RagStrategy::GraphTraversal {
            max_depth: 3,
            relationship_weights: true,
        };
        
        let hybrid_strategy = RagStrategy::Hybrid {
            embedding_weight: 0.6,
            graph_weight: 0.4,
            max_context_size: 12,
        };
        
        // Execute different strategies
        let embedding_context = rag_engine.assemble_context_with_strategy(
            query_entity, &embedding_strategy
        );
        let graph_context = rag_engine.assemble_context_with_strategy(
            query_entity, &graph_strategy  
        );
        let hybrid_context = rag_engine.assemble_context_with_strategy(
            query_entity, &hybrid_strategy
        );
        
        // Verify strategy differences
        assert_ne!(embedding_context.entities, graph_context.entities);
        
        // Hybrid should combine aspects of both
        let embedding_overlap = calculate_set_overlap(&hybrid_context.entities, &embedding_context.entities);
        let graph_overlap = calculate_set_overlap(&hybrid_context.entities, &graph_context.entities);
        
        assert!(embedding_overlap > 0.2, "Hybrid strategy should include embedding results");
        assert!(graph_overlap > 0.2, "Hybrid strategy should include graph results");
        
        // Compare context quality across strategies
        let embedding_quality = rag_engine.evaluate_context_quality(&embedding_context, query_entity);
        let graph_quality = rag_engine.evaluate_context_quality(&graph_context, query_entity);
        let hybrid_quality = rag_engine.evaluate_context_quality(&hybrid_context, query_entity);
        
        // Hybrid should generally perform well across multiple metrics
        assert!(hybrid_quality.overall_score >= 
               (embedding_quality.overall_score * 0.8).max(graph_quality.overall_score * 0.8),
               "Hybrid strategy should be competitive with specialized strategies");
    }
    
    #[test]
    fn test_rag_performance_characteristics() {
        let sizes = vec![100, 500, 1000, 2000];
        
        for &size in &sizes {
            let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(size, size * 2);
            let rag_engine = GraphRagEngine::new(graph, embeddings);
            
            let query_entity = EntityKey::from_hash(&format!("perf_test_{}", size));
            
            // Measure context assembly time
            let start_time = std::time::Instant::now();
            let context = rag_engine.assemble_context(query_entity, &RagParameters::default());
            let assembly_time = start_time.elapsed();
            
            println!("RAG context assembly for {} entities: {:?}", size, assembly_time);
            
            // Should scale sub-linearly with graph size
            let time_per_entity = assembly_time.as_nanos() as f64 / size as f64;
            
            // For size = 100, establish baseline
            if size == 100 {
                assert!(assembly_time < Duration::from_millis(50),
                       "RAG assembly too slow for small graph: {:?}", assembly_time);
            }
            
            // For larger sizes, should not grow linearly
            if size >= 1000 {
                assert!(assembly_time < Duration::from_millis(500),
                       "RAG assembly too slow for graph size {}: {:?}", size, assembly_time);
            }
            
            // Verify context quality doesn't degrade with size
            let quality = rag_engine.evaluate_context_quality(&context, query_entity);
            assert!(quality.overall_score >= 0.5,
                   "Context quality degraded for size {}: {}", size, quality.overall_score);
            
            // Memory usage should be reasonable
            let memory_usage = rag_engine.memory_usage();
            let memory_per_entity = memory_usage / size as u64;
            
            assert!(memory_per_entity < 1000, // < 1KB per entity
                   "Memory usage too high: {} bytes per entity", memory_per_entity);
        }
    }

    #[test]
    fn test_rag_context_caching() {
        let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(200, 400);
        let mut rag_engine = GraphRagEngine::new(graph, embeddings);
        rag_engine.enable_caching(true);
        
        let query_entity = EntityKey::from_hash("cache_test");
        let params = RagParameters::default();
        
        // First context assembly (cache miss)
        let (context1, time1) = measure_execution_time(|| {
            rag_engine.assemble_context(query_entity, &params)
        });
        
        // Second context assembly (cache hit)
        let (context2, time2) = measure_execution_time(|| {
            rag_engine.assemble_context(query_entity, &params)
        });
        
        // Results should be identical
        assert_eq!(context1.entities, context2.entities);
        assert_vectors_equal(&context1.relevance_scores, &context2.relevance_scores, 1e-6);
        
        // Cache hit should be significantly faster
        let speedup = time1.as_nanos() as f64 / time2.as_nanos() as f64;
        assert!(speedup > 5.0, "Cache not providing expected speedup: {:.2}x", speedup);
        
        // Test cache invalidation
        rag_engine.invalidate_cache_for_entity(query_entity);
        
        let (context3, time3) = measure_execution_time(|| {
            rag_engine.assemble_context(query_entity, &params)
        });
        
        // Should take longer again (cache miss)
        assert!(time3 > time2 * 2, "Cache invalidation not working");
        assert_eq!(context1.entities, context3.entities); // Results still identical
    }

    #[test]
    fn test_rag_edge_cases() {
        let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(50, 100);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Test with non-existent entity
        let non_existent = EntityKey::from_hash("non_existent");
        let context = rag_engine.assemble_context(non_existent, &RagParameters::default());
        assert!(context.entities.is_empty(), "Should return empty context for non-existent entity");
        
        // Test with very restrictive parameters
        let restrictive_params = RagParameters {
            max_context_entities: 1,
            max_graph_depth: 0,
            similarity_threshold: 0.99,
            diversity_factor: 1.0,
        };
        
        let query_entity = EntityKey::from_hash("test_entity_0");
        let restrictive_context = rag_engine.assemble_context(query_entity, &restrictive_params);
        
        assert!(restrictive_context.entities.len() <= 1);
        if !restrictive_context.entities.is_empty() {
            assert_eq!(restrictive_context.entities[0], query_entity);
        }
        
        // Test with very permissive parameters
        let permissive_params = RagParameters {
            max_context_entities: 1000,
            max_graph_depth: 10,
            similarity_threshold: 0.0,
            diversity_factor: 0.0,
        };
        
        let permissive_context = rag_engine.assemble_context(query_entity, &permissive_params);
        assert!(permissive_context.entities.len() > restrictive_context.entities.len());
        
        // Test with empty similarity query
        let empty_embedding = vec![];
        let empty_results = rag_engine.similarity_search(&empty_embedding, 10);
        assert!(empty_results.is_empty(), "Empty embedding should return no results");
        
        // Test with mismatched embedding dimension
        let wrong_dim_embedding = vec![0.5; 256]; // Assuming embeddings are different dimension
        let wrong_dim_results = rag_engine.similarity_search(&wrong_dim_embedding, 10);
        // Should either handle gracefully or return empty results
        assert!(wrong_dim_results.is_empty() || wrong_dim_results.len() <= 10);
    }

    #[test]
    fn test_rag_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(200, 400);
        let rag_engine = Arc::new(GraphRagEngine::new(graph, embeddings));
        
        let thread_count = 4;
        let queries_per_thread = 50;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let engine_clone = Arc::clone(&rag_engine);
            
            let handle = thread::spawn(move || {
                let mut successful_queries = 0;
                
                for query_id in 0..queries_per_thread {
                    let query_entity = EntityKey::from_hash(&format!("thread_{}_query_{}", thread_id, query_id));
                    let context = engine_clone.assemble_context(query_entity, &RagParameters::default());
                    
                    // Context might be empty for non-existent entities, but call should succeed
                    successful_queries += 1;
                }
                
                successful_queries
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads and verify all queries succeeded
        for (thread_id, handle) in handles.into_iter().enumerate() {
            let successful_count = handle.join().unwrap();
            assert_eq!(successful_count, queries_per_thread, 
                      "Thread {} had failed queries", thread_id);
        }
    }

    #[test]
    fn test_rag_memory_efficiency() {
        let sizes = vec![100, 500, 1000];
        
        for &size in &sizes {
            let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(size, size * 2);
            let rag_engine = GraphRagEngine::new(graph, embeddings);
            
            let initial_memory = rag_engine.memory_usage();
            
            // Perform multiple queries
            for i in 0..20 {
                let query_entity = EntityKey::from_hash(&format!("memory_test_{}", i));
                let _context = rag_engine.assemble_context(query_entity, &RagParameters::default());
            }
            
            let final_memory = rag_engine.memory_usage();
            let memory_growth = final_memory - initial_memory;
            
            println!("Memory growth for size {}: {} bytes", size, memory_growth);
            
            // Memory growth should be minimal (caching might cause some growth)
            let growth_per_entity = memory_growth as f64 / size as f64;
            assert!(growth_per_entity < 100.0, 
                   "Excessive memory growth: {:.2} bytes per entity", growth_per_entity);
            
            // Total memory should scale reasonably
            let memory_per_entity = final_memory as f64 / size as f64;
            assert!(memory_per_entity < 2000.0, 
                   "Memory usage too high: {:.2} bytes per entity", memory_per_entity);
        }
    }
}

// Helper functions for RAG tests
fn create_academic_paper_graph(entity_count: usize, relationship_count: usize) -> (KnowledgeGraph, EmbeddingStore) {
    let mut rng = DeterministicRng::new(RAG_TEST_SEED);
    let mut graph = KnowledgeGraph::new();
    let embedding_dim = 128;
    let mut embeddings = EmbeddingStore::new(embedding_dim);
    
    // Create paper entities with a central hub
    let central_key = EntityKey::from_hash("paper_central");
    let central_entity = Entity::new(central_key, "Central Paper".to_string());
    graph.add_entity(central_entity).unwrap();
    
    let central_embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
    embeddings.add_embedding("paper_central", central_embedding).unwrap();
    
    // Create other entities
    for i in 0..entity_count-1 {
        let entity_key = EntityKey::from_hash(&format!("paper_{}", i));
        let entity = Entity::new(entity_key, format!("Paper {}", i));
        graph.add_entity(entity).unwrap();
        
        let embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        embeddings.add_embedding(&format!("paper_{}", i), embedding).unwrap();
    }
    
    // Add relationships with preference for central node
    for _ in 0..relationship_count {
        let use_central = rng.gen::<f64>() < 0.3; // 30% chance to connect to central
        
        let (source_idx, target_idx) = if use_central {
            if rng.gen::<bool>() {
                (entity_count, rng.gen_range(0..entity_count-1)) // Central to other
            } else {
                (rng.gen_range(0..entity_count-1), entity_count) // Other to central
            }
        } else {
            let s = rng.gen_range(0..entity_count-1);
            let t = rng.gen_range(0..entity_count-1);
            (s, t)
        };
        
        if source_idx != target_idx {
            let source_key = if source_idx == entity_count {
                central_key
            } else {
                EntityKey::from_hash(&format!("paper_{}", source_idx))
            };
            
            let target_key = if target_idx == entity_count {
                central_key
            } else {
                EntityKey::from_hash(&format!("paper_{}", target_idx))
            };
            
            let relationship = crate::core::Relationship::new(
                "cites".to_string(),
                rng.gen_range(0.5..1.0),
                crate::core::RelationshipType::Directed
            );
            
            let _ = graph.add_relationship(source_key, target_key, relationship);
        }
    }
    
    (graph, embeddings)
}

fn create_test_knowledge_graph_with_embeddings(entity_count: usize, relationship_count: usize) -> (KnowledgeGraph, EmbeddingStore) {
    let mut rng = DeterministicRng::new(RAG_TEST_SEED);
    let graph = create_test_graph(entity_count, relationship_count);
    let embedding_dim = 64;
    let mut embeddings = EmbeddingStore::new(embedding_dim);
    
    // Add embeddings for all entities
    for i in 0..entity_count {
        let embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        embeddings.add_embedding(&format!("test_entity_{}", i), embedding).unwrap();
    }
    
    (graph, embeddings)
}

fn create_test_knowledge_graph_with_clusters(entity_count: usize, relationship_count: usize, cluster_count: usize) -> (KnowledgeGraph, EmbeddingStore) {
    let mut rng = DeterministicRng::new(RAG_TEST_SEED);
    let mut graph = KnowledgeGraph::new();
    let embedding_dim = 64;
    let mut embeddings = EmbeddingStore::new(embedding_dim);
    
    let entities_per_cluster = entity_count / cluster_count;
    
    // Generate cluster centers
    let cluster_centers: Vec<Vec<f32>> = (0..cluster_count)
        .map(|_| (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    // Create entities in clusters
    for cluster_id in 0..cluster_count {
        for entity_id in 0..entities_per_cluster {
            let global_id = cluster_id * entities_per_cluster + entity_id;
            let entity_key = if entity_id == 0 {
                EntityKey::from_hash(&format!("cluster_{}_center", cluster_id))
            } else {
                EntityKey::from_hash(&format!("cluster_{}_{}", cluster_id, entity_id))
            };
            
            let entity_name = if entity_id == 0 {
                format!("Cluster {} Center", cluster_id)
            } else {
                format!("Entity {} in Cluster {}", entity_id, cluster_id)
            };
            
            let entity = Entity::new(entity_key, entity_name);
            graph.add_entity(entity).unwrap();
            
            // Create embedding near cluster center
            let center = &cluster_centers[cluster_id];
            let embedding: Vec<f32> = center.iter()
                .map(|&c| c + rng.gen_range(-0.2..0.2))
                .collect();
            
            let embedding_name = if entity_id == 0 {
                format!("cluster_{}_center", cluster_id)
            } else {
                format!("cluster_{}_{}", cluster_id, entity_id)
            };
            
            embeddings.add_embedding(&embedding_name, embedding).unwrap();
        }
    }
    
    // Add relationships within and between clusters
    for _ in 0..relationship_count {
        let source_cluster = rng.gen_range(0..cluster_count);
        let target_cluster = if rng.gen::<f64>() < 0.7 { 
            source_cluster // 70% intra-cluster
        } else { 
            rng.gen_range(0..cluster_count) // 30% inter-cluster
        };
        
        let source_entity = rng.gen_range(0..entities_per_cluster);
        let target_entity = rng.gen_range(0..entities_per_cluster);
        
        let source_key = if source_entity == 0 {
            EntityKey::from_hash(&format!("cluster_{}_center", source_cluster))
        } else {
            EntityKey::from_hash(&format!("cluster_{}_{}", source_cluster, source_entity))
        };
        
        let target_key = if target_entity == 0 {
            EntityKey::from_hash(&format!("cluster_{}_center", target_cluster))
        } else {
            EntityKey::from_hash(&format!("cluster_{}_{}", target_cluster, target_entity))
        };
        
        if source_key != target_key {
            let relationship = crate::core::Relationship::new(
                "relates_to".to_string(),
                rng.gen_range(0.3..1.0),
                crate::core::RelationshipType::Undirected
            );
            
            let _ = graph.add_relationship(source_key, target_key, relationship);
        }
    }
    
    (graph, embeddings)
}

fn create_complex_test_graph(entity_count: usize, relationship_count: usize) -> (KnowledgeGraph, EmbeddingStore) {
    // Create a more complex graph with multiple types of structures
    let mut rng = DeterministicRng::new(RAG_TEST_SEED);
    let mut graph = KnowledgeGraph::new();
    let embedding_dim = 96;
    let mut embeddings = EmbeddingStore::new(embedding_dim);
    
    // Create test entity
    let test_key = EntityKey::from_hash("multi_strategy_test");
    let test_entity = Entity::new(test_key, "Multi Strategy Test Entity".to_string());
    graph.add_entity(test_entity).unwrap();
    
    let test_embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
    embeddings.add_embedding("multi_strategy_test", test_embedding).unwrap();
    
    // Create other entities with various connection patterns
    for i in 0..entity_count-1 {
        let entity_key = EntityKey::from_hash(&format!("complex_entity_{}", i));
        let entity = Entity::new(entity_key, format!("Complex Entity {}", i));
        graph.add_entity(entity).unwrap();
        
        let embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        embeddings.add_embedding(&format!("complex_entity_{}", i), embedding).unwrap();
    }
    
    // Add relationships with various patterns
    for _ in 0..relationship_count {
        let source_idx = rng.gen_range(0..entity_count);
        let target_idx = rng.gen_range(0..entity_count);
        
        if source_idx != target_idx {
            let source_key = if source_idx == 0 {
                test_key
            } else {
                EntityKey::from_hash(&format!("complex_entity_{}", source_idx - 1))
            };
            
            let target_key = if target_idx == 0 {
                test_key
            } else {
                EntityKey::from_hash(&format!("complex_entity_{}", target_idx - 1))
            };
            
            let relationship = crate::core::Relationship::new(
                "complex_relation".to_string(),
                rng.gen_range(0.1..1.0),
                crate::core::RelationshipType::Directed
            );
            
            let _ = graph.add_relationship(source_key, target_key, relationship);
        }
    }
    
    (graph, embeddings)
}

fn calculate_context_diversity(context: &RagContext, embeddings: &EmbeddingStore) -> f64 {
    if context.entities.len() < 2 {
        return 1.0;
    }
    
    let mut total_distance = 0.0;
    let mut pair_count = 0;
    
    for i in 0..context.entities.len() {
        for j in (i+1)..context.entities.len() {
            if let (Some(emb1), Some(emb2)) = (
                embeddings.get_embedding_by_name(&format!("entity_{}", i)),
                embeddings.get_embedding_by_name(&format!("entity_{}", j))
            ) {
                let distance = euclidean_distance(emb1, emb2);
                total_distance += distance as f64;
                pair_count += 1;
            }
        }
    }
    
    if pair_count > 0 {
        total_distance / pair_count as f64
    } else {
        0.0
    }
}

fn calculate_set_overlap<T: PartialEq>(set1: &[T], set2: &[T]) -> f64 {
    if set1.is_empty() || set2.is_empty() {
        return 0.0;
    }
    
    let overlap_count = set1.iter().filter(|item| set2.contains(item)).count();
    overlap_count as f64 / set1.len().max(set2.len()) as f64
}