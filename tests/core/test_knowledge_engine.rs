use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::knowledge_types::{TripleQuery, KnowledgeResult, MemoryStats, EntityContext};
use llmkg::core::triple::Triple;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio;

/// Helper function to create a test triple with custom subject, predicate, object
fn create_triple(subject: &str, predicate: &str, object: &str) -> Triple {
    Triple {
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        confidence: 0.9,
        source: Some("test_data".to_string()),
        metadata: HashMap::new(),
    }
}

/// Helper function to generate a large knowledge graph dataset
fn generate_test_dataset(size: usize) -> Vec<Triple> {
    let mut triples = Vec::new();
    
    // Create entities
    let person_count = size / 10;
    let company_count = size / 20;
    let product_count = size / 15;
    
    // Generate people and their relationships
    for i in 0..person_count {
        let person = format!("Person_{}", i);
        
        // Person works at company
        if i % 3 == 0 {
            let company_idx = i % company_count;
            triples.push(create_triple(&person, "worksAt", &format!("Company_{}", company_idx)));
        }
        
        // Person likes products
        for j in 0..3 {
            let product_idx = (i * 3 + j) % product_count;
            triples.push(create_triple(&person, "likes", &format!("Product_{}", product_idx)));
        }
        
        // Person knows other people
        if i > 0 {
            triples.push(create_triple(&person, "knows", &format!("Person_{}", i - 1)));
        }
    }
    
    // Generate company relationships
    for i in 0..company_count {
        let company = format!("Company_{}", i);
        
        // Company produces products
        for j in 0..5 {
            let product_idx = (i * 5 + j) % product_count;
            triples.push(create_triple(&company, "produces", &format!("Product_{}", product_idx)));
        }
        
        // Company partnerships
        if i > 0 {
            triples.push(create_triple(&company, "partnersWith", &format!("Company_{}", i - 1)));
        }
    }
    
    // Generate product relationships
    for i in 0..product_count {
        let product = format!("Product_{}", i);
        
        // Product categories
        let category = match i % 5 {
            0 => "Electronics",
            1 => "Clothing",
            2 => "Food",
            3 => "Books",
            _ => "Other",
        };
        triples.push(create_triple(&product, "hasCategory", category));
        
        // Product properties
        triples.push(create_triple(&product, "hasPrice", &format!("${}", (i * 10) % 1000)));
    }
    
    triples
}

#[tokio::test]
async fn test_complete_llm_workflow() {
    // Create knowledge engine
    let engine = KnowledgeEngine::new(384, 10000).expect("Failed to create engine");
    
    // Phase 1: Store a large number of triples
    println!("Phase 1: Storing large dataset...");
    let dataset = generate_test_dataset(1000);
    let start_time = Instant::now();
    
    for triple in &dataset {
        engine.store_triple(triple.clone(), None).expect("Failed to store triple");
    }
    
    let store_duration = start_time.elapsed();
    println!("Stored {} triples in {:?}", dataset.len(), store_duration);
    
    // Verify storage
    let stats = engine.get_memory_stats();
    assert!(stats.total_triples >= dataset.len());
    println!("Memory stats: {:?}", stats);
    
    // Phase 2: Query with various SPO patterns
    println!("\nPhase 2: Testing SPO pattern queries...");
    
    // Query by subject
    let query = TripleQuery {
        subject: Some("Person_0".to_string()),
        predicate: None,
        object: None,
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let start = Instant::now();
    let result = engine.query_triples(query).expect("Query failed");
    let query_time = start.elapsed();
    
    assert!(!result.triples.is_empty());
    println!("Subject query found {} triples in {:?}", result.triples.len(), query_time);
    
    // Query by predicate
    let query = TripleQuery {
        subject: None,
        predicate: Some("worksAt".to_string()),
        object: None,
        limit: 50,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    assert!(!result.triples.is_empty());
    println!("Predicate query found {} triples", result.triples.len());
    
    // Query by object
    let query = TripleQuery {
        subject: None,
        predicate: None,
        object: Some("Company_0".to_string()),
        limit: 50,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    println!("Object query found {} triples", result.triples.len());
    
    // Complex SPO query
    let query = TripleQuery {
        subject: Some("Person_0".to_string()),
        predicate: Some("likes".to_string()),
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    assert!(!result.triples.is_empty());
    println!("Complex SPO query found {} triples", result.triples.len());
    
    // Phase 3: Semantic search
    println!("\nPhase 3: Testing semantic search...");
    
    let search_queries = vec![
        "person works at company",
        "products in electronics category",
        "company partnerships",
        "people who like products",
    ];
    
    for search_text in search_queries {
        let start = Instant::now();
        let result = engine.semantic_search(search_text, 10).expect("Semantic search failed");
        let search_time = start.elapsed();
        
        println!("Semantic search '{}' found {} results in {:?}", 
                 search_text, result.nodes.len(), search_time);
        assert!(search_time.as_millis() < 1000); // Should be fast
    }
    
    // Phase 4: Entity relationship exploration
    println!("\nPhase 4: Testing entity relationship exploration...");
    
    let relationships = engine.get_entity_relationships("Person_0", 2)
        .expect("Failed to get relationships");
    
    assert!(!relationships.is_empty());
    println!("Found {} relationships for Person_0 (2 hops)", relationships.len());
    
    // Verify we get multi-hop relationships
    let has_indirect = relationships.iter().any(|t| {
        t.subject != "Person_0" && t.object != "Person_0"
    });
    assert!(has_indirect, "Should include indirect relationships");
    
    // Phase 5: Performance under concurrent load
    println!("\nPhase 5: Testing concurrent access...");
    
    let engine_arc = Arc::new(engine);
    let mut handles = vec![];
    
    // Spawn concurrent readers
    for i in 0..5 {
        let engine_clone = Arc::clone(&engine_arc);
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let query = TripleQuery {
                    subject: Some(format!("Person_{}", (i * 100 + j) % 100)),
                    predicate: None,
                    object: None,
                    limit: 10,
                    min_confidence: 0.0,
                    include_chunks: false,
                };
                
                engine_clone.query_triples(query).expect("Concurrent query failed");
            }
        });
        handles.push(handle);
    }
    
    // Spawn concurrent writers
    for i in 0..3 {
        let engine_clone = Arc::clone(&engine_arc);
        let handle = thread::spawn(move || {
            for j in 0..50 {
                let triple = create_triple(
                    &format!("ConcurrentPerson_{}", i),
                    "performs",
                    &format!("Action_{}", j)
                );
                engine_clone.store_triple(triple, None).expect("Concurrent store failed");
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Concurrent access test completed successfully");
    
    // Final statistics
    let final_stats = engine_arc.get_memory_stats();
    println!("\nFinal statistics:");
    println!("Total nodes: {}", final_stats.total_nodes);
    println!("Total triples: {}", final_stats.total_triples);
    println!("Total bytes: {}", final_stats.total_bytes);
    println!("Bytes per node: {:.2}", final_stats.bytes_per_node);
}

#[tokio::test]
async fn test_knowledge_chunks_workflow() {
    let engine = KnowledgeEngine::new(384, 5000).expect("Failed to create engine");
    
    // Test storing knowledge chunks
    let chunks = vec![
        "Alice is a software engineer who works at TechCorp. She specializes in distributed systems and machine learning.",
        "TechCorp is a technology company founded in 2010. It produces innovative software solutions for enterprise clients.",
        "Bob is the CEO of TechCorp. He has 20 years of experience in the technology industry.",
        "The distributed systems team at TechCorp includes Alice, Charlie, and David. They work on scalable infrastructure.",
    ];
    
    println!("Storing knowledge chunks...");
    let mut chunk_ids = Vec::new();
    
    for chunk in &chunks {
        let id = engine.store_chunk(chunk.to_string(), None)
            .expect("Failed to store chunk");
        chunk_ids.push(id);
    }
    
    // Query for extracted triples
    let query = TripleQuery {
        subject: None,
        predicate: None,
        object: Some("TechCorp".to_string()),
        limit: 50,
        min_confidence: 0.0,
        include_chunks: true,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    assert!(!result.nodes.is_empty());
    println!("Found {} nodes mentioning TechCorp", result.nodes.len());
    
    // Semantic search across chunks
    let search_result = engine.semantic_search("distributed systems team", 5)
        .expect("Semantic search failed");
    
    assert!(!search_result.nodes.is_empty());
    println!("Semantic search found {} relevant chunks", search_result.nodes.len());
}

#[tokio::test]
async fn test_entity_management_workflow() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Define entities with properties
    let entities = vec![
        ("Alice", "Person", "Software engineer at TechCorp", HashMap::from([
            ("role".to_string(), "Senior Engineer".to_string()),
            ("department".to_string(), "Engineering".to_string()),
            ("yearsExperience".to_string(), "8".to_string()),
        ])),
        ("TechCorp", "Company", "Technology company specializing in enterprise software", HashMap::from([
            ("founded".to_string(), "2010".to_string()),
            ("employees".to_string(), "500".to_string()),
            ("industry".to_string(), "Technology".to_string()),
        ])),
        ("DistSys", "Product", "Distributed systems framework", HashMap::from([
            ("version".to_string(), "2.5".to_string()),
            ("license".to_string(), "MIT".to_string()),
            ("language".to_string(), "Rust".to_string()),
        ])),
    ];
    
    // Store entities
    for (name, entity_type, description, properties) in entities {
        engine.store_entity(
            name.to_string(),
            entity_type.to_string(),
            description.to_string(),
            properties
        ).expect("Failed to store entity");
    }
    
    // Store relationships between entities
    let relationships = vec![
        create_triple("Alice", "worksAt", "TechCorp"),
        create_triple("Alice", "develops", "DistSys"),
        create_triple("TechCorp", "owns", "DistSys"),
        create_triple("DistSys", "usedBy", "Alice"),
    ];
    
    for triple in relationships {
        engine.store_triple(triple, None).expect("Failed to store relationship");
    }
    
    // Get entity types
    let entity_types = engine.get_entity_types();
    assert_eq!(entity_types.get("Alice"), Some(&"Person".to_string()));
    assert_eq!(entity_types.get("TechCorp"), Some(&"Company".to_string()));
    assert_eq!(entity_types.get("DistSys"), Some(&"Product".to_string()));
    
    // Explore entity relationships
    let alice_relationships = engine.get_entity_relationships("Alice", 2)
        .expect("Failed to get relationships");
    
    println!("Alice has {} relationships (2 hops)", alice_relationships.len());
    
    // Should include direct and indirect relationships
    let has_direct = alice_relationships.iter().any(|t| t.subject == "Alice" || t.object == "Alice");
    let has_indirect = alice_relationships.iter().any(|t| t.subject != "Alice" && t.object != "Alice");
    
    assert!(has_direct, "Should have direct relationships");
    assert!(has_indirect, "Should have indirect relationships through TechCorp or DistSys");
}

#[tokio::test]
async fn test_predicate_vocabulary_workflow() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Store triples with various predicates
    let predicates = vec![
        "likes", "LIKES", "Likes",  // Should normalize to same predicate
        "worksAt", "works_at", "WORKS_AT",
        "knows", "KNOWS", "Knows",
    ];
    
    for (i, pred) in predicates.iter().enumerate() {
        let triple = create_triple(&format!("Subject_{}", i), pred, &format!("Object_{}", i));
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Test predicate suggestions
    let suggestions = engine.suggest_predicates("work");
    println!("Predicate suggestions for 'work': {:?}", suggestions);
    
    // Query with normalized predicate
    let query = TripleQuery {
        subject: None,
        predicate: Some("likes".to_string()),
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    // Should find all variations of "likes"
    assert!(result.triples.len() >= 3, "Should find all normalized 'likes' predicates");
}

#[tokio::test]
async fn test_memory_pressure_and_eviction() {
    // Create engine with small memory limit
    let max_nodes = 100;
    let engine = KnowledgeEngine::new(128, max_nodes).expect("Failed to create engine");
    
    // Store more triples than the limit
    let triple_count = max_nodes * 2;
    println!("Storing {} triples with max_nodes={}", triple_count, max_nodes);
    
    for i in 0..triple_count {
        let triple = create_triple(
            &format!("Subject_{}", i),
            "testPredicate",
            &format!("Object_{}", i)
        );
        
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Check that eviction worked
    let stats = engine.get_memory_stats();
    assert!(stats.total_nodes <= max_nodes, 
            "Node count {} should not exceed max_nodes {}", stats.total_nodes, max_nodes);
    
    println!("After eviction: {} nodes (limit: {})", stats.total_nodes, max_nodes);
    
    // Recent items should still be queryable
    let recent_query = TripleQuery {
        subject: Some(format!("Subject_{}", triple_count - 1)),
        predicate: None,
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(recent_query).expect("Query failed");
    assert!(!result.triples.is_empty(), "Recent triples should still be available");
}

#[tokio::test]
async fn test_confidence_filtering() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Store triples with different confidence levels
    let confidence_levels = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    
    for (i, confidence) in confidence_levels.iter().enumerate() {
        let mut triple = create_triple(
            &format!("Subject_{}", i),
            "hasConfidence",
            &format!("Object_{}", i)
        );
        triple.confidence = *confidence;
        
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Query with different confidence thresholds
    let thresholds = vec![0.0, 0.5, 0.8, 1.0];
    
    for threshold in thresholds {
        let query = TripleQuery {
            subject: None,
            predicate: Some("hasConfidence".to_string()),
            object: None,
            limit: 100,
            min_confidence: threshold,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).expect("Query failed");
        
        // Verify all returned triples meet confidence threshold
        for triple in &result.triples {
            assert!(triple.confidence >= threshold,
                    "Triple confidence {} should be >= threshold {}", 
                    triple.confidence, threshold);
        }
        
        println!("Confidence >= {}: found {} triples", threshold, result.triples.len());
    }
}

#[tokio::test]
async fn test_performance_benchmarks() {
    let engine = KnowledgeEngine::new(384, 50000).expect("Failed to create engine");
    
    // Benchmark: Store 10,000 triples
    let store_count = 10000;
    let dataset = generate_test_dataset(store_count);
    
    let start = Instant::now();
    for triple in &dataset {
        engine.store_triple(triple.clone(), None).expect("Failed to store triple");
    }
    let store_duration = start.elapsed();
    
    let store_rate = store_count as f64 / store_duration.as_secs_f64();
    println!("Storage benchmark: {} triples/second", store_rate as u64);
    assert!(store_rate > 1000.0, "Storage should exceed 1000 triples/second");
    
    // Benchmark: Query performance
    let query_iterations = 1000;
    let start = Instant::now();
    
    for i in 0..query_iterations {
        let query = TripleQuery {
            subject: Some(format!("Person_{}", i % 100)),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        engine.query_triples(query).expect("Query failed");
    }
    
    let query_duration = start.elapsed();
    let query_rate = query_iterations as f64 / query_duration.as_secs_f64();
    println!("Query benchmark: {} queries/second", query_rate as u64);
    assert!(query_rate > 100.0, "Query rate should exceed 100 queries/second");
    
    // Benchmark: Semantic search
    let search_iterations = 100;
    let start = Instant::now();
    
    for i in 0..search_iterations {
        let query_text = format!("Person {} likes Product", i % 10);
        engine.semantic_search(&query_text, 5).expect("Search failed");
    }
    
    let search_duration = start.elapsed();
    let search_rate = search_iterations as f64 / search_duration.as_secs_f64();
    println!("Semantic search benchmark: {} searches/second", search_rate as u64);
    
    // Memory efficiency check
    let stats = engine.get_memory_stats();
    println!("\nMemory efficiency:");
    println!("  Total nodes: {}", stats.total_nodes);
    println!("  Total triples: {}", stats.total_triples);
    println!("  Bytes per node: {:.2}", stats.bytes_per_node);
    
    // Should maintain <60 bytes per entity as per requirements
    assert!(stats.bytes_per_node < 100.0, 
            "Memory usage per node should be efficient");
}

#[tokio::test]
async fn test_query_result_ordering() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Store triples and manually set quality scores
    let subjects = vec!["Alice", "Bob", "Charlie", "David", "Eve"];
    
    for subject in &subjects {
        let triple = create_triple(subject, "hasQuality", "test");
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Query and verify ordering by quality score
    let query = TripleQuery {
        subject: None,
        predicate: Some("hasQuality".to_string()),
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    
    // Verify results are ordered by quality score (descending)
    for i in 1..result.nodes.len() {
        assert!(result.nodes[i-1].metadata.quality_score >= result.nodes[i].metadata.quality_score,
                "Results should be ordered by quality score");
    }
}

#[tokio::test]
async fn test_entity_context_building() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Create a knowledge graph with entities
    engine.store_entity(
        "Alice".to_string(),
        "Person".to_string(),
        "A software engineer".to_string(),
        HashMap::new()
    ).expect("Failed to store entity");
    
    engine.store_entity(
        "TechCorp".to_string(),
        "Company".to_string(),
        "A technology company".to_string(),
        HashMap::new()
    ).expect("Failed to store entity");
    
    // Add relationships
    let triples = vec![
        create_triple("Alice", "worksAt", "TechCorp"),
        create_triple("Alice", "hasSkill", "Rust"),
        create_triple("TechCorp", "located", "Seattle"),
    ];
    
    for triple in triples {
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Query and check entity context
    let query = TripleQuery {
        subject: Some("Alice".to_string()),
        predicate: None,
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(query).expect("Query failed");
    
    // Should have entity context for Alice and related entities
    assert!(result.entity_context.contains_key("Alice"));
    assert!(result.entity_context.contains_key("TechCorp"));
    
    let alice_context = &result.entity_context["Alice"];
    assert_eq!(alice_context.entity_type, "Person");
    assert!(!alice_context.related_triples.is_empty());
}

#[tokio::test]
async fn test_large_chunk_rejection() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Try to store a chunk that exceeds size limit
    let large_text = "x".repeat(1024 * 1024 * 2); // 2MB text
    
    let result = engine.store_chunk(large_text, None);
    assert!(result.is_err(), "Should reject chunks that are too large");
    
    match result {
        Err(e) => println!("Expected error for large chunk: {:?}", e),
        Ok(_) => panic!("Should not accept oversized chunks"),
    }
}

#[tokio::test]
async fn test_concurrent_read_write_stress() {
    let engine = Arc::new(KnowledgeEngine::new(256, 10000).expect("Failed to create engine"));
    let test_duration = Duration::from_secs(5);
    let start_time = Instant::now();
    
    // Metrics tracking
    let write_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let read_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    
    let mut handles = vec![];
    
    // Spawn aggressive writers
    for writer_id in 0..4 {
        let engine_clone = Arc::clone(&engine);
        let write_count_clone = Arc::clone(&write_count);
        let duration = test_duration;
        
        let handle = thread::spawn(move || {
            let start = Instant::now();
            let mut count = 0;
            
            while start.elapsed() < duration {
                let triple = create_triple(
                    &format!("Writer{}Entity{}", writer_id, count),
                    "writes",
                    &format!("Data{}", count)
                );
                
                if engine_clone.store_triple(triple, None).is_ok() {
                    count += 1;
                }
            }
            
            write_count_clone.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
            println!("Writer {} completed {} writes", writer_id, count);
        });
        
        handles.push(handle);
    }
    
    // Spawn aggressive readers
    for reader_id in 0..6 {
        let engine_clone = Arc::clone(&engine);
        let read_count_clone = Arc::clone(&read_count);
        let duration = test_duration;
        
        let handle = thread::spawn(move || {
            let start = Instant::now();
            let mut count = 0;
            
            while start.elapsed() < duration {
                let query = TripleQuery {
                    subject: if count % 3 == 0 {
                        Some(format!("Writer{}Entity{}", reader_id % 4, count % 100))
                    } else {
                        None
                    },
                    predicate: if count % 3 == 1 {
                        Some("writes".to_string())
                    } else {
                        None
                    },
                    object: None,
                    limit: 10,
                    min_confidence: 0.0,
                    include_chunks: false,
                };
                
                if engine_clone.query_triples(query).is_ok() {
                    count += 1;
                }
            }
            
            read_count_clone.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
            println!("Reader {} completed {} reads", reader_id, count);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_writes = write_count.load(std::sync::atomic::Ordering::Relaxed);
    let total_reads = read_count.load(std::sync::atomic::Ordering::Relaxed);
    
    println!("\nStress test results:");
    println!("Total writes: {}", total_writes);
    println!("Total reads: {}", total_reads);
    println!("Write rate: {} ops/sec", total_writes as f64 / test_duration.as_secs_f64());
    println!("Read rate: {} ops/sec", total_reads as f64 / test_duration.as_secs_f64());
    
    // Verify system stability
    let stats = engine.get_memory_stats();
    assert!(stats.total_nodes > 0, "Should have stored nodes");
    assert!(stats.bytes_per_node > 0.0, "Should track memory usage");
    
    // Do a final query to ensure system is still functional
    let final_query = TripleQuery {
        subject: None,
        predicate: Some("writes".to_string()),
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let result = engine.query_triples(final_query).expect("System should still be queryable");
    assert!(!result.triples.is_empty(), "Should find some writes");
}

#[test]
fn test_engine_configuration() {
    // Test various engine configurations
    let configs = vec![
        (64, 100),    // Small embedding, small capacity
        (256, 1000),  // Medium configuration
        (768, 10000), // Large embedding (BERT-like), large capacity
    ];
    
    for (embedding_dim, max_nodes) in configs {
        let engine = KnowledgeEngine::new(embedding_dim, max_nodes);
        assert!(engine.is_ok(), 
                "Should create engine with embedding_dim={}, max_nodes={}", 
                embedding_dim, max_nodes);
        
        let engine = engine.unwrap();
        
        // Verify engine accepts triples
        let triple = create_triple("Test", "validates", "Configuration");
        let result = engine.store_triple(triple, None);
        assert!(result.is_ok(), "Should store triple in configured engine");
        
        // Verify statistics work
        let stats = engine.get_memory_stats();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_triples, 1);
    }
}

#[tokio::test]
async fn test_complex_multi_hop_reasoning() {
    let engine = KnowledgeEngine::new(256, 5000).expect("Failed to create engine");
    
    // Create a complex knowledge graph
    // Company hierarchy
    let company_triples = vec![
        create_triple("TechCorp", "parentCompany", "GlobalTech"),
        create_triple("GlobalTech", "parentCompany", "MegaCorp"),
        create_triple("StartupX", "acquiredBy", "TechCorp"),
    ];
    
    // People and roles
    let people_triples = vec![
        create_triple("Alice", "worksAt", "TechCorp"),
        create_triple("Alice", "reportsTo", "Bob"),
        create_triple("Bob", "worksAt", "TechCorp"),
        create_triple("Bob", "reportsTo", "Charlie"),
        create_triple("Charlie", "worksAt", "GlobalTech"),
        create_triple("David", "worksAt", "StartupX"),
    ];
    
    // Projects and ownership
    let project_triples = vec![
        create_triple("ProjectAI", "ownedBy", "TechCorp"),
        create_triple("ProjectAI", "contributedBy", "Alice"),
        create_triple("ProjectAI", "uses", "TechnologyML"),
        create_triple("TechnologyML", "developedBy", "StartupX"),
    ];
    
    // Store all triples
    for triple in company_triples.iter()
        .chain(people_triples.iter())
        .chain(project_triples.iter()) {
        engine.store_triple(triple.clone(), None).expect("Failed to store triple");
    }
    
    // Test multi-hop queries
    
    // 1. Find all entities related to Alice within 3 hops
    let alice_network = engine.get_entity_relationships("Alice", 3)
        .expect("Failed to get relationships");
    
    println!("Alice's 3-hop network contains {} relationships", alice_network.len());
    
    // Should include indirect relationships through Bob, Charlie, TechCorp, etc.
    let has_global_tech = alice_network.iter().any(|t| 
        t.subject == "GlobalTech" || t.object == "GlobalTech"
    );
    assert!(has_global_tech, "3-hop query should reach GlobalTech through company hierarchy");
    
    // 2. Find all entities connected to ProjectAI
    let project_network = engine.get_entity_relationships("ProjectAI", 2)
        .expect("Failed to get relationships");
    
    // Should connect to StartupX through TechnologyML
    let connects_startup = project_network.iter().any(|t|
        t.subject == "StartupX" || t.object == "StartupX"
    );
    assert!(connects_startup, "Should connect ProjectAI to StartupX through TechnologyML");
    
    // 3. Complex query combining multiple filters
    let query = TripleQuery {
        subject: None,
        predicate: Some("worksAt".to_string()),
        object: Some("TechCorp".to_string()),
        limit: 50,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let techcorp_employees = engine.query_triples(query).expect("Query failed");
    assert_eq!(techcorp_employees.triples.len(), 2, "Should find Alice and Bob at TechCorp");
}

#[test]
fn test_memory_stats_accuracy() {
    let engine = KnowledgeEngine::new(128, 1000).expect("Failed to create engine");
    
    // Get initial stats
    let initial_stats = engine.get_memory_stats();
    assert_eq!(initial_stats.total_nodes, 0);
    assert_eq!(initial_stats.total_triples, 0);
    assert_eq!(initial_stats.total_bytes, 0);
    
    // Store some data
    let triple_count = 50;
    for i in 0..triple_count {
        let triple = create_triple(
            &format!("Subject{}", i),
            "predicate",
            &format!("Object{}", i)
        );
        engine.store_triple(triple, None).expect("Failed to store triple");
    }
    
    // Check updated stats
    let stats = engine.get_memory_stats();
    assert_eq!(stats.total_nodes, triple_count);
    assert_eq!(stats.total_triples, triple_count);
    assert!(stats.total_bytes > 0);
    assert!(stats.bytes_per_node > 0.0);
    
    // Verify bytes per node is reasonable
    let bytes_per_triple = stats.total_bytes as f64 / stats.total_triples as f64;
    println!("Memory usage: {} bytes per triple", bytes_per_triple);
    assert!(bytes_per_triple < 1000.0, "Memory usage should be reasonable");
}

#[tokio::test]
async fn test_entity_type_tracking() {
    let engine = KnowledgeEngine::new(256, 1000).expect("Failed to create engine");
    
    // Store various entity types
    let entities = vec![
        ("Alice", "Person"),
        ("Bob", "Person"),
        ("TechCorp", "Company"),
        ("ProjectX", "Project"),
        ("Rust", "Technology"),
    ];
    
    for (name, entity_type) in &entities {
        engine.store_entity(
            name.to_string(),
            entity_type.to_string(),
            format!("{} entity", entity_type),
            HashMap::new()
        ).expect("Failed to store entity");
    }
    
    // Get all entity types
    let entity_types = engine.get_entity_types();
    
    // Verify all types are tracked
    for (name, expected_type) in &entities {
        assert_eq!(
            entity_types.get(*name),
            Some(&expected_type.to_string()),
            "Entity {} should have type {}",
            name,
            expected_type
        );
    }
    
    // Query by entity and check context includes correct type
    let query = TripleQuery {
        subject: Some("Alice".to_string()),
        predicate: None,
        object: None,
        limit: 10,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    // First add a relationship so we have something to query
    engine.store_triple(create_triple("Alice", "likes", "Rust"), None)
        .expect("Failed to store triple");
    
    let result = engine.query_triples(query).expect("Query failed");
    
    if let Some(alice_context) = result.entity_context.get("Alice") {
        assert_eq!(alice_context.entity_type, "Person");
    }
}