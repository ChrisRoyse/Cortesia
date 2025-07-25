use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use llmkg::core::knowledge_types::TripleQuery;
use std::time::Instant;

#[tokio::test]
async fn test_performance_requirements() {
    println!("Testing performance requirements...");
    
    let engine = KnowledgeEngine::new(384, 10_000).unwrap();
    
    // Test 1: Storage performance - should store 1000 triples in under 2 seconds
    println!("Test 1: Storage performance");
    let start = Instant::now();
    
    for i in 0..1000 {
        let triple = Triple::new(
            format!("Subject{}", i),
            format!("predicate{}", i % 10),
            format!("Object{}", i)
        ).unwrap();
        engine.store_triple(triple, None).unwrap();
    }
    
    let storage_time = start.elapsed();
    println!("✓ Stored 1000 triples in {:?}", storage_time);
    assert!(storage_time.as_secs() < 2, "Storage should complete within 2 seconds");
    
    // Test 2: Query performance - should perform 100 queries in under 1 second
    println!("Test 2: Query performance");
    let start = Instant::now();
    
    for i in 0..100 {
        let query = TripleQuery {
            subject: Some(format!("Subject{}", i)),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        engine.query_triples(query).unwrap();
    }
    
    let query_time = start.elapsed();
    println!("✓ Performed 100 queries in {:?}", query_time);
    assert!(query_time.as_secs() < 1, "Queries should complete within 1 second");
    
    // Test 3: Mixed workload performance
    println!("Test 3: Mixed workload performance");
    let start = Instant::now();
    
    for i in 0..100 {
        // Store a triple
        let triple = Triple::new(
            format!("MixedSubject{}", i),
            "mixed_predicate".to_string(),
            format!("MixedObject{}", i)
        ).unwrap();
        engine.store_triple(triple, None).unwrap();
        
        // Query every 10th iteration
        if i % 10 == 0 {
            let query = TripleQuery {
                subject: None,
                predicate: Some("mixed_predicate".to_string()),
                object: None,
                limit: 10,
                min_confidence: 0.0,
                include_chunks: false,
            };
            engine.query_triples(query).unwrap();
        }
    }
    
    let mixed_time = start.elapsed();
    println!("✓ Mixed workload completed in {:?}", mixed_time);
    assert!(mixed_time.as_secs() < 2, "Mixed workload should complete within 2 seconds");
    
    // Test 4: Memory usage (basic check)
    println!("Test 4: Memory usage check");
    let query = TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 10000,
        min_confidence: 0.0,
        include_chunks: false,
    };
    let all_results = engine.query_triples(query).unwrap();
    println!("✓ Retrieved {} total triples", all_results.len());
    assert!(all_results.len() >= 1100, "Should have stored at least 1100 triples");
    
    println!("🚀 All performance requirements met!");
}