use llmkg::core::triple::{
    Triple, KnowledgeNode, NodeType, NodeContent, NodeMetadata, 
    PredicateVocabulary, MAX_CHUNK_SIZE_BYTES
};
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::types::EntityKey;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

/// Helper function to create test embeddings
fn create_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.01).collect()
}

#[tokio::test]
async fn test_triple_knowledge_engine_integration() {
    // Create knowledge engine
    let temp_dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(temp_dir.path().to_path_buf()).await.unwrap()
    ));

    // Create test triples
    let triples = vec![
        Triple::new("Einstein".to_string(), "invented".to_string(), "relativity".to_string()).unwrap(),
        Triple::new("Einstein".to_string(), "born_in".to_string(), "1879".to_string()).unwrap(),
        Triple::new("relativity".to_string(), "is_a".to_string(), "theory".to_string()).unwrap(),
        Triple::new("Einstein".to_string(), "worked_at".to_string(), "Princeton".to_string()).unwrap(),
    ];

    // Store triples as knowledge nodes
    let mut stored_nodes = Vec::new();
    {
        let mut engine_write = engine.write().await;
        
        for triple in &triples {
            // Create knowledge node from triple
            let node = KnowledgeNode::new_triple(triple.clone(), create_test_embedding(128));
            
            // Store as entity in knowledge engine
            let entity_data = llmkg::core::types::EntityData {
                type_id: 1, // Triple type
                properties: serde_json::to_string(&node).unwrap(),
                embedding: node.embedding.clone(),
            };
            
            let entity_key = engine_write.add_entity(entity_data).await.unwrap();
            stored_nodes.push((entity_key, node));
        }
    }

    // Test retrieval and querying
    {
        let engine_read = engine.read().await;
        
        // Retrieve all triples about Einstein
        let mut einstein_triples = Vec::new();
        for (entity_key, node) in &stored_nodes {
            if let NodeContent::Triple(triple) = &node.content {
                if triple.subject == "Einstein" {
                    einstein_triples.push(triple.clone());
                }
            }
        }
        
        assert_eq!(einstein_triples.len(), 3, "Should find 3 triples about Einstein");
        
        // Test natural language conversion
        for triple in &einstein_triples {
            let nl = triple.to_natural_language();
            println!("Natural language: {}", nl);
            assert!(!nl.is_empty());
            assert!(nl.contains(&triple.subject));
        }
    }

    // Test relationship chains
    {
        let engine_read = engine.read().await;
        
        // Find chain: Einstein -> relativity -> theory
        let mut chain = Vec::new();
        
        // Find Einstein -> relativity
        for (_, node) in &stored_nodes {
            if let NodeContent::Triple(triple) = &node.content {
                if triple.subject == "Einstein" && triple.object == "relativity" {
                    chain.push(triple.clone());
                }
            }
        }
        
        // Find relativity -> theory
        for (_, node) in &stored_nodes {
            if let NodeContent::Triple(triple) = &node.content {
                if triple.subject == "relativity" && triple.object == "theory" {
                    chain.push(triple.clone());
                }
            }
        }
        
        assert_eq!(chain.len(), 2, "Should find complete chain");
        println!("Knowledge chain: {} -> {} -> {}", 
                 chain[0].subject, chain[0].object, chain[1].object);
    }
}

#[tokio::test]
async fn test_knowledge_representation_workflow() {
    // Create a complete knowledge representation system
    let temp_dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(temp_dir.path().to_path_buf()).await.unwrap()
    ));

    // Test different node types
    let test_text = "Artificial Intelligence (AI) is the simulation of human intelligence in machines. \
                     It encompasses machine learning, natural language processing, and computer vision.";
    
    // 1. Create chunk node with extracted triples
    let extracted_triples = vec![
        Triple::new("AI".to_string(), "is".to_string(), "simulation".to_string()).unwrap(),
        Triple::new("AI".to_string(), "encompasses".to_string(), "machine_learning".to_string()).unwrap(),
        Triple::new("AI".to_string(), "encompasses".to_string(), "NLP".to_string()).unwrap(),
    ];
    
    let chunk_node = KnowledgeNode::new_chunk(
        test_text.to_string(),
        create_test_embedding(128),
        extracted_triples.clone()
    ).unwrap();

    // 2. Create entity nodes
    let ai_entity = KnowledgeNode::new_entity(
        "Artificial Intelligence".to_string(),
        "The simulation of human intelligence in machines".to_string(),
        "Concept".to_string(),
        {
            let mut props = HashMap::new();
            props.insert("abbreviation".to_string(), "AI".to_string());
            props.insert("field".to_string(), "Computer Science".to_string());
            props
        },
        create_test_embedding(128)
    ).unwrap();

    // 3. Create relationship node
    let relationship_node = KnowledgeNode {
        id: "node_encompasses".to_string(), // Using string directly since generate_content_id is private
        node_type: NodeType::Relationship,
        content: NodeContent::Relationship {
            predicate: "encompasses".to_string(),
            description: "Indicates that one concept includes another as a component".to_string(),
            domain: "Concept".to_string(),
            range: "Concept".to_string(),
        },
        embedding: create_test_embedding(128),
        metadata: NodeMetadata {
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
            size_bytes: 100,
            quality_score: 1.0,
            tags: vec!["inclusion".to_string(), "hierarchy".to_string()],
        },
    };

    // Store all nodes
    {
        let mut engine_write = engine.write().await;
        
        for node in vec![chunk_node, ai_entity, relationship_node] {
            let entity_data = llmkg::core::types::EntityData {
                type_id: match node.node_type {
                    NodeType::Triple => 1,
                    NodeType::Chunk => 2,
                    NodeType::Entity => 3,
                    NodeType::Relationship => 4,
                },
                properties: serde_json::to_string(&node).unwrap(),
                embedding: node.embedding.clone(),
            };
            
            engine_write.add_entity(entity_data).await.unwrap();
        }
    }

    // Test retrieval and LLM formatting
    {
        let engine_read = engine.read().await;
        
        // In a real system, we'd query by type or content
        // For this test, we'll demonstrate the LLM formatting
        println!("\nKnowledge Representation LLM Output:");
        
        let test_nodes = vec![
            KnowledgeNode::new_triple(
                extracted_triples[0].clone(),
                create_test_embedding(128)
            ),
            KnowledgeNode::new_chunk(
                test_text.to_string(),
                create_test_embedding(128),
                extracted_triples.clone()
            ).unwrap(),
        ];
        
        for node in test_nodes {
            println!("\n{}", node.to_llm_format());
            
            // Test quality scoring
            let mut mutable_node = node.clone();
            mutable_node.mark_accessed();
            mutable_node.calculate_quality_score();
            
            assert!(mutable_node.metadata.quality_score >= 0.0);
            assert!(mutable_node.metadata.quality_score <= 1.0);
            assert_eq!(mutable_node.metadata.access_count, 1);
        }
    }
}

#[tokio::test]
async fn test_predicate_vocabulary_integration() {
    // Create predicate vocabulary
    let vocab = PredicateVocabulary::new();
    
    // Test integration with triple creation
    let test_predicates = vec![
        ("is a", "is"),                    // Should normalize to "is"
        ("LOCATED IN", "located_in"),      // Should normalize case and spaces
        ("connected to", "connected_to"),   // Should normalize spaces
        ("custom_predicate", "custom_predicate"), // Should pass through
    ];
    
    for (input, expected) in test_predicates {
        let normalized = vocab.normalize(input);
        assert_eq!(normalized, expected, "Failed to normalize '{}'", input);
        
        // Create triple with normalized predicate
        let triple = Triple::new(
            "Subject".to_string(),
            normalized.clone(),
            "Object".to_string()
        ).unwrap();
        
        assert_eq!(triple.predicate, expected);
    }

    // Test predicate suggestions for different contexts
    let contexts = vec![
        ("I want to express that Einstein created the theory of relativity", vec!["created_by"]),
        ("Show me entities that are similar", vec!["similar_to"]),
        ("Find causal relationships", vec!["causes"]),
        ("Unknown context xyz", vec!["is", "has", "connected_to"]), // Fallback
    ];
    
    for (context, expected_contains) in contexts {
        let suggestions = vocab.suggest_predicates(context);
        println!("Context: '{}' -> Suggestions: {:?}", context, suggestions);
        
        for expected in expected_contains {
            assert!(
                suggestions.iter().any(|s| s.contains(expected)),
                "Expected '{}' in suggestions for context '{}'", expected, context
            );
        }
    }
}

#[tokio::test]
async fn test_knowledge_node_memory_management() {
    // Test anti-bloat features
    
    // Test 1: Chunk size limits
    let large_text = "a".repeat(MAX_CHUNK_SIZE_BYTES + 1);
    let result = KnowledgeNode::new_chunk(
        large_text,
        create_test_embedding(128),
        Vec::new()
    );
    
    assert!(result.is_err(), "Should reject oversized chunks");
    
    // Test 2: Memory footprint tracking
    let triple = Triple::new(
        "Subject".to_string(),
        "predicate".to_string(),
        "Object".to_string()
    ).unwrap();
    
    let footprint = triple.memory_footprint();
    let expected = "Subject".len() + "predicate".len() + "Object".len() + std::mem::size_of::<f32>();
    assert_eq!(footprint, expected, "Memory footprint calculation incorrect");
    
    // Test 3: Node size tracking
    let node = KnowledgeNode::new_triple(triple.clone(), create_test_embedding(128));
    assert!(node.metadata.size_bytes > 0, "Node should track its size");
    assert_eq!(
        node.metadata.size_bytes,
        triple.memory_footprint() + 128 * 4, // embedding size
        "Node size calculation incorrect"
    );
    
    // Test 4: Entity data size limits
    let mut large_properties = HashMap::new();
    for i in 0..100 {
        large_properties.insert(
            format!("key_{}", i),
            "x".repeat(100) // 100 chars per value
        );
    }
    
    let result = KnowledgeNode::new_entity(
        "Entity".to_string(),
        "Description".to_string(),
        "Type".to_string(),
        large_properties,
        create_test_embedding(128)
    );
    
    // Should either succeed with reasonable size or fail if too large
    match result {
        Ok(node) => {
            assert!(node.metadata.size_bytes <= MAX_CHUNK_SIZE_BYTES * 2,
                    "Entity node size should be reasonable");
        },
        Err(e) => {
            // Expected if properties are too large
            println!("Entity creation failed as expected: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_complete_knowledge_workflow() {
    // Test complete workflow: extraction -> storage -> retrieval -> query
    let temp_dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(temp_dir.path().to_path_buf()).await.unwrap()
    ));

    // Simulate document processing
    let documents = vec![
        (
            "doc1",
            "The Eiffel Tower is located in Paris. It was built in 1889 for the World's Fair."
        ),
        (
            "doc2", 
            "Paris is the capital of France. The city is known for its art, culture, and architecture."
        ),
        (
            "doc3",
            "The Eiffel Tower is 330 meters tall. It is made of iron and weighs about 10,000 tons."
        ),
    ];

    // Process documents and extract knowledge
    let mut all_triples = Vec::new();
    let mut chunk_nodes = Vec::new();
    
    for (doc_id, text) in &documents {
        // Extract triples (simplified extraction)
        let mut doc_triples = Vec::new();
        
        if text.contains("Eiffel Tower") && text.contains("Paris") {
            doc_triples.push(Triple::new(
                "Eiffel_Tower".to_string(),
                "located_in".to_string(),
                "Paris".to_string()
            ).unwrap());
        }
        
        if text.contains("built in 1889") {
            doc_triples.push(Triple::new(
                "Eiffel_Tower".to_string(),
                "built_in".to_string(),
                "1889".to_string()
            ).unwrap());
        }
        
        if text.contains("Paris") && text.contains("capital") {
            doc_triples.push(Triple::new(
                "Paris".to_string(),
                "is".to_string(),
                "capital_of_France".to_string()
            ).unwrap());
        }
        
        if text.contains("330 meters") {
            doc_triples.push(Triple::new(
                "Eiffel_Tower".to_string(),
                "has_height".to_string(),
                "330_meters".to_string()
            ).unwrap());
        }
        
        // Create chunk node
        let chunk = KnowledgeNode::new_chunk(
            text.to_string(),
            create_test_embedding(128),
            doc_triples.clone()
        ).unwrap();
        
        all_triples.extend(doc_triples);
        chunk_nodes.push((doc_id, chunk));
    }

    // Store all knowledge
    {
        let mut engine_write = engine.write().await;
        
        // Store chunks
        for (doc_id, chunk) in &chunk_nodes {
            let entity_data = llmkg::core::types::EntityData {
                type_id: 2, // Chunk type
                properties: serde_json::to_string(&chunk).unwrap(),
                embedding: chunk.embedding.clone(),
            };
            
            engine_write.add_entity(entity_data).await.unwrap();
        }
        
        // Store individual triples
        for triple in &all_triples {
            let node = KnowledgeNode::new_triple(triple.clone(), create_test_embedding(128));
            let entity_data = llmkg::core::types::EntityData {
                type_id: 1, // Triple type
                properties: serde_json::to_string(&node).unwrap(),
                embedding: node.embedding.clone(),
            };
            
            engine_write.add_entity(entity_data).await.unwrap();
        }
    }

    // Query the knowledge base
    println!("\nKnowledge Base Query Results:");
    
    // Find all facts about the Eiffel Tower
    let eiffel_facts: Vec<_> = all_triples.iter()
        .filter(|t| t.subject == "Eiffel_Tower")
        .collect();
    
    println!("\nFacts about Eiffel Tower:");
    for fact in &eiffel_facts {
        println!("  - {}", fact.to_natural_language());
    }
    
    assert!(eiffel_facts.len() >= 3, "Should find multiple facts about Eiffel Tower");
    
    // Find relationships between entities
    let relationships: Vec<_> = all_triples.iter()
        .filter(|t| t.predicate == "located_in" || t.predicate == "is")
        .collect();
    
    println!("\nEntity relationships:");
    for rel in &relationships {
        println!("  - {}", rel.to_natural_language());
    }
    
    // Test confidence scoring
    for triple in &all_triples {
        assert_eq!(triple.confidence, 1.0, "Default confidence should be 1.0");
    }
    
    // Create weighted triple
    let weighted_triple = Triple::with_metadata(
        "Eiffel_Tower".to_string(),
        "similar_to".to_string(),
        "Tokyo_Tower".to_string(),
        0.7,
        Some("inference".to_string())
    ).unwrap();
    
    assert_eq!(weighted_triple.confidence, 0.7);
    assert_eq!(weighted_triple.source, Some("inference".to_string()));
}