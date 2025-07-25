use llmkg::core::entity_extractor::EntityExtractor;
use llmkg::core::relationship_extractor::RelationshipExtractor;
use llmkg::core::question_parser::QuestionParser;
use llmkg::core::answer_generator::AnswerGenerator;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use llmkg::core::knowledge_types::{TripleQuery, QuestionType, AnswerType};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_comprehensive_integration() {
    println!("Starting comprehensive integration test...");
    
    // Test 1: Basic knowledge storage and retrieval
    println!("Test 1: Basic knowledge storage and retrieval");
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(384, 10_000).unwrap()));
    
    // Store some basic facts
    {
        let engine_guard = engine.write().await;
        
        let facts = vec![
            ("Albert Einstein", "is", "physicist"),
            ("Albert Einstein", "developed", "Theory of Relativity"),
            ("Theory of Relativity", "published_in", "1905"),
            ("Eiffel Tower", "located_in", "Paris"),
            ("Paris", "is_in", "France"),
        ];
        
        for (s, p, o) in facts {
            let triple = Triple::new(s.to_string(), p.to_string(), o.to_string()).unwrap();
            engine_guard.store_triple(triple, None).unwrap();
        }
    }
    println!("✓ Basic facts stored successfully");
    
    // Test 2: Query functionality
    println!("Test 2: Query functionality");
    {
        let engine_guard = engine.read().await;
        
        let query = TripleQuery {
            subject: Some("Albert Einstein".to_string()),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        let results = engine_guard.query_triples(query).unwrap();
        assert!(!results.is_empty(), "Should find facts about Einstein");
        assert!(results.iter().any(|t| t.object == "physicist"), "Should find that Einstein is a physicist");
        assert!(results.iter().any(|t| t.object == "Theory of Relativity"), "Should find that Einstein developed Theory of Relativity");
    }
    println!("✓ Query functionality works");
    
    // Test 3: Entity extraction
    println!("Test 3: Entity extraction");
    let entity_extractor = EntityExtractor::new();
    let text = "Marie Curie discovered radium and polonium in her laboratory in Paris.";
    let entities = entity_extractor.extract_entities(text);
    
    assert!(entities.iter().any(|e| e.name == "Marie Curie"), "Should extract Marie Curie");
    assert!(entities.iter().any(|e| e.name == "Paris"), "Should extract Paris");
    println!("✓ Entity extraction works");
    
    // Test 4: Relationship extraction
    println!("Test 4: Relationship extraction");
    let rel_extractor = RelationshipExtractor::new();
    let relationships = rel_extractor.extract_relationships(text, &entities);
    
    assert!(!relationships.is_empty(), "Should extract relationships");
    assert!(relationships.iter().any(|r| r.predicate == "discovered"), "Should find discovery relationship");
    println!("✓ Relationship extraction works");
    
    // Test 5: Question parsing
    println!("Test 5: Question parsing");
    let question_tests = vec![
        ("What did Einstein develop?", QuestionType::What, AnswerType::Fact),
        ("Who invented the telephone?", QuestionType::Who, AnswerType::Entity),
        ("When was relativity published?", QuestionType::When, AnswerType::Time),
        ("Where is the Eiffel Tower?", QuestionType::Where, AnswerType::Location),
        ("Is Einstein a physicist?", QuestionType::Is, AnswerType::Boolean),
    ];
    
    for (question, expected_type, expected_answer_type) in question_tests {
        let intent = QuestionParser::parse(question);
        assert_eq!(intent.question_type, expected_type, "Question type mismatch for: {}", question);
        assert_eq!(intent.expected_answer_type, expected_answer_type, "Answer type mismatch for: {}", question);
    }
    println!("✓ Question parsing works");
    
    // Test 6: End-to-end question answering
    println!("Test 6: End-to-end question answering");
    {
        let engine_guard = engine.read().await;
        
        let test_questions = vec![
            ("What did Albert Einstein develop?", vec!["Theory of Relativity"]),
            ("Who is a physicist?", vec!["Albert Einstein"]),
            ("Where is the Eiffel Tower?", vec!["Paris"]),
            ("Is Einstein a physicist?", vec!["Yes"]),
        ];
        
        for (question, expected_answers) in test_questions {
            let intent = QuestionParser::parse(question);
            
            // Search for relevant facts
            let mut all_facts = Vec::new();
            for entity in &intent.entities {
                let query = TripleQuery {
                    subject: Some(entity.clone()),
                    predicate: None,
                    object: None,
                    limit: 100,
                    min_confidence: 0.0,
                    include_chunks: false,
                };
                
                if let Ok(results) = engine_guard.query_triples(query) {
                    all_facts.extend(results);
                }
            }
            
            // If no entities found, try broader search based on question type
            if all_facts.is_empty() && intent.question_type == QuestionType::Who {
                let query = TripleQuery {
                    subject: None,
                    predicate: None,
                    object: None,
                    limit: 100,
                    min_confidence: 0.0,
                    include_chunks: false,
                };
                
                if let Ok(results) = engine_guard.query_triples(query) {
                    all_facts.extend(results);
                }
            }
            
            if !all_facts.is_empty() {
                let answer = AnswerGenerator::generate_answer(all_facts, intent);
                
                let answer_matches = expected_answers.iter().any(|expected| {
                    answer.text.contains(expected) || answer.text == *expected
                });
                
                assert!(answer_matches, "Answer '{}' doesn't match expected answers {:?} for question: {}", 
                       answer.text, expected_answers, question);
                assert!(answer.confidence > 0.0, "Answer should have positive confidence");
            }
        }
    }
    println!("✓ End-to-end question answering works");
    
    // Test 7: Complex knowledge extraction and storage
    println!("Test 7: Complex knowledge extraction and storage");
    {
        let engine_guard = engine.write().await;
        
        let complex_text = "Isaac Newton formulated the laws of motion and universal gravitation. \
                           He published Principia Mathematica in 1687, which revolutionized physics.";
        
        let entities = entity_extractor.extract_entities(complex_text);
        let relationships = rel_extractor.extract_relationships(complex_text, &entities);
        
        // Store the extracted knowledge
        for rel in &relationships {
            if let Ok(triple) = Triple::new(
                rel.subject.clone(),
                rel.predicate.clone(),
                rel.object.clone()
            ) {
                let _ = engine_guard.store_triple(triple, None);
            }
        }
    }
    
    // Verify the complex knowledge was stored correctly
    {
        let engine_guard = engine.read().await;
        
        let query = TripleQuery {
            subject: Some("Isaac Newton".to_string()),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        let results = engine_guard.query_triples(query).unwrap();
        assert!(!results.is_empty(), "Should find facts about Newton");
        
        let has_formulated = results.iter().any(|t| t.predicate == "formulated");
        let has_published = results.iter().any(|t| t.predicate == "published");
        
        // Should find either the exact predicate or a similar one
        assert!(has_formulated || has_published || 
               results.iter().any(|t| t.object.contains("Principia")), 
               "Should find Newton's contributions");
    }
    println!("✓ Complex knowledge extraction and storage works");
    
    // Test 8: Performance test
    println!("Test 8: Performance test");
    let start_time = std::time::Instant::now();
    
    {
        let engine_guard = engine.write().await;
        
        // Store 100 facts
        for i in 0..100 {
            let triple = Triple::new(
                format!("Person{}", i),
                "likes".to_string(),
                format!("Activity{}", i % 10)
            ).unwrap();
            engine_guard.store_triple(triple, None).unwrap();
        }
    }
    
    {
        let engine_guard = engine.read().await;
        
        // Query 10 times
        for i in 0..10 {
            let query = TripleQuery {
                subject: Some(format!("Person{}", i)),
                predicate: None,
                object: None,
                limit: 10,
                min_confidence: 0.0,
                include_chunks: false,
            };
            engine_guard.query_triples(query).unwrap();
        }
    }
    
    let elapsed = start_time.elapsed();
    assert!(elapsed.as_secs() < 5, "Performance test should complete within 5 seconds");
    println!("✓ Performance test passed (took {:?})", elapsed);
    
    println!("🎉 All comprehensive integration tests passed!");
}