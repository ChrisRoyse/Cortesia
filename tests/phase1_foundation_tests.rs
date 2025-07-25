use llmkg::core::entity_extractor::{EntityExtractor, EntityType};
use llmkg::core::relationship_extractor::RelationshipExtractor;
use llmkg::core::question_parser::QuestionParser;
use llmkg::core::answer_generator::AnswerGenerator;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use llmkg::core::knowledge_types::{QuestionType, AnswerType, TripleQuery};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
mod entity_extraction_tests {
    use super::*;

    #[test]
    fn test_person_entity_extraction() {
        let extractor = EntityExtractor::new();
        let text = "Albert Einstein developed the Theory of Relativity in 1905.";
        let entities = extractor.extract_entities(text);

        // Should extract multi-word person name
        assert!(entities.iter().any(|e| e.name == "Albert Einstein" && e.entity_type == EntityType::Person));
        // Should extract multi-word concept
        assert!(entities.iter().any(|e| e.name == "Theory of Relativity" && e.entity_type == EntityType::Concept));
        // Should extract year
        assert!(entities.iter().any(|e| e.name == "1905" && e.entity_type == EntityType::Time));
    }

    #[test]
    fn test_organization_entity_extraction() {
        let extractor = EntityExtractor::new();
        let text = "Microsoft Corporation partnered with OpenAI Inc to develop advanced AI systems.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "Microsoft Corporation" && e.entity_type == EntityType::Organization));
        assert!(entities.iter().any(|e| e.name == "OpenAI Inc" && e.entity_type == EntityType::Organization));
        assert!(entities.iter().any(|e| e.name == "AI" && e.entity_type == EntityType::Unknown));
    }

    #[test]
    fn test_complex_entities_with_connectors() {
        let extractor = EntityExtractor::new();
        let text = "The University of California at Berkeley is located in the San Francisco Bay Area.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "University of California"));
        assert!(entities.iter().any(|e| e.name == "Berkeley"));
        assert!(entities.iter().any(|e| e.name == "San Francisco Bay Area"));
    }

    #[test]
    fn test_quoted_entities() {
        let extractor = EntityExtractor::new();
        let text = "The concept of 'quantum entanglement' was described as 'spooky action at a distance'.";
        let entities = extractor.extract_entities(text);

        assert!(entities.iter().any(|e| e.name == "quantum entanglement" && e.entity_type == EntityType::Concept));
        assert!(entities.iter().any(|e| e.name == "spooky action at a distance" && e.entity_type == EntityType::Concept));
    }

    #[test]
    fn test_no_overlapping_entities() {
        let extractor = EntityExtractor::new();
        let text = "New York City is different from New York State.";
        let entities = extractor.extract_entities(text);

        // Should extract both as separate entities
        let ny_entities: Vec<_> = entities.iter()
            .filter(|e| e.name.contains("New York"))
            .collect();
        
        assert!(ny_entities.len() >= 2);
        assert!(ny_entities.iter().any(|e| e.name == "New York City"));
        assert!(ny_entities.iter().any(|e| e.name == "New York State"));
    }
}

#[cfg(test)]
mod relationship_extraction_tests {
    use super::*;

    #[test]
    fn test_verb_relationships() {
        let entity_extractor = EntityExtractor::new();
        let rel_extractor = RelationshipExtractor::new();
        
        let text = "Einstein invented the equation E=mc².";
        let entities = entity_extractor.extract_entities(text);
        let relationships = rel_extractor.extract_relationships(text, &entities);

        assert!(!relationships.is_empty());
        assert!(relationships.iter().any(|r| 
            r.subject == "Einstein" && 
            r.predicate == "invented" && 
            r.object.contains("E")
        ));
    }

    #[test]
    fn test_location_relationships() {
        let entity_extractor = EntityExtractor::new();
        let rel_extractor = RelationshipExtractor::new();
        
        let text = "The Eiffel Tower is located in Paris, France.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = rel_extractor.extract_relationships(text, &entities);


        assert!(relationships.iter().any(|r| 
            r.subject == "Eiffel Tower" && 
            r.predicate == "located in" && 
            r.object == "Paris"
        ));
    }

    #[test]
    fn test_multiple_relationships() {
        let entity_extractor = EntityExtractor::new();
        let rel_extractor = RelationshipExtractor::new();
        
        let text = "Marie Curie discovered polonium and radium. She was born in Warsaw and worked in Paris.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = rel_extractor.extract_relationships(text, &entities);

        // Should find multiple relationships
        assert!(relationships.len() >= 2);
        assert!(relationships.iter().any(|r| r.subject == "Marie Curie" && r.predicate == "discovered"));
        assert!(relationships.iter().any(|r| r.predicate == "born in" || r.predicate == "in"));
    }

    #[test]
    fn test_causal_relationships() {
        let entity_extractor = EntityExtractor::new();
        let rel_extractor = RelationshipExtractor::new();
        
        let text = "Global warming causes sea levels to rise, which leads to coastal flooding.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = rel_extractor.extract_relationships(text, &entities);

        assert!(relationships.iter().any(|r| r.predicate == "causes" || r.predicate == "leads to"));
    }
}

#[cfg(test)]
mod question_parsing_tests {
    use super::*;

    #[test]
    fn test_what_question_parsing() {
        let intent = QuestionParser::parse("What did Einstein discover?");
        
        assert_eq!(intent.question_type, QuestionType::What);
        assert!(intent.entities.contains(&"Einstein".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Fact);
    }

    #[test]
    fn test_who_question_parsing() {
        let intent = QuestionParser::parse("Who invented the telephone?");
        
        assert_eq!(intent.question_type, QuestionType::Who);
        assert_eq!(intent.expected_answer_type, AnswerType::Entity);
    }

    #[test]
    fn test_when_question_with_temporal_context() {
        let intent = QuestionParser::parse("When was the Theory of Relativity published?");
        
        assert_eq!(intent.question_type, QuestionType::When);
        assert!(intent.entities.contains(&"Theory of Relativity".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Time);
    }

    #[test]
    fn test_temporal_range_extraction() {
        let intent = QuestionParser::parse("What happened between 1900 and 1920?");
        
        assert!(intent.temporal_context.is_some());
        let range = intent.temporal_context.unwrap();
        assert_eq!(range.start, Some("1900".to_string()));
        assert_eq!(range.end, Some("1920".to_string()));
    }

    #[test]
    fn test_boolean_question() {
        let intent = QuestionParser::parse("Is Einstein a physicist?");
        
        assert_eq!(intent.question_type, QuestionType::Is);
        assert!(intent.entities.contains(&"Einstein".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Boolean);
    }

    #[test]
    fn test_multi_entity_extraction() {
        let intent = QuestionParser::parse("What is the relationship between Albert Einstein and the Theory of Relativity?");
        
        assert!(intent.entities.contains(&"Albert Einstein".to_string()));
        assert!(intent.entities.contains(&"Theory of Relativity".to_string()));
    }
}

#[cfg(test)]
mod answer_generation_tests {
    use super::*;
    use llmkg::core::knowledge_types::QuestionIntent;

    fn create_test_intent(question_type: QuestionType, entities: Vec<&str>, answer_type: AnswerType) -> QuestionIntent {
        QuestionIntent {
            question_type,
            entities: entities.iter().map(|s| s.to_string()).collect(),
            expected_answer_type: answer_type,
            temporal_context: None,
        }
    }

    #[test]
    fn test_who_answer_generation() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "invented".to_string(), "E=mc²".to_string()).unwrap(),
            Triple::new("Einstein".to_string(), "developed".to_string(), "Theory of Relativity".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::Who, vec!["E=mc²"], AnswerType::Entity);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "Einstein");
        assert!(answer.confidence > 0.5);
        assert!(!answer.facts.is_empty());
    }

    #[test]
    fn test_what_answer_generation() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "is".to_string(), "physicist".to_string()).unwrap(),
            Triple::new("Einstein".to_string(), "has".to_string(), "Nobel Prize".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::What, vec!["Einstein"], AnswerType::Fact);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert!(answer.text.contains("Einstein is physicist"));
        assert!(!answer.entities.is_empty());
    }

    #[test]
    fn test_where_answer_generation() {
        let facts = vec![
            Triple::new("Eiffel Tower".to_string(), "located_in".to_string(), "Paris".to_string()).unwrap(),
            Triple::new("Paris".to_string(), "in".to_string(), "France".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::Where, vec!["Eiffel Tower"], AnswerType::Location);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "Paris");
        assert!(answer.confidence > 0.7);
    }

    #[test]
    fn test_boolean_answer_generation() {
        let facts = vec![
            Triple::new("Einstein".to_string(), "is".to_string(), "physicist".to_string()).unwrap(),
        ];
        
        let intent = create_test_intent(QuestionType::Is, vec!["Einstein", "physicist"], AnswerType::Boolean);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "Yes");
    }

    #[test]
    fn test_no_facts_answer() {
        let facts = vec![];
        let intent = create_test_intent(QuestionType::What, vec!["Unknown"], AnswerType::Fact);
        let answer = AnswerGenerator::generate_answer(facts, intent);
        
        assert_eq!(answer.text, "I don't have enough information to answer this question.");
        assert_eq!(answer.confidence, 0.0);
    }
}

#[tokio::test]
async fn test_end_to_end_integration() {
    // Create knowledge engine
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(512, 10_000).unwrap()));
    
    // Store some facts
    {
        let engine_guard = engine.write().await;
        
        // Store multi-word entities
        let triple1 = Triple::new(
            "Albert Einstein".to_string(),
            "developed".to_string(),
            "Theory of Relativity".to_string()
        ).unwrap();
        engine_guard.store_triple(triple1, None).unwrap();
        
        let triple2 = Triple::new(
            "Theory of Relativity".to_string(),
            "published_in".to_string(),
            "1905".to_string()
        ).unwrap();
        engine_guard.store_triple(triple2, None).unwrap();
        
        let triple3 = Triple::new(
            "Albert Einstein".to_string(),
            "is".to_string(),
            "physicist".to_string()
        ).unwrap();
        engine_guard.store_triple(triple3, None).unwrap();
    }
    
    // Test question answering
    {
        let engine_guard = engine.read().await;
        
        // Parse question
        let question = "What did Albert Einstein develop?";
        let intent = QuestionParser::parse(question);
        
        assert_eq!(intent.question_type, QuestionType::What);
        assert!(intent.entities.contains(&"Albert Einstein".to_string()));
        
        // Search for facts
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
        
        // Generate answer
        let answer = AnswerGenerator::generate_answer(all_facts, intent);
        
        assert!(answer.text.contains("Theory of Relativity"));
        assert!(answer.confidence > 0.5);
        assert!(!answer.facts.is_empty());
    }
}

#[tokio::test]
async fn test_complex_knowledge_extraction() {
    let text = "Marie Curie, born Maria Sklodowska in Warsaw, Poland, was a pioneering physicist and chemist. \
                She discovered the elements polonium and radium in 1898. Marie Curie was the first woman to \
                win a Nobel Prize and the only person to win Nobel Prizes in two different sciences.";
    
    // Extract entities
    let entity_extractor = EntityExtractor::new();
    let entities = entity_extractor.extract_entities(text);
    
    // Should extract multiple entity types
    assert!(entities.iter().any(|e| e.name == "Marie Curie" && e.entity_type == EntityType::Person));
    assert!(entities.iter().any(|e| e.name == "Warsaw"));
    assert!(entities.iter().any(|e| e.name == "Poland"));
    assert!(entities.iter().any(|e| e.name == "Nobel Prize"));
    assert!(entities.iter().any(|e| e.name == "1898" && e.entity_type == EntityType::Time));
    
    // Extract relationships
    let rel_extractor = RelationshipExtractor::new();
    let relationships = rel_extractor.extract_relationships(text, &entities);
    
    // Should find multiple relationship types
    assert!(relationships.iter().any(|r| r.subject == "Marie Curie" && r.predicate == "discovered"));
    assert!(relationships.iter().any(|r| r.predicate.contains("born") || r.predicate == "in"));
    
    // Store in knowledge engine
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(512, 10_000).unwrap()));
    {
        let engine_guard = engine.write().await;
        
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
    
    // Test question answering about the stored knowledge
    {
        let engine_guard = engine.read().await;
        
        // Test "Who discovered polonium?"
        let intent = QuestionParser::parse("Who discovered polonium?");
        let mut facts = Vec::new();
        
        let query = TripleQuery {
            subject: None,
            predicate: Some("discovered".to_string()),
            object: Some("polonium".to_string()),
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(results) = engine_guard.query_triples(query) {
            facts.extend(results);
        }
        
        if !facts.is_empty() {
            let answer = AnswerGenerator::generate_answer(facts, intent);
            // Check if the answer contains the correct person name
            let is_marie_curie = answer.text.contains("Marie Curie") || answer.text == "Marie Curie";
            let is_maria_sklodowska = answer.text.contains("Maria Sklodowska") || answer.text == "Maria Sklodowska";
            assert!(is_marie_curie || is_maria_sklodowska, "Expected 'Marie Curie' or 'Maria Sklodowska', got: '{}'", answer.text);
        }
    }
}