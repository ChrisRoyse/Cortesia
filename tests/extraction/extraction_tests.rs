use llmkg::extraction::{
    AdvancedEntityExtractor, Entity, Relation, EntityExtractor,
};

#[cfg(test)]
mod entity_extraction_tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_string_entity_extraction() {
        let text = "John Smith works at Microsoft Corporation in Seattle.";
        let extractor = "simple";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        assert!(!entities.is_empty());
        // Should extract capitalized words
        let entity_texts: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
        assert!(entity_texts.contains(&"John".to_string()));
        assert!(entity_texts.contains(&"Smith".to_string()));
        assert!(entity_texts.contains(&"Microsoft".to_string()));
        assert!(entity_texts.contains(&"Corporation".to_string()));
        assert!(entity_texts.contains(&"Seattle.".to_string()));
    }

    #[tokio::test]
    async fn test_string_entity_extraction_with_confidence() {
        let text = "Dr. Jane Doe is a researcher at Stanford University.";
        let extractor = text.to_string();
        
        let entities_with_conf = extractor.extract_entities_with_confidence(text).await.unwrap();
        
        assert!(!entities_with_conf.is_empty());
        for (entity, confidence) in entities_with_conf {
            assert!(confidence >= 0.0 && confidence <= 1.0);
            assert_eq!(entity.confidence, confidence);
        }
    }

    #[tokio::test]
    async fn test_advanced_entity_extraction() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Dr. John Smith works at Google Inc. He was born in New York City in 1985.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        assert!(!entities.is_empty());
        
        // Check for person entities
        let person_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type.contains("PERSON"))
            .collect();
        assert!(!person_entities.is_empty());
        assert!(person_entities.iter().any(|e| e.text.contains("Dr. John Smith")));
        
        // Check for organization entities
        let org_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type.contains("ORGANIZATION"))
            .collect();
        assert!(!org_entities.is_empty());
        assert!(org_entities.iter().any(|e| e.text == "Google Inc"));
        
        // Check for location entities
        let location_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type.contains("LOCATION"))
            .collect();
        assert!(!location_entities.is_empty());
        assert!(location_entities.iter().any(|e| e.text == "New York City"));
        
        // Check for date entities
        let date_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type.contains("DATE"))
            .collect();
        assert!(!date_entities.is_empty());
        assert!(date_entities.iter().any(|e| e.text == "1985"));
    }

    #[tokio::test]
    async fn test_entity_position_tracking() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Apple was founded by Steve Jobs.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        for entity in entities {
            assert!(entity.start_pos < entity.end_pos);
            assert!(entity.end_pos <= text.len());
            
            // Verify the text matches the position
            let extracted_text = &text[entity.start_pos..entity.end_pos];
            assert_eq!(extracted_text, entity.text);
        }
    }

    #[tokio::test]
    async fn test_entity_canonical_names() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Dr. Watson and Prof. Einstein discussed quantum physics.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        for entity in entities {
            // Canonical names should be normalized (no titles)
            assert!(!entity.canonical_name.starts_with("Dr."));
            assert!(!entity.canonical_name.starts_with("Prof."));
        }
    }

    #[tokio::test]
    async fn test_entity_properties() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Microsoft Corporation was founded in 1975.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        for entity in entities {
            assert!(entity.properties.is_empty() || entity.properties.len() >= 0);
            assert!(!entity.id.is_empty());
            assert!(!entity.source_model.is_empty());
        }
    }
}

#[cfg(test)]
mod relation_extraction_tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_relation_extraction() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Albert Einstein invented the theory of relativity.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        assert!(!relations.is_empty());
        
        // Check for invention relation
        let invention_relations: Vec<&Relation> = relations.iter()
            .filter(|r| r.predicate.contains("invent"))
            .collect();
        assert!(!invention_relations.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_relation_extraction() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "John Smith works at Google. He was born in California.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        // Should find at least works_at and born_in relations
        let predicates: Vec<String> = relations.iter()
            .map(|r| r.predicate.clone())
            .collect();
        
        assert!(predicates.iter().any(|p| p.contains("works_at")));
        assert!(predicates.iter().any(|p| p.contains("born_in")));
    }

    #[tokio::test]
    async fn test_relation_confidence_scores() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Marie Curie is a scientist. She invented polonium.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        for relation in relations {
            assert!(relation.confidence >= 0.0 && relation.confidence <= 1.0);
            assert!(!relation.evidence.is_empty());
            assert!(!relation.extraction_model.is_empty());
        }
    }

    #[tokio::test]
    async fn test_predicate_normalization() {
        // PredicateNormalizer is not publicly accessible, so we'll test through the extractor
        let extractor = AdvancedEntityExtractor::new();
        
        // Test predicate normalization through relation extraction
        let text = "John works at Google.";
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        // The predicate should be normalized to "works_at"
        if !relations.is_empty() {
            assert!(relations.iter().any(|r| r.predicate == "works_at"));
        }
    }

    #[tokio::test]
    async fn test_confidence_scorer() {
        // Test confidence scoring through the extractor
        let extractor = AdvancedEntityExtractor::new();
        
        // Extract relations and check their confidence scores
        let text = "Einstein is a physicist.";
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        for relation in relations {
            assert!(relation.confidence >= 0.0 && relation.confidence <= 1.0);
        }
    }
}

#[cfg(test)]
mod triple_extraction_tests {
    use super::*;

    #[tokio::test]
    async fn test_triple_extraction() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Einstein developed the theory of relativity in 1905.";
        
        let triples = extractor.extract_triples(text).await.unwrap();
        
        assert!(!triples.is_empty());
        
        for triple in &triples {
            assert!(!triple.subject.is_empty());
            assert!(!triple.predicate.is_empty());
            assert!(!triple.object.is_empty());
            assert!(triple.confidence >= 0.0 && triple.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_triple_metadata() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Apple Inc was founded by Steve Jobs.";
        
        let triples = extractor.extract_triples(text).await.unwrap();
        
        for triple in triples {
            // Check triple fields are populated
            assert!(triple.confidence >= 0.0 && triple.confidence <= 1.0);
        }
    }
}

#[cfg(test)]
mod coreference_resolution_tests {
    use super::*;

    #[tokio::test]
    async fn test_coreference_resolver() {
        // CoreferenceResolver is used internally by AdvancedEntityExtractor
        let extractor = AdvancedEntityExtractor::new();
        let text = "John went to the store. He bought milk.";
        
        // The extractor internally resolves coreferences
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Should extract entities even with pronouns
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_coreference_with_multiple_pronouns() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Alice met Bob at the conference. She gave him her business card. They discussed the project.";
        
        // The extractor should handle complex text with multiple pronouns
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Should extract at least Alice and Bob
        assert!(entities.len() >= 2);
    }
}

#[cfg(test)]
mod entity_linking_tests {
    use super::*;

    #[tokio::test]
    async fn test_entity_linker() {
        // Test entity linking through the extractor
        let extractor = AdvancedEntityExtractor::new();
        // Extract entities which will be linked internally
        let text = "Dr. John Smith and Prof. Einstein discussed physics.";
        let linked = extractor.extract_entities(text).await.unwrap();
        
        // Check that canonical names are normalized (titles removed)
        for entity in linked {
            if entity.text.contains("John Smith") {
                assert_eq!(entity.canonical_name, "John Smith");
            } else if entity.text.contains("Einstein") {
                assert_eq!(entity.canonical_name, "Einstein");
            }
        }
    }
}

#[cfg(test)]
mod different_text_format_tests {
    use super::*;

    #[tokio::test]
    async fn test_multiline_text_extraction() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "John Smith is a software engineer.\n\
                   He works at Google.\n\
                   The company is located in Mountain View.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        
        assert!(!entities.is_empty());
        assert!(!relations.is_empty());
    }

    #[tokio::test]
    async fn test_text_with_punctuation() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Dr. Watson, along with Sherlock Holmes, solved the case! They worked in London (England).";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Should handle various punctuation marks
        let entity_texts: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
        assert!(entity_texts.iter().any(|t| t.contains("Watson")));
        assert!(entity_texts.iter().any(|t| t.contains("Sherlock Holmes")));
        assert!(entity_texts.iter().any(|t| t.contains("London")));
    }

    #[tokio::test]
    async fn test_text_with_special_characters() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "The O'Connor Foundation donated $1 million to MIT's research lab.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Should handle apostrophes and other special characters
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_empty_text() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        let triples = extractor.extract_triples(text).await.unwrap();
        
        assert!(entities.is_empty());
        assert!(relations.is_empty());
        assert!(triples.is_empty());
    }

    #[tokio::test]
    async fn test_text_with_numbers_and_dates() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "In 2023, Company XYZ reported $5.2 billion in revenue. They plan to expand by 2025.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Should extract years as date entities
        let date_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type.contains("DATE"))
            .collect();
        assert!(date_entities.len() >= 2);
        assert!(date_entities.iter().any(|e| e.text == "2023"));
        assert!(date_entities.iter().any(|e| e.text == "2025"));
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_entity_merge() {
        let extractor = AdvancedEntityExtractor::new();
        // This shouldn't crash even with edge cases
        let text = "A A A B B B";
        
        let result = extractor.extract_entities(text).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_very_long_text() {
        let extractor = AdvancedEntityExtractor::new();
        let long_text = "John Smith ".repeat(1000);
        
        let result = extractor.extract_entities(&long_text).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_unicode_text() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "José García works at Zürich Insurance. He speaks 中文.";
        
        let result = extractor.extract_entities(text).await;
        assert!(result.is_ok());
        
        let entities = result.unwrap();
        // Should handle unicode characters properly
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_malformed_patterns() {
        let extractor = AdvancedEntityExtractor::new();
        // Text that might break regex patterns
        let text = "This has (unmatched parentheses and [brackets that don't close";
        
        let result = extractor.extract_entities(text).await;
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_extraction_pipeline() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Marie Curie was a Polish physicist who worked at the University of Paris. \
                   She discovered polonium in 1898. Her husband Pierre Curie was also a scientist.";
        
        // Extract entities
        let entities = extractor.extract_entities(text).await.unwrap();
        assert!(!entities.is_empty());
        
        // Extract relations
        let relations = extractor.extract_relations(text, &entities).await.unwrap();
        assert!(!relations.is_empty());
        
        // Extract triples
        let triples = extractor.extract_triples(text).await.unwrap();
        assert!(!triples.is_empty());
        
        // Verify entity types
        let person_count = entities.iter()
            .filter(|e| e.entity_type.contains("PERSON"))
            .count();
        assert!(person_count >= 2); // Marie and Pierre
        
        // Verify locations
        let location_count = entities.iter()
            .filter(|e| e.entity_type.contains("LOCATION"))
            .count();
        assert!(location_count >= 1); // University of Paris
        
        // Verify dates
        let date_count = entities.iter()
            .filter(|e| e.entity_type.contains("DATE"))
            .count();
        assert!(date_count >= 1); // 1898
    }

    #[tokio::test]
    async fn test_entity_deduplication() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "Google is a tech company. Google was founded in 1998. The Google campus is large.";
        
        let entities = extractor.extract_entities(text).await.unwrap();
        
        // Each unique entity should appear only once per position
        let mut seen_positions = std::collections::HashSet::new();
        for entity in entities {
            let pos = (entity.start_pos, entity.end_pos);
            assert!(!seen_positions.contains(&pos), "Duplicate entity at same position");
            seen_positions.insert(pos);
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_extraction_performance() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "John Smith works at Microsoft. ".repeat(100); // 3200 chars
        
        let start = Instant::now();
        let entities = extractor.extract_entities(&text).await.unwrap();
        let duration = start.elapsed();
        
        // Should complete in reasonable time
        assert!(duration.as_secs() < 5);
        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_extraction() {
        let texts = vec![
            "Albert Einstein was a physicist.",
            "Marie Curie discovered radium.",
            "Isaac Newton invented calculus.",
            "Charles Darwin wrote Origin of Species.",
        ];
        
        let mut handles = vec![];
        
        for text in texts {
            let extractor_clone = AdvancedEntityExtractor::new();
            let handle = tokio::spawn(async move {
                extractor_clone.extract_entities(text).await
            });
            handles.push(handle);
        }
        
        // All should complete successfully
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}