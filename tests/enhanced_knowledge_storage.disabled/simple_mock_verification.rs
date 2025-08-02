//! Simple Mock System Verification Tests
//! 
//! Basic verification tests for mock components to ensure they work correctly.

#[cfg(test)]
mod simple_mock_tests {
    use crate::enhanced_knowledge_storage::mocks::*;

    #[test]
    fn test_mock_hierarchical_storage_basic() {
        let storage = MockHierarchicalStorage::new();
        
        let entry = MockStorageEntry {
            id: "test_doc".to_string(),
            content: "Test content".to_string(),
            metadata: std::collections::HashMap::new(),
            relationships: vec!["related_1".to_string()],
        };
        
        storage.store_entry("test_doc".to_string(), entry.clone(), StorageTier::Hot);
        let retrieved = storage.retrieve_entry("test_doc");
        
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.id, "test_doc");
        assert_eq!(retrieved_entry.content, "Test content");
    }
    
    #[test]
    fn test_mock_text_processor_basic() {
        let processor = MockTextProcessor::new();
        let result = processor.process_text("Test text for processing");
        
        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.relationships.len(), 1);
        assert_eq!(result.quality_score, 0.85);
        assert!(result.quality_score > 0.8);
    }
    
    #[test]
    fn test_mock_embedding_generator_basic() {
        let generator = MockEmbeddingGenerator::new(384);
        let embedding = generator.generate_embedding("test text");
        
        assert_eq!(embedding.len(), 384);
        
        // Test consistency
        let embedding2 = generator.generate_embedding("test text");
        assert_eq!(embedding, embedding2);
    }
    
    #[test] 
    fn test_mock_entity_extractor_basic() {
        let extractor = MockEntityExtractor::new();
        let entities = extractor.extract_entities("Test text with entities");
        
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "MockEntity1");
        assert_eq!(entities[1].name, "MockEntity2");
    }
    
    #[test]
    fn test_mock_similarity_calculator_basic() {
        let calculator = MockSimilarityCalculator::new();
        
        let vec_a = vec![1.0, 0.0, 0.0];
        let similarity = calculator.cosine_similarity(&vec_a, &vec_a);
        assert!((similarity - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_mock_embedding_index_basic() {
        let index = MockEmbeddingIndex::new(3);
        
        let result = index.add_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0]);
        assert!(result.is_ok());
        
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search_similar(&query, 2);
        assert!(!results.is_empty());
    }
}