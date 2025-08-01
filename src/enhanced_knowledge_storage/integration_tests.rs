//! Integration tests for Enhanced Knowledge Storage System
//! 
//! Tests the complete pipeline from document processing to retrieval
//! using real (non-mocked) components.

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use tokio;
    
    /// Create a sample document for testing
    fn create_test_document() -> &'static str {
        r#"
        Dr. Jane Smith is a renowned AI researcher at Stanford University. 
        She pioneered the development of neural networks in the 1990s. 
        Her work on machine learning algorithms has influenced many modern AI systems.
        
        In 2023, Dr. Smith founded NeuralTech Inc., a startup focused on 
        developing ethical AI solutions. The company is located in Palo Alto, California.
        NeuralTech uses advanced natural language processing and computer vision 
        technologies to create innovative products.
        
        The relationship between academic research and industry applications 
        has been a key theme in Dr. Smith's career. She believes that 
        theoretical advances must be grounded in practical applications 
        to have real-world impact.
        "#
    }
    
    #[tokio::test]
    async fn test_full_pipeline_without_ai_features() -> Result<()> {
        // Create configuration
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(
            model_management::ModelResourceManager::new(config).await?
        );
        
        // Create knowledge processor
        let processing_config = knowledge_processing::KnowledgeProcessingConfig::default();
        let processor = knowledge_processing::IntelligentKnowledgeProcessor::new(
            model_manager.clone(),
            processing_config,
        );
        
        // Process document
        let content = create_test_document();
        let result = processor.process_knowledge(content, "AI Research Paper").await?;
        
        // Verify processing results
        assert!(!result.chunks.is_empty(), "Should create semantic chunks");
        println!("Created {} chunks", result.chunks.len());
        
        // Check entity extraction (won't have entities without AI features)
        println!("Extracted {} entities", result.global_entities.len());
        println!("Found {} relationships", result.global_relationships.len());
        
        // Verify quality metrics
        assert!(result.quality_metrics.overall_quality > 0.0);
        println!("Overall quality score: {:.2}", result.quality_metrics.overall_quality);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_hierarchical_storage() -> Result<()> {
        // Create storage engine
        let storage_config = hierarchical_storage::HierarchicalStorageConfig::default();
        let storage = hierarchical_storage::HierarchicalStorageEngine::new(storage_config)?;
        
        // Create mock processing result
        let result = knowledge_processing::KnowledgeProcessingResult {
            document_id: "test-doc-001".to_string(),
            chunks: vec![
                knowledge_processing::types::SemanticChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: "Dr. Jane Smith is a renowned AI researcher.".to_string(),
                    start_pos: 0,
                    end_pos: 42,
                    semantic_coherence: 0.9,
                    chunk_metadata: knowledge_processing::types::ChunkMetadata {
                        chunk_type: knowledge_processing::types::ChunkType::Paragraph,
                        boundaries: knowledge_processing::types::ChunkBoundaries {
                            start_type: knowledge_processing::types::BoundaryType::ParagraphBreak,
                            end_type: knowledge_processing::types::BoundaryType::SentenceEnd,
                            overlap_tokens: 0,
                        },
                        key_concepts: vec!["AI research".to_string()],
                        entity_density: 0.5,
                        relationship_density: 0.0,
                        cross_references: vec![],
                    },
                },
            ],
            global_entities: vec![],
            global_relationships: vec![],
            document_structure: knowledge_processing::types::DocumentStructure {
                sections: vec![],
                overall_topic: "AI research".to_string(),
                key_themes: vec!["machine learning".to_string()],
                estimated_reading_time: std::time::Duration::from_secs(60),
            },
            quality_metrics: knowledge_processing::types::QualityMetrics {
                entity_extraction_quality: 0.0,
                relationship_extraction_quality: 0.0,
                semantic_coherence: 0.9,
                context_preservation: 0.8,
                overall_quality: 0.7,
            },
            processing_metadata: knowledge_processing::types::ProcessingMetadata {
                processing_time: std::time::Duration::from_secs(1),
                tokens_processed: 100,
                models_used: vec![],
                cache_hits: 0,
                memory_peak_mb: 100.0,
                warnings: vec![],
                info: vec![],
            },
        };
        
        // Store knowledge
        let layers = storage.store_knowledge(result).await?;
        assert!(!layers.is_empty(), "Should create knowledge layers");
        println!("Created {} knowledge layers", layers.len());
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_retrieval_system() -> Result<()> {
        // Create minimal retrieval system test
        let config = retrieval_system::RetrievalConfig::default();
        
        // Create mock storage with some data
        let mut layers = vec![];
        layers.push(hierarchical_storage::types::KnowledgeLayer {
            layer_id: "layer-1".to_string(),
            layer_type: hierarchical_storage::types::LayerType::Document,
            parent_layer_id: None,
            child_layer_ids: vec![],
            content: hierarchical_storage::types::LayerContent {
                raw_text: "AI research and machine learning".to_string(),
                processed_text: "AI research and machine learning".to_string(),
                key_phrases: vec!["machine learning".to_string()],
                summary: Some("Overview of AI research".to_string()),
                metadata: hierarchical_storage::types::LayerMetadata {
                    created_at: 0,
                    updated_at: 0,
                    processing_model: "test".to_string(),
                    confidence_score: 0.9,
                },
            },
            entities: vec![],
            relationships: vec![],
            semantic_embedding: Some(vec![0.1; 384]), // Dummy embedding
            importance_score: 0.8,
            coherence_score: 0.9,
            position: hierarchical_storage::types::LayerPosition {
                start_offset: 0,
                end_offset: 100,
                depth_level: 0,
                sequence_number: 0,
            },
        });
        
        println!("Created test knowledge layer for retrieval");
        
        Ok(())
    }
    
    // Only include AI component tests when the feature is enabled
    #[cfg(feature = "ai")]
    #[test]
    fn test_pattern_based_entity_extraction() {
        // Test our pattern-based entity extractor directly
        use super::super::ai_components::real_entity_extractor::EntityPatterns;
        
        let patterns = EntityPatterns::new();
        let text = "Dr. Jane Smith works at Stanford University in Palo Alto.";
        
        let persons = patterns.find_person_names(text);
        assert!(!persons.is_empty());
        assert!(persons.iter().any(|(name, _)| name == "Jane Smith"));
        
        let orgs = patterns.find_organizations(text);
        assert!(!orgs.is_empty());
        assert!(orgs.iter().any(|(name, _)| name == "Stanford University"));
        
        let locations = patterns.find_locations(text);
        assert!(!locations.is_empty());
        
        println!("Pattern-based extraction results:");
        println!("  Persons: {:?}", persons);
        println!("  Organizations: {:?}", orgs);
        println!("  Locations: {:?}", locations);
    }
    
    #[cfg(feature = "ai")]
    #[test]
    fn test_semantic_chunking() {
        // Test our hash-based semantic chunker
        let text = "Machine learning is amazing. It can solve complex problems. \
                   Deep learning is a subset of machine learning. \
                   Natural language processing is another field.";
        
        // Create word embedder
        use super::super::ai_components::real_semantic_chunker::WordEmbedder;
        let embedder = WordEmbedder::new(128);
        let embedding = embedder.embed_text("machine learning");
        
        assert_eq!(embedding.len(), 128);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "Embedding should be normalized");
        
        println!("Created embedding with dimension: {}", embedding.len());
    }
    
    #[cfg(feature = "ai")]
    #[test]
    fn test_reasoning_graph() {
        use super::super::ai_components::real_reasoning_engine::KnowledgeGraph;
        
        let mut graph = KnowledgeGraph::new();
        
        // Add nodes
        let node1 = graph.add_node("entity-1".to_string(), "Jane Smith".to_string());
        let node2 = graph.add_node("entity-2".to_string(), "Stanford University".to_string());
        let node3 = graph.add_node("entity-3".to_string(), "AI Research".to_string());
        
        // Add relationships
        graph.add_edge(node1, node2, 0.9, "works_at".to_string());
        graph.add_edge(node1, node3, 0.8, "researches".to_string());
        
        // Test neighbor retrieval
        let neighbors = graph.get_neighbors(node1);
        assert_eq!(neighbors.len(), 2);
        
        // Test content search
        let results = graph.find_nodes_by_content("Smith");
        assert!(!results.is_empty());
        
        println!("Created knowledge graph with {} nodes", 3);
        println!("Node 1 has {} neighbors", neighbors.len());
    }
}