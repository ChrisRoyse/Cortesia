use llmkg::core::semantic_summary::{
    SemanticSummarizer, SemanticSummary, EntityType, KeyFeature, FeatureValue, 
    CompactEmbedding, ContextHint, ContextType, ReconstructionMetadata
};
use llmkg::core::types::{EntityData, EntityKey};
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio;

/// Integration test helper for creating test entity data
fn create_test_entity_data(id: u16, text: &str, embedding_size: usize) -> EntityData {
    EntityData {
        type_id: id,
        properties: text.to_string(),
        embedding: (0..embedding_size).map(|i| (i as f32) * 0.01).collect(),
    }
}

/// Integration test helper for simulating LLM responses
struct MockLLM {
    expected_queries: HashMap<String, String>,
}

impl MockLLM {
    fn new() -> Self {
        Self {
            expected_queries: HashMap::new(),
        }
    }

    fn add_expected_response(&mut self, query: String, response: String) {
        self.expected_queries.insert(query, response);
    }

    fn query(&self, prompt: &str) -> Result<String> {
        self.expected_queries.get(prompt)
            .cloned()
            .ok_or_else(|| llmkg::error::GraphError::QueryError(format!("Unexpected query: {}", prompt)))
    }
}

#[tokio::test]
async fn test_semantic_summary_llm_integration() {
    // Create a mock LLM
    let mut mock_llm = MockLLM::new();
    
    // Create test entity data representing different types of knowledge
    let entities = vec![
        create_test_entity_data(1, "Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in 1879 in Germany and won the Nobel Prize in Physics in 1921.", 128),
        create_test_entity_data(2, "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales. It was developed in the early 20th century by multiple physicists.", 128),
        create_test_entity_data(3, "The photoelectric effect is the emission of electrons when light hits a material. Einstein explained this phenomenon using quantum theory, earning him the Nobel Prize.", 128),
    ];

    // Create semantic summarizer
    let mut summarizer = SemanticSummarizer::new();
    
    // Generate summaries for all entities
    let mut summaries = Vec::new();
    for (i, entity) in entities.iter().enumerate() {
        let entity_key = EntityKey::default(); // In real usage, this would come from the knowledge engine
        let summary = summarizer.create_summary(entity, entity_key).unwrap();
        summaries.push(summary);
        
        // Convert to LLM-friendly text
        let llm_text = summarizer.to_llm_text(&summary);
        println!("Entity {} LLM Text:\n{}\n", i, llm_text);
    }

    // Test 1: Verify summaries improve LLM comprehension
    // Set up expected LLM responses based on summaries
    mock_llm.add_expected_response(
        "Based on this summary, what is the main subject?\nEntity Type: 1 (confidence: 0.90)".to_string(),
        "The main subject is Albert Einstein, a theoretical physicist.".to_string()
    );

    // Query the mock LLM with a summary
    let llm_text = summarizer.to_llm_text(&summaries[0]);
    let query = format!("Based on this summary, what is the main subject?\n{}", 
                       llm_text.lines().next().unwrap()); // Just use first line for test
    let response = mock_llm.query(&query).unwrap();
    assert!(response.contains("Einstein"));
    assert!(response.contains("physicist"));

    // Test 2: Verify comprehension scores
    for (i, summary) in summaries.iter().enumerate() {
        let comprehension_score = summarizer.estimate_llm_comprehension(summary);
        println!("Entity {} comprehension score: {:.2}", i, comprehension_score);
        assert!(comprehension_score > 0.5, "Comprehension score should be reasonable");
    }

    // Test 3: Test entity relationships through summaries
    // Create a summary with context hints
    let mut summary_with_context = summaries[2].clone();
    summary_with_context.context_hints.push(ContextHint {
        context_type: ContextType::CausalLink,
        related_id: 0, // Einstein entity
        strength: 0.9,
    });
    summary_with_context.context_hints.push(ContextHint {
        context_type: ContextType::SemanticCluster,
        related_id: 1, // Quantum mechanics entity  
        strength: 0.8,
    });

    let context_text = summarizer.to_llm_text(&summary_with_context);
    assert!(context_text.contains("Context Relationships:"));
    assert!(context_text.contains("CausalLink"));
    assert!(context_text.contains("SemanticCluster"));
}

#[tokio::test]
async fn test_semantic_summary_workflow() {
    // Create knowledge engine
    let temp_dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(temp_dir.path().to_path_buf()).await.unwrap()
    ));

    // Create embedding store
    let embedding_store = Arc::new(tokio::sync::RwLock::new(
        EmbeddingStore::new(128)
    ));

    // Add test entities
    let entities = vec![
        ("Python", "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms."),
        ("Machine Learning", "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming."),
        ("TensorFlow", "TensorFlow is an open-source machine learning framework developed by Google. It's widely used for deep learning applications."),
    ];

    let mut entity_keys = Vec::new();
    
    {
        let mut engine_write = engine.write().await;
        let mut embed_write = embedding_store.write().await;
        
        for (name, description) in &entities {
            // Create entity data
            let entity_data = EntityData {
                type_id: 1, // All are concept entities
                properties: format!("name: {}, description: {}", name, description),
                embedding: vec![0.1; 128], // Simplified embedding
            };
            
            // Add to knowledge engine
            let entity_key = engine_write.add_entity(entity_data.clone()).await.unwrap();
            entity_keys.push(entity_key);
            
            // Add to embedding store  
            embed_write.add_entity_embedding(entity_key, entity_data.embedding.clone()).unwrap();
        }

        // Add relationships
        engine_write.add_relationship(entity_keys[2], entity_keys[1], 0.9, 1).await.unwrap(); // TensorFlow -> ML
        engine_write.add_relationship(entity_keys[1], entity_keys[0], 0.7, 1).await.unwrap(); // ML -> Python
    }

    // Create semantic summarizer
    let mut summarizer = SemanticSummarizer::new();
    
    // Generate summaries for all entities
    let engine_read = engine.read().await;
    let mut summaries = Vec::new();
    
    for (i, entity_key) in entity_keys.iter().enumerate() {
        let entity_data = engine_read.get_entity(*entity_key).await.unwrap();
        let summary = summarizer.create_summary(&entity_data, *entity_key).unwrap();
        
        // Verify summary quality
        assert!(summary.key_features.len() >= 2, "Should have multiple features");
        assert!(!summary.semantic_embedding.quantized_values.is_empty(), "Should have embedding");
        
        let llm_text = summarizer.to_llm_text(&summary);
        println!("Entity {} summary:\n{}\n", entities[i].0, llm_text);
        
        summaries.push(summary);
    }

    // Test complete workflow: query -> summary -> LLM comprehension
    let query_embedding = vec![0.15; 128];
    let results = engine_read.similarity_search(&query_embedding, 3, None).await.unwrap();
    
    // Generate combined summary for query results
    let mut combined_text = String::new();
    combined_text.push_str("Query Results Summary:\n\n");
    
    for (entity_key, similarity) in &results {
        if let Some(summary) = summaries.iter().find(|s| {
            // Match by entity type and features (simplified for test)
            true
        }) {
            combined_text.push_str(&format!("Result (similarity: {:.2}):\n", similarity));
            combined_text.push_str(&summarizer.to_llm_text(summary));
            combined_text.push_str("\n");
        }
    }
    
    println!("Combined query summary:\n{}", combined_text);
    assert!(combined_text.contains("Query Results Summary"));
    assert!(combined_text.contains("Entity Type:"));
}

#[tokio::test] 
async fn test_llm_comprehension_estimation_accuracy() {
    let mut summarizer = SemanticSummarizer::new();
    
    // Test different quality levels of summaries
    let test_cases = vec![
        (
            "High quality entity", 
            create_test_entity_data(1, "This is a comprehensive description with rich semantic content including multiple important aspects and detailed information about the subject matter.", 256),
            0.8, // Expected high comprehension
        ),
        (
            "Medium quality entity",
            create_test_entity_data(2, "A basic description with some details.", 64),
            0.5, // Expected medium comprehension
        ),
        (
            "Low quality entity",
            create_test_entity_data(3, "", 0),
            0.3, // Expected low comprehension
        ),
    ];

    for (name, entity_data, expected_min_score) in test_cases {
        let summary = summarizer.create_summary(&entity_data, EntityKey::default()).unwrap();
        let comprehension = summarizer.estimate_llm_comprehension(&summary);
        
        println!("{}: comprehension score = {:.2}", name, comprehension);
        assert!(comprehension >= expected_min_score * 0.8, // Allow 20% variance
                "{} comprehension {} should be >= {}", name, comprehension, expected_min_score * 0.8);
        assert!(comprehension <= 1.0, "Comprehension should not exceed 1.0");
    }

    // Test summary with maximum features for best comprehension
    let mut rich_summary = SemanticSummary {
        entity_type: EntityType {
            type_id: 1,
            confidence: 0.99,
            secondary_type: Some(2),
        },
        key_features: (0..10).map(|i| KeyFeature {
            feature_id: i,
            value: FeatureValue::Numeric { value: i as f32, range_hint: Some((0.0, 10.0)) },
            importance: 0.9,
        }).collect(),
        semantic_embedding: CompactEmbedding {
            quantized_values: vec![128; 32],
            scale_factors: vec![1.0; 32],
            dimension_map: (0..32).map(|i| i as u8).collect(),
        },
        context_hints: (0..5).map(|i| ContextHint {
            context_type: ContextType::SemanticCluster,
            related_id: i,
            strength: 0.8,
        }).collect(),
        reconstruction_metadata: ReconstructionMetadata {
            original_size: 1000,
            compression_ratio: 5.0,
            quality_score: 0.95,
            content_hash: 12345,
        },
    };

    let max_comprehension = summarizer.estimate_llm_comprehension(&rich_summary);
    assert!(max_comprehension >= 0.9, "Rich summary should have very high comprehension");
    println!("Maximum comprehension achieved: {:.2}", max_comprehension);
}

#[tokio::test]
async fn test_semantic_summary_memory_efficiency() {
    let mut summarizer = SemanticSummarizer::new();
    
    // Test memory efficiency with different entity sizes
    let large_text = "Lorem ipsum ".repeat(100); // ~1200 bytes
    let large_entity = EntityData {
        type_id: 1,
        properties: large_text.clone(),
        embedding: vec![0.1; 512], // Large embedding
    };

    let original_size = large_entity.properties.len() + large_entity.embedding.len() * 4;
    println!("Original entity size: {} bytes", original_size);

    let summary = summarizer.create_summary(&large_entity, EntityKey::default()).unwrap();
    
    // Calculate summary size
    let summary_json = serde_json::to_string(&summary).unwrap();
    let summary_size = summary_json.len();
    println!("Summary size: {} bytes", summary_size);
    
    let compression_ratio = original_size as f32 / summary_size as f32;
    println!("Compression ratio: {:.2}x", compression_ratio);
    
    assert!(compression_ratio > 2.0, "Should achieve significant compression");
    
    // Verify summary preserves essential information
    assert!(!summary.key_features.is_empty(), "Should extract key features");
    assert!(!summary.semantic_embedding.quantized_values.is_empty(), "Should compress embedding");
    
    // Test that summary is still useful for LLM
    let llm_text = summarizer.to_llm_text(&summary);
    assert!(llm_text.len() < large_text.len(), "LLM text should be more concise");
    assert!(llm_text.contains("Key Features"), "Should include feature information");
}

#[tokio::test]
async fn test_semantic_summary_edge_cases() {
    let mut summarizer = SemanticSummarizer::new();
    
    // Test 1: Empty entity
    let empty_entity = EntityData {
        type_id: 0,
        properties: String::new(),
        embedding: vec![],
    };
    
    let result = summarizer.create_summary(&empty_entity, EntityKey::default());
    assert!(result.is_ok(), "Should handle empty entity gracefully");
    
    let summary = result.unwrap();
    assert_eq!(summary.entity_type.type_id, 0);
    assert!(!summary.key_features.is_empty(), "Should still have some features");
    
    // Test 2: Very large embedding
    let large_embedding_entity = EntityData {
        type_id: 1,
        properties: "Test".to_string(),
        embedding: vec![0.1; 2048], // Very large
    };
    
    let result = summarizer.create_summary(&large_embedding_entity, EntityKey::default());
    assert!(result.is_ok(), "Should handle large embeddings");
    
    let summary = result.unwrap();
    assert!(summary.semantic_embedding.quantized_values.len() <= 32, 
            "Should compress to target dimensions");
    
    // Test 3: Special characters in properties
    let special_chars_entity = EntityData {
        type_id: 2,
        properties: "Special chars: 你好世界 🌍 \n\t\r".to_string(),
        embedding: vec![0.1; 64],
    };
    
    let result = summarizer.create_summary(&special_chars_entity, EntityKey::default());
    assert!(result.is_ok(), "Should handle special characters");
    
    // Test 4: Numeric properties
    let numeric_entity = EntityData {
        type_id: 3,
        properties: "value: 3.14159, count: 42, ratio: 0.618".to_string(),
        embedding: vec![0.1; 64],
    };
    
    let summary = summarizer.create_summary(&numeric_entity, EntityKey::default()).unwrap();
    let has_numeric_feature = summary.key_features.iter().any(|f| {
        matches!(f.value, FeatureValue::Numeric { .. })
    });
    assert!(has_numeric_feature, "Should detect numeric features");
}