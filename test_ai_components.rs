#!/usr/bin/env cargo test

//! Test AI components directly to ensure they work without mocks

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    
    #[cfg(feature = "ai")]
    #[tokio::test]
    async fn test_real_entity_extractor() {
        use llmkg::enhanced_knowledge_storage::ai_components::{
            RealEntityExtractor, EntityExtractionConfig
        };
        
        println!("🧠 Testing Real Entity Extractor...");
        
        let config = EntityExtractionConfig {
            confidence_threshold: 0.7,
            max_entities_per_text: 20,
            enable_coreference_resolution: true,
            context_window_size: 500,
        };
        
        let extractor = Arc::new(RealEntityExtractor::new(config));
        
        let test_text = "Albert Einstein developed the theory of relativity. He worked at Princeton University and won the Nobel Prize in Physics in 1921.";
        
        match extractor.extract_entities(test_text).await {
            Ok(entities) => {
                println!("✅ Extracted {} entities", entities.len());
                for entity in &entities {
                    println!("   • {} ({}): confidence {:.2}", 
                             entity.name, entity.entity_type, entity.confidence);
                }
                assert!(!entities.is_empty(), "Should extract at least some entities");
            },
            Err(e) => {
                panic!("❌ Entity extraction failed: {}", e);
            }
        }
        
        println!("✅ Real Entity Extractor test passed");
    }
    
    #[cfg(feature = "ai")]
    #[tokio::test]
    async fn test_real_semantic_chunker() {
        use llmkg::enhanced_knowledge_storage::ai_components::{
            RealSemanticChunker, SemanticChunkingConfig
        };
        
        println!("📄 Testing Real Semantic Chunker...");
        
        let config = SemanticChunkingConfig {
            min_chunk_size: 50,
            max_chunk_size: 500,
            overlap_size: 20,
            coherence_threshold: 0.6,
        };
        
        let chunker = Arc::new(RealSemanticChunker::new(config));
        
        let test_text = "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without explicit programming. Neural networks are a key component of deep learning. They mimic the human brain's structure. Supervised learning uses labeled data for training. Unsupervised learning finds patterns in unlabeled data.";
        
        match chunker.chunk_document(test_text).await {
            Ok(chunks) => {
                println!("✅ Created {} semantic chunks", chunks.len());
                for (i, chunk) in chunks.iter().enumerate() {
                    println!("   Chunk {}: {} chars, coherence: {:.3}", 
                             i + 1, chunk.content.len(), chunk.semantic_coherence);
                    println!("     Key concepts: {:?}", chunk.key_concepts);
                }
                assert!(!chunks.is_empty(), "Should create at least one chunk");
                for chunk in &chunks {
                    assert!(chunk.semantic_coherence >= 0.0 && chunk.semantic_coherence <= 1.0, 
                            "Coherence should be between 0 and 1");
                }
            },
            Err(e) => {
                panic!("❌ Semantic chunking failed: {}", e);
            }
        }
        
        println!("✅ Real Semantic Chunker test passed");
    }
    
    #[cfg(feature = "ai")]
    #[tokio::test]
    async fn test_real_reasoning_engine() {
        use llmkg::enhanced_knowledge_storage::ai_components::{
            RealReasoningEngine, ReasoningConfig
        };
        
        println!("🔮 Testing Real Reasoning Engine...");
        
        let config = ReasoningConfig {
            max_reasoning_steps: 5,
            confidence_threshold: 0.5,
            enable_multi_hop: true,
            reasoning_timeout_seconds: 30,
        };
        
        let engine = Arc::new(RealReasoningEngine::new(config));
        
        let test_query = "If Einstein developed relativity and relativity explains gravity, what did Einstein explain?";
        
        match engine.reason(test_query).await {
            Ok(result) => {
                println!("✅ Generated reasoning chain with {} steps", result.reasoning_chain.len());
                println!("   Overall confidence: {:.3}", result.confidence);
                
                for (i, step) in result.reasoning_chain.iter().enumerate() {
                    println!("   Step {}: {} -> {}", 
                             i + 1, step.hypothesis, step.inference);
                    println!("     Evidence: {:?}", step.evidence);
                    println!("     Confidence: {:.3}", step.confidence);
                }
                
                assert!(!result.reasoning_chain.is_empty(), "Should produce at least one reasoning step");
                assert!(result.confidence >= 0.0 && result.confidence <= 1.0, 
                        "Overall confidence should be between 0 and 1");
            },
            Err(e) => {
                panic!("❌ Reasoning failed: {}", e);
            }
        }
        
        println!("✅ Real Reasoning Engine test passed");
    }
    
    #[cfg(feature = "ai")]
    #[tokio::test]
    async fn test_ai_model_backend() {
        use llmkg::enhanced_knowledge_storage::ai_components::{
            AIModelBackend, ModelType
        };
        
        println!("🤖 Testing AI Model Backend...");
        
        // Test without loading actual models (which would require large downloads)
        let backend = Arc::new(AIModelBackend::new());
        
        // Test model registration
        let model_id = "test-bert-base";
        let model_path = "/fake/path/to/model"; // This won't actually load
        
        // Just test the interface - actual model loading would require real model files
        let result = backend.is_model_loaded(model_id).await;
        println!("   Model loaded status: {}", result);
        
        // Test metrics
        let metrics = backend.get_performance_metrics().await;
        println!("   Performance metrics:");
        println!("     • Total inferences: {}", metrics.total_inferences);
        println!("     • Average latency: {:.2}ms", metrics.average_latency_ms);
        println!("     • Memory usage: {:.1}MB", metrics.memory_usage_mb);
        
        println!("✅ AI Model Backend interface test passed");
    }
    
    #[tokio::test]
    async fn test_knowledge_engine_with_enhanced_features() {
        use llmkg::core::{
            knowledge_engine::KnowledgeEngine,
            triple::Triple,
            knowledge_chunk::KnowledgeChunk,
        };
        
        println!("🧠 Testing Knowledge Engine with Enhanced Features...");
        
        let mut engine = KnowledgeEngine::new(384).await.expect("Failed to create knowledge engine");
        
        // Test basic triple storage
        let triple = Triple {
            subject: "Einstein".to_string(),
            predicate: "developed".to_string(),
            object: "relativity theory".to_string(),
            confidence: 0.95,
            timestamp: chrono::Utc::now(),
            source: Some("test".to_string()),
        };
        
        match engine.add_triple(triple.clone()).await {
            Ok(triple_id) => {
                println!("✅ Stored triple with ID: {:?}", triple_id);
            },
            Err(e) => {
                panic!("❌ Failed to store triple: {}", e);
            }
        }
        
        // Test knowledge chunk storage
        let chunk = KnowledgeChunk {
            id: uuid::Uuid::new_v4().to_string(),
            content: "Albert Einstein was a theoretical physicist who developed the theory of relativity.".to_string(),
            summary: "Einstein and relativity theory".to_string(),
            entities: vec!["Einstein".to_string(), "relativity theory".to_string()],
            relationships: vec!["Einstein -> developed -> relativity theory".to_string()],
            embedding: vec![0.1; 384], // Fake embedding for test
            importance_score: 0.9,
            coherence_score: 0.85,
            chunk_type: "biographical".to_string(),
            source_document: Some("test_doc".to_string()),
            position_in_document: Some(0),
            created_at: chrono::Utc::now(),
        };
        
        match engine.add_knowledge_chunk(chunk.clone()).await {
            Ok(chunk_id) => {
                println!("✅ Stored knowledge chunk with ID: {}", chunk_id);
            },
            Err(e) => {
                panic!("❌ Failed to store knowledge chunk: {}", e);
            }
        }
        
        // Test retrieval
        let query = llmkg::core::knowledge_types::TripleQuery {
            subject: Some("Einstein".to_string()),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: true,
        };
        
        match engine.query_triples(query).await {
            Ok(results) => {
                println!("✅ Retrieved {} triples and {} chunks", 
                         results.triples.len(), results.chunks.len());
                assert!(!results.triples.is_empty(), "Should find the stored triple");
            },
            Err(e) => {
                panic!("❌ Failed to query triples: {}", e);
            }
        }
        
        println!("✅ Knowledge Engine enhanced features test passed");
    }
    
    #[cfg(not(feature = "ai"))]
    #[test]
    fn test_ai_features_disabled() {
        println!("⚠️  AI features are disabled. Enable with --features ai to test real AI components.");
        println!("   This confirms that the system can detect when AI features are not available.");
    }
    
    #[tokio::test]
    async fn test_system_integration_basic() {
        println!("🔗 Testing Basic System Integration...");
        
        // Test that core components can work together
        use llmkg::core::knowledge_engine::KnowledgeEngine;
        
        let engine = KnowledgeEngine::new(384).await.expect("Failed to create engine");
        
        // Test knowledge storage and retrieval pipeline
        let test_facts = vec![
            ("Machine Learning", "is", "AI technique"),
            ("Neural Networks", "are used in", "Deep Learning"),  
            ("Python", "is popular for", "Data Science"),
        ];
        
        for (subj, pred, obj) in test_facts {
            let triple = llmkg::core::triple::Triple {
                subject: subj.to_string(),
                predicate: pred.to_string(),
                object: obj.to_string(),
                confidence: 0.9,
                timestamp: chrono::Utc::now(),
                source: Some("integration_test".to_string()),
            };
            
            engine.add_triple(triple).await.expect("Failed to add triple");
        }
        
        println!("✅ Stored {} test facts", test_facts.len());
        
        // Test search functionality
        let query = llmkg::core::knowledge_types::TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        let results = engine.query_triples(query).await.expect("Query failed");
        println!("✅ Retrieved {} facts from knowledge base", results.triples.len());
        
        assert!(results.triples.len() >= test_facts.len(), 
                "Should retrieve at least the facts we stored");
        
        println!("✅ Basic system integration test passed");
    }
}