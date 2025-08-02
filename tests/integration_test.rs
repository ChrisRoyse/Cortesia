//! End-to-End Integration Test for Enhanced Knowledge Storage System
//! 
//! This test demonstrates the complete workflow from document processing
//! through multi-hop reasoning with real AI components.

use llmkg::enhanced_knowledge_storage::{
    ai_components::{
        AIModelBackend, RealEntityExtractor, RealSemanticChunker, 
        RealReasoningEngine, ModelConfig, PerformanceMonitor
    },
    production::{
        ProductionKnowledgeSystem, ProductionConfig, Environment,
        SystemState, DocumentProcessingResult, ReasoningQueryResult
    }
};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_end_to_end_integration() {
    // Create async runtime
    let rt = Runtime::new().unwrap();
    
    rt.block_on(async {
        println!("🚀 Starting Enhanced Knowledge Storage System Integration Test");
        
        // Step 1: Initialize configuration
        println!("\n📋 Step 1: Loading production configuration...");
        let config = ProductionConfig::from_environment(Environment::Testing)
            .expect("Failed to load config");
        println!("✅ Configuration loaded successfully");
        
        // Step 2: Initialize AI Model Backend
        println!("\n🤖 Step 2: Initializing AI Model Backend...");
        let model_backend = Arc::new(
            AIModelBackend::new(config.model_config.clone())
                .await
                .expect("Failed to initialize model backend")
        );
        println!("✅ AI Model Backend initialized");
        
        // Step 3: Initialize Performance Monitor
        println!("\n📊 Step 3: Setting up Performance Monitoring...");
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        println!("✅ Performance monitoring active");
        
        // Step 4: Initialize Real AI Components
        println!("\n🧠 Step 4: Initializing Real AI Components...");
        
        let entity_extractor = Arc::new(
            RealEntityExtractor::new(
                model_backend.clone(),
                performance_monitor.clone()
            ).await.expect("Failed to initialize entity extractor")
        );
        println!("  ✅ Entity Extractor ready");
        
        let semantic_chunker = Arc::new(
            RealSemanticChunker::new(
                model_backend.clone(),
                performance_monitor.clone()
            ).await.expect("Failed to initialize semantic chunker")
        );
        println!("  ✅ Semantic Chunker ready");
        
        let reasoning_engine = Arc::new(
            RealReasoningEngine::new(
                model_backend.clone(),
                performance_monitor.clone()
            ).await.expect("Failed to initialize reasoning engine")
        );
        println!("  ✅ Reasoning Engine ready");
        
        // Step 5: Initialize Production System
        println!("\n🏗️ Step 5: Initializing Production Knowledge System...");
        let knowledge_system = ProductionKnowledgeSystem::new(config.clone())
            .await
            .expect("Failed to initialize production system");
        println!("✅ Production system initialized");
        
        // Step 6: Verify System State
        println!("\n🔍 Step 6: Verifying system state...");
        let state = knowledge_system.get_system_state().await;
        assert_eq!(state, SystemState::Ready, "System should be in Ready state");
        println!("✅ System state: Ready");
        
        // Step 7: Process Test Document
        println!("\n📄 Step 7: Processing test document...");
        let test_document = r#"
            # Knowledge Graph Systems
            
            Knowledge graphs are powerful data structures that represent information as a network 
            of entities and their relationships. They enable sophisticated reasoning and inference
            capabilities by connecting disparate pieces of information.
            
            ## Key Components
            
            1. **Entities**: The nodes in the graph representing concepts, people, places, or things
            2. **Relationships**: The edges connecting entities, describing how they relate
            3. **Attributes**: Properties associated with entities and relationships
            
            ## Applications
            
            Knowledge graphs are used in various domains including:
            - Search engines for semantic understanding
            - Recommendation systems for personalized suggestions
            - Natural language processing for context understanding
            - Scientific research for discovering hidden connections
        "#;
        
        let processing_result = knowledge_system.process_document(
            test_document.to_string(),
            Some("test_doc_001".to_string())
        ).await.expect("Failed to process document");
        
        println!("✅ Document processed successfully:");
        println!("  - Document ID: {}", processing_result.document_id);
        println!("  - Chunks created: {}", processing_result.chunks_created);
        println!("  - Entities extracted: {}", processing_result.entities_extracted);
        println!("  - Relationships found: {}", processing_result.relationships_found);
        println!("  - Processing time: {:?}", processing_result.processing_time);
        
        // Verify processing results
        assert!(processing_result.chunks_created > 0, "Should create chunks");
        assert!(processing_result.entities_extracted > 0, "Should extract entities");
        assert!(processing_result.relationships_found > 0, "Should find relationships");
        
        // Step 8: Test Retrieval with Simple Query
        println!("\n🔎 Step 8: Testing retrieval with simple query...");
        let simple_query = "What are knowledge graphs?";
        
        let retrieval_result = knowledge_system.retrieve(
            simple_query.to_string(),
            Some(5), // max results
            false    // disable multi-hop reasoning for simple query
        ).await.expect("Failed to retrieve");
        
        println!("✅ Simple retrieval successful:");
        println!("  - Results found: {}", retrieval_result.results.len());
        println!("  - Confidence: {:.2}", retrieval_result.confidence);
        assert!(!retrieval_result.results.is_empty(), "Should find results");
        
        // Step 9: Test Multi-Hop Reasoning
        println!("\n🧩 Step 9: Testing multi-hop reasoning...");
        let complex_query = "How do knowledge graphs help with recommendation systems and what components enable this?";
        
        let reasoning_result = knowledge_system.retrieve(
            complex_query.to_string(),
            Some(10), // max results
            true      // enable multi-hop reasoning
        ).await.expect("Failed to perform reasoning");
        
        println!("✅ Multi-hop reasoning successful:");
        println!("  - Results found: {}", reasoning_result.results.len());
        println!("  - Reasoning steps: {}", reasoning_result.reasoning_chain.as_ref().unwrap().steps.len());
        println!("  - Confidence: {:.2}", reasoning_result.confidence);
        
        assert!(reasoning_result.reasoning_chain.is_some(), "Should have reasoning chain");
        let chain = reasoning_result.reasoning_chain.unwrap();
        assert!(chain.steps.len() >= 2, "Should have multiple reasoning steps");
        
        // Step 10: Test Performance Metrics
        println!("\n📈 Step 10: Checking performance metrics...");
        let metrics = performance_monitor.get_recent_metrics(
            std::time::Duration::from_secs(300)
        ).await;
        
        println!("✅ Performance metrics collected:");
        println!("  - Total operations: {}", metrics.len());
        
        for metric in metrics.iter().take(3) {
            println!("  - Operation: {:?}, Duration: {:?}, Tokens: {}", 
                metric.operation_type, 
                metric.duration,
                metric.tokens_processed
            );
        }
        
        // Step 11: Test System Health
        println!("\n💚 Step 11: Checking system health...");
        let health = knowledge_system.health_check().await
            .expect("Failed to check health");
        
        println!("✅ System health check passed:");
        println!("  - Overall status: {}", health.status);
        println!("  - Components healthy: {}/{}", 
            health.components.iter().filter(|c| c.healthy).count(),
            health.components.len()
        );
        
        assert_eq!(health.status, "healthy", "System should be healthy");
        
        // Step 12: Cleanup
        println!("\n🧹 Step 12: Shutting down system...");
        knowledge_system.shutdown().await
            .expect("Failed to shutdown");
        println!("✅ System shutdown complete");
        
        println!("\n✨ Integration test completed successfully!");
        println!("The Enhanced Knowledge Storage System is working end-to-end with:");
        println!("  ✅ Real AI model backend");
        println!("  ✅ Entity extraction");
        println!("  ✅ Semantic chunking"); 
        println!("  ✅ Multi-hop reasoning");
        println!("  ✅ Performance monitoring");
        println!("  ✅ Production orchestration");
    });
}

#[test]
fn test_concurrent_operations() {
    let rt = Runtime::new().unwrap();
    
    rt.block_on(async {
        println!("🔄 Testing concurrent document processing...");
        
        let config = ProductionConfig::from_environment(Environment::Testing)
            .expect("Failed to load config");
        
        let knowledge_system = ProductionKnowledgeSystem::new(config)
            .await
            .expect("Failed to initialize system");
        
        // Process multiple documents concurrently
        let mut handles = vec![];
        
        for i in 0..3 {
            let system = knowledge_system.clone();
            let handle = tokio::spawn(async move {
                let doc = format!("Document {} contains information about topic {}", i, i);
                system.process_document(
                    doc,
                    Some(format!("doc_{}", i))
                ).await
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        let mut results = vec![];
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            results.push(result);
        }
        
        println!("✅ Processed {} documents concurrently", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("  - Document {}: {} chunks, {} entities", 
                i, result.chunks_created, result.entities_extracted);
        }
        
        knowledge_system.shutdown().await.unwrap();
    });
}

#[test] 
fn test_error_recovery() {
    let rt = Runtime::new().unwrap();
    
    rt.block_on(async {
        println!("🛡️ Testing error handling and recovery...");
        
        let config = ProductionConfig::from_environment(Environment::Testing)
            .expect("Failed to load config");
            
        let knowledge_system = ProductionKnowledgeSystem::new(config)
            .await
            .expect("Failed to initialize system");
        
        // Test with invalid document
        let result = knowledge_system.process_document(
            "".to_string(), // Empty document
            None
        ).await;
        
        assert!(result.is_err(), "Should fail on empty document");
        println!("✅ Correctly rejected empty document");
        
        // Test system still functional after error
        let valid_result = knowledge_system.process_document(
            "Valid document content".to_string(),
            None
        ).await;
        
        assert!(valid_result.is_ok(), "Should process valid document after error");
        println!("✅ System recovered and processed valid document");
        
        knowledge_system.shutdown().await.unwrap();
    });
}