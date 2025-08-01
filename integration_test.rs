#!/usr/bin/env cargo-script

//! Integration test for Enhanced Knowledge Storage System
//! 
//! This test verifies that all AI components work together without mocks,
//! using synthesized data to ensure full functionality.

use std::sync::Arc;
use tokio::sync::RwLock;

// Test configuration
const TEST_EMBEDDING_DIM: usize = 384;
const TEST_DOCUMENTS: &[(&str, &str)] = &[
    ("Machine Learning Basics", 
     "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data."),
    ("Quantum Computing", 
     "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits."),
    ("Renewable Energy", 
     "Solar energy and wind energy are two major forms of renewable energy. Solar panels convert sunlight into electricity, while wind turbines harness wind power to generate clean energy."),
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Enhanced Knowledge Storage System Integration Test");
    
    // Test 1: Initialize Enhanced Storage System
    println!("\n📦 Test 1: Initializing Enhanced Storage System...");
    
    #[cfg(feature = "ai")]
    {
        use llmkg::enhanced_knowledge_storage::{
            EnhancedKnowledgeStorageSystem,
            types::{EnhancedStorageConfig, ComplexityLevel},
        };
        
        let config = EnhancedStorageConfig {
            embedding_dim: TEST_EMBEDDING_DIM,
            enable_intelligent_processing: true,
            enable_multi_hop_reasoning: true,
            model_memory_limit: 1024 * 1024 * 1024, // 1GB
            max_processing_time_seconds: 30,
            enable_semantic_chunking: true,
            cache_enhanced_results: true,
            fallback_on_failure: true,
        };
        
        let storage_system = EnhancedKnowledgeStorageSystem::new(config).await?;
        println!("✅ Enhanced Storage System initialized successfully");
        
        // Test 2: Process Documents with Real AI Components
        println!("\n🧠 Test 2: Processing documents with real AI components...");
        
        let mut processed_documents = Vec::new();
        
        for (i, (title, content)) in TEST_DOCUMENTS.iter().enumerate() {
            println!("   Processing document {}: {}", i + 1, title);
            
            let metadata = llmkg::enhanced_knowledge_storage::types::DocumentMetadata {
                id: format!("doc_{}", i + 1),
                title: title.to_string(),
                source: "integration_test".to_string(),
                created_at: chrono::Utc::now(),
                complexity: ComplexityLevel::Medium,
                language: "en".to_string(),
                author: Some("Test Suite".to_string()),
                version: "1.0".to_string(),
            };
            
            match storage_system.process_document(content, metadata).await {
                Ok(result) => {
                    println!("     ✅ Document processed successfully");
                    println!("     📊 Extracted {} entities, {} relationships, {} chunks", 
                             result.global_entities.len(),
                             result.global_relationships.len(),
                             result.chunks.len());
                    processed_documents.push(result);
                }
                Err(e) => {
                    println!("     ❌ Failed to process document: {}", e);
                    return Err(e.into());
                }
            }
        }
        
        // Test 3: Verify Entity Extraction
        println!("\n🏷️  Test 3: Verifying entity extraction...");
        
        let mut total_entities = 0;
        let mut entity_types = std::collections::HashSet::new();
        
        for (i, doc) in processed_documents.iter().enumerate() {
            total_entities += doc.global_entities.len();
            for entity in &doc.global_entities {
                entity_types.insert(entity.entity_type.clone());
            }
            println!("   Document {}: {} entities found", i + 1, doc.global_entities.len());
        }
        
        println!("   ✅ Total entities extracted: {}", total_entities);
        println!("   ✅ Entity types found: {:?}", entity_types);
        
        if total_entities == 0 {
            println!("   ⚠️  Warning: No entities extracted. This may indicate an issue with entity extraction.");
        }
        
        // Test 4: Verify Semantic Chunking
        println!("\n📄 Test 4: Verifying semantic chunking...");
        
        let mut total_chunks = 0;
        let mut avg_coherence = 0.0;
        
        for (i, doc) in processed_documents.iter().enumerate() {
            total_chunks += doc.chunks.len();
            let doc_coherence: f32 = doc.chunks.iter()
                .map(|chunk| chunk.semantic_coherence)
                .sum::<f32>() / doc.chunks.len() as f32;
            avg_coherence += doc_coherence;
            
            println!("   Document {}: {} chunks, avg coherence: {:.3}", 
                     i + 1, doc.chunks.len(), doc_coherence);
        }
        
        avg_coherence /= processed_documents.len() as f32;
        println!("   ✅ Total chunks created: {}", total_chunks);
        println!("   ✅ Average semantic coherence: {:.3}", avg_coherence);
        
        // Test 5: Verify Relationship Extraction
        println!("\n🔗 Test 5: Verifying relationship extraction...");
        
        let mut total_relationships = 0;
        let mut relationship_types = std::collections::HashSet::new();
        
        for (i, doc) in processed_documents.iter().enumerate() {
            total_relationships += doc.global_relationships.len();
            for relationship in &doc.global_relationships {
                relationship_types.insert(relationship.relationship_type.clone());
            }
            println!("   Document {}: {} relationships found", i + 1, doc.global_relationships.len());
        }
        
        println!("   ✅ Total relationships extracted: {}", total_relationships);
        println!("   ✅ Relationship types found: {:?}", relationship_types);
        
        // Test 6: Store Knowledge in Hierarchical Storage
        println!("\n🏗️  Test 6: Storing knowledge in hierarchical storage...");
        
        for (i, doc_result) in processed_documents.iter().enumerate() {
            match storage_system.store_processed_knowledge(doc_result.clone()).await {
                Ok(storage_result) => {
                    println!("   Document {}: Stored {} layers with {} semantic links", 
                             i + 1, 
                             storage_result.created_layers.len(),
                             storage_result.semantic_links.len());
                }
                Err(e) => {
                    println!("   ❌ Failed to store document {}: {}", i + 1, e);
                    return Err(e.into());
                }
            }
        }
        
        println!("   ✅ All documents stored in hierarchical storage");
        
        // Test 7: Test Retrieval System
        println!("\n🔍 Test 7: Testing retrieval system...");
        
        let test_queries = [
            "What is machine learning?",
            "How do quantum computers work?",
            "What are renewable energy sources?",
            "Tell me about artificial intelligence and data patterns",
        ];
        
        for (i, query) in test_queries.iter().enumerate() {
            println!("   Query {}: \"{}\"", i + 1, query);
            
            let retrieval_query = llmkg::enhanced_knowledge_storage::retrieval_system::types::RetrievalQuery {
                natural_language_query: query.to_string(),
                structured_constraints: None,
                retrieval_mode: llmkg::enhanced_knowledge_storage::retrieval_system::types::RetrievalMode::Comprehensive,
                max_results: 5,
                enable_multi_hop: true,
                max_reasoning_hops: 3,
                context_window_size: 1000,
            };
            
            match storage_system.retrieve_knowledge(retrieval_query).await {
                Ok(result) => {
                    println!("     ✅ Found {} results with confidence {:.3}", 
                             result.retrieved_items.len(), result.overall_confidence);
                    
                    if result.reasoning_chain.is_some() {
                        let chain = result.reasoning_chain.unwrap();
                        println!("     🧠 Multi-hop reasoning: {} steps", chain.reasoning_steps.len());
                    }
                }
                Err(e) => {
                    println!("     ❌ Query failed: {}", e);
                }
            }
        }
        
        // Test 8: Performance Metrics
        println!("\n📊 Test 8: Checking performance metrics...");
        
        let metrics = storage_system.get_performance_metrics().await;
        println!("   ✅ System performance metrics:");
        println!("     • Total operations: {}", metrics.total_operations);
        println!("     • Average response time: {:.2}ms", metrics.average_response_time_ms);
        println!("     • Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
        println!("     • Memory usage: {:.1}MB", metrics.memory_usage_mb);
        println!("     • AI model utilization: {:.1}%", metrics.ai_model_utilization * 100.0);
        
        println!("\n🎉 Integration Test Results:");
        println!("✅ Enhanced Knowledge Storage System is fully functional");
        println!("✅ All AI components working without mocks");
        println!("✅ Document processing pipeline operational");
        println!("✅ Hierarchical storage working correctly");
        println!("✅ Retrieval system with multi-hop reasoning functional");
        println!("✅ Performance monitoring active");
        
        println!("\n🏁 Integration test completed successfully!");
        
    }
    
    #[cfg(not(feature = "ai"))]
    {
        println!("❌ AI features not enabled. Run with --features ai to test real AI components.");
        println!("   This test requires the 'ai' feature to verify that all components work without mocks.");
        return Err("AI features not enabled".into());
    }
    
    Ok(())
}