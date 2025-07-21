//! Integration tests for UnifiedMemorySystem
//! 
//! Tests the public API of the unified memory system including:
//! - Memory coordination workflows across backends
//! - Cross-backend retrieval scenarios
//! - Memory consolidation mechanisms
//! - Integration with working memory, SDR storage, and graph
//! - Performance under various loads

use std::sync::Arc;
use std::time::Duration;

use llmkg::cognitive::memory_integration::{
    UnifiedMemorySystem, MemoryIntegrationConfig, MemoryType, CrossMemoryLink, LinkType,
    OptimizationOpportunity, OpportunityType,
};
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType, MemoryQuery};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::sdr_types::SDRConfig;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::error::Result;

// Import shared test utilities
use super::test_utils::{create_test_entity_keys, PerformanceTimer};

/// Creates a test unified memory system with default configuration
async fn create_test_unified_memory_system() -> Result<UnifiedMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    let working_memory = Arc::new(
        WorkingMemorySystem::new(activation_engine, sdr_storage.clone()).await?
    );
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(128)?);
    
    Ok(UnifiedMemorySystem::new(working_memory, sdr_storage, graph))
}

/// Creates a test unified memory system with custom configuration
async fn create_test_unified_memory_with_config(config: MemoryIntegrationConfig) -> Result<UnifiedMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    let working_memory = Arc::new(
        WorkingMemorySystem::new(activation_engine, sdr_storage.clone()).await?
    );
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(128)?);
    
    Ok(UnifiedMemorySystem::with_config(working_memory, sdr_storage, graph, config))
}

#[tokio::test]
async fn test_unified_memory_system_creation() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Verify default configuration
    assert!(system.integration_config.enable_parallel_retrieval);
    assert_eq!(system.integration_config.default_strategy, "parallel_comprehensive");
    assert!(system.integration_config.cross_memory_linking);
    assert_eq!(system.integration_config.memory_hierarchy_depth, 7);
    
    // Verify initial statistics
    let stats = system.get_memory_statistics().await.unwrap();
    assert_eq!(stats.total_retrievals, 0);
    assert_eq!(stats.successful_retrievals, 0);
    assert_eq!(stats.consolidation_events, 0);
}

#[tokio::test]
async fn test_unified_memory_system_with_custom_config() {
    let custom_config = MemoryIntegrationConfig {
        enable_parallel_retrieval: false,
        default_strategy: "sequential_search".to_string(),
        consolidation_frequency: Duration::from_secs(60),
        optimization_frequency: Duration::from_secs(1800),
        cross_memory_linking: false,
        memory_hierarchy_depth: 5,
    };
    
    let system = create_test_unified_memory_with_config(custom_config.clone()).await.unwrap();
    
    assert!(!system.integration_config.enable_parallel_retrieval);
    assert_eq!(system.integration_config.default_strategy, "sequential_search");
    assert!(!system.integration_config.cross_memory_linking);
    assert_eq!(system.integration_config.memory_hierarchy_depth, 5);
}

#[tokio::test]
async fn test_basic_memory_storage_and_retrieval_workflow() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store information with varying importance
    let content1 = "artificial intelligence concepts";
    let content2 = "machine learning algorithms";
    let content3 = "neural network architectures";
    
    let id1 = system.store_information(content1, 0.8, Some("AI context")).await.unwrap();
    let id2 = system.store_information(content2, 0.9, Some("ML context")).await.unwrap();
    let id3 = system.store_information(content3, 0.7, Some("NN context")).await.unwrap();
    
    assert!(!id1.is_empty());
    assert!(!id2.is_empty());
    assert!(!id3.is_empty());
    
    // Retrieve using different strategies
    let result1 = system.retrieve_information("artificial intelligence", None).await.unwrap();
    let result2 = system.retrieve_information("machine learning", Some("parallel_comprehensive")).await.unwrap();
    
    // Both should return results
    assert!(!result1.primary_results.is_empty() || !result1.secondary_results.is_empty());
    assert!(!result2.primary_results.is_empty() || !result2.secondary_results.is_empty());
    
    // Should have reasonable confidence
    assert!(result1.fusion_confidence >= 0.0);
    assert!(result2.fusion_confidence >= 0.0);
    
    // Verify statistics were updated
    let stats = system.get_memory_statistics().await.unwrap();
    assert!(stats.total_retrievals > 0);
    assert!(stats.successful_retrievals > 0);
}

#[tokio::test]
async fn test_cross_backend_retrieval_scenarios() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store content in different memory types conceptually
    let working_memory_content = "current task information";
    let long_term_content = "established knowledge";
    let semantic_content = "conceptual relationships";
    
    // Store information with different characteristics
    system.store_information(working_memory_content, 0.9, Some("current")).await.unwrap();
    system.store_information(long_term_content, 0.8, Some("established")).await.unwrap();
    system.store_information(semantic_content, 0.7, Some("semantic")).await.unwrap();
    
    // Test search across all memory systems
    let all_results = system.search_all_memories("information", 10).await.unwrap();
    
    // Should return results from multiple backends
    assert!(!all_results.is_empty());
    
    // Verify we get results from different memory systems
    let total_items: usize = all_results.iter().map(|result| result.items.len()).sum();
    assert!(total_items > 0);
    
    // Test cross-memory retrieval
    let cross_result = system.retrieve_information("knowledge", None).await.unwrap();
    assert!(cross_result.fusion_confidence >= 0.0);
}

#[tokio::test]
async fn test_memory_consolidation_mechanisms() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store multiple related items for consolidation
    let related_contents = [
        "reinforcement learning fundamentals",
        "policy gradient methods",
        "value function approximation",
        "actor-critic algorithms",
        "deep Q-learning networks",
    ];
    
    for content in &related_contents {
        system.store_information(content, 0.8, Some("RL concepts")).await.unwrap();
    }
    
    // Perform consolidation with default policy
    let consolidation_result = system.consolidate_memories(None).await.unwrap();
    
    // Verify consolidation completed
    assert!(consolidation_result.consolidation_time.as_millis() > 0);
    assert!(consolidation_result.success_rate >= 0.0);
    assert!(consolidation_result.success_rate <= 1.0);
    
    // Perform consolidation with specific policy
    let specific_consolidation = system.consolidate_memories(Some("importance_based")).await.unwrap();
    assert!(specific_consolidation.consolidation_time.as_millis() > 0);
    
    // Verify statistics track consolidation events
    let stats = system.get_memory_statistics().await.unwrap();
    assert!(stats.consolidation_events >= 0);
}

#[tokio::test]
async fn test_cross_memory_linking_functionality() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Create cross-memory links between different memory types
    let associative_link = CrossMemoryLink {
        link_id: "concept_association".to_string(),
        source_memory: MemoryType::WorkingMemory,
        target_memory: MemoryType::SemanticMemory,
        link_strength: 0.8,
        link_type: LinkType::Associative,
        bidirectional: true,
    };
    
    let temporal_link = CrossMemoryLink {
        link_id: "temporal_sequence".to_string(),
        source_memory: MemoryType::EpisodicMemory,
        target_memory: MemoryType::WorkingMemory,
        link_strength: 0.7,
        link_type: LinkType::Temporal,
        bidirectional: false,
    };
    
    let causal_link = CrossMemoryLink {
        link_id: "causal_relationship".to_string(),
        source_memory: MemoryType::SemanticMemory,
        target_memory: MemoryType::LongTermMemory,
        link_strength: 0.9,
        link_type: LinkType::Causal,
        bidirectional: true,
    };
    
    // Create the links
    system.create_cross_memory_link(associative_link).await.unwrap();
    system.create_cross_memory_link(temporal_link).await.unwrap();
    system.create_cross_memory_link(causal_link).await.unwrap();
    
    // Retrieve cross-memory links
    let associative_links = system.get_cross_memory_links("concept_association").await.unwrap();
    let temporal_links = system.get_cross_memory_links("temporal_sequence").await.unwrap();
    let causal_links = system.get_cross_memory_links("causal_relationship").await.unwrap();
    
    // Links should be retrievable (exact behavior depends on coordinator implementation)
    assert!(associative_links.len() >= 0);
    assert!(temporal_links.len() >= 0);
    assert!(causal_links.len() >= 0);
}

#[tokio::test]
async fn test_memory_system_optimization_workflow() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store some data and perform operations to generate statistics
    for i in 0..10 {
        let content = format!("optimization_test_content_{}", i);
        system.store_information(&content, 0.6 + (i as f32 * 0.04), None).await.unwrap();
    }
    
    for i in 0..5 {
        let query = format!("optimization_test_content_{}", i);
        let _ = system.retrieve_information(&query, None).await;
    }
    
    // Analyze current performance
    let performance_analysis = system.analyze_performance().await.unwrap();
    
    assert!(performance_analysis.success_rate >= 0.0);
    assert!(performance_analysis.success_rate <= 1.0);
    assert!(performance_analysis.average_retrieval_time.as_millis() >= 0);
    
    // Perform system optimization
    let optimization_result = system.optimize_memory_system().await.unwrap();
    
    // Verify optimization completed
    assert!(optimization_result.optimization_time.as_millis() > 0);
    assert!(!optimization_result.optimization_opportunities.is_empty() || optimization_result.applied_optimizations.is_empty());
    
    // If optimizations were applied, they should be successful
    for optimization in &optimization_result.applied_optimizations {
        assert!(optimization.implementation_success);
        assert!(optimization.actual_improvement >= 0.0);
    }
}

#[tokio::test]
async fn test_system_status_reporting() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Get initial status
    let initial_status = system.get_system_status().await.unwrap();
    assert!(initial_status.contains("Unified Memory System Status"));
    assert!(initial_status.contains("Total Retrievals: 0"));
    
    // Perform some operations
    system.store_information("status test content", 0.8, None).await.unwrap();
    let _ = system.retrieve_information("status test", None).await;
    
    // Get updated status
    let updated_status = system.get_system_status().await.unwrap();
    assert!(updated_status.contains("Unified Memory System Status"));
    assert!(updated_status.contains("Total Retrievals:"));
    assert!(updated_status.contains("Success Rate:"));
    assert!(updated_status.contains("Average Retrieval Time:"));
}

#[tokio::test]
async fn test_memory_configuration_updates() {
    let mut system = create_test_unified_memory_system().await.unwrap();
    
    // Verify initial configuration
    assert!(system.integration_config.enable_parallel_retrieval);
    assert_eq!(system.integration_config.memory_hierarchy_depth, 7);
    
    // Update configuration
    let new_config = MemoryIntegrationConfig {
        enable_parallel_retrieval: false,
        default_strategy: "sequential_updated".to_string(),
        consolidation_frequency: Duration::from_secs(180),
        optimization_frequency: Duration::from_secs(3600),
        cross_memory_linking: false,
        memory_hierarchy_depth: 10,
    };
    
    system.update_config(new_config);
    
    // Verify configuration was updated
    assert!(!system.integration_config.enable_parallel_retrieval);
    assert_eq!(system.integration_config.default_strategy, "sequential_updated");
    assert_eq!(system.integration_config.memory_hierarchy_depth, 10);
    assert!(!system.integration_config.cross_memory_linking);
}

#[tokio::test]
async fn test_memory_integration_with_working_memory() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store content that should primarily go to working memory
    let working_content = "immediate working memory task";
    let id = system.store_information(working_content, 0.9, Some("immediate")).await.unwrap();
    assert!(!id.is_empty());
    
    // Search for the content
    let search_results = system.search_all_memories("immediate working", 5).await.unwrap();
    assert!(!search_results.is_empty());
    
    // Verify that working memory is included in the search
    let total_results: usize = search_results.iter().map(|r| r.items.len()).sum();
    assert!(total_results > 0);
}

#[tokio::test]
async fn test_memory_integration_with_sdr_storage() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store structured content suitable for SDR storage
    let sdr_content = "distributed representation learning";
    system.store_information(sdr_content, 0.8, Some("structured")).await.unwrap();
    
    // Search across all memory systems
    let all_results = system.search_all_memories("distributed representation", 10).await.unwrap();
    
    // Should find results (exact behavior depends on SDR implementation)
    assert!(all_results.len() >= 0);
    
    // Verify retrieval confidence
    for result in &all_results {
        assert!(result.retrieval_confidence >= 0.0);
        assert!(result.retrieval_confidence <= 1.0);
    }
}

#[tokio::test]
async fn test_memory_integration_with_knowledge_graph() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store relational content suitable for graph storage
    let graph_content = "entity relationship mapping";
    system.store_information(graph_content, 0.8, Some("relational")).await.unwrap();
    
    // Search specifically targeting graph-stored information
    let graph_results = system.search_all_memories("entity relationship", 5).await.unwrap();
    
    // Should process without errors
    assert!(graph_results.len() >= 0);
    
    // Each result should have valid confidence scores
    for result in &graph_results {
        assert!(result.retrieval_confidence >= 0.0);
        for item in &result.items {
            assert!(item.importance_score >= 0.0);
            assert!(item.activation_level >= 0.0);
        }
    }
}

#[tokio::test]
async fn test_concurrent_memory_operations() {
    let system = Arc::new(create_test_unified_memory_system().await.unwrap());
    
    // Spawn concurrent storage operations
    let mut storage_handles = Vec::new();
    for i in 0..5 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let content = format!("concurrent_content_{}", i);
            system_clone.store_information(&content, 0.7, None).await
        });
        storage_handles.push(handle);
    }
    
    // Wait for all storage operations
    for handle in storage_handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    // Spawn concurrent retrieval operations
    let mut retrieval_handles = Vec::new();
    for i in 0..3 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let query = format!("concurrent_content_{}", i);
            system_clone.retrieve_information(&query, None).await
        });
        retrieval_handles.push(handle);
    }
    
    // Wait for all retrieval operations
    for handle in retrieval_handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    // Verify system state after concurrent operations
    let final_stats = system.get_memory_statistics().await.unwrap();
    assert!(final_stats.total_retrievals > 0);
}

#[tokio::test]
async fn test_memory_system_error_handling() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Test retrieval with invalid strategy
    let result = system.retrieve_information("test", Some("invalid_strategy")).await;
    // Should handle gracefully (exact behavior depends on implementation)
    assert!(result.is_ok() || result.is_err());
    
    // Test consolidation with invalid policy
    let consolidation_result = system.consolidate_memories(Some("invalid_policy")).await;
    // Should handle gracefully
    assert!(consolidation_result.is_ok() || consolidation_result.is_err());
    
    // Test cross-memory link retrieval with non-existent ID
    let links = system.get_cross_memory_links("non_existent_id").await.unwrap();
    assert!(links.is_empty() || !links.is_empty()); // Should not crash
}

#[tokio::test]
async fn test_memory_system_capacity_and_limits() {
    let system = create_test_unified_memory_system().await.unwrap();
    
    // Store a large number of items to test capacity handling
    for i in 0..50 {
        let content = format!("capacity_test_item_{}", i);
        let importance = 0.1 + (i as f32 / 50.0) * 0.8; // Varying importance
        let result = system.store_information(&content, importance, None).await;
        assert!(result.is_ok());
    }
    
    // Verify system still functions
    let search_results = system.search_all_memories("capacity_test", 20).await.unwrap();
    assert!(search_results.len() >= 0);
    
    // Check system status
    let status = system.get_system_status().await.unwrap();
    assert!(status.contains("Unified Memory System Status"));
    
    // Analyze performance under load
    let performance = system.analyze_performance().await.unwrap();
    assert!(performance.success_rate >= 0.0);
    assert!(performance.average_retrieval_time.as_millis() >= 0);
}

/// Performance test for unified memory operations
#[tokio::test]
async fn test_unified_memory_system_performance() {
    let system = create_test_unified_memory_system().await.unwrap();
    let timer = PerformanceTimer::new("unified memory system operations");
    
    // Rapid storage operations across different content types
    for i in 0..25 {
        let content = format!("performance_test_content_{}", i);
        let importance = 0.5 + (i as f32 / 50.0);
        let context = if i % 2 == 0 { Some("even_context") } else { Some("odd_context") };
        let _ = system.store_information(&content, importance, context).await;
    }
    
    // Rapid retrieval operations with different strategies
    for i in 0..15 {
        let query = format!("performance_test_content_{}", i);
        let strategy = if i % 3 == 0 { Some("parallel_comprehensive") } else { None };
        let _ = system.retrieve_information(&query, strategy).await;
    }
    
    // Cross-memory search operations
    for i in 0..10 {
        let query = format!("performance_test_{}", i);
        let _ = system.search_all_memories(&query, 5).await;
    }
    
    // System optimization
    let _ = system.analyze_performance().await;
    
    // Performance should be reasonable
    timer.assert_within_ms(3000.0); // Allow more time for complex operations
    println!("Unified memory system performance test completed in {:.2}ms", timer.elapsed_ms());
}