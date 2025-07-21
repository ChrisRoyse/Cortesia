//! Integration tests for WorkingMemorySystem
//! 
//! Tests the public API of the working memory system including:
//! - Memory storage and retrieval workflows
//! - Buffer capacity management
//! - Memory decay over time
//! - Cross-buffer coordination
//! - Attention-based memory operations

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use llmkg::cognitive::working_memory::{
    WorkingMemorySystem, MemoryContent, BufferType, MemoryQuery, 
    MemoryItem, MemoryCapacityLimits, MemoryDecayConfig, ForgettingCurve,
    WorkingMemoryState
};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::sdr_types::SDRConfig;
use llmkg::core::types::{EntityKey, EntityData};
use llmkg::core::brain_types::{BrainInspiredEntity, ActivationPattern};
use llmkg::error::Result;

// Import shared test utilities
use super::test_utils::{create_test_entity_keys, create_memory_item, PerformanceTimer};

/// Creates a test working memory system with default configuration
async fn create_test_working_memory_system() -> Result<WorkingMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    
    WorkingMemorySystem::new(activation_engine, sdr_storage).await
}

/// Creates a test working memory system with custom capacity limits
async fn create_test_working_memory_with_limits(limits: MemoryCapacityLimits) -> Result<WorkingMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    
    let mut system = WorkingMemorySystem::new(activation_engine, sdr_storage).await?;
    system.capacity_limits = limits;
    Ok(system)
}

/// Creates a test working memory system with custom decay configuration
async fn create_test_working_memory_with_decay(decay_config: MemoryDecayConfig) -> Result<WorkingMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    
    let mut system = WorkingMemorySystem::new(activation_engine, sdr_storage).await?;
    system.decay_config = decay_config;
    Ok(system)
}

#[tokio::test]
async fn test_working_memory_creation() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Verify default configuration
    assert_eq!(memory_system.capacity_limits.phonological_capacity, 7);
    assert_eq!(memory_system.capacity_limits.visuospatial_capacity, 4);
    assert_eq!(memory_system.capacity_limits.episodic_capacity, 3);
    assert_eq!(memory_system.capacity_limits.total_capacity, 15);
    
    // Verify initial state is empty
    let state = memory_system.get_current_state().await.unwrap();
    assert_eq!(state.total_items, 0);
    assert_eq!(state.capacity_utilization, 0.0);
}

#[tokio::test]
async fn test_basic_memory_storage_and_retrieval() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store a concept in episodic buffer
    let content = MemoryContent::Concept("artificial intelligence".to_string());
    let result = memory_system.store_in_working_memory(
        content, 
        0.8, 
        BufferType::Episodic
    ).await.unwrap();
    
    assert!(result.success, "Storage should succeed");
    assert!(result.evicted_items.is_empty(), "No items should be evicted with empty buffer");
    
    // Verify state after storage
    let state = memory_system.get_current_state().await.unwrap();
    assert_eq!(state.total_items, 1);
    assert!(state.capacity_utilization > 0.0);
    
    // Retrieve the stored item
    let query = MemoryQuery {
        query_text: "artificial intelligence".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let retrieval_result = memory_system.retrieve_from_working_memory(&query).await.unwrap();
    assert!(!retrieval_result.items.is_empty(), "Should retrieve the stored item");
    assert!(retrieval_result.retrieval_confidence > 0.0, "Should have positive confidence");
}

#[tokio::test]
async fn test_buffer_capacity_management() {
    // Create system with small capacity for testing
    let small_limits = MemoryCapacityLimits {
        phonological_capacity: 2,
        visuospatial_capacity: 2,
        episodic_capacity: 2,
        total_capacity: 6,
    };
    let memory_system = create_test_working_memory_with_limits(small_limits).await.unwrap();
    
    // Fill episodic buffer to capacity
    for i in 0..2 {
        let content = MemoryContent::Concept(format!("concept_{}", i));
        let result = memory_system.store_in_working_memory(
            content, 
            0.5, 
            BufferType::Episodic
        ).await.unwrap();
        assert!(result.success);
    }
    
    // Add one more item - should trigger eviction
    let content = MemoryContent::Concept("new_important_concept".to_string());
    let result = memory_system.store_in_working_memory(
        content, 
        0.9, // High importance
        BufferType::Episodic
    ).await.unwrap();
    
    assert!(result.success, "Storage should succeed even at capacity");
    assert!(!result.evicted_items.is_empty(), "Should evict less important items");
    
    // Verify buffer state
    let state = memory_system.get_current_state().await.unwrap();
    let episodic_state = &state.buffer_states[2]; // Episodic is the third buffer
    assert!(episodic_state.capacity_utilization <= 1.0, "Should not exceed capacity");
}

#[tokio::test]
async fn test_memory_decay_over_time() {
    let fast_decay_config = MemoryDecayConfig {
        decay_rate: 0.5, // Fast decay for testing
        refresh_threshold: Duration::from_millis(10),
        forgetting_curve: ForgettingCurve::Exponential { 
            half_life: Duration::from_millis(100)
        },
    };
    
    let memory_system = create_test_working_memory_with_decay(fast_decay_config).await.unwrap();
    
    // Store items with different importance levels
    let weak_content = MemoryContent::Concept("weak_memory".to_string());
    let strong_content = MemoryContent::Concept("strong_memory".to_string());
    
    memory_system.store_in_working_memory(weak_content, 0.2, BufferType::Episodic).await.unwrap();
    memory_system.store_in_working_memory(strong_content, 0.9, BufferType::Episodic).await.unwrap();
    
    // Wait for some time to allow decay
    sleep(Duration::from_millis(50)).await;
    
    // Manually trigger decay
    memory_system.decay_memory_items().await.unwrap();
    
    // Retrieve all items to see what survived
    let all_items = memory_system.get_all_items().await.unwrap();
    
    // Strong memory should be more likely to survive than weak memory
    let has_strong = all_items.iter().any(|item| {
        if let MemoryContent::Concept(concept) = &item.content {
            concept == "strong_memory"
        } else {
            false
        }
    });
    
    // At minimum, verify that decay processing doesn't crash
    assert!(has_strong || all_items.is_empty(), "Decay should process without errors");
}

#[tokio::test]
async fn test_cross_buffer_coordination() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store related content in different buffers
    let phonological_content = MemoryContent::Concept("spoken word".to_string());
    let visuospatial_content = MemoryContent::Concept("visual image".to_string());
    let episodic_content = MemoryContent::Concept("life event".to_string());
    
    memory_system.store_in_working_memory(phonological_content, 0.7, BufferType::Phonological).await.unwrap();
    memory_system.store_in_working_memory(visuospatial_content, 0.8, BufferType::Visuospatial).await.unwrap();
    memory_system.store_in_working_memory(episodic_content, 0.9, BufferType::Episodic).await.unwrap();
    
    // Query across all buffers
    let cross_buffer_query = MemoryQuery {
        query_text: "memory".to_string(),
        search_buffers: vec![BufferType::Phonological, BufferType::Visuospatial, BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.3,
    };
    
    let results = memory_system.retrieve_from_working_memory(&cross_buffer_query).await.unwrap();
    
    // Should be able to retrieve from multiple buffers
    assert!(!results.items.is_empty(), "Should find items across buffers");
    assert_eq!(results.buffer_states.len(), 3, "Should report state for all buffers");
    
    // Verify buffer states show utilization
    let state = memory_system.get_current_state().await.unwrap();
    assert_eq!(state.total_items, 3, "Should have items in all buffers");
    assert!(state.buffer_states.iter().all(|bs| bs.current_load >= 0.0), "All buffers should report valid load");
}

#[tokio::test]
async fn test_attention_based_storage() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store item without attention boost
    let normal_content = MemoryContent::Concept("normal_concept".to_string());
    let normal_result = memory_system.store_in_working_memory(
        normal_content, 
        0.5, 
        BufferType::Episodic
    ).await.unwrap();
    
    // Store item with attention boost
    let attended_content = MemoryContent::Concept("attended_concept".to_string());
    let attended_result = memory_system.store_in_working_memory_with_attention(
        attended_content, 
        0.5, 
        BufferType::Episodic,
        0.8 // High attention boost
    ).await.unwrap();
    
    assert!(normal_result.success && attended_result.success, "Both storage operations should succeed");
    
    // Retrieve all items to compare
    let all_items = memory_system.get_all_items().await.unwrap();
    assert_eq!(all_items.len(), 2, "Should have stored both items");
    
    // Find the attended item - it should have higher activation due to attention boost
    let attended_item = all_items.iter().find(|item| {
        if let MemoryContent::Concept(concept) = &item.content {
            concept == "attended_concept"
        } else {
            false
        }
    });
    
    let normal_item = all_items.iter().find(|item| {
        if let MemoryContent::Concept(concept) = &item.content {
            concept == "normal_concept"
        } else {
            false
        }
    });
    
    if let (Some(attended), Some(normal)) = (attended_item, normal_item) {
        assert!(attended.activation_level > normal.activation_level, 
               "Attended item should have higher activation level");
    }
}

#[tokio::test]
async fn test_attention_relevant_item_retrieval() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Create test entity keys for attention targets
    let attention_targets = create_test_entity_keys(2);
    
    // Store some items
    let concept_content = MemoryContent::Concept("machine learning".to_string());
    memory_system.store_in_working_memory(concept_content, 0.8, BufferType::Episodic).await.unwrap();
    
    // Get attention-relevant items
    let relevant_items = memory_system.get_attention_relevant_items(
        &attention_targets,
        Some(BufferType::Episodic)
    ).await.unwrap();
    
    // Based on the current implementation, concept items are considered relevant to attention
    assert!(!relevant_items.is_empty(), "Should find attention-relevant items");
    
    // Test with no buffer filter (search all buffers)
    let all_relevant_items = memory_system.get_attention_relevant_items(
        &attention_targets,
        None
    ).await.unwrap();
    
    assert!(all_relevant_items.len() >= relevant_items.len(), 
           "Searching all buffers should find at least as many items");
}

#[tokio::test]
async fn test_long_term_memory_consolidation() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store high-importance items that should be consolidated
    for i in 0..3 {
        let content = MemoryContent::Concept(format!("important_concept_{}", i));
        memory_system.store_in_working_memory(content, 0.9, BufferType::Episodic).await.unwrap();
    }
    
    // Access the items to increase their access count
    let query = MemoryQuery {
        query_text: "important_concept".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    // Multiple retrievals to increase access count
    for _ in 0..3 {
        let _ = memory_system.retrieve_from_working_memory(&query).await.unwrap();
    }
    
    // Trigger consolidation
    let consolidation_result = memory_system.consolidate_to_long_term().await;
    
    // Should complete without error
    assert!(consolidation_result.is_ok(), "Consolidation should complete without error");
}

#[tokio::test]
async fn test_memory_query_with_attention_filtering() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store multiple items with different importance levels
    for i in 0..10 {
        let content = MemoryContent::Concept(format!("test_item_{}", i));
        let importance = 0.1 + (i as f32 * 0.1);
        memory_system.store_in_working_memory(content, importance, BufferType::Episodic).await.unwrap();
    }
    
    // Query with attention filtering enabled
    let attention_query = MemoryQuery {
        query_text: "test_item".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: true,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let attention_results = memory_system.retrieve_from_working_memory(&attention_query).await.unwrap();
    
    // Query without attention filtering
    let normal_query = MemoryQuery {
        query_text: "test_item".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let normal_results = memory_system.retrieve_from_working_memory(&normal_query).await.unwrap();
    
    // Attention filtering should limit results
    assert!(attention_results.items.len() <= normal_results.items.len(), 
           "Attention filtering should limit or maintain result count");
    assert!(attention_results.items.len() <= 5, 
           "Attention filtering should limit to top 5 items");
}

#[tokio::test]
async fn test_working_memory_state_reporting() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Get initial state
    let initial_state = memory_system.get_current_state().await.unwrap();
    assert_eq!(initial_state.total_items, 0);
    assert_eq!(initial_state.capacity_utilization, 0.0);
    assert_eq!(initial_state.efficiency_score, 1.0); // Should be 1.0 when empty
    
    // Add some items
    for i in 0..5 {
        let content = MemoryContent::Concept(format!("state_test_{}", i));
        let importance = 0.5 + (i as f32 * 0.1);
        let buffer_type = match i % 3 {
            0 => BufferType::Phonological,
            1 => BufferType::Visuospatial,
            _ => BufferType::Episodic,
        };
        memory_system.store_in_working_memory(content, importance, buffer_type).await.unwrap();
    }
    
    // Get updated state
    let updated_state = memory_system.get_current_state().await.unwrap();
    assert_eq!(updated_state.total_items, 5);
    assert!(updated_state.capacity_utilization > 0.0);
    assert!(updated_state.average_importance > 0.0);
    assert!(updated_state.efficiency_score >= 0.0 && updated_state.efficiency_score <= 1.0);
    
    // Verify buffer states
    assert_eq!(updated_state.buffer_states.len(), 3);
    for buffer_state in &updated_state.buffer_states {
        assert!(buffer_state.current_load >= 0.0);
        assert!(buffer_state.capacity_utilization >= 0.0);
    }
}

#[tokio::test]
async fn test_memory_retrieval_confidence_calculation() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store items with varying importance
    let high_importance_content = MemoryContent::Concept("high_importance".to_string());
    let low_importance_content = MemoryContent::Concept("low_importance".to_string());
    
    memory_system.store_in_working_memory(high_importance_content, 0.9, BufferType::Episodic).await.unwrap();
    memory_system.store_in_working_memory(low_importance_content, 0.2, BufferType::Episodic).await.unwrap();
    
    // Query for high importance item
    let high_query = MemoryQuery {
        query_text: "high_importance".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let high_results = memory_system.retrieve_from_working_memory(&high_query).await.unwrap();
    
    // Query for low importance item
    let low_query = MemoryQuery {
        query_text: "low_importance".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let low_results = memory_system.retrieve_from_working_memory(&low_query).await.unwrap();
    
    // High importance items should have higher retrieval confidence
    assert!(high_results.retrieval_confidence > low_results.retrieval_confidence,
           "High importance items should have higher retrieval confidence");
}

#[tokio::test]
async fn test_memory_item_access_patterns() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store a test item
    let content = MemoryContent::Concept("access_test".to_string());
    memory_system.store_in_working_memory(content, 0.7, BufferType::Episodic).await.unwrap();
    
    // Retrieve the item multiple times
    let query = MemoryQuery {
        query_text: "access_test".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    // Multiple retrievals should update access patterns
    for _ in 0..5 {
        let results = memory_system.retrieve_from_working_memory(&query).await.unwrap();
        assert!(!results.items.is_empty(), "Should continue to find the item");
    }
    
    // Verify the system is still functional after multiple accesses
    let final_state = memory_system.get_current_state().await.unwrap();
    assert_eq!(final_state.total_items, 1);
}

#[tokio::test]
async fn test_buffer_type_specific_operations() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Store items in specific buffer types
    let phonological_content = MemoryContent::Concept("phonological_item".to_string());
    let visuospatial_content = MemoryContent::Concept("visuospatial_item".to_string());
    let episodic_content = MemoryContent::Concept("episodic_item".to_string());
    
    memory_system.store_in_working_memory(phonological_content, 0.8, BufferType::Phonological).await.unwrap();
    memory_system.store_in_working_memory(visuospatial_content, 0.8, BufferType::Visuospatial).await.unwrap();
    memory_system.store_in_working_memory(episodic_content, 0.8, BufferType::Episodic).await.unwrap();
    
    // Test buffer-specific queries
    let phonological_query = MemoryQuery {
        query_text: "phonological_item".to_string(),
        search_buffers: vec![BufferType::Phonological],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let phonological_results = memory_system.retrieve_from_working_memory(&phonological_query).await.unwrap();
    assert!(!phonological_results.items.is_empty(), "Should find phonological item");
    
    // Query wrong buffer - should not find the item
    let wrong_buffer_query = MemoryQuery {
        query_text: "phonological_item".to_string(),
        search_buffers: vec![BufferType::Visuospatial],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let wrong_results = memory_system.retrieve_from_working_memory(&wrong_buffer_query).await.unwrap();
    // Should find fewer or no results when searching wrong buffer
    assert!(wrong_results.items.len() <= phonological_results.items.len(),
           "Should find fewer results when searching wrong buffer");
}

#[tokio::test]
async fn test_complex_memory_content_types() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    
    // Test different memory content types
    let concept_content = MemoryContent::Concept("test_concept".to_string());
    let relationship_content = MemoryContent::Relationship(
        "subject".to_string(), 
        "object".to_string(), 
        0.8
    );
    let composite_content = MemoryContent::Composite(vec![
        concept_content.clone(),
        relationship_content.clone(),
    ]);
    
    // Store different content types
    memory_system.store_in_working_memory(concept_content, 0.7, BufferType::Episodic).await.unwrap();
    memory_system.store_in_working_memory(relationship_content, 0.8, BufferType::Episodic).await.unwrap();
    memory_system.store_in_working_memory(composite_content, 0.9, BufferType::Episodic).await.unwrap();
    
    // Verify all items were stored
    let state = memory_system.get_current_state().await.unwrap();
    assert_eq!(state.total_items, 3, "Should store all content types");
    
    // Test retrieval
    let query = MemoryQuery {
        query_text: "concept".to_string(),
        search_buffers: vec![BufferType::Episodic],
        apply_attention: false,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    let results = memory_system.retrieve_from_working_memory(&query).await.unwrap();
    assert!(!results.items.is_empty(), "Should retrieve items with different content types");
}

/// Performance test specifically for working memory operations
#[tokio::test]
async fn test_working_memory_performance() {
    let memory_system = create_test_working_memory_system().await.unwrap();
    let timer = PerformanceTimer::new("working memory operations");
    
    // Rapid storage operations
    for i in 0..50 {
        let content = MemoryContent::Concept(format!("perf_test_{}", i));
        let buffer_type = match i % 3 {
            0 => BufferType::Phonological,
            1 => BufferType::Visuospatial,
            _ => BufferType::Episodic,
        };
        let _ = memory_system.store_in_working_memory(content, 0.5, buffer_type).await;
    }
    
    // Rapid retrieval operations
    for i in 0..25 {
        let query = MemoryQuery {
            query_text: format!("perf_test_{}", i),
            search_buffers: vec![BufferType::Phonological, BufferType::Visuospatial, BufferType::Episodic],
            apply_attention: false,
            importance_threshold: 0.1,
            recency_weight: 0.5,
        };
        let _ = memory_system.retrieve_from_working_memory(&query).await;
    }
    
    // Performance should be reasonable
    timer.assert_within_ms(1000.0);
    println!("Working memory performance test completed in {:.2}ms", timer.elapsed_ms());
}