use std::sync::Arc;
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType};
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::error::Result;

#[tokio::test]
async fn test_basic_working_memory_attention_coordination() -> Result<()> {
    // Basic test to verify working memory and attention coordination
    let sdr_storage = Arc::new(SDRStorage::new(512)?);
    let activation_engine = Arc::new(ActivationPropagationEngine::new(sdr_storage.clone()).await?);
    let working_memory = Arc::new(WorkingMemorySystem::new(activation_engine.clone(), sdr_storage.clone()).await?);

    // Test basic working memory storage
    let content = MemoryContent::Concept("test_concept".to_string());
    let result = working_memory.store_in_working_memory(
        content,
        0.8,
        BufferType::Phonological,
    ).await?;

    assert!(result.success);
    println!("✓ Basic working memory storage successful");

    // Test attention-enhanced storage
    let enhanced_content = MemoryContent::Concept("attention_enhanced_concept".to_string());
    let enhanced_result = working_memory.store_in_working_memory_with_attention(
        enhanced_content,
        0.6,
        BufferType::Episodic,
        0.5, // Attention boost
    ).await?;

    assert!(enhanced_result.success);
    println!("✓ Attention-enhanced memory storage successful");

    Ok(())
}

#[tokio::test]
async fn test_basic_inhibitory_learning() -> Result<()> {
    // Test the learning mechanisms in inhibitory logic
    use llmkg::cognitive::inhibitory_logic::{CompetitiveInhibitionSystem, InhibitionPerformanceMetrics};
    use llmkg::cognitive::critical::CriticalThinking;

    let sdr_storage = Arc::new(SDRStorage::new(512)?);
    let activation_engine = Arc::new(ActivationPropagationEngine::new(sdr_storage.clone()).await?);
    let critical_thinking = Arc::new(CriticalThinking::new(activation_engine.clone()).await?);
    
    let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new(
        activation_engine,
        critical_thinking,
    ).await?);

    // Create mock performance history
    let performance_history = vec![
        InhibitionPerformanceMetrics {
            timestamp: std::time::SystemTime::now(),
            processing_time_ms: 100.0,
            entities_processed: 10,
            competition_groups_resolved: 3,
            exceptions_handled: 1,
            efficiency_score: 0.7,
            effectiveness_score: 0.8,
        },
        InhibitionPerformanceMetrics {
            timestamp: std::time::SystemTime::now(),
            processing_time_ms: 80.0,
            entities_processed: 12,
            competition_groups_resolved: 4,
            exceptions_handled: 0,
            efficiency_score: 0.8,
            effectiveness_score: 0.9,
        },
    ];

    // Test learning mechanisms
    let learning_result = inhibition_system.apply_learning_mechanisms(
        &performance_history,
        0.1, // Learning rate
    ).await?;

    assert!(learning_result.learning_confidence > 0.0);
    println!("✓ Inhibitory learning mechanisms working: confidence {:.2}", learning_result.learning_confidence);

    Ok(())
}

#[tokio::test]
async fn test_basic_memory_decay() -> Result<()> {
    // Test working memory decay mechanisms
    let sdr_storage = Arc::new(SDRStorage::new(512)?);
    let activation_engine = Arc::new(ActivationPropagationEngine::new(sdr_storage.clone()).await?);
    let working_memory = Arc::new(WorkingMemorySystem::new(activation_engine, sdr_storage).await?);

    // Store multiple items to trigger capacity management
    for i in 0..10 {
        let content = MemoryContent::Concept(format!("concept_{}", i));
        let importance = if i < 3 { 0.9 } else { 0.3 }; // First 3 are high importance
        
        let result = working_memory.store_in_working_memory(
            content,
            importance,
            BufferType::Phonological,
        ).await?;
        
        assert!(result.success);
    }

    // Check that capacity limits are respected
    let buffers = working_memory.memory_buffers.read().await;
    assert!(buffers.phonological_buffer.len() <= 9); // 7±2 capacity
    
    // High importance items should be retained
    let high_importance_retained = buffers.phonological_buffer.iter()
        .any(|item| item.importance_score > 0.8);
    assert!(high_importance_retained);
    
    println!("✓ Memory decay and capacity management working correctly");
    println!("  Buffer size: {}", buffers.phonological_buffer.len());
    println!("  High importance items retained: {}", high_importance_retained);

    Ok(())
}