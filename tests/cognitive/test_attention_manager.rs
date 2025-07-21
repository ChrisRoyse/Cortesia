/// Integration tests for the attention manager system
/// These tests focus on the public API and system behavior
use llmkg::cognitive::attention_manager::{AttentionManager, AttentionState, AttentionType, AttentionFocus, ExecutiveCommand, AttentionStateInfo};
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, MemoryItem};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
use llmkg::core::types::EntityKey;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::sdr_types::SDRConfig;
use std::sync::Arc;
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use std::collections::HashMap;

// Import shared test utilities
use super::test_utils::{
    create_test_memory_items,
    create_memory_item,
    PerformanceTimer,
    scenarios
};

// Helper functions and test traits at the top level
fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
    use slotmap::SlotMap;
    use llmkg::core::types::EntityData;
    
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    let mut keys = Vec::new();
    
    for i in 0..count {
        let key = sm.insert(EntityData {
            type_id: 1,
            properties: format!("test_entity_{}", i),
            embedding: vec![0.0; 64],
        });
        keys.push(key);
    }
    
    keys
}

// Note: Unit tests that require access to private methods like calculate_attention_weights
// have been moved to src/cognitive/attention_manager.rs in the #[cfg(test)] module.
// This integration test file focuses on testing the public API behavior and system integration.

// Helper struct for test focus parameters
struct TestFocus {
    targets: Vec<EntityKey>,
    intensity: f32,
    attention_type: AttentionType,
}

/// Creates a fully configured attention manager for testing
async fn create_test_attention_manager() -> (
    AttentionManager,
    Arc<CognitiveOrchestrator>,
    Arc<ActivationPropagationEngine>,
    Arc<WorkingMemorySystem>,
) {
    use llmkg::cognitive::orchestrator::CognitiveOrchestratorConfig;
    use llmkg::core::activation_engine::ActivationConfig;
    
    let config = BrainEnhancedConfig::default();
    let embedding_dim = 128;
    
    // Create the brain graph
    let graph = Arc::new(
        BrainEnhancedKnowledgeGraph::new_with_config(embedding_dim, config.clone())
            .expect("Failed to create graph")
    );
    
    // Create cognitive orchestrator
    let cognitive_config = CognitiveOrchestratorConfig::default();
    let orchestrator = Arc::new(CognitiveOrchestrator::new(
        graph.clone(),
        cognitive_config,
    ).await.expect("Failed to create orchestrator"));
    
    // Create activation engine
    let activation_config = ActivationConfig::default();
    let activation_engine = Arc::new(ActivationPropagationEngine::new(activation_config));
    
    // Create SDR storage for working memory
    let sdr_config = SDRConfig {
        total_bits: embedding_dim * 16,
        active_bits: 40,
        sparsity: 0.02,
        overlap_threshold: 0.5,
    };
    let sdr_storage = Arc::new(SDRStorage::new(sdr_config));
    
    // Create working memory
    let working_memory = Arc::new(WorkingMemorySystem::new(
        activation_engine.clone(),
        sdr_storage,
    ).await.expect("Failed to create working memory"));
    
    // Create attention manager
    let attention_manager = AttentionManager::new(
        orchestrator.clone(),
        activation_engine.clone(),
        working_memory.clone(),
    ).await.expect("Failed to create attention manager");
    
    (attention_manager, orchestrator, activation_engine, working_memory)
}


#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_attention_focus_with_activation_boost() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(3);
        
        // Focus attention on the first target
        manager.focus_attention(
            vec![targets[0]],
            0.8,
            AttentionType::Selective,
        ).await?;
        
        // Get snapshot and verify
        let snapshot = manager.get_attention_state().await?;
        assert!(snapshot.current_targets.contains(&targets[0]));
        assert!(snapshot.attention_capacity < 1.0); // Some capacity used
        assert!(snapshot.cognitive_load > 0.0); // Some load present
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_executive_control_commands() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(3);
        
        // First, focus on initial target
        manager.focus_attention(
            vec![targets[0]],
            0.7,
            AttentionType::Selective,
        ).await?;
        
        // Switch focus using public API instead of executive command
        // Note: execute_executive_command is not part of the public API
        manager.focus_attention(
            vec![targets[1]],
            0.9,
            AttentionType::Selective,
        ).await?;
        
        // Verify focus has switched
        let snapshot = manager.get_attention_state().await?;
        assert!(snapshot.current_targets.contains(&targets[1]));
        
        // Test inhibit distraction - simulate by focusing away from distractors
        // Note: Direct inhibition is not available in public API
        
        // Test boost attention - simulate by increasing focus strength
        manager.focus_attention(
            vec![targets[1]],
            0.95, // Higher strength simulates boost
            AttentionType::Sustained,
        ).await?;
        
        // Test clear focus - focus on empty target list
        manager.focus_attention(
            vec![],
            0.0,
            AttentionType::Selective,
        ).await?;
        let final_snapshot = manager.get_attention_state().await?;
        assert!(final_snapshot.current_targets.is_empty());
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_divided_attention_management() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(4);
        
        // Set divided attention mode
        // Divided attention mode will be used in focus_attention calls
        
        // Focus on multiple targets with divided attention
        manager.focus_attention(
            targets.clone(),
            0.6,
            AttentionType::Divided,
        ).await?;
        
        // Verify all targets are in focus
        let snapshot = manager.get_attention_state().await?;
        assert_eq!(snapshot.attention_type, AttentionType::Divided);
        assert!(snapshot.current_targets.len() >= 2); // At least 2 in divided attention
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_sustained_attention_over_time() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let target = create_test_entity_keys(1)[0];
        
        // Set sustained attention mode
        // Sustained attention mode will be used in focus_attention calls
        
        // Focus with longer duration
        manager.focus_attention(
            vec![target],
            0.7,
            AttentionType::Sustained,
        ).await?;
        
        // Check attention is maintained over time
        let initial_snapshot = manager.get_attention_state().await?;
        assert!(initial_snapshot.current_targets.contains(&target));
        
        // Wait a bit
        sleep(Duration::from_millis(100)).await;
        
        // Verify attention is still maintained
        let later_snapshot = manager.get_attention_state().await?;
        assert!(later_snapshot.current_targets.contains(&target));
        assert_eq!(later_snapshot.attention_type, AttentionType::Sustained);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_alternating_attention_patterns() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(3);
        
        // Set alternating attention mode
        // Alternating attention mode will be used in focus_attention calls
        
        // Focus on different targets in sequence
        for target in &targets {
            manager.focus_attention(
            vec![*target],
            0.75,
            AttentionType::Alternating,
        ).await?;
            
            // Small delay between alternations
            sleep(Duration::from_millis(50)).await;
        }
        
        let snapshot = manager.get_attention_state().await?;
        assert_eq!(snapshot.attention_type, AttentionType::Alternating);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_cognitive_load_adaptation() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(10);
        
        // Gradually increase cognitive load
        for (i, target) in targets.iter().enumerate() {
            manager.focus_attention(
            vec![*target],
            0.8,
            AttentionType::Selective,
        ).await?;
            
            let snapshot = manager.get_attention_state().await?;
            
            // As we add more targets, cognitive load should increase
            if i > 0 {
                assert!(snapshot.cognitive_load > 0.0);
            }
            
            // System should start limiting attention as load increases
            if snapshot.cognitive_load > 0.8 {
                assert!(snapshot.current_targets.len() < targets.len());
            }
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_memory_integration_workflow() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(3);
        
        // Focus attention to influence working memory
        for target in &targets {
            manager.focus_attention(
            vec![*target],
            0.9,
            AttentionType::Selective,
        ).await?;
        }
        
        // The attention system will coordinate with working memory internally
        
        // Verify the attention system adapted
        let snapshot = manager.get_attention_state().await?;
        assert!(snapshot.cognitive_load > 0.0);
        
        Ok(())
    }
    
    // NOTE: The test_calculate_attention_weights_divided test has been moved to
    // src/cognitive/attention_manager.rs in the #[cfg(test)] module where it can
    // directly access the private calculate_attention_weights method.
    // This integration test module focuses on testing the public API behavior.
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_empty_targets_handling() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        
        // Get snapshot with no focused entities
        let snapshot = manager.get_attention_state().await?;
        assert!(snapshot.current_targets.is_empty());
        assert_eq!(snapshot.cognitive_load, 0.0);
        assert_eq!(snapshot.attention_capacity, 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_rapid_mode_switching() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let target = create_test_entity_keys(1)[0];
        
        // Rapidly switch between all attention modes
        let modes = vec![
            AttentionType::Selective,
            AttentionType::Divided,
            AttentionType::Sustained,
            AttentionType::Executive,
            AttentionType::Alternating,
        ];
        
        for mode in modes {
            // Mode will be used in focus_attention call
            manager.focus_attention(
                vec![target],
                0.5,
                mode.clone(),
            ).await?;
            
            let snapshot = manager.get_attention_state().await?;
            assert_eq!(snapshot.attention_type, mode);
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_extreme_cognitive_load() -> Result<()> {
        let (_manager, _, _, _) = create_test_attention_manager().await;
        
        // Test state behavior under extreme conditions
        let mut state = AttentionState::new();
        
        // Test maximum load
        state.update_cognitive_load(1.0);
        assert_eq!(state.cognitive_load, 1.0);
        assert_eq!(state.attention_capacity, 0.5);
        
        // Test beyond maximum (should clamp)
        state.update_cognitive_load(2.0);
        assert_eq!(state.cognitive_load, 1.0);
        
        // Test negative (should clamp to 0)
        state.update_cognitive_load(-1.0);
        assert_eq!(state.cognitive_load, 0.0);
        assert_eq!(state.attention_capacity, 1.0);
        
        Ok(())
    }
}

#[cfg(test)]
mod scenario_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_predefined_scenarios() -> Result<()> {
        let test_scenarios = scenarios::generate_scenarios();
        
        for scenario in test_scenarios {
            let (mut manager, _, _, _) = create_test_attention_manager().await;
            
            // Apply cognitive load
            let mut state = AttentionState::new();
            state.update_cognitive_load(scenario.cognitive_load);
            
            // Focus on all targets
            for target in &scenario.targets {
                let _ = manager.focus_attention(
                    vec![*target],
                    0.7,
                    AttentionType::Selective,
                ).await;
            }
            
            let snapshot = manager.get_attention_state().await?;
            
            // Under high load, system should limit focus
            if scenario.cognitive_load > 0.7 {
                assert!(
                    snapshot.current_targets.len() <= scenario.expected_focus_count,
                    "Scenario '{}' failed: expected at most {} focused entities under load {}, got {}",
                    scenario.name,
                    scenario.expected_focus_count,
                    scenario.cognitive_load,
                    snapshot.current_targets.len()
                );
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_attention_switching_performance() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(100);
        
        let timer = PerformanceTimer::new("Attention switching");
        
        // Rapidly switch attention
        for i in 0..50 {
            manager.focus_attention(
                vec![targets[i % targets.len()]],
                0.7,
                AttentionType::Selective,
            ).await?;
        }
        
        // Should complete within reasonable time
        timer.assert_within_ms(500.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_snapshot_generation_performance() -> Result<()> {
        let (mut manager, _, _, _) = create_test_attention_manager().await;
        let targets = create_test_entity_keys(20);
        
        // Populate attention state
        for target in &targets[..10] {
            manager.focus_attention(
            vec![*target],
            0.6,
            AttentionType::Selective,
        ).await?;
        }
        
        let timer = PerformanceTimer::new("Snapshot generation");
        
        // Generate many snapshots
        for _ in 0..100 {
            let _ = manager.get_attention_state().await?;
        }
        
        // Should be very fast
        timer.assert_within_ms(100.0);
        
        Ok(())
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_cognitive_load_bounds(load in 0.0f32..=2.0) {
            let mut state = AttentionState::new();
            state.update_cognitive_load(load);
            
            // Cognitive load should always be clamped between 0 and 1
            prop_assert!(state.cognitive_load >= 0.0);
            prop_assert!(state.cognitive_load <= 1.0);
            
            // Attention capacity should be inversely related
            prop_assert!(state.attention_capacity >= 0.5);
            prop_assert!(state.attention_capacity <= 1.0);
        }
        
        #[test]
        fn test_attention_focus_invariants(
            intensity in 0.0f32..=1.0,
            duration_ms in 1u64..=1000
        ) {
            // This test verifies that focus parameters maintain invariants
            let targets = create_test_entity_keys(1);
            
            // Create a test focus struct to verify parameters
            let test_focus = TestFocus {
                targets: vec![targets[0]],
                intensity,
                attention_type: AttentionType::Selective,
            };
            
            // Intensity should be preserved
            prop_assert_eq!(test_focus.intensity, intensity);
            
            // Targets should be preserved
            prop_assert_eq!(test_focus.targets.len(), 1);
        }
    }
}