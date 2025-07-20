/// Integration tests for the attention manager system
/// These tests focus on the public API and system behavior
use llmkg::cognitive::attention_manager::{AttentionManager, AttentionState, AttentionSnapshot, AttentionType, AttentionFocus, ExecutiveCommand};
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::types::EntityKey;
use std::sync::Arc;
use anyhow::Result;
use std::time::Duration;
use tokio::time::sleep;

// Import shared test utilities
use crate::cognitive::test_utils::{
    create_test_entity_keys, 
    create_test_memory_items,
    create_memory_item,
    PerformanceTimer,
    scenarios
};

/// Creates a fully configured attention manager for testing
async fn create_test_attention_manager() -> Result<AttentionManager> {
    let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(Default::default())?);
    let orchestrator = Arc::new(CognitiveOrchestrator::new(
        brain_graph.clone(),
        CognitiveOrchestratorConfig::default()
    ).await?);
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(Default::default()));
    let working_memory = Arc::new(WorkingMemorySystem::new(activation_engine.clone(), sdr_storage).await?);
    
    AttentionManager::new(orchestrator, activation_engine, working_memory).await
}

mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_attention_focus_with_activation_boost() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(3);
        
        // Focus attention on the first target
        let focus = AttentionFocus {
            target: targets[0],
            intensity: 0.8,
            duration: Duration::from_millis(100),
        };
        
        manager.focus_attention(focus).await?;
        
        // Get snapshot and verify
        let snapshot = manager.get_attention_snapshot().await?;
        assert!(snapshot.focused_entities.contains(&targets[0]));
        assert!(snapshot.attention_capacity < 1.0); // Some capacity used
        assert!(snapshot.cognitive_load > 0.0); // Some load present
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_executive_control_commands() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(3);
        
        // First, focus on initial target
        let initial_focus = AttentionFocus {
            target: targets[0],
            intensity: 0.7,
            duration: Duration::from_millis(50),
        };
        manager.focus_attention(initial_focus).await?;
        
        // Switch focus using executive command
        let switch_command = ExecutiveCommand::SwitchFocus {
            from: targets[0],
            to: targets[1],
            urgency: 0.9,
        };
        
        manager.execute_executive_command(switch_command).await?;
        
        // Verify focus has switched
        let snapshot = manager.get_attention_snapshot().await?;
        assert!(snapshot.focused_entities.contains(&targets[1]));
        
        // Test inhibit distraction
        let inhibit_command = ExecutiveCommand::InhibitDistraction {
            distractors: vec![targets[2]],
        };
        manager.execute_executive_command(inhibit_command).await?;
        
        // Test boost attention
        let boost_command = ExecutiveCommand::BoostAttention {
            target: targets[1],
            boost_factor: 1.5,
        };
        manager.execute_executive_command(boost_command).await?;
        
        // Test clear focus
        manager.execute_executive_command(ExecutiveCommand::ClearFocus).await?;
        let final_snapshot = manager.get_attention_snapshot().await?;
        assert!(final_snapshot.focused_entities.is_empty());
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_divided_attention_management() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(4);
        
        // Set divided attention mode
        manager.set_attention_mode(AttentionType::Divided).await?;
        
        // Focus on multiple targets
        for (i, target) in targets.iter().enumerate() {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.6 - (i as f32 * 0.1), // Decreasing intensity
                duration: Duration::from_millis(50),
            };
            manager.focus_attention(focus).await?;
        }
        
        // Verify all targets are in focus
        let snapshot = manager.get_attention_snapshot().await?;
        assert_eq!(snapshot.attention_type, AttentionType::Divided);
        assert!(snapshot.focused_entities.len() >= 2); // At least 2 in divided attention
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_sustained_attention_over_time() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let target = create_test_entity_keys(1)[0];
        
        // Set sustained attention mode
        manager.set_attention_mode(AttentionType::Sustained).await?;
        
        // Focus with longer duration
        let focus = AttentionFocus {
            target,
            intensity: 0.7,
            duration: Duration::from_millis(200),
        };
        
        manager.focus_attention(focus).await?;
        
        // Check attention is maintained over time
        let initial_snapshot = manager.get_attention_snapshot().await?;
        assert!(initial_snapshot.focused_entities.contains(&target));
        
        // Wait a bit
        sleep(Duration::from_millis(100)).await;
        
        // Verify attention is still maintained
        let later_snapshot = manager.get_attention_snapshot().await?;
        assert!(later_snapshot.focused_entities.contains(&target));
        assert_eq!(later_snapshot.attention_type, AttentionType::Sustained);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_alternating_attention_patterns() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(3);
        
        // Set alternating attention mode
        manager.set_attention_mode(AttentionType::Alternating).await?;
        
        // Focus on different targets in sequence
        for target in &targets {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.75,
                duration: Duration::from_millis(75),
            };
            manager.focus_attention(focus).await?;
            
            // Small delay between alternations
            sleep(Duration::from_millis(50)).await;
        }
        
        let snapshot = manager.get_attention_snapshot().await?;
        assert_eq!(snapshot.attention_type, AttentionType::Alternating);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_cognitive_load_adaptation() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(10);
        
        // Gradually increase cognitive load
        for (i, target) in targets.iter().enumerate() {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.8,
                duration: Duration::from_millis(50),
            };
            manager.focus_attention(focus).await?;
            
            let snapshot = manager.get_attention_snapshot().await?;
            
            // As we add more targets, cognitive load should increase
            if i > 0 {
                assert!(snapshot.cognitive_load > 0.0);
            }
            
            // System should start limiting attention as load increases
            if snapshot.cognitive_load > 0.8 {
                assert!(snapshot.focused_entities.len() < targets.len());
            }
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_memory_integration_workflow() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(3);
        
        // Focus attention to influence working memory
        for target in &targets {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.9,
                duration: Duration::from_millis(100),
            };
            manager.focus_attention(focus).await?;
        }
        
        // Update based on working memory state
        manager.update_from_working_memory().await?;
        
        // Verify the attention system adapted
        let snapshot = manager.get_attention_snapshot().await?;
        assert!(snapshot.cognitive_load > 0.0);
        
        Ok(())
    }
}

mod scenario_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_predefined_scenarios() -> Result<()> {
        let test_scenarios = scenarios::generate_scenarios();
        
        for scenario in test_scenarios {
            let mut manager = create_test_attention_manager().await?;
            
            // Apply cognitive load
            let mut state = AttentionState::new();
            state.update_cognitive_load(scenario.cognitive_load);
            
            // Focus on all targets
            for target in &scenario.targets {
                let focus = AttentionFocus {
                    target: *target,
                    intensity: 0.7,
                    duration: Duration::from_millis(50),
                };
                let _ = manager.focus_attention(focus).await;
            }
            
            let snapshot = manager.get_attention_snapshot().await?;
            
            // Under high load, system should limit focus
            if scenario.cognitive_load > 0.7 {
                assert!(
                    snapshot.focused_entities.len() <= scenario.expected_focus_count,
                    "Scenario '{}' failed: expected at most {} focused entities under load {}, got {}",
                    scenario.name,
                    scenario.expected_focus_count,
                    scenario.cognitive_load,
                    snapshot.focused_entities.len()
                );
            }
        }
        
        Ok(())
    }
}

mod performance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_attention_switching_performance() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(100);
        
        let timer = PerformanceTimer::new("Attention switching");
        
        // Rapidly switch attention
        for i in 0..50 {
            let focus = AttentionFocus {
                target: targets[i % targets.len()],
                intensity: 0.7,
                duration: Duration::from_millis(10),
            };
            manager.focus_attention(focus).await?;
        }
        
        // Should complete within reasonable time
        timer.assert_within_ms(500.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_snapshot_generation_performance() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        let targets = create_test_entity_keys(20);
        
        // Populate attention state
        for target in &targets[..10] {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.6,
                duration: Duration::from_millis(20),
            };
            manager.focus_attention(focus).await?;
        }
        
        let timer = PerformanceTimer::new("Snapshot generation");
        
        // Generate many snapshots
        for _ in 0..100 {
            let _ = manager.get_attention_snapshot().await?;
        }
        
        // Should be very fast
        timer.assert_within_ms(100.0);
        
        Ok(())
    }
}

mod edge_case_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_empty_targets_handling() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
        
        // Get snapshot with no focused entities
        let snapshot = manager.get_attention_snapshot().await?;
        assert!(snapshot.focused_entities.is_empty());
        assert_eq!(snapshot.cognitive_load, 0.0);
        assert_eq!(snapshot.attention_capacity, 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_rapid_mode_switching() -> Result<()> {
        let mut manager = create_test_attention_manager().await?;
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
            manager.set_attention_mode(mode.clone()).await?;
            
            let focus = AttentionFocus {
                target,
                intensity: 0.5,
                duration: Duration::from_millis(10),
            };
            manager.focus_attention(focus).await?;
            
            let snapshot = manager.get_attention_snapshot().await?;
            assert_eq!(snapshot.attention_type, mode);
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_extreme_cognitive_load() -> Result<()> {
        let manager = create_test_attention_manager().await?;
        
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
            // This test verifies that AttentionFocus creation maintains invariants
            let targets = create_test_entity_keys(1);
            let focus = AttentionFocus {
                target: targets[0],
                intensity,
                duration: Duration::from_millis(duration_ms),
            };
            
            // Intensity should be preserved
            prop_assert_eq!(focus.intensity, intensity);
            
            // Duration should be preserved
            prop_assert_eq!(focus.duration.as_millis() as u64, duration_ms);
        }
    }
}