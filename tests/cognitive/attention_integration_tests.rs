//! Integration tests for AttentionManager public API
//! Tests the complete attention management workflow through public interfaces

use llmkg::cognitive::attention_manager::{
    AttentionManager, AttentionType, ExecutiveCommand, AttentionTarget, AttentionTargetType,
};
use llmkg::cognitive::CognitivePatternType;
use std::time::Duration;
use tokio::time::sleep;
use anyhow::Result;

// Import test support utilities for integration tests
use llmkg::test_support::data::{create_standard_test_entities};
use llmkg::core::types::{EntityKey, EntityData};
use std::time::Instant;

/// Creates test EntityKeys for integration testing
fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
    use slotmap::SlotMap;
    
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    let mut keys = Vec::new();
    
    for i in 0..count {
        let key = sm.insert(EntityData::new(
            1,
            format!("integration_test_entity_{}", i),
            vec![0.0; 64],
        ));
        keys.push(key);
    }
    
    keys
}

/// Performance timer for integration tests
struct PerformanceTimer {
    start: Instant,
    operation: String,
}

impl PerformanceTimer {
    fn new(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
        }
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    fn assert_within_ms(&self, max_ms: f64) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed <= max_ms,
            "{} took {:.2}ms, expected less than {:.2}ms",
            self.operation,
            elapsed,
            max_ms
        );
    }
}

/// Creates a fully configured attention manager for integration testing
async fn create_integration_test_manager() -> Result<AttentionManager> {
    use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
    use llmkg::cognitive::working_memory::WorkingMemorySystem;
    use llmkg::core::activation_engine::{ActivationPropagationEngine, ActivationConfig};
    use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
    use llmkg::core::sdr_storage::SDRStorage;
    use llmkg::core::sdr_types::SDRConfig;
    use std::sync::Arc;

    let config = BrainEnhancedConfig::default();
    let embedding_dim = 128;

    // Create the brain graph
    let graph = Arc::new(
        BrainEnhancedKnowledgeGraph::new_with_config(embedding_dim, config.clone())?
    );

    // Create cognitive orchestrator
    let cognitive_config = CognitiveOrchestratorConfig::default();
    let orchestrator = Arc::new(CognitiveOrchestrator::new(
        graph.clone(),
        cognitive_config,
    ).await?);

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
    ).await?);

    // Create attention manager
    Ok(AttentionManager::new(
        orchestrator,
        activation_engine,
        working_memory,
    ).await?)
}

#[cfg(test)]
mod attention_focus_integration {
    use super::*;

    #[tokio::test]
    async fn test_attention_focus_and_shift() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(3);

        // Test focusing attention
        let result = attention_manager.focus_attention(
            targets.clone(),
            1.0,
            AttentionType::Selective
        ).await?;

        assert_eq!(result.focused_entities.len(), 3);
        assert_eq!(result.attention_strength, 1.0);
        assert!(result.cognitive_load_change > 0.0);

        // Verify through public API
        let snapshot = attention_manager.get_attention_state().await?;
        assert!(snapshot.current_targets.len() >= 1);
        assert_eq!(snapshot.attention_type, AttentionType::Selective);
        assert!(snapshot.focus_strength > 0.0);

        // Test shifting to new targets
        let new_targets = create_test_entity_keys(2);
        let shift_result = attention_manager.shift_attention(
            targets,
            new_targets.clone(),
            0.8
        ).await?;

        assert!(shift_result.shift_success);
        assert!(shift_result.shift_duration.as_millis() < 1000);
        assert!(shift_result.attention_continuity >= 0.0);

        // Verify shift worked
        let new_snapshot = attention_manager.get_attention_state().await?;
        assert!(new_snapshot.current_targets.len() >= 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_divided_attention_workflow() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let entity_keys = create_test_entity_keys(4);

        // Create attention targets
        let targets = entity_keys.iter().enumerate().map(|(_i, &key)| AttentionTarget {
            entity_key: key,
            attention_weight: 0.8 - (_i as f32 * 0.1),
            priority: 1.0 - (_i as f32 * 0.2),
            duration: Duration::from_millis(500),
            target_type: AttentionTargetType::Entity,
        }).collect();

        // Test divided attention management
        let result = attention_manager.manage_divided_attention(targets).await?;

        assert_eq!(result.focused_entities.len(), 4);
        assert!(result.attention_strength > 0.0);
        assert!(result.cognitive_load_change > 0.0);

        // Verify state reflects divided attention
        let snapshot = attention_manager.get_attention_state().await?;
        assert_eq!(snapshot.attention_type, AttentionType::Divided);
        assert!(snapshot.current_targets.len() >= 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_sustained_attention_over_time() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let target = create_test_entity_keys(1);

        // Focus with sustained attention
        let result = attention_manager.focus_attention(
            target.clone(),
            0.9,
            AttentionType::Sustained
        ).await?;

        assert!(result.attention_strength > 0.0);

        // Verify initial state
        let initial_snapshot = attention_manager.get_attention_state().await?;
        assert_eq!(initial_snapshot.attention_type, AttentionType::Sustained);
        assert!(initial_snapshot.current_targets.contains(&target[0]));

        // Wait and verify attention is maintained
        sleep(Duration::from_millis(100)).await;

        let later_snapshot = attention_manager.get_attention_state().await?;
        assert!(later_snapshot.current_targets.contains(&target[0]));
        assert!(later_snapshot.focus_strength > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_alternating_attention_patterns() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(3);

        // Test alternating between targets
        for (_i, &target) in targets.iter().enumerate() {
            let result = attention_manager.focus_attention(
                vec![target],
                0.7,
                AttentionType::Alternating
            ).await?;

            assert_eq!(result.focused_entities.len(), 1);
            assert_eq!(result.focused_entities[0], target);

            // Small delay between alternations
            sleep(Duration::from_millis(50)).await;

            // Verify each focus
            let snapshot = attention_manager.get_attention_state().await?;
            assert_eq!(snapshot.attention_type, AttentionType::Alternating);
            assert!(snapshot.current_targets.contains(&target));
        }

        Ok(())
    }
}

#[cfg(test)]
mod executive_control_integration {
    use super::*;

    #[tokio::test]
    async fn test_executive_command_workflow() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(3);

        // Test switch focus command
        let switch_command = ExecutiveCommand::SwitchFocus {
            from: targets[0],
            to: targets[1],
            urgency: 0.9,
        };

        let result = attention_manager.executive_control(switch_command).await?;
        assert!(result.attention_strength > 0.0);

        // Test inhibit distraction command
        let inhibit_command = ExecutiveCommand::InhibitDistraction {
            distractors: vec![targets[2]],
        };

        let inhibit_result = attention_manager.executive_control(inhibit_command).await?;
        assert!(inhibit_result.working_memory_updates.len() > 0);

        // Test boost attention command
        let boost_command = ExecutiveCommand::BoostAttention {
            target: targets[1],
            boost_factor: 1.2,
        };

        let boost_result = attention_manager.executive_control(boost_command).await?;
        assert!(boost_result.attention_strength > 0.0);

        // Test clear focus command
        let clear_command = ExecutiveCommand::ClearFocus;
        let _clear_result = attention_manager.executive_control(clear_command).await?;
        
        // Verify focus was cleared
        let snapshot = attention_manager.get_attention_state().await?;
        assert!(snapshot.current_targets.is_empty() || snapshot.focus_strength == 0.0);

        Ok(())
    }
}

#[cfg(test)]
mod memory_integration_workflow {
    use super::*;

    #[tokio::test]
    async fn test_attention_memory_coordination() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(3);

        // Test memory-coordinated attention focus
        let result = attention_manager.focus_attention_with_memory_coordination(
            targets.clone(),
            0.8,
            AttentionType::Selective
        ).await?;

        assert_eq!(result.focused_entities, targets);
        assert!(result.attention_strength > 0.0);

        // Check memory state after focus
        let memory_state = attention_manager.get_attention_memory_state().await?;
        assert!(!memory_state.current_attention_targets.is_empty());
        assert!(memory_state.memory_load >= 0.0);
        assert!(!memory_state.attention_memory_coordination.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_preserving_attention_shift() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let from_targets = create_test_entity_keys(2);
        let to_targets = create_test_entity_keys(2);

        // First focus on initial targets
        attention_manager.focus_attention_with_memory_coordination(
            from_targets.clone(),
            0.7,
            AttentionType::Selective
        ).await?;

        // Test memory-preserving shift
        let shift_result = attention_manager.shift_attention_with_memory_preservation(
            from_targets,
            to_targets.clone(),
            0.6
        ).await?;

        assert!(shift_result.shift_success);
        assert!(shift_result.working_memory_impact >= 0.0);

        // Verify final state
        let final_state = attention_manager.get_attention_state().await?;
        assert!(!final_state.current_targets.is_empty());

        Ok(())
    }
}

#[cfg(test)]
mod cognitive_pattern_integration {
    use super::*;

    #[tokio::test]
    async fn test_convergent_pattern_coordination() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;

        let result = attention_manager.coordinate_with_cognitive_patterns(
            CognitivePatternType::Convergent,
            "Focus on specific solution"
        ).await?;

        assert_eq!(result.attention_config.attention_type, AttentionType::Selective);
        assert!(result.attention_config.focus_strength >= 0.8);
        assert!(result.pattern_compatibility > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_divergent_pattern_coordination() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;

        let result = attention_manager.coordinate_with_cognitive_patterns(
            CognitivePatternType::Divergent,
            "Explore multiple possibilities"
        ).await?;

        assert_eq!(result.attention_config.attention_type, AttentionType::Divided);
        assert!(result.attention_config.focus_strength >= 0.5);
        assert!(result.pattern_compatibility > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_critical_pattern_coordination() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;

        let result = attention_manager.coordinate_with_cognitive_patterns(
            CognitivePatternType::Critical,
            "Analyze critical decision point"
        ).await?;

        assert_eq!(result.attention_config.attention_type, AttentionType::Executive);
        assert!(result.attention_config.focus_strength >= 0.9);
        assert!(result.pattern_compatibility > 0.0);

        Ok(())
    }
}

#[cfg(test)]
mod performance_validation {
    use super::*;

    #[tokio::test]
    async fn test_attention_state_query_performance() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(5);

        // Set up some attention state
        attention_manager.focus_attention(
            targets,
            0.8,
            AttentionType::Divided
        ).await?;

        let timer = PerformanceTimer::new("Attention state queries");

        // Perform many state queries
        for _i in 0..100 {
            let _state = attention_manager.get_attention_state().await?;
            let _memory_state = attention_manager.get_attention_memory_state().await?;
        }

        timer.assert_within_ms(500.0); // Should be very fast

        Ok(())
    }

    #[tokio::test]
    async fn test_rapid_attention_switching() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(10);

        let timer = PerformanceTimer::new("Rapid attention switching");

        // Rapidly switch between targets
        for i in 0..20 {
            let target_index = i % targets.len();
            attention_manager.focus_attention(
                vec![targets[target_index]],
                0.7,
                AttentionType::Selective
            ).await?;
        }

        timer.assert_within_ms(1000.0); // Should complete within 1 second

        // Verify final state is coherent
        let final_state = attention_manager.get_attention_state().await?;
        assert!(final_state.current_targets.len() >= 1);

        Ok(())
    }
}

#[cfg(test)]
mod error_handling_and_recovery {
    use super::*;

    #[tokio::test]
    async fn test_empty_input_handling() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;

        // Test with empty targets
        let result = attention_manager.focus_attention(
            vec![],
            0.5,
            AttentionType::Selective
        ).await?;

        assert!(result.focused_entities.is_empty());
        assert_eq!(result.attention_strength, 0.5);

        // Verify state is valid
        let state = attention_manager.get_attention_state().await?;
        assert!(state.current_targets.is_empty() || state.focus_strength == 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_attention_recovery_after_operations() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let targets = create_test_entity_keys(3);

        // Perform multiple operations
        for _i in 0..10 {
            let _result = attention_manager.focus_attention(
                targets.clone(),
                1.0,
                AttentionType::Divided
            ).await;
        }

        // System should still be responsive
        let recovery_result = attention_manager.focus_attention(
            vec![targets[0]],
            0.5,
            AttentionType::Selective
        ).await?;

        assert!(recovery_result.attention_strength > 0.0);

        // State should be valid
        let state = attention_manager.get_attention_state().await?;
        assert!(state.cognitive_load >= 0.0 && state.cognitive_load <= 1.0);
        assert!(state.attention_capacity > 0.0 && state.attention_capacity <= 1.0);

        Ok(())
    }
}

#[cfg(test)]
mod real_world_scenarios {
    use super::*;

    #[tokio::test]
    async fn test_research_task_scenario() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let concepts = create_test_entity_keys(5);

        // Simulate research task: broad exploration then focused analysis
        
        // 1. Initial divergent exploration
        attention_manager.coordinate_with_cognitive_patterns(
            CognitivePatternType::Divergent,
            "Explore research domain"
        ).await?;

        // 2. Focus on multiple concepts
        attention_manager.focus_attention(
            concepts.clone(),
            0.6,
            AttentionType::Divided
        ).await?;

        // 3. Narrow to critical analysis
        attention_manager.coordinate_with_cognitive_patterns(
            CognitivePatternType::Critical,
            "Analyze key findings"
        ).await?;

        // 4. Final convergent focus
        attention_manager.focus_attention_with_memory_coordination(
            vec![concepts[0]],
            0.9,
            AttentionType::Sustained
        ).await?;

        // Verify final state represents focused attention
        let final_state = attention_manager.get_attention_state().await?;
        assert!(final_state.focus_strength > 0.8);
        assert!(!final_state.current_targets.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_problem_solving_scenario() -> Result<()> {
        let attention_manager = create_integration_test_manager().await?;
        let problem_aspects = create_test_entity_keys(4);

        // Simulate problem-solving workflow
        
        // 1. Understand problem (sustained attention)
        attention_manager.focus_attention(
            vec![problem_aspects[0]],
            0.8,
            AttentionType::Sustained
        ).await?;

        // 2. Generate alternatives (divided attention)
        attention_manager.manage_divided_attention(
            problem_aspects.iter().map(|&key| AttentionTarget {
                entity_key: key,
                attention_weight: 0.7,
                priority: 1.0,
                duration: Duration::from_millis(200),
                target_type: AttentionTargetType::Concept,
            }).collect()
        ).await?;

        // 3. Evaluate options (alternating attention)
        for aspect in &problem_aspects {
            attention_manager.focus_attention(
                vec![*aspect],
                0.8,
                AttentionType::Alternating
            ).await?;
            sleep(Duration::from_millis(50)).await;
        }

        // 4. Make decision (executive control)
        attention_manager.executive_control(ExecutiveCommand::SwitchFocus {
            from: problem_aspects[1],
            to: problem_aspects[0],
            urgency: 0.9,
        }).await?;

        // Verify decision state
        let decision_state = attention_manager.get_attention_state().await?;
        assert!(decision_state.focus_strength > 0.7);

        Ok(())
    }
}