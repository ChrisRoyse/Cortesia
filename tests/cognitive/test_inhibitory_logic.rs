//! Integration tests for inhibitory logic
//! 
//! Tests the public APIs of the competitive inhibition system,
//! including multi-entity competition scenarios, hierarchical inhibition patterns,
//! and learning/adaptation mechanisms.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;
use slotmap::SlotMap;

use llmkg::cognitive::inhibitory::{
    CompetitiveInhibitionSystem, InhibitionMatrix, CompetitionGroup, CompetitionType,
    TemporalDynamics, InhibitionConfig, InhibitionException, InhibitionChangeReason
};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::ActivationPattern;
use llmkg::core::types::EntityKey;
use llmkg::cognitive::critical::CriticalThinking;

/// Helper function to create a test system
async fn create_test_inhibitory_system() -> CompetitiveInhibitionSystem {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).unwrap());
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let critical_thinking = Arc::new(CriticalThinking::new(graph));
    
    CompetitiveInhibitionSystem::new(activation_engine, critical_thinking)
}

/// Helper function to create test activation pattern with specific entities
fn create_test_activation_pattern(strengths: Vec<f32>) -> (ActivationPattern, Vec<EntityKey>) {
    let mut entity_map = SlotMap::new();
    let mut activations = HashMap::new();
    let mut entity_keys = Vec::new();
    
    for strength in strengths {
        let entity = entity_map.insert(());
        activations.insert(entity, strength);
        entity_keys.push(entity);
    }
    
    (ActivationPattern { activations }, entity_keys)
}

/// Helper function to create competition groups for testing
fn create_test_competition_groups(entities: &[EntityKey]) -> Vec<CompetitionGroup> {
    vec![
        CompetitionGroup {
            group_id: "semantic_group".to_string(),
            competing_entities: vec![entities[0], entities[1]],
            competition_type: CompetitionType::Semantic,
            winner_takes_all: false,
            inhibition_strength: 0.8,
            priority: 0.9,
            temporal_dynamics: TemporalDynamics::default(),
        },
        CompetitionGroup {
            group_id: "hierarchical_group".to_string(),
            competing_entities: vec![entities[2], entities[3]],
            competition_type: CompetitionType::Hierarchical,
            winner_takes_all: true,
            inhibition_strength: 0.6,
            priority: 0.7,
            temporal_dynamics: TemporalDynamics::default(),
        },
    ]
}

#[tokio::test]
async fn test_competitive_inhibition_basic_workflow() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.9, 0.7, 0.5, 0.3]);
    
    // Test basic inhibition without any groups
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should return valid result structure
    assert!(!result.final_pattern.activations.is_empty());
    assert_eq!(result.final_pattern.activations.len(), 4);
    assert_eq!(result.inhibition_strength_applied, 0.5);
    assert!(result.competition_results.is_empty()); // No groups initially
}

#[tokio::test]
async fn test_competitive_inhibition_with_semantic_competition() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.9, 0.7, 0.5, 0.3]);
    
    // Add semantic competition group
    let group = CompetitionGroup {
        group_id: "semantic_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Semantic,
        winner_takes_all: false,
        inhibition_strength: 0.8,
        priority: 0.9,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should have competition results
    assert_eq!(result.competition_results.len(), 1);
    let comp_result = &result.competition_results[0];
    assert_eq!(comp_result.group_id, "semantic_test");
    assert!(comp_result.winner.is_some());
    assert_eq!(comp_result.winner.unwrap(), entities[0]); // Stronger entity wins
    
    // Winner should maintain strength, loser should be reduced
    let winner_strength = result.final_pattern.activations[&entities[0]];
    let loser_strength = result.final_pattern.activations[&entities[1]];
    assert_eq!(winner_strength, 0.9); // Original strength
    assert!(loser_strength < 0.7); // Reduced from original
}

#[tokio::test]
async fn test_competitive_inhibition_winner_takes_all() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.85, 0.6]);
    
    // Add winner-takes-all competition group
    let group = CompetitionGroup {
        group_id: "winner_takes_all_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Semantic,
        winner_takes_all: true,
        inhibition_strength: 0.8,
        priority: 0.9,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should trigger winner-takes-all since winner > threshold (0.8)
    assert_eq!(result.competition_results.len(), 1);
    let comp_result = &result.competition_results[0];
    assert_eq!(comp_result.winner.unwrap(), entities[0]);
    assert!(!comp_result.suppressed_entities.is_empty());
    
    // Loser should be completely suppressed
    assert_eq!(result.final_pattern.activations[&entities[1]], 0.0);
}

#[tokio::test]
async fn test_multiple_competition_groups() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.9, 0.7, 0.6, 0.4]);
    
    let groups = create_test_competition_groups(&entities);
    for group in groups {
        system.add_competition_group(group).await.unwrap();
    }
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should process both competition groups
    assert_eq!(result.competition_results.len(), 2);
    
    // Results should be sorted by priority (semantic group first with priority 0.9)
    assert_eq!(result.competition_results[0].group_id, "semantic_group");
    assert_eq!(result.competition_results[1].group_id, "hierarchical_group");
    
    // Both groups should have winners
    assert!(result.competition_results[0].winner.is_some());
    assert!(result.competition_results[1].winner.is_some());
}

#[tokio::test]
async fn test_temporal_competition() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.7, 0.6, 0.5]);
    
    // Add temporal competition group
    let group = CompetitionGroup {
        group_id: "temporal_test".to_string(),
        competing_entities: entities.clone(),
        competition_type: CompetitionType::Temporal,
        winner_takes_all: false,
        inhibition_strength: 0.7,
        priority: 0.8,
        temporal_dynamics: TemporalDynamics {
            onset_delay: Duration::from_millis(5),
            peak_time: Duration::from_millis(25),
            decay_time: Duration::from_millis(100),
            oscillation_frequency: Some(10.0),
        },
    };
    
    system.add_competition_group(group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should apply temporal competition
    assert_eq!(result.competition_results.len(), 1);
    let comp_result = &result.competition_results[0];
    assert_eq!(comp_result.group_id, "temporal_test");
    
    // Temporal competition applies alternating suppression
    // Even indices should maintain strength, odd indices should be reduced
    assert_eq!(result.final_pattern.activations[&entities[0]], 0.8); // Even: maintained
    assert!(result.final_pattern.activations[&entities[1]] < 0.7); // Odd: reduced
    assert_eq!(result.final_pattern.activations[&entities[2]], 0.6); // Even: maintained
    assert!(result.final_pattern.activations[&entities[3]] < 0.5); // Odd: reduced
}

#[tokio::test]
async fn test_hierarchical_inhibition() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.9, 0.6, 0.3, 0.8, 0.1]);
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Verify hierarchical inhibition was applied
    assert!(!result.hierarchical_result.hierarchical_layers.is_empty());
    assert_eq!(result.hierarchical_result.abstraction_levels.len(), 5);
    
    // Should identify specificity winners (most specific level entities)
    assert!(!result.hierarchical_result.specificity_winners.is_empty());
    
    // Verify abstraction levels are assigned correctly
    for (entity, level) in &result.hierarchical_result.abstraction_levels {
        assert!(entities.contains(entity));
        assert!(*level <= 2); // Should be 0, 1, or 2
    }
}

#[tokio::test]
async fn test_exception_handling() {
    let system = create_test_inhibitory_system().await;
    
    // Create pattern with resource contention (many highly active entities)
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.9, 0.7, 0.8, 0.9, 0.85]);
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should detect and handle exceptions
    assert!(!result.exception_result.exceptions_detected.is_empty());
    
    // Should detect resource contention
    let has_resource_contention = result.exception_result.exceptions_detected
        .iter()
        .any(|e| matches!(e, InhibitionException::ResourceContention(_)));
    assert!(has_resource_contention);
    
    // Should attempt to resolve exceptions
    if !result.exception_result.exceptions_detected.is_empty() {
        assert!(!result.exception_result.resolutions_applied.is_empty() || 
                !result.exception_result.unresolved_conflicts.is_empty());
    }
}

#[tokio::test]
async fn test_learning_mechanisms() {
    let mut system = create_test_inhibitory_system().await;
    system.inhibition_config.enable_learning = true;
    
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6, 0.4]);
    
    // Add competition group
    let group = CompetitionGroup {
        group_id: "learning_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Semantic,
        winner_takes_all: false,
        inhibition_strength: 0.5,
        priority: 0.8,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Learning should be applied
    assert!(system.inhibition_config.enable_learning);
    
    // Check learning status
    let learning_status = system.check_learning_status().await.unwrap();
    assert!(learning_status.learning_enabled);
    assert!(!learning_status.parameters_learned.is_empty());
    assert!(learning_status.learning_confidence > 0.0);
}

#[tokio::test]
async fn test_would_compete_integration() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6, 0.4]);
    
    // Initially, entities should not compete
    let competes = system.would_compete(entities[0], entities[1]).await.unwrap();
    assert!(!competes);
    
    // Add competition group
    let group = CompetitionGroup {
        group_id: "would_compete_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Semantic,
        winner_takes_all: false,
        inhibition_strength: 0.7,
        priority: 0.8,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    // Now they should compete
    let competes = system.would_compete(entities[0], entities[1]).await.unwrap();
    assert!(competes);
    
    // Entity not in group should not compete
    let competes = system.would_compete(entities[0], entities[2]).await.unwrap();
    assert!(!competes);
}

#[tokio::test]
async fn test_update_competition_strength_integration() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6]);
    
    // Update competition strength
    let change = system.update_competition_strength(
        entities[0], 
        entities[1], 
        0.7
    ).await.unwrap();
    
    assert_eq!(change.entities_affected.len(), 2);
    assert_eq!(change.strength_change, 0.7);
    assert!(matches!(change.change_reason, InhibitionChangeReason::LearningAdjustment));
    
    // Should create a new competition group
    let groups = system.competition_groups.read().await;
    assert_eq!(groups.len(), 1);
    assert!(groups[0].competing_entities.contains(&entities[0]));
    assert!(groups[0].competing_entities.contains(&entities[1]));
    assert_eq!(groups[0].inhibition_strength, 0.7);
    
    // Entities should now compete
    let competes = system.would_compete(entities[0], entities[1]).await.unwrap();
    assert!(competes);
}

#[tokio::test]
async fn test_create_learned_competition_groups_integration() {
    let system = create_test_inhibitory_system().await;
    
    // Create activation history where entities rarely co-activate
    let (pattern1, entities) = create_test_activation_pattern(vec![0.8, 0.1, 0.7, 0.2]);
    let (pattern2, _) = create_test_activation_pattern(vec![0.2, 0.9, 0.1, 0.8]);
    let (pattern3, _) = create_test_activation_pattern(vec![0.9, 0.0, 0.8, 0.1]);
    
    // Update activation patterns to use same entities
    let mut pattern2_updated = ActivationPattern { activations: HashMap::new() };
    let mut pattern3_updated = ActivationPattern { activations: HashMap::new() };
    
    for (i, &entity) in entities.iter().enumerate() {
        pattern2_updated.activations.insert(entity, if i % 2 == 0 { 0.2 } else { 0.9 });
        pattern3_updated.activations.insert(entity, if i % 2 == 0 { 0.9 } else { 0.0 });
    }
    
    let history = vec![pattern1, pattern2_updated, pattern3_updated];
    
    let learned_groups = system.create_learned_competition_groups(&history, 0.8).await.unwrap();
    
    // Should create learned competition groups
    assert!(!learned_groups.is_empty());
    
    for group in &learned_groups {
        assert_eq!(group.competing_entities.len(), 2);
        assert!(group.inhibition_strength > 0.0);
        assert_eq!(group.competition_type, CompetitionType::Semantic);
        assert!(group.group_id.contains("learned_competition"));
    }
    
    // Groups should be added to the system
    let groups = system.competition_groups.read().await;
    assert_eq!(groups.len(), learned_groups.len());
}

#[tokio::test]
async fn test_inhibition_with_domain_context() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6, 0.4]);
    
    // Add competition group
    let group = CompetitionGroup {
        group_id: "context_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Contextual,
        winner_takes_all: false,
        inhibition_strength: 0.7,
        priority: 0.8,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    // Test with domain context
    let result = system.apply_competitive_inhibition(
        &pattern, 
        Some("cognitive_processing".to_string())
    ).await.unwrap();
    
    // Should process with context
    assert_eq!(result.competition_results.len(), 1);
    assert!(result.competition_results[0].winner.is_some());
}

#[tokio::test]
async fn test_spatial_and_causal_competition() {
    let system = create_test_inhibitory_system().await;
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6, 0.7, 0.5]);
    
    // Add spatial competition group
    let spatial_group = CompetitionGroup {
        group_id: "spatial_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Spatial,
        winner_takes_all: false,
        inhibition_strength: 0.6,
        priority: 0.7,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    // Add causal competition group
    let causal_group = CompetitionGroup {
        group_id: "causal_test".to_string(),
        competing_entities: vec![entities[2], entities[3]],
        competition_type: CompetitionType::Causal,
        winner_takes_all: false,
        inhibition_strength: 0.8,
        priority: 0.6,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(spatial_group).await.unwrap();
    system.add_competition_group(causal_group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should process both competition types
    assert_eq!(result.competition_results.len(), 2);
    
    // Both should have winners
    assert!(result.competition_results[0].winner.is_some());
    assert!(result.competition_results[1].winner.is_some());
}

#[tokio::test]
async fn test_integration_with_custom_config() {
    // Create system with custom configuration
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).unwrap());
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let critical_thinking = Arc::new(CriticalThinking::new(graph));
    
    let mut system = CompetitiveInhibitionSystem::new(activation_engine, critical_thinking);
    
    // Customize inhibition config
    system.inhibition_config = InhibitionConfig {
        global_inhibition_strength: 0.7,
        lateral_inhibition_strength: 0.8,
        hierarchical_inhibition_strength: 0.9,
        contextual_inhibition_strength: 0.5,
        winner_takes_all_threshold: 0.7,
        soft_competition_factor: 0.4,
        temporal_integration_window: Duration::from_millis(150),
        enable_learning: true,
    };
    
    let (pattern, entities) = create_test_activation_pattern(vec![0.8, 0.6]);
    
    // Add competition group
    let group = CompetitionGroup {
        group_id: "custom_config_test".to_string(),
        competing_entities: vec![entities[0], entities[1]],
        competition_type: CompetitionType::Semantic,
        winner_takes_all: true,
        inhibition_strength: 0.9,
        priority: 0.8,
        temporal_dynamics: TemporalDynamics::default(),
    };
    
    system.add_competition_group(group).await.unwrap();
    
    let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
    
    // Should use custom configuration
    assert_eq!(result.inhibition_strength_applied, 0.7); // Custom global strength
    assert_eq!(result.competition_results.len(), 1);
    
    // With lower threshold (0.7) and strong entity (0.8), should trigger winner-takes-all
    assert_eq!(result.final_pattern.activations[&entities[1]], 0.0);
}