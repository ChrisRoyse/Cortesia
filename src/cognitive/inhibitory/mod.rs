//! Competitive Inhibition System
//! 
//! This module implements competitive inhibition mechanisms inspired by neural processing,
//! including lateral inhibition, hierarchical competition, and temporal dynamics.

pub mod types;
pub mod matrix;
pub mod competition;
pub mod hierarchical;
pub mod exceptions;
pub mod learning;
pub mod integration;
pub mod dynamics;
pub mod metrics;

pub use types::*;
pub use matrix::InhibitionMatrixOps;

use crate::core::activation_engine::ActivationPropagationEngine;
use crate::core::brain_types::ActivationPattern;
use crate::core::types::EntityKey;
use crate::cognitive::critical::CriticalThinking;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use uuid;

#[derive(Clone)]
pub struct CompetitiveInhibitionSystem {
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub critical_thinking: Arc<CriticalThinking>,
    pub inhibition_matrix: Arc<RwLock<InhibitionMatrix>>,
    pub competition_groups: Arc<RwLock<Vec<CompetitionGroup>>>,
    pub inhibition_config: InhibitionConfig,
}

impl std::fmt::Debug for CompetitiveInhibitionSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompetitiveInhibitionSystem")
            .field("activation_engine", &"ActivationPropagationEngine")
            .field("critical_thinking", &"CriticalThinking")
            .field("inhibition_matrix", &"Arc<RwLock<InhibitionMatrix>>")
            .field("competition_groups", &"Arc<RwLock<Vec<CompetitionGroup>>>")
            .field("inhibition_config", &self.inhibition_config)
            .finish()
    }
}

impl CompetitiveInhibitionSystem {
    pub fn new(
        activation_engine: Arc<ActivationPropagationEngine>,
        critical_thinking: Arc<CriticalThinking>,
    ) -> Self {
        Self {
            activation_engine,
            critical_thinking,
            inhibition_matrix: Arc::new(RwLock::new(InhibitionMatrix::new())),
            competition_groups: Arc::new(RwLock::new(Vec::new())),
            inhibition_config: InhibitionConfig::default(),
        }
    }

    /// Main entry point for applying competitive inhibition
    pub async fn apply_competitive_inhibition(
        &self,
        activation_pattern: &ActivationPattern,
        _domain_context: Option<String>,
    ) -> Result<InhibitionResult> {
        // Create a working copy of the activation pattern
        let mut working_pattern = activation_pattern.clone();
        let mut competition_results = Vec::new();

        // Step 1: Apply group-based competition
        let group_results = competition::apply_group_competition(
            self,
            &mut working_pattern,
            &self.competition_groups,
            &self.inhibition_config,
        ).await?;
        competition_results.extend(group_results);

        // Step 2: Apply hierarchical inhibition
        let hierarchical_result = hierarchical::apply_hierarchical_inhibition(
            self,
            &mut working_pattern,
            &self.inhibition_matrix,
            &self.inhibition_config,
        ).await?;

        // Step 3: Handle special cases and exceptions
        let exception_result = exceptions::handle_inhibition_exceptions(
            self,
            &mut working_pattern,
            &competition_results,
            &hierarchical_result,
        ).await?;

        // Step 4: Apply temporal dynamics
        dynamics::apply_temporal_dynamics(
            &mut working_pattern,
            &competition_results,
            &self.inhibition_config,
        ).await?;

        // Step 5: Apply learning mechanisms if enabled
        if self.inhibition_config.enable_learning {
            learning::apply_adaptive_learning(
                self,
                &working_pattern,
                &competition_results,
            ).await?;
        }

        Ok(InhibitionResult {
            competition_results,
            hierarchical_result,
            exception_result,
            final_pattern: working_pattern,
            inhibition_strength_applied: self.inhibition_config.global_inhibition_strength,
        })
    }

    /// Add a new competition group
    pub async fn add_competition_group(&self, group: CompetitionGroup) -> Result<()> {
        let mut groups = self.competition_groups.write().await;
        groups.push(group);
        Ok(())
    }

    /// Check if two entities would compete
    pub async fn would_compete(&self, entity_a: EntityKey, entity_b: EntityKey) -> Result<bool> {
        let competition_groups = self.competition_groups.read().await;
        
        for group in competition_groups.iter() {
            if group.competing_entities.contains(&entity_a) && 
               group.competing_entities.contains(&entity_b) {
                return Ok(true);
            }
        }
        
        // Also check semantic similarity for potential competition
        let inhibition_matrix = self.inhibition_matrix.read().await;
        if let Some(&strength) = inhibition_matrix.lateral_inhibition.get(&(entity_a, entity_b)) {
            return Ok(strength > 0.1); // Threshold for competition
        }
        if let Some(&strength) = inhibition_matrix.lateral_inhibition.get(&(entity_b, entity_a)) {
            return Ok(strength > 0.1);
        }
        
        Ok(false)
    }

    /// Update competition strength between two entities
    pub async fn update_competition_strength(
        &self,
        entity_a: EntityKey,
        entity_b: EntityKey,
        strength_change: f32,
    ) -> Result<InhibitionChange> {
        let mut inhibition_matrix = self.inhibition_matrix.write().await;
        
        // Get current strength
        let current_strength = inhibition_matrix.lateral_inhibition
            .get(&(entity_a, entity_b))
            .or_else(|| inhibition_matrix.lateral_inhibition.get(&(entity_b, entity_a)))
            .copied()
            .unwrap_or(0.0);
        
        let new_strength = (current_strength + strength_change).clamp(0.0, 1.0);
        
        // Update the inhibition matrix
        inhibition_matrix.lateral_inhibition.insert((entity_a, entity_b), new_strength);
        
        // Check if we need to create or update a competition group
        let mut competition_groups = self.competition_groups.write().await;
        let mut found_group = false;
        
        for group in competition_groups.iter_mut() {
            if group.competing_entities.contains(&entity_a) || 
               group.competing_entities.contains(&entity_b) {
                // Update existing group
                if !group.competing_entities.contains(&entity_a) {
                    group.competing_entities.push(entity_a);
                }
                if !group.competing_entities.contains(&entity_b) {
                    group.competing_entities.push(entity_b);
                }
                group.inhibition_strength = new_strength;
                found_group = true;
                break;
            }
        }
        
        if !found_group && new_strength > 0.1 {
            // Create new competition group
            competition_groups.push(CompetitionGroup {
                group_id: format!("learned_group_{}", uuid::Uuid::new_v4()),
                competing_entities: vec![entity_a, entity_b],
                competition_type: CompetitionType::Semantic,
                winner_takes_all: false,
                inhibition_strength: new_strength,
                priority: 0.5,
                temporal_dynamics: TemporalDynamics::default(),
            });
        }
        
        Ok(InhibitionChange {
            competition_group: format!("entities_{:?}_{:?}", entity_a, entity_b),
            entities_affected: vec![entity_a, entity_b],
            strength_change,
            change_reason: InhibitionChangeReason::LearningAdjustment,
        })
    }

    /// Create learned competition groups based on activation history
    pub async fn create_learned_competition_groups(
        &self,
        activation_history: &[ActivationPattern],
        correlation_threshold: f32,
    ) -> Result<Vec<CompetitionGroup>> {
        // Analyze co-activation patterns
        let mut co_activation_counts: HashMap<(EntityKey, EntityKey), usize> = HashMap::new();
        let mut total_patterns = 0;
        
        for pattern in activation_history {
            total_patterns += 1;
            let active_entities: Vec<_> = pattern.activations
                .iter()
                .filter(|(_, &strength)| strength > 0.5)
                .map(|(key, _)| *key)
                .collect();
            
            // Count co-activations
            for i in 0..active_entities.len() {
                for j in (i + 1)..active_entities.len() {
                    let pair = {
                        use slotmap::{Key, KeyData};
                        let data_i: KeyData = active_entities[i].data();
                        let data_j: KeyData = active_entities[j].data();
                        if data_i < data_j {
                            (active_entities[i], active_entities[j])
                        } else {
                            (active_entities[j], active_entities[i])
                        }
                    };
                    *co_activation_counts.entry(pair).or_insert(0) += 1;
                }
            }
        }
        
        // Find entities that rarely co-activate (potential competitors)
        let mut new_groups = Vec::new();
        for ((entity_a, entity_b), count) in co_activation_counts {
            let co_activation_rate = count as f32 / total_patterns as f32;
            
            // Low co-activation suggests competition
            if co_activation_rate < (1.0 - correlation_threshold) {
                new_groups.push(CompetitionGroup {
                    group_id: format!("learned_competition_{}", uuid::Uuid::new_v4()),
                    competing_entities: vec![entity_a, entity_b],
                    competition_type: CompetitionType::Semantic,
                    winner_takes_all: false,
                    inhibition_strength: 1.0 - co_activation_rate,
                    priority: 0.6,
                    temporal_dynamics: TemporalDynamics::default(),
                });
            }
        }
        
        // Add to system
        for group in &new_groups {
            self.add_competition_group(group.clone()).await?;
        }
        
        Ok(new_groups)
    }

    /// Check learning status
    pub async fn check_learning_status(&self) -> Result<LearningStatus> {
        Ok(LearningStatus {
            learning_enabled: self.inhibition_config.enable_learning,
            parameters_learned: vec![
                "lateral_inhibition_strength".to_string(),
                "competition_aggressiveness".to_string(),
                "temporal_decay_rate".to_string(),
            ],
            learning_confidence: 0.7,
            adaptation_count: 0,
            last_learning_timestamp: std::time::SystemTime::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::core::brain_types::ActivationPattern;
    use crate::cognitive::critical::CriticalThinking;
    use std::collections::HashMap;

    fn create_test_system() -> CompetitiveInhibitionSystem {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).unwrap());
        let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
        let critical_thinking = Arc::new(CriticalThinking::new(graph));
        
        CompetitiveInhibitionSystem::new(activation_engine, critical_thinking)
    }

    fn create_test_activation_pattern() -> ActivationPattern {
        let mut activations = HashMap::new();
        
        // Create test entities
        let entity1 = EntityKey::from_hash("entity1");
        let entity2 = EntityKey::from_hash("entity2");
        let entity3 = EntityKey::from_hash("entity3");
        
        activations.insert(entity1, 0.8);
        activations.insert(entity2, 0.6);
        activations.insert(entity3, 0.4);
        
        let mut pattern = ActivationPattern::new("test".to_string());
        pattern.activations = activations;
        pattern
    }

    #[tokio::test]
    async fn test_system_creation() {
        let system = create_test_system();
        
        // Verify system is properly initialized
        assert_eq!(system.inhibition_config.global_inhibition_strength, 0.5);
        assert_eq!(system.inhibition_config.lateral_inhibition_strength, 0.7);
        assert!(system.inhibition_config.enable_learning);
        
        // Verify empty initial state
        let groups = system.competition_groups.read().await;
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn test_add_competition_group() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        let group = CompetitionGroup {
            group_id: "test_group".to_string(),
            competing_entities: entity_keys.clone(),
            competition_type: CompetitionType::Semantic,
            winner_takes_all: false,
            inhibition_strength: 0.8,
            priority: 0.7,
            temporal_dynamics: TemporalDynamics::default(),
        };

        system.add_competition_group(group).await.unwrap();
        
        let groups = system.competition_groups.read().await;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_id, "test_group");
        assert_eq!(groups[0].competing_entities.len(), entity_keys.len());
    }

    #[tokio::test]
    async fn test_would_compete_with_groups() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        let group = CompetitionGroup {
            group_id: "test_group".to_string(),
            competing_entities: vec![entity_keys[0], entity_keys[1]],
            competition_type: CompetitionType::Semantic,
            winner_takes_all: false,
            inhibition_strength: 0.8,
            priority: 0.7,
            temporal_dynamics: TemporalDynamics::default(),
        };

        system.add_competition_group(group).await.unwrap();
        
        // Test entities in same group compete
        let competes = system.would_compete(entity_keys[0], entity_keys[1]).await.unwrap();
        assert!(competes);
        
        // Test entities not in same group don't compete
        let competes = system.would_compete(entity_keys[0], entity_keys[2]).await.unwrap();
        assert!(!competes);
    }

    #[tokio::test]
    async fn test_would_compete_with_matrix() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        // Add inhibition relationship to matrix
        {
            let mut matrix = system.inhibition_matrix.write().await;
            matrix.lateral_inhibition.insert((entity_keys[0], entity_keys[1]), 0.5);
        }
        
        let competes = system.would_compete(entity_keys[0], entity_keys[1]).await.unwrap();
        assert!(competes);
        
        // Test below threshold
        {
            let mut matrix = system.inhibition_matrix.write().await;
            matrix.lateral_inhibition.insert((entity_keys[0], entity_keys[2]), 0.05);
        }
        
        let competes = system.would_compete(entity_keys[0], entity_keys[2]).await.unwrap();
        assert!(!competes);
    }

    #[tokio::test]
    async fn test_update_competition_strength() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        let result = system.update_competition_strength(
            entity_keys[0], 
            entity_keys[1], 
            0.5
        ).await.unwrap();
        
        assert_eq!(result.entities_affected.len(), 2);
        assert_eq!(result.strength_change, 0.5);
        
        // Verify matrix was updated
        let matrix = system.inhibition_matrix.read().await;
        let strength = matrix.lateral_inhibition.get(&(entity_keys[0], entity_keys[1]));
        assert!(strength.is_some());
        assert_eq!(*strength.unwrap(), 0.5);
    }

    #[tokio::test]
    async fn test_update_competition_strength_creates_group() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        system.update_competition_strength(
            entity_keys[0], 
            entity_keys[1], 
            0.5
        ).await.unwrap();
        
        // Should create a new competition group
        let groups = system.competition_groups.read().await;
        assert_eq!(groups.len(), 1);
        assert!(groups[0].competing_entities.contains(&entity_keys[0]));
        assert!(groups[0].competing_entities.contains(&entity_keys[1]));
    }

    #[tokio::test]
    async fn test_create_learned_competition_groups() {
        let system = create_test_system();
        let pattern1 = create_test_activation_pattern();
        let pattern2 = create_test_activation_pattern();
        
        // Create history where entities rarely co-activate
        let history = vec![pattern1, pattern2];
        
        let groups = system.create_learned_competition_groups(&history, 0.8).await.unwrap();
        
        // Should identify competing pairs
        assert!(!groups.is_empty());
        for group in &groups {
            assert_eq!(group.competing_entities.len(), 2);
            assert!(group.inhibition_strength > 0.0);
            assert_eq!(group.competition_type, CompetitionType::Semantic);
        }
    }

    #[tokio::test]
    async fn test_check_learning_status() {
        let system = create_test_system();
        
        let status = system.check_learning_status().await.unwrap();
        
        assert!(status.learning_enabled);
        assert!(!status.parameters_learned.is_empty());
        assert!(status.learning_confidence > 0.0);
        assert_eq!(status.adaptation_count, 0);
    }

    #[tokio::test]
    async fn test_apply_competitive_inhibition() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        
        let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
        
        // Verify result structure
        assert!(result.competition_results.is_empty()); // No groups initially
        assert!(!result.final_pattern.activations.is_empty());
        assert_eq!(result.inhibition_strength_applied, 0.5);
    }

    #[tokio::test]
    async fn test_apply_competitive_inhibition_with_group() {
        let system = create_test_system();
        let pattern = create_test_activation_pattern();
        let entity_keys: Vec<_> = pattern.activations.keys().copied().collect();
        
        // Add a competition group
        let group = CompetitionGroup {
            group_id: "test_group".to_string(),
            competing_entities: entity_keys.clone(),
            competition_type: CompetitionType::Semantic,
            winner_takes_all: true,
            inhibition_strength: 0.8,
            priority: 0.7,
            temporal_dynamics: TemporalDynamics::default(),
        };
        
        system.add_competition_group(group).await.unwrap();
        
        let result = system.apply_competitive_inhibition(&pattern, None).await.unwrap();
        
        // Should have competition results
        assert_eq!(result.competition_results.len(), 1);
        assert!(result.competition_results[0].winner.is_some());
        
        // Winner should have high activation, others should be suppressed
        let winner = result.competition_results[0].winner.unwrap();
        let winner_strength = result.final_pattern.activations.get(&winner).unwrap();
        assert!(*winner_strength > 0.5);
    }
}