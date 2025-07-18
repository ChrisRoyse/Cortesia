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
use slotmap::Key;
use crate::core::types::EntityKey;
use crate::cognitive::critical::CriticalThinking;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

#[derive(Clone)]
pub struct CompetitiveInhibitionSystem {
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub critical_thinking: Arc<CriticalThinking>,
    pub inhibition_matrix: Arc<RwLock<InhibitionMatrix>>,
    pub competition_groups: Arc<RwLock<Vec<CompetitionGroup>>>,
    pub inhibition_config: InhibitionConfig,
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
        domain_context: Option<String>,
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
                    let pair = if active_entities[i].data() < active_entities[j].data() {
                        (active_entities[i], active_entities[j])
                    } else {
                        (active_entities[j], active_entities[i])
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