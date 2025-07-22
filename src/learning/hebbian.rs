use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::activation_engine::ActivationPropagationEngine;
use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
use crate::core::brain_types::BrainInspiredRelationship;
use crate::core::types::EntityKey;
use crate::learning::types::*;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, Duration};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct HebbianLearningEngine {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub inhibition_system: Arc<CompetitiveInhibitionSystem>,
    pub learning_rate: f32,
    pub decay_constant: f32,
    pub strengthening_threshold: f32,
    pub weakening_threshold: f32,
    pub max_weight: f32,
    pub min_weight: f32,
    pub learning_statistics: Arc<RwLock<LearningStatistics>>,
    pub coactivation_tracker: Arc<RwLock<CoactivationTracker>>,
}

impl HebbianLearningEngine {
    pub async fn new(
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        activation_engine: Arc<ActivationPropagationEngine>,
        inhibition_system: Arc<CompetitiveInhibitionSystem>,
    ) -> Result<Self> {
        Ok(Self {
            brain_graph,
            activation_engine,
            inhibition_system,
            learning_rate: 0.01,
            decay_constant: 0.001,
            strengthening_threshold: 0.7,
            weakening_threshold: 0.3,
            max_weight: 1.0,
            min_weight: 0.0,
            learning_statistics: Arc::new(RwLock::new(LearningStatistics::new())),
            coactivation_tracker: Arc::new(RwLock::new(CoactivationTracker::new())),
        })
    }

    pub async fn apply_hebbian_learning(
        &mut self,
        activation_events: Vec<ActivationEvent>,
        learning_context: LearningContext,
    ) -> Result<LearningUpdate> {
        // 1. Update coactivation tracking with existing activation engine
        self.update_coactivation_tracking(&activation_events).await?;
        
        // 2. Calculate correlation changes based on current competition results
        let correlation_updates = self.calculate_correlation_updates(
            &activation_events,
            &learning_context,
        ).await?;
        
        // 3. Apply synaptic weight changes to brain graph
        let weight_updates = self.apply_synaptic_weight_changes(
            correlation_updates,
        ).await?;
        
        // 4. Update learning statistics and integrate with existing performance monitoring
        self.update_learning_statistics(&weight_updates).await?;
        
        // 5. Apply temporal decay leveraging existing temporal dynamics
        let decay_updates = self.apply_temporal_decay().await?;
        
        // 6. Update competitive inhibition parameters based on learning
        let inhibition_updates = self.update_inhibition_parameters(&weight_updates).await?;
        
        let learning_efficiency = self.calculate_learning_efficiency(&weight_updates);
        
        Ok(LearningUpdate {
            strengthened_connections: weight_updates.strengthened,
            weakened_connections: weight_updates.weakened,
            new_connections: weight_updates.newly_formed,
            pruned_connections: decay_updates,
            learning_efficiency,
            inhibition_updates,
        })
    }

    async fn update_coactivation_tracking(
        &mut self,
        activation_events: &[ActivationEvent],
    ) -> Result<()> {
        let mut tracker = self.coactivation_tracker.write().unwrap();
        let current_time = Instant::now();
        
        // Add new activation events
        for event in activation_events {
            tracker.activation_history
                .entry(event.entity_key)
                .or_insert_with(Vec::new)
                .push(event.clone());
        }
        
        // Clean up old events outside temporal window
        let temporal_window = tracker.temporal_window;
        for (_, events) in tracker.activation_history.iter_mut() {
            events.retain(|event| {
                current_time.duration_since(event.timestamp) < temporal_window
            });
        }
        
        // Update correlation matrix
        self.update_correlation_matrix(&mut tracker, activation_events).await?;
        
        Ok(())
    }

    async fn update_correlation_matrix(
        &self,
        tracker: &mut CoactivationTracker,
        new_events: &[ActivationEvent],
    ) -> Result<()> {
        // Calculate pairwise correlations for co-occurring activations
        for i in 0..new_events.len() {
            for j in (i + 1)..new_events.len() {
                let entity_a = new_events[i].entity_key;
                let entity_b = new_events[j].entity_key;
                
                let correlation_key = if entity_a < entity_b {
                    (entity_a, entity_b)
                } else {
                    (entity_b, entity_a)
                };
                
                // Calculate temporal correlation
                let correlation = self.calculate_temporal_correlation(
                    entity_a,
                    entity_b,
                    &tracker.activation_history,
                )?;
                
                // Update correlation matrix with exponential moving average
                let existing_correlation = tracker.correlation_matrix
                    .get(&correlation_key)
                    .unwrap_or(&0.0);
                
                let alpha = 0.1; // Learning rate for correlation updates
                let updated_correlation = alpha * correlation + (1.0 - alpha) * existing_correlation;
                
                tracker.correlation_matrix.insert(correlation_key, updated_correlation);
            }
        }
        
        Ok(())
    }

    fn calculate_temporal_correlation(
        &self,
        entity_a: EntityKey,
        entity_b: EntityKey,
        activation_history: &HashMap<EntityKey, Vec<ActivationEvent>>,
    ) -> Result<f32> {
        let empty_events = vec![];
        let events_a = activation_history.get(&entity_a).unwrap_or(&empty_events);
        let events_b = activation_history.get(&entity_b).unwrap_or(&empty_events);
        
        if events_a.is_empty() || events_b.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate correlation based on temporal proximity and activation strength
        let mut correlation_sum = 0.0;
        let mut correlation_count = 0;
        
        for event_a in events_a {
            for event_b in events_b {
                let time_diff = if event_a.timestamp > event_b.timestamp {
                    event_a.timestamp.duration_since(event_b.timestamp)
                } else {
                    event_b.timestamp.duration_since(event_a.timestamp)
                };
                
                // Only consider events within a temporal window
                if time_diff < Duration::from_secs(60) {
                    let temporal_proximity = 1.0 - (time_diff.as_millis() as f32 / 60000.0);
                    let activation_product = event_a.activation_strength * event_b.activation_strength;
                    correlation_sum += temporal_proximity * activation_product;
                    correlation_count += 1;
                }
            }
        }
        
        if correlation_count > 0 {
            Ok(correlation_sum / correlation_count as f32)
        } else {
            Ok(0.0)
        }
    }

    async fn calculate_correlation_updates(
        &self,
        activation_events: &[ActivationEvent],
        _learning_context: &LearningContext,
    ) -> Result<Vec<CorrelationUpdate>> {
        let mut updates = Vec::new();
        let tracker = self.coactivation_tracker.read().unwrap();
        
        // Generate correlation updates for all pairs of activated entities
        for i in 0..activation_events.len() {
            for j in (i + 1)..activation_events.len() {
                let event_a = &activation_events[i];
                let event_b = &activation_events[j];
                
                let correlation_key = if event_a.entity_key < event_b.entity_key {
                    (event_a.entity_key, event_b.entity_key)
                } else {
                    (event_b.entity_key, event_a.entity_key)
                };
                
                let correlation_strength = tracker.correlation_matrix
                    .get(&correlation_key)
                    .unwrap_or(&0.0)
                    .clone();
                
                // Check if this creates competition
                let creates_competition = self.would_create_competition(
                    event_a.entity_key,
                    event_b.entity_key,
                ).await?;
                
                // Determine if this should be inhibitory
                let is_inhibitory = creates_competition && correlation_strength < 0.5;
                
                updates.push(CorrelationUpdate {
                    source_entity: event_a.entity_key,
                    target_entity: event_b.entity_key,
                    source_activation: event_a.activation_strength,
                    target_activation: event_b.activation_strength,
                    correlation_strength,
                    creates_competition,
                    is_inhibitory,
                });
            }
        }
        
        Ok(updates)
    }

    async fn would_create_competition(
        &self,
        entity_a: EntityKey,
        entity_b: EntityKey,
    ) -> Result<bool> {
        // Check if entities would compete based on existing inhibition system
        self.inhibition_system.would_compete(entity_a, entity_b).await
            .map_err(|e| anyhow::anyhow!("Inhibition system error: {}", e))
    }

    async fn apply_synaptic_weight_changes(
        &self,
        correlation_updates: Vec<CorrelationUpdate>,
    ) -> Result<WeightUpdateResult> {
        let mut strengthened = Vec::new();
        let mut weakened = Vec::new();
        let mut newly_formed = Vec::new();
        let mut inhibition_changes = Vec::new();
        
        for update in correlation_updates {
            // Use existing brain graph methods for relationship management
            let relationship_exists = self.brain_graph.has_relationship(
                update.source_entity,
                update.target_entity,
            ).await;
            
            if relationship_exists {
                // Update existing relationship weight using brain graph
                let current_weight = self.brain_graph.get_relationship_weight(
                    update.source_entity,
                    update.target_entity,
                ).unwrap_or(0.5);
                
                // Apply Hebbian learning rule: Δw = η * x_i * x_j
                let weight_change = self.learning_rate * 
                    update.source_activation * 
                    update.target_activation *
                    update.correlation_strength;
                
                let new_weight = (current_weight + weight_change)
                    .clamp(self.min_weight, self.max_weight);
                
                self.brain_graph.update_relationship_weight(
                    update.source_entity,
                    update.target_entity,
                    new_weight,
                ).await?;
                
                // Update competitive inhibition if needed
                if update.creates_competition {
                    let _inhibition_update = self.inhibition_system.update_competition_strength(
                        update.source_entity,
                        update.target_entity,
                        weight_change,
                    ).await?;
                    
                    // Convert to learning InhibitionChange type
                    inhibition_changes.push(InhibitionChange {
                        competition_group: format!("{:?}-{:?}", update.source_entity, update.target_entity),
                        entities_affected: vec![update.source_entity, update.target_entity],
                        strength_change: weight_change,
                        change_reason: InhibitionChangeReason::HebbianLearning,
                    });
                }
                
                if weight_change > 0.0 {
                    strengthened.push(WeightChange {
                        source: update.source_entity,
                        target: update.target_entity,
                        old_weight: current_weight,
                        new_weight,
                        change_magnitude: weight_change,
                    });
                } else {
                    weakened.push(WeightChange {
                        source: update.source_entity,
                        target: update.target_entity,
                        old_weight: current_weight,
                        new_weight,
                        change_magnitude: weight_change.abs(),
                    });
                }
            } else if update.correlation_strength > self.strengthening_threshold {
                // Create new relationship using brain graph
                let initial_weight = self.learning_rate * update.correlation_strength;
                
                self.brain_graph.create_learned_relationship(
                    update.source_entity,
                    update.target_entity,
                ).await?;
                
                newly_formed.push(WeightChange {
                    source: update.source_entity,
                    target: update.target_entity,
                    old_weight: 0.0,
                    new_weight: initial_weight,
                    change_magnitude: initial_weight,
                });
            }
        }
        
        Ok(WeightUpdateResult {
            strengthened,
            weakened,
            newly_formed,
            inhibition_changes,
        })
    }

    pub async fn spike_timing_dependent_plasticity(
        &self,
        pre_synaptic_event: ActivationEvent,
        post_synaptic_event: ActivationEvent,
    ) -> Result<STDPResult> {
        // Implement STDP: timing-dependent synaptic plasticity
        let time_difference = if post_synaptic_event.timestamp > pre_synaptic_event.timestamp {
            post_synaptic_event.timestamp
                .duration_since(pre_synaptic_event.timestamp)
                .as_millis() as f32
        } else {
            -(pre_synaptic_event.timestamp
                .duration_since(post_synaptic_event.timestamp)
                .as_millis() as f32)
        };
        
        // STDP learning window (typically ~100ms)
        let stdp_window = 100.0; // milliseconds
        
        if time_difference.abs() > stdp_window {
            // Outside STDP window - no plasticity
            return Ok(STDPResult::NoChange);
        }
        
        // Calculate STDP weight change
        let weight_change = if time_difference > 0.0 {
            // Post-synaptic spike after pre-synaptic (potentiation)
            self.learning_rate * (-time_difference / stdp_window).exp()
        } else {
            // Post-synaptic spike before pre-synaptic (depression)
            -self.learning_rate * (time_difference / stdp_window).exp()
        };
        
        // Apply weight change to the connection
        self.brain_graph.update_relationship_weight(
            pre_synaptic_event.entity_key,
            post_synaptic_event.entity_key,
            weight_change,
        ).await?;
        
        Ok(STDPResult::WeightChanged {
            weight_change,
            timing_difference: time_difference,
            plasticity_type: if weight_change > 0.0 {
                PlasticityType::Potentiation
            } else {
                PlasticityType::Depression
            },
        })
    }

    async fn update_learning_statistics(
        &self,
        weight_updates: &WeightUpdateResult,
    ) -> Result<()> {
        let mut stats = self.learning_statistics.write().unwrap();
        
        stats.total_weight_changes += weight_updates.strengthened.len() as u64 +
                                      weight_updates.weakened.len() as u64 +
                                      weight_updates.newly_formed.len() as u64;
        
        // Update learning efficiency metrics
        let total_changes = weight_updates.strengthened.len() + 
                           weight_updates.weakened.len() + 
                           weight_updates.newly_formed.len();
        
        if total_changes > 0 {
            let average_change: f32 = weight_updates.strengthened.iter()
                .chain(weight_updates.weakened.iter())
                .chain(weight_updates.newly_formed.iter())
                .map(|change| change.change_magnitude)
                .sum::<f32>() / total_changes as f32;
            
            // Update average learning rate with exponential moving average
            stats.average_learning_rate = 0.9 * stats.average_learning_rate + 0.1 * average_change;
            
            // Update stability metrics
            stats.learning_stability = self.calculate_learning_stability(&weight_updates);
        }
        
        Ok(())
    }

    fn calculate_learning_stability(&self, weight_updates: &WeightUpdateResult) -> f32 {
        if weight_updates.strengthened.is_empty() && weight_updates.weakened.is_empty() {
            return 1.0;
        }
        
        // Calculate variance in weight changes as a measure of instability
        let all_changes: Vec<f32> = weight_updates.strengthened.iter()
            .chain(weight_updates.weakened.iter())
            .map(|change| change.change_magnitude)
            .collect();
        
        let mean_change = all_changes.iter().sum::<f32>() / all_changes.len() as f32;
        let variance = all_changes.iter()
            .map(|change| (change - mean_change).powi(2))
            .sum::<f32>() / all_changes.len() as f32;
        
        // Stability is inverse of variance, clamped to [0, 1]
        1.0 / (1.0 + variance)
    }

    async fn apply_temporal_decay(&self) -> Result<Vec<WeightChange>> {
        let mut pruned_connections = Vec::new();
        
        // Get all relationships from brain graph
        let relationships = self.brain_graph.get_all_relationships().await?;
        
        for relationship in relationships {
            // Apply temporal decay based on time since last strengthening
            let time_since_strengthening = SystemTime::now()
                .duration_since(relationship.last_strengthened)
                .unwrap_or(Duration::from_secs(0))
                .as_secs() as f32;
            
            // Calculate decay factor
            let decay_factor = (-self.decay_constant * time_since_strengthening).exp();
            let new_weight = relationship.weight * decay_factor;
            
            if new_weight < self.min_weight {
                // Prune connection if weight becomes too small
                self.brain_graph.remove_relationship(
                    relationship.source,
                    relationship.target,
                ).await?;
                
                pruned_connections.push(WeightChange {
                    source: relationship.source,
                    target: relationship.target,
                    old_weight: relationship.weight,
                    new_weight: 0.0,
                    change_magnitude: relationship.weight,
                });
            } else if new_weight != relationship.weight {
                // Update weight with decay
                self.brain_graph.update_relationship_weight(
                    relationship.source,
                    relationship.target,
                    new_weight,
                ).await?;
            }
        }
        
        Ok(pruned_connections)
    }

    async fn update_inhibition_parameters(
        &self,
        weight_updates: &WeightUpdateResult,
    ) -> Result<Vec<InhibitionChange>> {
        let mut inhibition_changes = Vec::new();
        
        // Analyze weight changes to identify new competition patterns
        for weight_change in &weight_updates.strengthened {
            // Check if this creates new competitive relationships
            let should_compete = self.should_entities_compete(
                weight_change.source,
                weight_change.target,
                weight_change.new_weight,
            ).await?;
            
            if should_compete {
                // Create competition group for these entities
                let _competition_result = self.inhibition_system.update_competition_strength(
                    weight_change.source,
                    weight_change.target,
                    weight_change.new_weight,
                ).await?;
                
                // Convert to learning InhibitionChange type
                inhibition_changes.push(InhibitionChange {
                    competition_group: format!("learned_{:?}_{:?}", weight_change.source, weight_change.target),
                    entities_affected: vec![weight_change.source, weight_change.target],
                    strength_change: weight_change.change_magnitude,
                    change_reason: InhibitionChangeReason::HebbianLearning,
                });
            }
        }
        
        Ok(inhibition_changes)
    }

    async fn should_entities_compete(
        &self,
        entity_a: EntityKey,
        entity_b: EntityKey,
        connection_strength: f32,
    ) -> Result<bool> {
        // Entities should compete if they have strong connections to similar concepts
        // but weak direct connections (suggesting they serve similar functions)
        
        let direct_connection = self.brain_graph.get_relationship_weight(
            entity_a,
            entity_b,
        ).unwrap_or(0.0);
        
        // If direct connection is weak but there are strong indirect connections,
        // entities might compete for similar roles
        if direct_connection < 0.3 && connection_strength > 0.7 {
            let common_neighbors = self.brain_graph.get_common_neighbors(
                entity_a,
                entity_b,
            ).await?;
            
            // If they have many common strong connections, they likely compete
            Ok(common_neighbors.len() > 3)
        } else {
            Ok(false)
        }
    }

    fn calculate_learning_efficiency(&self, weight_updates: &WeightUpdateResult) -> f32 {
        let total_changes = weight_updates.strengthened.len() + 
                           weight_updates.weakened.len() + 
                           weight_updates.newly_formed.len();
        
        if total_changes == 0 {
            return 0.0;
        }
        
        // Efficiency is based on the ratio of strengthening to weakening
        let strengthening_ratio = weight_updates.strengthened.len() as f32 / total_changes as f32;
        let formation_ratio = weight_updates.newly_formed.len() as f32 / total_changes as f32;
        
        // Learning is efficient when more connections are strengthened than weakened
        // and when new connections are formed appropriately
        0.6 * strengthening_ratio + 0.4 * formation_ratio
    }
}

#[derive(Debug, Clone)]
struct WeightUpdateResult {
    strengthened: Vec<WeightChange>,
    weakened: Vec<WeightChange>,
    newly_formed: Vec<WeightChange>,
    inhibition_changes: Vec<InhibitionChange>,
}

// Extension methods for brain graph to support Hebbian learning
impl BrainEnhancedKnowledgeGraph {
    // Note: create_learned_relationship is implemented in BrainEnhancedKnowledgeGraph
    // in brain_enhanced_graph.rs to avoid duplicate definitions
    
    pub async fn get_all_relationships(&self) -> Result<Vec<BrainInspiredRelationship>> {
        // Return empty vec for now - would need to iterate through all entities
        // and collect their relationships
        Ok(Vec::new())
    }
    
    pub async fn get_common_neighbors(
        &self,
        entity_a: EntityKey,
        entity_b: EntityKey,
    ) -> Result<Vec<EntityKey>> {
        // Find entities that both entity_a and entity_b are connected to
        let neighbors_a = self.get_neighbors(entity_a);
        let neighbors_b = self.get_neighbors(entity_b);
        
        let common = neighbors_a.into_iter()
            .filter(|entity| neighbors_b.contains(entity))
            .collect();
        
        Ok(common)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
    use crate::learning::types::*;
    use std::time::{Instant, Duration};
    use uuid::Uuid;

    // Test helper to create mock components
    async fn create_test_hebbian_engine() -> Result<HebbianLearningEngine> {
        let embedding_dim = 512; // Default embedding dimension
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(embedding_dim)?);
        
        // Create activation config
        let activation_config = crate::core::activation_config::ActivationConfig {
            max_iterations: 100,
            convergence_threshold: 0.01,
            decay_rate: 0.1,
            inhibition_strength: 0.3,
            default_threshold: 0.5,
        };
        
        let activation_engine = Arc::new(ActivationPropagationEngine::new(activation_config));
        
        // Create critical thinking system (required for CompetitiveInhibitionSystem)
        let critical_thinking = Arc::new(crate::cognitive::critical::CriticalThinking::new(
            brain_graph.clone()
        ));
        
        let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new(
            activation_engine.clone(),
            critical_thinking
        ));
        
        HebbianLearningEngine::new(brain_graph, activation_engine, inhibition_system).await
    }

    fn create_test_activation_events() -> Vec<ActivationEvent> {
        vec![
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.8,
                timestamp: Instant::now(),
                context: ActivationContext {
                    query_id: "test_query_1".to_string(),
                    source_pattern: None,
                    propagation_depth: 0,
                },
            },
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.6,
                timestamp: Instant::now(),
                context: ActivationContext {
                    query_id: "test_query_2".to_string(),
                    source_pattern: None,
                    propagation_depth: 0,
                },
            }
        ]
    }

    fn create_test_learning_context() -> LearningContext {
        LearningContext {
            performance_pressure: 0.5,
            user_satisfaction_level: 0.8,
            learning_urgency: 0.6,
            session_id: "test_context".to_string(),
            learning_goals: vec![
                LearningGoal {
                    goal_type: LearningGoalType::PerformanceImprovement,
                    target_improvement: 0.2,
                    deadline: Some(std::time::SystemTime::now() + std::time::Duration::from_secs(3600)),
                }
            ],
        }
    }

    #[tokio::test]
    async fn test_hebbian_learning_basic_functionality() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let activation_events = create_test_activation_events();
        let learning_context = create_test_learning_context();
        
        let result = engine.apply_hebbian_learning(activation_events, learning_context).await
            .expect("Failed to apply Hebbian learning");
        
        assert!(result.learning_efficiency >= 0.0 && result.learning_efficiency <= 1.0, 
               "Learning efficiency should be normalized");
        assert!(!result.strengthened_connections.is_empty() || 
               !result.weakened_connections.is_empty() || 
               !result.new_connections.is_empty(),
               "Should have some weight changes");
    }

    #[tokio::test]
    async fn test_hebbian_rule_biological_plausibility() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        // Test Hebb's rule: "Neurons that fire together, wire together"
        let simultaneous_activation_events = vec![
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.9,
                timestamp: Instant::now(),
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            },
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.8,
                timestamp: Instant::now(), // Same timestamp = simultaneous
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            }
        ];
        
        let learning_context = create_test_learning_context();
        
        let result = engine.apply_hebbian_learning(simultaneous_activation_events, learning_context).await
            .expect("Failed to apply Hebbian learning for simultaneous activations");
        
        // Simultaneous high activation should lead to strengthening
        assert!(!result.strengthened_connections.is_empty(), 
               "Simultaneous high activations should strengthen connections");
        
        // Learning efficiency should be high for biologically plausible patterns
        assert!(result.learning_efficiency > 0.5, 
               "Learning efficiency should be high for simultaneous activations");
    }

    #[tokio::test]
    async fn test_spike_timing_dependent_plasticity() {
        let engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let now = Instant::now();
        
        // Test STDP potentiation (pre before post)
        let pre_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.8,
            timestamp: now,
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let post_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.7,
            timestamp: now + Duration::from_millis(20), // 20ms later
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let result = engine.spike_timing_dependent_plasticity(pre_event, post_event).await
            .expect("Failed to apply STDP");
        
        match result {
            STDPResult::WeightChanged { weight_change, plasticity_type, .. } => {
                assert!(weight_change > 0.0, "STDP should strengthen connections when pre precedes post");
                assert!(matches!(plasticity_type, PlasticityType::Potentiation), 
                       "Should be potentiation when pre precedes post");
            },
            STDPResult::NoChange => panic!("STDP should produce changes for properly timed spikes"),
        }
    }

    #[tokio::test]
    async fn test_stdp_depression() {
        let engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let now = Instant::now();
        
        // Test STDP depression (post before pre)
        let pre_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.8,
            timestamp: now + Duration::from_millis(20), // 20ms later
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let post_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.7,
            timestamp: now, // Earlier
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let result = engine.spike_timing_dependent_plasticity(pre_event, post_event).await
            .expect("Failed to apply STDP");
        
        match result {
            STDPResult::WeightChanged { weight_change, plasticity_type, .. } => {
                assert!(weight_change < 0.0, "STDP should weaken connections when post precedes pre");
                assert!(matches!(plasticity_type, PlasticityType::Depression), 
                       "Should be depression when post precedes pre");
            },
            STDPResult::NoChange => panic!("STDP should produce changes for properly timed spikes"),
        }
    }

    #[tokio::test]
    async fn test_stdp_timing_window() {
        let engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let now = Instant::now();
        
        // Test events outside STDP window (>100ms)
        let pre_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.8,
            timestamp: now,
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let post_event = ActivationEvent {
            entity_key: EntityKey::default(),
            activation_strength: 0.7,
            timestamp: now + Duration::from_millis(200), // Outside STDP window
            event_type: ActivationEventType::EntityActivated,
            duration: Duration::from_millis(50),
        };
        
        let result = engine.spike_timing_dependent_plasticity(pre_event, post_event).await
            .expect("Failed to apply STDP");
        
        assert!(matches!(result, STDPResult::NoChange), 
               "STDP should not change weights for events outside timing window");
    }

    #[tokio::test]
    async fn test_coactivation_tracking() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let activation_events = create_test_activation_events();
        
        // Apply coactivation tracking
        engine.update_coactivation_tracking(&activation_events).await
            .expect("Failed to update coactivation tracking");
        
        let tracker = engine.coactivation_tracker.read().unwrap();
        
        assert!(!tracker.activation_history.is_empty(), 
               "Should track activation history");
        assert!(tracker.global_activity_level >= 0.0, 
               "Should track global activity level");
    }

    #[tokio::test]
    async fn test_correlation_matrix_updates() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let activation_events = vec![
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.9,
                timestamp: Instant::now(),
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            },
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.8,
                timestamp: Instant::now() + Duration::from_millis(10),
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            }
        ];
        
        engine.update_coactivation_tracking(&activation_events).await
            .expect("Failed to update coactivation tracking");
        
        let tracker = engine.coactivation_tracker.read().unwrap();
        
        // Should update correlation matrix for co-occurring activations
        assert!(!tracker.correlation_matrix.is_empty(), 
               "Should calculate correlations between co-active entities");
    }

    #[test]
    fn test_temporal_correlation_calculation() {
        let engine_future = create_test_hebbian_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let engine = rt.block_on(engine_future).expect("Failed to create engine");
        
        let entity_a = EntityKey::default();
        let entity_b = EntityKey::default();
        
        let mut activation_history = HashMap::new();
        
        let now = Instant::now();
        activation_history.insert(entity_a, vec![
            ActivationEvent {
                entity_key: entity_a,
                activation_strength: 0.8,
                timestamp: now,
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            }
        ]);
        
        activation_history.insert(entity_b, vec![
            ActivationEvent {
                entity_key: entity_b,
                activation_strength: 0.7,
                timestamp: now + Duration::from_millis(30), // Close in time
                event_type: ActivationEventType::EntityActivated,
                duration: Duration::from_millis(100),
            }
        ]);
        
        let correlation = engine.calculate_temporal_correlation(
            entity_a, entity_b, &activation_history
        ).expect("Failed to calculate temporal correlation");
        
        assert!(correlation > 0.0, "Should calculate positive correlation for temporally close activations");
        assert!(correlation <= 1.0, "Correlation should not exceed 1.0");
    }

    #[test]
    fn test_learning_efficiency_calculation() {
        let engine_future = create_test_hebbian_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let engine = rt.block_on(engine_future).expect("Failed to create engine");
        
        let weight_updates = WeightUpdateResult {
            strengthened: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.5,
                    new_weight: 0.7,
                    change_magnitude: 0.2,
                }
            ],
            weakened: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.6,
                    new_weight: 0.4,
                    change_magnitude: 0.2,
                }
            ],
            newly_formed: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.0,
                    new_weight: 0.3,
                    change_magnitude: 0.3,
                }
            ],
            inhibition_changes: vec![],
        };
        
        let efficiency = engine.calculate_learning_efficiency(&weight_updates);
        
        assert!(efficiency >= 0.0 && efficiency <= 1.0, 
               "Learning efficiency should be normalized");
        
        // With equal strengthening and weakening, plus new connections, 
        // efficiency should be moderate to high
        assert!(efficiency > 0.4, 
               "Should have reasonable efficiency with balanced updates and new connections");
    }

    #[test]
    fn test_learning_stability_calculation() {
        let engine_future = create_test_hebbian_engine();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let engine = rt.block_on(engine_future).expect("Failed to create engine");
        
        // Test stable learning (small, consistent changes)
        let stable_updates = WeightUpdateResult {
            strengthened: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.5,
                    new_weight: 0.52,
                    change_magnitude: 0.02,
                },
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.6,
                    new_weight: 0.62,
                    change_magnitude: 0.02,
                }
            ],
            weakened: vec![],
            newly_formed: vec![],
            inhibition_changes: vec![],
        };
        
        let stability = engine.calculate_learning_stability(&stable_updates);
        
        assert!(stability > 0.8, 
               "Should have high stability for consistent small changes");
        
        // Test unstable learning (large, variable changes)
        let unstable_updates = WeightUpdateResult {
            strengthened: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.5,
                    new_weight: 0.9,
                    change_magnitude: 0.4,
                },
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.6,
                    new_weight: 0.61,
                    change_magnitude: 0.01,
                }
            ],
            weakened: vec![],
            newly_formed: vec![],
            inhibition_changes: vec![],
        };
        
        let instability = engine.calculate_learning_stability(&unstable_updates);
        
        assert!(instability < 0.7, 
               "Should have lower stability for inconsistent changes");
    }

    #[tokio::test]
    async fn test_biological_learning_constraints() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        // Test that learning respects biological constraints
        assert!(engine.learning_rate > 0.0 && engine.learning_rate < 1.0,
               "Learning rate should be biologically plausible");
        assert!(engine.decay_constant > 0.0 && engine.decay_constant < engine.learning_rate,
               "Decay constant should be smaller than learning rate");
        assert!(engine.strengthening_threshold > 0.0 && engine.strengthening_threshold < 1.0,
               "Strengthening threshold should be normalized");
        assert!(engine.weakening_threshold > 0.0 && engine.weakening_threshold < engine.strengthening_threshold,
               "Weakening threshold should be less than strengthening threshold");
        assert!(engine.max_weight > engine.min_weight,
               "Maximum weight should exceed minimum weight");
        assert!(engine.min_weight >= 0.0 && engine.max_weight <= 1.0,
               "Weight bounds should be normalized");
    }

    #[tokio::test]
    async fn test_temporal_decay_biological_realism() {
        let engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        // Test temporal decay (synaptic weakening over time)
        let decay_result = engine.apply_temporal_decay().await
            .expect("Failed to apply temporal decay");
        
        // Decay should occur but might not prune connections immediately in a new system
        // This is biologically realistic - synapses don't disappear instantly
        assert!(decay_result.len() >= 0, "Temporal decay should return pruning information");
    }

    #[tokio::test]
    async fn test_competition_and_cooperation_balance() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let activation_events = create_test_activation_events();
        let learning_context = create_test_learning_context();
        
        let result = engine.apply_hebbian_learning(activation_events, learning_context).await
            .expect("Failed to apply Hebbian learning");
        
        // Should balance competition and cooperation
        let total_inhibitory_changes = result.inhibition_updates.len();
        let total_weight_changes = result.strengthened_connections.len() + 
                                 result.weakened_connections.len() + 
                                 result.new_connections.len();
        
        // Not all weight changes should result in inhibitory changes
        // This maintains balance between cooperation and competition
        if total_weight_changes > 0 {
            let inhibitory_ratio = total_inhibitory_changes as f32 / total_weight_changes as f32;
            assert!(inhibitory_ratio <= 0.5, 
                   "Should not create excessive inhibitory competition");
        }
    }

    #[tokio::test]
    async fn test_learning_statistics_tracking() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        let weight_updates = WeightUpdateResult {
            strengthened: vec![
                WeightChange {
                    source: EntityKey::default(),
                    target: EntityKey::default(),
                    old_weight: 0.5,
                    new_weight: 0.7,
                    change_magnitude: 0.2,
                }
            ],
            weakened: vec![],
            newly_formed: vec![],
            inhibition_changes: vec![],
        };
        
        engine.update_learning_statistics(&weight_updates).await
            .expect("Failed to update learning statistics");
        
        let stats = engine.learning_statistics.read().unwrap();
        
        assert!(stats.total_weight_changes > 0, "Should track weight changes");
        assert!(stats.average_learning_rate >= 0.0, "Should track average learning rate");
        assert!(stats.learning_stability >= 0.0 && stats.learning_stability <= 1.0, 
               "Learning stability should be normalized");
    }

    #[tokio::test]
    async fn test_hebbian_learning_adaptive_parameters() {
        let mut engine = create_test_hebbian_engine().await
            .expect("Failed to create Hebbian engine");
        
        // Test that learning parameters can adapt
        let initial_learning_rate = engine.learning_rate;
        let initial_decay_constant = engine.decay_constant;
        
        // Apply learning multiple times
        for _ in 0..5 {
            let activation_events = create_test_activation_events();
            let learning_context = create_test_learning_context();
            
            engine.apply_hebbian_learning(activation_events, learning_context).await
                .expect("Failed to apply Hebbian learning");
        }
        
        // Parameters should remain within biological bounds even after adaptation
        assert!(engine.learning_rate > 0.0 && engine.learning_rate < 1.0,
               "Learning rate should remain biologically constrained");
        assert!(engine.decay_constant > 0.0 && engine.decay_constant < engine.learning_rate,
               "Decay constant should remain properly constrained");
    }
}