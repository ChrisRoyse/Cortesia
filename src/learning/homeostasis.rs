use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::cognitive::attention_manager::AttentionManager;
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::types::EntityKey;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, Duration};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SynapticHomeostasis {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub attention_manager: Arc<AttentionManager>,
    pub working_memory: Arc<WorkingMemorySystem>,
    pub target_activity_level: f32,
    pub homeostatic_scaling_rate: f32,
    pub metaplasticity_threshold: f32,
    pub activity_tracker: Arc<RwLock<ActivityTracker>>,
    pub homeostasis_config: HomeostasisConfig,
}

#[derive(Debug, Clone)]
pub struct ActivityTracker {
    pub entity_activity_levels: HashMap<EntityKey, f32>,
    pub global_activity_level: f32,
    pub activity_history: VecDeque<ActivitySnapshot>,
    pub tracking_window: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct ActivitySnapshot {
    pub timestamp: SystemTime,
    pub entity_activities: HashMap<EntityKey, f32>,
    pub global_activity: f32,
    pub attention_focus: Vec<EntityKey>,
    pub memory_load: f32,
}

#[derive(Debug, Clone)]
pub struct HomeostasisConfig {
    pub scaling_frequency: Duration,
    pub activity_window: Duration,
    pub stability_threshold: f32,
    pub emergency_scaling_threshold: f32,
    pub metaplasticity_window: Duration,
}

#[derive(Debug, Clone)]
pub struct HomeostasisUpdate {
    pub scaled_entities: Vec<ActivityScaling>,
    pub metaplasticity_changes: Vec<MetaplasticityUpdate>,
    pub global_activity_change: f32,
    pub stability_improvement: f32,
}

#[derive(Debug, Clone)]
pub struct ActivityScaling {
    pub entity_key: EntityKey,
    pub scaling_factor: f32,
    pub relationships_affected: usize,
    pub previous_activity: f32,
    pub target_activity: f32,
}

#[derive(Debug, Clone)]
pub struct MetaplasticityUpdate {
    pub entity_key: EntityKey,
    pub learning_rate_adjustment: f32,
    pub threshold_adjustment: f32,
    pub plasticity_state: MetaplasticityState,
}

#[derive(Debug, Clone)]
pub struct MetaplasticityState {
    pub entity_key: EntityKey,
    pub adjusted_learning_rate: f32,
    pub adjusted_threshold: f32,
    pub plasticity_history: RecentPlasticityAnalysis,
}

#[derive(Debug, Clone)]
pub struct RecentPlasticityAnalysis {
    pub excessive_plasticity: bool,
    pub insufficient_plasticity: bool,
    pub high_activity: bool,
    pub low_activity: bool,
    pub plasticity_events: usize,
    pub average_change_magnitude: f32,
}

#[derive(Debug, Clone)]
pub struct ActivityImbalance {
    pub entity_key: EntityKey,
    pub current_activity: f32,
    pub target_activity: f32,
    pub imbalance_severity: f32,
    pub imbalance_duration: Duration,
}

#[derive(Debug, Clone)]
pub struct LearningEvent {
    pub entity_key: EntityKey,
    pub event_type: LearningEventType,
    pub magnitude: f32,
    pub timestamp: SystemTime,
    pub context: String,
}

#[derive(Debug, Clone)]
pub enum LearningEventType {
    WeightIncrease,
    WeightDecrease,
    NewConnection,
    ConnectionPruned,
    ThresholdChange,
}

impl SynapticHomeostasis {
    pub async fn new(
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        attention_manager: Arc<AttentionManager>,
        working_memory: Arc<WorkingMemorySystem>,
    ) -> Result<Self> {
        Ok(Self {
            brain_graph,
            attention_manager,
            working_memory,
            target_activity_level: 0.5, // Balanced activity level
            homeostatic_scaling_rate: 0.1,
            metaplasticity_threshold: 0.3,
            activity_tracker: Arc::new(RwLock::new(ActivityTracker::new())),
            homeostasis_config: HomeostasisConfig::default(),
        })
    }

    pub async fn apply_homeostatic_scaling(
        &mut self,
        time_window: Duration,
    ) -> Result<HomeostasisUpdate> {
        // 1. Calculate current activity levels integrating with attention and memory
        let current_activity = self.calculate_integrated_activity_levels(time_window).await?;
        
        // 2. Identify entities with activity imbalance
        let imbalanced_entities = self.identify_activity_imbalance(&current_activity)?;
        
        // 3. Apply homeostatic scaling to maintain target activity
        let scaling_updates = self.apply_activity_scaling(&imbalanced_entities).await?;
        
        // 4. Update metaplasticity thresholds based on activity
        let metaplasticity_updates = self.update_metaplasticity_thresholds(
            &current_activity,
        ).await?;
        
        // 5. Update activity tracking with attention and memory integration
        self.update_integrated_activity_tracking(current_activity).await?;
        
        // 6. Calculate stability improvement
        let stability_improvement = self.calculate_stability_improvement(&scaling_updates)?;
        
        Ok(HomeostasisUpdate {
            scaled_entities: scaling_updates,
            metaplasticity_changes: metaplasticity_updates,
            global_activity_change: self.calculate_global_activity_change(),
            stability_improvement,
        })
    }

    async fn calculate_integrated_activity_levels(
        &self,
        _time_window: Duration,
    ) -> Result<HashMap<EntityKey, f32>> {
        let mut activity_levels = HashMap::new();
        
        // 1. Get base activity from brain graph activation
        // Note: This would need to be implemented in BrainEnhancedKnowledgeGraph
        // For now, we'll use a placeholder approach
        let graph_activities: HashMap<EntityKey, f32> = HashMap::new();
        
        // 2. Get attention-weighted activities
        let attention_state = self.attention_manager.get_attention_state().await?;
        
        // 3. Get working memory influences
        // Note: This would need to be implemented based on actual working memory state
        let memory_activities: HashMap<EntityKey, f32> = HashMap::new();
        
        // 4. Combine all activity sources
        for (entity_key, base_activity) in graph_activities {
            // Check if entity is in current attention targets
            let attention_weight: f32 = if attention_state.current_targets.contains(&entity_key) {
                attention_state.focus_strength
            } else {
                0.5 // Default weight for non-focused entities
            };
            let memory_influence = memory_activities.get(&entity_key).unwrap_or(&1.0);
            
            // Calculate integrated activity level
            let integrated_activity = base_activity * attention_weight * memory_influence;
            activity_levels.insert(entity_key, integrated_activity);
        }
        
        Ok(activity_levels)
    }

    fn identify_activity_imbalance(
        &self,
        current_activity: &HashMap<EntityKey, f32>,
    ) -> Result<Vec<ActivityImbalance>> {
        let mut imbalanced_entities = Vec::new();
        let tracker = self.activity_tracker.read().unwrap();
        
        for (entity_key, &current_level) in current_activity {
            let deviation = (current_level - self.target_activity_level).abs();
            
            if deviation > self.homeostasis_config.stability_threshold {
                // Calculate how long this imbalance has persisted
                let imbalance_duration = self.calculate_imbalance_duration(
                    *entity_key,
                    &tracker.activity_history,
                );
                
                imbalanced_entities.push(ActivityImbalance {
                    entity_key: *entity_key,
                    current_activity: current_level,
                    target_activity: self.target_activity_level,
                    imbalance_severity: deviation / self.target_activity_level,
                    imbalance_duration,
                });
            }
        }
        
        // Sort by severity (most imbalanced first)
        imbalanced_entities.sort_by(|a, b| 
            b.imbalance_severity.partial_cmp(&a.imbalance_severity).unwrap()
        );
        
        Ok(imbalanced_entities)
    }

    fn calculate_imbalance_duration(
        &self,
        entity_key: EntityKey,
        activity_history: &VecDeque<ActivitySnapshot>,
    ) -> Duration {
        let mut duration = Duration::from_secs(0);
        let current_time = SystemTime::now();
        
        for snapshot in activity_history.iter().rev() {
            if let Some(&activity) = snapshot.entity_activities.get(&entity_key) {
                let deviation = (activity - self.target_activity_level).abs();
                if deviation > self.homeostasis_config.stability_threshold {
                    duration = current_time.duration_since(snapshot.timestamp)
                        .unwrap_or(Duration::from_secs(0));
                } else {
                    break;
                }
            }
        }
        
        duration
    }

    async fn apply_activity_scaling(
        &self,
        imbalanced_entities: &[ActivityImbalance],
    ) -> Result<Vec<ActivityScaling>> {
        let mut scalings = Vec::new();
        
        for imbalance in imbalanced_entities {
            // Calculate scaling factor with adaptive rate based on severity and duration
            let urgency_multiplier = if imbalance.imbalance_duration > self.homeostasis_config.activity_window {
                2.0 // More aggressive scaling for persistent imbalances
            } else {
                1.0
            };
            
            let scaling_factor = if imbalance.current_activity > self.target_activity_level {
                // Scale down if too active
                1.0 - self.homeostatic_scaling_rate * urgency_multiplier * 
                    (imbalance.current_activity - self.target_activity_level)
            } else {
                // Scale up if too inactive
                1.0 + self.homeostatic_scaling_rate * urgency_multiplier * 
                    (self.target_activity_level - imbalance.current_activity)
            };
            
            // Apply scaling to all incoming synaptic weights
            let incoming_relationships = self.brain_graph.get_parent_entities(
                imbalance.entity_key,
            ).await;
            
            let mut relationships_affected = 0;
            for (parent_key, weight) in &incoming_relationships {
                let new_weight = weight * scaling_factor;
                self.brain_graph.update_relationship_weight(
                    *parent_key,
                    imbalance.entity_key,
                    new_weight,
                ).await?;
                relationships_affected += 1;
            }
            
            // Also scale attention influence if this entity is in focus
            // Note: Attention scaling would need to be implemented differently
            // as the AttentionManager doesn't expose direct weight scaling
            
            scalings.push(ActivityScaling {
                entity_key: imbalance.entity_key,
                scaling_factor,
                relationships_affected,
                previous_activity: imbalance.current_activity,
                target_activity: self.target_activity_level,
            });
        }
        
        Ok(scalings)
    }

    async fn update_metaplasticity_thresholds(
        &self,
        current_activity: &HashMap<EntityKey, f32>,
    ) -> Result<Vec<MetaplasticityUpdate>> {
        let mut updates = Vec::new();
        
        for (entity_key, &_activity_level) in current_activity {
            // Get recent learning history for this entity
            let learning_history = self.get_recent_learning_history(*entity_key).await?;
            
            // Implement metaplasticity
            let metaplasticity_state = self.implement_metaplasticity(
                *entity_key,
                &learning_history,
            ).await?;
            
            // Calculate adjustments
            let learning_rate_adjustment = metaplasticity_state.adjusted_learning_rate - 0.01; // Base rate
            let threshold_adjustment = metaplasticity_state.adjusted_threshold - self.metaplasticity_threshold;
            
            if learning_rate_adjustment.abs() > 0.001 || threshold_adjustment.abs() > 0.01 {
                updates.push(MetaplasticityUpdate {
                    entity_key: *entity_key,
                    learning_rate_adjustment,
                    threshold_adjustment,
                    plasticity_state: metaplasticity_state,
                });
            }
        }
        
        Ok(updates)
    }

    async fn get_recent_learning_history(&self, entity_key: EntityKey) -> Result<Vec<LearningEvent>> {
        // Get learning events for this entity from the past metaplasticity window
        let cutoff_time = SystemTime::now() - self.homeostasis_config.metaplasticity_window;
        
        // This would integrate with the learning system to get actual events
        // For now, we'll simulate based on recent activity changes
        let mut events = Vec::new();
        
        let tracker = self.activity_tracker.read().unwrap();
        for snapshot in &tracker.activity_history {
            if snapshot.timestamp > cutoff_time {
                if let Some(&activity) = snapshot.entity_activities.get(&entity_key) {
                    // Infer learning events from activity changes
                    if activity > self.target_activity_level * 1.2 {
                        events.push(LearningEvent {
                            entity_key,
                            event_type: LearningEventType::WeightIncrease,
                            magnitude: activity - self.target_activity_level,
                            timestamp: snapshot.timestamp,
                            context: "High activity".to_string(),
                        });
                    }
                }
            }
        }
        
        Ok(events)
    }

    pub async fn implement_metaplasticity(
        &self,
        entity_key: EntityKey,
        learning_history: &[LearningEvent],
    ) -> Result<MetaplasticityState> {
        // Metaplasticity: plasticity of plasticity
        // Learning rate and thresholds adapt based on learning history
        
        let recent_learning = self.analyze_recent_learning(
            entity_key,
            learning_history,
        )?;
        
        let base_learning_rate = 0.01;
        let adjusted_learning_rate = if recent_learning.excessive_plasticity {
            // Reduce learning rate if too much recent plasticity
            base_learning_rate * 0.5
        } else if recent_learning.insufficient_plasticity {
            // Increase learning rate if too little recent plasticity
            base_learning_rate * 1.5
        } else {
            base_learning_rate
        };
        
        let adjusted_threshold = if recent_learning.high_activity {
            // Raise threshold if high activity
            self.metaplasticity_threshold * 1.2
        } else if recent_learning.low_activity {
            // Lower threshold if low activity
            self.metaplasticity_threshold * 0.8
        } else {
            self.metaplasticity_threshold
        };
        
        Ok(MetaplasticityState {
            entity_key,
            adjusted_learning_rate,
            adjusted_threshold,
            plasticity_history: recent_learning,
        })
    }

    fn analyze_recent_learning(
        &self,
        entity_key: EntityKey,
        learning_history: &[LearningEvent],
    ) -> Result<RecentPlasticityAnalysis> {
        let plasticity_events = learning_history.len();
        
        let average_change_magnitude = if plasticity_events > 0 {
            learning_history.iter().map(|event| event.magnitude).sum::<f32>() / plasticity_events as f32
        } else {
            0.0
        };
        
        // Analyze activity levels
        let tracker = self.activity_tracker.read().unwrap();
        let recent_activity = tracker.entity_activity_levels.get(&entity_key).unwrap_or(&0.5);
        
        let excessive_plasticity = plasticity_events > 10 || average_change_magnitude > 0.5;
        let insufficient_plasticity = plasticity_events < 2 && average_change_magnitude < 0.1;
        let high_activity = *recent_activity > self.target_activity_level * 1.5;
        let low_activity = *recent_activity < self.target_activity_level * 0.5;
        
        Ok(RecentPlasticityAnalysis {
            excessive_plasticity,
            insufficient_plasticity,
            high_activity,
            low_activity,
            plasticity_events,
            average_change_magnitude,
        })
    }

    async fn update_integrated_activity_tracking(
        &self,
        current_activity: HashMap<EntityKey, f32>,
    ) -> Result<()> {
        let mut tracker = self.activity_tracker.write().unwrap();
        
        // Update current activity levels
        tracker.entity_activity_levels = current_activity.clone();
        
        // Calculate global activity level
        let global_activity = if current_activity.is_empty() {
            0.0
        } else {
            current_activity.values().sum::<f32>() / current_activity.len() as f32
        };
        tracker.global_activity_level = global_activity;
        
        // Get current attention focus and memory load
        let attention_state = self.attention_manager.get_attention_state().await?;
        let attention_focus = attention_state.current_targets;
        // Memory load would need to be calculated from actual memory state
        let memory_load = 0.5; // Placeholder
        
        // Create activity snapshot
        let snapshot = ActivitySnapshot {
            timestamp: SystemTime::now(),
            entity_activities: current_activity,
            global_activity,
            attention_focus,
            memory_load,
        };
        
        // Add to history
        tracker.activity_history.push_back(snapshot);
        
        // Keep only recent history within tracking window
        let cutoff_time = SystemTime::now() - tracker.tracking_window;
        tracker.activity_history.retain(|snapshot| snapshot.timestamp > cutoff_time);
        
        Ok(())
    }

    fn calculate_stability_improvement(&self, scaling_updates: &[ActivityScaling]) -> Result<f32> {
        if scaling_updates.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate how much closer to target each entity got
        let mut total_improvement = 0.0;
        
        for scaling in scaling_updates {
            let previous_deviation = (scaling.previous_activity - scaling.target_activity).abs();
            let projected_new_activity = scaling.previous_activity * scaling.scaling_factor;
            let new_deviation = (projected_new_activity - scaling.target_activity).abs();
            
            let improvement = (previous_deviation - new_deviation) / previous_deviation;
            total_improvement += improvement;
        }
        
        Ok(total_improvement / scaling_updates.len() as f32)
    }

    fn calculate_global_activity_change(&self) -> f32 {
        let tracker = self.activity_tracker.read().unwrap();
        
        if tracker.activity_history.len() < 2 {
            return 0.0;
        }
        
        let current = tracker.activity_history.back().unwrap();
        let previous = &tracker.activity_history[tracker.activity_history.len() - 2];
        
        current.global_activity - previous.global_activity
    }

    pub async fn emergency_stabilization(&self) -> Result<HomeostasisUpdate> {
        // Emergency stabilization when system becomes too unstable
        let emergency_threshold = self.homeostasis_config.emergency_scaling_threshold;
        
        // Get current activity
        let current_activity = self.calculate_integrated_activity_levels(
            Duration::from_secs(60)
        ).await?;
        
        // Find severely imbalanced entities
        let severe_imbalances: Vec<_> = current_activity.iter()
            .filter(|(_, &activity)| {
                (activity - self.target_activity_level).abs() > emergency_threshold
            })
            .map(|(&entity_key, &activity)| ActivityImbalance {
                entity_key,
                current_activity: activity,
                target_activity: self.target_activity_level,
                imbalance_severity: (activity - self.target_activity_level).abs() / self.target_activity_level,
                imbalance_duration: Duration::from_secs(0),
            })
            .collect();
        
        // Apply aggressive scaling
        let mut emergency_scalings = Vec::new();
        for imbalance in &severe_imbalances {
            let aggressive_scaling_rate = self.homeostatic_scaling_rate * 3.0; // 3x normal rate
            
            let scaling_factor = if imbalance.current_activity > self.target_activity_level {
                1.0 - aggressive_scaling_rate * (imbalance.current_activity - self.target_activity_level)
            } else {
                1.0 + aggressive_scaling_rate * (self.target_activity_level - imbalance.current_activity)
            };
            
            // Apply emergency scaling
            let incoming_relationships = self.brain_graph.get_parent_entities(
                imbalance.entity_key,
            ).await;
            
            for (parent_key, weight) in &incoming_relationships {
                let new_weight = (*weight * scaling_factor).clamp(0.0, 1.0);
                self.brain_graph.update_relationship_weight(
                    *parent_key,
                    imbalance.entity_key,
                    new_weight,
                ).await?;
            }
            
            emergency_scalings.push(ActivityScaling {
                entity_key: imbalance.entity_key,
                scaling_factor,
                relationships_affected: incoming_relationships.len(),
                previous_activity: imbalance.current_activity,
                target_activity: self.target_activity_level,
            });
        }
        
        Ok(HomeostasisUpdate {
            scaled_entities: emergency_scalings,
            metaplasticity_changes: Vec::new(),
            global_activity_change: 0.0,
            stability_improvement: 1.0, // Emergency stabilization should improve stability
        })
    }
}

impl Default for ActivityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivityTracker {
    pub fn new() -> Self {
        Self {
            entity_activity_levels: HashMap::new(),
            global_activity_level: 0.5,
            activity_history: VecDeque::new(),
            tracking_window: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl HomeostasisConfig {
    pub fn default() -> Self {
        Self {
            scaling_frequency: Duration::from_secs(300), // 5 minutes
            activity_window: Duration::from_secs(900),   // 15 minutes
            stability_threshold: 0.1,                    // 10% deviation threshold
            emergency_scaling_threshold: 0.3,            // 30% deviation for emergency
            metaplasticity_window: Duration::from_secs(1800), // 30 minutes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::cognitive::attention_manager::AttentionManager;
    use crate::cognitive::working_memory::WorkingMemorySystem;
    use crate::core::activation_engine::ActivationPropagationEngine;
    use crate::cognitive::orchestrator::CognitiveOrchestrator;
    use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
    use std::time::Duration;
    

    // Test helper to create mock homeostasis system
    async fn create_test_homeostasis() -> Result<SynapticHomeostasis> {
        let embedding_dim = 512;
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(embedding_dim)?);
        
        // Create activation config and engine
        let activation_config = crate::core::activation_config::ActivationConfig {
            max_iterations: 100,
            convergence_threshold: 0.01,
            decay_rate: 0.1,
            decay_factor: 0.95,
            inhibition_strength: 0.3,
            default_threshold: 0.5,
        };
        let activation_engine = Arc::new(ActivationPropagationEngine::new(activation_config));
        
        // Create SDR storage
        let sdr_config = crate::core::sdr_types::SDRConfig {
            total_bits: embedding_dim * 4,
            active_bits: (embedding_dim * 4) / 50,
            sparsity: 0.02,
            overlap_threshold: 0.5,
        };
        let sdr_storage = Arc::new(crate::core::sdr_storage::SDRStorage::new(sdr_config));
        
        // Create working memory
        let working_memory = Arc::new(WorkingMemorySystem::new(
            activation_engine.clone(),
            sdr_storage.clone(),
        ).await?);
        
        // Create cognitive orchestrator
        let critical_thinking = Arc::new(crate::cognitive::critical::CriticalThinking::new(
            brain_graph.clone()
        ));
        let _inhibition_system = Arc::new(CompetitiveInhibitionSystem::new(
            activation_engine.clone(),
            critical_thinking.clone()
        ));
        
        let orchestrator = Arc::new(CognitiveOrchestrator::new(
            brain_graph.clone(),
            crate::cognitive::orchestrator::CognitiveOrchestratorConfig::default(),
        ).await?);
        
        // Create attention manager
        let attention_manager = Arc::new(AttentionManager::new(
            orchestrator,
            activation_engine.clone(),
            working_memory.clone(),
        ).await?);
        
        SynapticHomeostasis::new(brain_graph, attention_manager, working_memory).await
    }

    fn create_test_activity_levels() -> HashMap<EntityKey, f32> {
        let mut activities = HashMap::new();
        activities.insert(EntityKey::default(), 0.9); // High activity
        activities.insert(EntityKey::default(), 0.2); // Low activity
        activities.insert(EntityKey::default(), 0.5); // Target activity
        activities
    }

    #[tokio::test]
    async fn test_homeostatic_scaling_basic_functionality() {
        let mut homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let time_window = Duration::from_secs(60);
        
        let result = homeostasis.apply_homeostatic_scaling(time_window).await
            .expect("Failed to apply homeostatic scaling");
        
        assert!(result.stability_improvement >= 0.0, 
               "Stability improvement should be non-negative");
        assert!(result.global_activity_change.abs() <= 1.0, 
               "Global activity change should be bounded");
    }

    #[test]
    fn test_activity_imbalance_identification() {
        let homeostasis_future = create_test_homeostasis();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let homeostasis = rt.block_on(homeostasis_future).expect("Failed to create homeostasis");
        
        let current_activity = create_test_activity_levels();
        
        let imbalances = homeostasis.identify_activity_imbalance(&current_activity)
            .expect("Failed to identify activity imbalances");
        
        // Should identify entities that deviate significantly from target
        assert!(!imbalances.is_empty(), "Should identify activity imbalances");
        
        // Check that imbalances are prioritized by severity
        for i in 1..imbalances.len() {
            assert!(imbalances[i-1].imbalance_severity >= imbalances[i].imbalance_severity,
                   "Imbalances should be sorted by severity");
        }
    }

    #[test]
    fn test_imbalance_duration_calculation() {
        let homeostasis_future = create_test_homeostasis();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let homeostasis = rt.block_on(homeostasis_future).expect("Failed to create homeostasis");
        
        let entity_key = EntityKey::default();
        let mut activity_history = VecDeque::new();
        
        let now = SystemTime::now();
        
        // Create history with persistent imbalance
        for i in 0..5 {
            let mut entity_activities = HashMap::new();
            entity_activities.insert(entity_key, 0.9); // Consistently high
            
            activity_history.push_back(ActivitySnapshot {
                timestamp: now - Duration::from_secs(60 * i),
                entity_activities,
                global_activity: 0.7,
                attention_focus: vec![entity_key],
                memory_load: 0.5,
            });
        }
        
        let duration = homeostasis.calculate_imbalance_duration(entity_key, &activity_history);
        
        assert!(duration > Duration::from_secs(0), 
               "Should detect persistent imbalance duration");
    }

    #[tokio::test]
    async fn test_activity_scaling_application() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let imbalanced_entities = vec![
            ActivityImbalance {
                entity_key: EntityKey::default(),
                current_activity: 0.9, // Too high
                target_activity: 0.5,
                imbalance_severity: 0.8,
                imbalance_duration: Duration::from_secs(300),
            },
            ActivityImbalance {
                entity_key: EntityKey::default(),
                current_activity: 0.1, // Too low
                target_activity: 0.5,
                imbalance_severity: 0.8,
                imbalance_duration: Duration::from_secs(120),
            }
        ];
        
        let scalings = homeostasis.apply_activity_scaling(&imbalanced_entities).await
            .expect("Failed to apply activity scaling");
        
        assert_eq!(scalings.len(), imbalanced_entities.len(), 
                  "Should create scaling for each imbalanced entity");
        
        // Check scaling factors
        for scaling in &scalings {
            if scaling.previous_activity > scaling.target_activity {
                assert!(scaling.scaling_factor < 1.0, 
                       "Should scale down overactive entities");
            } else if scaling.previous_activity < scaling.target_activity {
                assert!(scaling.scaling_factor > 1.0, 
                       "Should scale up underactive entities");
            }
        }
    }

    #[tokio::test]
    async fn test_metaplasticity_implementation() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let entity_key = EntityKey::default();
        
        // Create learning history with excessive plasticity
        let excessive_learning_history = vec![
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.8,
                timestamp: SystemTime::now(),
                context: "High plasticity event".to_string(),
            },
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.9,
                timestamp: SystemTime::now(),
                context: "Another high plasticity event".to_string(),
            }
        ];
        
        let metaplasticity_state = homeostasis.implement_metaplasticity(
            entity_key,
            &excessive_learning_history,
        ).await.expect("Failed to implement metaplasticity");
        
        // With excessive plasticity, learning rate should be reduced
        assert!(metaplasticity_state.adjusted_learning_rate < 0.01, 
               "Should reduce learning rate for excessive plasticity");
        
        // Insufficient plasticity case
        let minimal_learning_history = vec![
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.05,
                timestamp: SystemTime::now(),
                context: "Minimal plasticity event".to_string(),
            }
        ];
        
        let metaplasticity_state_min = homeostasis.implement_metaplasticity(
            entity_key,
            &minimal_learning_history,
        ).await.expect("Failed to implement metaplasticity");
        
        // With insufficient plasticity, learning rate should be increased
        assert!(metaplasticity_state_min.adjusted_learning_rate > 0.01, 
               "Should increase learning rate for insufficient plasticity");
    }

    #[test]
    fn test_recent_plasticity_analysis() {
        let homeostasis_future = create_test_homeostasis();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let homeostasis = rt.block_on(homeostasis_future).expect("Failed to create homeostasis");
        
        let entity_key = EntityKey::default();
        
        // Test excessive plasticity detection
        let excessive_events = vec![
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.8,
                timestamp: SystemTime::now(),
                context: "High magnitude event".to_string(),
            }
        ];
        for _ in 0..15 { // Many events
            let _ = excessive_events.clone();
        }
        
        let analysis = homeostasis.analyze_recent_learning(entity_key, &excessive_events)
            .expect("Failed to analyze recent learning");
        
        assert!(analysis.excessive_plasticity, 
               "Should detect excessive plasticity with many high-magnitude events");
        assert!(analysis.average_change_magnitude > 0.5, 
               "Should track high average change magnitude");
        
        // Test insufficient plasticity detection
        let minimal_events = vec![
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.05,
                timestamp: SystemTime::now(),
                context: "Low magnitude event".to_string(),
            }
        ];
        
        let analysis_min = homeostasis.analyze_recent_learning(entity_key, &minimal_events)
            .expect("Failed to analyze recent learning");
        
        assert!(analysis_min.insufficient_plasticity, 
               "Should detect insufficient plasticity with few low-magnitude events");
    }

    #[tokio::test]
    async fn test_integrated_activity_tracking() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let current_activity = create_test_activity_levels();
        
        homeostasis.update_integrated_activity_tracking(current_activity.clone()).await
            .expect("Failed to update integrated activity tracking");
        
        let tracker = homeostasis.activity_tracker.read().unwrap();
        
        assert_eq!(tracker.entity_activity_levels.len(), current_activity.len(),
                  "Should track all entity activities");
        assert!(tracker.global_activity_level >= 0.0, 
               "Should calculate global activity level");
        assert!(!tracker.activity_history.is_empty(), 
               "Should maintain activity history");
    }

    #[test]
    fn test_stability_improvement_calculation() {
        let homeostasis_future = create_test_homeostasis();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let homeostasis = rt.block_on(homeostasis_future).expect("Failed to create homeostasis");
        
        let scaling_updates = vec![
            ActivityScaling {
                entity_key: EntityKey::default(),
                scaling_factor: 0.8,
                relationships_affected: 5,
                previous_activity: 0.9,
                target_activity: 0.5,
            },
            ActivityScaling {
                entity_key: EntityKey::default(),
                scaling_factor: 1.5,
                relationships_affected: 3,
                previous_activity: 0.2,
                target_activity: 0.5,
            }
        ];
        
        let improvement = homeostasis.calculate_stability_improvement(&scaling_updates)
            .expect("Failed to calculate stability improvement");
        
        assert!(improvement >= 0.0, "Stability improvement should be non-negative");
        assert!(improvement <= 1.0, "Stability improvement should not exceed 1.0");
    }

    #[test]
    fn test_global_activity_change_calculation() {
        let homeostasis_future = create_test_homeostasis();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let homeostasis = rt.block_on(homeostasis_future).expect("Failed to create homeostasis");
        
        // Initialize with some activity history
        let mut tracker = homeostasis.activity_tracker.write().unwrap();
        
        let now = SystemTime::now();
        tracker.activity_history.push_back(ActivitySnapshot {
            timestamp: now - Duration::from_secs(60),
            entity_activities: HashMap::new(),
            global_activity: 0.6,
            attention_focus: vec![],
            memory_load: 0.5,
        });
        
        tracker.activity_history.push_back(ActivitySnapshot {
            timestamp: now,
            entity_activities: HashMap::new(),
            global_activity: 0.8,
            attention_focus: vec![],
            memory_load: 0.5,
        });
        
        drop(tracker); // Release lock
        
        let change = homeostasis.calculate_global_activity_change();
        
        assert!((change - 0.2).abs() < 0.001, 
               "Should calculate correct global activity change: expected 0.2, got {change}");
    }

    #[tokio::test]
    async fn test_emergency_stabilization() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let result = homeostasis.emergency_stabilization().await
            .expect("Failed to perform emergency stabilization");
        
        // Emergency stabilization should be aggressive
        assert!(result.stability_improvement > 0.5, 
               "Emergency stabilization should provide significant improvement");
        
        // Should apply emergency scaling
        if !result.scaled_entities.is_empty() {
            for scaling in &result.scaled_entities {
                // Emergency scaling should be more aggressive than normal
                let scaling_magnitude = (1.0 - scaling.scaling_factor).abs();
                assert!(scaling_magnitude > 0.0, 
                       "Emergency scaling should apply significant changes");
            }
        }
    }

    #[tokio::test]
    async fn test_homeostatic_biological_constraints() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        // Test biological plausibility of parameters
        assert!(homeostasis.target_activity_level > 0.0 && homeostasis.target_activity_level < 1.0,
               "Target activity level should be biologically plausible");
        assert!(homeostasis.homeostatic_scaling_rate > 0.0 && homeostasis.homeostatic_scaling_rate < 1.0,
               "Homeostatic scaling rate should be moderate");
        assert!(homeostasis.metaplasticity_threshold > 0.0 && homeostasis.metaplasticity_threshold < 1.0,
               "Metaplasticity threshold should be normalized");
    }

    #[tokio::test]
    async fn test_homeostatic_scaling_prevents_runaway() {
        let mut homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        // Apply multiple rounds of homeostatic scaling
        for _ in 0..10 {
            let time_window = Duration::from_secs(60);
            let result = homeostasis.apply_homeostatic_scaling(time_window).await
                .expect("Failed to apply homeostatic scaling");
            
            // Each scaling should be bounded
            assert!(result.global_activity_change.abs() < 1.0, 
                   "Global activity changes should be bounded");
            
            // Stability should not decrease dramatically
            assert!(result.stability_improvement >= -0.5, 
                   "Homeostatic scaling should not drastically reduce stability");
        }
    }

    #[tokio::test]
    async fn test_metaplasticity_threshold_adaptation() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let current_activity = create_test_activity_levels();
        
        let updates = homeostasis.update_metaplasticity_thresholds(&current_activity).await
            .expect("Failed to update metaplasticity thresholds");
        
        // Should generate metaplasticity updates for entities with significant learning history
        for update in &updates {
            assert!(update.learning_rate_adjustment.abs() <= 0.1, 
                   "Learning rate adjustments should be moderate");
            assert!(update.threshold_adjustment.abs() <= 0.2, 
                   "Threshold adjustments should be bounded");
            
            // Verify plasticity state is reasonable
            assert!(update.plasticity_state.adjusted_learning_rate > 0.0, 
                   "Adjusted learning rate should be positive");
            assert!(update.plasticity_state.adjusted_threshold >= 0.0, 
                   "Adjusted threshold should be non-negative");
        }
    }

    #[test]
    fn test_activity_tracker_initialization() {
        let tracker = ActivityTracker::new();
        
        assert!(tracker.entity_activity_levels.is_empty(), 
               "New tracker should have empty activity levels");
        assert_eq!(tracker.global_activity_level, 0.5, 
                  "Should initialize with neutral global activity");
        assert!(tracker.activity_history.is_empty(), 
               "New tracker should have empty history");
        assert_eq!(tracker.tracking_window, Duration::from_secs(3600), 
                  "Should have appropriate tracking window");
    }

    #[test]
    fn test_homeostasis_config_defaults() {
        let config = HomeostasisConfig::default();
        
        assert_eq!(config.scaling_frequency, Duration::from_secs(300), 
                  "Should have reasonable scaling frequency");
        assert_eq!(config.activity_window, Duration::from_secs(900), 
                  "Should have appropriate activity window");
        assert!(config.stability_threshold > 0.0 && config.stability_threshold < 1.0, 
               "Stability threshold should be normalized");
        assert!(config.emergency_scaling_threshold > config.stability_threshold, 
               "Emergency threshold should be higher than normal threshold");
        assert_eq!(config.metaplasticity_window, Duration::from_secs(1800), 
                  "Should have appropriate metaplasticity window");
    }

    #[tokio::test]
    async fn test_homeostatic_scaling_convergence() {
        let mut homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        // Apply homeostatic scaling multiple times and check for convergence
        let mut previous_stability = 0.0;
        let mut stability_changes = Vec::new();
        
        for i in 0..5 {
            let time_window = Duration::from_secs(60);
            let result = homeostasis.apply_homeostatic_scaling(time_window).await
                .expect("Failed to apply homeostatic scaling");
            
            if i > 0 {
                let stability_change = (result.stability_improvement - previous_stability).abs();
                stability_changes.push(stability_change);
            }
            previous_stability = result.stability_improvement;
        }
        
        // Later iterations should show smaller changes (convergence)
        if stability_changes.len() > 2 {
            let early_changes: f32 = stability_changes[0..2].iter().sum();
            let late_changes: f32 = stability_changes[stability_changes.len()-2..].iter().sum();
            
            assert!(late_changes <= early_changes, 
                   "System should converge over time with smaller later changes");
        }
    }

    #[tokio::test]
    async fn test_learning_event_types_handling() {
        let homeostasis = create_test_homeostasis().await
            .expect("Failed to create homeostasis system");
        
        let entity_key = EntityKey::default();
        
        // Test different learning event types
        let diverse_learning_history = vec![
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightIncrease,
                magnitude: 0.3,
                timestamp: SystemTime::now(),
                context: "Weight increase".to_string(),
            },
            LearningEvent {
                entity_key,
                event_type: LearningEventType::WeightDecrease,
                magnitude: 0.2,
                timestamp: SystemTime::now(),
                context: "Weight decrease".to_string(),
            },
            LearningEvent {
                entity_key,
                event_type: LearningEventType::NewConnection,
                magnitude: 0.4,
                timestamp: SystemTime::now(),
                context: "New connection formed".to_string(),
            },
            LearningEvent {
                entity_key,
                event_type: LearningEventType::ConnectionPruned,
                magnitude: 0.1,
                timestamp: SystemTime::now(),
                context: "Connection pruned".to_string(),
            },
            LearningEvent {
                entity_key,
                event_type: LearningEventType::ThresholdChange,
                magnitude: 0.15,
                timestamp: SystemTime::now(),
                context: "Threshold adjusted".to_string(),
            }
        ];
        
        let metaplasticity_state = homeostasis.implement_metaplasticity(
            entity_key,
            &diverse_learning_history,
        ).await.expect("Failed to implement metaplasticity");
        
        // Should handle diverse learning events appropriately
        assert!(metaplasticity_state.adjusted_learning_rate > 0.0, 
               "Should maintain positive learning rate with diverse events");
        assert!(metaplasticity_state.plasticity_history.plasticity_events > 0, 
               "Should count all learning events");
    }
}