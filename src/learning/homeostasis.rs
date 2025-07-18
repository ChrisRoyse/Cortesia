use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::cognitive::attention_manager::AttentionManager;
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::types::EntityKey;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
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
    pub tracking_window: Duration,
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
        time_window: Duration,
    ) -> Result<HashMap<EntityKey, f32>> {
        let mut activity_levels = HashMap::new();
        
        // 1. Get base activity from brain graph activation
        let graph_activities = self.brain_graph.get_entity_activity_levels(time_window).await?;
        
        // 2. Get attention-weighted activities
        let attention_state = self.attention_manager.get_current_attention_state().await?;
        
        // 3. Get working memory influences
        let memory_activities = self.working_memory.get_memory_activity_influences().await?;
        
        // 4. Combine all activity sources
        for (entity_key, base_activity) in graph_activities {
            let attention_weight = attention_state.attention_weights.get(&entity_key).unwrap_or(&1.0);
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
            let incoming_relationships = self.brain_graph.get_incoming_relationships(
                imbalance.entity_key,
            ).await?;
            
            let mut relationships_affected = 0;
            for relationship in &incoming_relationships {
                let new_weight = relationship.weight * scaling_factor;
                self.brain_graph.update_relationship_weight(
                    relationship.source,
                    relationship.target,
                    new_weight,
                ).await?;
                relationships_affected += 1;
            }
            
            // Also scale attention influence if this entity is in focus
            if self.attention_manager.is_entity_in_focus(imbalance.entity_key).await? {
                self.attention_manager.scale_attention_weight(
                    imbalance.entity_key,
                    scaling_factor,
                ).await?;
            }
            
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
        
        for (entity_key, &activity_level) in current_activity {
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
        let attention_focus = self.attention_manager.get_focused_entities().await?;
        let memory_load = self.working_memory.get_current_memory_load().await?;
        
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
            let incoming_relationships = self.brain_graph.get_incoming_relationships(
                imbalance.entity_key,
            ).await?;
            
            for relationship in &incoming_relationships {
                let new_weight = (relationship.weight * scaling_factor).clamp(0.0, 1.0);
                self.brain_graph.update_relationship_weight(
                    relationship.source,
                    relationship.target,
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

// Extension methods for integrating with existing systems
impl AttentionManager {
    pub async fn get_focused_entities(&self) -> Result<Vec<EntityKey>> {
        let attention_state = self.get_current_attention_state().await?;
        Ok(attention_state.current_focus.target_entities)
    }
    
    pub async fn is_entity_in_focus(&self, entity_key: EntityKey) -> Result<bool> {
        let focused_entities = self.get_focused_entities().await?;
        Ok(focused_entities.contains(&entity_key))
    }
    
    pub async fn scale_attention_weight(
        &self,
        entity_key: EntityKey,
        scaling_factor: f32,
    ) -> Result<()> {
        // Scale the attention weight for this entity
        let mut attention_state = self.attention_state.write().await;
        if let Some(weight) = attention_state.current_focus.attention_weights.get_mut(&entity_key) {
            *weight *= scaling_factor;
            *weight = weight.clamp(0.0, 1.0);
        }
        Ok(())
    }
}

impl WorkingMemorySystem {
    pub async fn get_memory_activity_influences(&self) -> Result<HashMap<EntityKey, f32>> {
        let mut influences = HashMap::new();
        let buffers = self.memory_buffers.read().await;
        
        // Entities in working memory have increased activity influence
        for buffer in [&buffers.phonological_buffer, &buffers.visuospatial_buffer, &buffers.episodic_buffer] {
            for memory_item in buffer {
                if let crate::cognitive::working_memory::MemoryContent::Entity(entity) = &memory_item.content {
                    influences.insert(entity.id, 1.0 + memory_item.importance_score);
                }
            }
        }
        
        Ok(influences)
    }
    
    pub async fn get_current_memory_load(&self) -> Result<f32> {
        let buffers = self.memory_buffers.read().await;
        let total_capacity = self.capacity_limits.total_capacity as f32;
        let current_load = (buffers.phonological_buffer.len() + 
                           buffers.visuospatial_buffer.len() + 
                           buffers.episodic_buffer.len()) as f32;
        Ok(current_load / total_capacity)
    }
}

impl BrainEnhancedKnowledgeGraph {
    pub async fn get_entity_activity_levels(&self, time_window: Duration) -> Result<HashMap<EntityKey, f32>> {
        // Get activity levels for all entities based on recent activation
        let mut activity_levels = HashMap::new();
        let cutoff_time = SystemTime::now() - time_window;
        
        let all_entities = self.get_all_entities().await?;
        for entity in all_entities {
            let activity = self.calculate_entity_activity(entity.id, cutoff_time).await?;
            activity_levels.insert(entity.id, activity);
        }
        
        Ok(activity_levels)
    }
    
    async fn calculate_entity_activity(&self, entity_key: EntityKey, since: SystemTime) -> Result<f32> {
        // Calculate activity based on recent activations and connections
        let recent_activations = self.get_recent_activations(entity_key, since).await?;
        let connection_strength = self.get_total_connection_strength(entity_key).await?;
        
        // Combine activation frequency with connection strength
        let activity = (recent_activations as f32 * 0.7) + (connection_strength * 0.3);
        Ok(activity.clamp(0.0, 1.0))
    }
}